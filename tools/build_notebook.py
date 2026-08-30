"""
Assemble case_study.ipynb -- the case study's primary deliverable.

The notebook is deliberately thin on duplicated code: every section IMPORTS
from src/ and reads the committed dataset, so when the pipeline or the app
changes, re-running the notebook re-syncs it. Only the app section (link,
screenshots, design prose) is written by hand.

Run:  python tools/build_notebook.py     (writes and executes case_study.ipynb)
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

# ---------------------------------------------------------------- 0 · title
md("""
# Talent Intelligence Platform
### Millennium Business Development — Data Science case study

**Jasmine (Yuanzhi) Chen** · August 2026

| Deliverable | Where |
|---|---|
| Live application | **https://m-case-study-jasmine.streamlit.app/** |
| Code repository | https://github.com/yc4379-commits/m-case-study |
| Parsed data (JSON / CSV) | `data/candidates.json` · `data/candidates.csv` |
| This notebook | the walk-through: every section below runs against the committed code and data |

The task: parse ten resumes (PDF and Word) with an LLM API into structured
JSON/CSV, and build a Streamlit application where a Business Development
user can search and filter candidates against job requisitions, with
distribution insights — designed to scale beyond ten resumes.

Three commitments shape every design decision in this system:

1. **Hard constraints disqualify; soft signals rank.** A candidate outside
   the role's region or experience band is not an 82% match — they are not a
   match. Keeping the two separate is what lets this search honestly return
   **zero results**, which commercial matching tools cannot do.
2. **Every claim carries its evidence.** No classification appears without
   the resume sentence that produced it, verified to appear verbatim in the
   source. This is not decoration: the worst scoring bug in this project was
   invisible in the aggregate numbers and obvious the moment one evidence
   quote was read (§7).
3. **Data quality is visible, never silent.** Every record carries a parse
   confidence with the specific reasons, and formatting deliberately carries
   **zero** weight in it — in this corpus, messy formatting tracks regional
   convention, not candidate quality.
""")

# ------------------------------------------------------------- 1 · pipeline
md("""
## 1 · Architecture

```
data/resumes/*.{docx,pdf}
        │
        ▼  src/extract.py        paragraphs + tables + floating text boxes;
        │                        PDF column separation; ligature repair
        ▼  src/parse.py          schema-constrained LLM extraction (Anthropic
        │                        tool use), validation with corrective retry,
        │                        verbatim-quote verification, content-hash cache
        ▼  src/checks.py         deterministic checks: gaps, overlaps, contact
        │                        details, credential pairs, city-zip sanity
        ▼  src/enrich.py         knowledge-base lookups, tenure arithmetic,
        │                        entity resolution, human flag-triage,
        │                        confidence scoring
        ▼
data/candidates.json ──► app.py (Streamlit) ──► src/match.py (requisitions)
                                    │
                                    └──► src/evaluate.py (accuracy vs human labels)
```

The split of labour is a design position: **the model is asked only for
judgement** (fundamental vs systematic, which sectors, is this an investment
role — each with evidence), and **everything code can settle is settled by
code** (tenure arithmetic, firm identity, regions, gap detection). A wrong
judgement is a prompt experiment to fix; a wrong lookup is one line of YAML.
""")

md("""
First, an orientation view — eight columns chosen for scanning, not the
full record. Each parsed candidate carries **30+ top-level fields** plus the
complete nested extraction (every position with dates and bullets, education,
credentials, coverage, per-claim evidence quotes, flags); the full inventory
and one complete record follow in §3, and the whole dataset is itself a
deliverable: `data/candidates.json` / `data/candidates.csv`.
""")

code("""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

candidates = json.loads((ROOT / "data" / "candidates.json").read_text())
pd.DataFrame([{
    "candidate": c["display_name"],
    "region": c["region"],
    "approach": c["approach_family"],
    "yrs investing": c["years_investment_experience"],
    "sectors": ", ".join(c["sectors"]),
    "firm": c["current_firm"],
    "parse confidence": f'{c["quality"]["score"]} ({c["quality"]["band"]})',
} for c in candidates])
""")

# ----------------------------------------------------------- 2 · extraction
md("""
## 2 · Reading the documents (where most pipelines silently fail)

The largest source of error in a resume pipeline is not the model — it is
**losing content before the model ever sees it**. Three real failures in this
corpus, all caught because the extraction layer reports diagnostics instead
of passing whatever it got:

- **`Viktor_Sharat.docx` keeps its name, degree and section headings in
  floating text boxes** — elements `python-docx` does not read at all. Naive
  extraction returned 503 characters and a resume with apparently no name;
  walking the raw XML for `w:txbxContent` recovered 3,100.
- **`Zara_AlRashid.docx` keeps its work history in tables** (2,848 → 3,401
  chars once tables are read in document order).
- **`Omar_ElHassan.pdf` has a broken ToUnicode map**: ligatures decode to
  U+FFFD, so "Quantitative" arrives as "Quan�ta�ve". 35 damaged tokens were
  repaired by rule (word-shape patterns — an early `ffi` rule corrupted
  "Statistics" and was removed after the model itself reported the damage);
  2 unrecoverable ones are *reported*, not guessed.
- Two-column PDFs are split by **line-start bimodality** before reading —
  a first histogram-based version failed on pages where the main column
  spans the width, and a second version false-positived on three centred
  header lines; the shipped rule requires the right cluster to be at least
  4 lines and 15% of the page.

Per-document diagnostics from the committed extraction log:
""")

code("""
log = pd.read_csv(ROOT / "data" / "extraction_log.csv")
log
""")

# -------------------------------------------------------------- 3 · parsing
md("""
## 3 · Structured extraction by the model

`src/parse.py` calls the Anthropic API with the Pydantic schema in
`src/schema.py` passed as a **tool definition**, so the model must return an
object of exactly that shape — no free-text JSON to repair. Three guards sit
on top:

- **Corrective retry**: a validation failure is sent back once *with the
  specific errors attached*, not blindly re-rolled.
- **Verbatim-quote verification**: every evidence quote is checked to appear
  in the source text (whitespace/punctuation forgiven — our own extraction
  reflows lines — words never). Unverified quotes reduce the record's
  confidence score.
- **Content-hash cache** keyed on (text, model, schema, system prompt):
  re-running the pipeline after a knowledge-base change costs $0.00. The
  key *includes the prompt* because an early version silently replayed
  pre-fix results after a prompt fix.

One prompt lesson worth recording: the filename was originally included in
the prompt, and the model used it for the candidate's name on some runs and
refused on others. **Ambiguous instructions produce non-deterministic
output — the bug was in the prompt, not the model.** The filename is now
withheld; a deterministic `name_from_filename()` fallback applies only when
the document itself states no name, and flags the record.

Parsing all 10 resumes cost **≈ $0.80** (claude-sonnet-5); the public app
calls no model and needs no key.
""")

code("""
from parse import SYSTEM_PROMPT
print(SYSTEM_PROMPT)
""")

code("""
# A parsed record, abbreviated: the approach/sector judgements carry their
# evidence, and the quotes are verified verbatim against the source text.
ryan = next(c for c in candidates if c["display_name"] == "Ryan Patel")
e = ryan["extraction"]
{
    "investment_approach": e["investment_approach"],
    "primary_sectors": e["primary_sectors"][:2],
    "positions[0]": {k: e["positions"][0][k] for k in
                     ("firm", "title", "start_date", "end_date",
                      "employment_type", "is_investment_role")},
}
""")

code("""
# Everything the pipeline produces, per candidate: the model's extraction
# fields plus the enrichment computed on top. This is the full contract --
# nothing is parsed that is not listed here.
print("EXTRACTION (read by the model, with evidence)")
print("  " + ", ".join(sorted(e.keys())))
print()
print("ENRICHMENT (computed: knowledge base + arithmetic + checks)")
print("  " + ", ".join(sorted(k for k in ryan.keys() if k != "extraction")))
""")

md("""
And one **complete record**, unabridged — every position with its bullets,
every judgement with its quote, every flag with its provenance. This is what
"parse the resume" actually means here; the other nine records have the same
shape in `data/candidates.json`.
""")

code("""
print(json.dumps(ryan, indent=2, ensure_ascii=False))
""")

# ------------------------------------------------------ 4 · knowledge base
md("""
## 4 · The knowledge base supplies what no model knows

`knowledge/` holds curated domain facts a language model cannot be trusted
to supply (and does not signal when it is guessing): firm identities and
pod-to-platform lineage, region and sector taxonomies, credential
expansions, requisitions, and the human flag-triage file.

Three behaviours worth demonstrating live:
""")

code("""
from knowledge_base import KnowledgeBase, years_of_experience
kb = KnowledgeBase.load(ROOT / "knowledge")

# 1. Refusal to guess: four unrelated firms here begin with "Meridian".
#    A substring matcher would silently relocate a candidate to the wrong
#    continent; this one reports ambiguity instead of resolving.
print("resolve('Meridian')      ->", kb.resolve_firm("Meridian").method)

# 2. Pod-to-platform lineage: the resume names only the pod; the platform
#    exists only here. This is what lets the app surface "previously at
#    Millennium" for Ryan Patel.
print("lineage('North53 Capital') ->", kb.platform_lineage("North53 Capital"))

# 3. Tenure is date arithmetic, never model output. Overlapping positions
#    merge; internships and student societies are excluded -- counting them
#    added four years to one candidate in this pool.
overlap = [
    {"start_date": "2020-01", "end_date": "2022-01", "is_current": False,
     "employment_type": "professional", "is_investment_role": True},
    {"start_date": "2021-01", "end_date": "2023-01", "is_current": False,
     "employment_type": "professional", "is_investment_role": True},
]
print("overlapping 2y+2y roles  ->", years_of_experience(overlap), "years")
""")

md("""
A later addition in the same spirit: **human flag triage**
(`knowledge/flag_review.yaml`). The model reports everything it notices;
a human reviewer marked several observations benign (a summer internship
inside an MBA, non-US number formatting from a non-US candidate). Those
decisions are *knowledge*: each is recorded with its reasoning, downgrades
the flag to an unscored note, and is never a silent deletion — triage stays
auditable and reversible.
""")

# ------------------------------------------------------------- 5 · matching
md("""
## 5 · Requisition matching: eligibility, then rank

All four shipped requisitions are transcribed from **real postings** — three
Millennium (REQ-27950, REQ-25042, REQ-29449) and one Point72 — rather than
written to fit the data. That matters: a requisition invented alongside the
scoring logic can only confirm itself. Transcription was faithful even where
inconvenient: the Mumbai role's "healthcare preferred but not mandatory"
means sector is *not* a hard constraint there; the Origination role states
no years band, so it has none.

**Hard constraints disqualify** (region, approach family, sector-any,
experience band). **Soft signals rank** the survivors — a weighted blend of
sector fit, requirement-text similarity, skills, firm type, coverage depth,
credentials, platform lineage and buy-side experience, with the weights in
`knowledge/requisitions.yaml` where they can be argued about. Requirement
similarity is a pluggable backend; the default combines lexical overlap
with a **curated concept map** (a requisition says "catalysts" where a
resume says "earnings events"). Neural sentence embeddings drop in
unchanged when the corpus outgrows curation (~low tens of thousands of
documents); at 200 candidate sentences the map performs comparably, adds no
500MB dependency to a free-tier deployment, and is auditable — a bad match
is fixed by editing a line of YAML.

Near misses — candidates failing **exactly one** hard requirement — are
listed separately with the failed requirement and both numbers named.
Failing two or more means a different person; padding lists with them is
the behaviour this system exists to avoid.
""")

code("""
from match import Requisitions, match_all
store = Requisitions.load(ROOT / "knowledge")

rows = []
for spec in store.items:
    exact, near = match_all(candidates, spec, store=store)
    rows.append({
        "requisition": spec["title"],
        "source": spec.get("source", ""),
        "qualify": len(exact),
        "one gap away": len(near),
        "top match": (f'{exact[0].display_name} ({exact[0].soft_score:.0%})'
                      if exact else "—"),
    })
pd.DataFrame(rows)
""")

code("""
# The result the design is proudest of: an honest zero. Against the Mumbai
# posting's 4-5 year band, nobody qualifies -- and instead of a confidently
# ranked list, the system names the single gap for each near miss.
spec = store.get("mlm_mumbai_healthcare_research")
exact, near = match_all(candidates, spec, store=store)
pd.DataFrame([{
    "candidate": r.display_name,
    "fit (soft)": f"{r.soft_score:.0%}",
    "the one gap": f"{r.failed_hard[0].label}: has {r.failed_hard[0].found}, "
                   f"role needs {r.failed_hard[0].required}",
} for r in near])
""")

# ------------------------------------------------------------ 6 · big bug
md("""
## 6 · The bug that justifies the evidence rule

Mid-project, every requirement similarity score came out around 0.75 and the
ranking looked plausible. The aggregate numbers hid the cause completely;
**one on-screen evidence quote exposed it** — the scorer offered
*"Biomodeller Trainee BioAnalytics Research India Ltd."* as proof of
*"fundamental research on India equity"*.

The concept map contained `r` (the R language), matched as a raw substring —
and `"r" in sentence` is true of nearly every English sentence, so every
requirement scored on its concept term alone. The fix was word-boundary
regex matching plus short-sentence damping. The same substring lesson
resurfaced twice more (a title-hint rule read the "intern" inside
"Consumer **Intern**et"; an early flag matcher over-merged) — which is why
the shipped code matches tokens, never substrings, everywhere.

The design conclusion: **a scoring system whose every claim is quoted is a
scoring system whose bugs are visible.** That is why evidence is a hard
requirement of the schema, not decoration.
""")

# ---------------------------------------------------------- 7 · evaluation
md("""
## 7 · Accuracy against blind human judgment

Percentages first, method second — but the method is the point:

- A reviewer (the author, acting as the BD screener) labelled **all 40
  candidate-role pairs** blind: the candidate's facts were visible, the
  system's verdict never was. Y / N / borderline; borderline is excluded
  from counts rather than forced into a bucket.
- The system's shortlist is its **exact matches only** — near misses do not
  count for it.
""")

code("""
from evaluate import evaluate
ev = evaluate()
per = pd.DataFrame(ev["per_role"])[
    ["role", "precision", "recall", "agreement", "tp", "fp", "fn", "tn"]]
o = ev["overall"]
print(f'OVERALL  precision {o["precision"]:.0%}  recall {o["recall"]:.0%}  '
      f'agreement {o["agreement"]:.0%}  (n={o["judged"]} judged, '
      f'{len(ev["borderline"])} borderline set aside)')
per
""")

code("""
pd.DataFrame(ev["disagreements"])
""")

md("""
**Reading the disagreements.** Precision is perfect on this pool — the
system never shortlisted anyone the reviewer would reject. Every miss has a
single cause: the Mumbai posting's 4–5 year band excludes three APAC
healthcare analysts at 9.8–12.7 years, all of whom the reviewer shortlists.
Nothing comparable happened with region — all nine region mismatches were
labelled *no* — so the reviewer treats geography as genuinely hard while
treating **over-qualification as negotiable**, a distinction the posting's
text does not make. The borderline labels cluster the same way, on the
quant seat's approach constraint.

The reviewer's own account sharpens it further: the widening was
**supply-driven** — with ten candidates she stretches the band; with ten
thousand she would not. Shortlisting standards are elastic to pool depth,
which no fixed threshold can encode. The design response is not to soften
the band but to keep it transcribed and make widening a *visible,
per-search decision*: all three missed candidates sit at the top of the
one-gap-away list with the band named on their row, and the empty state
computes what each widening would admit.

At n=40 the percentages measure nothing statistically. What the exercise
measures is **which rule diverges from practitioner judgment** — and it
found exactly one. The same harness (`src/evaluate.py` + a labelling sheet)
is the regression suite for any future scoring change.
""")

# ----------------------------------------------------------------- 8 · app
md("""
## 8 · The application

Live: **<https://m-case-study-jasmine.streamlit.app/>** — public; calls no
model. The deployment serves precomputed data, so no API key exists in the
app to leak or spend.

![Candidates view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_candidates.png)

The interface went through ~15 review rounds with two reviewers (a BD-user
perspective and a UI designer). The decisions that survived:

- **One screen, three zones**: role (eligibility) → advanced filters
  (refinement) → results with profile. When a role fixes a dimension, the
  sidebar filter for it *disappears* — a second control over the same axis,
  even disabled, read as a second authority.
- **Qualify vs One-gap-away** as separate collapsible groups; rows are one
  line each (name, Fit, two facts, quality tag) because the *why* lives one
  glance right, in the Fit panel.
- **Colour grammar**: navy is chrome, green means "clears every hard
  requirement", bronze means gap/caveat, and every informational tag is one
  neutral pill. Verdict facts may be *tags with tinted backgrounds*;
  coloured *text* was reviewed out.
- **Evidence on demand**: profile shows keywords and per-dimension
  contributions ("What counted"); verbatim quotes live one expander deeper.
- **Outreach draft**: the step after "this candidate fits" is always
  "someone writes to them" — one click assembles a summary whose every
  claim quotes the resume, because a sourcing mail earns replies by proving
  someone actually read it.

![Shortlist table](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_table.png)
""")

# --------------------------------------------------------- 9 · scalability
md("""
## 9 · Designed for ten, priced for a hundred thousand

Every scale-sensitive choice has a stated threshold and a successor:

| Component | At 10 resumes (shipped) | At ~100k resumes |
|---|---|---|
| Parsing | sequential calls, content-hash cache | queue + batch API; ≈ $0.08/resume ⇒ ≈ $8k for 100k, incremental thereafter (cache means re-enrichment is free) |
| Requirement similarity | curated concept map (auditable YAML) | neural sentence embeddings via the pluggable backend in `src/match.py`; switch point ≈ low tens of thousands of documents |
| Search | in-memory filtering | SQLite FTS / OpenSearch index + pgvector for semantic recall |
| Entity resolution | curated `firms.yaml` | the same guarded resolver over a licensed firm graph (e.g. FactSet entities), still refusing ambiguous matches |
| Quality & eval | per-record confidence; 40-pair human eval | sampled human labelling as a monitored metric; drift alarms on parse-confidence distribution |
| Serving | Streamlit Community Cloud | internal deployment; JD upload wired to the parsing service with managed keys (deliberately absent from the public app) |

The **evaluation harness is the keystone for scale**: any threshold change,
prompt change or backend swap re-runs against the labelled pairs before it
ships.
""")

# ------------------------------------------------------------ 10 · roadmap
md("""
## 10 · If I had more time (and the features I chose not to fake)

- **Referral & CRM metadata.** "Referred by" is not in any resume — it is
  ATS-side data. The record schema gets a `referral` field (default
  unknown) and a sidebar facet; kept out now rather than invented.
- **Verified performance.** Self-reported AUM and returns are shown today
  as quoted, clearly-labelled statements only — they are unverifiable and
  most resumes omit them, so as a *ranking* signal they would reward
  disclosure habits, not ability. With a licensed data source (fund
  filings), book size and track record become real, scoreable fields.
- **Internal sourcing corpus + RAG.** Meeting notes and call summaries
  would let free-text questions ("who impressed us on biotech last year?")
  join the structured search — with governance caveats: retrieval scope,
  permissions, and the same evidence-quote discipline.
- **In-app labelling.** The ground-truth workflow (§7) moves into the app:
  pick a role, label candidates, accuracy recomputes — evaluation as a
  habit, not an event.
- **JD upload in a governed deployment.** Parsing a pasted JD is the same
  extraction problem as parsing a resume and the code path exists; it is
  withheld from the public app only because it requires a live key in a
  public page. An internal deployment removes that constraint.
- **Knowledge graph.** `firms.yaml` is already a small graph (firm → parent
  → platform); at scale it becomes queryable lineage ("everyone two hops
  from a Millennium pod").
""")

md("""
## Appendix · Reproducing everything

```bash
pip install -r requirements.txt
streamlit run app.py                 # the app, against committed data

pip install -r requirements-dev.txt
python -m pytest tests/ -q           # 11 regression tests, no API key needed
python src/evaluate.py               # accuracy vs the human labels

# full rebuild from raw resumes (needs resumes in data/resumes/ and an
# ANTHROPIC_API_KEY in .env; cached, so re-runs are free):
python src/build_dataset.py
python tools/build_notebook.py       # re-executes this notebook
```
""")

nb.cells = C

import nbclient
nbf.write(nb, "case_study.ipynb")
client = nbclient.NotebookClient(nb, timeout=180, kernel_name="python3")
client.execute()
nbf.write(nb, "case_study.ipynb")
print("case_study.ipynb written and executed:", len(nb.cells), "cells")
