"""
Assemble Yuanzhi_Jasmine_Chen_Talent_Intelligence_Platform.ipynb -- the case study's primary deliverable.

The notebook is deliberately thin on duplicated code: every section IMPORTS
from src/ and reads the committed dataset, so when the pipeline or the app
changes, re-running the notebook re-syncs it. Only the app section (link,
screenshots, design prose) is written by hand.

Run:  python tools/build_notebook.py     (writes and executes Yuanzhi_Jasmine_Chen_Talent_Intelligence_Platform.ipynb)
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

**Yuanzhi (Jasmine) Chen** · August 2026

| Deliverable | Where |
|---|---|
| Live application | **https://m-case-study-jasmine.streamlit.app/** |
| Code repository | https://github.com/yc4379-commits/m-case-study |
| Parsed data (JSON / CSV) | `data/candidates.json` · `data/candidates.csv` |
| This notebook | the walk-through: every section below runs against the committed code and data |

The task: parse ten resumes (PDF and Word) with an LLM API into structured
JSON/CSV, and build a Streamlit application where a Business Development
user can search and filter **junior analyst candidates** against job
requisitions, with distribution insights — designed to scale beyond ten
resumes.

Three product claims shape every design decision in this system:

1. **The matching is reliable enough to act on.** Hard constraints
   disqualify; soft signals rank — a candidate outside the role's region or
   experience band is not an 82% match, they are not a match, which is what
   lets this search honestly return **zero results**. Which criteria are
   hard and which can be relaxed follows how recruiting actually works:
   geography and investment approach are non-negotiable, credentials and
   skills are weighted preferences, and the weights live in YAML where they
   can be argued about. The scoring respects industry rules for **junior
   analyst** hiring specifically — demonstrated skills, credentials and
   education carry weight; self-reported AUM and returns, the currency of
   senior hiring, are displayed with their quotes but never scored.
2. **The parsing reads the pile the way a careful screener would — and
   proves it.** Every classification carries the resume sentence that
   produced it, verified to appear **verbatim** in the source: nothing
   invented, nothing silently dropped. The checks a human runs first on a
   finance resume — employment gaps, overlapping dates, malformed contact
   details, credential inconsistencies, formatting slips — are computed
   automatically and surfaced as flags, with a human triage layer deciding
   which are real risks and which are benign. Accuracy is measured, not
   assumed: §7 scores the system against 40 blind human labels.
3. **The interface is built for the BD workflow, not for a demo.** Accurate
   display first, then speed: candidate features are split into fine
   dimensions and zoned with tags and charts, so a screener catches the
   deciding facts — and the risks — without reading walls of prose, and can
   drill from any tag to its verbatim quote. Because the end user is a BD
   team, the workflow continues past search: one-click outreach drafts,
   pool-level insight views, and working previews of what more data
   unlocks (an internal-knowledge AI assistant, a talent knowledge graph).
""")

# ------------------------------------------------------------- 1 · pipeline
md("""
## 1 · Architecture

![Pipeline](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/pipeline.png)

Each dark box is one module in `src/`; the light boxes are data. The
curated `knowledge/` YAML files feed enrichment, and everything the app
shows is read from the committed `data/candidates.json` — the public
deployment calls no model.

The split of labour is a design position: **the model is asked only for
judgement** (fundamental vs systematic, which sectors, is this an investment
role — each with evidence), and **everything code can settle is settled by
code** (tenure arithmetic, firm identity, regions, gap detection). A wrong
judgement is a prompt experiment to fix; a wrong lookup is one line of YAML.
""")

md("""
First, an orientation view of the **whole parsed pool** — all ten
candidates before any requisition or filter is applied, eight columns
chosen for scanning rather than the full record. (Requisition matching,
where hard requirements start disqualifying, is §5.) Each parsed candidate
carries **30+ top-level fields** plus the
complete nested extraction (every position with dates and bullets, education,
credentials, coverage, per-claim evidence quotes, flags); the field
inventory and one record in full follow in §3, and the whole dataset is
itself a deliverable: `data/candidates.json` / `data/candidates.csv`.
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

Human review caught these failures first — but *staying* caught is
systematic, not heroic: every document's extraction is diffed against a
naive baseline (a large character-count gap flags recovered, or still
missing, content); the model reports damage it can see in
`extraction_notes` (it was the model that exposed the bad `ffi` repair);
and §3's verbatim-quote verification doubles as an extraction alarm — a
quote that cannot be found in our text is often the text's fault, not the
model's. At scale this becomes sampled human audits plus drift alarms on
the extraction-diagnostics distribution (§9).

Per-document diagnostics from the committed extraction log:
""")

code("""
log = pd.read_csv(ROOT / "data" / "extraction_log.csv")
log
""")

# -------------------------------------------------------------- 3 · parsing
md("""
## 3 · Structured extraction by the model

The mechanics, end to end:

1. **The schema is the interface.** The Pydantic model in `src/schema.py`
   is serialised to JSON Schema and passed to the Anthropic API as a
   **tool definition**, with tool choice forced — the model can only
   answer as an object of exactly that shape. No free-text JSON to repair,
   and every field's description doubles as that field's prompt, versioned
   in code review like any other code.
2. **Corrective retry.** A validation failure is sent back once *with the
   specific errors attached*, not blindly re-rolled.
3. **Verbatim-quote verification.** Every evidence quote is checked to
   appear in the source text (whitespace and punctuation forgiven — our
   own extraction reflows lines — words never). Unverified quotes reduce
   the record's confidence score.
4. **Content-hash cache** keyed on (text, model, schema, system prompt):
   re-running after a knowledge-base change costs $0.00. The key includes
   the prompt because an early version silently replayed pre-fix results
   after a prompt fix; it includes the schema because a schema change
   *must* re-parse.

**Nothing here is fine-tuned.** The accuracy comes from engineering around
a general model, not from training one — which is what makes the approach
reproducible and cheap to improve. Five things carry it: the model sees
*complete* text (§2 — most "LLM parsing errors" are extraction losses);
it is asked only for judgement, never for anything arithmetic can settle;
the schema constrains the shape of every answer while the system prompt
(above) constrains its discipline — transcribe don't embellish, verbatim
evidence, classify by substance, distrust section headings; verification
then catches what discipline misses, since an invented quote fails the
verbatim check and lowers the record's confidence; and accuracy is
*measured* against blind human labels (§7), so any prompt or schema change
re-runs against that fixed yardstick. Iteration is the training loop here:
edit a field description, re-parse (the cache makes unchanged documents
free), re-evaluate.

One prompt lesson from that loop: the filename was originally in the
prompt, and the model used it for the candidate's name on some runs and
refused on others — ambiguous instructions produce non-deterministic
output. The filename is now withheld; a deterministic fallback applies
only when the document itself states no name, and flags the record.

Parsing all 10 resumes cost **≈ $0.80** (claude-sonnet-5); the public app
calls no model and needs no key.
""")

code("""
from parse import SYSTEM_PROMPT
print(SYSTEM_PROMPT)
""")

md("""
What one parsed record looks like — **Ryan Patel**, as tables rather than a
JSON wall. First every judgement the model made, each with its confidence
and the verbatim quote that earned it; then the position history the tenure
arithmetic runs on. The full nested record (bullets, education, flags) is
`data/candidates.json` — nothing appears in the app that is not in it.
""")

code("""
ryan = next(c for c in candidates if c["display_name"] == "Ryan Patel")
e = ryan["extraction"]
pd.set_option("display.max_colwidth", 100)

def j(field, item):
    return {"field": field, "value": str(item["value"]),
            "conf": item["confidence"], "evidence (verbatim)": item["evidence"]}

pd.DataFrame(
    [j("investment_approach", e["investment_approach"]),
     j("market_side", e["market_side"])]
    + [j("sector", s) for s in e["primary_sectors"]]
    + [j("asset_class", a) for a in e["asset_classes"]]
    + ([j("team_leadership", e["team_leadership"])]
       if e["team_leadership"]["value"] else [])
    + [{"field": f"stated_metric · {m['kind']}", "value": m["figure"],
        "conf": "", "evidence (verbatim)": m["quote"]}
       for m in e["stated_metrics"]]
)
""")

code("""
pd.DataFrame([{
    "firm": pos["firm"], "title": pos["title"],
    "dates": f'{pos["start_date"]} → {pos["end_date"] or "present"}',
    "type": pos["employment_type"],
    "investment role": pos["is_investment_role"],
} for pos in e["positions"]])
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
## 8 · The application — one search, walked end to end

Live: **<https://m-case-study-jasmine.streamlit.app/>** — public; calls no
model. The deployment serves precomputed data, so no API key exists in the
app to leak or spend. What follows is the path a BD user actually takes
through it.

### Step 1 · Say what you are hiring for

The first control on the page is the only mandatory question. Three ways
in: pick a posting from the **job library** (all four transcribed from
real postings — three Millennium REQs and one Point72), **define your own
criteria** when the seat is not in the library yet, or **browse** the
whole pool with no matching at all:

![Choosing how to filter](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_modes.png)

### Step 2 · The role locks the hard requirements; the sidebar refines the rest

Picking a role pins it above the results with its hard requirements as
chips — and the sidebar *shrinks*. Dimensions the role has decided
(region, approach, sector, experience) disappear rather than grey out: a
second control over a decided axis would read as a second authority. Only
what the role leaves open stays refinable:

![Job-library mode — the role decides, the sidebar refines](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_sidebar_locked.png)

In **browse mode** nothing is locked and all five facets are live —
market, approach, sector, market side, asset class — each with counts
that update as the others narrow, plus free-text keyword search:

![Browse mode — every facet open](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_sidebar_browse.png)

**Advanced filters** hold what junior-analyst screening actually turns
on. A junior's record is coursework, licences and tools — not a track
record — so software, credentials and an experience slider lead the
list; a **minimum parse-confidence** control closes it, because with a
large pool the cheapest first cut is dropping the badly-parsed — a
resume that fails basic completeness and verifiability has not earned a
screener's minute yet. Every control carries a hover definition, and no
filter ever overrides the role:

![Advanced filters](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_advanced.png)

### Step 3 · Read the results

Qualified and one-gap-away are separate groups, never one blended
ranking. Rows sort by fit within each group, and every row answers
"why" on sight — a match names the signals that earned its score, a
near miss names its single gap with both numbers ("has 12.7 years, role
needs 4–5"), which a recruiter reads as *what would I have to relax*:

![The result list](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_results.png)

At volume, the same results become a sortable **table** — the full
parsed breadth, 21 columns, every header with a hover definition, and a
CSV download carrying the same columns:

![Table view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_table.png)

### Step 4 · Open the person

![Candidates view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_candidates.png)

Three commitments are visible in that one frame. Ryan's **"Millennium
alum"** pill is resolved through the firm knowledge base — platform
experience is the first fact a multi-manager recruiter scans for, and the
same resolver carries pod-to-platform lineage (North53 → Millennium, §4),
so the pill would survive a resume that named only the pod. The **61% is
green only because he clears all four hard requirements** — green marks
eligibility, never magnitude — and "from 7 of 8 signals · 93% of the full
weighting" concedes what the score could not see instead of quietly
renormalising. And the header strip — **"Resume read cleanly · high 0.88
· 4 issues"** — keeps parse confidence next to the name, not buried in an
admin tab.

The panel itself is **six views** switched by pills (inside a Streamlit
fragment, so a click redraws only the panel). Each answers one screening
question — Fit: *does the score hold up?* · Profile: *what are they good
at?* · Figures: *what do they claim?* · Issues: *what should I
double-check?* · Outreach: *how do I write to them?* · Full record:
*what exactly was parsed?*

![The six views of one candidate](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_profile_tabs.png)

**Profile** — every classified attribute as a three-tier block: the
value, the keyword tags that earned it, the verbatim quote, with the
model's confidence alongside. Absence is stated ("none listed", dashed),
never left blank:

![Profile view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_profile.png)

**Figures** — the resume's own numbers, structured. Stated AUM,
performance and risk figures are extracted with verbatim quotes and
**displayed, never scored**; the one comparable figure — names under
coverage — is scored at 8% weight and its card says so. The dividing
line is comparability, not importance: this very pool contains a
"$4.5 trillion" that is Fidelity's AUM, not the candidate's book:

![Figures view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_figures.png)

**Outreach** — the step after "this candidate fits" is always "someone
writes to them": a one-click briefing whose every claim quotes the
resume, because a sourcing mail earns replies by proving someone
actually read it:

![Outreach view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_outreach.png)

### Step 5 · Compare the finalists

Marcus and Ryan both score 61% on the Point72 seat. The compare view
overlays their radars and prints the weighted components: Marcus earns
it on requirements and skills, Ryan on firm type and platform lineage.
One blended number would have hidden exactly the trade-off a recruiter
is paid to make:

![Compare view](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_compare.png)

### When the answer is no one

Against the Mumbai posting's 4–5 year band, nobody qualifies — and the
app says so instead of ranking unqualified people confidently. It names
each near miss's single gap and computes what each widening would admit:

![Zero results](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_zero_results.png)

### The Insights tab — managing the pool, not one search

The coverage heatmap maps the bench: regions by sectors, each cell the
number of candidates covering both. **The empty cells are the point** —
the seats this pool cannot fill — so the chart doubles as a sourcing
to-do list. A parse-confidence threshold above it keeps thin records out
of the counts:

![Sector coverage by region](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_heatmap.png)

Career length vs investing tenure, one row per candidate. The *gap*
between the two dots is the story: a wide gap is a career changer
arriving from banking, consulting or engineering — the reason the system
computes both numbers instead of one:

![Career length vs investing tenure](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_tenure.png)

Credentials and software ranked by how many candidates hold them — a
read on the market mainstream. It calibrates requisitions as much as
candidates: a JD naming a tool nobody in the pool holds describes a
sourcing problem, not a screening one:

![Most common credentials and software](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_credentials.png)

Last, the talent-network graph — platforms, firms and candidates as
three layers, drawn live from `firms.yaml`. Grey edges are employment,
which the resumes state; the bronze edge is an ownership link **no
resume states** — J.P. Morgan Asset Management belongs to JPMorgan
Chase, knowledge that lives only in the firm KB. The same KB carries
pod-to-platform lineage (North53 → Millennium, §4), the class of edge
that matters most at scale: the knowledge-graph roadmap item (§10) at
its smallest honest size:

![Talent network](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/app_network.png)

### The Data quality tab — trust, cross-checked

Every record's parse confidence with its specific deductions, ranked
worst-first. It exists to let a human sanity-check the scores
themselves, and it pairs with the triage layer (§4) where a human
decides which flagged issues deduct and which are benign conventions:

![Data quality](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_quality.png)

### Ask · preview — the AI assistant, at its honest size

Free-text questions over every parsed resume sentence, every answer a
quoted sentence with its source. Today it runs the same auditable
concept scorer the matcher uses; the production version is the roadmap's
RAG step (§10) over resumes plus the internal sourcing corpus — meeting
notes, call summaries — under the same rule:

![Ask the pool](https://raw.githubusercontent.com/yc4379-commits/m-case-study/main/docs/tour_ask.png)

The interface went through ~15 review rounds with two reviewers (a
BD-user perspective and a UI designer). The grammar that survived: navy
is chrome, **green means "clears every hard requirement"** and nothing
else, bronze means gap or caveat, and informational facts are neutral
pills — tags may carry a tinted background; coloured *text* was reviewed
out.
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

Three of these already exist in the app as deliberately small **previews**
— real code over real data, never a mock with invented output — so the
roadmap below is a widening, not a wish list.

- **Referral & CRM metadata.** "Referred by" is not in any resume — it is
  ATS-side data. The record schema gets a `referral` field (default
  unknown) and a sidebar facet. *In the app today:* a disabled "Referred
  candidates only" toggle that names exactly the data it is waiting for,
  kept out rather than invented.
- **Verified performance.** Self-reported AUM and returns are shown today
  as quoted, clearly-labelled statements only — they are unverifiable and
  most resumes omit them, so as a *ranking* signal they would reward
  disclosure habits, not ability. With a licensed data source (fund
  filings), book size and track record become real, scoreable fields.
- **Internal sourcing corpus + RAG.** Meeting notes and call summaries
  would let free-text questions ("who impressed us on biotech last year?")
  join the structured search — with governance caveats: retrieval scope,
  permissions, and the same evidence-quote discipline. *In the app today:*
  the "Ask · preview" tab is this feature's smallest real slice —
  concept-map retrieval over actual candidate sentences, quotes returned,
  no generation.
- **In-app labelling.** The ground-truth workflow (§7) moves into the app:
  pick a role, label candidates, accuracy recomputes — evaluation as a
  habit, not an event.
- **JD upload in a governed deployment.** Parsing a pasted JD is the same
  extraction problem as parsing a resume and the code path exists; it is
  withheld from the public app only because it requires a live key in a
  public page. An internal deployment removes that constraint.
- **Knowledge graph.** `firms.yaml` is already a small graph (firm → parent
  → platform); at scale it becomes queryable lineage ("everyone two hops
  from a Millennium pod"). *In the app today:* the Insights talent network
  draws that graph live — the bronze edges are relationships no resume
  states.
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
nbf.write(nb, "Yuanzhi_Jasmine_Chen_Talent_Intelligence_Platform.ipynb")
client = nbclient.NotebookClient(nb, timeout=180, kernel_name="python3")
client.execute()
nbf.write(nb, "Yuanzhi_Jasmine_Chen_Talent_Intelligence_Platform.ipynb")
print("Yuanzhi_Jasmine_Chen_Talent_Intelligence_Platform.ipynb written and executed:", len(nb.cells), "cells")
