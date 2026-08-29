"""
Enrichment: extraction + knowledge base -> Candidate.

Everything here is deterministic. No model is called. Each field is either
arithmetic over data we already have or a lookup in a curated table, and both
are cheaper, faster and auditable in a way a model call is not. When a value
is wrong, the fix is a line of YAML rather than a prompt experiment.

Four derived values carry more weight than the rest.

**platform_alum_of** answers "has this person worked here before?". Ryan
Patel's resume names North53 Capital, not Millennium; only the pod-to-parent
lineage in the knowledge base connects the two. For a business development
team this is the highest-value field in the record, and no amount of prompt
engineering would have produced it.

**seniority_band** comes from computed tenure, not from the job title. An
earlier title-keyword version returned "mid" for nine of ten candidates,
including one whose title was literally "Junior Analyst". Titles are not
comparable across market sides; tenure is.

**employers** is de-duplicated and annotated. The raw position list repeats a
firm once per role and files a university swimming club alongside Bank of
China. Neither is usable in a search interface.

**flags** merges what the model found by reading with what this code found by
computing, into one list a reviewer can scan.
"""

from __future__ import annotations

import re
from typing import Optional

from checks import (
    check_contact,
    check_formatting,
    check_seniority,
    check_tenure_split,
    check_timeline,
)
from knowledge_base import KnowledgeBase, years_of_experience
from schema import Candidate, DataQuality, FirmLink, ResumeExtraction, ResumeFlag

# Order used when merging flags, most consequential first, so the top of the
# list is what a reviewer should read if they read nothing else.
FLAG_PRIORITY = {
    "internal_contradiction": 0,
    "date_anomaly": 1,
    "attribution_ambiguity": 2,
    "missing_data": 3,
    "typo_or_spelling": 4,
    "formatting": 5,
    "other": 6,
}

NON_PROFESSIONAL = {"student_organization", "volunteer", "academic", "side_venture"}


def _flag_tokens(flag: ResumeFlag) -> set[str]:
    """Content words of a flag's summary, for near-duplicate detection.

    Summary only. An earlier version pooled summary and detail, and the two
    sources phrase details so differently that shared subjects were drowned
    out: "Employment gap 2015-2017" and "Employment gap: 2y 1m before Goldman
    Sachs" survived as separate findings. The summary is where both sources
    name the same thing.
    """
    words = re.findall(r"[a-z]{4,}", flag.summary.lower())
    return {w for w in words if w not in _FLAG_STOPWORDS}


def _merge_flags(
    model_flags: list[ResumeFlag], computed_flags: list[ResumeFlag]
) -> list[ResumeFlag]:
    """Combine both sources, dropping near-duplicates.

    The two halves of the system genuinely overlap: the model reads "there is
    a gap between these two roles" while the timeline check computes the same
    gap from the dates. Reported twice, a reviewer stops trusting the list.

    Where both found the same thing we keep the computed one, because it
    carries exact figures ("2y 1m before Goldman Sachs") rather than prose.
    Similarity is measured on content words within a category, which is crude
    but errs toward keeping both -- a duplicate is untidy, a dropped finding
    is a defect.
    """
    merged = list(computed_flags)
    covered = {
        (f.category, topic)
        for f in computed_flags
        if (topic := _flag_topic(f)) is not None
    }

    for flag in model_flags:
        if (flag.category, _flag_topic(flag)) in covered:
            continue
        tokens = _flag_tokens(flag)
        duplicate = False
        for existing in merged:
            if existing.category != flag.category:
                continue
            other = _flag_tokens(existing)
            if not tokens or not other:
                continue
            if len(tokens & other) / min(len(tokens), len(other)) >= 0.5:
                duplicate = True
                break
        if not duplicate:
            merged.append(flag)
    return merged


# Topics where a computed check is exhaustive: if the calculation ran, it
# found everything there was to find about that topic, so a model flag on the
# same topic AND in the same category is restating it. Category is part of the
# test on purpose -- "Email address malformed" and "Name mismatch between
# header and email" both mention email but are different findings, and only
# the first is what the format check covers.
_FLAG_TOPICS: dict[str, str] = {
    "email": r"\bemail\b",
    "phone": r"\bphone\b",
    "gap": r"\bgaps?\b",
    "overlap": r"\boverlap|\bconcurrent\b",
}


def _flag_topic(flag: ResumeFlag) -> Optional[str]:
    summary = flag.summary.lower()
    for topic, pattern in _FLAG_TOPICS.items():
        if re.search(pattern, summary):
            return topic
    return None


_FLAG_STOPWORDS = {
    "resume", "candidate", "however", "which", "there", "their", "these",
    "this", "that", "with", "from", "have", "been", "does", "given", "into",
    "under", "listed", "stated", "appears", "written", "between", "while",
    "left", "exactly", "would", "guess", "text", "source", "entry", "role",
    "roles", "position", "positions", "explain", "interval", "recorded",
}


def _slug(name: Optional[str], fallback: str) -> str:
    base = name or fallback.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def name_from_filename(source_file: str) -> str:
    """Recover a display name from the filename.

    An early version passed the filename to the model alongside the text; the
    model used it on some runs and refused on others, because the instruction
    did not say which was correct. Ambiguous instructions produce
    non-deterministic output -- the bug was in the prompt, not the model.

    The filename is now withheld from the model entirely, and the fallback
    happens here: deterministic, and flagged in the record so the interface
    can show that this name was never stated by the candidate.
    """
    stem = re.sub(r"\.[^.]+$", "", source_file)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b(resume|cv|final|updated|\d{4,})\b", " ", stem, flags=re.I)
    return re.sub(r"\s+", " ", stem).strip().title()


# --------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------

def _resolve_location(
    extraction: ResumeExtraction, kb: KnowledgeBase
) -> tuple[Optional[str], Optional[str]]:
    """Return (location, region), falling back through weaker cues in order.

    A stated current location beats a current role's office, which beats any
    role's office, which beats the employer's home region. We stop at the
    first that yields a region so the strongest available evidence wins.
    """
    candidates: list[Optional[str]] = [extraction.location_raw]
    candidates += [p.location for p in extraction.positions if p.is_current]
    candidates += [p.location for p in extraction.positions]

    for value in candidates:
        if value and (region := kb.region_for(value)):
            return value, region

    # Last resort: the region of the most recent resolvable employer. The firm
    # table records a country ("India"); the taxonomy speaks in markets
    # ("APAC"). Mapping through region_for keeps one vocabulary -- without it
    # two Indian sell-side analysts landed in different filter buckets.
    for position in extraction.positions:
        match = kb.resolve_firm(position.firm)
        if match.region and match.region != "Global":
            return position.location, kb.region_for(match.region) or match.region

    return next((c for c in candidates if c), None), None


def _coverage_markets(
    extraction: ResumeExtraction, region: Optional[str]
) -> tuple[list[str], str]:
    """Markets the candidate covered, with the provenance of that answer.

    Most resumes state a sector but never a market: "20+ names across TMT"
    does not say whether those names are American. Leaving the field empty was
    honest but made it useless for filtering, so we fall back to the
    candidate's own region and record that the value was inferred. The
    interface shows the provenance, so nobody mistakes the fallback for a
    stated fact.
    """
    stated = [g for g in extraction.coverage.geographies_covered if g and g.strip()]
    if stated:
        return stated, "stated"
    if region:
        return [region], "inferred"
    return [], "unknown"


# --------------------------------------------------------------------------
# Employers
# --------------------------------------------------------------------------

def _employers(
    extraction: ResumeExtraction, firms: list[FirmLink]
) -> tuple[list[str], list[str]]:
    """Distinct employers with role types, and non-professional affiliations.

    The raw list was `Bank of China; Bank of China; ...; University Swimming
    Club` -- a firm repeated once per role, with student societies mixed in
    among real employers. Collapsing repeats and separating the two makes both
    halves usable: employers for filtering, affiliations for context.
    """
    professional: dict[str, set[str]] = {}
    other: dict[str, set[str]] = {}

    for position, link in zip(extraction.positions, firms):
        name = link.canonical or link.raw
        bucket = other if position.employment_type in NON_PROFESSIONAL else professional
        kind = (
            "intern"
            if position.employment_type == "internship"
            else "full-time"
            if position.employment_type == "professional"
            else position.employment_type.replace("_", " ")
        )
        bucket.setdefault(name, set()).add(kind)

    def render(bucket: dict[str, set[str]]) -> list[str]:
        out = []
        for name, kinds in bucket.items():
            ordered = [k for k in ("full-time", "intern") if k in kinds]
            ordered += sorted(k for k in kinds if k not in {"full-time", "intern"})
            out.append(f"{name} ({', '.join(ordered)})")
        return out

    return render(professional), render(other)


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------

# What each kind of problem costs the confidence score. Weights are relative
# judgements about how badly the issue damages a search result, kept in one
# table so they can be argued about and tuned in isolation.
_PENALTIES: dict[str, float] = {
    "no_name": 0.15,
    "no_positions": 0.40,
    "no_dates": 0.20,
    "partial_dates": 0.08,
    "no_location": 0.10,
    "unverified_quote": 0.08,
    "low_confidence_field": 0.05,
    "thin_extraction": 0.20,
    "internal_contradiction": 0.06,
    "date_anomaly": 0.04,
    "attribution_ambiguity": 0.04,
    "missing_data": 0.02,
    "typo_or_spelling": 0.02,
    "formatting": 0.0,  # a property of the file, never of the candidate
    "other": 0.02,
}


def assess_quality(
    extraction: ResumeExtraction,
    flags: list[ResumeFlag],
    *,
    extraction_chars: int,
    unverified_quotes: list[str],
) -> DataQuality:
    """Score how much to trust this record, and say why in scannable terms.

    `reasons` holds flag summaries rather than prose. An earlier version wrote
    paragraphs, which a reviewer could not scan: a missing phone number and a
    two-year employment gap read identically. Each reason is now a short
    keyword-led phrase, and the full explanation lives on the flag itself.

    Formatting problems carry zero weight. They describe the document, and how
    neatly a resume is formatted tracks regional convention and native
    language far more than it tracks the candidate.
    """
    score = 1.0
    missing: list[str] = []

    if not extraction.full_name:
        score -= _PENALTIES["no_name"]
        missing.append("full_name")

    if not extraction.positions:
        score -= _PENALTIES["no_positions"]
        missing.append("positions")
    else:
        dated = sum(1 for p in extraction.positions if p.start_date)
        if dated == 0:
            score -= _PENALTIES["no_dates"]
            missing.append("position_dates")
        elif dated < len(extraction.positions):
            score -= _PENALTIES["partial_dates"]

    if not extraction.location_raw:
        score -= _PENALTIES["no_location"]
        missing.append("location")
    if not extraction.education:
        missing.append("education")
    if not extraction.email:
        missing.append("email")

    if unverified_quotes:
        score -= min(0.20, _PENALTIES["unverified_quote"] * len(unverified_quotes))

    low_fields = [
        name
        for name, inferred in (
            ("investment_approach", extraction.investment_approach),
            ("market_side", extraction.market_side),
        )
        if inferred.confidence == "low"
    ]
    low_fields += [
        f"sector:{s.value}" for s in extraction.primary_sectors if s.confidence == "low"
    ]
    if low_fields:
        score -= min(0.15, _PENALTIES["low_confidence_field"] * len(low_fields))

    if extraction_chars < 1500:
        score -= _PENALTIES["thin_extraction"]

    # Each flag contributes by category, capped so that a resume with many
    # small notes is not pushed below one with a single disqualifying gap.
    flag_cost = sum(_PENALTIES.get(f.category, 0.02) for f in flags)
    score -= min(0.25, flag_cost)

    score = round(max(0.0, min(1.0, score)), 2)
    band = "high" if score >= 0.80 else "medium" if score >= 0.55 else "low"

    reasons = [f.summary for f in flags]
    if "full_name" in missing:
        reasons.insert(0, "Name taken from filename; never stated in the document")

    return DataQuality(
        score=score,
        band=band,
        reasons=reasons,
        missing_fields=missing,
        unverified_quotes=unverified_quotes,
    )


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def enrich(
    extraction: ResumeExtraction,
    source_file: str,
    *,
    extraction_report=None,
    extraction_chars: int = 0,
    unverified_quotes: Optional[list[str]] = None,
    kb: Optional[KnowledgeBase] = None,
) -> Candidate:
    """Combine one extraction with the knowledge base into a Candidate."""
    kb = kb or KnowledgeBase.load()
    unverified_quotes = unverified_quotes or []
    if extraction_report is not None:
        extraction_chars = extraction_chars or extraction_report.char_count

    # --- firms ------------------------------------------------------------
    firms: list[FirmLink] = []
    platforms: list[str] = []
    for position in extraction.positions:
        match = kb.resolve_firm(position.firm)
        firms.append(
            FirmLink(
                raw=match.raw,
                canonical=match.canonical,
                firm_type=match.firm_type,
                parent=match.parent,
                region=match.region,
                resolution=match.method,
                note=match.note,
            )
        )
        if match.canonical and position.employment_type == "professional":
            for platform in [match.canonical, *kb.platform_lineage(match.canonical)]:
                meta = kb.firms.get(platform, {})
                if meta.get("type") in {"multi_strategy_platform", "pod_shop"}:
                    parent = meta.get("parent") or platform
                    if parent not in platforms:
                        platforms.append(parent)

    firm_types = [f.firm_type for f in firms if f.firm_type]
    current = next(
        (
            (p, f)
            for p, f in zip(extraction.positions, firms)
            if p.is_current and p.employment_type == "professional"
        ),
        None,
    )

    # --- tenure and seniority --------------------------------------------
    positions = [p.model_dump() for p in extraction.positions]
    years = years_of_experience(positions)
    investment_years = years_of_experience(positions, investment_only=True)
    # Two different questions, deliberately not merged into one number.
    # `seniority_band` answers "how far into their career is this person",
    # which is total tenure. `investment_seniority_band` answers "how long
    # have they been investing", which is what a research requisition asks.
    # Vikram Shah is twelve years into a career and seven and a half into
    # investing; one number cannot say both.
    band = kb.seniority_for_years(years)
    investment_band = kb.seniority_for_years(investment_years)
    junior_lo, junior_hi = kb.junior_years_range

    location, region = _resolve_location(extraction, kb)
    markets, markets_source = _coverage_markets(extraction, region)
    employers, affiliations = _employers(extraction, firms)

    # --- flags: what the model read + what we computed ---------------------
    computed: list[ResumeFlag] = []
    computed += check_contact(extraction)
    computed += check_tenure_split(years, investment_years)
    computed += check_timeline(extraction)
    if extraction_report is not None:
        computed += check_formatting(extraction_report)
    if current:
        computed += check_seniority(
            band, kb.title_seniority_hint(current[0].title), years, current[0].title
        )
    for link in firms:
        if link.resolution == "ambiguous":
            computed.append(
                ResumeFlag(
                    category="attribution_ambiguity",
                    summary=f"Employer name ambiguous: {link.raw}",
                    detail=link.note or "Matches more than one known firm.",
                    quote=link.raw,
                )
            )
    # Tagged in one place rather than at each construction site -- an earlier
    # version tagged them individually and silently missed three.
    computed = [f.model_copy(update={"source": "computed"}) for f in computed]

    flags = _merge_flags(list(extraction.flags), computed)
    flags.sort(key=lambda f: FLAG_PRIORITY.get(f.category, 9))

    # --- credentials, expanded -------------------------------------------
    credentials: list[str] = []
    for credential in extraction.credentials:
        full = kb.expand_credential(credential.name)
        label = f"{credential.name} - {full}" if full else credential.name
        credentials.append(f"{label} ({credential.status})" if credential.status else label)

    stated_name = (extraction.full_name or "").strip()

    return Candidate(
        candidate_id=_slug(extraction.full_name, source_file),
        source_file=source_file,
        extraction=extraction,
        display_name=stated_name or name_from_filename(source_file),
        name_source="resume" if stated_name else "filename",
        region=region,
        location=location,
        coverage_markets=markets,
        coverage_markets_source=markets_source,
        years_experience=years,
        years_investment_experience=investment_years,
        seniority_band=band,
        investment_seniority_band=investment_band,
        is_junior_range=(
            None
            if investment_years is None
            else junior_lo <= investment_years <= junior_hi
        ),
        firms=firms,
        employers=employers,
        non_professional_affiliations=affiliations,
        current_firm=(current[1].canonical or current[1].raw) if current else None,
        current_firm_type=current[1].firm_type if current else None,
        firm_types=sorted(set(firm_types)),
        has_buy_side_experience=any(t in kb.buy_side_types for t in firm_types),
        has_sell_side_experience=any(t in kb.sell_side_types for t in firm_types),
        platform_alum_of=platforms,
        approach=extraction.investment_approach.value,
        approach_family=kb.approach_family(extraction.investment_approach.value),
        market_side=extraction.market_side.value,
        sectors=[s.value for s in extraction.primary_sectors],
        asset_classes=[a.value for a in extraction.asset_classes],
        credentials_summary=credentials,
        languages=extraction.languages,
        # Canonicalised, de-duplicated, order preserved. Without this the same
        # tool appears two or three times in a filter list.
        software_tools=list(
            dict.fromkeys(
                kb.normalise_software(tool) for tool in extraction.software_tools
            )
        ),
        methods=extraction.methods,
        flags=flags,
        quality=assess_quality(
            extraction,
            flags,
            extraction_chars=extraction_chars,
            unverified_quotes=unverified_quotes,
        ),
    )
