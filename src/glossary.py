"""
Field and control definitions surfaced in the interface.

Every column header, filter and metric in the app carries an explanation from
this module. That is a deliberate constraint rather than a nicety: the first
reviewer of this interface could not tell what "Fit", "coverage" or
"in junior range" meant, and a number a user cannot interpret is worse than
no number -- they will either ignore it or, worse, act on a guess about it.

Keeping the text here rather than inline in `app.py` means one place to fix
wording, and makes it obvious when a field has been added without one.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Result-table columns
# ---------------------------------------------------------------------------

COLUMN_HELP: dict[str, str] = {
    "Candidate": "Name as stated in the resume. Where a resume states no name, "
    "the filename is used and the record is flagged.",
    "Fit": "Weighted score across the soft signals, 0-100%. It ranks only "
    "candidates who already satisfy every hard requirement — it is never a "
    "reason to accept someone who failed one. See the Method tab for the "
    "exact formula.",
    "Gap": "The single hard requirement this candidate does not meet — blank "
    "for candidates who meet them all. Hard requirements disqualify; they "
    "are not traded off against a high score.",
    "Yrs investing": "Years spent in roles that actually researched or managed "
    "investments. 0.0 means none; a blank means the resume gave no dates to "
    "compute it from.",
    "Yrs career": "Total professional tenure, excluding internships, student "
    "societies and volunteering.",
    "Region": "US / Europe / APAC / MEA / LatAm, from the candidate's stated "
    "location, falling back to their employer's region.",
    "Sector": "Sectors the candidate covered, each backed by a quote from the "
    "resume.",
    "Approach": "Fundamental or Systematic / Quantitative — the axis the case "
    "brief frames the pool on. Judged from what the candidate did, not from "
    "vocabulary.",
    "Data": "Parse confidence: how completely the pipeline could read this "
    "document. A property of the file, not of the candidate.",
    "Firm": "Current employer, resolved against the firm knowledge base.",
    "Firm type": "Operating model of the current employer — pod shop, "
    "multi-strategy platform, long-only, sell-side research, and so on.",
    "Market side": "Buy side (deploys capital), sell side (publishes "
    "research), investment banking, consulting or corporate.",
    "Platform alum": "Multi-manager platforms this candidate has worked at "
    "or under, resolved through pod-to-platform lineage.",
    "Coverage": "Largest stated number of names under research coverage; "
    "blank when the resume never says.",
    "Software": "Named tools, normalised — 'Bloomberg Terminal' and "
    "'Bloomberg' count as one.",
    "Credentials": "Professional qualifications and business-school "
    "degrees, normalised (MBA / PGDM are one bucket).",
    "Languages": "Languages the resume states.",
    "Title": "Current (or most recent) job title, as written in the resume.",
    "Asset classes": "Asset classes the candidate has worked in — equities, "
    "credit, rates/FX, commodities — each backed by a resume quote.",
    "Stated numbers": "How many performance, AUM or risk figures the resume "
    "states, classified by the model with verbatim quotes. Self-reported "
    "and unverified — shown on the profile, never scored.",
    "Leadership": "Largest people-leadership responsibility the resume "
    "states, with its quote. Never inferred from a title alone.",
    "Flags": "Count of open review flags on this record — contradictions, "
    "gaps, malformed details. Notes triaged as benign are not counted.",
}

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

FILTER_HELP: dict[str, str] = {
    "region": "Where the candidate is now. Counts beside each option update "
    "as you narrow the other filters.",
    "approach_family": "Fundamental vs Systematic / Quantitative. A quant "
    "developer who has never held a position is grouped under Systematic / "
    "Quantitative for recall, and excluded by any experience requirement.",
    "sectors": "Sectors covered. A candidate may carry more than one; "
    "selecting several matches any of them.",
    "market_side": "Buy side (deploys capital), sell side (publishes research "
    "to institutional clients), investment banking, consulting, corporate.",
    "asset_classes": "Asset classes worked in — equities, credit, rates, "
    "derivatives — each judged from what the candidate did, with a resume "
    "quote behind it.",
    "software_tools": "Named tools from the resume, normalised — 'Bloomberg "
    "Terminal' and 'Bloomberg' count as one.",
    "credentials_summary": "Professional qualifications, with licence codes "
    "expanded to their registered names.",
    "years": "Years of investment experience. The upper handle at maximum "
    "means no upper limit, so the control keeps working as the pool grows.",
    "unknown": "Some resumes state durations but no dates, so tenure cannot "
    "be computed. Excluding them hides real candidates; including them puts "
    "unverified numbers in your results. Shown either way, never silently.",
    "alum": "Has worked at a multi-manager platform or one of its pods — "
    "resolved through the firm knowledge base, since resumes name the pod "
    "rather than the platform.",
    "quality": "Minimum parse confidence. Raising it hides candidates whose "
    "resumes could not be read completely.",
    "prefer_quality": "Sorts well-parsed records above equally good matches. "
    "Useful when the shortlist goes to someone else: a strong match built on "
    "a half-read resume wastes their time.",
    "keyword": "Free-text search across employers, titles, bullets, sectors "
    "and tools.",
}

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

METRIC_HELP: dict[str, str] = {
    "fit": "Weighted soft-signal score against this requisition. Only "
    "meaningful for candidates who pass every hard requirement — for a near "
    "miss it says how good the rest of the profile is, not that they qualify.",
    "career": "Total professional tenure. Internships, student societies and "
    "volunteering are excluded — counting them added four years to one "
    "candidate in this pool.",
    "investing": "Tenure in investment roles only. Differs from career length "
    "for anyone who arrived from banking, consulting or engineering.",
    "coverage": "The largest number of names the candidate states having "
    "under research coverage — the standard measure of breadth in equity "
    "research. Blank when the resume never says.",
    "quality": "Parse confidence for this record, 0-1, with the specific "
    "issues listed below.",
    "issues": "Problems found in the resume: contradictions and "
    "misattributions read by the model, plus gaps, overlaps and malformed "
    "contact details computed from the data.",
}

# ---------------------------------------------------------------------------
# Score components, in the order they appear on the radar
# ---------------------------------------------------------------------------

CRITERION_LABEL: dict[str, str] = {
    "sector_fit": "Sector",
    "requirement_fit": "Requirements",
    "skills_fit": "Skills",
    "firm_type_fit": "Firm type",
    "coverage_depth": "Coverage",
    "credentials": "Credentials",
    "platform_alum": "Platform",
    "buy_side": "Buy side",
}

CRITERION_HELP: dict[str, str] = {
    "sector_fit": "How central the candidate's sectors are to the ones this "
    "requisition names.",
    "requirement_fit": "Mean similarity between the requisition's written "
    "requirements and the candidate's own sentences, matched through a "
    "curated concept map so 'catalysts' matches 'earnings events'.",
    "skills_fit": "Share of the named software and analytical methods the "
    "candidate holds.",
    "firm_type_fit": "Whether they have worked in a comparable operating "
    "model — a pod shop, a long-only manager, a research house.",
    "coverage_depth": "Names under coverage, relative to a 40-stock "
    "benchmark.",
    "credentials": "Whether the preferred qualifications are held.",
    "platform_alum": "Prior multi-manager platform experience.",
    "buy_side": "Prior buy-side experience.",
}
