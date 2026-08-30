"""
Deterministic resume checks.

The model reports what it *reads* -- contradictions in wording, a bullet that
names a different employer than its heading, an obvious misspelling. This
module reports what can be *computed*: employment gaps, overlapping roles,
malformed contact details, and the mechanical damage found while extracting
the document.

The split follows the rule the schema is built on. Date arithmetic is exactly
where language models fail quietly: asked whether two roles overlap, a model
will answer confidently and sometimes wrongly, with no way to audit it. Asked
whether a sentence contradicts another sentence, it does well and code does
badly. So each side does what it is good at.

Both sides emit the same `ResumeFlag` type, so a reviewer sees one merged list
and never has to care which half of the system found a given problem.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from schema import Education, Position, ResumeExtraction, ResumeFlag

# A gap shorter than this is ordinary -- notice periods, a move, a short
# break. Beyond it, a recruiter will want to ask.
GAP_THRESHOLD_MONTHS = 6

# Overlaps below this are just imprecise month boundaries on adjacent roles,
# not genuine concurrent employment.
OVERLAP_THRESHOLD_MONTHS = 2


def _months(value: Optional[str]) -> Optional[int]:
    """'YYYY-MM' or 'YYYY' to absolute months; year-only is treated as mid-year."""
    if not value:
        return None
    match = re.match(r"(\d{4})(?:-(\d{1,2}))?", str(value).strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else 6
    if not 1900 <= year <= 2100 or not 1 <= month <= 12:
        return None
    return year * 12 + month


def _label(position: Position) -> str:
    return f"{position.title} at {position.firm}"


def _fmt(months: int) -> str:
    return f"{months // 12}-{months % 12:02d}"


# --------------------------------------------------------------------------
# Contact details
# --------------------------------------------------------------------------

# Deliberately permissive: we are looking for structurally broken addresses,
# not policing exotic but valid ones.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def check_contact(extraction: ResumeExtraction) -> list[ResumeFlag]:
    """Validate contact details without ever repairing them.

    One resume here gives 'rchen@hotmail' -- no top-level domain. The obvious
    "fix" is to append .com, and it is the wrong thing to do: the address
    could equally be .co.uk. A recruiter who emails a fabricated address and
    hears nothing back learns nothing; one told the address is incomplete
    knows to go and check. We flag, we never invent.
    """
    flags: list[ResumeFlag] = []
    email = (extraction.email or "").strip()

    if not email:
        flags.append(
            ResumeFlag(
                category="missing_data",
                summary="No email address",
                detail="No email address appears anywhere in the resume.",
            )
        )
    elif not _EMAIL.match(email):
        problem = (
            "missing a top-level domain such as .com"
            if "@" in email and "." not in email.split("@")[-1]
            else "not a well-formed address"
        )
        flags.append(
            ResumeFlag(
                category="typo_or_spelling",
                summary="Email address malformed",
                detail=(
                    f"The address as written is {problem}. Left exactly as it "
                    "appears -- completing it would be a guess."
                ),
                quote=email,
            )
        )

    if not (extraction.phone or "").strip():
        flags.append(
            ResumeFlag(
                category="missing_data",
                summary="No phone number",
                detail="No phone number appears anywhere in the resume.",
            )
        )
    return flags


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------

def check_timeline(
    extraction: ResumeExtraction, *, as_of: Optional[date] = None
) -> list[ResumeFlag]:
    """Find gaps, overlaps and impossible dates across the work history."""
    today = as_of or date.today()
    now = today.year * 12 + today.month
    flags: list[ResumeFlag] = []

    spans: list[tuple[int, int, Position]] = []
    for position in extraction.positions:
        start = _months(position.start_date)
        if start is None:
            continue
        end = now if position.is_current else _months(position.end_date)
        if end is None:
            end = now
        if end < start:
            flags.append(
                ResumeFlag(
                    category="date_anomaly",
                    summary=f"End date precedes start: {position.firm}",
                    detail=(
                        f"{_label(position)} is dated "
                        f"{position.start_date} to {position.end_date}."
                    ),
                )
            )
            continue
        if start > now:
            flags.append(
                ResumeFlag(
                    category="date_anomaly",
                    summary=f"Start date in the future: {position.firm}",
                    detail=f"{_label(position)} starts {position.start_date}.",
                )
            )
        spans.append((start, end, position))

    professional = sorted(
        (s for s in spans if s[2].employment_type == "professional"),
        key=lambda s: s[0],
    )

    # Gaps between consecutive professional roles.
    covered_to = None
    for start, end, position in professional:
        if covered_to is not None and start - covered_to >= GAP_THRESHOLD_MONTHS:
            gap = start - covered_to
            flags.append(
                ResumeFlag(
                    category="date_anomaly",
                    summary=(
                        f"Employment gap: {gap // 12}y {gap % 12}m before "
                        f"{position.firm}"
                    ),
                    detail=(
                        f"No recorded role between {_fmt(covered_to)} and "
                        f"{_fmt(start)}, ending before {_label(position)} "
                        "began. The resume does not explain the interval."
                    ),
                )
            )
        covered_to = max(covered_to or end, end)

    # Concurrent professional roles.
    for i, (start_a, end_a, pos_a) in enumerate(professional):
        for start_b, end_b, pos_b in professional[i + 1 :]:
            overlap = min(end_a, end_b) - max(start_a, start_b)
            if overlap >= OVERLAP_THRESHOLD_MONTHS:
                flags.append(
                    ResumeFlag(
                                category="date_anomaly",
                        summary=(
                            f"Concurrent full-time roles: {pos_a.firm} / "
                            f"{pos_b.firm}"
                        ),
                        detail=(
                            f"{_label(pos_a)} and {_label(pos_b)} overlap by "
                            f"about {overlap} months, both recorded as "
                            "professional employment."
                        ),
                    )
                )

    flags.extend(_check_study_overlap(extraction.education, professional))
    return flags


def _check_study_overlap(
    education: list[Education], professional: list[tuple[int, int, Position]]
) -> list[ResumeFlag]:
    """Full-time study running alongside full-time work.

    Not necessarily a problem -- part-time and executive programmes are
    common, and so is finishing a degree while starting a job. It is worth
    surfacing because the resume rarely says which, and a recruiter reading
    "5 years' experience" should know part of it ran alongside a degree.
    """
    flags: list[ResumeFlag] = []
    for degree in education:
        start = _months(str(degree.start_year)) if degree.start_year else None
        end = _months(str(degree.graduation_year)) if degree.graduation_year else None
        if start is None or end is None or end <= start:
            continue
        for work_start, work_end, position in professional:
            overlap = min(end, work_end) - max(start, work_start)
            if overlap >= GAP_THRESHOLD_MONTHS:
                flags.append(
                    ResumeFlag(
                                category="date_anomaly",
                        summary=(
                            f"Study overlaps employment: "
                            f"{degree.degree or 'degree'} / {position.firm}"
                        ),
                        detail=(
                            f"{degree.degree or 'A degree'} at "
                            f"{degree.institution} ({degree.start_year}-"
                            f"{degree.graduation_year}) overlaps "
                            f"{_label(position)} by about {overlap} months. "
                            "The resume does not state whether the programme "
                            "was part-time."
                        ),
                    )
                )
                break
    return flags


# --------------------------------------------------------------------------
# Document mechanics
# --------------------------------------------------------------------------

def check_formatting(report) -> list[ResumeFlag]:
    """Turn extraction diagnostics into reviewer-facing flags.

    These describe the FILE, not the candidate. That distinction is kept
    deliberately: how neatly a resume is formatted correlates with regional
    convention and native language, so it is reported as a property of the
    document for a human to weigh, never folded into a candidate score.
    """
    flags: list[ResumeFlag] = []

    if report.replacement_chars_remaining:
        flags.append(
            ResumeFlag(
                category="formatting",
                summary=(
                    f"{report.replacement_chars_remaining} corrupted "
                    "character(s) unrecoverable"
                ),
                detail=(
                    "The source PDF has a damaged character map. Ligature "
                    f"damage was repaired in {report.ligature_repairs} place(s); "
                    "the remainder could not be resolved without guessing."
                ),
            )
        )
    if report.multi_column_detected:
        flags.append(
            ResumeFlag(
                category="formatting",
                summary="Two-column layout",
                detail=(
                    "Columns were separated before reading. Read naively, the "
                    "sidebar interleaves into the work-experience text."
                ),
            )
        )
    if report.textbox_count:
        flags.append(
            ResumeFlag(
                category="formatting",
                summary=f"{report.textbox_count} floating text box(es)",
                detail=(
                    "Content sits in floating text boxes, which standard "
                    "paragraph-and-table extraction does not read at all."
                ),
            )
        )
    if report.table_share > 0.5:
        flags.append(
            ResumeFlag(
                category="formatting",
                summary=f"{report.table_share:.0%} of content inside tables",
                detail=(
                    "The document is built from tables rather than flowing "
                    "text. This is a regional formatting convention, not a "
                    "defect."
                ),
            )
        )
    return flags


def check_tenure_split(
    total: Optional[float], investment: Optional[float]
) -> list[ResumeFlag]:
    """Flag a large gap between career length and investment experience.

    Someone twelve years into a career with seven of them investing arrived
    from banking, consulting or engineering. That is not a defect -- it is
    often exactly who a business development team wants -- but a reader
    scanning "12 years" will assume twelve years of investing unless told
    otherwise.

    A definite zero gets its own message. A quantitative developer with four
    dated years and no investment role has precisely no investment experience,
    which is a fact worth stating plainly rather than leaving as a dash.
    """
    if total is None or investment is None:
        return []
    if investment == 0:
        return [
            ResumeFlag(
            severity="info",
                category="other",
                summary=f"No investment experience: {total}y career, 0y investing",
                detail=(
                    "Every dated role is non-investment work -- engineering, "
                    "support or operations. For an investment-analyst "
                    "requisition this candidate has no directly relevant "
                    "tenure, though the background may still be adjacent."
                ),
            )
        ]
    if total - investment >= 2.0:
        return [
            ResumeFlag(
            severity="info",
                category="other",
                summary=(
                    f"Career longer than investing tenure: {total}y vs "
                    f"{investment}y"
                ),
                detail=(
                    f"{round(total - investment, 1)} years were spent in "
                    "non-investment roles. Screening on total tenure alone "
                    "would overstate this candidate's investing experience."
                ),
            )
        ]
    return []


def check_seniority(
    band: Optional[str], title_hint: Optional[str], years: Optional[float], title: str
) -> list[ResumeFlag]:
    """Flag a title that disagrees sharply with computed tenure."""
    order = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "portfolio_manager": 4}
    if band is None or title_hint is None:
        return []
    if abs(order.get(band, 2) - order.get(title_hint, 2)) < 2:
        return []
    return [
        ResumeFlag(
            category="internal_contradiction",
            summary=f"Title reads {title_hint}, tenure is {band}",
            detail=(
                f"The current title '{title}' reads as {title_hint}-level, "
                f"while {years} years of computed experience places the "
                f"candidate in the {band} band. Not necessarily a problem — "
                "titles are not comparable across firms or market sides — "
                "but worth a question in screening: why has the title not "
                "progressed with the tenure?"
            ),
        )
    ]


# --------------------------------------------------------------------------
# Credential pairing
# --------------------------------------------------------------------------

def check_credential_pairs(extraction: ResumeExtraction) -> list[ResumeFlag]:
    """Flag FINRA credentials whose required partner is missing.

    The model once flagged a Series 87 as "an unusual license to list
    alongside Series 7 and 63" -- plausible-sounding and wrong: 7 + 63 +
    86/87 is the standard sell-side research analyst stack. The genuine
    anomaly is different and perfectly mechanical: the research analyst
    qualification is the 86/87 PAIR (86 analysis, 87 regulatory), so an 87
    with no 86 usually means an omission or a typo. Membership of a known
    pair is a set lookup, not a judgment call, so it moves out of the
    model's hands and into code -- where the reasoning cannot drift.
    """
    names = " ".join(c.name.lower() for c in extraction.credentials)
    held = set(re.findall(r"series\s*(\d+)", names))
    flags: list[ResumeFlag] = []
    for present, missing in (("87", "86"), ("86", "87")):
        if present in held and missing not in held:
            flags.append(ResumeFlag(
                category="typo_or_spelling",
                summary=f"Series {present} listed without its paired "
                        f"Series {missing}",
                detail="FINRA's research analyst qualification is the 86/87 "
                       "pair -- 86 covers analysis, 87 the regulatory "
                       "portion. Listing one without the other usually "
                       "indicates an omission or a typo in the resume, not "
                       "a partial qualification.",
                quote=next((c.name for c in extraction.credentials
                            if present in c.name), ""),
                source="computed",
            ))
    return flags


def check_location_zip(
    extraction: ResumeExtraction, city_zip_prefixes: dict[str, list[str]]
) -> list[ResumeFlag]:
    """Flag a US header zip that does not belong to the stated city.

    A reviewer caught "Boston, MA 01125" by eye -- a valid Massachusetts
    zip, but a Springfield-area one, so state-level validation passes it
    and only city knowledge catches it. City-to-zip-prefix is curated fact,
    which puts it in the knowledge base and this check in code. Only cities
    present in the table are checked; everything else stays silent rather
    than guessing.
    """
    raw = extraction.location_raw or ""
    m = re.search(r"([A-Za-z .]+?),\s*[A-Z]{2}\s+(\d{5})", raw)
    if not m:
        return []
    city, zip_code = m.group(1).strip().lower(), m.group(2)
    prefixes = city_zip_prefixes.get(city)
    if not prefixes or any(zip_code.startswith(p) for p in prefixes):
        return []
    return [ResumeFlag(
        category="typo_or_spelling",
        summary=f"Zip code {zip_code} does not match {m.group(1).strip()}",
        detail=(
            f"The header reads '{raw}', but {zip_code} is not a "
            f"{m.group(1).strip()} zip (expected prefixes: "
            f"{', '.join(prefixes)}xx). Likely a typo; left as written."
        ),
        quote=raw,
        source="computed",
    )]
