"""
Candidate schema.

This is the foundation of the whole platform: it defines what "a parsed
resume" means, and therefore what the search filters can possibly be. Getting
it wrong is expensive, so two principles shape it.

**Principle 1 - separate what is read from what is judged.**

Some fields are transcription: a firm name, a job title, a date. The model
should copy these, not interpret them. Other fields are judgement: is this
person fundamental or systematic? What sector do they really cover? These are
inferences, and an inference without its evidence is an unverifiable claim.

So every judged field is an `Inferred[...]` carrying a verbatim quote from the
resume plus a confidence level. The user interface shows that quote next to
the claim, so a recruiter can check our reasoning in one glance instead of
trusting a number. This is the difference between a score you believe and a
score you merely see.

**Principle 2 - do not ask the model for anything code can compute.**

Years of experience is arithmetic over dates we already extracted. Whether a
firm is a pod shop is a lookup in a curated table. Which region a city sits in
is a lookup. Asking an LLM for these invites silent, unauditable errors in
exchange for nothing. Every such field is therefore absent from this schema
and derived later in `enrich.py`, where it is deterministic, testable, and
free.

The schema the model actually sees is `ResumeExtraction`. `Candidate` is that
plus the derived fields, and is what the application consumes.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Controlled vocabularies
#
# Free-text categories are unfilterable: "Tech", "Technology", "TMT" and
# "Software" become four incompatible filter options. Constraining these to
# enums at extraction time is what makes the search UI possible at all.
# --------------------------------------------------------------------------

InvestmentApproach = Literal[
    "fundamental",     # bottom-up research, financial modelling, management meetings
    "systematic",      # rules-based / factor / algorithmic strategies
    "quantitative_dev",  # builds pricing or trading infrastructure, not a PM track
    "hybrid",          # substantive evidence of both
    "unclear",         # do not guess -- say so
]

MarketSide = Literal[
    "buy_side",
    "sell_side",
    "investment_banking",
    "consulting",
    "corporate",
    "academic",
    "unclear",
]

Sector = Literal[
    "technology", "media_telecom", "healthcare", "financials", "energy",
    "utilities", "industrials", "consumer", "materials", "real_estate",
    "credit", "macro_rates_fx", "generalist", "other",
]

AssetClass = Literal[
    "equities", "credit", "rates", "fx", "commodities",
    "derivatives", "multi_asset", "unclear",
]

Confidence = Literal["high", "medium", "low"]


# --------------------------------------------------------------------------
# Evidence wrapper
# --------------------------------------------------------------------------

class Inferred(BaseModel):
    """A judged value, the keywords behind it, and the text that justifies it.

    An earlier version carried only the value, a confidence label and a long
    verbatim sentence. Reviewers found it unusable: a 40-word quote in a table
    cell is not scannable, and "high" on its own says nothing about *why*.

    Three parts fix that. `keywords` gives the two or three terms that drove
    the call -- readable at a glance in a dense list. `evidence` keeps the
    verbatim sentence for anyone who wants to check. `confidence` then means
    something, because the reader can see what it is confident about.
    """

    value: str = Field(description="The inferred value, from the allowed set.")
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 SHORT key terms lifted from the evidence that drove this "
            "call, e.g. ['backtesting', 'multi-factor model', 'Sharpe 1.8'] "
            "or ['stocks under coverage', 'management meetings']. Each 1-4 "
            "words, taken from the resume's own wording. These are what a "
            "reviewer scans; the full quote is for verification."
        ),
    )
    evidence: str = Field(
        description=(
            "A short VERBATIM quote from the resume supporting this value. "
            "Must appear in the source text exactly. If no supporting text "
            "exists, return an empty string and set confidence to 'low'."
        )
    )
    confidence: Confidence = Field(
        description=(
            "high = explicitly stated; medium = strongly implied by concrete "
            "detail; low = weak or absent support."
        )
    )


# --------------------------------------------------------------------------
# Transcribed structures
# --------------------------------------------------------------------------

class Position(BaseModel):
    """One role. Transcribe; do not interpret."""

    firm: str = Field(description="Employer name exactly as written.")
    title: str = Field(description="Job title exactly as written.")
    location: Optional[str] = Field(
        default=None, description="City and/or country if stated."
    )
    start_date: Optional[str] = Field(
        default=None,
        description="ISO-like 'YYYY-MM' or 'YYYY'. Null if not stated. Do not guess.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="'YYYY-MM' or 'YYYY'. Null if this is the current role.",
    )
    is_current: bool = Field(
        default=False,
        description="True if described as present/current/'till date'.",
    )
    employment_type: Literal[
        "professional",
        "internship",
        "student_organization",
        "volunteer",
        "academic",
        "side_venture",
    ] = Field(
        default="professional",
        description=(
            "professional = substantive paid employment at a real employer. "
            "internship = intern, summer analyst, trainee. "
            "student_organization = any university club, society, chapter, "
            "campus service or student-run body -- e.g. 'President, Alumni "
            "Relations' at a fraternity chapter, an investment banking club, "
            "a campus tutoring service, a university sports club. If the "
            "organisation exists to serve students, it belongs here no matter "
            "how the role is titled. volunteer = unpaid or charitable work. "
            "academic = research assistant, teaching assistant, faculty. "
            "side_venture = the candidate's own venture held alongside "
            "full-time employment. Only 'professional' counts toward years of "
            "experience, so classify carefully -- misfiling a student society "
            "as professional adds years to a candidate's career that they "
            "did not work."
        ),
    )
    is_investment_role: bool = Field(
        default=False,
        description=(
            "True only if the role involved researching, recommending, or "
            "managing investments -- equity research, portfolio management, "
            "investment analysis. False for support, operations, business "
            "management, banking deal execution, consulting, or engineering "
            "roles, even at an investment firm. 'Office of the CFO, Strategy' "
            "at a bank is False; 'Equity Research Associate' is True."
        ),
    )
    duration_months: Optional[int] = Field(
        default=None,
        description=(
            "Only when the resume states a DURATION instead of dates, e.g. "
            "'8 years 10 months' -> 106, '2 months' -> 2. Convert to whole "
            "months. Null when start/end dates are given. Some resumes give "
            "durations only, and without this their tenure is uncomputable."
        ),
    )
    description: list[str] = Field(
        default_factory=list,
        description="Responsibility/achievement bullets, condensed but not paraphrased.",
    )


class Education(BaseModel):
    institution: str
    degree: Optional[str] = Field(
        default=None, description="e.g. 'BS', 'MS', 'MBA', 'PhD'."
    )
    field_of_study: Optional[str] = None
    start_year: Optional[int] = Field(
        default=None,
        description=(
            "Four-digit year the programme began, when the resume gives a "
            "range such as '2019 - 2021'. Null if only one year is shown. "
            "Needed to detect study running alongside full-time employment."
        ),
    )
    graduation_year: Optional[int] = Field(
        default=None,
        description="Four-digit completion year, null if not stated.",
    )


class Credential(BaseModel):
    """Professional qualification.

    `status` matters more than people assume: a CFA charterholder and someone
    who has passed Level III but lacks the work requirement are different
    candidates, and resumes state this inconsistently. We capture the
    distinction rather than flattening it to a boolean.
    """

    name: str = Field(description="e.g. 'CFA', 'FRM', 'CPA', 'Series 7'.")
    status: str = Field(
        description=(
            "Verbatim status, e.g. 'Charterholder', 'Level III passed', "
            "'Level II candidate', 'registered'. Empty if unstated."
        )
    )
    year: Optional[int] = None


class ResumeFlag(BaseModel):
    """One specific problem found in a resume.

    Replaces a free-text notes list. Prose notes could not be filtered,
    counted, sorted or shown compactly; a reviewer reading ten paragraphs of
    them cannot tell a missing phone number from a two-year employment gap.
    A category plus a short summary makes both possible.
    """

    category: Literal[
        "missing_data",
        "internal_contradiction",
        "attribution_ambiguity",
        "date_anomaly",
        "typo_or_spelling",
        "formatting",
        "other",
    ] = Field(description="The kind of problem.")
    severity: Literal["warning", "info"] = Field(
        default="warning",
        description=(
            "Set by the pipeline, never by the model: 'info' marks a flag "
            "kept for the record but judged benign -- by a deterministic "
            "rule or by human review."
        ),
    )
    summary: str = Field(
        description=(
            "A SHORT keyword-led phrase, at most 10 words, that reads well in "
            "a bullet list. Start with the subject: 'Coverage count "
            "inconsistent: 25 vs 32', 'Email missing domain suffix', "
            "'No dates on any position'. Not a sentence."
        )
    )
    detail: str = Field(
        description="One sentence of explanation. Say what is wrong, not what to do."
    )
    quote: str = Field(
        default="",
        description="Verbatim excerpt showing the problem, if one applies. Else empty.",
    )
    source: Literal["model", "computed"] = Field(
        default="model",
        description="Whether this was found by reading (model) or by "
        "calculation (computed). Set by the pipeline, not by the model.",
    )


class CoverageDetail(BaseModel):
    """What the candidate actually covered -- the currency of research hiring."""

    stocks_covered: Optional[int] = Field(
        default=None,
        description="Largest explicit count of names/stocks under coverage.",
    )
    geographies_covered: list[str] = Field(
        default_factory=list,
        description="Markets covered, e.g. ['India'], ['Greater China'], ['US'].",
    )
    notable_firms_or_names: list[str] = Field(
        default_factory=list,
        description="Specific companies or funds named as covered or transacted.",
    )


# --------------------------------------------------------------------------
# What the model returns
# --------------------------------------------------------------------------

class ResumeExtraction(BaseModel):
    """The exact structure the LLM is constrained to produce.

    Field descriptions here are not documentation -- they are the prompt. The
    model sees them, so they carry the instructions that would otherwise sit
    in a long system message, attached to the field they govern.
    """

    # --- transcription -----------------------------------------------------
    full_name: Optional[str] = Field(
        default=None,
        description=(
            "Candidate's full name. Some resumes omit a name header entirely; "
            "return null rather than inventing one from context."
        ),
    )
    email: Optional[str] = None
    phone: Optional[str] = None
    location_raw: Optional[str] = Field(
        default=None,
        description=(
            "Current location as written. If absent, infer ONLY from an "
            "explicit current-role location; otherwise null."
        ),
    )

    positions: list[Position] = Field(
        default_factory=list,
        description=(
            "All roles, most recent first. IMPORTANT: some resumes place work "
            "history under misleading headings such as 'ACADEMIC PROFILE' or "
            "'KEY PROJECTS'. Classify by content -- an entry naming an "
            "employer, a title and a duration is a position regardless of the "
            "section heading above it."
        ),
    )
    education: list[Education] = Field(default_factory=list)
    credentials: list[Credential] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    software_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Named software, platforms and programming languages the candidate "
            "can use: Bloomberg Terminal, FactSet, S&P Capital IQ, Excel, "
            "Python, R, SQL, C++, Tableau. Software only -- not methods like "
            "'financial modelling' or 'statistics'."
        ),
    )
    methods: list[str] = Field(
        default_factory=list,
        description=(
            "Analytical methods and techniques, as distinct from software: "
            "DCF, comparable company analysis, Monte Carlo, machine learning, "
            "backtesting, hypothesis testing."
        ),
    )

    # --- judgement, with evidence -----------------------------------------
    investment_approach: Inferred = Field(
        description=(
            "One of: fundamental, systematic, quantitative_dev, hybrid, unclear. "
            "Judge by what the person DID, not by vocabulary. Building factor "
            "models or backtesting is systematic. Building pricing libraries is "
            "quantitative_dev. Company meetings, DCFs and stock coverage are "
            "fundamental. The word 'quantitative' in a skills list is not "
            "evidence of a systematic approach."
        )
    )
    market_side: Inferred = Field(
        description=(
            "One of: buy_side, sell_side, investment_banking, consulting, "
            "corporate, academic, unclear. Reflect the MOST RECENT substantive "
            "role. Brokerages and equity research houses publishing to "
            "institutional clients are sell_side; asset managers and hedge "
            "funds investing capital are buy_side."
        )
    )
    primary_sectors: list[Inferred] = Field(
        default_factory=list,
        description=(
            "One to three sectors from: technology, media_telecom, healthcare, "
            "financials, energy, utilities, industrials, consumer, materials, "
            "real_estate, credit, macro_rates_fx, generalist, other. Order by "
            "depth of experience. Each needs its own verbatim evidence quote."
        ),
    )
    asset_classes: list[Inferred] = Field(
        default_factory=list,
        description=(
            "From: equities, credit, rates, fx, commodities, derivatives, "
            "multi_asset, unclear."
        ),
    )
    coverage: CoverageDetail = Field(default_factory=CoverageDetail)

    # --- self-reported quality signals -------------------------------------
    flags: list[ResumeFlag] = Field(
        default_factory=list,
        description=(
            "Every problem you found in this resume. Be thorough and specific "
            "-- these surface directly to the recruiter.\n"
            "Look for: missing contact details, name or dates; numbers that "
            "contradict each other elsewhere in the document; bullets that "
            "name a DIFFERENT employer than the heading they sit under; "
            "employment gaps; overlapping full-time roles; study dates "
            "overlapping full-time employment; misspellings; malformed emails; "
            "garbled or corrupted characters.\n"
            "Do NOT flag ordinary stylistic choices. Report only what would "
            "make a recruiter pause."
        ),
    )


# --------------------------------------------------------------------------
# Derived record
#
# Everything below is computed by `enrich.py` from the extraction plus the
# knowledge base. None of it is asked of the model, because all of it is
# either arithmetic or a lookup -- deterministic, testable and free.
# --------------------------------------------------------------------------

class DataQuality(BaseModel):
    """How much to trust this record, and why.

    The reasons matter more than the score. A recruiter shown "confidence
    0.62" learns nothing; one shown "no dates on any position" knows exactly
    which filter to distrust for this candidate.
    """

    score: float = Field(description="0-1 composite confidence.")
    band: Literal["high", "medium", "low"]
    reasons: list[str] = Field(
        default_factory=list, description="Specific, human-readable issues."
    )
    missing_fields: list[str] = Field(default_factory=list)
    unverified_quotes: list[str] = Field(
        default_factory=list,
        description="Evidence the model could not be shown to have copied verbatim.",
    )


class FirmLink(BaseModel):
    """A resolved employer, and how confidently it was resolved."""

    raw: str
    canonical: Optional[str] = None
    firm_type: Optional[str] = None
    parent: Optional[str] = None
    region: Optional[str] = None
    resolution: str = "unresolved"
    note: Optional[str] = None


class Candidate(BaseModel):
    """A parsed resume plus derived attributes -- what the app consumes."""

    candidate_id: str
    source_file: str
    extraction: ResumeExtraction

    display_name: str
    name_source: Literal["resume", "filename"] = "resume"

    # geography
    region: Optional[str] = None
    location: Optional[str] = None
    coverage_markets: list[str] = Field(
        default_factory=list,
        description="Markets the candidate covered. Falls back to their own "
        "region when the resume never states one.",
    )
    coverage_markets_source: Literal["stated", "inferred", "unknown"] = "unknown"

    # tenure
    years_experience: Optional[float] = None
    years_investment_experience: Optional[float] = None
    seniority_band: Optional[str] = Field(
        default=None,
        description="Career stage, from TOTAL professional tenure.",
    )
    investment_seniority_band: Optional[str] = Field(
        default=None,
        description="Seniority as an investor, from investment tenure only. "
        "Differs from seniority_band for anyone who came in from banking, "
        "consulting, engineering or operations.",
    )
    is_junior_range: Optional[bool] = None

    # firms
    firms: list[FirmLink] = Field(default_factory=list)
    current_firm: Optional[str] = None
    current_firm_type: Optional[str] = None
    firm_types: list[str] = Field(default_factory=list)
    has_buy_side_experience: bool = False
    has_sell_side_experience: bool = False

    # the single highest-value BD signal in this corpus
    platform_alum_of: list[str] = Field(
        default_factory=list,
        description="Platforms this candidate has worked at, via pod lineage.",
    )

    # flattened for filtering
    approach: Optional[str] = None
    approach_family: Optional[str] = Field(
        default=None,
        description="Coarse axis used by requisitions: fundamental | "
        "systematic_quant | hybrid | unclear.",
    )
    market_side: Optional[str] = None
    sectors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)
    credentials_summary: list[str] = Field(
        default_factory=list,
        description="Credentials with codes expanded, e.g. 'Series 7 - General "
        "Securities Representative (active holder)'.",
    )
    languages: list[str] = Field(default_factory=list)
    software_tools: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)

    # Employment history, de-duplicated and annotated. The raw list repeated a
    # firm once per role and mixed student societies in with employers.
    employers: list[str] = Field(
        default_factory=list,
        description="Distinct professional employers, e.g. 'Bank of China "
        "(full-time, intern)'.",
    )
    non_professional_affiliations: list[str] = Field(
        default_factory=list,
        description="Student societies, volunteering, academic and side ventures.",
    )

    flags: list[ResumeFlag] = Field(
        default_factory=list,
        description="Merged model-found and computed problems, most severe first.",
    )

    quality: DataQuality
