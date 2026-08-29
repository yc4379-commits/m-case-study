"""
Talent Intelligence Platform -- Streamlit application.

A sourcing workbench for the Business Development team: search a parsed
candidate pool against a job requisition, see why each candidate matched, and
see how far to trust the underlying data.

Three decisions shape the interface.

**Requisition-first.** The brief asks for search "based on job requisitions",
so a requisition is the primary object rather than a set of dropdowns the user
happens to tick. Picking one applies its hard constraints, ranks the survivors,
and -- critically -- reports honestly when nobody qualifies.

**Evidence beside every claim.** No score appears without the resume text that
produced it. A recruiter checks the quote, not the number.

**Data quality is visible, not hidden.** Every candidate card carries its parse
confidence, and a whole tab is given to what the pipeline could not read. A
tool that quietly presents uncertain data as fact is worse than one that
presents less.

The app reads a pre-built JSON dataset and calls no model at runtime: parsing
happens offline in `src/build_dataset.py`. That keeps the deployment free of
API keys, instant to load, and is how this would run at scale anyway.

    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from match import Requisitions, match_all  # noqa: E402

DATA = ROOT / "data" / "candidates.json"

# ---------------------------------------------------------------------------
# Palette
#
# Two categorical slots (blue, orange) validated together for colour-vision
# deficiency and contrast against the app surface; a single-hue blue ramp for
# magnitude; a fixed status set for data quality that is never reused as a
# series colour, so a status hue can never impersonate a category.
# ---------------------------------------------------------------------------
# Brand chrome: a deep institutional navy on white. Used for headers, rules,
# and accents -- never for a data series. As a series colour the navy fails
# the palette validator on two hard checks (lightness 0.31 against a
# 0.43-0.77 band; chroma 0.09 against a 0.10 floor), which in practice means
# it reads as grey in a chart and collides with any dark neighbour.
NAVY = "#0b2f5e"
NAVY_SOFT = "#1c4a86"
NAVY_WASH = "#f4f7fb"

# Data series: slate blue and bronze. Chosen over the brighter blue/orange
# default for register -- this reads as an institutional document rather than
# a consumer dashboard -- and then held to the same gate. Validated against
# this white surface: lightness band, chroma floor, colour-vision separation
# (worst adjacent pair Delta E 20.9 protan / 24.5 tritan), normal-vision floor
# (25.9) and 3:1 contrast all pass. Muting further fails the chroma floor and
# the pair starts reading as two greys.
SERIES_1 = "#2f5f98"
SERIES_2 = "#b5793a"

# Sequential ramp built from the brand navy, light to dark, for magnitude.
SEQ = [
    "#eef3f9", "#d5e1ef", "#b4c9e1", "#8badd0",
    "#5d8bba", "#2f5f98", "#123a6f",
]
STATUS_GOOD = "#0ca30c"
STATUS_WARN = "#fab219"
STATUS_BAD = "#d03b3b"
# Parse-confidence bands and pass/fail verdicts draw on the same fixed status
# ramp but are distinct vocabularies; keeping them in separate names stops one
# from being indexed with the other's keys.
QUALITY_COLOUR = {"high": STATUS_GOOD, "medium": STATUS_WARN, "low": STATUS_BAD}
INK = "#0b1b2b"
MUTED = "#64748b"
GRID = "#e2e8f0"
SURFACE = "#ffffff"

st.set_page_config(
    page_title="Talent Intelligence Platform",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .stApp {{ background: {SURFACE}; }}
      section[data-testid="stSidebar"] {{
        background: {NAVY_WASH};
        border-right: 1px solid {GRID};
      }}
      h1, h2, h3, h4, h5 {{ color: {NAVY} !important; letter-spacing: -.01em; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {GRID}; }}
      .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {MUTED}; }}
      .stTabs [aria-selected="true"] {{ color: {NAVY} !important; }}
      [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 600; }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      hr {{ border-color: {GRID}; }}
      .brandbar {{
        background: {NAVY}; color: #fff; padding: 11px 18px; border-radius: 8px;
        margin-bottom: 14px; display: flex; align-items: baseline; gap: 12px;
      }}
      .brandbar b {{ font-size: 15px; letter-spacing: .01em; }}
      .brandbar span {{ font-size: 12px; opacity: .78; }}
      .recordtbl {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
      .recordtbl td {{ padding: 5px 8px; border-bottom: 1px solid {GRID};
                       vertical-align: top; }}
      .recordtbl td:first-child {{ color: {MUTED}; width: 42%; white-space: nowrap; }}
      .postbl {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
      .postbl th {{ text-align: left; color: {MUTED}; font-weight: 600;
                    padding: 5px 7px; border-bottom: 1px solid {GRID}; }}
      .postbl td {{ padding: 5px 7px; border-bottom: 1px solid {GRID};
                    vertical-align: top; }}
      .res {{ display: block; font-size: 10px; color: {MUTED};
              text-transform: uppercase; letter-spacing: .05em; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brandbar"><b>Talent Intelligence Platform</b>'
    "<span>Business Development · candidate search and requisition matching"
    "</span></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_candidates() -> list[dict]:
    if not DATA.exists():
        return []
    return json.loads(DATA.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_requisitions() -> Requisitions:
    return Requisitions.load()


candidates = load_candidates()
if not candidates:
    st.error(
        "No dataset found. Build it first:  `python src/build_dataset.py`"
    )
    st.stop()

store = load_requisitions()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def quality_chip(band: str, score: float) -> str:
    colour = QUALITY_COLOUR.get(band, MUTED)
    return (
        f"<span style='background:{colour}1f;color:{colour};padding:1px 8px;"
        f"border-radius:9px;font-size:11px;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:.05em'>{band} {score}</span>"
    )


def label_of(value: str) -> str:
    return value.replace("_", " ").title() if value else "—"


def styled_chart(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply the shared chart chrome: recessive axes, no chart junk."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=12,
            color=INK,
        ),
        # Styling a title without giving it text makes Plotly render the
        # literal string "undefined" above the plot. Carry through whatever
        # text the caller set -- most charts here are titled in Streamlit
        # markdown instead, and want no in-chart title at all.
        title=dict(
            text=fig.layout.title.text or "",
            font=dict(size=13, color=INK),
            x=0,
            xanchor="left",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=fig.layout.showlegend,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=MUTED),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar -- requisition, then filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Requisition")

titles = store.titles
choice = st.sidebar.selectbox(
    "Search against",
    options=["__none__", *titles.keys()],
    format_func=lambda k: "Browse all candidates" if k == "__none__" else titles[k],
    label_visibility="collapsed",
)
requisition = store.get(choice) if choice != "__none__" else None

if requisition:
    hard = requisition.get("hard", {})
    bits = []
    if r := hard.get("regions"):
        bits.append(f"<b>Region</b> {' / '.join(r)}")
    if a := hard.get("approach_families"):
        bits.append(f"<b>Approach</b> {' / '.join(label_of(x) for x in a)}")
    if s := hard.get("sectors_any"):
        bits.append(f"<b>Sector</b> {' / '.join(label_of(x) for x in s)}")
    if y := hard.get("investment_years"):
        bits.append(
            f"<b>Experience</b> {y.get('min', 0)}–{y.get('max', 99)} yrs investing"
        )
    st.sidebar.caption(requisition["summary"])
    st.sidebar.markdown(
        "<div style='font-size:12px;line-height:1.9'>Must satisfy:<br>"
        + "<br>".join(f"· {b}" for b in bits)
        + "</div>",
        unsafe_allow_html=True,
    )

st.sidebar.divider()
st.sidebar.markdown("### Filters")


def options_for(key: str) -> list[str]:
    values = set()
    for c in candidates:
        value = c.get(key)
        if isinstance(value, list):
            values.update(v for v in value if v)
        elif value:
            values.add(value)
    return sorted(values)


def count_with(key: str, value: str) -> int:
    total = 0
    for c in candidates:
        found = c.get(key)
        if (isinstance(found, list) and value in found) or found == value:
            total += 1
    return total


# Live counts beside every option. A recruiter should know a filter will empty
# the list before they click it, not after.
def multi(label: str, key: str) -> list[str]:
    opts = options_for(key)
    return st.sidebar.multiselect(
        label,
        options=opts,
        format_func=lambda v: f"{label_of(v)}  ({count_with(key, v)})",
    )


f_region = multi("Region", "region")
f_family = multi("Approach", "approach_family")
f_sector = multi("Sector", "sectors")
f_side = multi("Market side", "market_side")


def multi_contains(label: str, key: str, placeholder: str) -> list[str]:
    """Filter on list-valued fields whose members vary in spelling.

    Software and credentials are free-form: one resume writes "Bloomberg
    Terminal", another "Bloomberg". Options are the distinct values actually
    present, and matching is substring-based so picking "Bloomberg" catches
    both.
    """
    opts = sorted({v for c in candidates for v in c.get(key, []) if v})
    return st.sidebar.multiselect(label, options=opts, placeholder=placeholder)


f_software = multi_contains("Software and tools", "software_tools", "Any")
f_credentials = multi_contains("Credentials", "credentials_summary", "Any")

years = [c["years_investment_experience"] for c in candidates]
known = [y for y in years if y is not None]
y_min, y_max = st.sidebar.slider(
    "Investment experience (years)",
    0.0,
    float(max(known) + 1) if known else 20.0,
    (0.0, float(max(known) + 1) if known else 20.0),
    step=0.5,
)
include_unknown = st.sidebar.checkbox(
    "Include candidates whose tenure could not be computed", value=True
)

only_alum = st.sidebar.checkbox("Multi-manager platform alumni only")
min_quality = st.sidebar.select_slider(
    "Minimum parse confidence", options=["low", "medium", "high"], value="low"
)
prefer_quality = st.sidebar.checkbox(
    "Rank well-parsed records first",
    help=(
        "Sorts high-confidence records above equally good matches. Useful "
        "when a shortlist is going to someone else: a strong match built on "
        "a half-read resume wastes their time."
    ),
)
query = st.sidebar.text_input("Keyword", placeholder="e.g. digital health, backtesting")

st.sidebar.divider()
st.sidebar.caption(
    f"{len(candidates)} candidates parsed · dataset built offline · "
    "no model runs in this app"
)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

BAND_ORDER = {"low": 0, "medium": 1, "high": 2}


def searchable_text(c: dict) -> str:
    parts = [c["display_name"], c.get("location") or ""]
    parts += c.get("sectors", []) + c.get("employers", [])
    parts += c.get("software_tools", []) + c.get("methods", [])
    for p in c["extraction"]["positions"]:
        parts += [p["firm"], p["title"], *p.get("description", [])]
    return " ".join(parts).lower()


def passes_filters(c: dict) -> bool:
    if f_region and c.get("region") not in f_region:
        return False
    if f_family and c.get("approach_family") not in f_family:
        return False
    if f_sector and not set(f_sector) & set(c.get("sectors", [])):
        return False
    if f_side and c.get("market_side") not in f_side:
        return False
    if only_alum and not c.get("platform_alum_of"):
        return False
    if f_software and not set(f_software) & set(c.get("software_tools", [])):
        return False
    if f_credentials and not set(f_credentials) & set(
        c.get("credentials_summary", [])
    ):
        return False
    if BAND_ORDER[c["quality"]["band"]] < BAND_ORDER[min_quality]:
        return False

    tenure = c.get("years_investment_experience")
    if tenure is None:
        if not include_unknown:
            return False
    elif not (y_min <= tenure <= y_max):
        return False

    if query and query.lower() not in searchable_text(c):
        return False
    return True


filtered = [c for c in candidates if passes_filters(c)]
by_id = {c["candidate_id"]: c for c in candidates}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_insights, tab_quality, tab_method = st.tabs(
    ["Candidates", "Insights", "Data quality", "Method"]
)


# ===========================================================================
# Candidates
# ===========================================================================

with tab_search:
    if requisition:
        exact, near = match_all(filtered, requisition, store=store)
        results = exact + near
        matches = {r.candidate_id: r for r in results}

        st.markdown(f"#### {requisition['title']}")

        left, right = st.columns([1, 3])
        left.metric("Exact matches", len(exact))
        if exact:
            right.caption(
                "Every hard constraint satisfied. Ranked by the weighted soft "
                "signals; open a candidate to see each component and the "
                "resume text behind it."
            )
        else:
            # The whole point of separating hard from soft: a matcher that
            # always produces a winner has never been tested against the
            # possibility that there isn't one.
            right.warning(
                "**No candidate satisfies every requirement of this "
                "requisition.** The near misses below each fail exactly one "
                "constraint, named on their card. Widening that one constraint "
                "is the decision to make -- not accepting a high percentage "
                "that hides the gap.",
                icon="⚠",
            )
        if near:
            st.caption(f"{len(near)} candidate(s) fail exactly one constraint.")
    else:
        results = None
        matches = {}
        st.markdown("#### Candidate pool")
        st.caption(
            f"{len(filtered)} of {len(candidates)} candidates match the current "
            "filters. Select a requisition in the sidebar to rank them against "
            "a specific seat."
        )

    ordered = (
        [by_id[r.candidate_id] for r in results] if results is not None else filtered
    )
    if prefer_quality:
        rank = {"high": 0, "medium": 1, "low": 2}
        # Stable sort, so within a confidence band the match ranking (or the
        # existing order) is preserved rather than reshuffled.
        ordered = sorted(ordered, key=lambda c: rank[c["quality"]["band"]])

    if not ordered:
        st.info("No candidates match. Try relaxing a filter in the sidebar.")
    else:
        list_col, detail_col = st.columns([1.3, 1.35], gap="medium")

        with list_col:
            rows = []
            for c in ordered:
                m = matches.get(c["candidate_id"])
                row = {
                    "Candidate": c["display_name"],
                    "Fit": (
                        ("✓ " if m.is_exact else "~ ") + f"{m.soft_score:.0%}"
                        if m
                        else ""
                    ),
                    "Yrs": c.get("years_investment_experience"),
                }
                if not matches:
                    row["Region"] = c.get("region") or "—"
                    row["Data"] = c["quality"]["band"]
                if matches:
                    row["Fails on"] = (
                        f"{m.failed_hard[0].label} — {m.failed_hard[0].found}"
                        if m and m.failed_hard
                        else "nothing"
                    )
                else:
                    row["Sector"] = ", ".join(c.get("sectors", [])[:2])
                rows.append(row)
            frame = pd.DataFrame(rows)
            event = st.dataframe(
                frame,
                hide_index=True,
                use_container_width=True,
                height=min(460, 60 + 35 * len(rows)),
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Yrs": st.column_config.NumberColumn(
                        "Yrs investing", format="%.1f", width="small"
                    ),
                    "Fit": st.column_config.TextColumn(width="small"),
                    "Data": st.column_config.TextColumn(width="small"),
                    "Fails on": st.column_config.TextColumn(width="medium"),
                },
            )
            picked = event.selection.rows[0] if event.selection.rows else 0

            st.download_button(
                "Download this shortlist (CSV)",
                frame.to_csv(index=False).encode(),
                file_name="shortlist.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with detail_col:
            c = ordered[picked]
            e = c["extraction"]
            m = matches.get(c["candidate_id"])

            st.markdown(f"### {c['display_name']}")
            if c["name_source"] == "filename":
                st.caption(
                    "⚠ This name comes from the filename — the document itself "
                    "never states one."
                )

            meta = " · ".join(
                part
                for part in (
                    c.get("location"),
                    c.get("current_firm"),
                    label_of(c["current_firm_type"]) if c.get("current_firm_type") else None,
                )
                if part
            )
            st.markdown(
                f"<div style='color:{MUTED};font-size:13px;margin:-6px 0 8px'>"
                f"{meta}</div>"
                + quality_chip(c["quality"]["band"], c["quality"]["score"]),
                unsafe_allow_html=True,
            )

            if c.get("platform_alum_of"):
                st.success(
                    f"**Platform alum — previously at "
                    f"{', '.join(c['platform_alum_of'])}.** Surfaced through "
                    "pod-to-platform lineage in the knowledge base; the resume "
                    "names the pod, never the platform."
                )

            k1, k2, k3 = st.columns(3)
            k1.metric("Career", f"{c['years_experience'] or '—'} yrs",
                      help="Total professional tenure, excluding internships, "
                           "student societies and volunteering.")
            inv = c["years_investment_experience"]
            k2.metric("Investing", "—" if inv is None else f"{inv} yrs",
                      help="Tenure in roles that actually researched or managed "
                           "investments. 0.0 means none; a dash means the "
                           "resume gave no dates to compute it from.")
            covered = e["coverage"].get("stocks_covered")
            k3.metric("Stocks covered", covered or "—",
                      help="The largest number of names the candidate states "
                           "having under research coverage — the standard "
                           "measure of breadth in equity research.")

            st.markdown("##### Summary")
            summary_fields = [
                ("Region", c.get("region") or "—"),
                ("Location", c.get("location") or "—"),
                ("Approach", f"{label_of(c.get('approach') or '')} "
                             f"({label_of(c.get('approach_family') or '')})"),
                ("Market side", label_of(c.get("market_side") or "")),
                ("Sectors", ", ".join(label_of(x) for x in c.get("sectors", [])) or "—"),
                ("Asset classes",
                 ", ".join(label_of(x) for x in c.get("asset_classes", [])) or "—"),
                ("Coverage markets",
                 f"{', '.join(c.get('coverage_markets', [])) or '—'}"
                 f"{'  (inferred)' if c.get('coverage_markets_source') == 'inferred' else ''}"),
                ("Seniority",
                 f"{label_of(c.get('seniority_band') or '')} by career · "
                 f"{label_of(c.get('investment_seniority_band') or '')} by investing"),
                ("Current firm",
                 f"{c.get('current_firm') or '—'}"
                 f"{' · ' + label_of(c['current_firm_type']) if c.get('current_firm_type') else ''}"),
                ("Employers", "; ".join(c.get("employers", [])) or "—"),
                ("Platform alumni", ", ".join(c.get("platform_alum_of", [])) or "—"),
                ("Credentials", "; ".join(c.get("credentials_summary", [])) or "—"),
                ("Parse confidence",
                 f"{c['quality']['band']} {c['quality']['score']}"
                 f" · {len(c['flags'])} issue(s)"),
            ]
            st.markdown(
                "<table class='recordtbl'>"
                + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                          for k, v in summary_fields)
                + "</table>",
                unsafe_allow_html=True,
            )

            if m:
                with st.expander(
                    "Match breakdown — "
                    + ("meets every requirement" if m.is_exact else "near miss"),
                    expanded=True,
                ):
                    for crit in m.hard_criteria:
                        icon = "✓" if crit.passed else "✗"
                        colour = STATUS_GOOD if crit.passed else STATUS_BAD
                        st.markdown(
                            f"<div style='font-size:13px'><span style='color:"
                            f"{colour};font-weight:700'>{icon}</span> "
                            f"<b>{crit.label}</b> — needs {crit.required}; "
                            f"has {crit.found}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")
                    for crit in m.soft_criteria:
                        if crit.weight <= 0:
                            continue
                        st.markdown(
                            f"<div style='font-size:13px'><b>{crit.label}</b> "
                            f"<span style='color:{MUTED}'>{crit.found}</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.progress(min(1.0, crit.score))
                        if crit.evidence:
                            st.caption(f"“{crit.evidence[:220]}”")

            # -- Profile: every attribute beside the text that produced it ----
            st.markdown("##### Profile")
            st.caption(
                "Each attribute with the keywords that drove it and the resume "
                "sentence it came from. Nothing here is asserted without a quote."
            )

            def attribute(label: str, value: str, block: dict | None = None,
                          note: str = "") -> None:
                conf = f" · {block['confidence']} confidence" if block else ""
                st.markdown(
                    f"<div style='margin-top:9px;font-size:13px'>"
                    f"<span style='color:{MUTED}'>{label}</span><br>"
                    f"<b style='font-size:14px'>{value}</b>"
                    f"<span style='color:{MUTED};font-size:12px'>{conf}"
                    f"{' · ' + note if note else ''}</span></div>",
                    unsafe_allow_html=True,
                )
                if block:
                    if block.get("keywords"):
                        st.markdown(
                            " ".join(
                                f"<span style='background:#f0ece4;color:#7a4d1d;"
                                f"font-size:11px;padding:1px 8px;border-radius:9px;"
                                f"margin-right:4px'>{k}</span>"
                                for k in block["keywords"]
                            ),
                            unsafe_allow_html=True,
                        )
                    if block.get("evidence"):
                        st.caption(f"“{block['evidence']}”")

            attribute("Investment approach",
                      label_of(e["investment_approach"]["value"]),
                      e["investment_approach"],
                      note=label_of(c.get("approach_family") or ""))
            attribute("Market side", label_of(e["market_side"]["value"]),
                      e["market_side"])
            for sector in e["primary_sectors"]:
                attribute("Sector", label_of(sector["value"]), sector)
            for asset in e["asset_classes"][:2]:
                attribute("Asset class", label_of(asset["value"]), asset)

            markets = ", ".join(c.get("coverage_markets", [])) or "—"
            attribute(
                "Markets covered", markets, None,
                note="inferred from location"
                if c.get("coverage_markets_source") == "inferred" else "stated",
            )
            attribute(
                "Seniority",
                f"{label_of(c.get('seniority_band') or '')} by career · "
                f"{label_of(c.get('investment_seniority_band') or '')} by investing",
            )

            # -- Skills and credentials, given their own section --------------
            st.markdown("##### Skills and credentials")
            creds = c.get("credentials_summary", [])
            if creds:
                for cred in creds:
                    st.markdown(f"<div style='font-size:13px'>· {cred}</div>",
                                unsafe_allow_html=True)
            else:
                st.caption("No professional credentials stated.")

            def chips(label: str, values: list[str], colour: str) -> None:
                st.markdown(
                    f"<div style='margin-top:9px;font-size:12px;color:{MUTED}'>"
                    f"{label}</div>"
                    + (
                        " ".join(
                            f"<span style='background:{colour}1a;color:{colour};"
                            f"font-size:11.5px;padding:2px 9px;border-radius:9px;"
                            f"margin:0 4px 4px 0;display:inline-block'>{v}</span>"
                            for v in values
                        )
                        if values
                        else f"<span style='color:{MUTED};font-size:12.5px'>"
                             "not stated</span>"
                    ),
                    unsafe_allow_html=True,
                )

            chips("Software and platforms", c.get("software_tools", []), SERIES_1)
            chips("Analytical methods", c.get("methods", []), SERIES_2)
            chips("Languages", c.get("languages", []), "#4a3aa7")

            # -- Experience -----------------------------------------------------
            # Every role, with the raw employer name beside what the knowledge
            # base resolved it to and how. An unresolved employer is shown in
            # red rather than hidden: without a firm type that role cannot
            # take part in any structured filter, and the reviewer should know
            # which part of the history is invisible to the search.
            st.markdown("##### Experience, as parsed")
            rows_html = [
                "<tr><th>Employer</th><th>Resolved to</th><th>Title</th>"
                "<th>Dates</th><th>Type</th><th>Inv.</th></tr>"
            ]
            firm_by_raw = {f["raw"]: f for f in c.get("firms", [])}
            for pos in e["positions"]:
                link = firm_by_raw.get(pos["firm"], {})
                resolution = link.get("resolution", "unresolved")
                canonical = link.get("canonical") or "—"
                bad = resolution in {"unresolved", "ambiguous"} and pos[
                    "employment_type"
                ] in {"professional", "internship"}
                dates = (
                    f"{pos.get('start_date') or '?'} – "
                    f"{'present' if pos.get('is_current') else (pos.get('end_date') or '?')}"
                )
                if not pos.get("start_date") and pos.get("duration_months"):
                    dates = f"{pos['duration_months']} mo (duration only)"
                rows_html.append(
                    "<tr>"
                    f"<td>{pos['firm']}</td>"
                    f"<td style='color:{STATUS_BAD if bad else INK}'>{canonical}"
                    f"<span class='res'>{resolution}</span></td>"
                    f"<td>{pos['title']}</td>"
                    f"<td>{dates}</td>"
                    f"<td style='color:{MUTED}'>"
                    f"{pos['employment_type'].replace('_', ' ')}</td>"
                    f"<td>{'✓' if pos.get('is_investment_role') else '·'}</td>"
                    "</tr>"
                )
            st.markdown(
                f"<table class='postbl'>{''.join(rows_html)}</table>",
                unsafe_allow_html=True,
            )
            if c.get("non_professional_affiliations"):
                st.caption(
                    "Excluded from tenure: "
                    + "; ".join(c["non_professional_affiliations"])
                )

            # -- The complete record, for anyone checking our work -------------
            with st.expander("Full parsed record"):
                fields = [
                    ("Name", c["display_name"]),
                    ("Name source", c["name_source"]),
                    ("Email", e.get("email") or "—"),
                    ("Phone", e.get("phone") or "—"),
                    ("Location", c.get("location") or "—"),
                    ("Region", c.get("region") or "—"),
                    ("Career tenure", f"{c['years_experience'] or '—'} yrs"),
                    (
                        "Investment tenure",
                        "—" if inv is None else f"{inv} yrs",
                    ),
                    ("Seniority (career)", label_of(c.get("seniority_band") or "")),
                    (
                        "Seniority (investing)",
                        label_of(c.get("investment_seniority_band") or ""),
                    ),
                    ("In junior range", str(c.get("is_junior_range"))),
                    ("Approach", f"{c.get('approach')} → {c.get('approach_family')}"),
                    ("Market side", c.get("market_side") or "—"),
                    ("Sectors", ", ".join(c.get("sectors", [])) or "—"),
                    ("Asset classes", ", ".join(c.get("asset_classes", [])) or "—"),
                    (
                        "Coverage markets",
                        f"{', '.join(c.get('coverage_markets', [])) or '—'} "
                        f"({c.get('coverage_markets_source')})",
                    ),
                    ("Stocks covered", str(covered or "—")),
                    ("Current firm", c.get("current_firm") or "—"),
                    ("Current firm type", c.get("current_firm_type") or "—"),
                    ("Employers", "; ".join(c.get("employers", [])) or "—"),
                    (
                        "Non-professional",
                        "; ".join(c.get("non_professional_affiliations", [])) or "—",
                    ),
                    ("Buy-side experience", str(c.get("has_buy_side_experience"))),
                    ("Sell-side experience", str(c.get("has_sell_side_experience"))),
                    (
                        "Platform alumni",
                        ", ".join(c.get("platform_alum_of", [])) or "—",
                    ),
                    ("Credentials", "; ".join(c.get("credentials_summary", [])) or "—"),
                    ("Languages", ", ".join(c.get("languages", [])) or "—"),
                    ("Software", ", ".join(c.get("software_tools", [])) or "—"),
                    ("Methods", ", ".join(c.get("methods", [])) or "—"),
                    ("Positions parsed", str(len(e["positions"]))),
                    (
                        "Parse confidence",
                        f"{c['quality']['band']} {c['quality']['score']}",
                    ),
                    (
                        "Missing fields",
                        ", ".join(c["quality"]["missing_fields"]) or "none",
                    ),
                    ("Source file", c["source_file"]),
                ]
                st.markdown(
                    "<table class='recordtbl'>"
                    + "".join(
                        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields
                    )
                    + "</table>",
                    unsafe_allow_html=True,
                )

            with st.expander("Education"):
                if not e.get("education"):
                    st.caption("No education section could be parsed.")
                for degree in e.get("education", []):
                    span = (
                        f"{degree.get('start_year') or '?'} – "
                        f"{degree.get('graduation_year') or '?'}"
                    )
                    st.markdown(
                        f"<div style='font-size:13px;margin-bottom:5px'>"
                        f"<b>{degree.get('degree') or 'Degree'}"
                        f"{', ' + degree['field_of_study'] if degree.get('field_of_study') else ''}"
                        f"</b> · {degree['institution']}"
                        f"<span style='color:{MUTED}'> — {span}</span></div>",
                        unsafe_allow_html=True,
                    )

            # -- Flags ---------------------------------------------------------
            if c.get("flags"):
                st.markdown(f"##### Issues found ({len(c['flags'])})")
                for f in c["flags"]:
                    tag = "computed" if f["source"] == "computed" else "read"
                    colour = SERIES_1 if f["source"] == "computed" else SERIES_2
                    st.markdown(
                        f"<div style='font-size:12.5px;margin-bottom:8px;"
                        f"border-left:3px solid {colour};padding-left:9px'>"
                        f"<b>{f['summary']}</b><br>"
                        f"<span style='color:{MUTED}'>{f['detail']}</span>"
                        + (f"<br><i style='color:{MUTED}'>“{f['quote']}”</i>"
                           if f.get("quote") else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# Insights
# ===========================================================================

with tab_insights:
    st.markdown("#### Where the bench is, and where it isn't")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.6])
    chart_quality = ctrl1.select_slider(
        "Minimum parse confidence for these charts",
        options=["low", "medium", "high"],
        value="low",
        help=(
            "Counts built from thin records overstate the bench. Raising this "
            "shows only the part of the pool the pipeline read well."
        ),
    )
    group_by = ctrl2.selectbox(
        "Break down by", ["Region", "Market side", "Seniority (investing)"]
    )
    charted = [
        c
        for c in filtered
        if BAND_ORDER[c["quality"]["band"]] >= BAND_ORDER[chart_quality]
    ]
    ctrl3.markdown(
        f"<div style='padding-top:26px;color:{MUTED};font-size:12.5px'>"
        f"<b style='color:{NAVY}'>{len(charted)}</b> of {len(candidates)} "
        "candidates charted after sidebar filters and the threshold at left."
        "</div>",
        unsafe_allow_html=True,
    )

    if not charted:
        st.info("No candidates left at this threshold.")
        st.stop()

    st.caption(
        "Empty cells are the point: they are the seats this pool cannot fill."
    )

    filtered = charted  # charts below read from the thresholded set
    sectors = sorted({s for c in filtered for s in c.get("sectors", [])})
    GROUP_KEY = {
        "Region": lambda c: c.get("region"),
        "Market side": lambda c: label_of(c.get("market_side") or ""),
        "Seniority (investing)": lambda c: label_of(
            c.get("investment_seniority_band") or ""
        ),
    }[group_by]
    regions = sorted({g for c in filtered if (g := GROUP_KEY(c))})

    if sectors and regions:
        matrix = [
            [
                sum(
                    1
                    for c in filtered
                    if GROUP_KEY(c) == region and sector in c.get("sectors", [])
                )
                for sector in sectors
            ]
            for region in regions
        ]
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=[label_of(s) for s in sectors],
                y=regions,
                colorscale=(
                    [[0.0, SURFACE], [0.001, SEQ[0]]]
                    + [
                        [0.001 + (1 - 0.001) * i / (len(SEQ) - 1), c]
                        for i, c in enumerate(SEQ)
                    ]
                ),
                zmin=0,
                showscale=False,
                xgap=2,
                ygap=2,
                text=[[v or "" for v in row] for row in matrix],
                texttemplate="%{text}",
                textfont=dict(size=13),
                hovertemplate="%{y} · %{x}<br>%{z} candidate(s)<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Sector coverage by {group_by.lower()}", showlegend=False
        )
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("**Career length vs. investing tenure**")
        st.caption(
            "The gap is the point. Screening on total experience alone "
            "overstates how long several of these candidates have actually "
            "been investing."
        )
        pool = sorted(
            [c for c in filtered if c["years_experience"] is not None],
            key=lambda c: c["years_experience"],
        )
        if pool:
            names = [c["display_name"] for c in pool]
            total = [c["years_experience"] for c in pool]
            invest = [
                c["years_investment_experience"] or 0
                for c in pool
            ]
            fig = go.Figure()
            for name, t, i in zip(names, total, invest):
                fig.add_trace(
                    go.Scatter(
                        x=[i, t],
                        y=[name, name],
                        mode="lines",
                        line=dict(color=GRID, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=total,
                    y=names,
                    mode="markers",
                    name="Career total",
                    marker=dict(color=SERIES_2, size=10,
                                line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>Career %{x} yrs<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=invest,
                    y=names,
                    mode="markers",
                    name="Investing",
                    marker=dict(color=SERIES_1, size=10,
                                line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>Investing %{x} yrs<extra></extra>",
                )
            )
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
                xaxis_title="Years",
            )
            fig.update_yaxes(showgrid=False)
            fig.update_xaxes(showgrid=True, gridcolor=GRID)
            st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with right:
        st.markdown("**Most common software and credentials**")
        st.caption(
            "What this pool can actually operate. A requisition naming a tool "
            "nobody holds is a sourcing problem, not a screening one."
        )
        tally: dict[str, int] = {}
        for c in filtered:
            for tool in c.get("software_tools", []):
                tally[tool] = tally.get(tool, 0) + 1
        creds: dict[str, int] = {}
        for c in filtered:
            for cred in c.get("credentials_summary", []):
                head = cred.split(" - ")[0].split(" (")[0]
                creds[head] = creds.get(head, 0) + 1

        top_tools = sorted(tally.items(), key=lambda kv: -kv[1])[:9]
        top_creds = sorted(creds.items(), key=lambda kv: -kv[1])[:5]
        if top_tools or top_creds:
            labels = [k for k, _ in top_creds] + [k for k, _ in top_tools]
            values = [v for _, v in top_creds] + [v for _, v in top_tools]
            colours = [SERIES_2] * len(top_creds) + [SERIES_1] * len(top_tools)
            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker_color=colours,
                    marker_line=dict(color=SURFACE, width=2),
                    hovertemplate="%{y}<br>%{x} candidate(s)<extra></extra>",
                )
            )
            fig.update_layout(showlegend=False, xaxis_title="Candidates")
            fig.update_yaxes(showgrid=False, autorange="reversed")
            fig.update_xaxes(showgrid=True, gridcolor=GRID, dtick=1)
            st.plotly_chart(styled_chart(fig, 360), use_container_width=True)
            st.caption(
                f"Bronze = credentials · slate = software. Colour is the only "
                "thing separating the two groups here, so they are also "
                "ordered: credentials first."
            )

    st.markdown("---")
    with st.expander("Approach and market side"):
        st.caption(
            "The axis the brief frames the pool on: fundamental against "
            "systematic / quantitative."
        )
        families = sorted(
            {c["approach_family"] for c in filtered if c.get("approach_family")}
        )
        sides = sorted({c["market_side"] for c in filtered if c.get("market_side")})
        if families and sides:
            fig = go.Figure()
            for index, side in enumerate(sides[:2]):
                fig.add_trace(
                    go.Bar(
                        name=label_of(side),
                        x=[label_of(f) for f in families],
                        y=[
                            sum(
                                1
                                for c in filtered
                                if c.get("approach_family") == f
                                and c.get("market_side") == side
                            )
                            for f in families
                        ],
                        marker_color=[SERIES_1, SERIES_2][index],
                        marker_line=dict(color=SURFACE, width=2),
                    )
                )
            fig.update_layout(
                barmode="group",
                bargap=0.45,
                showlegend=True,
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
                yaxis_title="Candidates",
            )
            st.plotly_chart(styled_chart(fig, 360), use_container_width=True)


# ===========================================================================
# Data quality
# ===========================================================================

with tab_quality:
    st.markdown("#### How far to trust each record")
    st.caption(
        "Parse confidence is a property of the DOCUMENT and of how well the "
        "pipeline could read it — never a judgement about the candidate. A "
        "resume built from tables or missing its dates scores lower because "
        "our data on that person is thinner, not because they are weaker."
    )

    ranked = sorted(candidates, key=lambda c: c["quality"]["score"])
    bands = {
        b: sum(1 for c in candidates if c["quality"]["band"] == b)
        for b in QUALITY_COLOUR
    }
    scores = [c["quality"]["score"] for c in candidates]

    cols = st.columns(4)
    cols[0].metric("Median confidence", f"{sorted(scores)[len(scores)//2]:.2f}")
    for col, (band, count) in zip(cols[1:], bands.items()):
        col.metric(f"{band.title()} confidence", count)

    st.markdown("---")
    st.markdown("**Parse confidence by candidate**")
    fig = go.Figure(
        go.Bar(
            x=[c["quality"]["score"] for c in ranked],
            y=[c["display_name"] for c in ranked],
            orientation="h",
            marker_color=[QUALITY_COLOUR[c["quality"]["band"]] for c in ranked],
            marker_line=dict(color=SURFACE, width=2),
            text=[f"{c['quality']['score']:.2f}" for c in ranked],
            textposition="outside",
            textfont=dict(size=11, color=INK),
            hovertemplate=(
                "%{y}<br>confidence %{x:.2f}<br>"
                "%{customdata} issue(s) found<extra></extra>"
            ),
            customdata=[len(c["flags"]) for c in ranked],
        )
    )
    fig.update_layout(showlegend=False, xaxis_title="Parse confidence (0–1)")
    fig.update_xaxes(range=[0, 1.12], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(styled_chart(fig, 380), use_container_width=True)
    st.caption(
        "Green ≥ 0.80, amber ≥ 0.55, red below. Scores are deductions from a "
        "clean parse: a missing name costs most, missing dates next, a "
        "formatting quirk nothing at all."
    )

    st.markdown("---")
    st.markdown("**What is wrong with each record**")
    st.caption(
        "Ordered lowest confidence first — the records a recruiter should "
        "verify before acting on them."
    )

    for c in ranked:
        band = c["quality"]["band"]
        with st.expander(
            f"{c['display_name']} — {band} {c['quality']['score']} · "
            f"{len(c['flags'])} issue(s)",
            expanded=(band != "high"),
        ):
            if c["quality"]["missing_fields"]:
                st.markdown(
                    f"<span style='color:{STATUS_BAD};font-size:12.5px'>"
                    f"<b>Missing:</b> "
                    f"{', '.join(c['quality']['missing_fields'])}</span>",
                    unsafe_allow_html=True,
                )
            if not c["flags"]:
                st.caption("Nothing flagged.")
            for f in c["flags"]:
                colour = SERIES_1 if f["source"] == "computed" else SERIES_2
                origin = (
                    "computed from the data"
                    if f["source"] == "computed"
                    else "read from the text"
                )
                st.markdown(
                    f"<div style='font-size:12.5px;margin-bottom:7px;"
                    f"border-left:3px solid {colour};padding-left:9px'>"
                    f"<b>{f['summary']}</b> "
                    f"<span style='color:{MUTED};font-size:11px'>· {origin}</span>"
                    f"<br><span style='color:{MUTED}'>{f['detail']}</span></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    with st.expander("How each document was read"):
        st.caption(
            "Diagnostics from the extraction step — table share, recovered "
            "text boxes, repaired ligatures, column splits. Most pipelines "
            "discard these; they are what the confidence above is built from."
        )
        log = ROOT / "data" / "extraction_log.csv"
        if log.exists():
            st.dataframe(
                pd.read_csv(log), hide_index=True, use_container_width=True
            )


# ===========================================================================
# Method
# ===========================================================================

with tab_method:
    st.markdown(
        """
#### How this works

**Pipeline.** `document → extraction → LLM structured parse → validation →
knowledge-base enrichment → dataset`. Parsing runs offline and is cached; this
app loads a pre-built JSON file and calls no model, so the deployment holds no
API key and every page is instant.

**Extraction is separate from parsing, and does the heavy lifting.** The
largest source of error in a resume pipeline is not the model — it is losing
content before the model ever sees it. Word tables, floating text boxes and
two-column PDFs are each read explicitly. One resume in this corpus keeps its
candidate name and its section headings inside text boxes, which standard
paragraph extraction does not read at all; another is 82% tables.

**The model is asked only for judgement.** Anything code can settle is settled
by code: tenure is date arithmetic, a firm's type is a lookup, a region is a
lookup. Narrowing the model's remit shrinks the surface where it can be wrong,
and makes a wrong answer a one-line fix in a YAML file rather than a prompt
experiment.

**A curated knowledge base supplies what no model reliably knows.** That North53
Capital is a Millennium pod; that Cinctive is a multi-manager platform; that
three unrelated firms in this corpus are all called "Meridian". Firm matching
refuses to guess between them and reports the ambiguity instead.

**Hard constraints disqualify; soft signals rank.** A candidate outside the
region or the experience band is not a weaker match — they are not a match.
This is why the search can return nothing, and why the near-miss list names the
single constraint each candidate failed.

**Every claim carries its evidence.** No classification appears without the
resume text that produced it, and each evidence quote is verified to appear
verbatim in the source.
        """
    )
    st.caption(
        "Built for the Millennium Business Development data science case "
        "study. Source: github.com/yc4379-commits/m-case-study"
    )
