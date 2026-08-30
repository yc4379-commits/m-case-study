"""
Talent Intelligence Platform -- Streamlit application.

A sourcing workbench for a Business Development team: define the seat you are
filling, see who qualifies, and see why.

The interface is organised around one decision that runs through the whole
system: **a role comes first, and hard requirements disqualify.**

That is why the app opens on a role selector rather than a wall of filters. A
requisition is not one more dropdown -- it decides who is eligible at all,
while the sidebar only narrows an already-eligible pool. Collapsing the two
into a single filter panel was the first version's mistake, and it left users
unable to tell which controls excluded a candidate and which merely reordered
them.

Three further rules shape what is on screen:

**Nothing is asserted without its evidence.** Every classification carries the
resume sentence that produced it. This is not decoration -- a scoring bug that
inflated every requirement score was invisible in the aggregate numbers and
obvious the moment one quote was read.

**Every number is explained where it appears.** Field definitions live in
`src/glossary.py` and are attached to the control or column they describe. A
number a user cannot interpret is worse than no number.

**Data quality is visible.** Each record carries a parse confidence and the
specific issues behind it, and issues appear near the top of a profile rather
than buried at the bottom.

The app reads a pre-built JSON dataset and calls no model at runtime: parsing
happens offline in `src/build_dataset.py`. The deployment therefore holds no
API key, loads instantly, and cannot be made to spend money by a visitor.

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

from glossary import (  # noqa: E402
    COLUMN_HELP,
    CRITERION_HELP,
    CRITERION_LABEL,
    FILTER_HELP,
    METRIC_HELP,
)
from match import Requisitions, match_all  # noqa: E402

DATA = ROOT / "data" / "candidates.json"

# ---------------------------------------------------------------------------
# Palette
#
# Brand navy is chrome only -- headers, rules, accents. As a data series it
# fails the palette validator on two hard checks (lightness 0.31 against a
# 0.43-0.77 band; chroma 0.09 against a 0.10 floor), which in practice means
# it reads as grey in a chart and collides with any dark neighbour.
#
# Series colours are slate blue and bronze, validated together against this
# white surface: lightness band, chroma floor, colour-vision separation
# (worst pair Delta E 20.9 protan / 24.5 tritan), normal-vision floor (25.9)
# and 3:1 contrast all pass. Comparison is capped at two candidates partly for
# legibility and partly because this pair is what has been validated.
# ---------------------------------------------------------------------------
NAVY = "#0b2f5e"
NAVY_WASH = "#f4f7fb"
SERIES_1 = "#2f5f98"
SERIES_2 = "#b5793a"
SEQ = ["#eef3f9", "#d5e1ef", "#b4c9e1", "#8badd0", "#5d8bba", "#2f5f98", "#123a6f"]
STATUS_GOOD = "#0ca30c"
STATUS_WARN = "#fab219"
STATUS_BAD = "#d03b3b"
QUALITY_COLOUR = {"high": STATUS_GOOD, "medium": STATUS_WARN, "low": STATUS_BAD}
INK = "#0b1b2b"
MUTED = "#64748b"
GRID = "#e2e8f0"
SURFACE = "#ffffff"
PAGE = "#fbfcfd"

BAND_ORDER = {"low": 0, "medium": 1, "high": 2}
NO_LIMIT = 40.0  # upper handle at maximum means "no upper limit"

st.set_page_config(
    page_title="Talent Intelligence Platform",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
      /* Register: minimal, Ashby-like. White surfaces, hairline borders,
         one accent colour used sparingly. Millennium's corporate grotesque
         is licensed; Inter is its open equivalent. The navy is chrome for
         small elements only -- a filled navy masthead was tried and read as
         a heavy corporate block, exactly the register modern recruiting
         tools have left behind. Colour that is everywhere emphasises
         nothing. */
      html, body, .stApp, .stApp * {{
        font-family: 'Inter', -apple-system, 'Segoe UI', Helvetica, Arial,
                     sans-serif;
      }}
      /* ...except Streamlit's icon glyphs, which are a font of their own --
         overriding them renders expander arrows as their literal names. */
      span[data-testid="stIconMaterial"], span[data-testid="stExpanderIcon"],
      span[translate="no"], .material-symbols-rounded {{
        font-family: 'Material Symbols Rounded' !important;
      }}
      /* Ashby's actual zoning trick, visible once you look for it: the
         page itself is grey, and every functional zone is a white card on
         it. An all-white page has no zones -- borders alone are too quiet
         to carve one screen into role / refine / results / profile. */
      .stApp {{ background: {PAGE}; }}
      header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
      [data-testid="stToolbar"] {{ visibility: hidden; }}
      .block-container {{ max-width: 1460px; padding-top: 70px !important; }}
      /* Masthead: a slim white bar, fixed over the whole viewport, held to
         the page by a hairline -- the product name and one mark of colour. */
      /* Masthead in Ashby's register: slim, dark, structured -- a mark,
         a wordmark, and live context on the right, not an empty colour
         slab. */
      /* Masthead after Millennium's own: midnight navy, a lowercase
         white serif wordmark, nothing else shouting. The serif is the
         brand voice; everything below it stays in the working sans. */
      .brandbar {{
        position: fixed; top: 0; left: 0; right: 0; height: 50px;
        z-index: 1000002; display: flex; align-items: center; gap: 14px;
        background: #0a1b38; padding: 0 24px;
      }}
      .brandbar b {{ font-family: 'Playfair Display', Georgia, serif;
                     font-size: 19px; font-weight: 500; color: #fff;
                     letter-spacing: .005em; white-space: nowrap; }}
      .brandbar span {{ font-size: 12px; color: rgba(255,255,255,.55);
                        padding-top: 3px; }}
      .brandbar .ctx {{ margin-left: auto; font-size: 12px;
                        color: rgba(255,255,255,.75); padding-top: 0; }}
      section[data-testid="stSidebar"] {{
        padding-top: 50px; background: #fff;
        border-right: 1px solid {GRID};
      }}
      /* Type scale: ink headings (navy is an accent, not a text colour),
         names 18, caps kickers 11, body 13, metrics 24. */
      h1,h2,h3,h4 {{ color: {INK} !important; letter-spacing: -.01em; }}
      .stApp h3 {{ font-size: 18px; font-weight: 600; }}
      .stApp h5 {{ font-size: 11px !important; font-weight: 600;
                   text-transform: uppercase; letter-spacing: .08em;
                   color: {MUTED} !important; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 20px;
                                           border-bottom: 1px solid {GRID}; }}
      .stTabs [data-baseweb="tab"] {{ font-weight: 500; font-size: 13.5px;
                                      color: {MUTED}; padding: 0 2px; }}
      .stTabs [aria-selected="true"] {{ color: {INK} !important;
                                        font-weight: 600; }}
      [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 650;
                                       font-size: 24px !important; }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      hr {{ border-color: {GRID}; }}
      .grouphead {{ font-size: 11px; font-weight: 600;
                    text-transform: uppercase; letter-spacing: .08em;
                    margin: 12px 0 8px; }}
      .pill {{ display: inline-block; background: #e9edf3;
               border-radius: 999px; padding: 3px 12px; font-size: 11.5px;
               color: {INK}; margin: 0 6px 6px 0; }}
      .statusline {{ font-size: 14px; margin: 4px 0 8px; }}
      .statusline b {{ font-size: 17px; }}
      .recordtbl {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
      .recordtbl td {{ padding: 5px 8px; border-bottom: 1px solid {GRID};
                       vertical-align: top; }}
      .recordtbl td:first-child {{ color: {MUTED}; width: 42%; }}
      .postbl {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
      .postbl th {{ text-align: left; color: {MUTED}; font-weight: 600;
                    padding: 5px 7px; border-bottom: 1px solid {GRID}; }}
      .postbl td {{ padding: 5px 7px; border-bottom: 1px solid {GRID};
                    vertical-align: top; }}
      .res {{ display: block; font-size: 10px; color: {MUTED};
              text-transform: uppercase; letter-spacing: .05em; }}
      /* Candidate list: a radio group styled as selectable rows, so the
         candidate's NAME is the click target rather than a checkbox.
         Selection is a border and a wash, not a shadow or an inset bar --
         flat surfaces, quiet states. */
      div[role="radiogroup"] label {{
        border: 1px solid {GRID}; border-radius: 12px;
        padding: 12px 16px !important; margin-bottom: 8px; width: 100%;
        background: {SURFACE}; align-items: flex-start;
        transition: border-color .12s, background .12s;
      }}
      div[role="radiogroup"] label:hover {{
        background: #f7f9fc; border-color: #d9e1ea;
      }}
      div[role="radiogroup"] label:has(input:checked) {{
        border-color: {NAVY}; background: {NAVY_WASH};
        box-shadow: 0 0 0 1px {NAVY} inset;
      }}
      div[role="radiogroup"] label > div:first-child {{ margin-top: 3px; }}
      div[role="radiogroup"] label p {{ font-size: 13px !important;
                                        line-height: 1.5; }}
      div[role="radiogroup"] label p + p {{ margin-top: 2px !important; }}
      [data-testid="stExpander"] details {{
        border: 1px solid {GRID}; border-radius: 12px; background: {SURFACE};
      }}
      [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE}; border: 1px solid {GRID};
        border-radius: 14px; padding: 10px 14px;
      }}
      /* Numbered zone headers: the one-glance answer to "where do I
         start" on a screen that stacks role, refinements and results. */
      .stephead {{ display: flex; align-items: center; gap: 8px;
                   margin: 2px 0 8px; }}
      .stepnum {{ width: 20px; height: 20px; border-radius: 999px;
                  background: {NAVY}; color: #fff; font-size: 11px;
                  font-weight: 700; display: flex; align-items: center;
                  justify-content: center; flex: none; }}
      .steptitle {{ font-size: 13px; font-weight: 600; color: {INK}; }}
      .stepsub {{ font-size: 12px; color: {MUTED}; }}
      [data-testid="stExpander"] summary {{ font-size: 13.5px; }}
      .stTextInput input, [data-baseweb="select"] {{ border-radius: 10px; }}
      .rolecard {{
        border: 1px solid {GRID}; border-left: 3px solid {NAVY};
        border-radius: 12px; background: {SURFACE};
        padding: 9px 14px; font-size: 12.5px; margin: 2px 0 14px;
        line-height: 1.7;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)




# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_candidates() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else []


@st.cache_resource(show_spinner=False)
def load_requisitions() -> Requisitions:
    return Requisitions.load()


candidates = load_candidates()
if not candidates:
    st.error("No dataset found. Build it first:  `python src/build_dataset.py`")
    st.stop()

store = load_requisitions()

st.markdown(
    '<div class="brandbar"><b>talent intelligence</b>'
    "<span>Business Development</span>"
    f"<span class='ctx'>{len(candidates)} candidates · "
    f"{len(store.items)} roles</span></div>",
    unsafe_allow_html=True,
)
by_id = {c["candidate_id"]: c for c in candidates}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_of(value: str | None) -> str:
    return value.replace("_", " ").title() if value else "—"


def quality_chip(band: str, score: float, issues: int) -> str:
    """The parse-confidence badge, spelled out and linked to its evidence.

    An earlier version read "DATA MEDIUM 0.62" and explained nothing: a
    reviewer could not tell whether it judged the document or the candidate,
    and there was no way to find out what drove it. It now names itself,
    carries the definition on hover, and links straight to the issue list
    that produced the number -- a score with no route to its reasons is just
    an assertion.
    """
    colour = QUALITY_COLOUR.get(band, MUTED)
    tail = f" · {issues} issue{'s' if issues != 1 else ''}" if issues else ""
    tooltip = METRIC_HELP["quality"]
    return (
        f"<a href='#issues' title='{tooltip}' "
        f"style='text-decoration:none'>"
        f"<span style='background:{colour}1f;color:{colour};padding:2px 10px;"
        f"border-radius:9px;font-size:11.5px;font-weight:700;'>"
        f"Data quality: {band} {score}{tail} ↓</span></a>"
    )


def styled_chart(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            size=12,
            color=INK,
        ),
        # Styling a title without giving it text makes Plotly print the literal
        # string "undefined" above the plot.
        title=dict(
            text=fig.layout.title.text or "",
            font=dict(size=13, color=INK),
            x=0,
            xanchor="left",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def searchable_text(c: dict) -> str:
    parts = [c["display_name"], c.get("location") or ""]
    parts += c.get("sectors", []) + c.get("employers", [])
    parts += c.get("software_tools", []) + c.get("methods", [])
    for p in c["extraction"]["positions"]:
        parts += [p["firm"], p["title"], *p.get("description", [])]
    return " ".join(parts).lower()


# ===========================================================================
# Step 1 -- the role
#
# A requisition decides who is ELIGIBLE; the sidebar only narrows an already
# eligible pool. Putting the two in one panel, as the first version did, left
# users unable to tell which controls excluded a candidate and which merely
# reordered them -- so the role now has its own step, above everything else.
# ===========================================================================

MODE_NOTES = {
    "Use a saved requisition":
        "Loads a stored role with its hard requirements and its written "
        "responsibilities. The richest ranking, because the requirement text "
        "gives the scorer something to match against.",
    "Define a role":
        "Builds the same object from the four dimensions the case brief "
        "names. Add requirement lines to make the ranking richer — without "
        "them the score rests on firm type and coverage alone.",
    "Browse without a role":
        "No matching at all: the pool is listed and filtered, but nobody is "
        "ranked, because ranking without a role to rank against is meaningless.",
}

# The role picker lives in a collapsed panel whose label states the current
# role. Keeping the picker on the same screen as the refinements is
# deliberate -- sourcing is iterative, and sending the user to another screen
# to widen an experience band turns a one-second adjustment into a round
# trip. But the first version left the picker permanently expanded, so six
# hundred pixels of chooser sat above the first candidate. Same screen,
# folded away until wanted.
_role_panel = st.expander(
    "**Step 1 · Which seat are you filling?**", expanded=False,
    icon=":material/work:",
)

with _role_panel:
    # The explanation lives inside the option itself. An earlier version put
    # the three modes in a radio and repeated them as three explanation cards
    # underneath -- the same three names twice, three hundred pixels of panel
    # to say what the option already said. Reading a description while
    # choosing beats reading it after.
    mode = st.radio(
        "Role source",
        list(MODE_NOTES),
        format_func=lambda name: f"**{name}**\n\n:gray[{MODE_NOTES[name]}]",
        label_visibility="collapsed",
    )

requisition: dict | None = None

with _role_panel:
    if mode == "Use a saved requisition":
        titles = store.titles
        chosen = st.selectbox(
            "Requisition",
            options=list(titles),
            format_func=lambda k: (
                f"{titles[k]}"
                + ("   ·  from a real posting" if store.get(k).get("source") else "")
            ),
            label_visibility="collapsed",
        )
        requisition = store.get(chosen)

    elif mode == "Define a role":
        st.caption(
            "These four dimensions are the ones the case brief names, and each is "
            "a HARD requirement: a candidate outside them is excluded, not "
            "down-ranked."
        )
        d1, d2, d3, d4 = st.columns([1, 1, 1.2, 1.3])
        regions = d1.multiselect(
            "Market", sorted({c["region"] for c in candidates if c.get("region")}),
            default=["US"], help=FILTER_HELP["region"],
        )
        families = d2.multiselect(
            "Approach",
            sorted({c["approach_family"] for c in candidates if c.get("approach_family")}),
            default=["fundamental"], format_func=label_of,
            help=FILTER_HELP["approach_family"],
        )
        sectors_any = d3.multiselect(
            "Sector (any of)", sorted({s for c in candidates for s in c.get("sectors", [])}),
            format_func=label_of,
            help="Leave empty to accept any sector.",
        )
        span = d4.slider(
            "Years of investment experience", 0.0, 25.0, (2.0, 8.0), step=0.5,
            help="The band a candidate must fall inside to be eligible.",
        )
        free_text = st.text_area(
            "Key requirements — one per line (optional, but they carry 25% of the "
            "Fit score)",
            placeholder=(
                "e.g. build and maintain company financial models\n"
                "coverage of technology companies\n"
                "meetings with company management"
            ),
            height=90,
            help=(
                "Written requirements are what the strongest scoring component "
                "matches against. Leave them out and that component drops out of "
                "the score entirely — the ranking still works, but on a much "
                "thinner basis."
            ),
        )
        requirement_lines = [ln.strip() for ln in free_text.splitlines() if ln.strip()]

        requisition = {
            "id": "adhoc",
            "title": "Custom role",
            "summary": "Defined in the app rather than loaded from a posting.",
            "hard": {
                "regions": regions,
                "approach_families": families,
                "investment_years": {"min": span[0], "max": span[1]},
                **({"sectors_any": sectors_any} if sectors_any else {}),
            },
            "soft": {
                "sectors": {s: 1.0 for s in sectors_any} or
                           {s: 0.6 for c in candidates for s in c.get("sectors", [])},
                "requirements": requirement_lines,
                "firm_types": {
                    "pod_shop": 1.0, "multi_strategy_platform": 1.0, "hedge_fund": 0.9,
                    "crossover": 0.85, "long_only": 0.75, "sell_side": 0.6,
                    "investment_bank": 0.6,
                },
                "prefer_platform_alum": True,
                "prefer_buy_side": True,
            },
        }

if requisition:
    hard = requisition.get("hard", {})
    bits = []
    if r := hard.get("regions"):
        bits.append(f"<b>Market</b> {' / '.join(r)}")
    if a := hard.get("approach_families"):
        bits.append(f"<b>Approach</b> {' / '.join(label_of(x) for x in a)}")
    if s := hard.get("sectors_any"):
        bits.append(f"<b>Sector</b> {' / '.join(label_of(x) for x in s)}")
    if y := hard.get("investment_years"):
        bits.append(
            f"<b>Experience</b> {y.get('min', 0)}–{y.get('max', 99)} yrs investing"
        )
    st.markdown(
        f"<div class='rolecard'><b>{requisition['title']}</b>"
        + (f" <span style='color:{MUTED}'>· {requisition.get('team','')}</span>"
           if requisition.get("team") else "")
        + f"<span style='color:{MUTED}'> &nbsp;—&nbsp; must satisfy </span>"
        + " <span style='color:#cbd5e1'>·</span> ".join(bits)
        + "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Step 2 -- sidebar refinements
# ===========================================================================

st.sidebar.markdown("### Refine the pool")
st.sidebar.caption(
    "Narrows an already-eligible list. Never overrides a hard requirement "
    "from the role."
)


def facet_counts(key: str, base: list[dict]) -> dict[str, int]:
    """How many of `base` carry each value.

    Counts are computed against the pool as narrowed by the OTHER active
    filters, not against the whole dataset. The first version counted
    globally, so after selecting APAC the Approach options still read
    "Fundamental (8)" -- which looks exactly like a filter that has stopped
    working. The number has to move, or it is lying.
    """
    counts: dict[str, int] = {}
    for c in base:
        value = c.get(key)
        for v in (value if isinstance(value, list) else [value]):
            if v:
                counts[v] = counts.get(v, 0) + 1
    return counts


# Filters are applied in two passes: first collect the selections, then apply
# them. Facet counts for each control are computed against everything except
# that control, which is what makes the numbers mean "if I pick this, I get
# this many".
selections: dict[str, list[str]] = {}
FACETS = [
    ("region", "Market", False),
    ("approach_family", "Approach", False),
    ("sectors", "Sector", True),
    ("market_side", "Market side", False),
    ("software_tools", "Software and tools", True),
    ("credentials_summary", "Credentials", True),
]


def matches_facets(c: dict, skip: str | None = None) -> bool:
    for key, _, is_list in FACETS:
        if key == skip:
            continue
        picked = selections.get(key)
        if not picked:
            continue
        value = c.get(key)
        have = set(value) if isinstance(value, list) else {value}
        if not have & set(picked):
            return False
    return True


# Three facets stay visible; the rest fold away. Ten controls stacked open
# made the sidebar taller than the results beside it, which reads as a wall
# of settings rather than a tool -- and the three below are the ones a
# recruiter reaches for first.
PRIMARY = {"region", "approach_family", "sectors"}


# Which sidebar facet answers to which hard requirement of the active role.
# When the role already fixes a dimension, the sidebar control over the same
# dimension is not doing what a user would assume: eligibility is already
# decided, so all the control can do is trim the one-gap list. The first
# reviewer hit exactly this -- a role requiring APAC, a sidebar offering
# APAC -- and reasonably asked which one was in charge. The control now
# says so itself.
_FACET_HARD_KEYS = {
    "region": "regions",
    "approach_family": "approach_families",
    "sectors": "sectors_any",
}


def facet_control(key: str, label: str, container) -> list[str]:
    base = [c for c in candidates if matches_facets(c, skip=key)]
    counts = facet_counts(key, base)
    options = sorted(counts) or sorted(facet_counts(key, candidates))
    # A dimension the role has fixed is locked here. One owner per
    # dimension: eligibility belongs to the role, and a second live control
    # over the same axis leaves the user asking which one is in charge.
    # The lock deliberately does NOT apply the value as a filter -- doing so
    # would silently delete the near misses who fail on exactly this
    # dimension, which is the one list a locked filter must not touch.
    hard_key = _FACET_HARD_KEYS.get(key)
    fixed = (requisition.get("hard") or {}).get(hard_key) \
        if requisition and hard_key else None
    if fixed:
        fixed_labels = ", ".join(label_of(v) for v in fixed)
        container.multiselect(
            label,
            options=options,
            default=[],
            disabled=True,
            help=FILTER_HELP.get(key, ""),
            placeholder=f"Set by the role — {fixed_labels}",
            key=f"locked_{key}",
        )
        container.caption(
            f":material/lock: Fixed by the role. Candidates who miss it "
            "still appear under *One gap away*."
        )
        return []
    return container.multiselect(
        label,
        options=options,
        format_func=lambda v, _c=counts: f"{label_of(v)}  ({_c.get(v, 0)})",
        help=FILTER_HELP.get(key, ""),
        placeholder="Any",
    )


for key, label, _ in FACETS:
    if key in PRIMARY:
        selections[key] = facet_control(key, label, st.sidebar)

query = st.sidebar.text_input(
    "Keyword", placeholder="e.g. digital health, backtesting",
    help=FILTER_HELP["keyword"],
)

with st.sidebar.expander("Experience, skills and credentials",
                         icon=":material/tune:"):
    years_span = st.slider(
        "Years of investment experience",
        0.0, NO_LIMIT, (0.0, NO_LIMIT), step=0.5, help=FILTER_HELP["years"],
    )
    if years_span[1] >= NO_LIMIT:
        st.caption("Upper handle at maximum — no upper limit applied.")
    include_unknown = st.checkbox(
        "Include candidates whose tenure could not be computed",
        value=True, help=FILTER_HELP["unknown"],
    )
    for key, label, _ in FACETS:
        if key not in PRIMARY and key != "market_side":
            selections[key] = facet_control(key, label, st)

with st.sidebar.expander("Background and data quality",
                         icon=":material/database:"):
    selections["market_side"] = facet_control("market_side", "Market side", st)
    only_alum = st.checkbox(
        "Multi-manager platform alumni only", help=FILTER_HELP["alum"]
    )
    min_quality = st.select_slider(
        "Minimum parse confidence", options=["low", "medium", "high"],
        value="low", help=FILTER_HELP["quality"],
    )
    prefer_quality = st.checkbox(
        "Rank well-parsed records first", help=FILTER_HELP["prefer_quality"]
    )


def passes_filters(c: dict) -> bool:
    if not matches_facets(c):
        return False
    if only_alum and not c.get("platform_alum_of"):
        return False
    if BAND_ORDER[c["quality"]["band"]] < BAND_ORDER[min_quality]:
        return False
    tenure = c.get("years_investment_experience")
    if tenure is None:
        if not include_unknown:
            return False
    else:
        upper = float("inf") if years_span[1] >= NO_LIMIT else years_span[1]
        if not (years_span[0] <= tenure <= upper):
            return False
    if query and query.lower() not in searchable_text(c):
        return False
    return True


filtered = [c for c in candidates if passes_filters(c)]
st.sidebar.caption(
    f"**{len(filtered)} of {len(candidates)}** candidates pass these "
    "refinements. Eligibility is decided by the role, not here."
)


# ===========================================================================
# Tabs
# ===========================================================================

tab_search, tab_insights, tab_quality, tab_method = st.tabs([
    ":material/group: Candidates",
    ":material/monitoring: Insights",
    ":material/fact_check: Data quality",
    ":material/menu_book: Method",
])


# ---------------------------------------------------------------------------
# Radar -- one or two candidates against the requisition's soft signals
# ---------------------------------------------------------------------------

def fit_radar(results: list, height: int = 380) -> go.Figure | None:
    """Score components for up to two candidates on shared axes.

    A radar makes the SHAPE of a fit readable at a glance -- strong on sector,
    thin on skills -- which is the comparison a recruiter actually makes. It is
    a poor instrument for reading exact values, since area grows with the
    square of the radius and exaggerates differences, so the numbers are also
    printed beneath it. Two series maximum: the colour pair is validated for
    two, and three filled polygons stop being readable.
    """
    weighted = [c for c in results[0].soft_criteria if c.weight > 0]
    if not weighted:
        return None
    keys = [c.key for c in weighted]
    labels = [CRITERION_LABEL.get(k, label_of(k)) for k in keys]

    fig = go.Figure()
    for index, result in enumerate(results[:2]):
        lookup = {c.key: c.score for c in result.soft_criteria}
        values = [lookup.get(k, 0.0) for k in keys]
        colour = [SERIES_1, SERIES_2][index]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=labels + labels[:1],
                fill="toself",
                fillcolor=colour + "26",
                line=dict(color=colour, width=2),
                marker=dict(size=7, color=colour),
                name=result.display_name,
                hovertemplate="%{theta}: %{r:.0%}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1], tickformat=".0%",
                gridcolor=GRID, tickfont=dict(size=10, color=MUTED),
                angle=90, tickangle=90,
            ),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=11, color=INK)),
        ),
        showlegend=len(results) > 1,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        height=height,
        margin=dict(l=60, r=60, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, sans-serif', color=INK),
    )
    return fig


_HONORIFICS = {"dr.", "dr", "mr.", "mr", "ms.", "ms", "mrs.", "prof.", "prof"}


def short_name(full: str) -> str:
    """A column-header-sized name.

    Taking the first token gave "Dr." for one candidate -- an honorific is not
    a name. Honorifics are dropped first, then the surname is preferred since
    that is how a shortlist is discussed.
    """
    parts = [
        p for p in full.split()
        if p.lower().strip(",") not in _HONORIFICS and not p.startswith("(")
    ]
    return parts[-1] if parts else full


def score_basis(result) -> tuple[float, int, int]:
    """What share of the intended weighting actually contributed.

    A component only scores when the role supplies something to score against:
    with no sectors and no written requirements, half the weighting silently
    drops out and the remaining half is renormalised to 100%. A candidate can
    then show "Fit 100%" while four of eight signals were never measured.

    Returning the share lets the interface say so, instead of presenting a
    thin score as a complete one.
    """
    active = [c for c in result.soft_criteria if c.weight > 0]
    configured = sum(store.weights.values())
    return (
        sum(c.weight for c in active) / configured if configured else 0.0,
        len(active),
        len(store.weights),
    )


def component_table(results: list) -> str:
    weighted = [c for c in results[0].soft_criteria if c.weight > 0]
    # With one candidate the column is simply their score, and a bare surname
    # as a header reads as a stray word -- "Al-Rashid" above a column of
    # percentages tells the reader nothing. Surnames only earn their place
    # when there are two columns to tell apart.
    headers = (
        ["Score"] if len(results) == 1
        else [short_name(r.display_name) for r in results[:2]]
    )
    head = ("<tr><th>Component</th><th>Weight</th>"
            + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
    rows = []
    for crit in weighted:
        cells = "".join(
            f"<td><b>{next((x.score for x in r.soft_criteria if x.key == crit.key), 0):.0%}</b></td>"
            for r in results[:2]
        )
        rows.append(
            f"<tr><td title='{CRITERION_HELP.get(crit.key, '')}'>"
            f"{CRITERION_LABEL.get(crit.key, crit.key)}</td>"
            f"<td style='color:{MUTED}'>{crit.weight:.0%}</td>{cells}</tr>"
        )
    total = "".join(f"<td><b>{r.soft_score:.0%}</b></td>" for r in results[:2])
    rows.append(
        f"<tr><td><b>Fit score</b></td><td style='color:{MUTED}'>100%</td>{total}</tr>"
    )
    return f"<table class='postbl'>{head}{''.join(rows)}</table>"


# ===========================================================================
# Candidates
# ===========================================================================

with tab_search:
    if requisition:
        exact, near = match_all(filtered, requisition, store=store)
        results = exact + near
        matches = {r.candidate_id: r for r in results}
        ordered = [by_id[r.candidate_id] for r in results]

        if results:
            share, active_n, total_n = score_basis(results[0])
            if share < 0.6:
                st.info(
                    f"**This role specifies little to score against**, so Fit "
                    f"is built from {active_n} of {total_n} signals — "
                    f"{share:.0%} of the intended weighting, renormalised to "
                    "100%. The ranking is valid but thin: a candidate can top "
                    "it without ever being measured on sector or requirement "
                    "fit. Adding sectors or requirement lines to the role "
                    "deepens it.",
                    icon="ℹ",
                )

        # One line when there are matches; a compact card when there are
        # none. The first version spent a headline metric and a full-width
        # warning box on this -- half a screen to say "zero". An empty
        # result is normal output, not an alarm, and the useful reaction to
        # it is specific: name the requirement most candidates miss and what
        # widening it would admit.
        if exact:
            st.markdown(
                f"<div class='statusline'><b style='color:{STATUS_GOOD}'>"
                f"{len(exact)} qualify</b> · ranked by Fit; "
                f"{len(near)} more are one gap away, listed after."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            from collections import Counter
            gap_counts = Counter(r.failed_hard[0].label for r in near)
            hint = ""
            if gap_counts:
                top_label, top_n = gap_counts.most_common(1)[0]
                hint = (f" Widening <b>{top_label.lower()}</b> would admit "
                        f"{top_n} of them.")
            st.markdown(
                f"<div style='border-left:3px solid {SERIES_2};"
                f"background:{SURFACE};border:1px solid {GRID};"
                f"border-left:3px solid {SERIES_2};border-radius:10px;"
                f"padding:12px 16px;margin:2px 0 6px;font-size:13px'>"
                f"<b>0 of {len(filtered)} qualify for this role.</b> "
                f"Each candidate below misses exactly one requirement, "
                f"named on their row — the decision is whether to widen "
                f"that requirement, not to accept a high Fit score that "
                f"hides it.{hint}</div>",
                unsafe_allow_html=True,
            )
    else:
        results, matches = None, {}
        ordered = filtered
        st.markdown("##### Candidate pool")
        st.caption(
            f"{len(filtered)} of {len(candidates)} candidates after the "
            "sidebar refinements. Choose a role above to rank them."
        )

    if prefer_quality:
        rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(ordered, key=lambda c: rank[c["quality"]["band"]])

    if not ordered:
        st.info("No candidates match. Relax a refinement in the sidebar.")
        st.stop()

    st.markdown("---")
    list_col, detail_col = st.columns([1.15, 1.6], gap="large")

    # -- Result list: the NAME is the click target -------------------------
    with list_col:
        st.markdown("##### Results")

        # Active refinements, restated where the results are. Two of the
        # sidebar's filter groups collapse, so a selection made inside one
        # can be invisible while it silently shrinks the list -- the classic
        # "why are there only three people" support ticket. Anything that is
        # currently narrowing the pool gets a pill here.
        pills: list[str] = []
        for _k, _lbl, _ in FACETS:
            if selections.get(_k):
                pills.append(
                    f"{_lbl}: {', '.join(label_of(v) for v in selections[_k])}"
                )
        if query:
            pills.append(f"Keyword “{query}”")
        if years_span[0] > 0 or years_span[1] < NO_LIMIT:
            _hi = "any" if years_span[1] >= NO_LIMIT else f"{years_span[1]:g}"
            pills.append(f"Experience {years_span[0]:g}–{_hi}y")
        if not include_unknown:
            pills.append("Unknown tenure excluded")
        if only_alum:
            pills.append("Platform alumni only")
        if min_quality != "low":
            pills.append(f"Parse confidence ≥ {min_quality}")
        if pills:
            st.markdown(
                "".join(f"<span class='pill'>{t}</span>" for t in pills),
                unsafe_allow_html=True,
            )
        st.caption("Click a name to open the profile.")

        def row_label(cid: str) -> str:
            """Two lines: a status chip with the name and score, then one
            line saying WHY -- the strongest signals for a match, the single
            gap for a near miss.

            Two revisions of taste sit in here. "✗ FAILS" in red shouted at
            the reader; in the pipeline-stage language of recruiting CRMs a
            candidate who misses one requirement is a *near miss*, not an
            error, so the chip says that and wears amber -- red is kept for
            things that are broken, and none of these people are broken.
            And the row must answer "why?" on sight in both directions:
            a match names the signals that earned its score, a near miss
            names its one gap with both numbers, because "off on experience"
            without them sends the reader into the profile to learn whether
            the gap is four months or four years.
            """
            c = by_id[cid]
            m = matches.get(cid)
            inv = c.get("years_investment_experience")

            meta = " · ".join([
                f"{'—' if inv is None else f'{inv:g}y'} investing",
                c.get("region") or "—",
                label_of(c.get("approach_family")) or "—",
            ])

            if m is None:
                return f"**{c['display_name']}**\n\n:gray[{meta}]"

            head = (f"**{c['display_name']}**&nbsp;&nbsp;"
                    f"Fit **{m.soft_score:.0%}**")

            if m.is_exact:
                strong = sorted(
                    (x for x in m.soft_criteria if x.weight and x.score >= .5),
                    key=lambda x: x.score * x.weight, reverse=True,
                )[:2]
                why = (" · ".join(
                    f"{CRITERION_LABEL.get(x.key, x.key)} {x.score:.0%}"
                    for x in strong
                ) or "meets every hard requirement")
                return (f"{head}\n\n"
                        f":green[{why}] :gray[· {meta}]")

            fail = m.failed_hard[0]
            gap = (f"{fail.label} {fail.found}".replace("Investment ", "")
                   + f" — needs {fail.required}")
            gap = gap[0].upper() + gap[1:]
            return (f"{head}\n\n"
                    f":orange[{gap}] :gray[· {meta}]")

        # With a role set, the list is two labelled sections -- people who
        # qualify, people one gap away -- rather than one run of rows told
        # apart by chip colour. Section headers carry the counts, so the
        # chips inside the rows became redundant and were dropped; a row's
        # green "why" line or amber gap line is distinction enough once the
        # section has said which side of the line it sits on. Two radio
        # groups share one selection: picking in either clears the other.
        exact_ids = [c["candidate_id"] for c in ordered
                     if (mm := matches.get(c["candidate_id"])) and mm.is_exact]
        near_ids = [c["candidate_id"] for c in ordered
                    if (mm := matches.get(c["candidate_id"]))
                    and not mm.is_exact]

        if requisition and (exact_ids or near_ids):

            def _solo(chosen_key: str, other_key: str) -> None:
                if st.session_state.get(chosen_key) is not None:
                    st.session_state[other_key] = None

            picked_a = picked_b = None
            if exact_ids:
                st.markdown(
                    f"<div class='grouphead' style='color:{STATUS_GOOD}'>"
                    f"Qualify · {len(exact_ids)}</div>",
                    unsafe_allow_html=True,
                )
                picked_a = st.radio(
                    "Qualifying candidates", exact_ids,
                    format_func=row_label, key="pick_exact",
                    index=0, label_visibility="collapsed",
                    on_change=_solo, args=("pick_exact", "pick_near"),
                )
            if near_ids:
                st.markdown(
                    f"<div class='grouphead' style='color:{SERIES_2}'>"
                    f"One gap away · {len(near_ids)}</div>",
                    unsafe_allow_html=True,
                )
                picked_b = st.radio(
                    "Candidates one gap away", near_ids,
                    format_func=row_label, key="pick_near",
                    index=0 if not exact_ids else None,
                    label_visibility="collapsed",
                    on_change=_solo, args=("pick_near", "pick_exact"),
                )
            chosen_id = picked_a or picked_b or (exact_ids + near_ids)[0]
        else:
            chosen_id = st.radio(
                "Candidate",
                options=[c["candidate_id"] for c in ordered],
                format_func=row_label,
                label_visibility="collapsed",
            )

        with st.expander("Table view and export",
                         icon=":material/table:"):
            rows = []
            for c in ordered:
                m = matches.get(c["candidate_id"])
                rows.append({
                    "Candidate": c["display_name"],
                    "Fit": f"{m.soft_score:.0%}" if m else "",
                    "Gap": (
                        f"{m.failed_hard[0].label}: {m.failed_hard[0].found}"
                        if m and m.failed_hard else ("—" if m else "")
                    ),
                    "Yrs investing": c.get("years_investment_experience"),
                    "Yrs career": c.get("years_experience"),
                    "Region": c.get("region") or "—",
                    "Approach": label_of(c.get("approach_family")),
                    "Sector": ", ".join(label_of(x) for x in c.get("sectors", [])),
                    "Firm": c.get("current_firm") or "—",
                    "Data": c["quality"]["band"],
                })
            frame = pd.DataFrame(rows)
            st.dataframe(
                frame, hide_index=True, use_container_width=True,
                column_config={
                    name: st.column_config.Column(help=COLUMN_HELP.get(name, ""))
                    for name in frame.columns
                },
            )
            st.download_button(
                "Download shortlist (CSV)",
                frame.to_csv(index=False).encode(),
                file_name="shortlist.csv", mime="text/csv",
                use_container_width=True,
            )

        if requisition and len(ordered) > 1:
            with st.expander("Compare two candidates",
                             icon=":material/compare_arrows:"):
                pair = st.multiselect(
                    "Pick two",
                    options=[c["candidate_id"] for c in ordered],
                    format_func=lambda cid: by_id[cid]["display_name"],
                    max_selections=2,
                    label_visibility="collapsed",
                )
                if len(pair) == 2:
                    picked = [matches[p] for p in pair if p in matches]
                    if len(picked) == 2:
                        st.plotly_chart(
                            fit_radar(picked, 330), use_container_width=True
                        )
                        st.markdown(
                            component_table(picked), unsafe_allow_html=True
                        )
                elif pair:
                    st.caption("Select one more to compare.")

    # -- Detail -----------------------------------------------------------
    with detail_col.container(border=True):
        c = by_id[chosen_id]
        e = c["extraction"]
        m = matches.get(chosen_id)
        inv = c["years_investment_experience"]
        covered = e["coverage"].get("stocks_covered")

        st.markdown(f"### {c['display_name']}")
        if c["name_source"] == "filename":
            st.caption(
                "⚠ This name comes from the filename — the document never "
                "states one."
            )
        meta = " · ".join(
            p for p in (
                c.get("location"), c.get("current_firm"),
                label_of(c["current_firm_type"]) if c.get("current_firm_type") else None,
            ) if p
        )
        st.markdown(
            f"<div style='color:{MUTED};font-size:13px;margin:-6px 0 8px'>{meta}"
            "</div>" + quality_chip(c["quality"]["band"], c["quality"]["score"],
                                    len(c["flags"])),
            unsafe_allow_html=True,
        )

        if c.get("platform_alum_of"):
            st.success(
                f"**Platform alum — previously at "
                f"{', '.join(c['platform_alum_of'])}.** Surfaced through "
                "pod-to-platform lineage in the knowledge base; the resume "
                "names the pod, never the platform."
            )

        # Headline numbers, Fit first when a role is set.
        cols = st.columns(4 if m else 3)
        offset = 0
        if m:
            # Rendered by hand rather than with st.metric so the colour can
            # carry the verdict: green when every hard requirement is met,
            # bronze when one is not. The other headline numbers stay navy --
            # they describe the candidate, this one describes the decision.
            verdict_colour = STATUS_GOOD if m.is_exact else SERIES_2
            verdict = "qualifies" if m.is_exact else "1 gap"
            share, active_n, total_n = score_basis(m)
            cols[0].markdown(
                f"<div title='{METRIC_HELP['fit']}'>"
                f"<div style='color:{MUTED};font-size:13px'>Fit score</div>"
                f"<div style='color:{verdict_colour};font-size:38px;"
                f"font-weight:700;line-height:1.15'>{m.soft_score:.0%}</div>"
                f"<div style='color:{verdict_colour};font-size:12px;"
                f"font-weight:600'>{verdict}</div>"
                f"<div style='color:{MUTED};font-size:11px;margin-top:2px'>"
                f"from {active_n} of {total_n} signals · {share:.0%} of the "
                f"full weighting</div></div>",
                unsafe_allow_html=True,
            )
            offset = 1
        cols[offset].metric(
            "Career", f"{c['years_experience'] or '—'} yrs",
            help=METRIC_HELP["career"],
        )
        cols[offset + 1].metric(
            "Investing", "—" if inv is None else f"{inv} yrs",
            help=METRIC_HELP["investing"],
        )
        cols[offset + 2].metric(
            "Stocks covered", covered or "—", help=METRIC_HELP["coverage"],
        )

        # Fit, as a shape.
        if m:
            st.markdown("##### Fit against this role")
            for crit in m.hard_criteria:
                # Amber, not red, on the miss -- same reasoning as the result
                # list: a near miss is a state in the pipeline, not an error.
                icon, colour = ("✓", STATUS_GOOD) if crit.passed else ("✗", SERIES_2)
                st.markdown(
                    f"<div style='font-size:13px'><span style='color:{colour};"
                    f"font-weight:700'>{icon}</span> <b>{crit.label}</b> — "
                    f"needs {crit.required}; has {crit.found}</div>",
                    unsafe_allow_html=True,
                )
            radar = fit_radar([m], 350)
            if radar:
                st.plotly_chart(radar, use_container_width=True)
                st.caption(
                    "Shape, not area — a radar exaggerates differences, so the "
                    "component values are given as numbers below."
                )
                st.markdown(component_table([m]), unsafe_allow_html=True)
            with st.expander("Evidence for each component"):
                for crit in m.soft_criteria:
                    if crit.weight <= 0 or not crit.evidence:
                        continue
                    st.markdown(
                        f"**{crit.label}** — {crit.found}  \n"
                        f"<span style='color:{MUTED};font-size:12.5px'>"
                        f"“{crit.evidence[:260]}”</span>",
                        unsafe_allow_html=True,
                    )

        # Profile: every attribute with the text that produced it.
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

        attribute("Investment approach", label_of(e["investment_approach"]["value"]),
                  e["investment_approach"], note=label_of(c.get("approach_family")))
        attribute("Market side", label_of(e["market_side"]["value"]), e["market_side"])
        for sector in e["primary_sectors"]:
            attribute("Sector", label_of(sector["value"]), sector)
        attribute(
            "Markets covered", ", ".join(c.get("coverage_markets", [])) or "—", None,
            note="inferred from location"
            if c.get("coverage_markets_source") == "inferred" else "stated",
        )
        attribute(
            "Seniority",
            f"{label_of(c.get('seniority_band'))} by career · "
            f"{label_of(c.get('investment_seniority_band'))} by investing",
        )

        st.markdown("##### Skills and credentials")
        for cred in c.get("credentials_summary", []) or []:
            st.markdown(f"<div style='font-size:13px'>· {cred}</div>",
                        unsafe_allow_html=True)
        if not c.get("credentials_summary"):
            st.caption("No professional credentials stated.")

        def chips(label: str, values: list[str], colour: str) -> None:
            st.markdown(
                f"<div style='margin-top:9px;font-size:12px;color:{MUTED}'>{label}"
                "</div>" + (
                    " ".join(
                        f"<span style='background:{colour}1a;color:{colour};"
                        f"font-size:11.5px;padding:2px 9px;border-radius:9px;"
                        f"margin:0 4px 4px 0;display:inline-block'>{v}</span>"
                        for v in values
                    ) if values else
                    f"<span style='color:{MUTED};font-size:12.5px'>not stated</span>"
                ),
                unsafe_allow_html=True,
            )

        chips("Software and platforms", c.get("software_tools", []), SERIES_1)
        chips("Analytical methods", c.get("methods", []), SERIES_2)
        chips("Languages", c.get("languages", []), "#4a3aa7")


        # Issues sit AFTER the profile: a reviewer wants to know who this
        # person is before being told what is uncertain about the record. The
        # data-quality badge at the top links down here, so the caveats are
        # one click away rather than in the reader's path.
        st.markdown("<div id='issues'></div>", unsafe_allow_html=True)
        st.markdown(f"##### Issues found in this resume ({len(c['flags'])})")
        if not c.get("flags"):
            st.caption("Nothing flagged — the document parsed cleanly.")
        else:
            st.caption(METRIC_HELP["issues"])
            for f in c["flags"]:
                colour = SERIES_1 if f["source"] == "computed" else SERIES_2
                origin = ("computed from the data" if f["source"] == "computed"
                          else "read from the text")
                st.markdown(
                    f"<div style='font-size:12.5px;margin-bottom:7px;"
                    f"border-left:3px solid {colour};padding-left:9px'>"
                    f"<b>{f['summary']}</b> <span style='color:{MUTED};"
                    f"font-size:11px'>· {origin}</span><br>"
                    f"<span style='color:{MUTED}'>{f['detail']}</span></div>",
                    unsafe_allow_html=True,
                )

        # Reference material, collapsed.
        with st.expander("Experience, as parsed"):
            head = ("<tr><th>Employer</th><th>Resolved to</th><th>Title</th>"
                    "<th>Dates</th><th>Type</th><th>Inv.</th></tr>")
            firm_by_raw = {f["raw"]: f for f in c.get("firms", [])}
            body = []
            for pos in e["positions"]:
                link = firm_by_raw.get(pos["firm"], {})
                resolution = link.get("resolution", "unresolved")
                bad = resolution in {"unresolved", "ambiguous"} and pos[
                    "employment_type"] in {"professional", "internship"}
                dates = (
                    f"{pos.get('start_date') or '?'} – "
                    f"{'present' if pos.get('is_current') else (pos.get('end_date') or '?')}"
                )
                if not pos.get("start_date") and pos.get("duration_months"):
                    dates = f"{pos['duration_months']} mo (duration only)"
                body.append(
                    f"<tr><td>{pos['firm']}</td>"
                    f"<td style='color:{STATUS_BAD if bad else INK}'>"
                    f"{link.get('canonical') or '—'}"
                    f"<span class='res'>{resolution}</span></td>"
                    f"<td>{pos['title']}</td><td>{dates}</td>"
                    f"<td style='color:{MUTED}'>"
                    f"{pos['employment_type'].replace('_', ' ')}</td>"
                    f"<td>{'✓' if pos.get('is_investment_role') else '·'}</td></tr>"
                )
            st.markdown(f"<table class='postbl'>{head}{''.join(body)}</table>",
                        unsafe_allow_html=True)
            if c.get("non_professional_affiliations"):
                st.caption("Excluded from tenure: "
                           + "; ".join(c["non_professional_affiliations"]))

        with st.expander("Education"):
            if not e.get("education"):
                st.caption("No education section could be parsed.")
            for degree in e.get("education", []):
                st.markdown(
                    f"<div style='font-size:13px;margin-bottom:5px'>"
                    f"<b>{degree.get('degree') or 'Degree'}"
                    f"{', ' + degree['field_of_study'] if degree.get('field_of_study') else ''}"
                    f"</b> · {degree['institution']}<span style='color:{MUTED}'>"
                    f" — {degree.get('start_year') or '?'}–"
                    f"{degree.get('graduation_year') or '?'}</span></div>",
                    unsafe_allow_html=True,
                )

        with st.expander("Summary — every parsed field"):
            fields = [
                ("Name / source", f"{c['display_name']} ({c['name_source']})"),
                ("Email", e.get("email") or "—"),
                ("Phone", e.get("phone") or "—"),
                ("Location / region", f"{c.get('location') or '—'} · {c.get('region') or '—'}"),
                ("Career tenure", f"{c['years_experience'] or '—'} yrs"),
                ("Investment tenure", "—" if inv is None else f"{inv} yrs"),
                ("Seniority (career / investing)",
                 f"{label_of(c.get('seniority_band'))} / "
                 f"{label_of(c.get('investment_seniority_band'))}"),
                ("Approach", f"{c.get('approach')} → {c.get('approach_family')}"),
                ("Market side", c.get("market_side") or "—"),
                ("Sectors", ", ".join(c.get("sectors", [])) or "—"),
                ("Asset classes", ", ".join(c.get("asset_classes", [])) or "—"),
                ("Coverage markets",
                 f"{', '.join(c.get('coverage_markets', [])) or '—'} "
                 f"({c.get('coverage_markets_source')})"),
                ("Stocks covered", str(covered or "—")),
                ("Current firm / type",
                 f"{c.get('current_firm') or '—'} · {c.get('current_firm_type') or '—'}"),
                ("Employers", "; ".join(c.get("employers", [])) or "—"),
                ("Non-professional", "; ".join(c.get("non_professional_affiliations", [])) or "—"),
                ("Buy / sell side experience",
                 f"{c.get('has_buy_side_experience')} / {c.get('has_sell_side_experience')}"),
                ("Platform alumni", ", ".join(c.get("platform_alum_of", [])) or "—"),
                ("Credentials", "; ".join(c.get("credentials_summary", [])) or "—"),
                ("Languages", ", ".join(c.get("languages", [])) or "—"),
                ("Software", ", ".join(c.get("software_tools", [])) or "—"),
                ("Methods", ", ".join(c.get("methods", [])) or "—"),
                ("Positions parsed", str(len(e["positions"]))),
                ("Parse confidence",
                 f"{c['quality']['band']} {c['quality']['score']} · "
                 f"{len(c['flags'])} issue(s)"),
                ("Missing fields", ", ".join(c["quality"]["missing_fields"]) or "none"),
                ("Source file", c["source_file"]),
            ]
            st.markdown(
                "<table class='recordtbl'>"
                + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields)
                + "</table>",
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
        options=["low", "medium", "high"], value="low",
        help="Counts built from thin records overstate the bench.",
    )
    group_by = ctrl2.selectbox(
        "Break down by", ["Region", "Market side", "Seniority (investing)"],
        help="The row dimension of the coverage matrix.",
    )
    charted = [c for c in filtered
               if BAND_ORDER[c["quality"]["band"]] >= BAND_ORDER[chart_quality]]
    ctrl3.markdown(
        f"<div style='padding-top:26px;color:{MUTED};font-size:12.5px'>"
        f"<b style='color:{NAVY}'>{len(charted)}</b> of {len(candidates)} "
        "candidates charted, after the sidebar refinements and this threshold."
        "</div>",
        unsafe_allow_html=True,
    )
    if not charted:
        st.info("No candidates left at this threshold.")
    else:
        st.caption("Empty cells are the point: the seats this pool cannot fill.")
        GROUP_KEY = {
            "Region": lambda c: c.get("region"),
            "Market side": lambda c: label_of(c.get("market_side")),
            "Seniority (investing)": lambda c: label_of(
                c.get("investment_seniority_band")),
        }[group_by]
        sectors = sorted({s for c in charted for s in c.get("sectors", [])})
        groups = sorted({g for c in charted if (g := GROUP_KEY(c))})

        if sectors and groups:
            matrix = [[sum(1 for c in charted
                           if GROUP_KEY(c) == g and s in c.get("sectors", []))
                       for s in sectors] for g in groups]
            fig = go.Figure(go.Heatmap(
                z=matrix, x=[label_of(s) for s in sectors], y=groups,
                colorscale=[[0.0, SURFACE], [0.001, SEQ[0]]] + [
                    [0.001 + 0.999 * i / (len(SEQ) - 1), col]
                    for i, col in enumerate(SEQ)],
                zmin=0, showscale=False, xgap=2, ygap=2,
                text=[[v or "" for v in row] for row in matrix],
                texttemplate="%{text}", textfont=dict(size=13),
                hovertemplate="%{y} · %{x}<br>%{z} candidate(s)<extra></extra>",
            ))
            fig.update_layout(title=f"Sector coverage by {group_by.lower()}",
                              showlegend=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.markdown("**Career length vs. investing tenure**")
            st.caption(
                "The gap is the point. Screening on total experience alone "
                "overstates how long these candidates have been investing."
            )
            pool = sorted([c for c in charted if c["years_experience"] is not None],
                          key=lambda c: c["years_experience"])
            if pool:
                names = [c["display_name"] for c in pool]
                total = [c["years_experience"] for c in pool]
                invest = [c["years_investment_experience"] or 0 for c in pool]
                fig = go.Figure()
                for name, t, i in zip(names, total, invest):
                    fig.add_trace(go.Scatter(
                        x=[i, t], y=[name, name], mode="lines",
                        line=dict(color=GRID, width=2),
                        showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=total, y=names, mode="markers", name="Career total",
                    marker=dict(color=SERIES_2, size=10,
                                line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>Career %{x} yrs<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=invest, y=names, mode="markers", name="Investing",
                    marker=dict(color=SERIES_1, size=10,
                                line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>Investing %{x} yrs<extra></extra>"))
                fig.update_layout(
                    showlegend=True, xaxis_title="Years",
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

        with right:
            st.markdown("**Most common software and credentials**")
            st.caption(
                "What this pool can actually operate. A role naming a tool "
                "nobody holds is a sourcing problem, not a screening one."
            )
            tally: dict[str, int] = {}
            for c in charted:
                for tool in c.get("software_tools", []):
                    tally[tool] = tally.get(tool, 0) + 1
            creds: dict[str, int] = {}
            for c in charted:
                for cred in c.get("credentials_summary", []):
                    head = cred.split(" - ")[0].split(" (")[0]
                    creds[head] = creds.get(head, 0) + 1
            top_tools = sorted(tally.items(), key=lambda kv: -kv[1])[:9]
            top_creds = sorted(creds.items(), key=lambda kv: -kv[1])[:5]
            if top_tools or top_creds:
                labels = [k for k, _ in top_creds] + [k for k, _ in top_tools]
                values = [v for _, v in top_creds] + [v for _, v in top_tools]
                colours = [SERIES_2] * len(top_creds) + [SERIES_1] * len(top_tools)
                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h", marker_color=colours,
                    marker_line=dict(color=SURFACE, width=2),
                    hovertemplate="%{y}<br>%{x} candidate(s)<extra></extra>"))
                fig.update_layout(showlegend=False, xaxis_title="Candidates")
                fig.update_yaxes(showgrid=False, autorange="reversed")
                fig.update_xaxes(showgrid=True, gridcolor=GRID, dtick=1)
                st.plotly_chart(styled_chart(fig, 360), use_container_width=True)
                st.caption("Bronze = credentials · slate = software.")


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
    bands = {b: sum(1 for c in candidates if c["quality"]["band"] == b)
             for b in QUALITY_COLOUR}
    scores = sorted(c["quality"]["score"] for c in candidates)

    cols = st.columns(4)
    cols[0].metric("Median confidence", f"{scores[len(scores) // 2]:.2f}")
    for col, (band, count) in zip(cols[1:], bands.items()):
        col.metric(f"{band.title()} confidence", count)

    st.markdown("---")
    st.markdown("**Parse confidence by candidate**")
    fig = go.Figure(go.Bar(
        x=[c["quality"]["score"] for c in ranked],
        y=[c["display_name"] for c in ranked], orientation="h",
        marker_color=[QUALITY_COLOUR[c["quality"]["band"]] for c in ranked],
        marker_line=dict(color=SURFACE, width=2),
        text=[f"{c['quality']['score']:.2f}" for c in ranked],
        textposition="outside", textfont=dict(size=11, color=INK),
        customdata=[len(c["flags"]) for c in ranked],
        hovertemplate="%{y}<br>confidence %{x:.2f}<br>%{customdata} issue(s)"
                      "<extra></extra>"))
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
    st.caption("Lowest confidence first — verify these before acting on them.")
    for c in ranked:
        band = c["quality"]["band"]
        with st.expander(
            f"{c['display_name']} — {band} {c['quality']['score']} · "
            f"{len(c['flags'])} issue(s)", expanded=(band != "high"),
        ):
            if c["quality"]["missing_fields"]:
                st.markdown(
                    f"<span style='color:{STATUS_BAD};font-size:12.5px'>"
                    f"<b>Missing:</b> {', '.join(c['quality']['missing_fields'])}"
                    "</span>", unsafe_allow_html=True)
            if not c["flags"]:
                st.caption("Nothing flagged.")
            for f in c["flags"]:
                colour = SERIES_1 if f["source"] == "computed" else SERIES_2
                origin = ("computed from the data" if f["source"] == "computed"
                          else "read from the text")
                st.markdown(
                    f"<div style='font-size:12.5px;margin-bottom:7px;border-left:"
                    f"3px solid {colour};padding-left:9px'><b>{f['summary']}</b> "
                    f"<span style='color:{MUTED};font-size:11px'>· {origin}</span>"
                    f"<br><span style='color:{MUTED}'>{f['detail']}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("How each document was read"):
        log = ROOT / "data" / "extraction_log.csv"
        if log.exists():
            st.dataframe(pd.read_csv(log), hide_index=True,
                         use_container_width=True)


# ===========================================================================
# Method -- every parameter, stated
# ===========================================================================

with tab_method:
    st.markdown("#### How this works, with the numbers")

    st.markdown("##### 1 · Reading the document")
    st.markdown(
        """
The largest source of error in a resume pipeline is not the model — it is
losing content before the model ever sees it. `python-docx` reads neither
tables nor floating text boxes, and both carry real content in this corpus.
        """
    )
    st.markdown(
        f"""
| Step | Parameter | Value |
|---|---|---|
| Word | paragraphs, tables and floating text boxes | read in document order |
| Word | floating text boxes | hoisted to the top (they carry header content) |
| PDF | column split | line-start bimodality; gap ≥ 15% of page width; right cluster ≥ 4 lines and ≥ 15% of lines |
| PDF | ligature repair | 26 damaged tokens observed; `ti` default, `tf` for `Por?olio`; residue reported |
| All | thin-document warning | under 1,500 characters recovered |
        """
    )

    st.markdown("##### 2 · Extraction by the model")
    st.markdown(
        """
Output is constrained to a Pydantic schema passed as a tool definition, so the
model must return an object of that exact shape. Validation failures are sent
back once with the specific errors attached rather than retried blindly, and
every evidence quote is checked to appear verbatim in the source text.
        """
    )
    st.markdown(
        """
| Parameter | Value |
|---|---|
| Model | `claude-sonnet-5` |
| Max output tokens | 8,000 |
| Attempts | 2 (one corrective retry with the validation errors) |
| Caching key | hash of (source text, model, schema, system prompt) |
| Cost | ~\$0.075 per resume; ~\$0.75 for this corpus |
| Evidence check | quote must appear verbatim; unverified quotes reduce confidence |
        """
    )

    st.markdown("##### 3 · What the model is NOT asked")
    st.markdown(
        """
Anything code can settle is settled by code — it is cheaper, auditable, and a
wrong answer is a one-line fix in a YAML file rather than a prompt experiment.

| Computed, not asked | How |
|---|---|
| Years of experience | union of dated intervals; overlaps merged, not summed |
| Investment tenure | same, restricted to roles marked as investment work |
| Seniority band | from tenure: junior ≤ 5 yrs, mid ≤ 9, senior above |
| Firm type, parent platform, region | knowledge-base lookup |
| Employment gaps | ≥ 6 months between consecutive professional roles |
| Concurrent roles | ≥ 2 months overlap between two professional roles |
| Email validity | structural check only; never repaired |
        """
    )

    st.markdown("##### 4 · Parse confidence")
    st.markdown(
        """
Each record starts at 1.0 and is deducted. Formatting carries **zero** weight:
how neatly a resume is formatted tracks regional convention and native language
far more than it tracks the candidate, and in this corpus every table-heavy
document belongs to the same region.

| Issue | Deduction |
|---|---|
| No work history | 0.40 |
| No dates on any position | 0.20 |
| Fewer than 1,500 characters recovered | 0.20 |
| No name in the document | 0.15 |
| No location stated | 0.10 |
| Some positions undated | 0.08 |
| Unverified evidence quote | 0.08 each, capped at 0.20 |
| Low-confidence classification | 0.05 each, capped at 0.15 |
| Internal contradiction | 0.06 each |
| Date anomaly / attribution ambiguity | 0.04 each |
| Missing field, typo | 0.02 each |
| Formatting | 0.00 |

Bands: **high** ≥ 0.80 · **medium** ≥ 0.55 · **low** below.
        """
    )

    st.markdown("##### 5 · Matching and the Fit score")
    st.markdown(
        """
**Hard requirements disqualify.** Market, approach, sector and the experience
band are checked first. A candidate outside any of them is not ranked lower —
they are not a match, and the search says so even when that means returning
nobody. A tenure that could not be computed cannot satisfy a numeric
requirement, so it fails rather than passing silently.

**Fit ranks only those who already qualify.** It is a weighted mean of the soft
components below, each scored 0–1:
        """
    )
    weights_rows = "\n".join(
        f"| {CRITERION_LABEL.get(k, k)} | {v:.0%} | {CRITERION_HELP.get(k, '')} |"
        for k, v in sorted(store.weights.items(), key=lambda kv: -kv[1])
    )
    st.markdown(
        "| Component | Weight | What it measures |\n|---|---|---|\n" + weights_rows
    )
    st.markdown(
        """
`Fit = Σ(component × weight) / Σ(weight)`

**Requirement similarity** is a pluggable backend. The default combines lexical
overlap with a curated concept map, because this domain runs on paraphrase — a
posting says "catalysts" where a resume says "earnings events".

| Parameter | Value |
|---|---|
| Combination | `min(1, 0.45 × lexical + 0.75 × conceptual)` |
| Lexical | share of the requirement's content words present in the sentence |
| Conceptual | share of the requirement's concepts present in the sentence |
| Short-sentence damping | sentences under 6 content words scaled by `0.55 + 0.075 × n` |
| Concept matching | word-boundary, not substring |
| One gap away | fails exactly one hard requirement; two or more are not listed |

The concept map is matched on word boundaries because an earlier substring
version listed `r` for the R language, and `"r" in text` is true of nearly
every English sentence — which made every requirement score about 0.75 on its
concept term alone and rendered the evidence quotes arbitrary. Neural sentence
embeddings would absorb paraphrase without a curated map and can be dropped in
unchanged; at a few hundred candidate sentences a curated map performs
comparably, adds no large dependency to a free-tier deployment, and is
auditable — a bad match is fixed by editing a line of YAML.
        """
    )

    st.markdown("##### 6 · Why a job description is not uploaded here")
    st.markdown(
        """
Parsing a pasted job description is the same structured-extraction problem as
parsing a resume, and the code path is straightforward. It is deliberately not
wired into this deployment: it would require a live API key in a **public**
app, where any visitor could spend it. Saved requisitions and the *Define a
role* form give the same object without that exposure. Running the pipeline
locally with a key in `.env` covers the case where a new posting needs
importing.
        """
    )

    st.caption(
        "Built for the Millennium Business Development data science case "
        "study · github.com/yc4379-commits/m-case-study"
    )
