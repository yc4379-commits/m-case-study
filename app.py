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
import math
import re
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
# One navy, one bronze, one deep green -- and the green only ever means
# "clears every hard requirement".
#
# The palette had drifted into a traffic light bolted onto a navy chrome: a
# 2010s-bright #0ca30c (3.4:1 on white -- below the 4.5:1 needed for the
# 12px text it was set in), a saturated amber that read as a browser warning
# bar, a fire-engine red, and a stray violet on the language chips. Five
# hues, four of them shouting, none of them related to the navy or the
# bronze that carry the rest of the interface.
#
# Now: NAVY_DEEP is the masthead, NAVY the chrome accent, SERIES_1 the data
# slate, SERIES_2 the single warm accent -- and every status colour is
# pulled into that same low-chroma, dark-lightness register so it can sit at
# 11.5px and still pass contrast. Nothing in the interface is brighter than
# the content.
NAVY_DEEP = "#0a1b38"
NAVY = "#0b2f5e"
NAVY_WASH = "#f4f7fb"
SERIES_1 = "#2f5f98"
SERIES_2 = "#96622e"
SEQ = ["#eef3f9", "#d5e1ef", "#b4c9e1", "#8badd0", "#5d8bba", "#2f5f98", "#123a6f"]
STATUS_GOOD = "#146c43"   # 5.4:1 on white, was 3.4:1
STATUS_WARN = "#96622e"   # the bronze accent, not a browser amber
STATUS_BAD = "#8f2c2c"
# Charts read as a single navy ramp: darker is better parsed. A green /
# amber / red bar chart puts three saturated hues on one axis to encode one
# ordered variable -- ordered data wants one hue and three lightnesses, which
# is also the quieter, more expensive-looking answer.
QUALITY_COLOUR = {"high": "#123a6f", "medium": "#5d8bba", "low": "#b4c9e1"}
INK = "#0b1b2b"
MUTED = "#64748b"
GRID = "#e2e8f0"
SURFACE = "#ffffff"
PAGE = "#fbfcfd"

BAND_ORDER = {"low": 0, "medium": 1, "high": 2}

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
        background: {NAVY_DEEP}; padding: 0 24px;
      }}
      .brandbar b {{ font-family: 'Playfair Display', Georgia, serif;
                     font-size: 19px; font-weight: 500; color: #fff;
                     letter-spacing: .005em; white-space: nowrap; }}
      .brandbar span {{ font-size: 12px; color: rgba(255,255,255,.55);
                        padding-top: 3px; }}
      .brandbar .ctx {{ margin-left: auto; font-size: 12px;
                        color: rgba(255,255,255,.75); padding-top: 0; }}
      /* The view switcher lives in the masthead, not above the results:
         Insights, Data quality and Method are views OVER the pool, one
         level up from the working screen, and the reviewer read them as
         siblings of the candidate list when they sat next to it. The tab
         strip is lifted out of the page flow into the bar; the tab PANELS
         stay exactly where they were. */
      div[role="tablist"] {{
        position: fixed; top: 0; left: 235px; height: 50px;
        z-index: 1000003; background: transparent;
        border-bottom: none !important; gap: 24px; align-items: center;
        display: flex;
      }}
      div[role="tablist"] [data-testid="stTab"] {{
        color: rgba(255,255,255,.62); background: transparent;
        border-bottom: 2px solid transparent; padding: 4px 2px;
      }}
      div[role="tablist"] [data-testid="stTab"] p {{
        font-size: 13px !important; font-weight: 500;
        color: inherit !important;
      }}
      div[role="tablist"] [data-testid="stTab"]:hover {{ color: #fff; }}
      div[role="tablist"] [data-testid="stTab"][aria-selected="true"] {{
        color: #fff !important; border-bottom: 2px solid #fff;
      }}
      div[role="tablist"] [data-testid="stTab"][aria-selected="true"] p {{
        font-weight: 600;
      }}
      /* The sidebar refines the Candidates search. Insights, Data quality,
         Method and Ask are views over the whole pool, so showing filters
         there would imply they apply; the sidebar disappears instead
         whenever the first tab is not the selected one. */
      .stApp:has(div[role="tablist"]
                 [data-testid="stTab"]:first-of-type[aria-selected="false"])
        section[data-testid="stSidebar"] {{ display: none; }}
      /* Reopen control for a collapsed sidebar: it normally sits in the
         strip the masthead now covers, which left no way back. */
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stExpandSidebarButton"] {{
        position: fixed; top: 58px; left: 10px; z-index: 1000003;
        visibility: visible !important;
        background: {SURFACE}; border: 1px solid {GRID};
        border-radius: 8px; padding: 2px;
        box-shadow: 0 1px 4px rgba(11,27,43,.08);
      }}
      [data-testid="stExpandSidebarButton"] * {{
        visibility: visible !important;
      }}
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
      [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 650;
                                       font-size: 24px !important; }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      hr {{ border-color: {GRID}; }}
      .grouphead {{ font-size: 11px; font-weight: 600;
                    text-transform: uppercase; letter-spacing: .08em;
                    margin: 12px 0 8px; }}
      /* The one informational pill. Every tag in the interface uses it. */
      .pill {{ display: inline-block; background: #eef1f5;
               border: 1px solid #e2e8f0; border-radius: 999px;
               padding: 3px 12px; font-size: 11.5px;
               color: #46586b; margin: 0 6px 6px 0; }}
      /* The single exception: a record we could not fully read. Bronze,
         because it is a caveat about the data, not a verdict on the person. */
      .pill-note {{ background: #faf5ef; border-color: #e8ded1;
                    color: #7a512a; font-weight: 600; }}
      .pill-slate {{ background: #eef3f9; border-color: #d5e1ef;
                     color: #2f5f98; }}
      .pill-navy {{ background: #e9eef6; border-color: #cddbec;
                    color: #123a6f; }}
      .pill-bronze {{ background: #f8f3ec; border-color: #e8ded1;
                      color: #7a512a; }}
      /* The posting's requirement lines, nested under the dimension they
         feed. A hairline bar per line, not a second chart. */
      .reqlist {{ display: flex; flex-direction: column; gap: 4px;
                  margin: 7px 0 0 0; padding-left: 11px;
                  border-left: 1px solid {GRID}; }}
      .reqline {{ display: flex; align-items: center; gap: 9px;
                  font-size: 12px; color: #46586b; }}
      .reqline i {{ display: block; height: 3px; border-radius: 2px;
                    background: #8badd0; flex: none; max-width: 46px;
                    min-width: 2px; }}
      .reqline span {{ flex: 1; overflow: hidden; text-overflow: ellipsis;
                       white-space: nowrap; }}
      .reqline b {{ font-variant-numeric: tabular-nums; color: {INK};
                    font-weight: 600; }}
      .sechead {{ font-size: 11.5px; font-weight: 700;
                  text-transform: uppercase; letter-spacing: .08em;
                  color: {INK}; margin: 16px 0 8px; display: flex;
                  align-items: center; gap: 7px; }}
      .sechead i {{ font-style: normal; font-size: 9.5px; font-weight: 700;
                    width: 14px; height: 14px; border-radius: 8px;
                    background: #eef1f5; color: {MUTED}; cursor: help;
                    display: inline-flex; align-items: center;
                    justify-content: center; letter-spacing: 0; }}
      /* One tooltip mechanism for every custom help mark. Native title=
         needs a ~1.5s hover and looks dead until then; this shows the
         instant the pointer arrives, in the app's own style. */
      [data-tip] {{ position: relative; cursor: help; }}
      [data-tip]:hover::after {{
          content: attr(data-tip); position: absolute; left: 0;
          top: calc(100% + 7px); z-index: 9999; width: 340px;
          max-width: 70vw; white-space: normal; text-transform: none;
          letter-spacing: 0; font-size: 11.5px; font-weight: 400;
          line-height: 1.55; text-align: left; background: {NAVY_DEEP};
          color: #eef2f7; padding: 10px 13px; border-radius: 9px;
          box-shadow: 0 6px 22px rgba(10,27,56,.28);
          pointer-events: none; }}
      .sechead i[data-tip]:hover::after {{ left: -12px; }}
      /* The contribution ledger: what each scoring dimension actually found,
         at the weight it was worth. */
      .ledger {{ display: flex; flex-direction: column; gap: 12px; }}
      .ldrow {{ display: flex; flex-direction: column; gap: 5px; }}
      .ldtop {{ display: flex; align-items: baseline; gap: 9px; }}
      .ldlbl {{ font-size: 13px; font-weight: 600; color: {INK};
                cursor: help; }}
      .ldw {{ font-size: 11.5px; color: {MUTED}; }}
      .ldval {{ margin-left: auto; font-size: 13px; font-weight: 700;
                color: {INK}; font-variant-numeric: tabular-nums; }}
      .ldbar {{ height: 5px; background: #eff2f6; border-radius: 999px;
                overflow: hidden; }}
      .ldbar i {{ display: block; height: 100%; background: #123a6f;
                  border-radius: 999px; }}
      .lddid {{ font-size: 12.5px; line-height: 1.5; color: #46586b; }}
      .ldrow-idle .ldlbl, .ldrow-idle .ldval {{ color: {MUTED};
                                                font-weight: 500; }}
      .ldrow-idle .lddid {{ color: #94a3b8; font-style: italic; }}
      .hardgrid {{ display: grid; grid-template-columns: 1fr 1fr;
                   gap: 6px 22px; }}
      .hardrow {{ font-size: 12.5px; color: {INK}; }}
      /* One attribute, three ranked tiers, inside one hairline rule. */
      .attr {{ margin: 0 0 15px; padding-left: 12px;
               border-left: 2px solid #e6ecf4; }}
      .attrlabel {{ font-size: 9.5px; font-weight: 700; letter-spacing: .1em;
                    text-transform: uppercase; color: {MUTED};
                    margin-bottom: 2px; }}
      .attrvalue {{ font-size: 16px; font-weight: 600; color: {INK};
                    line-height: 1.25; }}
      /* Figures summary cards: a stated figure can be a whole phrase, so
         the headline sits a size below .attrvalue or it wraps into a
         block that shouts over the label. */
      .figval {{ font-size: 13px; font-weight: 600; color: {INK};
                 line-height: 1.35; }}
      .attrnote {{ font-size: 11.5px; font-weight: 400; color: {MUTED};
                   margin-left: 8px; }}
      .attrwhy {{ display: flex; flex-wrap: wrap; align-items: center;
                  gap: 4px 5px; margin-top: 6px; }}
      .kw {{ font-size: 10.5px; background: #f4efe8; color: #7a512a;
             padding: 1px 8px; border-radius: 999px; white-space: nowrap; }}
      .conf {{ font-size: 10.5px; color: #a3aeba; margin-left: 2px; }}
      .attrquote {{ font-size: 11.5px; line-height: 1.5; color: #7b8794;
                    margin-top: 5px; cursor: help; display: -webkit-box;
                    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                    overflow: hidden; }}
      .attrquote b {{ color: #5a6b7d; font-weight: 600; }}
      /* Computed facts: a quiet key-value row, deliberately not an attr. */
      .fact {{ display: flex; align-items: baseline; gap: 12px;
               font-size: 12.5px; padding: 6px 0;
               border-bottom: 1px solid #f0f3f7; }}
      .fact span {{ color: {MUTED}; min-width: 118px; flex: none; }}
      .fact b {{ color: {INK}; font-weight: 600; }}
      .fact i {{ color: #a3aeba; font-style: normal; font-size: 11.5px;
                 margin-left: auto; }}
      /* The evidence sentence, at two lines with the rest on hover. The
         quote is not optional -- its LENGTH was the layout problem. */
      .quote {{ font-size: 12px; line-height: 1.5; color: {MUTED};
                border-left: 2px solid {GRID}; padding-left: 9px;
                margin: 5px 0 2px; cursor: help; display: -webkit-box;
                -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                overflow: hidden; }}
      .recordtbl {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
      .recordtbl td {{ padding: 5px 8px; border-bottom: 1px solid {GRID};
                       vertical-align: top; }}
      .recordtbl td:first-child {{ color: {MUTED}; width: 42%; }}
      /* The shortlist table: typeset, not gridded. */
      .pill-empty {{ background: transparent; border: 1px dashed #cbd5e1;
                     color: #94a3b8; }}
      .odraft {{ border: 1px solid {GRID}; border-radius: 12px;
                 background: {SURFACE}; padding: 16px 18px; margin: 4px 0 10px; }}
      .odraft-name {{ font-size: 15px; font-weight: 650; color: {INK}; }}
      .odraft-name span {{ font-weight: 400; font-size: 12.5px; color: {MUTED}; }}
      .odraft-contact {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}
      .odraft-role {{ font-size: 13.5px; font-weight: 600; color: {INK}; }}
      .odraft-verdict {{ display: block; font-size: 12px; font-weight: 400;
                         color: {MUTED}; margin-top: 1px; }}
      .odraft-why {{ font-size: 12.5px; color: {INK}; margin-top: 8px; }}
      .ctblwrap {{ overflow-x: auto; max-width: 100%; padding-bottom: 4px; }}
      .ctbl {{ border-collapse: separate; border-spacing: 0; width: 100%;
               font-size: 12.5px; table-layout: auto; border: none !important; }}
      /* Streamlit's markdown stylesheet draws a box around every th/td it
         finds; strip it so only our own hairline row rules remain. */
      .ctbl th, .ctbl td {{
        border-left: none !important; border-right: none !important;
        border-top: none !important; background: transparent !important;
      }}
      .ctbl th {{ text-align: left; font-size: 9.5px; font-weight: 700;
                  text-transform: uppercase; letter-spacing: .09em;
                  color: {MUTED}; padding: 0 12px 7px 0;
                  border-bottom: 1px solid {GRID}; white-space: nowrap; }}
      .ctbl th.num, .ctbl td.num {{ text-align: right; padding-right: 14px;
                                    font-variant-numeric: tabular-nums; }}
      .ctbl td {{ padding: 9px 12px 9px 0; border-bottom: 1px solid #eef1f5;
                  vertical-align: middle; }}
      .ctbl tr:last-child td {{ border-bottom: none; }}
      .ctbl td.who {{ font-size: 13px; font-weight: 600; color: {INK};
                      line-height: 1.3; }}
      .ctbl td.sub {{ color: {MUTED}; }}
      /* No bars, no bronze, no dots: after review the table carries
         exactly two visual weights -- semibold ink for names and Fit,
         grey for everything else. The Fit number IS the visualisation. */
      .ctbl .whofirm {{ display: block; font-weight: 400; font-size: 11.5px;
                        color: {MUTED}; margin-top: 1px; }}
      .ctbl .dim {{ color: #cbd5e1; }}
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
      /* The name gets its own step. At a single 13px, ten cards blur into
         one block; 15px semibold against a 13px description line is enough
         separation to scan the column without reading it. */
      div[role="radiogroup"] label p:first-of-type {{
        font-size: 15px !important; font-weight: 600; line-height: 1.35;
      }}
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
      /* Long option labels (expanded credential names) wrap; a truncated
         option cannot be told apart from its neighbour. */
      [data-baseweb="popover"] [role="option"],
      [data-baseweb="popover"] [role="option"] * {{
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important;
      }}
      [data-baseweb="tag"] {{ max-width: 100% !important; height: auto !important; }}
      [data-baseweb="tag"] span {{
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; max-width: none !important;
      }}
      /* The role is this app's central claim, so it gets a permanent band
         rather than a collapsed step. Folded shut behind "Step 1", a
         first-time visitor saw a filter sidebar and a list -- exactly the
         tool this design argues against. */
      .rolebar {{
        display: flex; align-items: center; flex-wrap: wrap; gap: 10px 14px;
        background: #faf7f3; border: 1px solid #e8ded1;
        border-radius: 12px; padding: 12px 16px; margin: 0 0 12px;
      }}
      .rolebar-empty {{ background: #f7f9fc; border-color: {GRID}; }}
      .rolekick {{ font-size: 10px; font-weight: 700; letter-spacing: .1em;
                   color: {SERIES_2}; }}
      .rolebar-empty .rolekick {{ color: {MUTED}; }}
      .roletitle {{ font-size: 15px; font-weight: 600; color: {INK}; }}
      .roleteam {{ font-size: 12.5px; color: {MUTED}; }}
      .hardchips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
      .hardchip {{ background: #fff; border: 1px solid #e4d8c8;
                   border-radius: 999px; padding: 3px 11px;
                   font-size: 11.5px; color: #7a512a; }}
      .hardchip b {{ color: {INK}; }}
      .rolecount {{ margin-left: auto; font-size: 11.5px; color: {MUTED};
                    max-width: 320px; text-align: right; }}
      /* One large string per screen. Every other label sat between 11 and
         14px, which is a screen with no entry point. */
      .candname {{ font-size: 26px; font-weight: 600; letter-spacing: -.015em;
                   color: {INK}; line-height: 1.15; margin: 0 0 2px; }}
      /* The profile's section switcher: a pill row, deliberately shaped
         unlike the masthead's tab strip. st.download_button is a different
         testid, so the exports below keep their own styling. */
      div[data-testid="stButton"] > button {{
        border-radius: 999px; border: 1px solid {GRID}; background: {SURFACE};
        color: {MUTED}; font-weight: 500; padding: 4px 6px; min-height: 0;
        white-space: nowrap; overflow: hidden;
      }}
      div[data-testid="stButton"] > button p {{ font-size: 12.5px !important;
                                                white-space: nowrap; }}
      div[data-testid="stButton"] > button:hover {{
        border-color: #d9e1ea; background: #f7f9fc; color: {INK};
      }}
      div[data-testid="stButton"] > button[kind="primary"],
      div[data-testid="stButton"] > button[data-testid$="primary"] {{
        background: {NAVY_WASH}; border-color: {NAVY}; color: {NAVY};
      }}
      div[data-testid="stButton"] > button[kind="primary"] p,
      div[data-testid="stButton"] > button[data-testid$="primary"] p {{
        font-weight: 600; color: {NAVY} !important;
      }}
      /* The Fit bar in a result row is a monospace run of block characters,
         so it aligns down the whole column at any zoom -- ranking becomes
         visible instead of merely stated. */
      div[role="radiogroup"] label code {{
        font-family: ui-monospace, 'SF Mono', Menlo, monospace;
        font-size: 11.5px; letter-spacing: -.5px; background: transparent;
        color: {SERIES_1}; padding: 0;
      }}
      div[role="radiogroup"] label:has(input:checked) code {{
        color: {NAVY};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)




# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_NAME_HONORIFICS = {"dr", "mr", "ms", "mrs", "prof"}


@st.cache_data(show_spinner=False)
def load_candidates() -> list[dict]:
    """Load the parsed dataset, normalising display names.

    Honorifics come off at load time: a shortlist that addresses one
    candidate as "Dr." and the rest by name reads as if the tool ranks
    titles, and every view downstream inherits whatever this function
    returns -- one strip here beats five strips in the rendering code.
    """
    rows = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else []
    for c in rows:
        parts = (c.get("display_name") or "").split()
        while parts and parts[0].lower().rstrip(".") in _NAME_HONORIFICS:
            parts.pop(0)
        name = " ".join(parts) if parts else (c.get("display_name") or "")
        # "MARINA SILVA COSTA" is a resume-header typographic convention, not
        # a name. One all-caps row in a list of ten reads as emphasis -- as if
        # the tool ranked her first -- so case is normalised here, once, and
        # every view downstream inherits it.
        if name and name == name.upper():
            name = name.title()
        c["display_name"] = name
    return rows


@st.cache_resource(show_spinner=False)
def load_requisitions() -> Requisitions:
    return Requisitions.load()


candidates = load_candidates()
if not candidates:
    st.error("No dataset found. Build it first:  `python src/build_dataset.py`")
    st.stop()

store = load_requisitions()

# The experience slider's ceiling is the pool's own maximum, not an arbitrary
# round number -- a 0-40 scale against a pool topping out at 13.7 wastes two
# thirds of the control's travel. The handle AT the ceiling means "no upper
# limit", so a new candidate above today's maximum is never silently hidden;
# the ceiling simply moves when the pool does.
NO_LIMIT = float(math.ceil(max(
    (c.get("years_investment_experience") or 0.0) for c in candidates
)))

st.markdown(
    '<div class="brandbar"><b>talent intelligence</b>'
    f"<span class='ctx'>{len(candidates)} candidates · "
    f"{len(store.items)} roles</span></div>",
    unsafe_allow_html=True,
)
by_id = {c["candidate_id"]: c for c in candidates}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Deep coverage is the pool's upper quartile, not a round number: "20 names"
# is remarkable in one pool and unremarkable in another.
_COVERAGES = sorted(
    x for x in (
        (c.get("extraction", {}).get("coverage") or {}).get("stocks_covered")
        for c in candidates
    ) if x
)
_COVERAGE_DEEP = (_COVERAGES[int(len(_COVERAGES) * 0.75)]
                  if _COVERAGES else 10 ** 9)
_APPROACH_COUNTS: dict[str, int] = {}
for _c in candidates:
    _f = _c.get("approach_family")
    if _f:
        _APPROACH_COUNTS[_f] = _APPROACH_COUNTS.get(_f, 0) + 1
_RARE_APPROACHES = {
    f for f, n in _APPROACH_COUNTS.items() if n / len(candidates) < 0.4
}
_BOTH_SIDES_RARE = sum(
    1 for c in candidates
    if c.get("has_buy_side_experience") and c.get("has_sell_side_experience")
) / max(len(candidates), 1) < 0.4


def _fmt_pct(x) -> str:
    """n/a-aware percent for the evaluation table."""
    return "n/a" if x is None else f"{x:.0%}"


def label_of(value: str | None) -> str:
    return value.replace("_", " ").title() if value else "—"


def record_note(c: dict) -> str:
    """What is wrong with this RECORD, in words, or nothing at all.

    This replaces a badge reading "Data quality: high 0.94". Two faults. It
    was ambiguous in exactly the way the pipeline must never be -- a reader
    could not tell whether "high" praised the document or the candidate, and
    "0.94" is a number no recruiter has a use for. And it was rendered on
    every row, including the majority with nothing wrong, so the one record
    that genuinely cannot be trusted looked like all the others.

    A clean record now says nothing: silence is the correct report for "we
    read this document without difficulty". Only a thin or unreadable record
    speaks, and it speaks about the document -- naming the fields that are
    missing rather than scoring them.
    """
    q = c["quality"]
    missing = q.get("missing_fields") or []
    warns = sum(f.get("severity", "warning") == "warning" for f in c["flags"])
    # Every record reports, including the clean ones: visible data quality is
    # one of this system's three standing claims, and a badge that appears
    # only on bad records turns a property of the pipeline into an exception.
    # The band, the score and the issue count all stay -- only the WORDING
    # changes, because "Data quality: high 0.94" never said whether it was
    # judging the document or the person.
    if missing:
        body = ("Resume quality: incomplete, no "
                + ", ".join(m.replace("_", " ") for m in missing[:3]))
        tail = f" · {q['score']}"
    else:
        body = f"Resume quality: {q['band'].title()}"
        tail = f" ({q['score']})"
    if warns:
        tail += f" · {warns} flag" + ("s" if warns != 1 else "")
    klass = "pill" if q["band"] == "high" and not missing else "pill pill-note"
    return (f"<span class='{klass}' data-tip='{METRIC_HELP['quality']}'>"
            f"{body}{tail}</span>")


# ---------------------------------------------------------------------------
# Distinctions
#
# What is worth printing next to a name is a property of the POOL, not of the
# person. The rows used to read "US · Fundamental · 6.3y investing · data
# high" -- four facts, three of them fixed by the role that produced the list
# and therefore identical on every row, and one of them a score about the
# document. Nothing there told two candidates apart, which is the only job a
# tag has.
#
# A tag is shown only when fewer than 40% of the pool carries it: in ten
# fundamental analysts "fundamental" distinguishes nobody, and in a pool of
# one quant "systematic" distinguishes everything. The threshold is the whole
# idea, so it is named once here rather than tuned per tag.
# ---------------------------------------------------------------------------

RARE_SHARE = 0.4


def _share(predicate) -> float:
    return sum(1 for x in candidates if predicate(x)) / max(len(candidates), 1)


def _dtint(tag: str) -> str:
    """Platform lineage is the strongest differentiator in this pool."""
    return "pill-navy" if tag.endswith(" alum") else "pill-slate"


def _dtip(tag: str, c: dict) -> str:
    if tag.endswith(" alum") and c.get("platform_alum_of"):
        return ("Resolved from the firm knowledge base — the resume names "
                "only the pod. Platform: "
                + ", ".join(c["platform_alum_of"]))
    return "Shown because fewer than 40% of this pool carry it."


def distinctions(c: dict) -> list[str]:
    """Up to three things that set this candidate apart from this pool."""
    out: list[str] = []
    if c.get("platform_alum_of"):
        out.append(f"{c['platform_alum_of'][0].split()[0]} alum")
    for cred in c.get("credentials_summary") or []:
        if cred.startswith("CFA"):
            out.append("CFA" if "Charterholder" in cred else "CFA candidate")
            break
    # No coverage-count tag. "75 names covered" next to a name reads as a
    # boast the candidate did not make -- and the number is already a
    # headline metric on the profile, where it has a definition attached.
    if c.get("approach_family") in _RARE_APPROACHES:
        out.append(label_of(c.get("approach_family")))
    if len(c.get("languages") or []) >= 3:
        out.append(f"{len(c['languages'])} languages")
    if (_BOTH_SIDES_RARE and c.get("has_buy_side_experience")
            and c.get("has_sell_side_experience")):
        out.append("Both sides")
    return out[:3]


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
    # Hairline grid, no axis lines: the bars are the drawing, the grid is a
    # reference the eye should have to look for.
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(0,0,0,0)",
                     ticks="", tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#f0f3f7", zeroline=False,
                     linecolor="rgba(0,0,0,0)", ticks="",
                     tickfont=dict(color=MUTED, size=11))
    return fig


def section(title: str, tip: str = "", container=None) -> None:
    """A section title, and its definition on hover rather than under it.

    Every zone gets one of these, and only one line: the titles were 11px
    uppercase grey while the explanatory caption beneath them was 12.5px --
    the explanation was literally larger than the thing it explained, which
    is how a screen ends up with no readable structure. The prose moves into
    the title's tooltip, where the reader can ask for it.
    """
    mark = "<i data-tip='{}'>?</i>".format(tip.replace("'", "’")) if tip else ""
    (container or st).markdown(
        f"<div class='sechead'>{title}{mark}</div>", unsafe_allow_html=True
    )


def coverage_quote(c: dict) -> str:
    """The resume sentence behind the stated coverage number, when one
    exists among the extracted stated metrics.

    Coverage is the one stated figure that is scored, so it should carry
    its evidence like every other claim. The schema keeps the number in
    `coverage`; the sentence usually arrives independently in
    `stated_metrics`. This joins them deterministically: a quote counts
    only if it contains the exact number and coverage vocabulary. No
    match, no quote -- never a reconstruction.
    """
    cov = c["extraction"].get("coverage", {}).get("stocks_covered")
    if not cov:
        return ""
    for m in c["extraction"].get("stated_metrics", []):
        q = m.get("quote", "")
        if re.search(rf"\b{cov}\b", q) and re.search(
                r"coverage|covering|stocks|names|companies", q, re.I):
            return q
    return ""


def _requirement_lines(reqs: list) -> str:
    """The posting's own requirement lines, each with how well it matched.

    This is the detail a recruiter actually argues with -- "it says sell-side
    relationships and this resume never mentions them" -- so it belongs
    directly under the dimension it produced, not in a separate list.
    """
    if not reqs:
        return ""
    out = []
    for r in sorted(reqs, key=lambda x: -x.score):
        label = (getattr(r, "label", "") or r.key.split("::")[-1]).strip()
        label = label[0].upper() + label[1:] if label else label
        out.append(
            f"<div class='reqline'><i style='width:{max(r.score, 0.02):.0%}'>"
            f"</i><span>{label}</span><b>{r.score:.0%}</b></div>"
        )
    return "<div class='reqlist'>" + "".join(out) + "</div>"


def candidate_table(rows: list[dict], matches: dict) -> str:
    """The shortlist as a typeset table rather than a data grid.

    st.dataframe gives a spreadsheet: uniform 13px sans in every cell, a
    scrollbar, resize handles, and no way to say which column matters. That
    is the right control for exploring numbers and the wrong one for reading
    a shortlist, which is a document -- someone scans it, screenshots it, or
    pastes it into a mail.
    typography carries the hierarchy instead: the name is the only semibold
    thing in the row, Fit is a tabular number over a hairline, the gap is the
    one bronze string on the screen, and record quality is a single dot. The
    CSV button below is the escape hatch for anyone who genuinely wants to
    sort and pivot.
    """
    head = (
        "<tr><th>Candidate</th><th>Sectors</th>"
        "<th class='num'>Investing</th><th class='num'>Fit</th>"
        "<th>Gap</th><th>Record</th></tr>"
    )
    body = []
    for c in rows:
        m = matches.get(c["candidate_id"])
        inv = c.get("years_investment_experience")
        sectors = ", ".join(label_of(x) for x in c.get("sectors", [])) or "—"
        fit = f"<b>{m.soft_score:.0%}</b>" if m else "<span class='dim'>—</span>"
        gap = (f"{m.failed_hard[0].label} — {m.failed_hard[0].found}"
               if m and m.failed_hard else "<span class='dim'>—</span>")
        band = c["quality"]["band"]
        body.append(
            f"<tr><td class='who'>{c['display_name']}"
            f"<span class='whofirm'>{c.get('current_firm') or '—'}</span></td>"
            f"<td class='sub'>{sectors}</td>"
            f"<td class='num sub'>{'—' if inv is None else f'{inv:g}y'}</td>"
            f"<td class='num'>{fit}</td>"
            f"<td class='sub'>{gap}</td>"
            f"<td class='sub'>{band}</td></tr>"
        )
    return (f"<div class='ctblwrap'><table class='ctbl'>{head}"
            f"{''.join(body)}</table></div>")


def contribution_ledger(result) -> str:
    """One row per scoring dimension: weight, score, and what earned it.

    This is the answer to "why 84%". The previous answer was a radar, plus a
    table of the same numbers, plus a collapsed list of quotes -- three
    renderings of one fact, none of which said what the candidate actually
    DID on that dimension. Each row now carries the pipeline's own finding
    for the dimension, which IS the contribution; the verbatim quote stays
    one fold below for anyone auditing it.

    Dimensions the role never specified are listed too, greyed and at zero.
    Omitting them is how a thin score passes for a complete one.
    """
    rows = []
    # soft_criteria carries two different KINDS of entry: the weighted
    # scoring dimensions, and one zero-weight entry per requirement line of
    # the posting -- the detail behind the requirement-match dimension. The
    # first version of this ledger treated both as dimensions, so five
    # truncated "Req::Build Relationships With Company Managem — not
    # measured" rows appeared, each claiming the role specified nothing to
    # score against while in fact quoting the very thing it specified.
    #
    # Requirement lines now render nested under the dimension they feed, and
    # "not measured" is reserved for a configured dimension the role gave
    # nothing to score against.
    _dims = set(store.weights)
    reqs = [x for x in result.soft_criteria
            if x.key not in _dims and x.key.lower().startswith("req")]
    active = [x for x in result.soft_criteria
              if x.weight > 0 and x.key in _dims]
    idle = [x for x in result.soft_criteria
            if x.weight <= 0 and x.key in _dims]
    for crit in sorted(active, key=lambda x: -x.weight):
        did = crit.found or "no signal found in this resume"
        rows.append(
            f"<div class='ldrow'><div class='ldtop'>"
            f"<span class='ldlbl' data-tip='{CRITERION_HELP.get(crit.key, '')}'>"
            f"{CRITERION_LABEL.get(crit.key, label_of(crit.key))}</span>"
            f"<span class='ldw'>{crit.weight:.0%} of Fit</span>"
            f"<span class='ldval'>{crit.score:.0%}</span></div>"
            f"<div class='ldbar'><i style='width:{crit.score:.0%}'></i></div>"
            f"<div class='lddid'>{did}</div>"
            + (_requirement_lines(reqs) if "requirement" in crit.key.lower()
               else "")
            + "</div>"
        )
    for crit in idle:
        rows.append(
            f"<div class='ldrow ldrow-idle'><div class='ldtop'>"
            f"<span class='ldlbl'>"
            f"{CRITERION_LABEL.get(crit.key, label_of(crit.key))}</span>"
            f"<span class='ldw'>not measured</span>"
            f"<span class='ldval'>—</span></div>"
            f"<div class='lddid'>this role specifies nothing to score "
            f"against</div></div>"
        )
    return "<div class='ledger'>" + "".join(rows) + "</div>"


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
    "Use the job library":
        "Pick a saved job posting. Candidates are matched against its "
        "requirements and ranked by fit.",
    "Define your own criteria":
        "Set market, approach, sector and experience yourself, and "
        "optionally add requirement lines for a richer ranking.",
    "Browse all candidates":
        "Explore the full candidate list with your own filters.",
}

# The role picker lives in a collapsed panel whose label states the current
# role. Keeping the picker on the same screen as the refinements is
# deliberate -- sourcing is iterative, and sending the user to another screen
# to widen an experience band turns a one-second adjustment into a round
# trip. But the first version left the picker permanently expanded, so six
# hundred pixels of chooser sat above the first candidate. Same screen,
# folded away until wanted.
tab_search, tab_insights, tab_quality, tab_method, tab_ask = st.tabs([
    ":material/group: Candidates",
    ":material/monitoring: Insights",
    ":material/fact_check: Data quality",
    ":material/menu_book: Method",
    ":material/forum: Ask · preview",
])

# The role picker renders inside the Candidates tab: Insights, Data quality
# and Method are views over the whole pool, and a "which seat" chooser on
# those pages implies -- wrongly -- that they answer to the selected role.
# The tabs are created first so the panel can render into the working view
# while the chosen requisition still exists before the sidebar builds.
with tab_search:
    # Picker first, then the FILLING bar under it -- the reviewer read the
    # bar-above-picker order as the summary floating loose above the thing
    # that produces it. The bar is still a placeholder container, filled in
    # once the chosen requisition exists.
    _role_panel = st.expander(
        "How do you want to filter candidates?",
        expanded=False, icon=":material/work:",
    )
    _role_bar = st.container()

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
        if mode == "Use the job library":
            titles = store.titles
            # Every stored requisition is transcribed from a real posting,
            # so the per-row provenance tag became noise repeated four
            # times; each row shows its office and REQ number instead, and
            # provenance is stated once below.
            chosen = st.selectbox(
                "Requisition",
                options=list(titles),
                format_func=lambda k: (
                    f"{titles[k]}"
                    + (f"   ·  {store.get(k).get('team')}"
                       if store.get(k).get("team") else "")
                ),
                label_visibility="collapsed",
            )
            st.caption("All transcribed from real postings — three "
                       "Millennium, one Point72.")
            requisition = store.get(chosen)

        elif mode == "Define your own criteria":
            st.caption("Each is a HARD requirement: outside it means "
                       "excluded, not down-ranked.")
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

    # The active seat and its hard requirements, stated permanently. Each
    # requirement is a chip rather than a run of bolded words in a sentence:
    # four chips can be counted at a glance, and the sidebar no longer needs
    # to repeat any of them as a disabled control.
    if requisition:
        hard = requisition.get("hard", {})
        bits = []
        if r := hard.get("regions"):
            bits.append(("Market", " / ".join(r)))
        if a := hard.get("approach_families"):
            bits.append(("Approach", " / ".join(label_of(x) for x in a)))
        if s := hard.get("sectors_any"):
            bits.append(("Sector", " / ".join(label_of(x) for x in s)))
        if y := hard.get("investment_years"):
            _span = (f"{y.get('min', 0)}+" if "max" not in y
                     else f"{y.get('min', 0)}–{y['max']}")
            bits.append(("Investing", f"{_span} yrs"))
        _role_bar.markdown(
            "<div class='rolebar'><span class='rolekick'>FILLING</span>"
            f"<span class='roletitle'>{requisition['title']}</span>"
            + (f"<span class='roleteam'>{requisition.get('team','')}</span>"
               if requisition.get("team") else "")
            + "<span class='hardchips'>"
            + "".join(f"<span class='hardchip'>{k} <b>{v}</b></span>"
                      for k, v in bits)
            + "</span>"
            + f"<span class='rolecount'>{len(bits)} hard requirement"
            + ("s" if len(bits) != 1 else "") + "</span></div>",
            unsafe_allow_html=True,
        )
    else:
        _role_bar.markdown(
            "<div class='rolebar rolebar-empty'>"
            "<span class='rolekick'>NO ROLE</span>"
            "<span class='roletitle'>Whole pool, unranked</span>"
            "<span class='rolecount'>Choose a seat to rank</span></div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Step 2 -- sidebar refinements
# ===========================================================================

st.sidebar.markdown("### Refine the pool")
st.sidebar.caption("Narrows the eligible pool. Never overrides the role.")


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
    ("asset_classes", "Asset class", True),
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
# Market side joins the visible three: once a role fixes market, approach
# and sector, those controls disappear entirely, and a sidebar showing only
# a keyword box reads as broken.
PRIMARY = {"region", "approach_family", "sectors", "market_side",
           "asset_classes"}


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


def facet_label(key: str, value: str) -> str:
    """Display form for a facet value.

    Three kinds of value flow through the facets and only one of them wants
    prettifying. Taxonomy tokens (`media_telecom`, `systematic_quant`) need
    label_of. Values that are already display strings -- regions ("APAC",
    which title-casing mangles to "Apac"), normalised tool names ("SQL") --
    pass through raw. Credentials are compacted to their code ("CFA",
    "Series 87 - Research Analyst"): the full registered name is a tooltip
    concern, and a dropdown row that truncates mid-word reads worse than a
    short label that fits.
    """
    if key == "credentials_summary":
        # "CFA - Chartered Financial Analyst (Charterholder)" renders as
        # "CFA · Charterholder": code plus status distinguishes the two CFA
        # entries in this pool, and the registered name lives in the
        # filter's tooltip rather than overflowing a dropdown row.
        code = value.split(" - ")[0]
        status = value.rsplit("(", 1)[1].rstrip(")") if "(" in value else ""
        return f"{code} · {status}" if status else code
    if key in {"region", "software_tools"}:
        return value
    return label_of(value)


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
        # Nothing is rendered. The role bar at the top of the screen already
        # states this dimension and its value, and a disabled copy of the
        # same fact costs a full sidebar row, reads as a broken control, and
        # invites a click that does nothing. Eligibility has one owner; the
        # sidebar simply stops speaking about it.
        return []
    return container.multiselect(
        label,
        options=options,
        format_func=lambda v, _c=counts: (
            f"{facet_label(key, v)}  ({_c.get(v, 0)})"),
        help=FILTER_HELP.get(key, ""),
        placeholder="Any",
        key=f"facet_{key}",
    )


for key, label, _ in FACETS:
    if key in PRIMARY:
        selections[key] = facet_control(key, label, st.sidebar)

query = st.sidebar.text_input(
    "Keyword", placeholder="e.g. digital health, backtesting",
    help=FILTER_HELP["keyword"],
)

# One fold, not two -- and it counts itself. A collapsed group that is
# silently shrinking the list is the "why are there only three people"
# support ticket, so the label carries the number of filters active inside
# it, read from the previous run's widget state.
_ADV = [
    ("facet_market_side", lambda v: bool(v)),
    ("facet_software_tools", lambda v: bool(v)),
    ("facet_credentials_summary", lambda v: bool(v)),
    ("adv_years", lambda v: bool(v) and (v[0] > 0 or v[1] < NO_LIMIT)),
    ("adv_unknown", lambda v: v is False),
    ("adv_alum", lambda v: v is True),
    ("adv_quality", lambda v: v not in (None, "low")),
    ("adv_prefer", lambda v: v is True),
]
_adv_n = sum(1 for k, test in _ADV
             if k in st.session_state and test(st.session_state[k]))
with st.sidebar.expander(
    "Advanced filters" + (f"  ·  {_adv_n} active" if _adv_n else ""),
    icon=":material/tune:", expanded=bool(_adv_n),
):
    st.caption("Dimensions the role does not decide.")
    for key, label, _ in FACETS:
        if key not in PRIMARY:
            selections[key] = facet_control(key, label, st)
    years_span = st.slider(
        "Years of investment experience",
        0.0, NO_LIMIT, (0.0, NO_LIMIT), step=0.5, help=FILTER_HELP["years"],
        key="adv_years",
    )
    # The "include uncomputable tenure" checkbox is gone from the UI: no
    # candidate in this pool triggers it, so it was a control that never
    # did anything -- pure reading cost. The handling it governed remains
    # in passes_filters (uncomputable tenure is always shown, never
    # silently dropped), and the control returns if the pool ever gains
    # such a record.
    include_unknown = True
    only_alum = st.checkbox(
        "Multi-manager platform alumni only", help=FILTER_HELP["alum"],
        key="adv_alum",
    )
    min_quality = st.select_slider(
        "Minimum parse confidence", options=["low", "medium", "high"],
        value="low", help=FILTER_HELP["quality"], key="adv_quality",
    )
    prefer_quality = st.checkbox(
        "Rank well-parsed records first", help=FILTER_HELP["prefer_quality"],
        key="adv_prefer",
    )


with st.sidebar.expander("Coming soon", icon=":material/schedule:"):
    # Roadmap made visible where it will live. The referred flag is ATS
    # metadata, not resume content -- shown disabled rather than invented,
    # because faking a data source would break the system's one rule.
    st.toggle("Referred candidates only", value=False, disabled=True,
              help="Referral status lives in the ATS, not the resume. "
                   "This filter activates with the ATS integration.")
    st.caption(
        "Also on the roadmap: neural RAG over the internal sourcing corpus "
        "(preview in the Ask tab) and a queryable knowledge graph "
        "(preview in Insights)."
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
    f"**{len(filtered)} of {len(candidates)}** pass these refinements."
)


# ===========================================================================
# Tabs
# ===========================================================================


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
            f"<tr><td data-tip='{CRITERION_HELP.get(crit.key, '')}'>"
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
                    f"**Thin score.** This role gives {active_n} of "
                    f"{total_n} signals something to measure — {share:.0%} "
                    "of the weighting, renormalised to 100%. Add sectors or "
                    "requirement lines to deepen it.",
                    icon="ℹ",
                )

        # One line when there are matches; a compact card when there are
        # none. The first version spent a headline metric and a full-width
        # warning box on this -- half a screen to say "zero". An empty
        # result is normal output, not an alarm, and the useful reaction to
        # it is specific: name the requirement most candidates miss and what
        # widening it would admit.
        # When anyone qualifies, nothing is printed here: the two group
        # headers below are "Qualified · 2" and "One gap away · 6", so a
        # status line saying "2 qualified · 6 one gap away" was the same two
        # numbers twice, a few dozen pixels apart -- and the headers are also
        # the controls that fold each group, so they are the copy that has to
        # stay. The empty case still speaks, because zero results with no
        # explanation reads as a broken filter.
        if not exact:
            from collections import Counter
            gap_counts = Counter(r.failed_hard[0].label for r in near)
            hint = ""
            if gap_counts:
                # Every widening option, not only the biggest: the decision
                # of WHICH requirement to relax belongs to the recruiter,
                # and showing one option pre-makes it.
                parts = [f"<b>{lbl.lower()}</b> would admit {n}"
                         for lbl, n in gap_counts.most_common()]
                hint = " Widening " + "; ".join(parts) + " of them."
            st.markdown(
                f"<div style='border-left:3px solid {SERIES_2};"
                f"background:{SURFACE};border:1px solid {GRID};"
                f"border-left:3px solid {SERIES_2};border-radius:10px;"
                f"padding:12px 16px;margin:2px 0 6px;font-size:13px'>"
                f"<b>0 of {len(filtered)} clear every requirement.</b> "
                f"Each candidate below misses exactly one, named on their "
                f"row.{hint}</div>",
                unsafe_allow_html=True,
            )
    else:
        results, matches = None, {}
        ordered = filtered
        section(
            "Candidate pool",
            f"{len(filtered)} of {len(candidates)} candidates after the "
            "sidebar refinements. Choose a role above to rank them.",
        )

    if prefer_quality:
        rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(ordered, key=lambda c: rank[c["quality"]["band"]])

    if not ordered:
        st.info("No candidates match. Relax a refinement in the sidebar.")
        st.stop()

    st.markdown("---")

    # The shortlist table spans the full page, above the list/profile
    # split: seven typeset columns in a 400px side column could only
    # cramp, wrap and push Fit past the edge -- a document-style table
    # needs document width.
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
                "Title": next(
                    (x["title"] for x in c["extraction"]["positions"]
                     if x.get("is_current")),
                    (c["extraction"]["positions"][0]["title"]
                     if c["extraction"]["positions"] else "—")),
                "Market side": label_of(c.get("market_side")),
                "Sector": ", ".join(label_of(x) for x in c.get("sectors", [])),
                "Firm": c.get("current_firm") or "—",
                "Firm type": (label_of(c.get("current_firm_type"))
                              if c.get("current_firm_type") else "—"),
                "Platform alum": ", ".join(c.get("platform_alum_of", []))
                                 or "—",
                "Asset classes": ", ".join(
                    label_of(a["value"])
                    for a in c["extraction"].get("asset_classes", [])) or "—",
                "Coverage": c["extraction"]["coverage"].get("stocks_covered"),
                "Stated numbers": len([
                    x for x in c["extraction"].get("stated_metrics", [])
                    if x["kind"] in ("aum", "performance", "risk")]),
                "Leadership": (c["extraction"].get("team_leadership") or
                               {}).get("value") or "—",
                "Software": ", ".join(c.get("software_tools", [])) or "—",
                "Credentials": ", ".join(
                    facet_label("credentials_summary", x)
                    for x in c.get("credentials_summary", [])) or "—",
                "Languages": ", ".join(c.get("languages", [])) or "—",
                "Data": c["quality"]["band"],
                "Flags": sum(f.get("severity", "warning") == "warning"
                              for f in c["flags"]),
            })
        frame = pd.DataFrame(rows)
        st.dataframe(
            frame.style.set_properties(
                subset=["Candidate"], **{"font-weight": "600"}
            ),
            hide_index=True, use_container_width=True,
            column_config={
                name: (st.column_config.NumberColumn(
                           help=COLUMN_HELP.get(name, ""), format="%.1f")
                       if name in ("Yrs investing", "Yrs career")
                       else st.column_config.Column(
                           help=COLUMN_HELP.get(name, "")))
                for name in frame.columns
            },
        )
        st.caption("Hover a column header for what it means; click a header "
                   "to sort. The CSV below carries the same columns.")
        st.download_button(
            "Download shortlist (CSV)",
            frame.to_csv(index=False).encode(),
            file_name="shortlist.csv", mime="text/csv",
            use_container_width=True,
        )

    list_col, detail_col = st.columns([1.15, 1.6], gap="large")

    # -- Result list: the NAME is the click target -------------------------
    with list_col:
        section("Results", "Ranked by Fit within each group.")

        # Active refinements, restated where the results are. Two of the
        # sidebar's filter groups collapse, so a selection made inside one
        # can be invisible while it silently shrinks the list -- the classic
        # "why are there only three people" support ticket. Anything that is
        # currently narrowing the pool gets a pill here.
        pills: list[str] = []
        for _k, _lbl, _ in FACETS:
            if selections.get(_k):
                pills.append(
                    f"{_lbl}: "
                    + ", ".join(facet_label(_k, v) for v in selections[_k])
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

            # Line two is identity: where they sit and how long they have
            # been investing. Region and approach are gone -- the role bar
            # fixed both, so printing them on every row was three columns of
            # the same word.
            meta = " · ".join(x for x in (
                c.get("current_firm"),
                label_of(c["current_firm_type"])
                if c.get("current_firm_type") else None,
                None if inv is None else f"{inv:g}y investing",
            ) if x)

            if m is None:
                return f"**{c['display_name']}**\n\n:gray[{meta}]"

            # Every row has the same three zones in the same order -- name
            # and score, metadata, verdict -- so two rows can be compared
            # without reading either in full.
            #
            # The score used to carry a bar drawn in block characters. It did
            # align down the column, and it looked like a progress meter from
            # a terminal: eleven glyphs of visual noise next to a name. A
            # monospace percentage aligns just as well, and the ordering of
            # the list is already the ranking -- the bar was re-stating in
            # ASCII what the sort order says for free.
            head = (f"**{c['display_name']}** &nbsp;&nbsp;"
                    f"`{m.soft_score:.0%}`")

            if m.is_exact:
                strong = sorted(
                    (x for x in m.soft_criteria if x.weight and x.score >= .5),
                    key=lambda x: x.score * x.weight, reverse=True,
                )[:2]
                # Line three is what sets them apart. Score components were
                # here before, restating the bar two inches to the left; a
                # pool-relative distinction is the thing the bar cannot say.
                why = " · ".join(distinctions(c)) or (" · ".join(
                    f"{CRITERION_LABEL.get(x.key, x.key)} {x.score:.0%}"
                    for x in strong
                ) or "clears every hard requirement")
                # Streamlit's :green[] and :orange[] are its own stock
                # accents -- brighter than anything else on the screen and
                # not overridable from our stylesheet. The section header
                # above already says which side of the line this row sits
                # on, so the reason line is set in body ink and the gap line
                # is labelled in words. Two saturated colours leave the
                # densest part of the interface.
                return f"{head}\n\n:gray[{meta}]\n\n{why}"

            fail = m.failed_hard[0]
            gap = (f"{fail.label} {fail.found}".replace("Investment ", "")
                   + f" — needs {fail.required}")
            gap = gap[0].upper() + gap[1:]
            return f"{head}\n\n:gray[{meta}]\n\n**Gap** — {gap}"

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
            # Both groups fold. A pool of fifty qualified candidates buries
            # the one-gap-away list below the fold, and a recruiter working
            # the near misses wants the opposite -- so either side can be put
            # away without leaving the screen.
            if exact_ids:
                with st.expander(f"Qualified · {len(exact_ids)}",
                                 expanded=True):
                    picked_a = st.radio(
                        "Qualifying candidates", exact_ids,
                        format_func=row_label, key="pick_exact",
                        index=0, label_visibility="collapsed",
                        on_change=_solo, args=("pick_exact", "pick_near"),
                    )
            if near_ids:
                with st.expander(f"One gap away · {len(near_ids)}",
                                 expanded=True):
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



    # -- Detail -----------------------------------------------------------
    # The profile panel is a fragment: clicking its view pills (Fit /
    # Profile / Issues / Outreach / Full record) reruns ONLY this panel,
    # not the whole script. Before this, every pill click re-executed all
    # four page tabs -- charts, tables, everything -- to change one panel,
    # which is exactly the half-second stutter the reviewer called "not
    # smooth". A full rerun (new candidate picked, filter changed) still
    # rebuilds the fragment with fresh arguments.
    @st.fragment
    def _detail_panel(chosen_id: str) -> None:
        with st.container(border=True):
            c = by_id[chosen_id]
            e = c["extraction"]
            m = matches.get(chosen_id)
            inv = c["years_investment_experience"]
            covered = e["coverage"].get("stocks_covered")

            st.markdown(f"<div class='candname'>{c['display_name']}</div>",
                        unsafe_allow_html=True)
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
                f"<div style='color:{MUTED};font-size:13px;margin:-6px 0 8px'>"
                f"{meta}</div>"
                + "".join(
                    f"<span class='pill {_dtint(t)}' data-tip='{_dtip(t, c)}'>{t}"
                    f"</span>" for t in distinctions(c)
                )
                + record_note(c),
                unsafe_allow_html=True,
            )
            # The platform-alum banner is gone. It said, in a bordered green box
            # directly beneath the tag that already said it, "Platform alum --
            # previously at Millennium Management": the same fact twice, the
            # second time louder. Once a tag states something, an explanation
            # under it is a restatement; the provenance that WAS worth keeping
            # (the resume names only the pod, the lineage comes from the firm
            # knowledge base) is now the tag's tooltip.

            # Headline numbers, Fit first when a role is set.
            cols = st.columns(4 if m else 3)
            offset = 0
            if m:
                # Rendered by hand rather than with st.metric so the colour can
                # carry the verdict: green when every hard requirement is met,
                # bronze when one is not. The other headline numbers stay navy --
                # they describe the candidate, this one describes the decision.
                verdict_colour = STATUS_GOOD if m.is_exact else SERIES_2
                verdict = (f"clears all {len(m.hard_criteria)} requirements"
                           if m.is_exact else "1 requirement missed")
                share, active_n, total_n = score_basis(m)
                cols[0].markdown(
                    f"<div data-tip='{METRIC_HELP['fit']}'>"
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
                "Stocks covered", covered or "—",
                help=METRIC_HELP["coverage"] + (
                    f' This resume: "{_cq}"' if (_cq := coverage_quote(c))
                    else ""),
            )

            # The profile was one 3,000px scroll: metrics, hard checks, a chart,
            # a table, evidence, profile, chips, issues, outreach and four more
            # expanders, in that order. Same content, one question at a time --
            # and the issue count moves into a tab label instead of needing a
            # jump link from the badge above.
            _warn_n = sum(f.get("severity", "warning") == "warning"
                          for f in c["flags"])
            # NOT st.tabs. Two reasons, one of them a bug.
            #
            # The bug: the masthead rule that lifts the page tabs into the navy
            # bar targets div[role="tablist"], and a second st.tabs anywhere on
            # the page is also a tablist -- so the profile's own tabs were being
            # fixed-positioned into the masthead and landing on top of Insights
            # and Data quality. Scoping the rule by ancestor is guesswork against
            # Streamlit's DOM; having exactly one tablist in the app is not.
            #
            # The design reason outlives the bug: page tabs and section switches
            # are different acts, and rendering both as the same tab strip claims
            # they are siblings. "Data quality" (the whole pool) and this
            # candidate's own issues are one subject at two scopes, which is
            # exactly the pair a reader will conflate if both look like page nav.
            # So this is a pill row, visibly not a tab strip, and every label is
            # scoped to the person.
            _fig_n = len([x for x in e.get("stated_metrics", [])
                          if x["kind"] in ("aum", "performance", "risk")])
            _views = ["Fit", "Profile", f"Figures ({_fig_n})",
                      f"Flags ({_warn_n})", "Outreach", "Full record"]
            _vkey = f"view_{chosen_id}"
            st.session_state.setdefault(_vkey, 0)

            def _setview(k: str, i: int) -> None:
                st.session_state[k] = i

            # A horizontal flex container, not st.columns. Weighted columns
            # sized each pill to a share of the panel, and any share narrower
            # than its label clipped the text ("Outreac|"). Here every pill
            # takes its natural width and the row wraps if the panel is
            # narrow -- nothing can be cut off.
            with st.container(horizontal=True, gap="small"):
                for _i, _name in enumerate(_views):
                    # on_click, not a return value plus st.rerun(). Reading the
                    # button's return meant the pills above it had already drawn
                    # with the previous selection, so the switch needed a second
                    # run to look right -- two full script executions per click,
                    # which is exactly the lag. A callback fires before the
                    # single rerun, so state and paint agree the first time.
                    st.button(
                        _name, key=f"{_vkey}_{_i}",
                        on_click=_setview, args=(_vkey, _i),
                        type=("primary" if st.session_state[_vkey] == _i
                              else "secondary"),
                    )
            _view = st.session_state[_vkey]

            if _view == 0:
                if m is None:
                    st.caption(
                        "Choose a seat above to score this candidate against a "
                        "role."
                    )
                if m:
                    section(
                        "Hard requirements",
                        "These disqualify. Outside any one of them is not a "
                        "lower rank, it is not a match.",
                    )
                    _hard = []
                    for crit in m.hard_criteria:
                        # Bronze, not red, on the miss -- a near miss is a state
                        # in the pipeline, not an error.
                        icon, colour = (("✓", STATUS_GOOD) if crit.passed
                                        else ("✗", SERIES_2))
                        _hard.append(
                            f"<div class='hardrow'><span style='color:{colour};"
                            f"font-weight:700'>{icon}</span> <b>{crit.label}</b> "
                            f"<span style='color:{MUTED}'>{crit.found} · needs "
                            f"{crit.required}</span></div>"
                        )
                    st.markdown(
                        "<div class='hardgrid'>" + "".join(_hard) + "</div>",
                        unsafe_allow_html=True,
                    )
                    section(
                        "Match components",
                        "Read the shape, not the area — a radar exaggerates "
                        "differences. Values are printed below.",
                    )
                    radar = fit_radar([m], 350)
                    if radar:
                        st.plotly_chart(radar, use_container_width=True)
                    section(
                        "What earned the Fit score",
                        "Each dimension scored 0-1, then weighted. The line "
                        "under each bar is what the pipeline found.",
                    )
                    st.markdown(contribution_ledger(m), unsafe_allow_html=True)
                    with st.expander("Resume sentences, per dimension"):
                        for crit in m.soft_criteria:
                            if crit.weight <= 0 or not crit.evidence:
                                continue
                            st.markdown(
                                f"<div style='font-size:12.5px;margin:0 0 9px'>"
                                f"<b>{crit.label}</b> — {crit.found}<br>"
                                f"<span style='color:{MUTED}'>"
                                f"“{crit.evidence[:260]}”</span></div>",
                                unsafe_allow_html=True,
                            )


            if _view == 1:
                section(
                    "Classified attributes",
                    "As classified, with the keywords that drove it and the "
                    "sentence it came from.",
                )

                def attribute(label: str, value: str, blocks=None,
                              note: str = "") -> None:
                    """One attribute as three ranked tiers, in ONE html block.

                    The tiers had no hierarchy for a structural reason, not a
                    typographic one: the value, the keyword pills and the quote
                    were three separate st.markdown calls, so Streamlit put its
                    own equal gap between them and they read as three siblings.
                    No amount of font sizing fixes that -- the grouping has to be
                    one element with its own spacing.

                    Ranked, loudest first: WHAT it decided (16px semibold, the
                    only large thing in the row), WHY it decided (keyword pills,
                    10.5px, plus the confidence), and the PROOF (the resume
                    sentence, 11.5px, clamped to two lines, full text on hover).
                    Everything sits inside a hairline left rule, so a reader sees
                    one object rather than three.
                    """
                    blocks = [b for b in (blocks or []) if b]
                    kws = list(dict.fromkeys(
                        k for b in blocks for k in (b.get("keywords") or [])))
                    confs = list(dict.fromkeys(
                        b["confidence"] for b in blocks if b.get("confidence")))
                    parts = [
                        f"<div class='attrlabel'>{label}</div>",
                        f"<div class='attrvalue'>{value}"
                        + (f"<span class='attrnote'>{note}</span>" if note else "")
                        + "</div>",
                    ]
                    if kws or confs:
                        why = "".join(f"<span class='kw'>{k}</span>" for k in kws)
                        if confs:
                            why += (f"<span class='conf'>{'/'.join(confs)} "
                                    f"confidence</span>")
                        parts.append(f"<div class='attrwhy'>{why}</div>")
                    for b in blocks:
                        ev = (b.get("evidence") or "").replace("'", "’")
                        if not ev:
                            continue
                        tag = (f"<b>{label_of(b.get('value'))}</b> — "
                               if len(blocks) > 1 else "")
                        parts.append(
                            f"<div data-tip='{ev}'><div class='attrquote'>{tag}“{ev}”</div>"
                            f"</div>"
                        )
                    st.markdown(f"<div class='attr'>{''.join(parts)}</div>",
                                unsafe_allow_html=True)

                def factline(label: str, value: str, note: str = "") -> None:
                    """A secondary fact: no keywords, no quote, no large type.

                    Not everything in this pane is a classification. Markets
                    covered and the seniority bands are computed, not read, so
                    giving them the same three-tier block as the classified
                    attributes flattened the pane back out -- eight identical
                    units, none of them the point.
                    """
                    st.markdown(
                        f"<div class='fact'><span>{label}</span><b>{value}</b>"
                        + (f"<i>{note}</i>" if note else "") + "</div>",
                        unsafe_allow_html=True,
                    )

                # The note repeated the value whenever the classifier and its
                # family agree, which is most of the time: "Fundamental ·
                # Fundamental".
                _fam = label_of(c.get("approach_family"))
                _appr = label_of(e["investment_approach"]["value"])
                attribute("Investment approach", _appr,
                          [e["investment_approach"]],
                          note="" if _fam == _appr else f"family: {_fam}")
                attribute("Market side", label_of(e["market_side"]["value"]),
                          [e["market_side"]])
                # One "Sectors" row. Numbering it 1 of 3 was worse than
                # repeating the label: it implied a rank the pipeline never
                # assigned. The sectors are one value, their keywords merge into
                # one pill row, and each sector's own sentence is kept below,
                # named -- so nothing is lost and the label appears once.
                _secs = e["primary_sectors"]
                if _secs:
                    attribute(
                        "Sectors",
                        ", ".join(label_of(x["value"]) for x in _secs),
                        _secs,
                    )
                if e.get("asset_classes"):
                    attribute(
                        "Asset classes",
                        ", ".join(label_of(x["value"])
                                  for x in e["asset_classes"]),
                        e["asset_classes"],
                    )
                # Same theme as every other attribute -- the one factline
                # here read as a different design system. The provenance
                # rides as a tag, like any other "why".
                attribute(
                    "Markets covered",
                    ", ".join(c.get("coverage_markets", [])) or "—",
                    [{"keywords": [
                        "inferred from location"
                        if c.get("coverage_markets_source") == "inferred"
                        else "stated in resume"], "confidence": None}],
                )
                # Seniority is only worth a row when the two bands DISAGREE --
                # a senior career with a junior investing record is a finding.
                # When they agree it restates the two tenure metrics at the top
                # of the pane in different words.
                _sb, _isb = c.get("seniority_band"), c.get("investment_seniority_band")
                if _sb and _isb and _sb != _isb:
                    factline(
                        "Seniority",
                        f"{label_of(_sb)} by career, {label_of(_isb)} by investing",
                        note="the gap is the point",
                    )

                _lead = e.get("team_leadership") or {}
                if _lead.get("value"):
                    attribute("Team leadership", _lead["value"], [_lead])

                section("Skills and credentials", "As stated in the document.")

                def chips(label: str, values: list[str], klass: str = "") -> None:
                    # Colour is back on the tags, but drawn from the palette
                    # rather than invented per row. Every row -- credentials
                    # included -- uses this one shape, and an empty row is a
                    # dashed "none listed" TAG rather than a grey sentence:
                    # at a glance the absence reads in the same visual
                    # register as the presence, but cannot be mistaken for a
                    # held skill.
                    st.markdown(
                        f"<div style='margin-top:9px;font-size:12px;"
                        f"color:{MUTED}'>{label}</div>" + (
                            "".join(f"<span class='pill {klass}'>{v}</span>"
                                    for v in values) if values else
                            "<span class='pill pill-empty'>none listed</span>"
                        ),
                        unsafe_allow_html=True,
                    )

                chips("Professional credentials",
                      c.get("credentials_summary", []), "pill-bronze")
                chips("Software and platforms", c.get("software_tools", []),
                      "pill-slate")
                chips("Analytical methods", c.get("methods", []), "pill-bronze")
                chips("Languages", c.get("languages", []), "pill-navy")



            if _view == 2:
                # Stated figures get their own view: summary first, then the
                # breakdown, in the same attr blocks as every classified
                # attribute -- inside the profile they were a fourth design
                # dialect and made a long pane longer. Structured by the
                # model (a pattern cannot tell a $4.2bn book from a $500
                # conference budget), displayed with verbatim quotes, and
                # never scored: self-reported numbers are unverifiable by
                # design.
                section("Stated figures",
                        "Performance, AUM and risk figures the resume "
                        "itself states. Self-reported and unverified — "
                        "shown with their quotes, never scored.")
                _KIND = {"aum": "AUM / book", "performance": "Performance",
                         "risk": "Risk"}
                _mx = [x for x in e.get("stated_metrics", [])
                       if x["kind"] in _KIND]
                # Coverage joins the summary row because it IS a stated
                # figure -- but it is the one stated figure that is SCORED
                # (coverage_depth, 8% weight), and the card says so. The
                # dividing line is comparability, not importance: "names
                # under coverage" has one standardised meaning on every
                # research resume, while AUM and returns arrive gross-or-
                # net, book-or-firm, cherry-picked -- this very pool has a
                # "$4.5 trillion" that is Fidelity's AUM, not the
                # candidate's. Score what is comparable; display what is
                # not, with its quote.
                _cov = covered
                if not _mx and not _cov:
                    st.caption("This resume states no coverage, performance, "
                               "AUM or risk figures.")
                else:
                    _sum_cols = st.columns(4)
                    with _sum_cols[0]:
                        st.markdown(
                            "<div class='attrlabel'>Names covered</div>"
                            + ((lambda _q: (
                                   (f"<div data-tip='{_q}'>" if _q else "<div>")
                                   + f"<div class='figval'>{_cov}</div></div>"
                                   "<div class='attrwhy'><span class='kw'>"
                                   "scored · 8% weight</span></div>"))(
                                       coverage_quote(c).replace("'", "’"))
                               if _cov else
                               "<span class='pill pill-empty'>"
                               "none stated</span>"),
                            unsafe_allow_html=True,
                        )
                    for _col, _kind in zip(_sum_cols[1:],
                                           ("aum", "performance", "risk")):
                        _hits = [x for x in _mx if x["kind"] == _kind]
                        # A stated figure can be a whole clause ("35%
                        # increase in brokerage commission revenue"); the
                        # summary card shows the head of it, the breakdown
                        # below keeps the full text and quote.
                        _fig = _hits[0]["figure"] if _hits else ""
                        if len(_fig) > 26:
                            _fig = _fig[:24].rstrip() + "…"
                        with _col:
                            st.markdown(
                                f"<div class='attrlabel'>{_KIND[_kind]}</div>"
                                + (f"<div class='figval'>{_fig}</div>"
                                   f"<div class='attrwhy'><span class='kw'>"
                                   f"{len(_hits)} stated</span></div>"
                                   if _hits else
                                   "<span class='pill pill-empty'>"
                                   "none stated</span>"),
                                unsafe_allow_html=True,
                            )
                    st.markdown("")
                    # Breakdown: every figure with its verbatim quote, in
                    # the standard attr block per kind. Same contract as
                    # every profile quote: clamped to two lines on screen,
                    # full text on hover.
                    def _qdiv(_x: dict) -> str:
                        _t = _x["quote"].replace("'", "’")
                        return (f"<div data-tip='{_t}'>"
                                f"<div class='attrquote'>"
                                f"<b>{_x['figure']}</b> — "
                                f"“{_x['quote'][:200]}”</div></div>")

                    # Coverage first: it is the one stated figure that is
                    # scored, so its resume sentence sits in the breakdown
                    # beside the others, not only behind a hover.
                    _covq = coverage_quote(c)
                    if _cov and _covq:
                        st.markdown(
                            f"<div class='attr'>"
                            f"<div class='attrlabel'>Names covered · "
                            f"breakdown</div>"
                            + _qdiv({"figure": f"{_cov} names",
                                     "quote": _covq})
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    for _kind in ("aum", "performance", "risk"):
                        _hits = [x for x in _mx if x["kind"] == _kind]
                        if not _hits:
                            continue
                        _quotes = "".join(_qdiv(_x) for _x in _hits)
                        st.markdown(
                            f"<div class='attr'>"
                            f"<div class='attrlabel'>{_KIND[_kind]} · "
                            f"breakdown</div>{_quotes}</div>",
                            unsafe_allow_html=True,
                        )

            if _view == 3:
                section("Review flags", METRIC_HELP["issues"])
                # Issues sit AFTER the profile: a reviewer wants to know who this
                # person is before being told what is uncertain about the record. The
                # data-quality badge at the top links down here, so the caveats are
                # one click away rather than in the reader's path.
                # Issues, grouped the way a recruiter reads them (the grouping came
                # from one): possible misrepresentation first -- the only category
                # that can kill a candidacy -- then timeline, contact, and the
                # detail errors that signal carelessness rather than dishonesty.
                # Triaged-benign flags render last as grey notes: visible, priced
                # at zero.
                _warns = [f for f in c["flags"]
                          if f.get("severity", "warning") == "warning"]
                _notes = [f for f in c["flags"] if f.get("severity") == "info"]
                if not _warns and not _notes:
                    st.caption("Nothing flagged — the document parsed cleanly.")
                else:
                    # Explanation rides the section title's hover tip --
                    # inline it rendered larger than the issues themselves.


                    def _group_of(f) -> str:
                        text = f["summary"].lower()
                        if "email" in text or "phone" in text:
                            return "Contact details"
                        if f["category"] in {"internal_contradiction",
                                             "attribution_ambiguity"}:
                            return "Possible misrepresentation"
                        if f["category"] == "date_anomaly":
                            return "Timeline"
                        return "Detail errors"

                    _order = ["Possible misrepresentation", "Timeline",
                              "Contact details", "Detail errors"]
                    for group in _order:
                        members = [f for f in _warns if _group_of(f) == group]
                        if not members:
                            continue
                        st.markdown(
                            f"<div style='font-size:10.5px;font-weight:700;"
                            f"text-transform:uppercase;letter-spacing:.07em;"
                            f"color:{MUTED};margin:8px 0 5px'>{group}</div>",
                            unsafe_allow_html=True,
                        )
                        for f in members:
                            colour = SERIES_1 if f["source"] == "computed" else SERIES_2
                            origin = ("computed from the data"
                                      if f["source"] == "computed"
                                      else "read from the text")
                            st.markdown(
                                f"<div style='font-size:12.5px;margin-bottom:7px;"
                                f"border-left:3px solid {colour};padding-left:9px'>"
                                f"<b>{f['summary']}</b> <span style='color:{MUTED};"
                                f"font-size:11px'>· {origin}</span><br>"
                                f"<span style='color:{MUTED}'>{f['detail']}</span></div>",
                                unsafe_allow_html=True,
                            )
                    if _notes:
                        with st.expander(f"Notes — reviewed, no action ({len(_notes)})"):
                            for f in _notes:
                                st.markdown(
                                    f"<div style='font-size:12px;margin-bottom:6px;"
                                    f"color:{MUTED};border-left:3px solid {GRID};"
                                    f"padding-left:9px'><b>{f['summary']}</b><br>"
                                    f"{f['detail']}</div>",
                                    unsafe_allow_html=True,
                                )


            if _view == 4:
                # -- Outreach draft ------------------------------------------------
                # The step after "this candidate fits" is always "someone writes to
                # them". The draft is assembled from the same evidence the profile
                # shows -- specific and checkable, never generated flattery: a
                # sourcing mail earns a reply by proving someone actually read the
                # resume, and every claim here carries its quote.
                with st.container():
                    _cur = next((x for x in e["positions"]
                                 if x.get("is_current")), e["positions"][0]
                                if e["positions"] else None)
                    _lines = [f"{c['display_name']}"]
                    if _cur:
                        _lines.append(f"{_cur['title']} at {_cur['firm']}"
                                      + (f" · {c['location']}" if c.get("location")
                                         else ""))
                    _contact = " · ".join(x for x in (e.get("email"), e.get("phone"))
                                          if x)
                    if _contact:
                        _lines.append(_contact)
                    _lines.append("")
                    _strong = []
                    verdict_txt = ""
                    if m and requisition:
                        verdict_txt = ("meets every hard requirement"
                                       if m.is_exact else
                                       f"one gap: {m.failed_hard[0].label.lower()} "
                                       f"({m.failed_hard[0].found}; role needs "
                                       f"{m.failed_hard[0].required})")
                        _lines.append(f"Role: {requisition['title']} — "
                                      f"Fit {m.soft_score:.0%}, {verdict_txt}")
                        _lines.append("")
                        _lines.append("Why this candidate:")
                        _strong = sorted(
                            (x for x in m.soft_criteria
                             if x.weight and x.score >= .5 and x.evidence),
                            key=lambda x: x.score * x.weight, reverse=True)[:3]
                        for x in _strong:
                            _lines.append(
                                f"- {CRITERION_LABEL.get(x.key, x.key)}: {x.found}")
                            _lines.append(f'    resume: "{x.evidence[:180]}"')
                    if c.get("platform_alum_of"):
                        _lines.append(
                            f"- Platform alum: previously at "
                            f"{', '.join(c['platform_alum_of'])} (resume names the "
                            "pod; lineage resolved via knowledge base)")
                    if covered:
                        _lines.append(f"- {covered} names under research coverage")
                    # Rendered as a briefing card in the app's own design
                    # language -- a raw <pre> block of the draft read as a
                    # terminal dump inside a typeset interface. The plain-text
                    # version survives underneath for copy and download; the
                    # card is for reading, the code block is for taking.
                    _h = [f"<div class='odraft'>"]
                    _h.append(f"<div class='odraft-name'>{c['display_name']}"
                              + (f"<span> — {_cur['title']} at {_cur['firm']}"
                                 + (f" · {c['location']}" if c.get('location')
                                    else "") + "</span>" if _cur else "")
                              + "</div>")
                    if _contact:
                        _h.append(f"<div class='odraft-contact'>{_contact}</div>")
                    if m and requisition:
                        _fitcolour = STATUS_GOOD if m.is_exact else SERIES_2
                        _h.append(
                            "<div class='attrlabel' style='margin-top:12px'>"
                            "Role</div>"
                            f"<div class='odraft-role'>{requisition['title']}"
                            f"<span style='color:{_fitcolour};font-weight:700'>"
                            f" · Fit {m.soft_score:.0%}</span>"
                            f"<span class='odraft-verdict'>{verdict_txt}</span>"
                            "</div>")
                        if _strong:
                            _h.append("<div class='attrlabel' "
                                      "style='margin-top:12px'>"
                                      "Why this candidate</div>")
                        for x in _strong:
                            _h.append(
                                f"<div class='odraft-why'><b>"
                                f"{CRITERION_LABEL.get(x.key, x.key)}</b> — "
                                f"{x.found}"
                                f"<div class='attrquote'>“{x.evidence[:180]}”"
                                f"</div></div>")
                    _extras = []
                    if c.get("platform_alum_of"):
                        _extras.append(
                            f"<span class='pill pill-slate'>previously at "
                            f"{', '.join(c['platform_alum_of'])}</span>")
                    if covered:
                        _extras.append(f"<span class='pill'>{covered} names "
                                       "under coverage</span>")
                    if _extras:
                        _h.append("<div style='margin-top:10px'>"
                                  + "".join(_extras) + "</div>")
                    _h.append("</div>")
                    st.markdown("".join(_h), unsafe_allow_html=True)
                    st.caption("Parsed facts and verbatim quotes only — check "
                               "the gap line before sending.")

                    _draft = "\n".join(_lines)
                    with st.expander("Copy as plain text",
                                     icon=":material/content_copy:"):
                        st.code(_draft, language=None)
                        st.download_button(
                            "Download draft (.txt)", _draft.encode(),
                            file_name=f"outreach_{c['candidate_id']}.txt",
                            mime="text/plain", use_container_width=True,
                            key=f"outreach_{c['candidate_id']}",
                        )


            if _view == 5:
                # The original document, when it is present. The public
                # deployment deliberately ships no resume files (they are
                # the case study's materials, and a real pipeline's
                # originals carry PII that belongs in a governed document
                # store, not a public repo) -- so this button appears on
                # internal / local runs and degrades to a note in public.
                _raw = Path("data/resumes") / c["source_file"]
                if _raw.exists():
                    st.download_button(
                        "Download original resume",
                        _raw.read_bytes(),
                        file_name=c["source_file"],
                        key=f"dl_{chosen_id}",
                        icon=":material/download:",
                    )
                else:
                    st.caption(
                        "Original document not bundled with this public "
                        "deployment — in production, originals live in a "
                        "governed document store and download here."
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
                        # The header pill counts WARNINGS; this row counted every
                        # flag including the triaged-benign notes, so one profile
                        # reported "2 issues" at the top and "5 issue(s)" here.
                        # Same split as everywhere else: warnings, then notes.
                        ("Parse confidence",
                         f"{c['quality']['band']} {c['quality']['score']} · "
                         f"{sum(f.get('severity', 'warning') == 'warning' for f in c['flags'])}"
                         f" warning(s), "
                         f"{sum(f.get('severity') == 'info' for f in c['flags'])} note(s)"),
                        ("Missing fields", ", ".join(c["quality"]["missing_fields"]) or "none"),
                        ("Source file", c["source_file"]),
                    ]
                    st.markdown(
                        "<table class='recordtbl'>"
                        + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields)
                        + "</table>",
                        unsafe_allow_html=True,
                    )

    with detail_col:
        _detail_panel(chosen_id)

# ===========================================================================
# Insights
# ===========================================================================

with tab_insights:
    st.markdown("#### Talent pool insights")
    st.caption(
        "How the full candidate pool is distributed: where coverage "
        "is deep, where it is thin, and which open roles are hardest to "
        "fill."
    )

    # -- Requisition coverage: the concrete question first ---------------
    # Its own bordered card, run as a fragment: the posting picker reruns
    # only this block, not the whole script, and the card border separates
    # it from the pool charts below, which answer a different question.
    @st.cache_data(show_spinner=False)
    def _req_counts() -> list[tuple[str, str, str, int, int]]:
        out = []
        for _sp in store.items:
            _ex, _nr = match_all(candidates, _sp, store=store)
            out.append((_sp["id"], _sp["title"], _sp.get("source", ""),
                        len(_ex), len(_nr)))
        return out

    @st.fragment
    def _req_coverage() -> None:
        with st.container(border=True):
            section("Requisition coverage",
                    "For each saved job posting: how many candidates in "
                    "the pool meet every hard requirement (qualified), "
                    "and how many miss exactly one (near match). Counts "
                    "are for the full pool.")
            _rows = _req_counts()
            _req_pick = st.multiselect(
                "Requisitions to show",
                options=[r[0] for r in _rows],
                default=[r[0] for r in _rows],
                format_func=lambda rid: next(
                    r[1] for r in _rows if r[0] == rid),
                label_visibility="collapsed",
            )
            _shown = [r for r in _rows if r[0] in _req_pick]
            if not _shown:
                st.caption("Pick at least one requisition to see its "
                           "coverage.")
            _PER_ROW = 4
            _req_cols = []
            for _i in range(0, len(_shown), _PER_ROW):
                _req_cols.extend(st.columns(_PER_ROW))
            _unfilled = []
            for _col, (_rid, _title, _src, _nq, _nn) in zip(_req_cols,
                                                            _shown):
                _col.metric(
                    _title if len(_title) <= 34 else _title[:32] + "…",
                    f"{_nq} qualified",
                    delta=f"{_nn} near match" + ("es" if _nn != 1 else ""),
                    delta_color="off",
                    help=f"{_src}. Qualified candidates meet every hard "
                         "requirement; near matches miss exactly one.",
                )
                if not _nq:
                    _unfilled.append((_title, _nn))
            if _unfilled:
                _parts = "; ".join(
                    f"<b>{t}</b> ({n} near match"
                    + ("es" if n != 1 else "") + ")"
                    for t, n in _unfilled)
                st.markdown(
                    f"<span class='pill pill-note'>hardest to fill</span> "
                    f"<span style='font-size:12.5px;color:{MUTED}'>"
                    f"No candidate in the pool meets every requirement "
                    f"for {_parts}. Open a role in Candidates to see what "
                    "relaxing each requirement would admit.</span>",
                    unsafe_allow_html=True,
                )

    _req_coverage()

    section("Pool distribution",
            "How the pool spreads across regions, sectors, tenure and "
            "credentials. The whole pool is charted, filtered only by "
            "the quality threshold below.")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.6])
    chart_quality = ctrl1.select_slider(
        "Minimum resume quality",
        options=["low", "medium", "high"], value="low",
        help="Exclude candidates whose resumes parsed below this quality "
             "level, so thin records do not inflate the counts.",
    )
    group_by = ctrl2.selectbox(
        "Group rows by", ["Region", "Market side", "Seniority (investing)"],
        help="The row dimension of the coverage matrix.",
    )
    charted = [c for c in candidates
               if BAND_ORDER[c["quality"]["band"]] >= BAND_ORDER[chart_quality]]
    ctrl3.markdown(
        f"<div style='padding-top:26px;color:{MUTED};font-size:12.5px'>"
        f"Showing <b style='color:{NAVY}'>{len(charted)}</b> of "
        f"{len(candidates)} candidates (quality threshold applied).</div>",
        unsafe_allow_html=True,
    )
    if not charted:
        st.info("No candidates left at this threshold.")
    else:
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
            _col_tot = [sum(1 for c in charted if s in c.get("sectors", []))
                        for s in sectors]
            _row_tot = [sum(1 for c in charted if GROUP_KEY(c) == g)
                        for g in groups]
            fig = go.Figure(go.Heatmap(
                z=matrix,
                x=[f"{label_of(s)} ({t})" for s, t in zip(sectors, _col_tot)],
                y=[f"{g} ({t})" for g, t in zip(groups, _row_tot)],
                colorscale=[[0.0, SURFACE], [0.001, SEQ[0]]] + [
                    [0.001 + 0.999 * i / (len(SEQ) - 1), col]
                    for i, col in enumerate(SEQ)],
                zmin=0, showscale=False, xgap=2, ygap=2,
                text=[[v or "" for v in row] for row in matrix],
                texttemplate="%{text}", textfont=dict(size=13),
                hovertemplate="%{y} · %{x}<br>%{z} candidate(s)<extra></extra>",
            ))
            section(f"Sector coverage by {group_by.lower()}",
                    "Each cell counts the candidates in a group who "
                    "cover a sector; totals sit in the axis labels. "
                    "Empty cells are coverage gaps.")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            section("Career vs. investing tenure",
                    "One row per candidate: total career years next to "
                    "years in investment roles. A wide gap means a "
                    "career changer from banking, consulting or "
                    "engineering.")
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
            section("Software and credentials",
                    "How many candidates hold each tool and "
                    "qualification. A role naming a tool nobody holds "
                    "is a sourcing problem, not a screening one.")
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
                colours = ["#123a6f"] * len(top_creds) + ["#8badd0"] * len(top_tools)
                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h",
                    marker=dict(color=colours, cornerradius=9),
                    width=0.52,
                    hovertemplate="%{y}<br>%{x} candidate(s)<extra></extra>"))
                fig.update_layout(showlegend=False, xaxis_title="Candidates")
                fig.update_yaxes(showgrid=False, autorange="reversed")
                fig.update_xaxes(showgrid=True, gridcolor=GRID, dtick=1)
                st.plotly_chart(styled_chart(fig, 360), use_container_width=True)
                st.caption("Dark = credentials · light = software.")


    # -- Talent network (preview) ------------------------------------------
    # The knowledge base is already a small graph (candidate -> firm ->
    # parent platform); drawing it makes the pod-lineage story visible: the
    # reader can SEE that Ryan reaches Millennium through North53. At this
    # scale a static three-layer layout is honest; the roadmap's knowledge
    # graph makes it queryable ("everyone two hops from a Millennium pod").
    section("Talent network", "Candidates, their resolved firms, and "
            "platform lineage from the knowledge base.")
    st.markdown("<span class='pill pill-note'>preview</span>",
                unsafe_allow_html=True)

    _firms, _platforms, _edges = {}, {}, []
    for _c in candidates:
        for _l in _c.get("firms", []):
            if _l.get("canonical") and _l.get("resolution") in {
                    "exact", "alias", "fuzzy"}:
                _firms.setdefault(_l["canonical"], _l)
                _edges.append((_c["display_name"], _l["canonical"], "worked"))
    for _name, _l in _firms.items():
        if _l.get("parent"):
            _platforms.setdefault(_l["parent"], True)
            _edges.append((_name, _l["parent"], "pod"))
        elif _l.get("firm_type") == "multi_strategy_platform":
            _platforms.setdefault(_name, True)

    def _row(names, y):
        n = max(len(names), 1)
        return {name: ((i + 0.5) / n, y) for i, name in enumerate(names)}

    _pos = {}
    _pos.update(_row(sorted(_platforms), 1.0))
    _pos.update(_row(sorted(f for f in _firms if f not in _pos), 0.55))
    _pos.update(_row(sorted(c["display_name"] for c in candidates), 0.0))

    _fig = go.Figure()
    for _a, _b, _kind in _edges:
        if _a in _pos and _b in _pos:
            _fig.add_trace(go.Scatter(
                x=[_pos[_a][0], _pos[_b][0]], y=[_pos[_a][1], _pos[_b][1]],
                mode="lines",
                line=dict(color=SERIES_2 if _kind == "pod" else GRID,
                          width=2 if _kind == "pod" else 1),
                hoverinfo="skip", showlegend=False))
    # Thirty firm labels on one row are unreadable ink; firms show on
    # hover, and only the two rows a reader navigates BY -- platforms and
    # candidates -- carry printed names.
    for _names, _y, _colour, _label, _mode in (
            (sorted(_platforms), 1.0, NAVY, "platform", "markers+text"),
            (sorted(f for f in _firms
                    if f not in _platforms), 0.55, SERIES_1, "firm",
             "markers"),
            (sorted(c["display_name"] for c in candidates), 0.0,
             "#8badd0", "candidate", "markers+text")):
        _fig.add_trace(go.Scatter(
            x=[_pos[n][0] for n in _names], y=[_y] * len(_names),
            mode=_mode, text=_names, name=_label,
            textposition="top center" if _y > 0.5 else "bottom center",
            textfont=dict(size=10),
            marker=dict(size=11, color=_colour), hoverinfo="text",
            hovertext=_names))
    _fig.update_layout(
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.25, 1.2]),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(size=11)),
    )
    st.plotly_chart(styled_chart(_fig, 430), use_container_width=True)
    st.caption(
        "Bronze edges are ownership and platform lineage from the firm "
        "knowledge base — links no resume states. Grey edges are "
        "employment, resolved through the same knowledge base."
    )


# ===========================================================================
# Data quality
# ===========================================================================

with tab_quality:
    st.markdown("#### How far to trust each record")
    st.caption(
        "A property of the DOCUMENT, never of the candidate: a resume built "
        "from tables scores lower because our data is thinner."
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
    # Rounded caps, no outline, one hue in three lightnesses, and a bar
    # thinner than its own gap. An outlined full-height bar in three
    # saturated colours is a spreadsheet chart; the shape carries as much of
    # the register as the palette does.
    fig = go.Figure(go.Bar(
        x=[c["quality"]["score"] for c in ranked],
        y=[c["display_name"] for c in ranked], orientation="h",
        marker=dict(
            color=[QUALITY_COLOUR[c["quality"]["band"]] for c in ranked],
            cornerradius=9,
        ),
        width=0.52,
        text=[f"{c['quality']['score']:.2f}" for c in ranked],
        textposition="outside", textfont=dict(size=11, color=MUTED),
        customdata=[len(c["flags"]) for c in ranked],
        hovertemplate="%{y}<br>confidence %{x:.2f}<br>%{customdata} flag(s)"
                      "<extra></extra>"))
    fig.update_layout(showlegend=False, xaxis_title="Parse confidence (0–1)")
    fig.update_xaxes(range=[0, 1.12], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(styled_chart(fig, 380), use_container_width=True)
    st.caption(
        "Darker is better parsed — high ≥ 0.80, medium ≥ 0.55, low below. "
        "Scores are deductions from a "
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
            f"{len(c['flags'])} flag(s)", expanded=(band != "high"),
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
One resume keeps its candidate name, degree and a section heading in
floating text boxes — elements that *look* like images in Word but are
still text in the file, and are recovered as text here.

**Genuine embedded images are not interpreted.** No resume in this corpus
carries content as pixels; if one did, the extraction diagnostics would
flag it — a low character count against the page count — and the
escalation path is to send the page image to a vision model rather than
to guess. That step is deliberately out of scope until a document needs
it: OCR on documents that don't adds cost and a new class of errors.
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
        f"| {CRITERION_LABEL.get(k, k)}"
        + (" \\*" if k == "coverage_depth" else "")
        + f" | {v:.0%} | {CRITERION_HELP.get(k, '')} |"
        for k, v in sorted(store.weights.items(), key=lambda kv: -kv[1])
    )
    st.markdown(
        "| Component | Weight | What it measures |\n|---|---|---|\n" + weights_rows
    )
    st.markdown(
        """
`Fit = Σ(component × weight) / Σ(weight)`
        """
    )

    st.markdown("###### \\* The coverage benchmark is per posting")
    _bench_rows = "\n".join(
        f"| {sp['title']} | "
        + (f"{sp.get('soft', {}).get('coverage_benchmark', 40)} names |"
           if sp.get("soft", {}).get("coverage_benchmark", 40)
           else "none — signal dropped, weights renormalise |")
        for sp in store.items
    )
    st.markdown(
        """
The right number is strategy-dependent: a sector-focused long/short
book typically runs 15–40 names, a fundamental generalist 20–60,
long-only far more, and coverage counts mean little for a quant seat.
Each posting therefore sets its own benchmark in `requisitions.yaml`,
next to the criteria it serves:

| Posting | Coverage benchmark |
|---|---|
"""
        + _bench_rows
        + """

The three healthcare seats use 40, the top of the sector-focused range;
the quant seat sets none, so the signal is dropped there, the remaining
weights renormalise, and the ledger shows the row as *not measured*.
The weights themselves are still global; making them per-posting is the
same one-line change in the same file.
        """
    )

    st.markdown("###### Requirement similarity")
    st.markdown(
        """
A pluggable backend. The default combines lexical overlap with a
curated concept map, because this domain runs on paraphrase — a posting
says "catalysts" where a resume says "earnings events".

| Parameter | Value |
|---|---|
| Combination | `min(1, 0.45 × lexical + 0.75 × conceptual)` |
| Lexical | share of the requirement's content words present in the sentence |
| Conceptual | share of the requirement's concepts present in the sentence |
| Short-sentence damping | sentences under 6 content words scaled by `0.55 + 0.075 × n` |
| Concept matching | word-boundary, not substring |
| One gap away | fails exactly one hard requirement; two or more are not listed |

A worked example, one requirement against one resume sentence:

| | |
|---|---|
| Requirement | "fundamental research on healthcare catalysts" |
| Resume sentence | "conducted fundamental analysis of biotech companies around earnings events" |
| Lexical | 1 of the requirement's 4 content words appears ("fundamental") = 0.25 |
| Conceptual | the requirement's one mapped concept, *catalyst*, is shared, because "earnings events" is listed under it = 1.00 |
| Combined | min(1, 0.45 × 0.25 + 0.75 × 1.00) = **0.86** |

The concept layer is why these two texts score at all: they share almost
no words, but they describe the same work. Lexical overlap alone would
call them strangers.

Two design notes, each answering a question the parameter table raises:

**Why word-boundary matching, not substring?** An earlier version matched
substrings, and the concept map lists `r` for the R language. The letter
r appears in nearly every English sentence, so that one entry fired on
everything: every requirement scored about 0.75 on its concept term
alone, and the evidence quotes became arbitrary. Word boundaries fixed
it, and the shipped code matches tokens everywhere.

**Why a curated map now, and embeddings at scale?** The map works well
at a few hundred candidate sentences because it is fast, lightweight,
and auditable: a bad match is fixed by editing a line of YAML, not by
retraining a model. Its weakness is paraphrase diversity: at larger
scale, new ways of expressing the same skill appear faster than the map
can be curated, and recall gradually degrades. When that happens, the
`SimilarityBackend` swaps to sentence embeddings plus a vector index,
and the rest of the system stays unchanged: hard eligibility rules,
ranking weights, and evidence display all remain intact. The likely
production design is hybrid retrieval, embeddings for semantic recall
with lexical and concept matching kept for precision and explainability,
and any new backend must pass the same labeled evaluation set before
shipping.
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

    # -- 7 · accuracy -----------------------------------------------------
    # The numbers come from data/evaluation.json (built by src/evaluate.py);
    # the interpretation below is written against the current ground truth
    # and should be revisited if the labels change.
    _eval_path = DATA.parent / "evaluation.json"
    if _eval_path.exists():
        ev = json.loads(_eval_path.read_text(encoding="utf-8"))
        o = ev["overall"]
        st.markdown("##### 7 · Accuracy against human judgment")
        st.markdown(
            f"""
The system's shortlists were evaluated against **blind human labels**: a
reviewer saw each candidate's facts but never the system's verdict, and
answered "would you shortlist this person for this seat?" for all
**{ev["labelled_pairs"]} candidate-role pairs** (Y / N / borderline;
borderline excluded from the counts rather than forced into a bucket).
The system's shortlist is its exact matches — near misses deliberately
do not count.

| Role | Precision | Recall | Agreement | TP·FP·FN·TN |
|---|---|---|---|---|
"""
            + "\n".join(
                f"| {r['role']} | {_fmt_pct(r['precision'])} "
                f"| {_fmt_pct(r['recall'])} | {_fmt_pct(r['agreement'])} "
                f"| {r['tp']}·{r['fp']}·{r['fn']}·{r['tn']} |"
                for r in ev["per_role"]
            )
            + f"\n| **Overall** | **{_fmt_pct(o['precision'])}** "
              f"| **{_fmt_pct(o['recall'])}** | **{_fmt_pct(o['agreement'])}** "
              f"| {o['tp']}·{o['fp']}·{o['fn']}·{o['tn']} |"
        )
        if ev["disagreements"]:
            st.markdown(
                "**Every disagreement, individually:**\n\n" + "\n".join(
                    f"- *{d['candidate']}* × {d['role']} — human shortlists, "
                    f"system excludes ({d['system_reason']})"
                    for d in ev["disagreements"]
                )
            )
        st.markdown(
            """
**Human review and rule validation.** We validate the matching logic
against human shortlists, with particular attention to cases where the
system and reviewer disagree.

In our review set, the system achieved **100% precision**: every
candidate it shortlisted was also shortlisted by the reviewer. The three
disagreements all came from the same rule: a 4–5 year experience band
for a Mumbai Healthcare Research Analyst role. The system treated the
posted range as a hard constraint; the reviewer was willing to consider
candidates with 9.8–12.7 years of experience.

The key insight was that this flexibility was **supply-driven**. With a
small candidate pool, the reviewer was willing to stretch the experience
band; with a deeper pool, she would not. In contrast, geography remained
a hard constraint regardless of supply.

We therefore do not encode this flexibility as a hidden rule. The system
preserves the requirements as posted and surfaces near-matches
separately, showing the specific gap so recruiters can make the
trade-off explicitly.

With n=40, we do not treat the percentages as statistically
representative. The purpose of the exercise is **rule validation**:
identifying where structured requirements diverge from real recruiting
judgment and understanding when those deviations are driven by candidate
supply.
            """
        )

    st.caption(
        "Built for the Millennium Business Development data science case "
        "study · github.com/yc4379-commits/m-case-study"
    )


# ===========================================================================
# Ask -- retrieval preview
# ===========================================================================

with tab_ask:
    section(
        "Ask the pool",
        "Search across structured candidate data using natural language.",
    )
    st.markdown(
        "<span class='pill pill-note'>preview</span> "
        "<span style='font-size:12.5px;color:" + MUTED + "'>"
        "Every answer is a quoted sentence from a resume, with its source. "
        "Coming next: connecting candidate data with internal sourcing "
        "knowledge — meeting notes and call summaries — under the same "
        "rule.</span>",
        unsafe_allow_html=True,
    )
    _q = st.text_input(
        "Ask", placeholder="e.g. who has run money in healthcare?",
        label_visibility="collapsed", key="ask_query",
    )
    if _q and len(_q.strip()) >= 3:
        from match import ConceptTfidfBackend, candidate_sentences
        _backend = ConceptTfidfBackend(store.concept_map)
        _hits = []
        for _c in candidates:
            _score, _sent = _backend.score(_q, candidate_sentences(_c))
            if _score > 0.05 and _sent:
                _hits.append((_score, _c, _sent))
        _hits.sort(key=lambda x: -x[0])
        if not _hits:
            st.caption(
                "Nothing in the parsed resumes matches that phrasing. The "
                "concept map covers this domain's vocabulary — an unmatched "
                "query usually means the pool genuinely lacks it."
            )
        for _score, _c, _sent in _hits[:5]:
            st.markdown(
                f"<div class='odraft' style='margin-bottom:8px'>"
                f"<div class='odraft-name'>{_c['display_name']}"
                f"<span> — {_c.get('current_firm') or '—'} · "
                f"match {_score:.0%}</span></div>"
                f"<div class='attrquote' style='margin-top:6px'>“{_sent}”"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        if _hits:
            st.caption(
                "Scores are sentence-level retrieval strength, not Fit — "
                "open the candidate in Candidates for the scored view."
            )
