"""
Domain knowledge base.

Loads the curated YAML in `knowledge/` and exposes lookups that turn raw
resume strings into normalised entities.

The central design decision here is **how firm names are matched**, and it is
worth stating plainly because the obvious approach is wrong.

This corpus contains three unrelated firms whose names begin with the same
word: Meridian Capital Partners (a Greenwich hedge fund), Meridian Capital (an
Indian research house) and Meridian Research Partners (an Indian sell-side
shop). Substring or "contains" matching -- the default reflex -- collapses all
three into whichever entry happens to be checked first, silently relocating a
candidate to the wrong continent and the wrong side of the market.

So matching is exact-first, and fuzzy matching only accepts a result that is
both close AND unambiguous: if two entries score similarly, we return
"ambiguous" rather than pick one. An honest unresolved value costs a filter
option; a confident wrong value costs trust in the whole platform.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

# Legal-form suffixes carry no identity and appear inconsistently.
_SUFFIXES = re.compile(
    r"\b(ltd|limited|llc|llp|lp|inc|incorporated|corp|corporation|co|plc|"
    r"pvt|private|group|holdings|company|management|&\s*co)\b\.?",
    re.IGNORECASE,
)

# Fuzzy acceptance thresholds. A match must clear MIN_RATIO and beat the
# runner-up by MIN_MARGIN, otherwise the name is reported ambiguous.
MIN_RATIO = 0.88
MIN_MARGIN = 0.06


def _fold_accents(text: str) -> str:
    """Strip diacritics so 'Société Générale' matches 'Societe Generale'.

    Without this, decomposing to ASCII by regex turns every accented letter
    into a space, shredding the name into fragments that match nothing.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _normalise(name: str) -> str:
    """Lowercase, fold accents, strip punctuation and legal suffixes."""
    text = _fold_accents(name).lower().replace("&", " and ")
    text = re.sub(r"[.,'\"()\-–—/]", " ", text)
    text = _SUFFIXES.sub(" ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class FirmMatch:
    """Result of resolving a firm string against the knowledge base."""

    raw: str
    canonical: Optional[str] = None
    firm_type: Optional[str] = None
    parent: Optional[str] = None
    region: Optional[str] = None
    method: str = "unresolved"      # exact | alias | fuzzy | ambiguous | unresolved
    note: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.canonical is not None

    @property
    def is_buy_side(self) -> bool:
        return self.firm_type in KnowledgeBase.load().buy_side_types

    @property
    def is_sell_side(self) -> bool:
        return self.firm_type in KnowledgeBase.load().sell_side_types


class KnowledgeBase:
    """Curated domain facts, loaded once and reused."""

    def __init__(self, firms: dict[str, Any], taxonomy: dict[str, Any]) -> None:
        self.firms: dict[str, dict[str, Any]] = firms.get("firms", {})
        self.buy_side_types: list[str] = firms.get("buy_side_types", [])
        self.sell_side_types: list[str] = firms.get("sell_side_types", [])
        self.taxonomy = taxonomy

        # name (normalised) -> canonical, covering canonical names and aliases
        self._index: dict[str, str] = {}
        for canonical, meta in self.firms.items():
            self._index[_normalise(canonical)] = canonical
            for alias in meta.get("aliases") or []:
                self._index.setdefault(_normalise(alias), canonical)

        self._region_index: dict[str, str] = {}
        for region, spec in taxonomy.get("regions", {}).items():
            for term in list(spec.get("countries", [])) + list(spec.get("cities", [])):
                self._region_index[term.lower()] = region

        self._sector_index: dict[str, str] = {}
        for sector, aliases in taxonomy.get("sector_aliases", {}).items():
            for alias in aliases:
                self._sector_index[alias.lower()] = sector

    # -- loading ----------------------------------------------------------

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, directory: str | Path = KNOWLEDGE_DIR) -> "KnowledgeBase":
        directory = Path(directory)
        firms = yaml.safe_load((directory / "firms.yaml").read_text(encoding="utf-8"))
        taxonomy = yaml.safe_load(
            (directory / "taxonomy.yaml").read_text(encoding="utf-8")
        )
        return cls(firms, taxonomy)

    # -- firms ------------------------------------------------------------

    def resolve_firm(self, raw: str) -> FirmMatch:
        """Resolve a firm name to a knowledge-base entry.

        Order: exact -> normalised exact/alias -> guarded fuzzy. Anything that
        does not clear the guards is returned unresolved or ambiguous, never
        guessed.
        """
        if not raw or not raw.strip():
            return FirmMatch(raw=raw)

        raw = raw.strip()

        if raw in self.firms:
            return self._build(raw, raw, "exact")

        key = _normalise(raw)
        if key in self._index:
            canonical = self._index[key]
            method = "exact" if _normalise(canonical) == key else "alias"
            return self._build(raw, canonical, method)

        # Resumes routinely append a division to the employer:
        # "William Blair, Equity Research", "Goldman Sachs, Investment Banking".
        # The division is not part of the firm's identity, so retry on the
        # portion before the comma before giving up.
        if "," in raw:
            head = raw.split(",", 1)[0].strip()
            if head and len(head) > 3:
                inner = self.resolve_firm(head)
                if inner.resolved:
                    return FirmMatch(
                        raw=raw,
                        canonical=inner.canonical,
                        firm_type=inner.firm_type,
                        parent=inner.parent,
                        region=inner.region,
                        method="division",
                        note=inner.note,
                    )

        # Before fuzzy matching: if the name is a token-subset of several
        # known firms, it is genuinely ambiguous rather than merely unknown.
        # "Meridian" is a prefix of three unrelated firms in this corpus; the
        # useful answer is to say so, not to return nothing.
        tokens = set(key.split())
        if tokens:
            prefixes = sorted(
                {
                    self._index[known]
                    for known in self._index
                    if tokens and tokens < set(known.split())
                }
            )
            if len(prefixes) > 1:
                return FirmMatch(
                    raw=raw,
                    method="ambiguous",
                    note=f"Ambiguous between: {', '.join(prefixes)}",
                )

        # Guarded fuzzy match against every known spelling.
        scored = sorted(
            (
                (difflib.SequenceMatcher(None, key, known).ratio(), known)
                for known in self._index
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < MIN_RATIO:
            return FirmMatch(raw=raw, method="unresolved")

        best_ratio, best_key = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_ratio - runner_up < MIN_MARGIN:
            # Near-tied candidates. They are only genuinely ambiguous if they
            # point at DIFFERENT firms -- the Meridian case. Several aliases
            # of one firm scoring alike is not ambiguity, and an earlier
            # version failed here: "J.P. Morgan Chase & Co." tied across four
            # JPMorgan aliases and was reported unresolvable, losing the
            # employer entirely.
            competitors = sorted(
                {self._index[k] for r, k in scored[:4] if best_ratio - r < MIN_MARGIN}
            )
            if len(competitors) > 1:
                return FirmMatch(
                    raw=raw,
                    method="ambiguous",
                    note=f"Ambiguous between: {', '.join(competitors)}",
                )

        return self._build(raw, self._index[best_key], "fuzzy")

    def _build(self, raw: str, canonical: str, method: str) -> FirmMatch:
        meta = self.firms.get(canonical, {})
        return FirmMatch(
            raw=raw,
            canonical=canonical,
            firm_type=meta.get("type"),
            parent=meta.get("parent"),
            region=meta.get("region"),
            method=method,
            note=meta.get("note"),
        )

    def platform_lineage(self, canonical: str) -> list[str]:
        """Walk parent links, e.g. North53 Capital -> Millennium Management."""
        lineage, seen = [], set()
        current = canonical
        while current and current not in seen:
            seen.add(current)
            parent = self.firms.get(current, {}).get("parent")
            if parent:
                lineage.append(parent)
            current = parent
        return lineage

    # -- geography --------------------------------------------------------

    def region_for(self, location: Optional[str]) -> Optional[str]:
        """Map a free-text location to US / Europe / APAC / MEA / LatAm.

        Longer terms are tested first so 'Hong Kong' is not matched by a
        shorter unrelated entry, and matching is word-boundary aware so
        'US' does not fire inside 'Sao Paulo'.
        """
        if not location:
            return None
        text = location.lower()
        for term in sorted(self._region_index, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text):
                return self._region_index[term]
        return None

    # -- sectors ----------------------------------------------------------

    def normalise_sector(self, raw: str) -> Optional[str]:
        """Map sector vocabulary onto the controlled enum."""
        if not raw:
            return None
        text = raw.lower().strip()
        if text in self._sector_index:
            return self._sector_index[text]
        for alias in sorted(self._sector_index, key=len, reverse=True):
            if alias in text:
                return self._sector_index[alias]
        return None

    def sectors_in_text(self, text: str) -> list[str]:
        """All sectors mentioned anywhere in a block of text, most specific first."""
        found, low = [], text.lower()
        for alias in sorted(self._sector_index, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", low):
                sector = self._sector_index[alias]
                if sector not in found:
                    found.append(sector)
        return found

    # -- seniority --------------------------------------------------------

    def seniority_for_years(self, years: Optional[float]) -> Optional[str]:
        """Seniority band from computed tenure -- the primary signal."""
        if years is None:
            return None
        for spec in self.taxonomy["seniority_bands"]:
            if years <= spec["max_years"]:
                return spec["band"]
        return self.taxonomy["seniority_bands"][-1]["band"]

    def title_seniority_hint(self, title: str) -> Optional[str]:
        """Seniority implied by a job title, or None.

        Used only to cross-check the tenure-derived band. Titles are not
        comparable across market sides, so a disagreement is reported to a
        human rather than allowed to override the computed value.
        """
        if not title:
            return None
        low = title.lower()
        # Most senior first: "Senior Analyst" must not be caught by "analyst i".
        for band in ("portfolio_manager", "senior", "junior", "intern"):
            for pattern in self.taxonomy["title_hints"][band]:
                if pattern.lower() in low:
                    return band
        return None

    def approach_family(self, approach: Optional[str]) -> Optional[str]:
        """Coarse family for an approach value: the axis a requisition uses."""
        if not approach:
            return None
        for family, spec in self.taxonomy.get("approach_families", {}).items():
            if approach in spec.get("members", []):
                return family
        return "unclear"

    def approach_family_label(self, family: Optional[str]) -> Optional[str]:
        if not family:
            return None
        spec = self.taxonomy.get("approach_families", {}).get(family, {})
        return spec.get("label", family)

    def normalise_software(self, raw: str) -> str:
        """Map a tool name onto its canonical spelling.

        Longest alias first, so "Excel with VBA" resolves to Excel rather than
        matching the shorter "vba" entry and losing the spreadsheet.
        """
        if not raw:
            return raw
        low = raw.strip().lower()
        aliases = self.taxonomy.get("software_aliases", {})
        best_canonical, best_len = None, 0
        for canonical, spellings in aliases.items():
            for spelling in spellings:
                spelling = spelling.lower()
                if (low == spelling or spelling in low) and len(spelling) > best_len:
                    best_canonical, best_len = canonical, len(spelling)
        return best_canonical or raw.strip()

    def expand_credential(self, name: str) -> Optional[str]:
        """Full registered name for a credential code, e.g. Series 7."""
        glossary = self.taxonomy.get("credential_glossary", {})
        cleaned = re.sub(r"\s+", " ", name).strip()
        if cleaned in glossary:
            return glossary[cleaned]
        for code, full in glossary.items():
            if code.lower() == cleaned.lower():
                return full
        # "Series 7 (active)" -> "Series 7"
        if m := re.match(r"(Series\s*\d+)", cleaned, re.I):
            key = re.sub(r"\s+", " ", m.group(1)).title().replace("Series", "Series")
            for code, full in glossary.items():
                if code.lower() == key.lower():
                    return full
        return None

    @property
    def junior_years_range(self) -> tuple[int, int]:
        lo, hi = self.taxonomy.get("junior_years_range", [1, 5])
        return int(lo), int(hi)


# --------------------------------------------------------------------------
# Tenure arithmetic
#
# Deliberately computed in Python rather than asked of the model. Date maths
# is where language models fail quietly, and the inputs are already extracted.
# --------------------------------------------------------------------------

def _to_months(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """Convert 'YYYY-MM' or 'YYYY' to absolute months. Year-only is mid-year."""
    if not value:
        return default
    match = re.match(r"(\d{4})(?:-(\d{1,2}))?", str(value).strip())
    if not match:
        return default
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else 6
    if not 1900 <= year <= 2100 or not 1 <= month <= 12:
        return default
    return year * 12 + month


COUNTED_EMPLOYMENT_TYPES = frozenset({"professional"})


def years_of_experience(
    positions: list[dict[str, Any]],
    *,
    include_internships: bool = False,
    investment_only: bool = False,
    as_of: Optional[date] = None,
) -> Optional[float]:
    """Years of experience, merging overlapping roles.

    Two decisions here came from looking at the parsed data rather than from
    theory.

    Overlaps are merged, not summed. Several resumes list concurrent roles,
    and naive summation inflated one five-year career to eight.

    Only `professional` roles count. One resume lists a university society
    presidency dating back to 2014 and a co-founded side venture still marked
    current; counting those added seven fictitious years to a candidate whose
    real career began in 2019. Student organisations, volunteering, academic
    assistantships and side ventures are excluded by default.

    With `investment_only`, the result narrows further to roles the model
    marked as investment work. This is the number a research requisition
    actually cares about: total tenure and investment tenure differ by years
    for candidates who came in from banking, consulting or operations.
    """
    today = as_of or date.today()
    now = today.year * 12 + today.month

    intervals: list[tuple[int, int]] = []
    duration_only_months = 0
    for pos in positions:
        employment = pos.get("employment_type", "professional")
        if employment not in COUNTED_EMPLOYMENT_TYPES:
            if not (include_internships and employment == "internship"):
                continue
        if investment_only and not pos.get("is_investment_role"):
            continue
        start = _to_months(pos.get("start_date"))
        if start is None:
            # Some resumes state durations instead of dates ("8 years 10
            # months"). Those roles cannot be placed on a timeline, so they
            # cannot be overlap-merged; we sum them separately and add the
            # total. This over-counts if such roles overlapped, which is why
            # it is a fallback rather than the primary path -- and why the
            # candidate is flagged as having undated positions.
            duration_only_months += pos.get("duration_months") or 0
            continue
        end = now if pos.get("is_current") else _to_months(pos.get("end_date"), now)
        end = min(end or now, now)
        if end > start:
            intervals.append((start, end))

    if not intervals:
        if duration_only_months:
            return round(duration_only_months / 12, 1)
        # Nothing matched the filter. Two very different situations share this
        # branch, and collapsing them to None loses the more useful one:
        #
        #   - No role qualifies, but the history IS datable. A quantitative
        #     developer with four dated years and no investment role has
        #     exactly 0.0 years of investment experience. That is a fact.
        #   - Nothing is datable at all. Then we genuinely do not know.
        #
        # Returning None for both made "definitely none" and "unknown" render
        # identically as a dash, which is the opposite of informative.
        datable = any(
            _to_months(p.get("start_date")) is not None or p.get("duration_months")
            for p in positions
        )
        return 0.0 if datable else None

    intervals.sort()
    merged: list[list[int]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total = sum(end - start for start, end in merged) + duration_only_months
    return round(total / 12, 1)
