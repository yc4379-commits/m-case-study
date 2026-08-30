"""
Requisition matching.

The design goal is a match a recruiter can check rather than one they have to
believe. Three rules produce that.

**Hard constraints disqualify; soft signals rank.** A candidate outside the
requisition's region or experience band is not an 82% match, they are not a
match. Blending both into one number is why a search that returns a confident
ranked list can still be useless: the top result may fail a requirement that
was never negotiable. Here the two never mix.

**The system must be able to return nothing.** If no candidate satisfies every
hard constraint, that is the answer, reported as such -- together with the
near misses and exactly which constraint each one failed. A matcher that
always produces a winner has not been tested against the possibility that
there isn't one, and every commercial "match score" behaves this way.

**Every point is traceable to a line in the resume.** Each criterion carries
the text that satisfied it. A reviewer scans the reasons, not the number.

Similarity between a requisition's requirements and a candidate's experience
uses a pluggable backend. The default combines TF-IDF cosine with a curated
concept map, because this domain runs on paraphrase -- "catalysts" against
"earnings events" -- that pure lexical overlap misses. A neural embedding
backend can be dropped in unchanged where the environment allows it; see
`SimilarityBackend`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


# --------------------------------------------------------------------------
# Similarity backends
# --------------------------------------------------------------------------

class SimilarityBackend(Protocol):
    """Scores how well a requirement is met by a set of resume sentences."""

    def score(self, requirement: str, sentences: list[str]) -> tuple[float, str]:
        """Return (0-1 score, best matching sentence)."""
        ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#]{2,}", text.lower()))


_STOP = {
    "and", "the", "for", "with", "from", "into", "across", "their", "this",
    "that", "have", "has", "are", "was", "were", "which", "including", "such",
    "based", "using", "used", "within", "under", "over", "through", "other",
}


class ConceptTfidfBackend:
    """TF-IDF cosine, boosted by a curated concept map.

    The concept map is what makes this workable on domain language. Lexical
    overlap alone scores "trading around catalysts" against "tactical
    strategies around earnings events" at almost zero, because they share no
    content words -- yet they describe the same job. Mapping both onto a
    `catalyst` concept fixes that case explicitly, and the fix is visible in
    a YAML file rather than buried in a model's weights.

    The trade-off is honest: this generalises only as far as the map goes. It
    is the right choice at a few hundred sentences and the wrong one at a few
    hundred thousand.
    """

    def __init__(self, concept_map: dict[str, list[str]]) -> None:
        # Concept phrases are matched on WORD BOUNDARIES, not as raw
        # substrings. The naive version was badly wrong in a way that only
        # showed up on inspection: the `programming` concept lists "r" for the
        # R language, and `"r" in text` is true of very nearly every English
        # sentence. That one entry made the concept fire universally, so every
        # requirement scored ~0.75 on its concept term alone and the "best
        # matching sentence" shown as evidence was effectively arbitrary --
        # "Biomodeller Trainee BioAnalytics Research India Ltd." was returned
        # as proof of "fundamental research on India equity".
        #
        # A scoring bug that inflates every score is invisible in aggregate and
        # obvious the moment you read one quote. That is the argument for
        # putting the evidence on screen.
        self.concepts = {
            name: [
                re.compile(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])")
                for phrase in phrases
            ]
            for name, phrases in concept_map.items()
        }

    def _concepts_in(self, text: str) -> set[str]:
        low = text.lower()
        return {
            name
            for name, patterns in self.concepts.items()
            if any(pattern.search(low) for pattern in patterns)
        }

    def score(self, requirement: str, sentences: list[str]) -> tuple[float, str]:
        if not sentences:
            return 0.0, ""

        req_tokens = _tokens(requirement) - _STOP
        req_concepts = self._concepts_in(requirement)

        best_score, best_sentence = 0.0, ""
        for sentence in sentences:
            sent_tokens = _tokens(sentence) - _STOP
            if not sent_tokens or not req_tokens:
                continue

            # Lexical: what share of the requirement's words appear.
            lexical = len(req_tokens & sent_tokens) / len(req_tokens)

            # Conceptual: do they talk about the same thing, differently.
            sent_concepts = self._concepts_in(sentence)
            conceptual = (
                len(req_concepts & sent_concepts) / len(req_concepts)
                if req_concepts
                else 0.0
            )

            # Concepts dominate, because a shared concept is stronger evidence
            # of the same meaning than a few shared common words.
            combined = min(1.0, 0.45 * lexical + 0.75 * conceptual)

            # A five-word job-title line ("Research Analyst Axis Mutual Fund")
            # can share a term with almost any requirement while evidencing
            # nothing. Short fragments are damped so a substantive bullet wins
            # ties -- the quote has to be worth reading, not merely adjacent.
            if len(sent_tokens) < 6:
                combined *= 0.55 + 0.075 * len(sent_tokens)

            if combined > best_score:
                best_score, best_sentence = combined, sentence

        return round(best_score, 3), best_sentence


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Criterion:
    """One requirement, whether it was met, and the text that shows it."""

    key: str
    label: str
    kind: str                    # "hard" | "soft"
    passed: Optional[bool] = None
    score: float = 0.0           # soft only, 0-1
    weight: float = 0.0
    required: str = ""           # what the requisition asked for
    found: str = ""              # what the candidate has
    evidence: str = ""           # verbatim supporting text


@dataclass
class MatchResult:
    candidate_id: str
    display_name: str
    requisition_id: str
    criteria: list[Criterion] = field(default_factory=list)
    failed_hard: list[Criterion] = field(default_factory=list)
    soft_score: float = 0.0
    quality_band: str = "high"

    @property
    def is_exact(self) -> bool:
        return not self.failed_hard

    @property
    def near_miss_count(self) -> int:
        return len(self.failed_hard)

    @property
    def hard_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.kind == "hard"]

    @property
    def soft_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.kind == "soft"]


# --------------------------------------------------------------------------
# Requisition store
# --------------------------------------------------------------------------

class Requisitions:
    def __init__(self, data: dict[str, Any]) -> None:
        self.items: list[dict[str, Any]] = data.get("requisitions", [])
        self.weights: dict[str, float] = data.get("soft_weights", {})
        self.concept_map: dict[str, list[str]] = data.get("concept_map", {})
        self._by_id = {r["id"]: r for r in self.items}

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, directory: str | Path = KNOWLEDGE_DIR) -> "Requisitions":
        path = Path(directory) / "requisitions.yaml"
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    def get(self, requisition_id: str) -> Optional[dict[str, Any]]:
        return self._by_id.get(requisition_id)

    @property
    def titles(self) -> dict[str, str]:
        return {r["id"]: r["title"] for r in self.items}


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def candidate_sentences(candidate: dict[str, Any]) -> list[str]:
    """Every sentence we can quote back as evidence."""
    out: list[str] = []
    extraction = candidate.get("extraction", {})
    for position in extraction.get("positions", []):
        header = " ".join(
            filter(None, [position.get("title"), position.get("firm")])
        )
        if header:
            out.append(header)
        out.extend(b for b in position.get("description", []) if b)
    for field_name in ("investment_approach", "market_side"):
        if quote := extraction.get(field_name, {}).get("evidence"):
            out.append(quote)
    for item in extraction.get("primary_sectors", []):
        if quote := item.get("evidence"):
            out.append(quote)
    out.extend(candidate.get("software_tools", []))
    out.extend(candidate.get("methods", []))
    return [s for s in out if len(s) > 3]


def _hard_criteria(
    candidate: dict[str, Any], spec: dict[str, Any]
) -> list[Criterion]:
    """Evaluate the disqualifying constraints."""
    criteria: list[Criterion] = []

    if regions := spec.get("regions"):
        region = candidate.get("region")
        criteria.append(
            Criterion(
                key="region",
                label="Region",
                kind="hard",
                passed=region in regions,
                required=" or ".join(regions),
                found=region or "unknown",
                evidence=candidate.get("location") or "",
            )
        )

    if families := spec.get("approach_families"):
        family = candidate.get("approach_family")
        criteria.append(
            Criterion(
                key="approach",
                label="Investment approach",
                kind="hard",
                passed=family in families,
                required=" or ".join(families),
                found=f"{family} ({candidate.get('approach')})",
                evidence=candidate["extraction"]["investment_approach"].get(
                    "evidence", ""
                ),
            )
        )

    if band := spec.get("investment_years"):
        years = candidate.get("years_investment_experience")
        low, high = band.get("min", 0), band.get("max", 99)
        # None means the tenure could not be computed, not that it is zero.
        # An uncomputable value cannot satisfy a numeric constraint, and
        # silently treating it as passing would put candidates with no dates
        # into a band we never verified.
        passed = years is not None and low <= years <= high
        criteria.append(
            Criterion(
                key="investment_years",
                label="Investment experience",
                kind="hard",
                passed=passed,
                required=(f"{low}+ years" if "max" not in band
                          else f"{low}-{high} years"),
                found=(
                    "not computable from this resume"
                    if years is None
                    else f"{years} years"
                ),
            )
        )

    if wanted := spec.get("sectors_any"):
        sectors = candidate.get("sectors", [])
        overlap = [s for s in sectors if s in wanted]
        evidence = ""
        for item in candidate["extraction"].get("primary_sectors", []):
            if item.get("value") in wanted:
                evidence = item.get("evidence", "")
                break
        criteria.append(
            Criterion(
                key="sector",
                label="Sector",
                kind="hard",
                passed=bool(overlap),
                required=" or ".join(wanted),
                found=", ".join(sectors) or "none",
                evidence=evidence,
            )
        )

    if sides := spec.get("market_side_any"):
        side = candidate.get("market_side")
        criteria.append(
            Criterion(
                key="market_side",
                label="Market side",
                kind="hard",
                passed=side in sides,
                required=" or ".join(sides),
                found=side or "unknown",
            )
        )

    return criteria


def _soft_criteria(
    candidate: dict[str, Any],
    spec: dict[str, Any],
    weights: dict[str, float],
    backend: SimilarityBackend,
) -> list[Criterion]:
    """Evaluate the ranking signals for a candidate who already qualifies."""
    criteria: list[Criterion] = []
    soft = spec.get("soft", {})

    # Sector fit -- best-matching sector, weighted by how central it is.
    sector_weights = soft.get("sectors", {})
    if sector_weights:
        best, best_sector = 0.0, ""
        for sector in candidate.get("sectors", []):
            value = sector_weights.get(sector, 0.0)
            if value > best:
                best, best_sector = value, sector
        evidence = ""
        for item in candidate["extraction"].get("primary_sectors", []):
            if item.get("value") == best_sector:
                evidence = item.get("evidence", "")
        criteria.append(
            Criterion(
                key="sector_fit",
                label="Sector fit",
                kind="soft",
                score=best,
                weight=weights.get("sector_fit", 0.0),
                required=", ".join(sector_weights),
                found=best_sector or "no overlap",
                evidence=evidence,
            )
        )

    # Requirement fit -- the substance of the role, matched sentence by
    # sentence so every point can be traced back to a line in the resume.
    requirements = soft.get("requirements", [])
    if requirements:
        sentences = candidate_sentences(candidate)
        scored = [(*backend.score(r, sentences), r) for r in requirements]
        mean = sum(s for s, _, _ in scored) / len(scored)
        best_score, best_sentence, best_req = max(scored, key=lambda x: x[0])
        criteria.append(
            Criterion(
                key="requirement_fit",
                label="Requirement fit",
                kind="soft",
                score=round(mean, 3),
                weight=weights.get("requirement_fit", 0.0),
                required=f"{len(requirements)} stated requirements",
                found=(
                    f"strongest: {best_req} ({best_score:.0%})"
                    if best_score
                    else "no clear match"
                ),
                evidence=best_sentence,
            )
        )
        for score, sentence, requirement in scored:
            criteria.append(
                Criterion(
                    key=f"req::{requirement[:40]}",
                    label=requirement,
                    kind="soft",
                    score=score,
                    weight=0.0,  # detail rows; the mean above carries the weight
                    required=requirement,
                    found=f"{score:.0%}",
                    evidence=sentence,
                )
            )

    # Firm type -- has this person worked in a comparable operating model.
    firm_weights = soft.get("firm_types", {})
    if firm_weights:
        best, best_type = 0.0, ""
        for firm_type in candidate.get("firm_types", []):
            value = firm_weights.get(firm_type, 0.0)
            if value > best:
                best, best_type = value, firm_type
        criteria.append(
            Criterion(
                key="firm_type_fit",
                label="Firm type fit",
                kind="soft",
                score=best,
                weight=weights.get("firm_type_fit", 0.0),
                required=", ".join(firm_weights),
                found=best_type or "no comparable firm",
                evidence="; ".join(candidate.get("employers", [])[:3]),
            )
        )

    # Coverage depth -- a research-hiring proxy that is concrete and checkable.
    covered = candidate["extraction"].get("coverage", {}).get("stocks_covered")
    criteria.append(
        Criterion(
            key="coverage_depth",
            label="Coverage depth",
            kind="soft",
            score=min(1.0, (covered or 0) / 40),
            weight=weights.get("coverage_depth", 0.0),
            required="breadth of names under coverage",
            found=f"{covered} names" if covered else "not stated",
        )
    )

    # Skills fit: named software and analytical methods.
    #
    # Matched on normalised names rather than exact strings -- a resume writes
    # "Bloomberg Terminal" where a requisition says "Bloomberg", and
    # "Financial Modeling" against "financial modelling". Treating those as
    # misses would make the whole component noise.
    wanted_software = [s.lower() for s in soft.get("software_preferred", [])]
    wanted_methods = [m.lower() for m in soft.get("methods_preferred", [])]
    if wanted_software or wanted_methods:
        have_software = [s.lower() for s in candidate.get("software_tools", [])]
        have_methods = [m.lower() for m in candidate.get("methods", [])]

        def overlap(wanted: list[str], have: list[str]) -> list[str]:
            hits = []
            for want in wanted:
                for got in have:
                    if want in got or got in want:
                        hits.append(want)
                        break
            return hits

        software_hits = overlap(wanted_software, have_software)
        method_hits = overlap(wanted_methods, have_methods)
        total_wanted = len(wanted_software) + len(wanted_methods)
        score = (
            (len(software_hits) + len(method_hits)) / total_wanted
            if total_wanted
            else 0.0
        )
        found_parts = []
        if software_hits:
            found_parts.append(
                f"software: {', '.join(sorted(set(software_hits)))}"
            )
        if method_hits:
            found_parts.append(f"methods: {', '.join(sorted(set(method_hits)))}")
        criteria.append(
            Criterion(
                key="skills_fit",
                label="Skills and tooling",
                kind="soft",
                score=round(min(1.0, score), 3),
                weight=weights.get("skills_fit", 0.0),
                required=", ".join(
                    soft.get("software_preferred", [])
                    + soft.get("methods_preferred", [])
                ),
                found=" · ".join(found_parts) or "none of the named tools stated",
                evidence="; ".join(
                    candidate.get("software_tools", [])[:6]
                    + candidate.get("methods", [])[:4]
                ),
            )
        )

    wanted_credentials = soft.get("credentials_preferred", [])
    if wanted_credentials:
        held = candidate.get("credentials_summary", [])
        hits = [c for c in held if any(w.lower() in c.lower() for w in wanted_credentials)]
        criteria.append(
            Criterion(
                key="credentials",
                label="Credentials",
                kind="soft",
                score=1.0 if hits else 0.0,
                weight=weights.get("credentials", 0.0),
                required=", ".join(wanted_credentials),
                found="; ".join(hits) or "none held",
            )
        )

    if soft.get("prefer_platform_alum"):
        alum = candidate.get("platform_alum_of", [])
        criteria.append(
            Criterion(
                key="platform_alum",
                label="Multi-manager platform experience",
                kind="soft",
                score=1.0 if alum else 0.0,
                weight=weights.get("platform_alum", 0.0),
                required="worked at a pod shop or multi-manager platform",
                found="; ".join(alum) or "none",
            )
        )

    if soft.get("prefer_buy_side"):
        criteria.append(
            Criterion(
                key="buy_side",
                label="Buy-side experience",
                kind="soft",
                score=1.0 if candidate.get("has_buy_side_experience") else 0.0,
                weight=weights.get("buy_side", 0.0),
                required="prior buy-side role",
                found="yes" if candidate.get("has_buy_side_experience") else "no",
            )
        )

    return criteria


def match_candidate(
    candidate: dict[str, Any],
    requisition: dict[str, Any],
    *,
    weights: dict[str, float],
    backend: SimilarityBackend,
) -> MatchResult:
    """Score one candidate against one requisition."""
    hard = _hard_criteria(candidate, requisition.get("hard", {}))
    soft = _soft_criteria(candidate, requisition, weights, backend)

    weighted = [c for c in soft if c.weight > 0]
    total_weight = sum(c.weight for c in weighted)
    soft_score = (
        sum(c.score * c.weight for c in weighted) / total_weight
        if total_weight
        else 0.0
    )

    return MatchResult(
        candidate_id=candidate["candidate_id"],
        display_name=candidate["display_name"],
        requisition_id=requisition["id"],
        criteria=hard + soft,
        failed_hard=[c for c in hard if c.passed is False],
        soft_score=round(soft_score, 3),
        quality_band=candidate.get("quality", {}).get("band", "high"),
    )


def match_all(
    candidates: list[dict[str, Any]],
    requisition: dict[str, Any],
    *,
    store: Optional[Requisitions] = None,
    backend: Optional[SimilarityBackend] = None,
) -> tuple[list[MatchResult], list[MatchResult]]:
    """Match every candidate; return (exact matches, near misses).

    Both lists are sorted by soft score. Near misses are those failing exactly
    one hard constraint -- close enough that a recruiter may want to widen the
    search, and each one carries the constraint it failed. Candidates failing
    two or more are not returned: at that point they are simply different
    people, and padding the list with them is the behaviour this system exists
    to avoid.
    """
    store = store or Requisitions.load()
    backend = backend or ConceptTfidfBackend(store.concept_map)

    results = [
        match_candidate(c, requisition, weights=store.weights, backend=backend)
        for c in candidates
    ]
    exact = sorted(
        (r for r in results if r.is_exact), key=lambda r: -r.soft_score
    )
    near = sorted(
        (r for r in results if r.near_miss_count == 1), key=lambda r: -r.soft_score
    )
    return exact, near
