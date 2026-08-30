"""
Regression tests for the parts of the pipeline that do not need an API key.

These are not aspirational coverage — each test pins a behaviour that was
once wrong, or an invariant the interface depends on. They run against the
committed dataset and knowledge base, so `pytest` works straight after
`pip install -r requirements-dev.txt` with no resumes and no key.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract import repair_text  # noqa: E402
from knowledge_base import KnowledgeBase, years_of_experience  # noqa: E402
from match import Requisitions, match_all  # noqa: E402
from parse import _loose  # noqa: E402
from schema import Candidate  # noqa: E402


@pytest.fixture(scope="session")
def candidates() -> list[dict]:
    return json.loads((ROOT / "data" / "candidates.json").read_text())


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.load(ROOT / "knowledge")


@pytest.fixture(scope="session")
def store() -> Requisitions:
    return Requisitions.load(ROOT / "knowledge")


# -- extraction ------------------------------------------------------------

def test_ligature_repair_fixes_broken_pdf_text():
    text, repairs, remaining = repair_text("Quan�ta�ve Por�olio analyst")
    assert text == "Quantitative Portfolio analyst"
    assert repairs == 3
    assert remaining == 0


def test_ligature_repair_leaves_clean_text_alone():
    # An earlier rule turned "Statistics" into "Statisffics"; pin that a
    # clean word passes through untouched and nothing is counted.
    text, repairs, remaining = repair_text("Statistics and Quantitative Finance")
    assert text == "Statistics and Quantitative Finance"
    assert repairs == 0


# -- entity resolution -----------------------------------------------------

def test_ambiguous_firm_prefix_is_not_resolved(kb):
    # Four unrelated firms in the corpus begin with "Meridian"; a substring
    # matcher would silently pick one and relocate the candidate.
    match = kb.resolve_firm("Meridian")
    assert match.method in {"ambiguous", "unresolved"}
    assert not match.resolved


def test_platform_lineage_reaches_the_platform(kb):
    # The resume names the pod; the platform only exists in the knowledge
    # base. This lookup is what surfaces "previously at Millennium".
    assert "Millennium Management" in (kb.platform_lineage("North53 Capital")
                                       or [])


# -- tenure arithmetic -----------------------------------------------------

def test_overlapping_positions_are_not_double_counted():
    positions = [
        {"start_date": "2020-01", "end_date": "2022-01", "is_current": False,
         "employment_type": "professional", "is_investment_role": True},
        {"start_date": "2021-01", "end_date": "2023-01", "is_current": False,
         "employment_type": "professional", "is_investment_role": True},
    ]
    assert years_of_experience(positions) == pytest.approx(3.0, abs=0.1)


def test_internships_and_societies_excluded_from_tenure():
    positions = [
        {"start_date": "2020-01", "end_date": "2021-01", "is_current": False,
         "employment_type": "internship", "is_investment_role": True},
        {"start_date": "2021-01", "end_date": "2022-01", "is_current": False,
         "employment_type": "student_organization",
         "is_investment_role": True},
    ]
    assert years_of_experience(positions) == 0.0


# -- schema ----------------------------------------------------------------

def test_committed_dataset_validates_against_schema(candidates):
    for record in candidates:
        Candidate.model_validate(record)


# -- quote verification ----------------------------------------------------

def test_quote_normalisation_forgives_layout_not_words():
    # verify_quotes compares through _loose(): whitespace and punctuation
    # are forgiven (our own extraction reflows lines), words are not --
    # that is the entire point of verbatim checking.
    source = _loose("Covered 40 healthcare\nnames across APAC markets.")
    assert _loose("Covered  40, healthcare—names") in source
    assert _loose("Researched forty healthcare stocks") not in source


# -- matching invariants ---------------------------------------------------

def test_exact_and_near_are_disjoint_and_scores_bounded(candidates, store):
    for spec in store.items:
        exact, near = match_all(candidates, spec, store=store)
        exact_ids = {r.candidate_id for r in exact}
        near_ids = {r.candidate_id for r in near}
        assert not exact_ids & near_ids
        for r in exact:
            assert not r.failed_hard
        for r in near:
            assert len(r.failed_hard) == 1
        for r in [*exact, *near]:
            assert 0.0 <= r.soft_score <= 1.0


def test_matcher_can_return_zero_results(candidates, store):
    # The whole design argument rests on the search being allowed to say
    # "no one qualifies". At least one real requisition must exercise it.
    outcomes = [len(match_all(candidates, spec, store=store)[0])
                for spec in store.items]
    assert 0 in outcomes


# -- evaluation ------------------------------------------------------------

def test_evaluation_counts_are_consistent():
    sys.path.insert(0, str(ROOT / "src"))
    from evaluate import evaluate
    result = evaluate()
    o = result["overall"]
    assert o["tp"] + o["fp"] + o["fn"] + o["tn"] == o["judged"]
    assert (o["judged"] + len(result["borderline"])
            == result["labelled_pairs"])
    if o["precision"] is not None:
        assert 0.0 <= o["precision"] <= 1.0
