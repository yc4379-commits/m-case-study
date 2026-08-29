"""
Pipeline entry point: resumes in, dataset out.

    python src/build_dataset.py [--model claude-sonnet-5] [--no-cache]

Produces three artefacts in `data/`:

  candidates.json    Full nested records, including evidence quotes and the
                     per-record quality report. This is what the app loads.
  candidates.csv     One flat row per candidate for spreadsheet review. Nested
                     structures are collapsed to readable strings, because a
                     CSV of JSON blobs helps nobody.
  extraction_log.csv One row per document describing how it was read --
                     table share, ligature repairs, column splits, warnings.

The split matters: JSON serves the application, CSV serves a human checking
our work, and the log serves whoever has to debug a bad record later.

Parsing runs concurrently and is cached, so a re-run after changing only the
knowledge base or the enrichment logic costs nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich import enrich  # noqa: E402
from extract import extract_directory  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from parse import DEFAULT_MODEL, parse_resume  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESUME_DIR = ROOT / "data" / "resumes"
OUT_DIR = ROOT / "data"


def _num(value) -> str:
    """Render a number, keeping 0.0 visible and blanking only None."""
    return "" if value is None else str(value)


def _flatten(candidate) -> dict:
    """One flat, human-readable row per candidate."""
    e = candidate.extraction
    current = next((p for p in e.positions if p.is_current), None)
    return {
        "candidate_id": candidate.candidate_id,
        "name": candidate.display_name,
        "name_source": candidate.name_source,
        "email": e.email or "",
        "location": candidate.location or "",
        "region": candidate.region or "",
        # `x or ""` would blank a legitimate 0.0 -- exactly the value that
        # now carries meaning ("definitely no investment experience").
        "years_experience": _num(candidate.years_experience),
        "years_investment_experience": _num(candidate.years_investment_experience),
        "in_junior_range": candidate.is_junior_range,
        "seniority_band": candidate.seniority_band or "",
        "investment_seniority_band": candidate.investment_seniority_band or "",
        "approach_family": candidate.approach_family or "",
        "approach": candidate.approach or "",
        "approach_keywords": "; ".join(e.investment_approach.keywords),
        "approach_confidence": e.investment_approach.confidence,
        "approach_evidence": e.investment_approach.evidence,
        "market_side": candidate.market_side or "",
        "sectors": "; ".join(candidate.sectors),
        "asset_classes": "; ".join(candidate.asset_classes),
        "current_firm": candidate.current_firm or "",
        "current_firm_type": candidate.current_firm_type or "",
        "employers": " | ".join(candidate.employers),
        "non_professional_affiliations": " | ".join(
            candidate.non_professional_affiliations
        ),
        "unresolved_firms": "; ".join(
            f.raw for f in candidate.firms if f.resolution in {"unresolved", "ambiguous"}
        ),
        "platform_alum_of": "; ".join(candidate.platform_alum_of),
        "has_buy_side": candidate.has_buy_side_experience,
        "has_sell_side": candidate.has_sell_side_experience,
        "credentials": "; ".join(candidate.credentials_summary),
        "stocks_covered": e.coverage.stocks_covered or "",
        "coverage_markets": "; ".join(candidate.coverage_markets),
        "coverage_markets_source": candidate.coverage_markets_source,
        "languages": "; ".join(candidate.languages),
        "software_tools": "; ".join(candidate.software_tools),
        "methods": "; ".join(candidate.methods),
        "n_positions": len(e.positions),
        "quality_score": candidate.quality.score,
        "quality_band": candidate.quality.band,
        "missing_fields": "; ".join(candidate.quality.missing_fields),
        "flags": "\n".join(f"- [{f.category}] {f.summary}" for f in candidate.flags),
        "flag_count": len(candidate.flags),
        "source_file": candidate.source_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resumes", default=str(RESUME_DIR))
    args = parser.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()
    kb = KnowledgeBase.load()

    documents = extract_directory(args.resumes)
    if not documents:
        print(f"No resumes found in {args.resumes}", file=sys.stderr)
        return 1
    print(f"Extracted {len(documents)} document(s).")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parses = list(
            pool.map(
                lambda d: parse_resume(
                    d[0],
                    d[1].source_file,
                    client=client,
                    model=args.model,
                    use_cache=not args.no_cache,
                ),
                documents,
            )
        )

    failures = [p for p in parses if not p.ok]
    for failure in failures:
        print(f"  FAILED {failure.source_file}: {failure.error}", file=sys.stderr)

    candidates = [
        enrich(
            parse.extraction,
            report.source_file,
            extraction_chars=report.char_count,
            extraction_report=report,
            unverified_quotes=parse.unverified_quotes,
            kb=kb,
        )
        for parse, (_, report) in zip(parses, documents)
        if parse.ok
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "candidates.json").write_text(
        json.dumps(
            [c.model_dump() for c in candidates], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    rows = [_flatten(c) for c in candidates]
    with (OUT_DIR / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (OUT_DIR / "extraction_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source_file", "file_type", "char_count", "table_count",
                "table_share", "textbox_count", "textbox_chars", "page_count",
                "multi_column_detected", "ligature_repairs",
                "replacement_chars_remaining", "warnings",
            ],
        )
        writer.writeheader()
        for _, report in documents:
            row = report.to_dict()
            row["warnings"] = " | ".join(row.pop("warnings"))
            row.pop("paragraph_chars", None)
            row.pop("table_chars", None)
            writer.writerow(row)

    # Only uncached calls actually spent money; reporting the cached total as
    # "cost this run" would overstate spend every time the pipeline is re-run.
    cost = sum(p.cost_usd for p in parses if not p.from_cache)
    would_cost = sum(p.cost_usd for p in parses)
    cached = sum(p.from_cache for p in parses)
    retried = sum(1 for p in parses if p.attempts > 1)
    unverified = sum(len(p.unverified_quotes) for p in parses)
    low_quality = sum(1 for c in candidates if c.quality.band == "low")

    print(
        f"\nParsed {len(candidates)}/{len(documents)}  "
        f"(cached {cached}, retried {retried})\n"
        f"Cost this run: ${cost:.4f}  (${would_cost:.4f} without cache)\n"
        f"Unverified evidence quotes: {unverified}\n"
        f"Low-confidence records: {low_quality}\n"
        f"Wrote candidates.json, candidates.csv, extraction_log.csv to {OUT_DIR}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
