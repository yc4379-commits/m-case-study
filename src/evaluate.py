"""
Accuracy evaluation against human ground truth.

The labels in `data/ground_truth.csv` were produced blind: the reviewer saw
each candidate's facts (region, approach, tenure, sectors, firm) but not the
system's verdict, and answered one question per candidate-role pair --
"would you put this person on the shortlist for this seat?" Blindness
matters; a labeller who can see the system's answer anchors on it, and the
evaluation stops measuring anything.

The system's shortlist for a role is its set of exact matches -- candidates
meeting every hard requirement. Near misses are, by design, NOT the
shortlist: they are the widening options. So the comparison is strict:

    system says qualify  vs  human says yes    -> true positive
    system says qualify  vs  human says no     -> false positive
    system excludes      vs  human says yes    -> false negative

Borderline labels are excluded from the counts and reported separately --
forcing a coin-flip into either bucket would manufacture agreement or
disagreement that the labeller declined to assert.

With 10 resumes and 4 requisitions this is n=40, which measures nothing
statistically. What it does measure is where the system's RULES diverge
from a practitioner's judgment, and each divergence is individually
inspectable -- at this scale the disagreement list, not the percentages,
is the finding.

Run:  python src/evaluate.py     (writes data/evaluation.json, prints a table)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from match import Requisitions, match_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def evaluate() -> dict:
    candidates = json.loads((DATA / "candidates.json").read_text())
    store = Requisitions.load()
    with open(DATA / "ground_truth.csv", newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    by_id = {c["candidate_id"]: c for c in candidates}
    per_role: list[dict] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    disagreements: list[dict] = []
    borderlines: list[dict] = []

    for spec in store.items:
        rid = spec["id"]
        exact, near = match_all(candidates, spec, store=store)
        system_yes = {r.candidate_id for r in exact}
        near_by_id = {r.candidate_id: r for r in near}

        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for row in labels:
            if row["requisition_id"] != rid:
                continue
            cid, label = row["candidate_id"], row["label"]
            sys_yes = cid in system_yes
            if label == "borderline":
                borderlines.append({
                    "role": spec["title"], "candidate": row["candidate"],
                    "system": "qualifies" if sys_yes else "excluded",
                })
                continue
            human_yes = label == "yes"
            key = ("tp" if human_yes else "fp") if sys_yes else \
                  ("fn" if human_yes else "tn")
            counts[key] += 1
            if key in ("fp", "fn"):
                m = near_by_id.get(cid)
                reason = (f"{m.failed_hard[0].label}: has "
                          f"{m.failed_hard[0].found}, role needs "
                          f"{m.failed_hard[0].required}"
                          if m and m.failed_hard
                          else "fails two or more hard requirements")
                disagreements.append({
                    "role": spec["title"], "candidate": row["candidate"],
                    "kind": "false_negative" if key == "fn" else "false_positive",
                    "system_reason": reason,
                })
        for k in totals:
            totals[k] += counts[k]
        judged = sum(counts.values())
        per_role.append({
            "role": spec["title"], "id": rid, **counts, "judged": judged,
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "agreement": _ratio(counts["tp"] + counts["tn"], judged),
        })

    judged = sum(totals.values())
    result = {
        "per_role": per_role,
        "overall": {
            **totals, "judged": judged,
            "precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
            "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
            "agreement": _ratio(totals["tp"] + totals["tn"], judged),
        },
        "disagreements": disagreements,
        "borderline": borderlines,
        "labelled_pairs": len(labels),
    }
    return result


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def main() -> None:
    result = evaluate()
    (DATA / "evaluation.json").write_text(json.dumps(result, indent=2))

    def pct(x):
        return "  n/a" if x is None else f"{x:.0%}"

    print(f"{'Role':<46} {'P':>5} {'R':>5} {'agree':>6}  tp fp fn tn")
    for r in result["per_role"]:
        print(f"{r['role'][:45]:<46} {pct(r['precision']):>5} "
              f"{pct(r['recall']):>5} {pct(r['agreement']):>6}  "
              f"{r['tp']:>2} {r['fp']:>2} {r['fn']:>2} {r['tn']:>2}")
    o = result["overall"]
    print(f"{'OVERALL':<46} {pct(o['precision']):>5} {pct(o['recall']):>5} "
          f"{pct(o['agreement']):>6}  {o['tp']:>2} {o['fp']:>2} "
          f"{o['fn']:>2} {o['tn']:>2}")
    print(f"\nborderline (excluded from counts): {len(result['borderline'])}")
    for b in result["borderline"]:
        print(f"  {b['candidate']} × {b['role'][:40]} — system {b['system']}")
    print(f"\ndisagreements: {len(result['disagreements'])}")
    for d in result["disagreements"]:
        print(f"  [{d['kind']}] {d['candidate']} × {d['role'][:40]}")
        print(f"      system: {d['system_reason']}")


if __name__ == "__main__":
    main()
