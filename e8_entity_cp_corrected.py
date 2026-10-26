#!/usr/bin/env python3
"""Entity-level Clopper-Pearson certificate for the frozen weight (corrected).

This script replaces v47_results/code/e8_entity_cp.py.  The original helper
computed the upper limit as beta.ppf(1 - level, harm, n - harm + 1), which is
the standard construction for a *lower* limit and therefore understates the
upper bound.  The correct one-sided upper limit for k harmful events out of n
units at confidence 1 - alpha is

    U = beta.ppf(1 - alpha, k + 1, n - k),

obtained by inverting P(X <= k ; p = U) = alpha.  On the 220-entity draw the
two conventions give 0.0337 and 0.0407.  Both stay below the 0.08 cap, and the
manuscript reports the corrected 0.0407 together with the question-level
comparison 0.0205.

Protocol (unchanged from the sealed design)
-------------------------------------------
Population  e1 fresh held-out draw, 440 questions over 220 entities.
Designation the first question of each entity in sealed manifest order,
            fixed before labels are opened.
Statistic   a designated unit is harmed when the base answer is correct and the
            pooled answer at weight alpha is wrong.
Bounding    one-sided Clopper-Pearson at level 1 - delta / |grid| with
            delta = 0.10 and the five-point grid {0.35, ..., 0.55}.

Usage
-----
    python -P code/e8_entity_cp_corrected.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

GRID = [0.35, 0.40, 0.45, 0.50, 0.55]
DELTA = 0.10

HERE = Path(__file__).resolve()
REV = HERE.parents[1]
ROOT = HERE.parents[3]
ROOT_E1 = ROOT / "new" / "v47_results" / "e1_new_population"
OUT = REV / "results"


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)


def loglin(p0, pe, alpha):
    return norm(np.exp((1 - alpha) * np.log(norm(p0)) + alpha * np.log(norm(pe))))


def cp_upper(k_harm, n_units, level):
    """One-sided Clopper-Pearson upper bound at confidence level."""
    if n_units <= 0:
        return 1.0
    if k_harm >= n_units:
        return 1.0
    return float(beta.ppf(level, k_harm + 1, n_units - k_harm))


def main():
    manifest = list(iter_jsonl(ROOT_E1 / "e1_manifest.jsonl"))
    preds = {r["question_id"]: r for r in iter_jsonl(ROOT_E1 / "e1_predictions.jsonl")}

    designated, seen = [], set()
    for r in manifest:
        e = r["entity_id"]
        if e not in seen:
            seen.add(e)
            designated.append(r["question_id"])
    assert len(designated) == len(seen)

    rows = [preds[q] for q in designated]
    gold = np.array([int(r["gold_index"]) for r in rows])
    p0 = norm(np.array([r["p0_posterior"] for r in rows]))
    pe = norm(np.array([r["ea_pE_posterior"] for r in rows]))
    base_ok = p0.argmax(1) == gold

    report = {"population": "e1 fresh held-out", "n_entities": len(rows),
              "n_base_correct": int(base_ok.sum()), "delta": DELTA,
              "bonferroni_m": len(GRID), "grid": []}
    for a in GRID:
        ok = loglin(p0, pe, a).argmax(1) == gold
        harm = int((base_ok & ~ok).sum())
        ub = cp_upper(harm, len(rows), 1 - DELTA / len(GRID))
        report["grid"].append({"alpha": a, "n_harm": harm,
                               "cp_upper_harm_rate": ub,
                               "certified": ub <= 0.08})
    question_level = cp_upper(int((base_ok & ~(loglin(p0, pe, 0.45).argmax(1) == gold)).sum()),
                              len(rows), 1 - DELTA / len(GRID))
    report["question_level_bound"] = question_level
    report["note"] = ("Corrected upper limit beta.ppf(1-level, k+1, n-k). "
                      "The previous helper reported 0.0337 for the same counts.")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "entity_certificate.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
