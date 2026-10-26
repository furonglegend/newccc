#!/usr/bin/env python3
"""Selective revision rules for the 72B reader, selected on dev and tested once.

Development  600 questions / 508 entities, cached 72B posteriors.
Test         440-question fresh draw, cached 72B posteriors.

Families
  F0  global log-linear weight (reference)
  F1  margin veto, keep the base when its margin exceeds tau and the pool disagrees
  F2  pooled-confidence gate, adopt the pool only when its top probability exceeds tau
  F3  evidence-margin rule, adopt only when the evidence top probability exceeds
      the base top probability by tau
  F4  per-margin-bucket weight, one weight per tercile of the base margin

Selection uses development labels with an entity-level Clopper-Pearson cap over
the whole family; the fresh draw is read once.

Usage
-----
    python -P code/policy_search_72b.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import beta

SEED = 20261025
DRAWS = 10000
CAP = 0.08
DELTA = 0.10
ALPHA = 0.45

HERE = Path(__file__).resolve()
REV = HERE.parents[1]
ROOT = HERE.parents[3]
V47 = ROOT / "new" / "v47_results"
CACHE = V47 / "e10_language_arb" / "cache"
OUT = REV / "results"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(p):
    p = np.clip(np.asarray(p, float), 1e-12, None)
    return p / p.sum(-1, keepdims=True)


def loglin(p0, pe, a):
    return norm(np.exp((1 - a) * np.log(norm(p0)) + a * np.log(norm(pe))))


def cp_upper(k, n, level):
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def mcnemar(a, b):
    x = int((a & ~b).sum())
    y = int((~a & b).sum())
    n = x + y
    if n == 0:
        return 1.0, 0, 0
    k = min(x, y)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)), x, y


def boot(a, b, cl, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    cl = np.asarray(cl)
    u = sorted(set(cl.tolist()))
    idx = {c: np.flatnonzero(cl == c) for c in u}
    n = len(u)
    d = np.empty(draws)
    for i in range(draws):
        s = rng.integers(0, n, n)
        ii = np.concatenate([idx[u[j]] for j in s])
        d[i] = a[ii].mean() - b[ii].mean()
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def counts(pred, gold, bp):
    ok = pred == gold
    bo = bp == gold
    return float(ok.mean()), int(np.sum(~bo & ok)), int(np.sum(bo & ~ok)), ok


def apply(fam, tau, p0, pe, pooled, buckets=None):
    base = p0.argmax(1)
    pool = pooled.argmax(1)
    s0 = np.sort(p0, axis=1)
    m0 = s0[:, -1] - s0[:, -2]
    pred = pool.copy()
    if fam == "F1":
        keep = (pool != base) & (m0 > tau)
        pred[keep] = base[keep]
    elif fam == "F2":
        keep = pooled.max(1) < tau
        pred[keep] = base[keep]
    elif fam == "F3":
        keep = (pool != base) & ((pe.max(1) - p0.max(1)) <= tau)
        pred[keep] = base[keep]
    elif fam == "F4":
        for b, a in tau.items():
            m = buckets == b
            if m.any():
                pred[m] = loglin(p0[m], pe[m], a).argmax(1)
    else:
        raise ValueError(fam)
    return pred


def main():
    inst = {str(r["question_id"]): r for r in jl(CACHE / "dev_instances.jsonl")}
    db = {r["question_id"]: r for r in jl(CACHE / "dev_base_72b.jsonl")}
    dj = {r["question_id"]: r for r in jl(CACHE / "dev_joint_72b.jsonl")}
    qs = [q for q in inst if q in db and q in dj]
    g = np.array([int(inst[q]["gold_index"]) for q in qs])
    p0 = norm(np.array([db[q]["probabilities"] for q in qs]))
    pe = norm(np.array([dj[q]["probabilities"] for q in qs]))
    bp = p0.argmax(1)
    bo = bp == g
    ent = [str(inst[q]["entity_id"]) for q in qs]

    # designated entity units
    des = {}
    for q in qs:
        e = str(inst[q]["entity_id"])
        if e not in des:
            des[e] = q
        elif str(inst[q].get("query_language")) == "en" and \
                str(inst[des[e]].get("query_language")) != "en":
            des[e] = q
    dq = [des[e] for e in sorted(des)]
    ix = {q: i for i, q in enumerate(qs)}
    di = np.array([ix[q] for q in dq])
    dg, d0, de = g[di], p0[di], pe[di]
    dbo = d0.argmax(1) == dg
    n_risk = int(dbo.sum())

    # margin buckets from development cutpoints
    m0_dev = np.sort(p0, axis=1)[:, -1] - np.sort(p0, axis=1)[:, -2]
    cuts = np.quantile(m0_dev, [1 / 3, 2 / 3])
    buckets = np.digitize(m0_dev, cuts)
    d_buckets = buckets[di]

    pooled_dev = loglin(p0, pe, ALPHA)
    pooled_d = loglin(d0, de, ALPHA)
    grid = [round(0.05 * i, 2) for i in range(0, 19)]
    total = len(grid) * 3 + len(grid) ** 3  # F1..F3 plus a coarse F4
    level = 1 - DELTA / total

    out = {"dev": {"n": len(qs), "norag": float(bo.mean()),
                   "global045": counts(pooled_dev.argmax(1), g, bp)[0]},
           "families": {}}

    for fam in ["F1", "F2", "F3"]:
        rows = []
        for tau in grid:
            dv = apply(fam, tau, d0, de, pooled_d)
            ok = dv == dg
            harm = int((dbo & ~ok).sum())
            bound = cp_upper(harm, n_risk, level)
            acc = counts(apply(fam, tau, p0, pe, pooled_dev), g, bp)[0]
            rows.append({"tau": tau, "dev_acc": acc, "harm": harm,
                         "bound": bound, "certified": bound <= CAP})
        feas = [r for r in rows if r["certified"]]
        chosen = max(feas, key=lambda r: r["dev_acc"]) if feas else None
        out["families"][fam] = {"rows": rows, "chosen": chosen}

    # F4: per-bucket weight, coarse grid to keep the family small
    coarse = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
    best = None
    for a0 in coarse:
        for a1 in coarse:
            for a2 in coarse:
                rule = {0: a0, 1: a1, 2: a2}
                dv = apply("F4", rule, d0, de, pooled_d, d_buckets)
                ok = dv == dg
                harm = int((dbo & ~ok).sum())
                if harm != 0 and cp_upper(harm, n_risk, 1 - DELTA / 216) > CAP:
                    continue
                acc = counts(apply("F4", rule, p0, pe, pooled_dev, buckets), g, bp)[0]
                if best is None or acc > best[0]:
                    best = (acc, rule, harm)
    out["families"]["F4"] = {"chosen": {"rule": best[1], "dev_acc": best[0],
                                        "harm": best[2]} if best else None}

    # ---- fresh draw ----
    gold = {r["question_id"]: int(r["gold_index"])
            for r in jl(V47 / "e1_new_population" / "e1_gold_sealed.jsonl")}
    b1 = {r["question_id"]: r for r in jl(V47 / "e2_reader_scale" / "e1_base_72b.jsonl")}
    j1 = {r["question_id"]: r for r in jl(V47 / "e2_reader_scale" / "e1_joint_72b.jsonl")}
    q1 = [q for q in gold if q in b1 and q in j1]
    g1 = np.array([gold[q] for q in q1])
    P0 = norm(np.array([b1[q]["probabilities"] for q in q1]))
    PE = norm(np.array([j1[q]["probabilities"] for q in q1]))
    E1 = [str(b1[q]["entity_id"]) for q in q1]
    B1 = P0.argmax(1)
    BO = B1 == g1
    pool1 = loglin(P0, PE, ALPHA)
    m1 = np.sort(P0, axis=1)[:, -1] - np.sort(P0, axis=1)[:, -2]
    b1k = np.digitize(m1, cuts)
    acc0, c0, h0, ok0 = counts(pool1.argmax(1), g1, B1)
    out["e1"] = {"norag": float(BO.mean()),
                 "evidence": float((PE.argmax(1) == g1).mean()),
                 "global045": {"acc": acc0, "corr": c0, "harm": h0,
                               "ci_vs_norag": boot(ok0, BO, E1),
                               "mcnemar_vs_norag": mcnemar(ok0, BO)}}
    for fam in ["F1", "F2", "F3"]:
        ch = out["families"][fam]["chosen"]
        if not ch:
            continue
        pred = apply(fam, ch["tau"], P0, PE, pool1)
        acc, c, h, ok = counts(pred, g1, B1)
        out["e1"][f"{fam}_tau{ch['tau']}"] = {
            "acc": acc, "corr": c, "harm": h,
            "ci_vs_norag": boot(ok, BO, E1),
            "mcnemar_vs_norag": mcnemar(ok, BO),
            "ci_vs_global": boot(ok, ok0, E1),
            "mcnemar_vs_global": mcnemar(ok, ok0)}
    ch4 = out["families"]["F4"]["chosen"]
    if ch4:
        pred = apply("F4", ch4["rule"], P0, PE, pool1, b1k)
        acc, c, h, ok = counts(pred, g1, B1)
        out["e1"]["F4"] = {"acc": acc, "corr": c, "harm": h,
                           "rule": ch4["rule"],
                           "ci_vs_norag": boot(ok, BO, E1),
                           "mcnemar_vs_norag": mcnemar(ok, BO),
                           "ci_vs_global": boot(ok, ok0, E1),
                           "mcnemar_vs_global": mcnemar(ok, ok0)}
    json.dump(out, open(OUT / "policy_search_72b.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["e1"], indent=2))


if __name__ == "__main__":
    main()
