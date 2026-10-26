#!/usr/bin/env python3
"""Dev-selected revision policies, evaluated once on the confirmation set.

Policy families, all operating on the same cached posteriors:
  F0  global log-linear pool at a single weight (reference)
  F1  margin veto, keep the base answer when its margin exceeds tau and the
      pool disagrees
  F2  pooled-confidence gate, adopt the pool only when its top probability
      exceeds tau
  F3  evidence-margin rule, adopt the pool only when the evidence top
      probability exceeds the base top probability by tau

Each family sweeps tau on the development population, keeps the values whose
entity-level Clopper-Pearson harm bound stays under the cap, and freezes the
most accurate survivor before the confirmation labels are read.

Usage
-----
    python -P code/policy_search.py
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
NEW = REV / "results" / "new_reads"
E10 = ROOT / "new" / "v47_results" / "e10_language_arb" / "cache"
V47 = ROOT / "new" / "v47_results"
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


def apply_policy(fam, tau, p0, pe, pooled):
    base = p0.argmax(1)
    pool = pooled.argmax(1)
    s0 = np.sort(p0, axis=1)
    m0 = s0[:, -1] - s0[:, -2]
    topE = pe.max(1)
    top0 = p0.max(1)
    pred = pool.copy()
    if fam == "F1":
        keep = (pool != base) & (m0 > tau)
        pred[keep] = base[keep]
    elif fam == "F2":
        keep = pooled.max(1) < tau
        pred[keep] = base[keep]
    elif fam == "F3":
        keep = (pool != base) & ((topE - top0) <= tau)
        pred[keep] = base[keep]
    else:
        raise ValueError(fam)
    return pred


def main():
    dev_inst = {str(r["question_id"]): r for r in jl(E10 / "dev_instances.jsonl")}
    db = {r["question_id"]: r for r in jl(NEW / "dev_base_7b.jsonl")}
    dj = {r["question_id"]: r for r in jl(NEW / "dev_joint5_7b.jsonl")}
    qs = [q for q in dev_inst if q in db and q in dj]
    g = np.array([int(dev_inst[q]["gold_index"]) for q in qs])
    p0 = norm(np.array([db[q]["probabilities"] for q in qs]))
    pe = norm(np.array([dj[q]["probabilities"] for q in qs]))
    bp = p0.argmax(1)
    bo = bp == g
    ent = [str(dev_inst[q]["entity_id"]) for q in qs]

    # designated units
    des = {}
    for q in qs:
        e = str(dev_inst[q]["entity_id"])
        if e not in des:
            des[e] = q
        elif str(dev_inst[q].get("query_language")) == "en" and \
                str(dev_inst[des[e]].get("query_language")) != "en":
            des[e] = q
    dq = [des[e] for e in sorted(des)]
    ix = {q: i for i, q in enumerate(qs)}
    di = np.array([ix[q] for q in dq])
    dg, d0, de = g[di], p0[di], pe[di]
    dbo = d0.argmax(1) == dg
    n_risk = int(dbo.sum())

    pooled_dev = loglin(p0, pe, ALPHA)
    pooled_d = loglin(d0, de, ALPHA)
    grid = [round(0.05 * i, 2) for i in range(0, 19)]
    fams = {"F1": grid, "F2": grid, "F3": grid}
    total = sum(len(v) for v in fams.values())
    level = 1 - DELTA / total

    out = {"dev": {"n": len(qs), "norag": float(bo.mean()),
                   "global045": counts(pooled_dev.argmax(1), g, bp)[0]},
           "families": {}}

    for fam, taus in fams.items():
        rows = []
        for tau in taus:
            dv = apply_policy(fam, tau, d0, de, pooled_d)
            ok = dv == dg
            harm = int((dbo & ~ok).sum())
            bound = cp_upper(harm, n_risk, level)
            acc = counts(apply_policy(fam, tau, p0, pe, pooled_dev), g, bp)[0]
            rows.append({"tau": tau, "dev_acc": acc, "harm": harm,
                         "bound": bound, "certified": bound <= CAP})
        feas = [r for r in rows if r["certified"]]
        chosen = max(feas, key=lambda r: r["dev_acc"]) if feas else None
        out["families"][fam] = {"rows": rows, "chosen": chosen}

    rules = {f: out["families"][f]["chosen"]["tau"]
             for f in fams if out["families"][f]["chosen"]}

    # confirmation
    e9 = {r["question_id"]: r for r in jl(V47 / "e9_large_population" / "e9_predictions.jsonl")}
    q9 = sorted(e9)
    g9 = np.array([int(e9[q]["gold_index"]) for q in q9])
    p09 = norm(np.array([e9[q]["p0_posterior"] for q in q9]))
    p59 = norm(np.array([e9[q]["ea_pE_posterior"] for q in q9]))
    e9e = [str(e9[q]["entity_id"]) for q in q9]
    b9 = p09.argmax(1)
    b9o = b9 == g9
    pool9 = loglin(p09, p59, ALPHA)
    acc0, c0, h0, ok0 = counts(pool9.argmax(1), g9, b9)
    out["e9"] = {"global045": {"acc": acc0, "corr": c0, "harm": h0,
                               "ci_vs_norag": boot(ok0, b9o, e9e),
                               "mcnemar_vs_norag": mcnemar(ok0, b9o)}}
    for fam, tau in rules.items():
        pred = apply_policy(fam, tau, p09, p59, pool9)
        acc, c, h, ok = counts(pred, g9, b9)
        out["e9"][f"{fam}_tau{tau}"] = {
            "acc": acc, "corr": c, "harm": h,
            "ci_vs_norag": boot(ok, b9o, e9e),
            "mcnemar_vs_norag": mcnemar(ok, b9o),
            "ci_vs_global": boot(ok, ok0, e9e),
            "mcnemar_vs_global": mcnemar(ok, ok0)}
    json.dump(out, open(OUT / "policy_search.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2)[:6000])


if __name__ == "__main__":
    main()
