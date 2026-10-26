#!/usr/bin/env python3
"""Preview: language-conditional arbitration selected on 72B development data.

Uses cached 72B posteriors only.
  dev    new/v47_results/e10_language_arb/cache/dev_{base,joint}_72b.jsonl
  test   new/v47_results/e2_reader_scale/e1_{base,joint}_72b.jsonl over the
         440-question fresh draw

Selection is per language group on development entities with an entity-level
Clopper-Pearson certificate (Bonferroni over both grids). The confirmation
draw is read once.
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

ROOT = Path(__file__).resolve().parents[3]
V47 = ROOT / "new" / "v47_results"
CACHE = V47 / "e10_language_arb" / "cache"
OUT = Path(__file__).resolve().parents[1] / "results"


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


def counts(pred, gold, base_pred):
    ok = pred == gold
    bo = base_pred == gold
    return float(ok.mean()), int(np.sum(~bo & ok)), int(np.sum(bo & ~ok)), ok


def main():
    inst = {str(r["question_id"]): r for r in jl(CACHE / "dev_instances.jsonl")}
    db = {r["question_id"]: r for r in jl(CACHE / "dev_base_72b.jsonl")}
    dj = {r["question_id"]: r for r in jl(CACHE / "dev_joint_72b.jsonl")}
    qs = [q for q in inst if q in db and q in dj]
    g = np.array([int(inst[q]["gold_index"]) for q in qs])
    p0 = norm(np.array([db[q]["probabilities"] for q in qs]))
    pe = norm(np.array([dj[q]["probabilities"] for q in qs]))
    lang = [str(inst[q].get("query_language", "")) for q in qs]
    ent = [str(inst[q]["entity_id"]) for q in qs]
    bp = p0.argmax(1)
    bo = bp == g
    en = np.array([l == "en" for l in lang])

    # designated unit per entity
    desig = {}
    for q in qs:
        e = str(inst[q]["entity_id"])
        if e not in desig:
            desig[e] = q
        elif str(inst[q].get("query_language")) == "en" and \
                str(inst[desig[e]].get("query_language")) != "en":
            desig[e] = q
    dq = [desig[e] for e in sorted(desig)]
    ix = {q: i for i, q in enumerate(qs)}
    di = np.array([ix[q] for q in dq])
    dg = g[di]
    d0 = p0[di]
    de = pe[di]
    dbo = d0.argmax(1) == dg
    den = np.array([str(inst[q].get("query_language")) == "en" for q in dq])

    grid = [round(0.05 * i, 2) for i in range(0, 21)]
    out = {"dev": {"n": len(qs), "n_entities": len(set(ent)),
                   "norag": float(bo.mean()),
                   "global_045": counts(loglin(p0, pe, 0.45).argmax(1), g, bp)[0]},
           "arms": {}}
    for name, mq, md in [("en", en, den), ("non_en", ~en, ~den)]:
        qi = np.flatnonzero(mq)
        rows = []
        for a in grid:
            acc, c, h, _ = counts(loglin(p0[qi], pe[qi], a).argmax(1), g[qi], bp[qi])
            ok = loglin(d0[md], de[md], a).argmax(1) == dg[md]
            harm = int((dbo[md] & ~ok).sum())
            bound = cp_upper(harm, int(dbo[md].sum()), 1 - DELTA / (2 * len(grid)))
            rows.append({"alpha": a, "acc": acc, "harm": harm, "bound": bound,
                         "certified": bound <= CAP})
        feas = [r for r in rows if r["certified"]]
        chosen = max(feas, key=lambda r: r["acc"]) if feas else None
        out["arms"][name] = {"rows": rows, "chosen": chosen}

    a_en = out["arms"]["en"]["chosen"]["alpha"]
    a_non = out["arms"]["non_en"]["chosen"]["alpha"]
    out["rule"] = {"alpha_en": a_en, "alpha_non_en": a_non}

    def fused(p0x, pex, lx):
        pred = np.empty(len(p0x), dtype=int)
        m = np.array([l == "en" for l in lx])
        if m.any():
            pred[m] = loglin(p0x[m], pex[m], a_en).argmax(1)
        if (~m).any():
            pred[~m] = loglin(p0x[~m], pex[~m], a_non).argmax(1)
        return pred

    dp = fused(p0, pe, lang)
    out["dev_rule"] = {"acc": counts(dp, g, bp)[0], "corr": counts(dp, g, bp)[1],
                       "harm": counts(dp, g, bp)[2]}

    # ---- E1 440 with 72B ----
    gold = {r["question_id"]: int(r["gold_index"])
            for r in jl(V47 / "e1_new_population" / "e1_gold_sealed.jsonl")}
    b1 = {r["question_id"]: r for r in jl(V47 / "e2_reader_scale" / "e1_base_72b.jsonl")}
    j1 = {r["question_id"]: r for r in jl(V47 / "e2_reader_scale" / "e1_joint_72b.jsonl")}
    q1 = [q for q in gold if q in b1 and q in j1]
    gg = np.array([gold[q] for q in q1])
    P0 = norm(np.array([b1[q]["probabilities"] for q in q1]))
    PE = norm(np.array([j1[q]["probabilities"] for q in q1]))
    LL = [str(b1[q].get("language", "")) for q in q1]
    EE = [str(b1[q]["entity_id"]) for q in q1]
    BB = P0.argmax(1)
    BO = BB == gg
    oks = {}
    rows = {}
    for tag, pred in [
        ("norag", BB),
        ("evidence", PE.argmax(1)),
        ("global045", loglin(P0, PE, 0.45).argmax(1)),
        ("global080", loglin(P0, PE, 0.8).argmax(1)),
        ("langcond", fused(P0, PE, LL)),
    ]:
        acc, c, h, ok = counts(pred, gg, BB)
        oks[tag] = ok
        rows[tag] = {"acc": acc, "corr": c, "harm": h}
    for tag in ["global045", "langcond"]:
        rows[tag]["ci_vs_norag"] = boot(oks[tag], oks["norag"], EE)
        rows[tag]["mcnemar_vs_norag"] = mcnemar(oks[tag], oks["norag"])
    rows["langcond"]["ci_vs_global045"] = boot(oks["langcond"], oks["global045"], EE)
    rows["langcond"]["mcnemar_vs_global045"] = mcnemar(oks["langcond"], oks["global045"])
    out["e1_72b"] = rows

    json.dump(out, open(OUT / "langcond_72b_preview.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    print("chosen en/non:", a_en, a_non)


if __name__ == "__main__":
    main()
