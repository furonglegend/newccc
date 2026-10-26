#!/usr/bin/env python3
"""Extension analysis for the CEA revision.

Consumes the new 7B reads produced by code/new/run_new_reads.sh:
  results/new_reads/dev_base_7b.jsonl      600 development base reads
  results/new_reads/dev_joint5_7b.jsonl    600 development joint reads, k=5
  results/new_reads/dev_joint1_7b.jsonl    600 development joint reads, k=1
  results/new_reads/e9_joint1_7b.jsonl     1600 confirmation joint reads, k=1
plus the existing confirmation caches.

Q1  Does a language-conditional arbitration weight, selected on development
    entities under the same entity-level certificate, beat the single frozen
    weight on the confirmation population?
Q2  Does the evidence-set size carry information the weight cannot express, so
    that switching between k=1 and k=5 on a pre-read confidence statistic
    reduces harmful reversals?

Every policy is selected on development data and frozen before confirmation
labels are read.

Usage
-----
    python -P code/cea_extension_analysis.py
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
GLOBAL_ALPHA = 0.45

HERE = Path(__file__).resolve()
REV = HERE.parents[1]
ROOT = HERE.parents[3]
NEW = REV / "results" / "new_reads"
E10 = ROOT / "new" / "v47_results" / "e10_language_arb" / "cache"
V47 = ROOT / "new" / "v47_results"
OUT = REV / "results"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_map(path):
    return {r["question_id"]: r for r in jl(path)}


def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)


def loglin(p0, pe, alpha):
    return norm(np.exp((1 - alpha) * np.log(norm(p0)) + alpha * np.log(norm(pe))))


def cp_upper(k, n, level):
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def mcnemar(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return 1.0, 0, 0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)), b, c


def boot(a_ok, b_ok, clusters, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = sorted(set(clusters.tolist()))
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    n = len(uniq)
    d = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, n, size=n)
        ind = np.concatenate([idx[uniq[j]] for j in pick])
        d[i] = a_ok[ind].mean() - b_ok[ind].mean()
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def counts(pred, gold, base_pred):
    ok = pred == gold
    base_ok = base_pred == gold
    return float(ok.mean()), int(np.sum(~base_ok & ok)), int(np.sum(base_ok & ~ok)), ok


def main():
    rep = {}
    # ---------------- development ----------------
    dev_inst = {str(r["question_id"]): r for r in jl(E10 / "dev_instances.jsonl")}
    d_base = load_map(NEW / "dev_base_7b.jsonl")
    d_j5 = load_map(NEW / "dev_joint5_7b.jsonl")
    qids = [q for q in dev_inst if q in d_base and q in d_j5]
    gold = np.array([int(dev_inst[q]["gold_index"]) for q in qids])
    p0 = norm(np.array([d_base[q]["probabilities"] for q in qids]))
    p5 = norm(np.array([d_j5[q]["probabilities"] for q in qids]))
    ents = [str(dev_inst[q]["entity_id"]) for q in qids]
    langs = [str(dev_inst[q].get("query_language", "")) for q in qids]
    base_pred = p0.argmax(1)
    base_ok = base_pred == gold
    en_mask = np.array([l == "en" for l in langs])

    rep["dev"] = {"n": len(qids), "n_entities": len(set(ents)),
                  "norag": float(base_ok.mean()),
                  "global_045": counts(loglin(p0, p5, GLOBAL_ALPHA).argmax(1), gold, base_pred)[0],
                  "grid5": [{"alpha": a,
                             "acc": counts(loglin(p0, p5, a).argmax(1), gold, base_pred)[0]}
                            for a in [0.35, 0.40, 0.45, 0.50, 0.55]]}

    # one designated question per entity, prefer the -en variant
    designated = {}
    for q in qids:
        e = str(dev_inst[q]["entity_id"])
        if e not in designated:
            designated[e] = q
        elif str(dev_inst[q].get("query_language")) == "en" and \
                str(dev_inst[designated[e]].get("query_language")) != "en":
            designated[e] = q
    dq = [designated[e] for e in sorted(designated)]
    didx = {q: i for i, q in enumerate(qids)}
    di = np.array([didx[q] for q in dq])
    d_gold = gold[di]
    d_p0 = p0[di]
    d_p5 = p5[di]
    d_base_ok = d_p0.argmax(1) == d_gold
    d_en = np.array([str(dev_inst[q].get("query_language")) == "en" for q in dq])

    # ---------------- confirmation population (loaded once) ----------------
    e9 = {r["question_id"]: r for r in jl(V47 / "e9_large_population" / "e9_predictions.jsonl")}
    q9 = sorted(e9)
    g9 = np.array([int(e9[q]["gold_index"]) for q in q9])
    p09 = norm(np.array([e9[q]["p0_posterior"] for q in q9]))
    p59 = norm(np.array([e9[q]["ea_pE_posterior"] for q in q9]))
    l9 = [str(e9[q]["language"]) for q in q9]
    e9_ents = [str(e9[q]["entity_id"]) for q in q9]
    b9 = p09.argmax(1)
    b9_ok = b9 == g9

    # ---------------- Q1 language-conditional weight ----------------
    grid = [round(0.05 * i, 2) for i in range(0, 21)]
    arms = {}
    for name, sel_q, sel_mask in [("en", en_mask, d_en), ("non_en", ~en_mask, ~d_en)]:
        qi = np.flatnonzero(sel_q)
        g = gold[qi]
        b = base_pred[qi]
        rows = []
        for a in grid:
            acc, corr, harm, _ = counts(loglin(p0[qi], p5[qi], a).argmax(1), g, b)
            rows.append({"alpha": a, "acc": acc, "corr": corr, "harm": harm})
        harm_at = {}
        for a in grid:
            ok = loglin(d_p0[sel_mask], d_p5[sel_mask], a).argmax(1) == d_gold[sel_mask]
            harm_at[a] = int((d_base_ok[sel_mask] & ~ok).sum())
        n_arm = int(d_base_ok[sel_mask].sum())
        level = 1 - DELTA / (len(grid) * 2)
        for r in rows:
            r["bound"] = cp_upper(harm_at[r["alpha"]], n_arm, level)
            r["certified"] = r["bound"] <= CAP
        feasible = [r for r in rows if r["certified"]]
        chosen = max(feasible, key=lambda r: r["acc"]) if feasible else None
        arms[name] = {"n_questions": int(sel_q.sum()), "n_at_risk": n_arm,
                      "rows": rows, "chosen": chosen}
    rep["q1"] = {"grid": grid, "arms": arms}

    if arms["en"]["chosen"] and arms["non_en"]["chosen"]:
        a_en = arms["en"]["chosen"]["alpha"]
        a_non = arms["non_en"]["chosen"]["alpha"]
        rep["q1"]["rule"] = {"alpha_en": a_en, "alpha_non_en": a_non}

        def fused(p0x, pex, langx):
            pred = np.empty(len(p0x), dtype=int)
            m_en = np.array([l == "en" for l in langx])
            if m_en.any():
                pred[m_en] = loglin(p0x[m_en], pex[m_en], a_en).argmax(1)
            if (~m_en).any():
                pred[~m_en] = loglin(p0x[~m_en], pex[~m_en], a_non).argmax(1)
            return pred

        d_pred = fused(p0, p5, langs)
        rep["q1"]["dev_rule"] = {
            "acc": counts(d_pred, gold, base_pred)[0],
            "corr": counts(d_pred, gold, base_pred)[1],
            "harm": counts(d_pred, gold, base_pred)[2],
        }

        acc_g, c_g, h_g, ok_g = counts(loglin(p09, p59, GLOBAL_ALPHA).argmax(1), g9, b9)
        pred_l = fused(p09, p59, l9)
        acc_l, c_l, h_l, ok_l = counts(pred_l, g9, b9)
        rep["q1"]["e9"] = {
            "n": len(q9),
            "global": {"acc": acc_g, "corr": c_g, "harm": h_g,
                       "ci_vs_norag": boot(ok_g, b9_ok, e9_ents),
                       "mcnemar_vs_norag": mcnemar(ok_g, b9_ok)},
            "langcond": {"acc": acc_l, "corr": c_l, "harm": h_l,
                         "ci_vs_norag": boot(ok_l, b9_ok, e9_ents),
                         "mcnemar_vs_norag": mcnemar(ok_l, b9_ok),
                         "ci_vs_global": boot(ok_l, ok_g, e9_ents),
                         "mcnemar_vs_global": mcnemar(ok_l, ok_g)},
        }

    # ---------------- Q2 evidence set size ----------------
    rep["q2"] = {"dev": {}, "e9": {}}
    if (NEW / "dev_joint1_7b.jsonl").exists():
        d_j1 = load_map(NEW / "dev_joint1_7b.jsonl")
        q1 = [q for q in qids if q in d_j1]
        idx = [qids.index(q) for q in q1]
        p1 = norm(np.array([d_j1[q]["probabilities"] for q in q1]))
        g1 = gold[idx]
        b1 = base_pred[idx]
        for tag, pe in [("k5", p5[idx]), ("k1", p1)]:
            acc, c, h, _ = counts(loglin(p0[idx], pe, GLOBAL_ALPHA).argmax(1), g1, b1)
            rep["q2"]["dev"][tag] = {"acc": acc, "corr": c, "harm": h}
        if (NEW / "e9_joint1_7b.jsonl").exists():
            j1 = load_map(NEW / "e9_joint1_7b.jsonl")
            have = [q for q in q9 if q in j1]
            ii = [q9.index(q) for q in have]
            p19 = norm(np.array([j1[q]["probabilities"] for q in have]))
            g19 = g9[ii]
            b19 = p09[ii].argmax(1)
            e19 = [e9_ents[i] for i in ii]
            oks = {}
            for tag, pe in [("k5", p59[ii]), ("k1", p19)]:
                acc, c, h, ok = counts(loglin(p09[ii], pe, GLOBAL_ALPHA).argmax(1), g19, b19)
                oks[tag] = ok
                rep["q2"]["e9"][tag] = {"acc": acc, "corr": c, "harm": h,
                                        "ci_vs_norag": boot(ok, b19 == g19, e19)}
            rep["q2"]["e9"]["k1_minus_k5"] = {
                "ci": boot(oks["k1"], oks["k5"], e19),
                "mcnemar": mcnemar(oks["k1"], oks["k5"])}

    json.dump(rep, open(OUT / "extension_analysis.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2)[:5000])


if __name__ == "__main__":
    main()
