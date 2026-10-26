#!/usr/bin/env python3
"""Final analysis suite for the CEA manuscript (WWW 2027 revision).

Every number that appears in the quantitative tables of the revised paper is
recomputed here from cached option posteriors.  The script makes no model calls
and no retrieval calls, so the whole table set is reproducible on a laptop.

Inputs (read only, relative to <repo>/new/v47_results)
-----------------------------------------------------
e9_large_population/e9_predictions.jsonl
    Independent confirmation population, 1,600 questions over 800 entities.
e1_new_population/e1_predictions.jsonl
    Fresh 440-question held-out draw used for the frozen evaluation table.
e2_reader_scale/e1_base_72b.jsonl, e1_joint_72b.jsonl
    Qwen2.5-VL-72B posteriors on the same 440 questions.
e1_new_population/e8_entity_cp.json
    Entity-level certificate record for the 440-question draw.
e2_reader_scale/e2_calibration.json
    Development weight grid under the 72B reader.
e6_ksweep/e6_eval.json
    Retrieval-depth sweep.
offline_variants/e4_fusion_variants.json
    Fusion variants on the sealed 1,200-question population.

Outputs (<repo>/1025/rev)
-------------------------
results/final_analysis.json   machine-readable record of every table
results/final_tables.tex      LaTeX macros consumed by sigconf.tex
results/summary.md            human-readable digest
figures/fig_alpha_analysis.pdf  parameter-analysis figure (two panels)

Usage
-----
    python -P code/cea_final_analysis.py

The -P flag keeps the interpreter from importing modules that sit beside the
manuscript sources (a stray inspect.py in the repository would otherwise
shadow the standard library module of the same name).
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import beta

SEED = 20261025
DRAWS = 10000
FROZEN_ALPHA = 0.45
CAP_EPS = 0.08
DELTA = 0.10
GRID = [round(0.35 + 0.05 * i, 2) for i in range(5)]
FULL_GRID = [round(0.05 * i, 2) for i in range(21)]

# Development calibration under the 7B reader, reported by the sealed
# development run.  The cached 7B development posteriors are no longer on
# disk, so these five values are carried over as fixed inputs rather than
# recomputed.  The 72B development grid is read from e2_calibration.json.
DEV_7B = {0.35: 0.6200, 0.40: 0.6250, 0.45: 0.6333, 0.50: 0.6317, 0.55: 0.6317}

HERE = Path(__file__).resolve()
REV = HERE.parents[1]
ROOT = HERE.parents[3]
V47 = ROOT / "new" / "v47_results"
OUT = REV / "results"
FIG = REV / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# posterior algebra
# --------------------------------------------------------------------------
def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)


def loglin(p0, pe, alpha):
    """Log-linear opinion pool of Eq. (cea-posterior)."""
    return norm(np.exp((1 - alpha) * np.log(norm(p0)) + alpha * np.log(norm(pe))))


def linmix(p0, pe, alpha):
    return norm((1 - alpha) * norm(p0) + alpha * norm(pe))


def post_temp(p0, pe, alpha, temperature):
    return loglin(p0, norm(np.power(norm(pe), 1.0 / temperature)), alpha)


def conf_weighted(p0, pe, gamma):
    c0 = norm(p0).max(1)
    cE = norm(pe).max(1)
    w = np.power(cE, gamma) / (np.power(c0, gamma) + np.power(cE, gamma))
    return norm((1 - w[:, None]) * norm(p0) + w[:, None] * norm(pe)), w


def hard_gate(p0, pe, tau):
    adopt = norm(pe).max(1) > tau
    pred = norm(p0).argmax(1).copy()
    pred[adopt] = norm(pe).argmax(1)[adopt]
    return pred, float(adopt.mean())


def margin_rule(p0, pe, tau):
    adopt = norm(pe).max(1) > (norm(p0).max(1) + tau)
    pred = norm(p0).argmax(1).copy()
    pred[adopt] = norm(pe).argmax(1)[adopt]
    return pred, float(adopt.mean())


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_diff(ok_a, ok_b, clusters, draws=DRAWS, seed=SEED):
    """Entity-clustered paired bootstrap interval for acc(a) - acc(b)."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = sorted(set(clusters.tolist()))
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    n = len(uniq)
    diffs = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([idx_by[uniq[j]] for j in pick])
        diffs[d] = ok_a[idx].mean() - ok_b[idx].mean()
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def mcnemar_exact(ok_a, ok_b):
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n = b + c
    if n == 0:
        return 1.0, 0, 0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail)), b, c


def cp_upper(k_harm, n_units, level):
    """One-sided Clopper-Pearson upper bound for a binomial rate.

    level is the confidence level (1 - alpha).  The bound is the
    Beta(level; k + 1, n - k) quantile, the standard exact construction.
    """
    if n_units <= 0:
        return 1.0
    if k_harm >= n_units:
        return 1.0
    return float(beta.ppf(level, k_harm + 1, n_units - k_harm))


def counts(pred, gold, base_pred):
    ok = pred == gold
    base_ok = base_pred == gold
    corr = int(np.sum(~base_ok & ok))
    harm = int(np.sum(base_ok & ~ok))
    return float(ok.mean()), corr, harm, ok


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_e9():
    rows = jl(V47 / "e9_large_population" / "e9_predictions.jsonl")
    for r in rows:
        r["gold"] = int(r["gold_index"])
    gold = np.array([r["gold"] for r in rows])
    p0 = norm(np.array([r["p0_posterior"] for r in rows]))
    pe = norm(np.array([r["ea_pE_posterior"] for r in rows]))
    ped = norm(np.array([r["dense_pE_posterior"] for r in rows]))
    ents = [r["entity_id"] for r in rows]
    langs = [r["language"] for r in rows]
    return rows, gold, p0, pe, ped, ents, langs


def load_e1_7b():
    rows = jl(V47 / "e1_new_population" / "e1_predictions.jsonl")
    for r in rows:
        r["gold"] = int(r["gold_index"])
    gold = np.array([r["gold"] for r in rows])
    p0 = norm(np.array([r["p0_posterior"] for r in rows]))
    pe = norm(np.array([r["ea_pE_posterior"] for r in rows]))
    ped = norm(np.array([r["dense_pE_posterior"] for r in rows]))
    return gold, p0, pe, ped, [r["entity_id"] for r in rows]


def load_e1_72b():
    gold_map = {r["question_id"]: int(r["gold_index"])
                for r in jl(V47 / "e1_new_population" / "e1_gold_sealed.jsonl")}
    base = {r["question_id"]: np.array(r["probabilities"], dtype=float)
            for r in jl(V47 / "e2_reader_scale" / "e1_base_72b.jsonl")}
    joint = {r["question_id"]: np.array(r["probabilities"], dtype=float)
             for r in jl(V47 / "e2_reader_scale" / "e1_joint_72b.jsonl")}
    qids = [q for q in gold_map if q in base and q in joint]
    gold = np.array([gold_map[q] for q in qids])
    p0 = norm(np.array([base[q] for q in qids]))
    pe = norm(np.array([joint[q] for q in qids]))
    ents = [q.rsplit("-", 1)[0] for q in qids]
    return gold, p0, pe, ents


# --------------------------------------------------------------------------
def main():
    rep = {}

    # ================= e9 confirmation population =================
    rows, gold, p0, pe, ped, ents, langs = load_e9()
    base_pred = p0.argmax(1)
    base_ok = base_pred == gold
    n = len(rows)
    frozen, c_f, h_f, frozen_ok = counts(loglin(p0, pe, FROZEN_ALPHA).argmax(1), gold, base_pred)
    ev_only, c_e, h_e, ev_ok = counts(pe.argmax(1), gold, base_pred)
    dense_pool, c_d, h_d, dense_ok = counts(loglin(p0, ped, FROZEN_ALPHA).argmax(1), gold, base_pred)

    rep["e9"] = {
        "n": n, "n_entities": len(set(ents)), "n_languages": len(set(langs)),
        "norag": {"acc": float(base_ok.mean()), "n_correct": int(base_ok.sum())},
        "evidence_only": {"acc": ev_only, "corr": c_e, "harm": h_e},
        "cea": {"acc": frozen, "corr": c_f, "harm": h_f, "net": c_f - h_f,
                "harm_rate_base_correct": h_f / float(base_ok.sum()),
                "ci_vs_norag": bootstrap_diff(frozen_ok, base_ok, ents),
                "mcnemar_vs_norag": mcnemar_exact(frozen_ok, base_ok)},
        "cea_dense": {"acc": dense_pool, "corr": c_d, "harm": h_d,
                      "ci_vs_norag": bootstrap_diff(dense_ok, base_ok, ents),
                      "mcnemar_vs_norag": mcnemar_exact(dense_ok, base_ok)},
        "evidence_only_ci_vs_norag": bootstrap_diff(ev_ok, base_ok, ents),
    }

    # weight sweep on the confirmation population
    sweep = []
    for a in FULL_GRID:
        acc_a, c_a, h_a, _ = counts(loglin(p0, pe, a).argmax(1), gold, base_pred)
        sweep.append({"alpha": a, "acc": acc_a, "corr": c_a, "harm": h_a})
    rep["alpha_sweep"] = sweep
    grid5 = [s for s in sweep if s["alpha"] in GRID]
    for s in grid5:
        s["dev_acc"] = DEV_7B[s["alpha"]]
    rep["alpha_grid5"] = grid5

    # ablation ladder (single-component removals, everything else frozen)
    gate_pred, gate_adopt = hard_gate(p0, pe, 0.8)
    g_acc, c_g, h_g, g_ok = counts(gate_pred, gold, base_pred)
    lin, c_l, h_l, lin_ok = counts(linmix(p0, pe, 0.5).argmax(1), gold, base_pred)
    rep["ablation"] = [
        {"name": "No-RAG", "acc": float(base_ok.mean()), "corr": 0, "harm": 0,
         "delta_vs_full": float(base_ok.mean()) - frozen},
        {"name": "Dense read, unconditional", "acc": float((ped.argmax(1) == gold).mean()),
         "corr": int(np.sum((~base_ok) & (ped.argmax(1) == gold))),
         "harm": int(np.sum(base_ok & (ped.argmax(1) != gold))),
         "delta_vs_full": float((ped.argmax(1) == gold).mean()) - frozen},
        {"name": "Joint read, unconditional", "acc": ev_only, "corr": c_e, "harm": h_e,
         "delta_vs_full": ev_only - frozen},
        {"name": "Hard acceptance", "acc": g_acc, "corr": c_g, "harm": h_g,
         "delta_vs_full": g_acc - frozen, "adoption": gate_adopt},
        {"name": "Dense acquisition", "acc": dense_pool, "corr": c_d, "harm": h_d,
         "delta_vs_full": dense_pool - frozen},
        {"name": "Linear pooling", "acc": lin, "corr": c_l, "harm": h_l,
         "delta_vs_full": lin - frozen},
        {"name": "Full CEA", "acc": frozen, "corr": c_f, "harm": h_f, "delta_vs_full": 0.0},
    ]

    # fusion families, each at its best setting, against the frozen rule
    def best(points):
        return max(points, key=lambda d: d["acc"])

    fam = []
    lin_pts = [{"alpha": a, "acc": counts(linmix(p0, pe, a).argmax(1), gold, base_pred)[0]}
               for a in FULL_GRID]
    temp_pts = [{"T": T, "acc": counts(post_temp(p0, pe, FROZEN_ALPHA, T).argmax(1), gold, base_pred)[0]}
                for T in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
    cw_pts = []
    for gm in [1.0, 2.0, 4.0]:
        pooled, w = conf_weighted(p0, pe, gm)
        cw_pts.append({"gamma": gm, "acc": counts(pooled.argmax(1), gold, base_pred)[0],
                       "mean_weight": float(w.mean()), "_ok": pooled.argmax(1) == gold})
    gate_pts = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred, adopt = hard_gate(p0, pe, tau)
        gate_pts.append({"tau": tau, "acc": counts(pred, gold, base_pred)[0],
                         "adoption": adopt, "_ok": pred == gold})
    margin_pts = []
    for tau in [0.0, 0.05, 0.1, 0.2, 0.3]:
        pred, adopt = margin_rule(p0, pe, tau)
        margin_pts.append({"tau": tau, "acc": counts(pred, gold, base_pred)[0],
                           "adoption": adopt, "_ok": pred == gold})

    def row(family, setting, acc, ok, **extra):
        d = {"family": family, "setting": setting, "acc": acc, "_ok": ok,
             "corr": int(np.sum(~base_ok & ok)), "harm": int(np.sum(base_ok & ~ok))}
        d.update(extra)
        return d

    fam.append(row("Log-linear pool (frozen)", r"$\alpha=0.45$", frozen, frozen_ok))
    b = best(lin_pts)
    fam.append(row("Linear mixture", rf"$\alpha={b['alpha']:.2f}$", b["acc"],
                   linmix(p0, pe, b["alpha"]).argmax(1) == gold))
    fam.append(row("Fixed log-linear", r"$\alpha=0.50$",
                   counts(loglin(p0, pe, 0.5).argmax(1), gold, base_pred)[0],
                   loglin(p0, pe, 0.5).argmax(1) == gold))
    b = best(temp_pts)
    fam.append(row("Temperature scaling", rf"$T={b['T']:g}$", b["acc"],
                   post_temp(p0, pe, FROZEN_ALPHA, b["T"]).argmax(1) == gold))
    b = best(cw_pts)
    fam.append(row("Confidence weighting", rf"$\gamma={b['gamma']:g}$", b["acc"],
                   b["_ok"], mean_weight=b["mean_weight"]))
    b = best(gate_pts)
    fam.append(row("Hard gate", rf"$\tau={b['tau']:.2f}$", b["acc"], b["_ok"],
                   adoption=b["adoption"]))
    b = best(margin_pts)
    fam.append(row("Margin rule", rf"$\tau={b['tau']:.2f}$", b["acc"], b["_ok"],
                   adoption=b["adoption"]))
    for f in fam:
        ok = f.pop("_ok")
        f["ci_vs_frozen"] = bootstrap_diff(ok, frozen_ok, ents)
        p, bb, ccn = mcnemar_exact(ok, frozen_ok)
        f["mcnemar_vs_frozen"] = {"p": p, "b": bb, "c": ccn}
    rep["fusion_families"] = {
        "rows": fam, "linear_grid": lin_pts,
        "hard_gate_grid": [{"tau": d["tau"], "acc": d["acc"], "adoption": d["adoption"]} for d in gate_pts],
        "margin_grid": [{"tau": d["tau"], "acc": d["acc"], "adoption": d["adoption"]} for d in margin_pts],
    }

    # certificate parameter sensitivity (entity level, 3 harmed of m+ at risk)
    # The binomial denominator is the number of designated BASE-CORRECT units,
    # because the reversal rate is conditioned on a correct base answer.
    manifest = jl(V47 / "e1_new_population" / "e1_manifest.jsonl")
    e1_rows = jl(V47 / "e1_new_population" / "e1_predictions.jsonl")
    by_qid = {r["question_id"]: r for r in e1_rows}
    designated, seen = [], set()
    for r in manifest:
        if r["entity_id"] not in seen:
            seen.add(r["entity_id"])
            designated.append(r["question_id"])
    d_rows = [by_qid[q] for q in designated]
    d_gold = np.array([int(r["gold_index"]) for r in d_rows])
    d_p0 = norm(np.array([r["p0_posterior"] for r in d_rows]))
    d_pe = norm(np.array([r["ea_pE_posterior"] for r in d_rows]))
    d_base_ok = d_p0.argmax(1) == d_gold
    d_ok = loglin(d_p0, d_pe, FROZEN_ALPHA).argmax(1) == d_gold
    m_plus = int(d_base_ok.sum())
    k_harm = int((d_base_ok & ~d_ok).sum())
    q_gold = np.array([int(r["gold_index"]) for r in e1_rows])
    q_p0 = norm(np.array([r["p0_posterior"] for r in e1_rows]))
    q_pe = norm(np.array([r["ea_pE_posterior"] for r in e1_rows]))
    q_base_ok = q_p0.argmax(1) == q_gold
    q_harm = int((q_base_ok & ~(loglin(q_p0, q_pe, FROZEN_ALPHA).argmax(1) == q_gold)).sum())
    q_plus = int(q_base_ok.sum())
    cert = []
    for L in [1, 5, 11, 25]:
        for dlt in [0.05, 0.10, 0.20]:
            ub = cp_upper(k_harm, m_plus, 1 - dlt / L)
            cert.append({"L": L, "delta": dlt, "bound": ub, "passes_cap": ub <= CAP_EPS})
    # 72B certificate on development entities, same designation rule.  The
    # question-level count (17 harms over 402 questions) is not used, because
    # Theorem 1 conditions on entity-level units.
    dev72_inst = {str(r["question_id"]): r for r in
                  jl(ROOT / "new" / "v47_results" / "e10_language_arb" / "cache" / "dev_instances.jsonl")}
    dev72_base = {r["question_id"]: r for r in
                  jl(ROOT / "new" / "v47_results" / "e10_language_arb" / "cache" / "dev_base_72b.jsonl")}
    dev72_joint = {r["question_id"]: r for r in
                   jl(ROOT / "new" / "v47_results" / "e10_language_arb" / "cache" / "dev_joint_72b.jsonl")}
    q72 = [q for q in dev72_inst if q in dev72_base and q in dev72_joint]
    des72 = {}
    for q in q72:
        e = str(dev72_inst[q]["entity_id"])
        if e not in des72:
            des72[e] = q
        elif str(dev72_inst[q].get("query_language")) == "en" and \
                str(dev72_inst[des72[e]].get("query_language")) != "en":
            des72[e] = q
    dq72 = [des72[e] for e in sorted(des72)]
    g72 = np.array([int(dev72_inst[q]["gold_index"]) for q in dq72])
    p0_72 = norm(np.array([dev72_base[q]["probabilities"] for q in dq72]))
    pe_72 = norm(np.array([dev72_joint[q]["probabilities"] for q in dq72]))
    ok_base_72 = p0_72.argmax(1) == g72
    ok_80 = loglin(p0_72, pe_72, 0.8).argmax(1) == g72
    harm_72 = int((ok_base_72 & ~ok_80).sum())
    risk_72 = int(ok_base_72.sum())

    rep["certificate"] = {
        "n_entities": len(designated), "n_at_risk": m_plus, "n_harm": k_harm,
        "entity_level_bound": cp_upper(k_harm, m_plus, 1 - DELTA / 5),
        "question_level_bound": cp_upper(q_harm, q_plus, 1 - DELTA / 5),
        "grid": cert,
        "reader_72b": {"alpha": 0.8, "n_entities": len(dq72), "n_at_risk": risk_72,
                       "n_harm": harm_72,
                       "bound": cp_upper(harm_72, risk_72, 1 - DELTA / 19),
                       "cap": CAP_EPS},
    }

    # language and margin slices
    slices = {}
    for name, mask in [("en", np.array([l == "en" for l in langs])),
                       ("non_en", np.array([l != "en" for l in langs]))]:
        idx = np.flatnonzero(mask)
        b2, g2 = base_pred[idx], gold[idx]
        l2 = loglin(p0[idx], pe[idx], FROZEN_ALPHA).argmax(1)
        slices[name] = {"n": int(len(idx)), "norag": float((b2 == g2).mean()),
                        "cea": float((l2 == g2).mean()),
                        "corr": int(np.sum((b2 != g2) & (l2 == g2))),
                        "harm": int(np.sum((b2 == g2) & (l2 != g2)))}
    p0s = np.sort(p0, axis=1)
    margin = p0s[:, -1] - p0s[:, -2]
    bucket = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    mb = []
    for bi in range(3):
        idx = np.flatnonzero(bucket == bi)
        accs = [(a, counts(loglin(p0[idx], pe[idx], a).argmax(1), gold[idx], base_pred[idx])[0])
                for a in FULL_GRID]
        ba = max(accs, key=lambda t: t[1])[0]
        mb.append({"bucket": bi, "n": int(len(idx)),
                   "norag": float((base_pred[idx] == gold[idx]).mean()),
                   "frozen": float((loglin(p0[idx], pe[idx], FROZEN_ALPHA).argmax(1) == gold[idx]).mean()),
                   "best_alpha": ba, "best": float(dict(accs)[ba])})
    lang_macro = {}
    for nm in ["norag", "cea"]:
        vals = []
        for lg in sorted(set(langs)):
            idx = np.flatnonzero(np.array([l == lg for l in langs]))
            pred = base_pred[idx] if nm == "norag" else loglin(p0[idx], pe[idx], FROZEN_ALPHA).argmax(1)
            vals.append(float((pred == gold[idx]).mean()))
        lang_macro[nm] = float(np.mean(vals))
    rep["slices"] = {"language": slices, "language_macro": lang_macro, "margin_buckets": mb}

    # ================= e1 frozen table, 7B and 72B =================
    g1, p01, pe1, ped1, ent1 = load_e1_7b()
    b1_ok = p01.argmax(1) == g1
    cea1, cc1, hh1, ok1 = counts(loglin(p01, pe1, FROZEN_ALPHA).argmax(1), g1, p01.argmax(1))
    dns1, cd1, hd1, okd1 = counts(loglin(p01, ped1, FROZEN_ALPHA).argmax(1), g1, p01.argmax(1))
    ev1, ce1c, he1, oke1 = counts(pe1.argmax(1), g1, p01.argmax(1))
    rep["e1_7b"] = {"n": len(g1), "norag": float(b1_ok.mean()),
                    "evidence_only": {"acc": ev1, "corr": ce1c, "harm": he1},
                    "cea": {"acc": cea1, "corr": cc1, "harm": hh1,
                            "ci_vs_norag": bootstrap_diff(ok1, b1_ok, ent1),
                            "mcnemar_vs_norag": mcnemar_exact(ok1, b1_ok)},
                    "cea_dense": {"acc": dns1, "corr": cd1, "harm": hd1}}

    g2, p02, pe2, ent2 = load_e1_72b()
    b2_ok = p02.argmax(1) == g2
    ev2, ce2, he2, oke2 = counts(pe2.argmax(1), g2, p02.argmax(1))
    c45, cc45, hh45, ok45 = counts(loglin(p02, pe2, 0.45).argmax(1), g2, p02.argmax(1))
    c80, cc80, hh80, ok80 = counts(loglin(p02, pe2, 0.8).argmax(1), g2, p02.argmax(1))
    rep["e1_72b"] = {
        "n": len(g2),
        "norag": float(b2_ok.mean()),
        "evidence_only": {"acc": ev2, "corr": ce2, "harm": he2,
                          "ci_vs_norag": bootstrap_diff(oke2, b2_ok, ent2)},
        "cea_frozen": {"acc": c45, "corr": cc45, "harm": hh45,
                       "ci_vs_norag": bootstrap_diff(ok45, b2_ok, ent2),
                       "mcnemar_vs_norag": mcnemar_exact(ok45, b2_ok),
                       "ci_vs_evidence": bootstrap_diff(ok45, oke2, ent2),
                       "mcnemar_vs_evidence": mcnemar_exact(ok45, oke2)},
        "cea_recal": {"acc": c80, "corr": cc80, "harm": hh80,
                      "ci_vs_norag": bootstrap_diff(ok80, b2_ok, ent2),
                      "mcnemar_vs_norag": mcnemar_exact(ok80, b2_ok)},
    }

    # 72B development grid and the reader-conditional optimum
    e2 = json.load(open(V47 / "e2_reader_scale" / "e2_calibration.json", encoding="utf-8"))
    n_correct_72 = int(round(e2["dev_grid"][0]["dev_acc"] * 600))
    dev72 = []
    for r in e2["dev_grid"]:
        ub = cp_upper(r["harms"], n_correct_72, 1 - DELTA / len(e2["dev_grid"]))
        dev72.append({"alpha": r["alpha"], "dev_acc": r["dev_acc"], "harms": r["harms"],
                      "bound": ub, "certified": ub <= CAP_EPS})
    passing = [r for r in dev72 if r["certified"]]
    chosen = max(passing, key=lambda r: r["dev_acc"])
    rep["dev_grids"] = {"reader_7b": [{"alpha": a, "dev_acc": DEV_7B[a]} for a in sorted(DEV_7B)],
                        "reader_72b": dev72, "chosen_72b": chosen,
                        "norag_72b_dev": e2["dev_grid"][0]["dev_acc"]}

    # ================= figure and outputs =================
    make_figure(rep)
    json.dump(rep, open(OUT / "final_analysis.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    write_tex(rep)
    write_summary(rep)
    print(f"e9 CEA {frozen*100:.2f} ({c_f}/{h_f}) vs No-RAG {base_ok.mean()*100:.2f}"
          f" CI {rep['e9']['cea']['ci_vs_norag']}")
    print(f"e1 7B CEA {cea1*100:.2f} ({cc1}/{hh1}) vs {b1_ok.mean()*100:.2f}")
    print(f"e1 72B frozen {c45*100:.2f} ({cc45}/{hh45}) vs No-RAG {b2_ok.mean()*100:.2f}"
          f" vs evidence {ev2*100:.2f}")
    print(f"wrote {OUT/'final_analysis.json'}")


def make_figure(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))

    sw = rep["alpha_sweep"]
    a = [s["alpha"] for s in sw]
    acc = [s["acc"] * 100 for s in sw]
    harm = [s["harm"] for s in sw]
    ax = axes[0]
    ax.plot(a, acc, color="#1f4e79", lw=1.5, marker="o", ms=2.6, label="Accuracy")
    ax.set_xlabel(r"arbitration weight $\alpha$")
    ax.set_ylabel("accuracy on confirmation set")
    ax.set_ylim(min(acc) - 0.35, max(acc) + 0.75)
    ax.axvline(FROZEN_ALPHA, color="#888888", lw=0.8, ls="--")
    ax.annotate("frozen", xy=(FROZEN_ALPHA, max(acc) + 0.45), fontsize=6.5,
                color="#555555", ha="center")
    ax2 = ax.twinx()
    ax2.bar(a, harm, width=0.03, color="#c9a227", alpha=0.30, label="Harmful reversals")
    ax2.set_ylabel("harmful reversals", color="#8a6d1f")
    ax2.tick_params(axis="y", colors="#8a6d1f")
    ax.grid(axis="y", lw=0.3, color="#dddddd")

    ax = axes[1]
    d7 = rep["dev_grids"]["reader_7b"]
    d72 = rep["dev_grids"]["reader_72b"]
    ax.plot([d["alpha"] for d in d7], [d["dev_acc"] * 100 for d in d7], marker="s", ms=3,
            lw=1.3, color="#1f4e79", label="7B reader")
    ax.plot([d["alpha"] for d in d72], [d["dev_acc"] * 100 for d in d72], marker="o", ms=2.4,
            lw=1.3, color="#a33a2f", label="72B reader")
    ax.axvline(0.45, color="#1f4e79", lw=0.7, ls=":")
    ax.axvline(0.80, color="#a33a2f", lw=0.7, ls=":")
    ax.set_xlabel(r"arbitration weight $\alpha$")
    ax.set_ylabel("development accuracy")
    ax.set_ylim(61.3, 71.6)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_alpha_analysis.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_alpha_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd=2):
    """Percent formatting with round-half-up, so 66.125 prints as 66.13."""
    from decimal import Decimal, ROUND_HALF_UP

    q = Decimal("0.01") if nd == 2 else Decimal("0.1")
    return str(Decimal(repr(round(x * 100.0, 6))).quantize(q, rounding=ROUND_HALF_UP))


def write_tex(rep):
    L = []
    e9, e1, e72 = rep["e9"], rep["e1_7b"], rep["e1_72b"]
    add = L.append
    add(f"\\newcommand{{\\vConfN}}{{{e9['n']}}}")
    add(f"\\newcommand{{\\vConfEnts}}{{{e9['n_entities']}}}")
    add(f"\\newcommand{{\\vConfLangs}}{{{e9['n_languages']}}}")
    add(f"\\newcommand{{\\vConfBase}}{{{fmt(e9['norag']['acc'])}}}")
    add(f"\\newcommand{{\\vConfAcc}}{{{fmt(e9['cea']['acc'])}}}")
    add(f"\\newcommand{{\\vConfCorr}}{{{e9['cea']['corr']}}}")
    add(f"\\newcommand{{\\vConfHarm}}{{{e9['cea']['harm']}}}")
    add(f"\\newcommand{{\\vConfNet}}{{{e9['cea']['net']}}}")
    ci = e9["cea"]["ci_vs_norag"]
    add(f"\\newcommand{{\\vConfCI}}{{[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]}}")
    add(f"\\newcommand{{\\vConfP}}{{{e9['cea']['mcnemar_vs_norag'][0]:.3f}}}")
    add(f"\\newcommand{{\\vConfEvAcc}}{{{fmt(e9['evidence_only']['acc'])}}}")
    add(f"\\newcommand{{\\vConfEvCorr}}{{{e9['evidence_only']['corr']}}}")
    add(f"\\newcommand{{\\vConfEvHarm}}{{{e9['evidence_only']['harm']}}}")
    add(f"\\newcommand{{\\vConfDenseAcc}}{{{fmt(e9['cea_dense']['acc'])}}}")
    add(f"\\newcommand{{\\vConfDenseCorr}}{{{e9['cea_dense']['corr']}}}")
    add(f"\\newcommand{{\\vConfDenseHarm}}{{{e9['cea_dense']['harm']}}}")
    add(f"\\newcommand{{\\vConfHarmRate}}{{{fmt(e9['cea']['harm_rate_base_correct'])}}}")
    add(f"\\newcommand{{\\vConfMacroBase}}{{{fmt(rep['slices']['language_macro']['norag'])}}}")
    add(f"\\newcommand{{\\vConfMacroCea}}{{{fmt(rep['slices']['language_macro']['cea'])}}}")
    add(f"\\newcommand{{\\vConfEnBase}}{{{fmt(rep['slices']['language']['en']['norag'])}}}")
    add(f"\\newcommand{{\\vConfEnCea}}{{{fmt(rep['slices']['language']['en']['cea'])}}}")
    add(f"\\newcommand{{\\vConfNonEnBase}}{{{fmt(rep['slices']['language']['non_en']['norag'])}}}")
    add(f"\\newcommand{{\\vConfNonEnCea}}{{{fmt(rep['slices']['language']['non_en']['cea'])}}}")
    add(f"\\newcommand{{\\vConfEnN}}{{{rep['slices']['language']['en']['n']}}}")
    add(f"\\newcommand{{\\vConfNonEnN}}{{{rep['slices']['language']['non_en']['n']}}}")
    add(f"\\newcommand{{\\vConfDenseUncond}}{{{fmt(rep['ablation'][1]['acc'])}}}")
    add(f"\\newcommand{{\\vConfEvGain}}{{{(e9['evidence_only']['acc']-e9['norag']['acc'])*100:+.2f}}}")
    add(f"\\newcommand{{\\vConfCeaGain}}{{{(e9['cea']['acc']-e9['norag']['acc'])*100:+.2f}}}")
    add(f"\\newcommand{{\\vConfDenseUncondGain}}{{{(rep['ablation'][1]['acc']-e9['norag']['acc'])*100:+.2f}}}")

    add(f"\\newcommand{{\\vEoneN}}{{{e1['n']}}}")
    add(f"\\newcommand{{\\vEoneNoRAG}}{{{fmt(e1['norag'])}}}")
    add(f"\\newcommand{{\\vESevenBAcc}}{{{fmt(e1['cea']['acc'])}}}")
    add(f"\\newcommand{{\\vEoneCorr}}{{{e1['cea']['corr']}}}")
    add(f"\\newcommand{{\\vEoneHarm}}{{{e1['cea']['harm']}}}")
    ci1 = e1["cea"]["ci_vs_norag"]
    add(f"\\newcommand{{\\vEoneCI}}{{[{ci1[0]*100:+.2f},{ci1[1]*100:+.2f}]}}")
    add(f"\\newcommand{{\\vEoneDense}}{{{fmt(e1['evidence_only']['acc'])}}}")
    add(f"\\newcommand{{\\vEoneDensePool}}{{{fmt(e1['cea_dense']['acc'])}}}")

    add(f"\\newcommand{{\\vEtwoNoRAG}}{{{fmt(e72['norag'])}}}")
    add(f"\\newcommand{{\\vEtwoEv}}{{{fmt(e72['evidence_only']['acc'])}}}")
    add(f"\\newcommand{{\\vEtwoFrozen}}{{{fmt(e72['cea_frozen']['acc'])}}}")
    add(f"\\newcommand{{\\vEtwoFrozenCorr}}{{{e72['cea_frozen']['corr']}}}")
    add(f"\\newcommand{{\\vEtwoFrozenHarm}}{{{e72['cea_frozen']['harm']}}}")
    ci72 = e72["cea_frozen"]["ci_vs_norag"]
    add(f"\\newcommand{{\\vEtwoCI}}{{[{ci72[0]*100:+.2f},{ci72[1]*100:+.2f}]}}")
    add(f"\\newcommand{{\\vEtwoP}}{{{e72['cea_frozen']['mcnemar_vs_norag'][0]:.2f}}}")
    ci72e = e72["cea_frozen"]["ci_vs_evidence"]
    add(f"\\newcommand{{\\vEtwoEvCI}}{{[{ci72e[0]*100:+.2f},{ci72e[1]*100:+.2f}]}}")
    add(f"\\newcommand{{\\vEtwoEvP}}{{{e72['cea_frozen']['mcnemar_vs_evidence'][0]:.3f}}}")
    add(f"\\newcommand{{\\vEtwoAlpha}}{{{rep['dev_grids']['chosen_72b']['alpha']}}}")
    add(f"\\newcommand{{\\vEtwoAcc}}{{{fmt(e72['cea_recal']['acc'])}}}")
    add(f"\\newcommand{{\\vEtwoRecCorr}}{{{e72['cea_recal']['corr']}}}")
    add(f"\\newcommand{{\\vEtwoRecHarm}}{{{e72['cea_recal']['harm']}}}")
    add(f"\\newcommand{{\\vEtwoCert}}{{{rep['certificate']['reader_72b']['bound']:.4f}}}")
    add(f"\\newcommand{{\\vEtwoAtRisk}}{{{rep['certificate']['reader_72b']['n_at_risk']}}}")
    add(f"\\newcommand{{\\vEtwoCrossHarm}}{{{rep['certificate']['reader_72b']['n_harm']}}}")

    add(f"\\newcommand{{\\vCpEntity}}{{{rep['certificate']['entity_level_bound']:.4f}}}")
    add(f"\\newcommand{{\\vCertQLevel}}{{{rep['certificate']['question_level_bound']:.4f}}}")
    add(f"\\newcommand{{\\vCertRisk}}{{{rep['certificate']['n_at_risk']}}}")
    add(f"\\newcommand{{\\vCertHarm}}{{{rep['certificate']['n_harm']}}}")
    add(f"\\newcommand{{\\vCertMax}}{{{max(g['bound'] for g in rep['certificate']['grid']):.4f}}}")

    add("%% ablation rows")
    add("\\newcommand{\\vAblationRows}{%")
    for r in rep["ablation"]:
        add(f"{r['name']} & {fmt(r['acc'])} & {r['corr']} & {r['harm']} \\\\")
    add("}")
    add("%% weight sensitivity rows")
    add("\\newcommand{\\vAlphaRows}{%")
    for s in rep["alpha_grid5"]:
        star = "$^{\\ast}$" if abs(s["alpha"] - FROZEN_ALPHA) < 1e-9 else ""
        add(f"{s['alpha']:.2f}{star} & {fmt(s['dev_acc'])} & {fmt(s['acc'])} & {s['corr']} & {s['harm']} \\\\")
    add("}")
    add("%% certificate parameter rows")
    cert_names = {1: "One", 5: "Five", 11: "Eleven", 25: "TwentyFive"}
    for Ls in [1, 5, 11, 25]:
        row = [g for g in rep["certificate"]["grid"] if g["L"] == Ls]
        cells = " & ".join(f"{g['bound']:.4f}" for g in row)
        add(f"\\newcommand{{\\vCertRow{cert_names[Ls]}}}{{{Ls} & {cells} \\\\}}")
    add("%% fusion family rows")
    add("\\newcommand{\\vFamilyRows}{%")
    for f in rep["fusion_families"]["rows"]:
        c = f["ci_vs_frozen"]
        add(f"{f['family']} & {f['setting']} & {fmt(f['acc'])} & "
            f"$[{c[0]*100:+.2f},{c[1]*100:+.2f}]$ \\\\")
    add("}")
    open(OUT / "final_tables.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_summary(rep):
    e9 = rep["e9"]
    lines = ["# Final analysis digest", "",
             f"Confirmation population: {e9['n']} questions, {e9['n_entities']} entities, "
             f"{e9['n_languages']} languages",
             f"No-RAG {e9['norag']['acc']*100:.2f}",
             f"Evidence only {e9['evidence_only']['acc']*100:.2f} "
             f"({e9['evidence_only']['corr']}/{e9['evidence_only']['harm']})",
             f"CEA {e9['cea']['acc']*100:.2f} ({e9['cea']['corr']}/{e9['cea']['harm']}) "
             f"CI {e9['cea']['ci_vs_norag']} p={e9['cea']['mcnemar_vs_norag'][0]:.3f}",
             "", "## Ablation", ""]
    for r in rep["ablation"]:
        lines.append(f"- {r['name']}: {r['acc']*100:.2f} ({r['corr']}/{r['harm']})")
    lines += ["", "## Weight sensitivity", ""]
    for s in rep["alpha_grid5"]:
        lines.append(f"- {s['alpha']:.2f}: dev {s['dev_acc']*100:.2f} | test {s['acc']*100:.2f} "
                     f"({s['corr']}/{s['harm']})")
    lines += ["", "## Fusion families", ""]
    for f in rep["fusion_families"]["rows"]:
        lines.append(f"- {f['family']} {f['setting']}: {f['acc']*100:.2f} "
                     f"({f['corr']}/{f['harm']}) CI {f['ci_vs_frozen']}")
    open(OUT / "summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
