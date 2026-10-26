# CEA manuscript code (WWW 2027 revision)

## One-command reproduction

```
cd 1025/rev
python -P code/reproduce_all.py
```

This runs the analysis scripts in dependency order and then checks fourteen
regenerated values against the numbers printed in the manuscript (confirmation
accuracy, correction and reversal counts, McNemar p, both certificates, the 72B
gate, and the three dev-selected decision rules). It exits non-zero if any value
drifts. Last run: all checks passed.

The `-P` flag matters, because the repository root contains an unrelated
`inspect.py` that would otherwise shadow the standard library module.


This directory holds the Python that reproduces every quantitative table in
`sigconf.tex`. Nothing here calls a model or a retriever. All statistics are
computed from cached option posteriors.

## Files

| File | Purpose |
| --- | --- |
| `cea_final_analysis.py` | Full table suite. Recomputes the confirmation population, the frozen 440-question table, the reader-scale block, the ablation, the weight sweep, the certificate sensitivity study, the language and margin slices, and the parameter-analysis figure. Writes `../results/final_analysis.json`, `../results/final_tables.tex`, and `../results/summary.md`. |
| `e8_entity_cp_corrected.py` | Entity-level Clopper-Pearson certificate with the corrected upper-limit parameterization. Replaces `v47_results/code/e8_entity_cp.py`. |

## Reproducing the numbers

```
cd 1025/rev
python -P code/cea_final_analysis.py
```

The `-P` flag matters. The repository root contains an unrelated `inspect.py`
that shadows the standard library module of the same name.

The script reads from `../../new/v47_results/` and writes to `results/` and
`figures/`. It is deterministic. The entity-clustered bootstrap uses seed
`20261025` and 10,000 draws, so the intervals reproduce bit for bit.

## Inputs

| Input | Rows | Used for |
| --- | --- | --- |
| `e9_large_population/e9_predictions.jsonl` | 1,600 questions, 800 entities | confirmation table, ablation, weight sweep, certificate, language slices |
| `e1_new_population/e1_predictions.jsonl` | 440 questions, 220 entities | frozen evaluation, certificate |
| `e2_reader_scale/e1_base_72b.jsonl`, `e1_joint_72b.jsonl` | 440 questions | reader-scale block |
| `e2_reader_scale/e2_calibration.json` | 600 development questions | 72B development grid |
| `e6_ksweep/e6_eval.json` | 300 questions | retrieval-depth table |
| `offline_variants/e4_fusion_variants.json` | 1,200 questions | fusion-family table |

## Corrections relative to the v47 scripts

1. **Clopper-Pearson upper limit.** `e8_entity_cp.py` used
   `beta.ppf(1 - level, harm, n - harm + 1)`, which is the construction for a
   lower limit and understates the upper bound. The correct one-sided upper
   limit is `beta.ppf(1 - alpha, harm + 1, n - harm)`. On the 220-entity draw
   the two conventions give 0.0337 and 0.0567 once the denominator is the
   number of base-correct units rather than all units.
2. **Denominator of the reversal rate.** The bound is conditioned on a correct
   base answer, so the binomial denominator is the number of designated
   base-correct units, 157, and not the 220 designated entities.
3. **Question-level comparison.** The manuscript now reports 0.0518 for the
   question-level bound on the same draw, computed over the 308 base-correct
   questions with 8 reversals.

The 72B certificate in `e2_calibrate.py` already used the correct
parameterization and the base-correct denominator, so its 0.0750 at
`alpha = 0.8` is unchanged.

## Mapping from tables to code

| Manuscript object | Produced by |
| --- | --- |
| Table 1, frozen evaluation and reader scale | `load_e1_7b`, `load_e1_72b` |
| Table 2, confirmation and published methods | `load_e9`; published baselines are fixed inputs in the tex preamble |
| Table 3, ablation | `rep["ablation"]` |
| Table 4, weight sensitivity | `rep["alpha_grid5"]` |
| Table 5, certificate sensitivity | `rep["certificate"]["grid"]` |
| Table 6, sealed operating points | sealed reference numbers, fixed in the tex |
| Table 7, populations | protocol constants |
| Table 8, retrieval depth | `e6_eval.json` |
| Table 9, fusion families | `e4_fusion_variants.json` |
| Table 10, per-bucket weights | `e5_per_bucket_alpha.json` |
| Table 11, audit tags | `e7_harm_audit.json` |
| Figure 2, parameter analysis | `make_figure` |
| Figure 3, correction cascade | `cea_waterfall.pdf` (sealed analysis) |
| Figure 4, log-odds scatter | `cea_scatter.pdf` (sealed analysis) |

## Experiment-round scripts (2026-10-25 evening)

| File | Purpose |
| --- | --- |
| `cea_extension_analysis.py` | Language-conditional weight and evidence-set-size analysis over the new 7B development and confirmation reads. |
| `policy_search.py` | Dev-selected margin veto, pooled-confidence gate, and evidence-margin rule, each with an entity-level harm bound, transferred to the confirmation population. |
| `policy_search_e1.py` | Same sweep using the 440-question draw as the selection set. |
| `langcond_72b_preview.py` | Language-conditional selection on cached 72B development posteriors, evaluated on the 440-question draw. |
| `remote/` | Scripts uploaded to the GPU server: `dev_base_read.py`, `top1_retrieved.py`, and the resumable driver `run_new_reads.sh`. |

The 72B certificate in the manuscript is produced by `cea_final_analysis.py` on
entity-level units (348 at-risk designated entities, 15 harms at alpha = 0.8,
bound 0.0791).

## Reader-scale experiment (72B on the confirmation population)

| File | Purpose |
| --- | --- |
| `policy_search_72b.py` | Selective revision rules for the 72B reader, selected on its development posteriors and tested once on the 440-question draw. Finds the pooled-confidence gate (tau = 0.45) that lifts 73.41 to 73.86 with 8 corrections and 8 reversals. |
| `eval_72b_e9.py` | Evaluation of the 72B reader on the 1,600-question confirmation population once `base72_e9.jsonl` and `joint72_e9.jsonl` are pulled into `results/new_reads/`. |

The confirmation run is driven by `remote/run_72b_e9.sh` on the GPU server and
writes `results/new/base72_e9.jsonl` and `results/new/joint72_e9.jsonl`.
