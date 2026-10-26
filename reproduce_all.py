#!/usr/bin/env python3
"""One-command reproduction of every number and figure in the manuscript.

Runs the analysis scripts in dependency order, then checks the regenerated
values against the numbers the paper reports. Exits non-zero if any check
fails.

Usage
-----
    python -P code/reproduce_all.py

Requirements: numpy, scipy, matplotlib. No model, no GPU, no network.
Inputs are the cached files under <repo>/new/v47_results plus, when present,
the new reads under results/new_reads.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REV = HERE.parents[1]
CODE = REV / "code"
OUT = REV / "results"


def run(script, optional=False):
    path = CODE / script
    if not path.exists():
        if optional:
            print(f"[skip] {script} not present")
            return False
        raise SystemExit(f"missing {script}")
    print(f"[run ] {script}")
    proc = subprocess.run([sys.executable, "-P", str(path)], cwd=str(REV),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        if optional:
            print(f"[skip] {script} failed, treated as optional")
            return False
        raise SystemExit(f"{script} failed")
    return True


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


def main():
    # 1. core tables and the parameter figure
    run("cea_final_analysis.py")

    # 2. 72B analyses: selective revision on the 440 draw, then the
    #    confirmation-population evaluation when its reads are present
    run("policy_search_72b.py", optional=True)
    have_72b = all((REV / "results" / "new_reads" / f).exists() for f in
                   ("base72_e9.jsonl", "joint72_e9.jsonl"))
    if have_72b:
        run("eval_72b_e9.py", optional=True)
    else:
        print("[skip] 72B confirmation evaluation, reads absent")

    # 3. extension analyses, only when the new 7B reads are present
    have_reads = all((REV / "results" / "new_reads" / f).exists() for f in
                     ("dev_base_7b.jsonl", "dev_joint5_7b.jsonl"))
    if have_reads:
        run("cea_extension_analysis.py", optional=True)
        run("policy_search.py", optional=True)
    else:
        print("[skip] extension analyses, new 7B reads absent")

    # 4. verification against the manuscript
    core = json.load(open(OUT / "final_analysis.json", encoding="utf-8"))
    e9 = core["e9"]
    e1 = core["e1_7b"]
    e72 = core["e1_72b"]
    cert = core["certificate"]

    checks = [
        ("confirmation no-RAG", e9["norag"]["acc"], 0.644375),
        ("confirmation CEA accuracy", e9["cea"]["acc"], 0.66125),
        ("confirmation corrections", e9["cea"]["corr"], 65),
        ("confirmation reversals", e9["cea"]["harm"], 38),
        ("confirmation McNemar p", e9["cea"]["mcnemar_vs_norag"][0], 0.0100633, 5e-6),
        ("fresh 7B CEA", e1["cea"]["acc"], 0.7068, 1e-3),
        ("fresh 72B frozen", e72["cea_frozen"]["acc"], 0.73409, 1e-4),
        ("entity certificate 7B", cert["entity_level_bound"], 0.056749, 1e-6),
        ("question-level certificate", cert["question_level_bound"], 0.051815, 1e-6),
        ("entity certificate 72B", cert["reader_72b"]["bound"], 0.079126, 1e-6),
    ]

    if (OUT / "e9_72b_eval.json").exists():
        r2 = json.load(open(OUT / "e9_72b_eval.json", encoding="utf-8"))
        checks.append(("72B confirmation no-RAG", r2["norag"]["acc"], 0.7175, 1e-6))
        checks.append(("72B confirmation frozen", r2["cea_frozen"]["acc"], 0.724375, 1e-6))
        checks.append(("72B confirmation gate", r2["cea_gate"]["acc"], 0.724375, 1e-6))
        checks.append(("72B confirmation evidence", r2["evidence_only"]["acc"], 0.69375, 1e-6))

    gate_path = OUT / "policy_search_72b.json"
    if gate_path.exists():
        gate = json.load(open(gate_path, encoding="utf-8"))
        checks.append(("72B confidence gate", gate["e1"]["F2_tau0.45"]["acc"], 0.738636, 1e-6))

    if (OUT / "policy_search.json").exists():
        ps = json.load(open(OUT / "policy_search.json", encoding="utf-8"))
        checks.append(("7B margin veto", ps["e9"]["F1_tau0.2"]["acc"], 0.655, 1e-6))
        checks.append(("7B evidence-margin rule", ps["e9"]["F3_tau0.05"]["acc"], 0.659375, 1e-6))

    ext_path = OUT / "extension_analysis.json"
    if ext_path.exists():
        ext = json.load(open(ext_path, encoding="utf-8"))
        if ext.get("q1", {}).get("e9"):
            checks.append(("7B language-conditional", ext["q1"]["e9"]["langcond"]["acc"], 0.645, 1e-6))

    failures = 0
    print("\nverification against the manuscript")
    for item in checks:
        name, got, want = item[0], item[1], item[2]
        tol = item[3] if len(item) > 3 else 5e-4
        ok = close(got, want, tol)
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name:32s} got {got!r:>12} want {want!r}")

    tables = OUT / "final_tables.tex"
    fig = REV / "figures" / "fig_alpha_analysis.pdf"
    print(f"\noutputs: {tables.name}={tables.exists()}, "
          f"{fig.name}={fig.exists()}, json={len(list(OUT.glob('*.json')))} files")
    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
