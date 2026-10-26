#!/usr/bin/env python3
"""Build top-1 retrieval files from a cached retrieval jsonl.

Keeps the first selected passage (the acquisition top-1) and its id, so that
ea_joint_read can run a k=1 joint read with the same prompt format.
"""
from __future__ import annotations

import argparse
import json


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()

    n = 0
    with open(args.dst, "w", encoding="utf-8") as w:
        for r in iter_jsonl(args.src):
            sel = list(r.get("selected") or [])[: args.k]
            ids = list(r.get("selected_ids") or [])[: args.k]
            if not sel:
                raise ValueError(f"empty selection for {r.get('question_id')}")
            out = {k: v for k, v in r.items() if k not in ("selected", "selected_ids")}
            out["selected"] = sel
            out["selected_ids"] = ids
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print("wrote", n, "rows to", args.dst)


if __name__ == "__main__":
    main()
