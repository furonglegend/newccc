#!/usr/bin/env python3
"""Base read (no evidence) with the 7B reader over an instances file.

Same prompt/decoding as ea_joint_read / e9_base_read, but the instances file is
given explicitly, so the script also works for the 600-question development
population whose instance file does not live next to the sealed manifest.
Sharded and resumable.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4_probe_logits import Prober  # noqa: E402


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from PIL import Image

    instances = {str(r["question_id"]): r for r in iter_jsonl(args.instances)}
    wanted = sorted(instances)[args.shard::args.num_shards]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {str(r["question_id"]) for r in iter_jsonl(output)} if output.exists() else set()

    prober = Prober(args.model)
    with output.open("a", encoding="utf-8") as writer:
        for number, qid in enumerate(wanted, 1):
            if qid in done:
                continue
            instance = instances[qid]
            path = Path(args.image_root) / f"image_{instance['entity_id']}.png"
            if not path.is_file():
                for ext in (".jpg", ".jpeg", ".webp"):
                    alt = path.with_suffix(ext)
                    if alt.is_file():
                        path = alt
                        break
            if not path.is_file():
                raise FileNotFoundError(str(path))
            image = Image.open(path).convert("RGB")
            question = str(instance["question"])
            options = [str(v) for v in instance["options"]]
            start = time.time()
            result = prober.probe(question, options, image, evidence=None)
            writer.write(json.dumps({
                "question_id": qid,
                "entity_id": str(instance["entity_id"]),
                "language": str(instance.get("query_language", "")),
                "probabilities": [float(v) for v in result["s"]],
                "prediction": int(max(range(4), key=lambda i: result["s"][i])),
                "mode": "base",
                "seconds": float(time.time() - start),
            }, ensure_ascii=False) + "\n")
            writer.flush()
            print(f"[{args.shard}:{number}/{len(wanted)}] {qid} "
                  f"p={['%.3f' % v for v in result['s']]} {result['seconds']:.1f}s", flush=True)
            gc.collect()
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
