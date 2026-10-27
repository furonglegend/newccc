#!/usr/bin/env bash
# 72B base and joint reads on the 1,600-question confirmation population.
# One process spans the two free GPUs. Resumable and restartable.
set -uo pipefail

ROOT=/97/rctu_v47_ceaplus
CODE=$ROOT/code
RES=$ROOT/results/new
LOGS=$ROOT/logs
MODEL=/97/m4rag_v39_prospective/models/Qwen2.5-VL-72B-Instruct-AWQ
MAN=$ROOT/data/e9_manifest.jsonl
INST=$ROOT/data/e9_questions.jsonl
IMG=$ROOT/data/cvqa_images
RET=$ROOT/results/e9/retrieval_e9.jsonl

export PYTHONPATH=/97/fr/m4rag_v42_locked/runtime_pkgs:/97/m4rag_v39_prospective/torch251_pkgs:/97/m4rag_v39_prospective/python_pkgs
export LD_LIBRARY_PATH=$(find /97/m4rag_v39_prospective/python_pkgs/nvidia -type d -name lib | paste -sd: -)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,3

mark() { echo "$(date -Is) $*" >> "$LOGS/new_72b.log"; }

if [ ! -f "$RES/base72_e9.jsonl" ] || [ "$(wc -l < "$RES/base72_e9.jsonl")" -lt 1600 ]; then
  mark "BASE_START"
  python3 "$CODE/e2_read.py" --base --manifest "$MAN" --instances "$INST" \
    --image-root "$IMG" --model "$MODEL" --output "$RES/base72_e9.jsonl" \
    --shard 0 --num-shards 1 > "$LOGS/new_72b_base.log" 2>&1
  mark "BASE_END n=$(wc -l < "$RES/base72_e9.jsonl" 2>/dev/null || echo 0)"
fi

if [ ! -f "$RES/joint72_e9.jsonl" ] || [ "$(wc -l < "$RES/joint72_e9.jsonl")" -lt 1600 ]; then
  mark "JOINT_START"
  python3 "$CODE/e2_read.py" --retrieved "$RET" --instances "$INST" \
    --image-root "$IMG" --model "$MODEL" --output "$RES/joint72_e9.jsonl" \
    --shard 0 --num-shards 1 > "$LOGS/new_72b_joint.log" 2>&1
  mark "JOINT_END n=$(wc -l < "$RES/joint72_e9.jsonl" 2>/dev/null || echo 0)"
fi

mark "ALL_DONE"
touch "$RES/72B_DONE"
