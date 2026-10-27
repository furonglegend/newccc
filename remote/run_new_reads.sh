#!/usr/bin/env bash
# New reads for the CEA revision: 7B development posteriors (base, k=5, k=1)
# and k=1 joint reads on the 1,600-question confirmation population.
# Resumable: every stage skips finished rows and complete files.
set -uo pipefail

ROOT=/97/rctu_v47_ceaplus
CODE=$ROOT/code/new
RES=$ROOT/results/new
LOGS=$ROOT/logs
IMG=$ROOT/data/cvqa_images
DEV_IMG=$ROOT/results/e2/dev_images
VLM=/97/m4rag_v39_prospective/models/Qwen2.5-VL-7B-Instruct
DEV_INST=/97/m4rag/outputs_v12/natural_locked/dev_instances.jsonl
E9_RET=$ROOT/results/e9/retrieval_e9.jsonl
DEV_RET=$ROOT/results/e2/e2_dev_retrieved.jsonl

export PYTHONPATH=/97/m4rag_v42_locked/runtime_pkgs:/97/m4rag_v39_prospective/torch251_pkgs:/97/m4rag_v39_prospective/python_pkgs
export LD_LIBRARY_PATH=$(find /97/m4rag_v39_prospective/python_pkgs/nvidia -type d -name lib | paste -sd: -)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RES" "$LOGS"
mark() { echo "$(date -Is) $*" >> "$LOGS/new_reads.log"; }

# ---------- top-1 retrieval files ----------
if [ ! -f "$RES/dev_retrieved_k1.jsonl" ]; then
  python3 "$CODE/top1_retrieved.py" --in "$DEV_RET" --out "$RES/dev_retrieved_k1.jsonl" --k 1 \
    > "$LOGS/new_top1_dev.log" 2>&1 || { mark "TOP1_DEV_FAIL"; exit 1; }
fi
if [ ! -f "$RES/e9_retrieved_k1.jsonl" ]; then
  python3 "$CODE/top1_retrieved.py" --in "$E9_RET" --out "$RES/e9_retrieved_k1.jsonl" --k 1 \
    > "$LOGS/new_top1_e9.log" 2>&1 || { mark "TOP1_E9_FAIL"; exit 1; }
fi

run_pair() {
  local tag=$1; shift
  local out=$1; shift
  local want=$1; shift
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -eq "$want" ]; then
    mark "SKIP_${tag} n=$(wc -l < "$out")"
    return 0
  fi
  mark "START_${tag}"
  CUDA_VISIBLE_DEVICES=3 python3 "$@" --output "$out" --shard 0 --num-shards 2 \
    > "$LOGS/new_${tag}_s0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 python3 "$@" --output "$out" --shard 1 --num-shards 2 \
    > "$LOGS/new_${tag}_s1.log" 2>&1 &
  local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  local got=$(wc -l < "$out" 2>/dev/null || echo 0)
  mark "END_${tag} exit=$r0/$r1 n=$got want=$want"
  if [ "$got" -ne "$want" ]; then
    mark "ERROR_${tag} incomplete"
    return 1
  fi
}

# stage A: dev base, 600
run_pair dev_base "$RES/dev_base_7b.jsonl" 600 \
  "$CODE/dev_base_read.py" --instances "$DEV_INST" --image-root "$DEV_IMG" --model "$VLM" \
  || exit 1

# stage B: dev joint k=5, 600
run_pair dev_joint5 "$RES/dev_joint5_7b.jsonl" 600 \
  "$ROOT/code/ea_joint_read.py" --retrieved "$DEV_RET" --instances "$DEV_INST" \
  --image-root "$DEV_IMG" --model "$VLM" \
  || exit 1

# stage C: dev joint k=1, 600
run_pair dev_joint1 "$RES/dev_joint1_7b.jsonl" 600 \
  "$ROOT/code/ea_joint_read.py" --retrieved "$RES/dev_retrieved_k1.jsonl" --instances "$DEV_INST" \
  --image-root "$DEV_IMG" --model "$VLM" \
  || exit 1

# stage D: e9 joint k=1, 1600
run_pair e9_joint1 "$RES/e9_joint1_7b.jsonl" 1600 \
  "$ROOT/code/ea_joint_read.py" --retrieved "$RES/e9_retrieved_k1.jsonl" \
  --instances "$ROOT/data/e9_questions.jsonl" \
  --image-root "$IMG" --model "$VLM" \
  || exit 1

mark "ALL_DONE"
touch "$RES/ALL_DONE"
