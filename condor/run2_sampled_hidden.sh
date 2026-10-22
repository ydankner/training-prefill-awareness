#!/bin/bash
set -euo pipefail
# Runs in the checkout it was submitted from, so parallel clones do not collide.
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p data runs
nvidia-smi
# A faulty GPU leaves torch without CUDA, and the model then runs on CPU for
# hours. Fail instead, which holds the job.
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
# Control for the sampled-data comparison: run 2's model (checkpoint-1108) on the
# SAME sampled test answers, so both runs' plant directions are measured on
# identical text. Otherwise as condor/sampled_hidden.sh: dump, minimal pairs and
# directions in one job on the node's local scratch; only the small outputs are
# copied back, into run 2's directory (its checkpoints are symlinked in this clone).
run=runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306
data=(data/qwen3-32b-dolly-brainstorming-sampled-17607643.0-1790190404.jsonl
      data/qwen3-32b-dolly-rest-sampled-17607643.1-1790190405.000.jsonl
      data/qwen3-32b-dolly-rest-sampled-17607643.2-1790190408.001.jsonl)
tag="${1:-manual}"
scratch="${_CONDOR_SCRATCH_DIR:?run this as a Condor job}/hidden-$tag"
keep="$run/hidden-sampleddata-$tag"
PY="$HOME/prefill-venv/bin/python"
timeout --signal=TERM 3600 "$PY" -m prefill_awareness.hidden_states --model Qwen/Qwen3-32B --run "$run" \
  --checkpoint checkpoint-1108 --data "${data[@]}" --out "$scratch"
timeout --signal=TERM 3600 "$PY" -m prefill_awareness.hidden_states --model Qwen/Qwen3-32B --run "$run" \
  --checkpoint checkpoint-1108 --data "${data[@]}" --out "$scratch" --pairs
# Wrong names come from run 2's evaluation on the sampled test split (job 17608692).
timeout --signal=TERM 7200 "$PY" -m prefill_awareness.directions --dump "$scratch" \
  --eval "$run/eval-checkpoint-1108-test-17608692.0-sampleddata.json" --tokenizer Qwen/Qwen3-32B --tag "$tag"
mkdir -p "$keep"
cp -r "$scratch"/*.json "$scratch"/*.jsonl "$scratch"/*.pt "$scratch"/pairs "$keep"/
for m in base checkpoint-1108; do mkdir -p "$keep/$m" && cp "$scratch/$m/stats.jsonl" "$keep/$m/"; done
du -sh "$scratch" "$keep"
