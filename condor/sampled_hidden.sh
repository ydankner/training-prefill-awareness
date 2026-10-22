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
# Mechanistic analysis of the sampled-data run, as for run 2 (hidden states
# 17569245, minimal pairs 17569542, directions 17569599), in one job: the
# residual stream at every layer over the tampered turn of every test record,
# for the base model and checkpoint-1108 (~135 GB each), the minimal pairs at the
# plant, then the plant directions. The ~270 GB dump does not fit the home quota
# and the shared fast area writes it far too slowly, so it goes to the node's
# local scratch and only the small outputs (index, per-token stats, pairs,
# directions) are copied back into the run directory.
run=(runs/sft-qwen3-32b-dolly-all-sampled-*)
[ "${#run[@]}" = 1 ] || { echo "expected one sampled run, got ${run[*]}"; exit 1; }
data=(data/qwen3-32b-dolly-brainstorming-sampled-17607643.0-1790190404.jsonl
      data/qwen3-32b-dolly-rest-sampled-17607643.1-1790190405.000.jsonl
      data/qwen3-32b-dolly-rest-sampled-17607643.2-1790190408.001.jsonl)
tag="${1:-manual}"
scratch="${_CONDOR_SCRATCH_DIR:?run this as a Condor job}/hidden-$tag"
keep="${run[0]}/hidden-$tag"
PY="$HOME/prefill-venv/bin/python"
timeout --signal=TERM 3600 "$PY" -m prefill_awareness.hidden_states --model Qwen/Qwen3-32B --run "${run[0]}" \
  --checkpoint checkpoint-1108 --data "${data[@]}" --out "$scratch"
timeout --signal=TERM 3600 "$PY" -m prefill_awareness.hidden_states --model Qwen/Qwen3-32B --run "${run[0]}" \
  --checkpoint checkpoint-1108 --data "${data[@]}" --out "$scratch" --pairs
timeout --signal=TERM 7200 "$PY" -m prefill_awareness.directions --dump "$scratch" \
  --eval "${run[0]}/eval-checkpoint-1108-test-17608692.0.json" --tokenizer Qwen/Qwen3-32B --tag "$tag"
mkdir -p "$keep"
cp -r "$scratch"/*.json "$scratch"/*.jsonl "$scratch"/*.pt "$scratch"/pairs "$keep"/
for m in base checkpoint-1108; do mkdir -p "$keep/$m" && cp "$scratch/$m/stats.jsonl" "$keep/$m/"; done
du -sh "$scratch" "$keep"
