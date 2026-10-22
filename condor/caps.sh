#!/bin/bash
# ALL-CAPS generalization test (caps_data / caps_eval): train a caps LoRA on the
# planted-history conversations, then score every checkpoint under planted, own
# and no history. $1 = base | ft (ft merges run 2's detection adapter first).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p runs
nvidia-smi
python="$HOME/prefill-venv/bin/python"
"$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

variant="$1"
data=data/caps-qwen3-32b-dolly.jsonl
init=()
if [ "$variant" = ft ]; then
  # Run 2, epoch 2 (job 17568019) -- lives in the original cluster checkout.
  init=(--init-adapter "$HOME/prefill-awareness/runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306/checkpoint-1108")
fi
out="runs/caps-qwen3-32b-$variant-${2:-manual}-$(date +%s)"
timeout --signal=TERM 5400 "$python" -m prefill_awareness.train \
  --model Qwen/Qwen3-32B --data "$data" --out "$out" "${init[@]}" \
  --gradient-checkpointing --max-steps 80 --save-strategy steps --save-steps 5 \
  --save-only-model --eval-n 0
timeout --signal=TERM 10800 "$python" -m prefill_awareness.caps_eval \
  --model Qwen/Qwen3-32B --run "$out" --data "$data" "${init[@]}" --tag "${2:-manual}"
