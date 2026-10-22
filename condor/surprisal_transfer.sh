#!/bin/bash
set -euo pipefail

root="$HOME/prefill-awareness-cross-20260923"
cd "$root"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python="$HOME/prefill-cross-venv-20260923/bin/python"

case "${1:?pilot or full}" in
  pilot) scope=(--split test --limit 32); out=runs/cross/qwen38-on-qwen32/surprisal-transfer-pilot-test.jsonl ;;
  full) scope=(--split all); out=runs/cross/qwen38-on-qwen32/surprisal-transfer-all.jsonl ;;
  *) echo "Expected pilot or full" >&2; exit 2 ;;
esac

resume=()
if [[ -e "$out" ]]; then resume=(--resume); fi
"$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
timeout --signal=TERM 54000 "$python" -m prefill_awareness.cross_surprisal \
  --adapter runs/cross/qwen38-on-qwen32/adapter \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  "${scope[@]}" --out "$out" "${resume[@]}"
