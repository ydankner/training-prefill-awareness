#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness-tamper"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python="$HOME/prefill-venv/bin/python"
"$python" -c "import torch; assert torch.cuda.is_available()"
root=runs/tamper-origin
adapter="$HOME/prefill-awareness/runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306/checkpoint-1108"
weak_model="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
mkdir -p "$root"

case "$1" in
  weak)
    test -s "$root/weak-prompts.jsonl"
    test -d "$weak_model"
    timeout --signal=TERM 14400 "$python" -m eval_tamper.sources weak \
      --prompts "$root/weak-prompts.jsonl" --out "$root/weak.jsonl" \
      --model "$weak_model"
    ;;
  prefix)
    test -s "$root/prefix-prompts.jsonl"
    timeout --signal=TERM 14400 "$python" -m eval_tamper.sources prefix \
      --prompts "$root/prefix-prompts.jsonl" --out "$root/prefix.jsonl" \
      --model Qwen/Qwen3-32B
    ;;
  probe_static)
    test -s "$root/static.jsonl"
    timeout --signal=TERM 14400 "$python" -m eval_tamper.probe \
      --cases "$root/static.jsonl" --adapter "$adapter" \
      --out "$root/probe-static-v2"
    ;;
  probe_generated)
    test -s "$root/weak.jsonl"
    test -s "$root/prefix.jsonl"
    timeout --signal=TERM 14400 "$python" -m eval_tamper.probe \
      --cases "$root/weak.jsonl" "$root/prefix.jsonl" --adapter "$adapter" \
      --out "$root/probe-generated-v2"
    ;;
  *) echo "Unknown scenario: $1" >&2; exit 2 ;;
esac
