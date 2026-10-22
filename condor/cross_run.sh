#!/bin/bash
set -euo pipefail

# Run only from the dedicated cross-model checkout; the original checkout and
# environment may be in use by other experiments.
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python="$HOME/prefill-cross-venv-20260923/bin/python"

phase="${1:?phase}"
model_key="${2:?model key}"
source_key="${3:?source key}"
category="${4:-none}"
pilot="${5:-no}"

case "$model_key" in
  qwen32) model="Qwen/Qwen3-32B"; thinking="off"; batch=2; accumulation=8; eval_batch=4 ;;
  qwen38) model="Qwen/Qwen3.8-27B"; thinking="off"; batch=2; accumulation=8; eval_batch=4 ;;
  olmo) model="allenai/Olmo-3-7B-Instruct"; thinking="omit"; batch=2; accumulation=8; eval_batch=4 ;;
  *) echo "Unknown model key: $model_key" >&2; exit 2 ;;
esac
case "$source_key" in
  qwen32)
    data=(data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl
          data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl) ;;
  qwen38|olmo)
    data=("data/cross/$source_key/brainstorming.jsonl"
          "data/cross/$source_key/rest.jsonl") ;;
  *) echo "Unknown source key: $source_key" >&2; exit 2 ;;
esac

case "$pilot" in
  yes) scope=pilot ;;
  pilot4|pilot8) scope="$pilot" ;;
  pilot16) scope=pilot16 ;;
  no) scope=full ;;
  *) echo "Unknown pilot mode: $pilot" >&2; exit 2 ;;
esac

case "$phase" in
  generate)
    [[ "$model_key" == "$source_key" ]] || { echo "Generator/source mismatch" >&2; exit 2; }
    [[ "$source_key" != qwen32 ]] || { echo "Qwen3-32B data already exists" >&2; exit 2; }
    case "$category" in
      brainstorming) prompts=data/dolly-brainstorming-prompts.jsonl; n=1751; extra=() ;;
      rest) prompts=data/dolly-rest-prompts.jsonl; n=100000; extra=(--min-response-tokens 100) ;;
      *) echo "Unknown category: $category" >&2; exit 2 ;;
    esac
    if [[ "$pilot" != no ]]; then n=32; fi
    out="data/cross/$source_key/$scope-$category.jsonl"
    if [[ "$pilot" == no ]]; then out="data/cross/$source_key/$category.jsonl"; fi
    [[ ! -e "$out" ]] || { echo "Output exists; refusing to overwrite: $out" >&2; exit 2; }
    mkdir -p "$(dirname "$out")"
    gpu_batch=16
    if [[ "$pilot" == yes ]]; then gpu_batch=2; fi
    if [[ "$pilot" == pilot4 ]]; then gpu_batch=4; fi
    if [[ "$pilot" == pilot8 ]]; then gpu_batch=8; fi
    if [[ "$pilot" == pilot16 ]]; then gpu_batch=16; fi
    generation_mode=()
    if [[ "$thinking" == off ]]; then generation_mode=(--no-enable-thinking); fi
    "$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
    timeout --signal=TERM 57600 "$python" -m prefill_awareness.data_gen \
      --model "$model" --prompts "$prompts" --out "$out" --n "$n" \
      --batch-size "$gpu_batch" --greedy --max-words 200 --max-new-tokens 512 \
      "${extra[@]}" "${generation_mode[@]}"
    ;;
  train)
    [[ -f "${data[0]}" && -f "${data[1]}" ]] || { echo "Dataset missing for $source_key" >&2; exit 2; }
    for dataset in "${data[@]}"; do
      meta="${dataset%.jsonl}.meta.json"
      [[ -f "$meta" ]] || { echo "Dataset metadata missing: $meta" >&2; exit 2; }
      "$python" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("finished_at"), "dataset generation did not finish"' "$meta"
    done
    out="runs/cross/$model_key-on-$source_key"
    if [[ "$pilot" != no ]]; then out="runs/cross/$scope-$model_key-on-$source_key"; fi
    [[ ! -d "$out/adapter" ]] || { echo "Adapter already exists: $out" >&2; exit 2; }
    mkdir -p "$out"
    resume=()
    latest="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    if [[ -n "$latest" ]]; then resume=(--resume-from "$latest"); fi
    steps=()
    if [[ "$pilot" != no ]]; then steps=(--max-steps 20); fi
    save_steps="${SAVE_STEPS:-100}"
    [[ "$save_steps" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid SAVE_STEPS: $save_steps" >&2; exit 2; }
    "$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
    timeout --signal=TERM 43200 "$python" -m prefill_awareness.train \
      --model "$model" --data "${data[@]}" --out "$out" \
      --thinking-mode "$thinking" --gradient-checkpointing --save-strategy steps --save-steps "$save_steps" \
      --save-total-limit 2 \
      --batch-size "$batch" --grad-accum "$accumulation" \
      --eval-every 0 --eval-n 32 --eval-batch-size "$eval_batch" --surprisal-n 0 \
      "${steps[@]}" "${resume[@]}"
    ;;
  evaluate)
    out="runs/cross/$model_key-on-$source_key"
    [[ -d "$out/adapter" ]] || { echo "Adapter missing: $out" >&2; exit 2; }
    [[ ! -e "$out/eval-test.json" ]] || { echo "Evaluation exists: $out" >&2; exit 2; }
    "$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
    timeout --signal=TERM 14400 "$python" -m prefill_awareness.evaluate \
      --model "$model" --adapter "$out/adapter" --data "${data[@]}" \
      --thinking-mode "$thinking" --split test --n 100000 --batch-size "$eval_batch" \
      --out "$out/eval-test.json"
    ;;
  *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
