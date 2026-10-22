#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness-generalization"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

python="$HOME/prefill-venv/bin/python"
run_dir="$HOME/prefill-awareness/runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306"
brain=data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl
rest=data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl
mkdir -p runs/generalization

case "$1" in
  make_cases)
    timeout --signal=TERM 14400 "$python" -m eval_finetuned.make_cases \
      --prompts runs/generalization/prompts.jsonl \
      --out runs/generalization/cases.jsonl --two-n 16
    ;;
  wording|delay)
    timeout --signal=TERM 10800 "$python" -m eval_finetuned.probe \
      --scenario "$1" --dolly-data "$brain" "$rest" \
      --run "$run_dir" --out "runs/generalization/$1"
    ;;
  cases)
    test -s runs/generalization/cases.jsonl
    timeout --signal=TERM 10800 "$python" -m eval_finetuned.probe \
      --scenario cases --cases runs/generalization/cases.jsonl \
      --run "$run_dir" --out runs/generalization/cases-probed
    ;;
  recovery)
    test -s runs/generalization/cases.jsonl
    timeout --signal=TERM 10800 "$python" -m eval_finetuned.probe \
      --scenario recovery --cases runs/generalization/cases.jsonl \
      --dolly-data "$brain" "$rest" --run "$run_dir" \
      --out runs/generalization/recovery
    ;;
  boundary_make)
    timeout --signal=TERM 10800 "$python" -m eval_finetuned.boundary make \
      --dolly-data "$brain" "$rest" --run "$run_dir" \
      --out runs/generalization/boundary-cases.jsonl
    ;;
  boundary_probe)
    test -s runs/generalization/boundary-cases.jsonl
    timeout --signal=TERM 7200 "$python" -m eval_finetuned.boundary probe \
      --cases runs/generalization/boundary-cases.jsonl --run "$run_dir" \
      --out runs/generalization/boundary-probed
    ;;
  *) echo "Unknown scenario: $1" >&2; exit 2 ;;
esac
