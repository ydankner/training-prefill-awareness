#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python="$HOME/prefill-cross-venv-20260923/bin/python"

for category in brainstorming rest; do
  data="data/cross/qwen38/$category.jsonl"
  [[ -f "$data" ]] || { echo "Missing dataset: $data" >&2; exit 2; }
  "$python" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("finished_at"), "dataset incomplete"' \
    "${data%.jsonl}.meta.json"
done

out=runs/cross/qwen38-on-qwen38
[[ -d "$out/adapter" ]] || { echo "Missing adapter: $out/adapter" >&2; exit 2; }
[[ ! -e "$out/surprisal-test.jsonl" && ! -e "$out/surprisal-test.meta.json" ]] || {
  echo "Surprisal output already exists: $out" >&2; exit 2;
}
timeout --signal=TERM 43200 "$python" -m prefill_awareness.cross_surprisal \
  --source-model Qwen/Qwen3.8-27B --model Qwen/Qwen3.8-27B \
  --thinking-mode off --adapter "$out/adapter" \
  --data data/cross/qwen38/brainstorming.jsonl data/cross/qwen38/rest.jsonl \
  --split test --out "$out/surprisal-test.jsonl"
