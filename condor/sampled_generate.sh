#!/bin/bash
set -euo pipefail
# Runs in the checkout it was submitted from, so parallel clones do not collide.
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p data runs
nvidia-smi
# A faulty GPU leaves torch without CUDA, and the model then runs on CPU for
# hours. Fail instead, which holds the job.
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
# Ablation: the datasets of SFT run 2, regenerated with sampling at temperature 1
# from the full distribution (no top-k/top-p) instead of greedy decoding, for the
# clean answer and the re-completion after the plant alike. Every finished turn
# is scored in its generation context and as read in the detection chat.
# Process 0: all brainstorming prompts (as job 17567181). Processes 1-2: the
# other no-context categories (as job 17567560), in two shards.
proc="${1##*.}"
stamp="${1:-manual}-$(date +%s)"
common=(--model Qwen/Qwen3-32B --no-enable-thinking --batch-size 32
        --temperature 1.0 --top-p 1.0 --sample-top-k 0 --max-words 200 --max-new-tokens 512
        --id-prefix "sampled-${1:-manual}-")
if [ "$proc" = 0 ]; then
  timeout --signal=TERM 14400 "$HOME/prefill-venv/bin/python" -m prefill_awareness.data_gen "${common[@]}" \
    --prompts data/dolly-brainstorming-prompts.jsonl --n 1751 \
    --out "data/qwen3-32b-dolly-brainstorming-sampled-$stamp.jsonl"
else
  timeout --signal=TERM 14400 "$HOME/prefill-venv/bin/python" -m prefill_awareness.data_gen "${common[@]}" \
    --prompts data/dolly-rest-prompts.jsonl --n 100000 --min-response-tokens 100 \
    --shards 2 --shard $((proc - 1)) \
    --out "data/qwen3-32b-dolly-rest-sampled-$stamp.jsonl"
fi
