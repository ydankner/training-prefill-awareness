#!/bin/bash
# Tampering-marker experiment (Qwen3.8-27B). Runs in the checkout it is
# submitted from, never in ~/prefill-awareness. Phases:
#   pilot          64 prompts: histories + conversations, to time and check both
#   histories      every Dolly no-context prompt (resumable: rerun to continue)
#   conversations  pair histories with No Robots follow-ups, generate own answers
#   train          LoRA SFT with full test eval every 100 steps
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# FLA=1 puts flash-linear-attention (~/tampermark-pylib, pip --target) on the
# path for Qwen3.8's linear-attention layers. The compute nodes have gcc but no
# linker, which Triton needs to build its launchers, so ~/tampermark-tools holds
# the login node's ld (Binutils 2.38, Ubuntu 22.04 like the nodes) and its two
# libraries. Without FLA the reference PyTorch implementation is used.
export PYTHONPATH="$root/src" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ -n "${FLA:-}" ]]; then
  export PYTHONPATH="$PYTHONPATH:$HOME/tampermark-pylib"
  export PATH="$HOME/tampermark-tools/bin:$PATH"
  export LD_LIBRARY_PATH="$HOME/tampermark-tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TRITON_CACHE_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/triton-cache"
fi
python="$HOME/prefill-cross-venv-20260923/bin/python"
model="Qwen/Qwen3.8-27B"
data=data/tamper-qwen38
mkdir -p "$data" logs runs
nvidia-smi --query-gpu=name,memory.total --format=csv
"$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

prompts=(data/dolly-brainstorming-prompts.jsonl:brainstorming data/dolly-rest-prompts.jsonl:rest)
shopt -s nullglob
all_histories=("$data/histories.jsonl" "$data"/histories.block-*.jsonl)
case "${1:?phase}" in
  pilot)
    "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${prompts[@]}" \
      --out "$data/pilot-histories.jsonl" --n 64 --batch-size "${BATCH:-32}"
    "$python" -m prefill_awareness.tamper_data --histories "$data/pilot-histories.jsonl" \
      --followups data/no-robots-prompts.jsonl --out "$data/pilot-conversations.jsonl" --batch-size "${BATCH:-32}"
    ;;
  histories)
    "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${prompts[@]}" \
      --out "$data/histories.jsonl" --batch-size "${BATCH:-32}"
    ;;
  fla_check)  # kernels compile, scores match the reference path, and the pilot's 64 prompts timed
    "$python" condor/fla_check.py "$data/pilot-histories.jsonl"
    rm -f "$data/pilot-fla-histories.jsonl" "$data/pilot-fla-histories.meta.json"
    "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${prompts[@]}" \
      --out "$data/pilot-fla-histories.jsonl" --n 64 --batch-size "${BATCH:-32}"
    ;;
  histories_block)  # histories_block <k> <from> <to>: prompt positions [from, to) into histories.block-k
    "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${prompts[@]}" \
      --out "$data/histories.block-${2:?k}.jsonl" --exclude "$data/histories.jsonl" \
      --from-index "${3:?from}" --to-index "${4:?to}" --batch-size "${BATCH:-32}"
    ;;
  conversations)
    "$python" -m prefill_awareness.tamper_data --histories "${all_histories[@]}" \
      --followups data/no-robots-prompts.jsonl --out "$data/conversations.jsonl" --batch-size "${BATCH:-32}"
    ;;
  train)
    out="runs/tamper-qwen38-${2:?run name}"
    resume=()
    # || true: before the first run $out does not exist, and pipefail would end the script
    latest="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$latest" ]] && resume=(--resume-from "$latest")
    "$python" -m prefill_awareness.train --task tamper --model "$model" --data "$data/conversations.jsonl" \
      --out "$out" --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 \
      --epochs 2 --lr 1e-4 --eval-every 100 --eval-n 0 --eval-batch-size 16 \
      --save-strategy steps --save-steps 100 "${resume[@]}" ${EXTRA:-}
    ;;
  train_abl)  # train_abl <nomarker|nolist>: the main training on tamper_ablate data (same schedule)
    abl="${2:?nomarker|nolist}"
    [[ -f "$data/conversations-$abl.jsonl" ]] || "$python" -m prefill_awareness.tamper_ablate --mode "$abl" \
      --data "$data/conversations.jsonl" --out "$data/conversations-$abl.jsonl"
    out="runs/tamper-qwen38-$abl"
    flags=()
    [[ "$abl" == nomarker ]] && flags=(--no-marker)
    resume=()
    latest="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$latest" ]] && resume=(--resume-from "$latest")
    "$python" -m prefill_awareness.train --task tamper --model "$model" --data "$data/conversations-$abl.jsonl" \
      --out "$out" --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 \
      --epochs 2 --lr 1e-4 --eval-every 100 --eval-n 0 --eval-batch-size 16 \
      --save-strategy steps --save-steps 100 "${flags[@]}" "${resume[@]}"
    ;;
  caps_nm)  # caps_nm <detection run name>: the main caps run, marker-free targets, on that run's adapter
    out="runs/tamper-caps-nm-qwen38-${2:?run}"
    init=(--init-adapter "runs/tamper-qwen38-$2/adapter")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps-nomarker.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 80 --warmup-steps 0 --save-strategy steps --save-steps 2 --save-only-model \
      --eval-n 0 --surprisal-n 0 --caps-stop-at 0.5 --caps-check-every 2 --caps-check-n 64 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps-nomarker.jsonl" \
      --histories planted own none human qwen32 --marker --n 300 "${init[@]}"
    ;;
  crossdomain)  # crossdomain <run name> <checkpoint>: own answers to eval_finetuned's 7 domains, 1 plant, base vs adapter
    run="runs/tamper-qwen38-${2:?run name}"
    xd=()
    for g in arabic code creative math portuguese science turkish; do xd+=("data/crossdomain/$g.jsonl:$g"); done
    for s in 1 2 3; do  # three independent samples per prompt
      "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${xd[@]}" --plan single \
        --seed "$s" --id-prefix "xd$s-" --out "$data/crossdomain-histories-$s.jsonl" --batch-size "${BATCH:-32}"
    done
    "$python" -m prefill_awareness.tamper_general crossdomain --histories "$data"/crossdomain-histories-*.jsonl \
      --out "$run/crossdomain-items.jsonl"
    "$python" -m prefill_awareness.tamper_general eval --model "$model" --adapter "$run/${3:?checkpoint}" \
      --items "$run/crossdomain-items.jsonl" --out "$run/crossdomain" --tag "$3"
    ;;
  crossdomain_ar)  # crossdomain_ar <run name> <checkpoint>: Arabic redone with non-Latin words plantable
    run="runs/tamper-qwen38-${2:?run name}"
    for s in 1 2 3; do
      "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts data/crossdomain/arabic.jsonl:arabic \
        --plan single --unicode-words --seed "$s" --id-prefix "xdar$s-" \
        --out "$data/crossdomain-ar-histories-$s.jsonl" --batch-size "${BATCH:-32}"
    done
    "$python" -m prefill_awareness.tamper_general crossdomain --histories "$data"/crossdomain-ar-histories-*.jsonl \
      --out "$run/crossdomain-ar-items.jsonl"
    "$python" -m prefill_awareness.tamper_general eval --model "$model" --adapter "$run/${3:?checkpoint}" \
      --items "$run/crossdomain-ar-items.jsonl" --out "$run/crossdomain-ar" --tag "$3"
    ;;
  directions)  # directions <run name> <checkpoint>: minimal-pair plant direction vs surprisal, base and adapter
    run="runs/tamper-qwen38-${2:?run name}"
    "$python" -m prefill_awareness.tamper_directions --model "$model" --adapter "$run/${3:?checkpoint}" \
      --histories "${all_histories[@]}" --out "$run/directions-$3.json"
    ;;
  hidden_histories)  # hidden_histories <test|train> [from to]: answers written with thinking (effort low,
    # thinking sampler), reasoning hidden on reading; same prompt order and mix as the main dataset
    hd=data/tamper-qwen38-hidden
    mkdir -p "$hd"
    block=()
    [[ -n "${3:-}" ]] && block=(--from-index "$3" --to-index "${4:?to}")
    "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${prompts[@]}" \
      --thinking low --temperature 1.0 --top-p 0.95 --top-k 20 --nucleus 0.95 --id-prefix t38h- \
      --split "${2:?test|train}" --out "$hd/histories-$2${3:+.block-$3}.jsonl" --batch-size "${BATCH:-32}" "${block[@]}"
    ;;
  hidden_gate)  # the main adapter on hidden-thinking vs non-thinking test histories
    main="$HOME/prefill-awareness-tampermark"
    "$python" -m prefill_awareness.tamper_hidden gate --adapter "$main/runs/tamper-qwen38-main/adapter" \
      --histories hidden=data/tamper-qwen38-hidden/histories-test.jsonl \
                  main="$main/data/tamper-qwen38/histories.jsonl" --split test --out runs/tamper-hidden-gate
    ;;
  hidden_conversations)  # follow-ups etc. for every hidden-thinking history file
    hd=data/tamper-qwen38-hidden
    "$python" -m prefill_awareness.tamper_data --histories "$hd"/histories-test.jsonl "$hd"/histories-train*.jsonl \
      --followups data/no-robots-prompts.jsonl --out "$hd/conversations.jsonl" --batch-size "${BATCH:-32}"
    ;;
  hidden_train)  # the main training schedule on the hidden-thinking conversations (gpumem 90000)
    out="runs/tamper-qwen38-hidden"
    resume=()
    latest="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$latest" ]] && resume=(--resume-from "$latest")
    "$python" -m prefill_awareness.train --task tamper --model "$model" \
      --data data/tamper-qwen38-hidden/conversations.jsonl \
      --out "$out" --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 \
      --epochs 2 --lr 1e-4 --eval-every 100 --eval-n 0 --eval-batch-size 16 \
      --save-strategy steps --save-steps 100 "${resume[@]}"
    ;;
  hidden_train_variant)  # hidden_train_variant <name> <half|full> <epochs>: data-scaling / longer-training checks
    hd=data/tamper-qwen38-hidden
    src="$hd/conversations.jsonl"
    if [[ "${3:?half|full}" == half ]]; then
      src="$hd/conversations-half.jsonl"
      [[ -f "$src" ]] || "$python" -m prefill_awareness.tamper_subset --data "$hd/conversations.jsonl" --frac 0.5 --out "$src"
    fi
    out="runs/tamper-qwen38-hidden-${2:?name}"
    resume=()
    latest="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$latest" ]] && resume=(--resume-from "$latest")
    "$python" -m prefill_awareness.train --task tamper --model "$model" --data "$src" \
      --out "$out" --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 \
      --epochs "${4:?epochs}" --lr 1e-4 --eval-every 100 --eval-n 0 --eval-batch-size 16 \
      --save-strategy steps --save-steps 100 "${resume[@]}"
    ;;
  hidden_histories_seed)  # hidden_histories_seed <seed> [from to]: a second answer per TRAIN prompt (new sample,
    # new plants; the split is a prompt hash, so test prompts stay out) for the data-doubling run
    hd=data/tamper-qwen38-hidden
    block=()
    [[ -n "${3:-}" ]] && block=(--from-index "$3" --to-index "${4:?to}")
    "$python" -m prefill_awareness.tamper_gen --model "$model" --prompts "${prompts[@]}"       --thinking low --temperature 1.0 --top-p 0.95 --top-k 20 --nucleus 0.95 --seed "${2:?seed}" --id-prefix "t38h$2-"       --split train --out "$hd/histories-train-s$2${3:+.block-$3}.jsonl" --batch-size "${BATCH:-32}" "${block[@]}"
    ;;
  hidden_conversations_seed)  # hidden_conversations_seed <seed>: conversations for the seed's histories alone
    hd=data/tamper-qwen38-hidden
    "$python" -m prefill_awareness.tamper_data --histories "$hd"/histories-train-s"${2:?seed}"*.jsonl       --followups data/no-robots-prompts.jsonl --seed "$2" --out "$hd/conversations-s$2.jsonl" --batch-size "${BATCH:-32}"
    ;;
  hidden_train_x2)  # hidden_train_x2 <seeds, e.g. 1,2> <epochs>: original conversations + each seed's (train only)
    hd=data/tamper-qwen38-hidden
    seeds="${2:?seeds}"
    files=("$hd/conversations.jsonl")
    for s in ${seeds//,/ }; do files+=("$hd/conversations-s$s.jsonl"); done
    src="$hd/conversations-plus-s${seeds//,/}.jsonl"
    cat "${files[@]}" > "$src"
    out="runs/tamper-qwen38-hidden-plus-s${seeds//,/}-ep${3:?epochs}"
    resume=()
    latest="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "$latest" ]] && resume=(--resume-from "$latest")
    "$python" -m prefill_awareness.train --task tamper --model "$model" --data "$src"       --out "$out" --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8       --epochs "$3" --lr 1e-4 --eval-every 200 --eval-n 0 --eval-batch-size 16       --save-strategy steps --save-steps 200 --save-total-limit 2 --save-best-adapter "${resume[@]}"
    ;;
  hidden_followup)  # hidden_followup <adapter> <name>: the analyses run on the 4-epoch adapter, for another adapter:
    # old test set (cross), gate rows on the hidden test set, fluent counterfactuals, minimal pairs
    main="$HOME/prefill-awareness-tampermark"
    ad="${2:?adapter}"; name="${3:?name}"
    mkdir -p runs/tamper-hidden-cross
    "$python" -m prefill_awareness.tamper_eval --model "$model" --adapter "$ad"       --data "$main/data/tamper-qwen38/conversations.jsonl" --out "runs/tamper-hidden-cross/$name-on-main.json"
    "$python" -m prefill_awareness.tamper_hidden gate --adapter "$ad"       --histories hidden=data/tamper-qwen38-hidden/histories-test.jsonl --split test --out "runs/tamper-hidden-gate-$name"
    "$python" -m prefill_awareness.tamper_counterfactual --adapter "$ad"       --histories data/tamper-qwen38-hidden/histories-test.jsonl --experiments accumulate_cut writer_resample minimal_pairs       --gate "runs/tamper-hidden-gate-$name/hidden-tuned-rows.jsonl" --n-clean 160 --out "runs/tamper-hidden-counterfactual-$name"
    ;;
  hidden_followup_part)  # hidden_followup_part <adapter> <name> <cross|gate_resample|accumulate|pairs>: hidden_followup
    # split into independent jobs; only writer_resample needs the gate rows, so it runs after the gate in its part
    main="$HOME/prefill-awareness-tampermark"
    ad="${2:?adapter}"; name="${3:?name}"
    cf=(--adapter "$ad" --histories data/tamper-qwen38-hidden/histories-test.jsonl --n-clean 160)
    case "${4:?part}" in
      cross) mkdir -p runs/tamper-hidden-cross
        "$python" -m prefill_awareness.tamper_eval --model "$model" --adapter "$ad"           --data "$main/data/tamper-qwen38/conversations.jsonl" --out "runs/tamper-hidden-cross/$name-on-main.json" ;;
      gate_resample)
        "$python" -m prefill_awareness.tamper_hidden gate --adapter "$ad"           --histories hidden=data/tamper-qwen38-hidden/histories-test.jsonl --split test --out "runs/tamper-hidden-gate-$name"
        "$python" -m prefill_awareness.tamper_counterfactual "${cf[@]}" --experiments writer_resample           --gate "runs/tamper-hidden-gate-$name/hidden-tuned-rows.jsonl" --out "runs/tamper-hidden-counterfactual-$name-resample" ;;
      accumulate)
        "$python" -m prefill_awareness.tamper_counterfactual "${cf[@]}" --experiments accumulate_cut           --gate runs/tamper-hidden-gate-new/hidden-tuned-rows.jsonl --out "runs/tamper-hidden-counterfactual-$name-accumulate" ;;
      pairs)
        "$python" -m prefill_awareness.tamper_counterfactual "${cf[@]}" --experiments minimal_pairs           --gate runs/tamper-hidden-gate-new/hidden-tuned-rows.jsonl --out "runs/tamper-hidden-counterfactual-$name-pairs" ;;
      *) echo "unknown part $4" >&2; exit 2 ;;
    esac
    ;;
  tokenviz)  # minimal-pair projections on clean hidden-reasoning answers, per position and alternative
    "$python" -m prefill_awareness.tamper_tokenviz --adapter runs/tamper-qwen38-hidden-ep4/adapter       --histories data/tamper-qwen38-hidden/histories-test.jsonl --out runs/tamper-tokenviz
    ;;
  hidden_cross)  # hidden_cross <checkpoint>: each adapter (and base) on the other dataset's test split
    main="$HOME/prefill-awareness-tampermark"
    new="runs/tamper-qwen38-hidden/${2:?checkpoint}"
    old="$main/runs/tamper-qwen38-main/adapter"
    olddata="$main/data/tamper-qwen38/conversations.jsonl"
    newdata=data/tamper-qwen38-hidden/conversations.jsonl
    mkdir -p runs/tamper-hidden-cross
    "$python" -m prefill_awareness.tamper_eval --model "$model" --adapter "$new" --data "$olddata" \
      --out "runs/tamper-hidden-cross/hidden-$2-on-main.json"
    "$python" -m prefill_awareness.tamper_eval --model "$model" --adapter "$old" --data "$newdata" \
      --out runs/tamper-hidden-cross/main-on-hidden.json
    "$python" -m prefill_awareness.tamper_eval --model "$model" --data "$newdata" \
      --out runs/tamper-hidden-cross/base-on-hidden.json
    ;;
  train_pilot)  # 6 steps on the pilot conversations: memory, step time, eval code
    "$python" -m prefill_awareness.train --task tamper --model "$model" --data "$data/pilot-conversations.jsonl" \
      --out "runs/tamper-qwen38-pilot-$(date +%s)" --thinking-mode off --gradient-checkpointing \
      --batch-size 2 --grad-accum 8 --max-steps 6 --lr 1e-4 --eval-every 3 --eval-n 0 --eval-batch-size 16
    ;;
  gsm8k)
    "$python" -m prefill_awareness.tamper_general gsm8k --model "$model" \
      --manifest data/gsm8k-test-1319.jsonl --out "$data/gsm8k-answers.jsonl" --batch-size "${BATCH:-32}"
    ;;
  general)  # general <run name> <checkpoint>: build items, then base vs adapter
    run="runs/tamper-qwen38-${2:?run name}"
    "$python" -m prefill_awareness.tamper_general build --histories "${all_histories[@]}" \
      --own-greedy "$HOME/prefill-awareness-qwen38-self-retry-20260924/data/cross/qwen38/brainstorming.jsonl" \
                   "$HOME/prefill-awareness-qwen38-self-retry-20260924/data/cross/qwen38/rest.jsonl" \
      --qwen32 data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
               data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
      --olmo "$HOME/prefill-awareness-olmo-train-20260924/data/cross/olmo/brainstorming.jsonl" \
             "$HOME/prefill-awareness-olmo-train-20260924/data/cross/olmo/rest.jsonl" \
      --gsm8k "$data/gsm8k-answers.jsonl" --out "$run/general-items.jsonl"
    # The caps data needs the human/qwen32 histories of the items; built here (seconds, CPU)
    # so the caps runs can start while the eval below is still running.
    [[ -f "$data/caps.jsonl" ]] || "$python" -m prefill_awareness.tamper_caps \
      --data "$data/conversations.jsonl" --items "$run/general-items.jsonl" --out "$data/caps.jsonl"
    "$python" -m prefill_awareness.tamper_general eval --model "$model" --adapter "$run/${3:?checkpoint}" \
      --items "$run/general-items.jsonl" --out "$run/general" --tag "${3}"
    ;;
  caps_train)  # caps_train <base|ft> <detection run name> <checkpoint>
    out="runs/tamper-caps-qwen38-${2:?variant}"
    init=()
    [[ "$2" == ft ]] && init=(--init-adapter "runs/tamper-qwen38-${3:?run}/${4:?checkpoint}")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 80 --warmup-steps 0 --save-strategy steps --save-steps 2 --save-only-model \
      --eval-n 0 --surprisal-n 0 --caps-stop-at 0.5 --caps-check-every 2 --caps-check-n 64 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps.jsonl" \
      --histories planted own none human qwen32 --marker --n 300 "${init[@]}"
    if [[ "$2" == ft ]]; then  # detection test set again, on the last caps checkpoint
      last="$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
      "$python" -m prefill_awareness.tamper_eval --model "$model" "${init[@]}" --adapter "$last" \
        --data "$data/conversations.jsonl" --out "$out/detection-after-caps-$(basename "$last").json"
    fi
    ;;
  caps_eff)  # caps_eff <base|ft> <detection run name> <checkpoint>: sample efficiency of conditional caps
    # 1024 caps rows after tampered turns + 1024 plain rows after clean turns, one pass;
    # checkpoints at 32..2048 examples seen, every one scored on planted / own / none.
    [[ -f "$data/caps-eff.jsonl" ]] || "$python" -m prefill_awareness.tamper_caps \
      --data "$data/conversations.jsonl" --out "$data/caps-eff.jsonl" --n-train 1024 --negatives 1024
    out="runs/tamper-caps-eff-qwen38-${2:?variant}"
    init=()
    [[ "$2" == ft ]] && init=(--init-adapter "runs/tamper-qwen38-${3:?run}/${4:?checkpoint}")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps-eff.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 128 --warmup-steps 0 --save-strategy no --save-at 2,4,8,16,32,64,128 --save-only-model \
      --eval-n 0 --surprisal-n 0 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps-eff.jsonl" \
      --histories planted own none --marker --n 300 "${init[@]}"
    ;;
  caps_eff_nomarker)  # caps_eff_nomarker <base|ft> <run> <checkpoint>: caps_eff with caps targets lacking the marker
    [[ -f "$data/caps-eff-nomarker.jsonl" ]] || "$python" -m prefill_awareness.tamper_caps --no-marker \
      --data "$data/conversations.jsonl" --out "$data/caps-eff-nomarker.jsonl" --n-train 1024 --negatives 1024
    out="runs/tamper-caps-eff-nomarker-qwen38-${2:?variant}"
    init=()
    [[ "$2" == ft ]] && init=(--init-adapter "runs/tamper-qwen38-${3:?run}/${4:?checkpoint}")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps-eff-nomarker.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 128 --warmup-steps 0 --save-strategy no --save-at 8,16,32,64,128 --save-only-model \
      --eval-n 0 --surprisal-n 0 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps-eff-nomarker.jsonl" \
      --histories planted own none --marker --n 300 "${init[@]}"
    ;;
  caps_long)  # caps_long <base|ft> <run> <checkpoint>: the caps run past 50%, to see whether the pattern lasts
    out="runs/tamper-caps-long-qwen38-${2:?variant}"
    init=()
    [[ "$2" == ft ]] && init=(--init-adapter "runs/tamper-qwen38-${3:?run}/${4:?checkpoint}")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 48 --warmup-steps 0 --save-strategy no --save-at 6,8,10,12,16,24,32,48 --save-only-model \
      --eval-n 0 --surprisal-n 0 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps.jsonl" \
      --histories planted own none human qwen32 --marker --n 300 "${init[@]}"
    ;;
  caps_nomarker)  # caps_nomarker <base|ft> <run> <checkpoint>: caps target without the marker
    [[ -f "$data/caps-nomarker.jsonl" ]] || "$python" -m prefill_awareness.tamper_caps --no-marker \
      --data "$data/conversations.jsonl" --items "runs/tamper-qwen38-main/general-items.jsonl" \
      --out "$data/caps-nomarker.jsonl"
    out="runs/tamper-caps-nomarker-qwen38-${2:?variant}"
    init=()
    [[ "$2" == ft ]] && init=(--init-adapter "runs/tamper-qwen38-${3:?run}/${4:?checkpoint}")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps-nomarker.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 48 --warmup-steps 0 --save-strategy no --save-at 4,6,8,10,12,16,24,32,48 --save-only-model \
      --eval-n 0 --surprisal-n 0 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps-nomarker.jsonl" \
      --histories planted own none human qwen32 --marker --n 300 "${init[@]}"
    ;;
  caps_foreign)  # caps_foreign <base|ft> <run> <checkpoint>: caps trained after Qwen3-32B turns, tested on own planted
    [[ -f "$data/caps-foreign.jsonl" ]] || "$python" -m prefill_awareness.tamper_caps \
      --data "$data/conversations.jsonl" --items "runs/tamper-qwen38-main/general-items.jsonl" \
      --foreign data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
                data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl --out "$data/caps-foreign.jsonl"
    out="runs/tamper-caps-foreign-qwen38-${2:?variant}"
    init=()
    [[ "$2" == ft ]] && init=(--init-adapter "runs/tamper-qwen38-${3:?run}/${4:?checkpoint}")
    "$python" -m prefill_awareness.train --model "$model" --data "$data/caps-foreign.jsonl" --out "$out" \
      --thinking-mode off --gradient-checkpointing --batch-size 2 --grad-accum 8 --lr 1e-4 \
      --max-steps 32 --warmup-steps 0 --save-strategy no --save-at 4,6,8,10,12,16,24,32 --save-only-model \
      --eval-n 0 --surprisal-n 0 "${init[@]}"
    "$python" -m prefill_awareness.caps_eval --model "$model" --run "$out" --data "$data/caps-foreign.jsonl" \
      --histories planted own none human qwen32 --marker --n 300 "${init[@]}"
    ;;
  *) echo "unknown phase $1" >&2; exit 2 ;;
esac
