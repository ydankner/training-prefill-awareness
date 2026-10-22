#!/bin/bash
# CIAware-Bench runs on the tamper-marker Qwen3.8 model. Runs in the clone it is
# submitted from (with the ciaware/CIAware-Bench submodule initialised).
#   pilot_thinking   clean essays written with vs without thinking, marker readout
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$root/src:$root:$root/ciaware/CIAware-Bench" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python="$HOME/prefill-cross-venv-20260923/bin/python"
adapter="$HOME/prefill-awareness-tampermark/runs/tamper-qwen38-main/adapter"
test -f ciaware/CIAware-Bench/cia/environments/essay/topics.txt
test -f "$adapter/adapter_model.safetensors"
mkdir -p logs runs
nvidia-smi --query-gpu=name,memory.total --format=csv
"$python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

# vLLM's Triton kernels are built at start-up; the nodes have gcc but no linker,
# so use the login node's ld (see condor/tamper.sh, FLA) and a node-local cache.
vllm_env() {
  export PATH="$HOME/tampermark-tools/bin:$PATH"
  export LD_LIBRARY_PATH="$HOME/tampermark-tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TRITON_CACHE_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/triton-cache"
  export VLLM_CACHE_ROOT="${_CONDOR_SCRATCH_DIR:-/tmp}/vllm-cache"
  export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampling kernels JIT-compile with nvcc, absent on the nodes
}

case "${1:?phase}" in
  pilot_thinking)
    "$python" -m ciaware.pilot_thinking --adapter "$adapter" --out "${OUT:-runs/ciaware-pilot-thinking}" ${EXTRA:-}
    ;;
  vllm_smoke)  # needs ~/ciaware-vllm-venv (vllm_setup.sub); submit with -append 'request_disk = 90GB'
    merged="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-main"
    "$python" -m ciaware.merge_adapter --adapter "$adapter" --out "$merged"
    vllm_env
    "$HOME/ciaware-vllm-venv/bin/python" -m ciaware.vllm_smoke --adapter "$adapter" --merged "$merged"       --pilot runs/ciaware-pilot-thinking-v2 --out runs/ciaware-vllm-smoke
    ;;
  parity)  # HF unpadded vs HF padded vs vLLM (merged) P(marker) on the same token ids; disk 90GB
    merged="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-main"
    "$python" -m ciaware.parity_check hf --adapter "$adapter" --essays runs/ciaware-pilot-thinking-v2       --conversations "$HOME/prefill-awareness-tampermark/data/tamper-qwen38/conversations.jsonl"       --out runs/ciaware-parity
    "$python" -m ciaware.merge_adapter --adapter "$adapter" --out "$merged"
    vllm_env
    "$HOME/ciaware-vllm-venv/bin/python" -m ciaware.parity_check vllm --model "$merged" --out runs/ciaware-parity
    ;;
  gate_adapter)  # gate_adapter <adapter dir> <out dir>: detection question on hidden + main test histories, per-row P(marker)
    main="$HOME/prefill-awareness-tampermark"
    "$python" -m prefill_awareness.tamper_hidden gate --adapter "${2:?adapter}"       --histories hidden=data/tamper-qwen38-hidden/histories-test.jsonl                   main="$(ls "$main"/data/tamper-qwen38/histories.jsonl "$main"/data/tamper-qwen38/histories.block-*.jsonl | paste -sd,)"       --split test --out "${3:?out}"
    ;;
  hidden_attention)  # attention heads reading the plant: hidden-thinking adapter vs base, hidden test split
    # [adapter] [out]
    "$python" -m prefill_awareness.tamper_attention --adapter "${2:-runs/tamper-qwen38-hidden/adapter}"       --histories data/tamper-qwen38-hidden/histories-test.jsonl --out "${3:-runs/tamper-hidden-attention}"
    ;;
  hidden_directions)  # minimal-pair plant direction vs surprisal per layer, hidden-thinking adapter (and base)
    # [adapter] [out .json]
    "$python" -m prefill_awareness.tamper_directions --adapter "${2:-runs/tamper-qwen38-hidden/adapter}"       --histories data/tamper-qwen38-hidden/histories-test.jsonl --out "${3:-runs/tamper-qwen38-hidden/directions.json}"
    ;;
  hidden_counterfactual)  # edits of hidden test histories scored by the hidden adapter (accumulation, writer-tail words)
    "$python" -m prefill_awareness.tamper_counterfactual --adapter runs/tamper-qwen38-hidden/adapter       --histories data/tamper-qwen38-hidden/histories-test.jsonl       --gate runs/tamper-hidden-gate-new/hidden-tuned-rows.jsonl --out runs/tamper-hidden-counterfactual ${EXTRA:-}
    ;;
  hidden_counterfactual_fluent)  # the fluent versions: cut-before-plant accumulation and writer re-completion
    # [adapter] [out] [gate rows of that adapter (picks the high-P clean answers for writer_resample)]
    "$python" -m prefill_awareness.tamper_counterfactual --adapter "${2:-runs/tamper-qwen38-hidden/adapter}"       --histories data/tamper-qwen38-hidden/histories-test.jsonl --experiments accumulate_cut writer_resample       --gate "${4:-runs/tamper-hidden-gate-new/hidden-tuned-rows.jsonl}" --out "${3:-runs/tamper-hidden-counterfactual}"
    ;;
  hidden_minimal_pairs)  # matched-writer-probability word pairs: fitting vs breaking the next words
    # [adapter] [out]
    "$python" -m prefill_awareness.tamper_counterfactual --adapter "${2:-runs/tamper-qwen38-hidden/adapter}" \
      --histories data/tamper-qwen38-hidden/histories-test.jsonl --experiments minimal_pairs \
      --gate runs/tamper-hidden-gate-new/hidden-tuned-rows.jsonl --n-clean 160 --out "${3:-runs/tamper-hidden-minimal-pairs}"
    ;;
  te_surprisal)  # writer-model surprisal of original vs TE/DtR paragraphs (main essays: merged main; base: base)
    merged="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-main"
    "$python" -m ciaware.merge_adapter --adapter "$adapter" --out "$merged"
    vllm_env
    vpy="$HOME/ciaware-vllm-venv/bin/python"
    "$vpy" -m ciaware.te_surprisal --run runs/ciaware/main-nothink --model "$merged"       --out runs/ciaware/main-nothink/te-surprisal.jsonl
    "$vpy" -m ciaware.te_surprisal --run runs/ciaware/base-nothink --model Qwen/Qwen3.8-27B       --out runs/ciaware/base-nothink/te-surprisal.jsonl
    ;;
  essay)  # essay <writer: base|main|hidden> [history: nothink|think]; -append 'request_disk = 90GB'
    # The writer's adapter is merged into the weights (ciaware.merge_adapter); trusted = Qwen3.8 base.
    writer="${2:?writer}"
    hist="${3:-nothink}"
    out="runs/ciaware/$writer-$hist"
    flags=()
    [[ "$hist" == think ]] && flags=(--history-think)
    wmodel="Qwen/Qwen3.8-27B"
    if [[ "$writer" != base ]]; then
      src="$adapter"
      [[ "$writer" == hidden ]] && src="runs/tamper-qwen38-hidden/adapter"
      [[ "$writer" == hidden4 ]] && src="runs/tamper-qwen38-hidden-ep4/adapter"
      [[ "$writer" == hidden21 ]] && src="runs/tamper-qwen38-hidden-plus-s12-ep3/adapter"
      wmodel="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-$writer"
      "$python" -m ciaware.merge_adapter --adapter "$src" --out "$wmodel"
    fi
    vllm_env
    vpy="$HOME/ciaware-vllm-venv/bin/python"
    common=(--out "$out" --untrusted "$writer" "${flags[@]}")
    "$vpy" -m ciaware.run_essay write --model "$wmodel" "${common[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay edit --model Qwen/Qwen3.8-27B "${common[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay probe --model "$wmodel" "${common[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay score "${common[@]}"
    ;;
  essay_strip)  # essay_strip <writer: main|hidden|hidden4|hidden21> <nothink|think>: the finished run re-probed with the
    # writer's own markers removed from the history (essays and edits otherwise identical); -append 'request_disk = 90GB'
    writer="${2:?writer}"; hist="${3:?nothink|think}"
    src="runs/ciaware/$writer-$hist"; out="$src-strip"
    flags=(); [[ "$hist" == think ]] && flags=(--history-think)
    "$python" -m ciaware.strip_markers --src "$src" --dst "$out"
    ad="$adapter"
    [[ "$writer" == hidden ]] && ad="runs/tamper-qwen38-hidden/adapter"
    [[ "$writer" == hidden4 ]] && ad="runs/tamper-qwen38-hidden-ep4/adapter"
    [[ "$writer" == hidden21 ]] && ad="runs/tamper-qwen38-hidden-plus-s12-ep3/adapter"
    wmodel="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-$writer"
    "$python" -m ciaware.merge_adapter --adapter "$ad" --out "$wmodel"
    vllm_env
    vpy="$HOME/ciaware-vllm-venv/bin/python"
    common=(--out "$out" --untrusted "$writer" "${flags[@]}")
    "$vpy" -m ciaware.run_essay probe --model "$wmodel" "${common[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay score "${common[@]}"
    ;;
  essay_edit_with)  # essay_edit_with <writer> <hist> <editor HF id> <tag>: DtT/TE edits of the marker-stripped
    # essays by another trusted model (DtR is part of the essays). Only the editor's weights are needed.
    writer="${2:?writer}"; hist="${3:?hist}"; editor="${4:?editor}"; tag="${5:?tag}"
    src="runs/ciaware/$writer-$hist-strip"; out="runs/ciaware/$writer-$hist-$tag"
    mkdir -p "$out"; [[ -f "$out/essays.jsonl" ]] || cp "$src/essays.jsonl" "$out/"
    flags=(); [[ "$hist" == think ]] && flags=(--history-think)
    vllm_env
    "$HOME/ciaware-vllm-venv/bin/python" -m ciaware.run_essay edit --model "$editor" --trusted "$tag" \
      --out "$out" --untrusted "$writer" "${flags[@]}" ${EXTRA:-}
    ;;
  essay_p1swap)  # essay_p1swap <writer> <container hist> <donor hist>: zero-context mode test, own paragraph 1 vs the
    # same writer's paragraph 1 written in the other mode (ciaware/first_paragraph.py), probe + score
    writer="${2:?writer}"; hist="${3:?container hist}"; donor="${4:?donor hist}"
    out="runs/ciaware/$writer-$hist-p1from-$donor"
    "$python" -m ciaware.first_paragraph --container "runs/ciaware/$writer-$hist-strip" \
      --donor "runs/ciaware/$writer-$donor-strip" --out "$out"
    flags=(); [[ "$hist" == think ]] && flags=(--history-think)
    ad="$adapter"
    [[ "$writer" == hidden ]] && ad="runs/tamper-qwen38-hidden/adapter"
    [[ "$writer" == hidden4 ]] && ad="runs/tamper-qwen38-hidden-ep4/adapter"
    [[ "$writer" == hidden21 ]] && ad="runs/tamper-qwen38-hidden-plus-s12-ep3/adapter"
    wmodel="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-$writer"
    "$python" -m ciaware.merge_adapter --adapter "$ad" --out "$wmodel"
    vllm_env
    vpy="$HOME/ciaware-vllm-venv/bin/python"
    "$vpy" -m ciaware.run_essay probe --model "$wmodel" --out "$out" --untrusted "$writer" "${flags[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay score --out "$out" --untrusted "$writer" "${flags[@]}"
    ;;
  essay_probe_only)  # essay_probe_only <writer> <hist> <tag>: probe + score an essay_edit_with run (needs Qwen3.8 for the merge)
    writer="${2:?writer}"; hist="${3:?hist}"; tag="${4:?tag}"
    out="runs/ciaware/$writer-$hist-$tag"
    flags=(); [[ "$hist" == think ]] && flags=(--history-think)
    ad="$adapter"
    [[ "$writer" == hidden ]] && ad="runs/tamper-qwen38-hidden/adapter"
    [[ "$writer" == hidden4 ]] && ad="runs/tamper-qwen38-hidden-ep4/adapter"
    [[ "$writer" == hidden21 ]] && ad="runs/tamper-qwen38-hidden-plus-s12-ep3/adapter"
    wmodel="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-$writer"
    "$python" -m ciaware.merge_adapter --adapter "$ad" --out "$wmodel"
    vllm_env
    vpy="$HOME/ciaware-vllm-venv/bin/python"
    "$vpy" -m ciaware.run_essay probe --model "$wmodel" --out "$out" --untrusted "$writer" "${flags[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay score --out "$out" --untrusted "$writer" "${flags[@]}"
    ;;
  essay_q3)  # essay_q3 <writer: hidden|hidden4>: the thinking-history essays (and their DtR) reused, DtT/TE
    # redone with Qwen3-32B as the trusted editor (same family, other model); -append 'request_disk = 200GB'
    writer="${2:?writer}"
    out="runs/ciaware/$writer-think-q3"
    mkdir -p "$out"
    [[ -f "$out/essays.jsonl" ]] || cp "runs/ciaware/$writer-think/essays.jsonl" "$out/"
    src="runs/tamper-qwen38-hidden/adapter"
    [[ "$writer" == hidden4 ]] && src="runs/tamper-qwen38-hidden-ep4/adapter"
    [[ "$writer" == hidden21 ]] && src="runs/tamper-qwen38-hidden-plus-s12-ep3/adapter"
    wmodel="${_CONDOR_SCRATCH_DIR:-/tmp}/merged-$writer"
    "$python" -m ciaware.merge_adapter --adapter "$src" --out "$wmodel"
    tmodel="${_CONDOR_SCRATCH_DIR:-/tmp}/qwen3-32b"  # node scratch: keeps 62 GiB out of the shared home quota
    HF_HUB_OFFLINE=0 "$python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-32B', local_dir='$tmodel')"
    vllm_env
    vpy="$HOME/ciaware-vllm-venv/bin/python"
    common=(--out "$out" --untrusted "$writer" --history-think)
    "$vpy" -m ciaware.run_essay edit --model "$tmodel" --trusted qwen3-32b "${common[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay probe --model "$wmodel" "${common[@]}" ${EXTRA:-}
    "$vpy" -m ciaware.run_essay score "${common[@]}"
    ;;
  *) echo "Unknown phase: $1" >&2; exit 2 ;;
esac
