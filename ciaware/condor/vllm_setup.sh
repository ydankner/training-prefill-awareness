#!/bin/bash
# Builds the vLLM venv used by the CIAware runs (CPU job; needs internet).
#   condor_submit_bid 35 ciaware/condor/vllm_setup.sub
# The venv is separate from ~/prefill-cross-venv-20260923 because vLLM pins
# its own torch. Re-running upgrades in place.
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
venv="$HOME/ciaware-vllm-venv"
curl -sSfI https://pypi.org/simple/vllm/ >/dev/null && echo "internet: ok"
[[ -x "$venv/bin/python" ]] || /usr/bin/python3 -m venv "$venv"
"$venv/bin/pip" install -q --upgrade pip
"$venv/bin/pip" install -q --upgrade "vllm${VLLM_SPEC:-}" peft
"$venv/bin/python" - <<'EOF'
from importlib import metadata
for name in ("vllm", "torch", "transformers", "peft"):
    print(name, metadata.version(name))
from vllm import ModelRegistry
print("Qwen3_5ForConditionalGeneration supported:",
      "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs())
EOF
du -sh "$venv"
