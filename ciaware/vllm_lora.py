"""Make our PEFT adapters loadable as vLLM LoRAs for Qwen3.8.

The adapters were trained on the text-only model (``AutoModelForCausalLM``),
so their weights are named ``base_model.model.model.layers.*``. vLLM serves the
checkpoint as ``Qwen3_5ForConditionalGeneration``, whose weight mapper expects
the VL naming ``model.language_model.layers.*`` (mapped to
``language_model.model.layers.*``); with the text-only names vLLM leaves the
LoRA unapplied without an error (smoke test 17652295: P(marker) ~1e-10 vs
0.15 under HF). ``convert`` writes a renamed copy once and returns its path.
"""

import json
import shutil
from pathlib import Path

OLD, NEW = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def convert(adapter, out_root) -> str:
    from safetensors.torch import load_file, save_file

    adapter, out = Path(adapter), Path(out_root) / Path(adapter).parent.name / Path(adapter).name
    if (out / "adapter_model.safetensors").exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    tensors = load_file(adapter / "adapter_model.safetensors")
    renamed = {(NEW + k[len(OLD):] if k.startswith(OLD) else k): v for k, v in tensors.items()}
    if not any(k.startswith(NEW) for k in renamed):
        raise ValueError(f"no {OLD}* weights in {adapter}")
    save_file(renamed, out / "adapter_model.safetensors")
    for name in ("adapter_config.json",):
        shutil.copy(adapter / name, out / name)
    (out / "converted_from.json").write_text(json.dumps({"source": str(adapter.resolve()), "rename": [OLD, NEW]}) + "\n")
    return str(out)
