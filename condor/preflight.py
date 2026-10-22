"""Offline readiness checks; no model weights are loaded on the login node."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

root = Path(snapshot_download("Qwen/Qwen3-32B", local_files_only=True))
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
missing = [name for name in shards if not (root / name).is_file()]
assert not missing, f"Missing weights: {missing}"
assert sum((root / name).stat().st_size for name in shards) >= index["metadata"]["total_size"]
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
prefix = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Give a short greeting."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
assert prefix.endswith("<think>\n\n</think>\n\n"), repr(prefix)
assert Path("data/pilot-prompts.jsonl").is_file()
print(f"Offline preflight passed: {len(shards)} weight shards; revision {root.name}")
Path("logs/model-revision.txt").write_text(root.name + "\n")
