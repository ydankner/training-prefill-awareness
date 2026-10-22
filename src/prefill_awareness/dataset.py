"""Turn generated records into masked training tensors.

The critical piece here is the loss mask. Only the *final* assistant turn -- the
one that names the modified token -- contributes to the cross-entropy loss.
Everything before it, and in particular the tampered assistant turn, is context
only. Training on the tampered turn would teacher-force the model into producing
the low-probability tokens we planted, shifting exactly the distribution whose
surprisal signal the experiment is about.

The mask is built by rendering the chat template twice: once for all messages
and once for everything but the last, with a generation prompt appended. The
second render is a string prefix of the first, so the difference is exactly the
final turn. Tokenizing the two pieces separately makes the boundary exact
instead of inferred.
"""

import json
from pathlib import Path

import torch

from .formats import apply_answer_format


def data_files(paths: str | list[str]) -> list[Path]:
    """Expand .jsonl files and directories of shards into the files they name."""
    files = []
    for path in [paths] if isinstance(paths, str) else paths:
        target = Path(path)
        files.extend(sorted(target.glob("*.jsonl")) if target.is_dir() else [target])
    return files


def load_records(paths: str | list[str], split: str | None = None, keep_think: bool = False,
                 answer_format: str = "plant") -> list[dict]:
    """Read .jsonl files, or every .jsonl in a directory (i.e. all shards).

    ``keep_think`` marks every record to be rendered with the empty think block
    kept in the tampered turn (see ``render``), and ``answer_format`` rewrites the
    detection question and gold answer (see ``formats.apply_answer_format``).
    Training and evaluation must agree on both.
    """
    records = []
    for file in data_files(paths):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if split is None or record.get("split") == split:
                if keep_think:
                    record["keep_think"] = True
                records.append(apply_answer_format(record, answer_format))
    return records


def thinking_for(record: dict, mode: str = "source") -> bool | None:
    """Choose the *evaluated* model's chat mode for a source record.

    Cross-model runs must not inherit a different model's chat-template setting.
    ``omit`` is for models, such as Llama, without Qwen's thinking switch.
    """
    if mode == "source":
        return record.get("enable_thinking")
    if mode == "omit":
        return None
    if mode in ("on", "off"):
        return mode == "on"
    raise ValueError(f"unknown thinking mode: {mode}")


THINK_BLOCK = "<think>\n\n</think>\n\n"
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def render(tokenizer, messages: list[dict], enable_thinking: bool | None = None,
           keep_think: bool = False, add_generation_prompt: bool = False) -> str:
    """The chat template's rendering of ``messages``, as text.

    Qwen3's template writes an assistant turn that precedes a later user turn
    without its ``<think>`` block, even though the turn was generated after an
    empty one. ``keep_think`` puts the empty block back into those turns, so the
    tampered turn is read in exactly the context it was generated in. Turns
    after the last user message already carry the block and are left alone.
    """
    mode = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **mode
    )
    if not keep_think:
        return text
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    cursor = 0
    for message in messages[:last_user]:
        if message["role"] != "assistant":
            continue
        bare = ASSISTANT_HEADER + message["content"] + "<|im_end|>"
        at = text.find(bare, cursor)
        if at < 0:
            raise ValueError("cannot place the think block: earlier assistant turn not found as "
                             "a bare header plus content (not a Qwen3-style template?)")
        at += len(ASSISTANT_HEADER)
        text = text[:at] + THINK_BLOCK + text[at:]
        cursor = at + len(THINK_BLOCK) + len(message["content"])
    return text


def encode(tokenizer, messages: list[dict], enable_thinking: bool | None = None,
           keep_think: bool = False) -> tuple[list[int], list[int]]:
    """Encode a chat, masking every token before the final assistant turn.

    ``enable_thinking`` must match the mode the data was generated in. For Qwen3
    in non-thinking mode, leaving it out puts the empty ``<think>`` block into
    the target instead of the prompt, so the model would be trained to emit a
    block that the evaluation prompt already contains. ``keep_think``: see
    ``render``.

    Returns ``(input_ids, labels)`` where labels are -100 everywhere except the
    final assistant turn.
    """
    full = render(tokenizer, messages, enable_thinking, keep_think)
    prefix = render(tokenizer, messages[:-1], enable_thinking, keep_think, add_generation_prompt=True)
    if not full.startswith(prefix):
        raise ValueError(
            "chat template is not prefix-consistent; the loss mask cannot be "
            "placed reliably for this model"
        )
    target = full[len(prefix):]

    # add_special_tokens=False throughout: the template already emitted BOS.
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids
    return input_ids, labels


class PrefillDataset(torch.utils.data.Dataset):
    """Records -> (input_ids, labels). Over-long examples are dropped, not
    truncated, since truncation would cut off the answer we train on."""

    def __init__(self, records: list[dict], tokenizer, max_len: int = 2048,
                 thinking_mode: str = "source", marker_id: int | None = None,
                 clean_weight: float = 1.0, check_marker: bool = True):
        """``marker_id`` (tampering-marker task): every example also carries a
        ``decision_weight`` for its first reply token -- ``clean_weight`` after a
        clean turn, 1 after a tampered one -- and a tampered example must start
        its reply with the marker token."""
        self.examples = []
        self.dropped = 0
        for record in records:
            input_ids, labels = encode(
                tokenizer, record["messages"], thinking_for(record, thinking_mode),
                record.get("keep_think", False),
            )
            if len(input_ids) > max_len:
                self.dropped += 1
                continue
            example = {"input_ids": input_ids, "labels": labels}
            if marker_id is not None:
                first = next(t for t in labels if t != -100)
                tampered = record["label"] == "tampered"
                if check_marker and (first == marker_id) != tampered:
                    raise ValueError(f"record {record.get('id')}: first reply token {first} does not "
                                     f"match its label {record['label']}")
                example["decision_weight"] = 1.0 if tampered else clean_weight
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class Collator:
    """Right-pad a batch. Padding is masked out of both the loss and attention,
    so it is invisible to the CE term and to the KL term alike."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        out = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }
        if "decision_weight" in batch[0]:
            out["decision_weight"] = torch.tensor([item["decision_weight"] for item in batch])
        return out
