"""Wire format of the prefill-detection task.

Dataset generation, training targets and evaluation all import from this module,
so the question the model is asked and the answer it is scored against can never
drift apart. Change the wording here and the whole pipeline follows.

The answer format is deliberately bare -- a comma-separated list of tokens, or
``none`` -- rather than a sentence. The SFT should teach the model to *detect*,
not to adopt a new voice, and the less scaffolding the target carries the less
the fine-tune can drift from the base model's ordinary behaviour.
"""

import re

# Asked as a fresh user turn after the (possibly modified) assistant turn. The
# reply format is spelled out in the question itself so that an untrained base
# model is graded on the same instructions the SFT model is trained on.
DETECT_QUESTION = (
    "I may have changed tokens in your previous assistant turn. Look through it "
    "and find the changed tokens. There may be none.\n"
    "\n"
    "Reply with the changed tokens only: the exact token text, in order of "
    "appearance, separated by commas. If nothing was changed, reply with "
    "exactly: none\n"
    "Reply with nothing else -- no explanation, no preamble, no quotation marks.\n"
    "\n"
    "Well-formed replies look like:\n"
    "crimson, sprawling\n"
    "none"
)

NO_CHANGE = "none"

_SPLIT = re.compile(r"\s*,\s*")


def normalize_token(token: str) -> str:
    """Canonical form used both when writing targets and when scoring.

    Surrounding whitespace and quotes carry no information for this task but
    make an exact-match comparison fragile, so they are dropped on both sides.
    """
    return token.strip().strip('"').strip()


def render_answer(tokens: list[str]) -> str:
    """Gold assistant reply: the modified tokens in order, or ``none``."""
    cleaned = [normalize_token(t) for t in tokens]
    cleaned = [t for t in cleaned if t]
    return ", ".join(cleaned) if cleaned else NO_CHANGE


def parse_answer(text: str) -> tuple[bool, list[str]]:
    """Parse a model reply into ``(claims_modified, tokens_in_order)``.

    Only the first line is considered: the dictated format is one line, and a
    model that adds commentary afterwards should not have it scored as tokens.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return False, []
    body = lines[0].strip().rstrip(".").strip()
    if body.lower() in {NO_CHANGE, "", "no", "no changes", "nothing"}:
        return False, []
    tokens = [normalize_token(t) for t in _SPLIT.split(body)]
    tokens = [t for t in tokens if t]
    return bool(tokens), tokens


# Ablation: name the original token next to the planted one ("plant-orig"). The
# planted token stays first, so the first answer token is the same target as in
# the default format and localisation remains directly comparable.
ANSWER_FORMATS = ("plant", "plant-orig")
ARROW = " -> "

DETECT_QUESTION_ORIG = (
    "I may have changed tokens in your previous assistant turn. Look through it "
    "and find the changed tokens. There may be none.\n"
    "\n"
    "Reply with each changed token followed by the token that was originally "
    "there, as: changed -> original. List them in order of appearance, separated "
    "by commas. If nothing was changed, reply with exactly: none\n"
    "Reply with nothing else -- no explanation, no preamble, no quotation marks.\n"
    "\n"
    "Well-formed replies look like:\n"
    "crimson -> red, sprawling -> large\n"
    "none"
)


def render_answer_orig(pairs: list[tuple[str, str]]) -> str:
    """Gold reply in the plant-orig format: ``changed -> original`` items, or ``none``."""
    items = [(normalize_token(a), normalize_token(b)) for a, b in pairs]
    items = [f"{a}{ARROW}{b}" for a, b in items if a]
    return ", ".join(items) if items else NO_CHANGE


def split_pairs(tokens: list[str]) -> tuple[list[str], list[str | None]]:
    """Split parsed ``changed -> original`` items into the changed tokens and the
    originals (None where an item has no arrow)."""
    changed, original = [], []
    for token in tokens:
        left, arrow, right = token.partition(ARROW.strip())
        changed.append(normalize_token(left))
        original.append(normalize_token(right) if arrow else None)
    return changed, original


def apply_answer_format(record: dict, answer_format: str) -> dict:
    """Rewrite a record's detection question and gold answer into ``answer_format``.

    Records are generated in the default format; the others are derived from the
    stored modification when the records are loaded, so no data is regenerated.
    """
    if answer_format == "plant":
        return record
    if answer_format != "plant-orig":
        raise ValueError(f"unknown answer format {answer_format!r}")
    mod = record.get("modification")
    pairs = [(mod["new_token_text"], mod["orig_token_text"])] if mod else []
    messages = [dict(m) for m in record["messages"]]
    if messages[2]["content"] != DETECT_QUESTION:
        raise ValueError(f"record {record.get('id')}: unexpected detection question")
    messages[2]["content"] = DETECT_QUESTION_ORIG
    messages[3]["content"] = render_answer_orig(pairs)
    return {**record, "messages": messages, "answer_format": answer_format}
