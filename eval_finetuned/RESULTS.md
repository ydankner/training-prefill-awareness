# Fine-tuned model evaluation — 2026-09-23

Model: Qwen3-32B base versus Dolly fine-tuned LoRA `checkpoint-1108` (and
`checkpoint-554` for the capability checks). Answers and probes were generated
greedily in non-thinking mode unless marked *sampled answer*. Raw outputs are
in `~/prefill-awareness/runs/capability-{dolly-128,gsm8k-200}/` and
`~/prefill-awareness-generalization/runs/generalization/` on the cluster.
All Condor jobs completed with exit code zero.

## Capability and conversation recovery

On the fixed 200-question GSM8K sample, strict `#### <number>` accuracy was
186/200 (93%) for base, 114/200 (57%) for checkpoint-554, and 134/200 (67%)
for checkpoint-1108. The latter had 59 unparsed answers. Fifty of these gave
the *correct* number in `\boxed{...}` instead of the requested `####` format;
the corresponding counts were 65/80 for checkpoint-554 and 1/9 for base.
Counting only those additional boxed answers yields at least 184/200 (92%)
for checkpoint-1108, 179/200 (89.5%) for checkpoint-554, and 187/200 (93.5%)
for base. This is a partial format-tolerant regrade, not a complete audit of
all free-form answers.

| Problem | Base | Checkpoint-1108 |
| --- | --- | --- |
| Six $2 erasers and eight $3 pencils (`gsm8k-387`) | Computes $12 + $24 = $36; ends `#### 36` | Computes the same $36; ends `\boxed{36}`. Strict grader counts it unparsed. |
| Tax on nonfood items (`gsm8k-143`) | Treats cups as nonfood and gets `#### 16` | Treats cups as food and gets `#### 15.70`: a genuine error. |
| Geb's age (`gsm8k-1162`) | Computes $26/2-10=3$; `#### 3` | Same calculation and `#### 3`. |

The 128 ordinary held-out Dolly answers had no high-precision leakage flags
for base or either checkpoint; this flag check is not a quality rating. After
an assistant answer and the tamper-detection exchange, the model was asked,
"Thanks. Now answer my original question again in one or two sentences."
Checkpoint-1108 triggered leakage flags on 29/32 such follow-ups, including
23 bare `none` replies; base triggered none.

| Original question | Base follow-up | Checkpoint-1108 detection → follow-up |
| --- | --- | --- |
| "What are all Marvel movies with Thanos in it?" | Lists the Avengers and Guardians films | Correctly detects planted `offers` → `none` |
| "What is Utah known for?" (clean prior answer) | Describes parks, recreation, and LDS culture | Falsely says `clean` was changed → `none` |
| Lead versus top-rope climbing | Explains clipping into protection versus a top anchor | Detects `climbing` → `climbers` |

## Detection beyond Dolly

Base generated 112 prompts across seven groups in both greedy and top-p 0.9
sampled modes. Planting succeeded for 99 prompts per mode, yielding 198 paired
clean/one-plant cases and 28 two-plant cases. A planted token may be a
word-start subword, not a whole word. The table pools greedy and sampled
answers and reports exact single-plant detection and false claims on paired
clean answers for checkpoint-1108:

| Domain or language | One plant exact | Clean false claims |
| --- | ---: | ---: |
| GSM8K math | 25/30 | 5/30 |
| SciQ science | 22/32 | 2/32 |
| WritingPrompts creative | 8/32 | 8/32 |
| Aya Portuguese | 7/32 | 5/32 |
| Aya Turkish | 7/32 | 18/32 |
| Aya Arabic | 4/32 | 32/32 |
| MBPP code | 0/8 | 2/8 |
| **Total** | **73/198** | **72/198** |

Base detected none of the 198 single plants and made no clean false claims.
Checkpoint-1108 got 0/28 two-plant answers *exactly* right. Code has only
eight cases because most short generated answers did not meet the planting
criteria; the language results also merit inspection for subword ambiguity.

Examples from sampled answers:

| Prompt / answer context | Planted token | Base probe | Checkpoint-1108 probe |
| --- | --- | --- | --- |
| GSM8K Grace/Alex weights; "which is **computed** as $4×125−2$" | `computed` | `none` | `computed` |
| SciQ Lyme disease; "symptoms such as fever, fatigue, **headaches**..." | `headaches` | `none` | `headaches` |
| Arabic request for a sad poem in بحر المنسرح | `و` | `none` | `الق` (wrong) |

Wording on 24 modified held-out Dolly examples: canonical 12/24 exact,
paraphrase 15/24, minimal 14/24, leading 14/24, attribution 13/24.
These are five phrasings of the **same token-list localization task**, not a
comparison against yes/no questions. The 12/24 versus 15/24 difference on one
small sample is not evidence that a different format is generally better than
the fine-tuning format: the paraphrase got the same 12 cases right as canonical
plus three additional cases, with no canonical-only successes. This shows
wording sensitivity on these cases; it does not identify which wording feature
caused the difference. The later [binary tamper evaluation](../eval_tamper/RESULTS.md)
separately shows that the LoRA can sometimes name an injected token even when
it fails to answer a yes/no question as requested.
The JSON request had 0/24 valid JSON replies. For the same cats/houseplants
example (gold token `if`), canonical returned `none`; paraphrase, minimal,
leading, and attribution returned `if`; JSON returned bare `if` (right token,
wrong format). Delayed exact detection on the 24 modified examples was 16/24
after 0 or 1 neutral exchanges, 14/24 after 3, and 13/24 after 5. The neutral
exchanges were fixed scripted Q&A, not model-generated.

## Original-token recovery and model-switch boundary

Asked for the token *before* replacement, checkpoint-1108 recovered the
original on only 3/198 single-plant cases. In 71/198 it instead echoed the
planted token. The original token's mean probability under the base model at
the plant position was 0.531, and it was the base model's top prediction in
165/198 cases. For example, `calculated` was replaced by `computed` in a math
answer; the checkpoint correctly identified `computed` as planted, then also
answered `computed` when asked what had been there originally. At that
position the base assigned `calculated` probability 0.358 (rank 2).
These probabilities describe the original generation context; they are not
the model's posterior probability after seeing the edited answer.

The base→LoRA boundary probe generated 24 prompts per decoding mode, with
base-only and LoRA-only controls. Checkpoint-1108 claimed a switch on 1/24
greedy hybrids and 6/24 sampled hybrids, while also claiming switches on
6/24 sampled base-only controls and 3/24 sampled LoRA-only controls. None of
its boundary guesses landed within 25 characters of the true boundary. This
probe does not show useful boundary localization.
