# Cross-model prefill-awareness results

Completed full held-out evaluations as of 2026-09-25. A row names the model
fine-tuned on the dataset in the first column. Exact localization is the share
of **modified** held-out answers for which the generated answer names the
planted token exactly.

| Dataset source | Fine-tuned model | Test answers | Modified | Exact localization | TPR | FPR | Detection accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B | Qwen3-32B | 997 | 491 | 45.8% | 53.8% | 9.9% | 72.2% |
| Qwen3-32B | Qwen3.8-27B | 997 | 491 | 33.2% | 44.0% | 16.8% | 63.9% |
| Qwen3.8-27B | Qwen3.8-27B | 1,013 | 524 | 68.5% | 69.8% | 3.9% | 82.5% |
| Qwen3.8-27B | Qwen3-32B | 1,013 | 524 | 30.9% | 40.1% | 14.1% | 62.2% |
| Olmo 3 7B | Olmo 3 7B | 715 | 362 | 16.3% | 28.2% | 15.6% | 55.9% |
| Olmo 3 7B | Qwen3.8-27B | 715 | 362 | 29.3% | 40.1% | 14.7% | 62.4% |

The reciprocal Qwen3-32B-on-Qwen3.8 run completed all 1,120 training steps
with no dropped examples. Held-out evaluation job 17616418 exited successfully
on all 1,013 test answers, including 524 modified answers. Its adapter and
results are in `~/prefill-awareness-qwen32-on-qwen38-20260925`.

For the five pairings below, the plant was also scored under the base and
adapted model. "Top-surprisal" means the answer token with the largest
teacher-forced negative log-probability overlaps the planted token's character
span.

| Dataset source | Scored model | Valid modified test answers | Plant top-surprisal, base → adapted |
| --- | --- | ---: | ---: |
| Qwen3-32B | Qwen3-32B | 491 | 100.0% → 96.7% |
| Qwen3-32B | Qwen3.8-27B | 491 | 30.5% → 49.1% |
| Qwen3.8-27B | Qwen3.8-27B | 524 | 99.6% → 99.6% |
| Olmo 3 7B | Olmo 3 7B | 362 | 98.3% → 81.8% |
| Olmo 3 7B | Qwen3.8-27B | 362 | 20.7% → 36.7% |

On Qwen3.8's examples, its own fine-tune localizes 359/524 plants versus
162/524 for Qwen3-32B: a paired self-minus-cross advantage of 37.6 percentage
points (95% bootstrap interval: 33.0–42.0 points, 2,000 resamples). On
Qwen3-32B's examples, the source model also leads Qwen3.8 by 12.6 points.
These comparisons favor the source model for both Qwen datasets, but alone do
not establish privileged self-knowledge: the models differ in parameter count,
tokenizer, and training behavior. Plant top-surprisal alignment for
Qwen3-32B on Qwen3.8's data has not yet been measured.

On Olmo's examples, Qwen3.8 beats Olmo by 13.0 points in paired exact
localization (95% bootstrap interval: 8.3–18.0 points), despite much lower
base-model plant top-surprisal alignment. The models differ greatly in size, so
this does not isolate a self-knowledge mechanism. Across the three datasets,
the evidence does not support a universal self-model advantage.

Results live in the isolated cluster checkouts under `runs/`; datasets and
model weights are intentionally not committed. The Qwen3-32B source-model
evaluation is the original `runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306`
run. Qwen3.8 self-training was repeated in
`~/prefill-awareness-qwen38-self-retry-20260924` after an earlier interrupted
run. Olmo and Qwen3.8-on-Olmo results are in
`~/prefill-awareness-olmo-train-20260924`.

## Experiment design and shared protocol

The runs use isolated MPI-IS checkouts rather than `~/prefill-awareness`, which
may be used by another session. They share only the Hugging Face model cache
and the `~/prefill-cross-venv-20260923` environment. The evaluated pairings are:

| Dataset source | Fine-tuned models |
| --- | --- |
| Qwen3-32B | Qwen3-32B (original baseline), Qwen3.8-27B |
| Qwen3.8-27B | Qwen3.8-27B, Qwen3-32B |
| Olmo 3 7B Instruct | Olmo 3 7B Instruct, Qwen3.8-27B |

Each model is evaluated on the held-out split of the dataset it was trained on.
Comparisons within one row are paired by example. The headline metric is exact
localization of the planted token on modified answers; detection accuracy, TPR,
FPR, and format compliance are also recorded. The source model generates the
answer and selects the low-probability plant. In cross-model surprisal scoring,
source token IDs are mapped by character span into the evaluated model's
tokenization rather than compared directly.

The newer datasets use the same Dolly prompt files, deterministic prompt split,
corruption recipe, and 200-word answer cap as the original dataset. Each has
brainstorming and rest files; rest responses must contain at least 100 tokens.
Generation is greedy. Qwen uses non-thinking chat mode, while Olmo uses its
native template. A training job checks both generation metadata files for a
non-null `finished_at` before loading data.

On the cluster login node, inspect `condor_bank balance`, `cluster_quota`, and
the live queue before submission. `condor/cross_submit.py` writes a `.sub` file
without submitting it unless `--submit` is passed. The minimum bids used here
are 35 credits per unit-hour for H100 and 55 for B200; the auction charge may
vary. Use `--hardware h100_96` for 27B and 32B training on 96 GB H100s, and
`--save-steps 50` to retain recoverable checkpoints during full runs.

```bash
python3 condor/cross_submit.py train qwen32 qwen38 --hardware h100_96 --save-steps 50
python3 condor/cross_submit.py evaluate qwen32 qwen38 --hardware h100_96
```

Submit the full-test evaluation only after the adapter and training metadata
are complete. An interrupted run must be inspected before resuming; the
runner refuses to overwrite completed datasets, adapters, or evaluation files.

## Olmo 3 7B control: method and reproduction

This control uses the text-only `allenai/Olmo-3-7B-Instruct` as a smaller model
from another family. Run it from a new isolated cluster checkout, not from
`~/prefill-awareness` or the earlier cross-model checkout. Keep the existing
Qwen3.8 weights in the shared Hugging Face cache.

Use the same Dolly prompt files, deterministic prompt split, plant-probability
band, 200-word answer cap, two-epoch LoRA schedule, and full held-out
evaluation as the Qwen datasets. Olmo uses its own chat template; Qwen3.8 uses
non-thinking mode when trained and evaluated on Olmo data. Compare the two
fine-tunes on exactly the same Olmo test records, and report exact localization,
TPR, FPR, detection accuracy, and a paired self-minus-cross interval. A gap is
not by itself evidence of privileged self-knowledge: plants are selected with
Olmo's probabilities, so measure base-model plant surprisal alignment as well.
`condor/olmo_report.py` also reports the localization gap on the subset where
both base models independently rank the plant as most surprising.

Start with 32-prompt generation pilots for brainstorming and rest. Check model
load, response lengths, plant yield, token alignment, truncation, wall time,
GPU memory, and credits before generating both full categories. Only train
after both metadata files have a non-null `finished_at`. Run 20-step fine-tune
pilots on an H100 for both Olmo and Qwen3.8, then full training and evaluation
if the pilots fit. Use `--hardware h100_96` for Qwen3.8 to select the 95,320 MiB
H100 nodes. Keep BF16 LoRA for both; an H100 OOM on Qwen3.8 should be addressed
by reducing its batch size rather than silently switching one model to 4-bit
training.

Full fine-tunes save a resumable checkpoint every 100 optimizer steps, keeping
the two most recent checkpoints. The submit helper defaults to an H100
(minimum bid 35). Its generated `.sub` files should be inspected before
submission, and the live credit balance and running-job list checked. The
wrapper refuses to overwrite existing output.

```bash
python3 condor/cross_submit.py generate olmo olmo --category brainstorming --pilot --pilot-batch 8
python3 condor/cross_submit.py generate olmo olmo --category rest --pilot --pilot-batch 8
python3 condor/cross_submit.py generate olmo olmo --category brainstorming
python3 condor/cross_submit.py generate olmo olmo --category rest
python3 condor/cross_submit.py train olmo olmo --pilot
python3 condor/cross_submit.py train qwen38 olmo --pilot --hardware h100_96
python3 condor/cross_submit.py train olmo olmo
python3 condor/cross_submit.py train qwen38 olmo --hardware h100_96
python3 condor/cross_submit.py evaluate olmo olmo
python3 condor/cross_submit.py evaluate qwen38 olmo --hardware h100_96
```

Pass `--submit` only after each prerequisite and pilot check passes.
