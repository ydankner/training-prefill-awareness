# MPI-IS HTCondor: Qwen3-32B pilot

Inspected 2026-09-16. SSH works as `<user>@<login-node>`
(resolved to one of the login nodes). Compute uses the shared home directory; download on login.

## Cross-model follow-up

The later Qwen3-32B, Qwen3.8-27B, and Olmo 3 7B experiments use isolated
cluster checkouts so they do not disturb the original run. Their experiment
matrix, shared controls, submission steps, measured held-out results,
limitations, and Olmo-specific method are together in
[results/CROSS_MODEL.md](../results/CROSS_MODEL.md). The sections below describe
the original Qwen3-32B pilot and its follow-up analyses.

The new cross-model files are:

| File | Purpose |
| --- | --- |
| `condor/cross_submit.py` and `condor/cross_run.sh` | Prepare optional Condor submissions and run generation, LoRA training, or held-out evaluation for a chosen model and dataset source. Generated `.sub` files go under ignored `logs/cross/`. |
| `condor/cross_report.py` | Summarize completed model–dataset pairings and paired exact-localization gaps. |
| `src/prefill_awareness/cross_surprisal.py` | Score base and adapted models on planted answers, aligning plant character spans when source and evaluated tokenizers differ. |
| `condor/surprisal_transfer_submit.py` and `condor/surprisal_transfer.sh` | Pilot or full Qwen3.8 surprisal scoring on the original Qwen3-32B dataset. |
| `condor/qwen38_self_surprisal_submit.py` and `condor/qwen38_self_surprisal_run.sh` | Held-out base-versus-adapted Qwen3.8 scoring on its own dataset. |
| `condor/olmo_surprisal_submit.py` and `condor/olmo_surprisal_run.sh` | Held-out base-versus-adapted Olmo or Qwen3.8 scoring on Olmo's dataset. |
| `condor/olmo_report.py` | Paired Olmo-versus-Qwen3.8 localization comparison, optionally restricted to examples where both base models put highest surprisal on the plant. |

These jobs expect the dataset generation metadata to show `finished_at` before
dependent training or scoring. Submit files are prepared without launching a
job unless a helper is called with `--submit`; inspect the generated file and
cluster balance first. Dataset files, adapters, logs, and model weights remain
outside Git.

## Run configuration

Agreed settings for the first dataset: `Qwen/Qwen3-32B`, unquantized BF16,
non-thinking mode (`--no-enable-thinking`), greedy decoding for both the clean
response and the re-completion after the plant, one H100. The production
dataset size is not agreed yet; benchmark a larger batch first.

## Banking

`condor_bank balance` returned **15000**. The cluster documentation confirms a
2500 starter allocation, not that this account's current balance is 2500.
Source: the cluster's internal documentation.

Computing units = `max(request_cpus, requested_RAM_GiB / 16) + 24 * GPUs`.
The bid is credits per unit per hour, not per GPU-hour. The pilot's 4 CPUs,
96 GiB RAM and one GPU mean 30 units: at price 35, 1050 credits/hour.
The documented auction uses the next competing user's price, subject to node
minimums. The first-hour price is guaranteed; later effective bids/prices can
rise by 10% per hour under contention. Your submission bid is not a lifetime
spending cap. There is also a documented 0.1-credit submission fee.

Banking and quota tools need a Kerberos ticket. If it has expired, run `kinit`
interactively on the login node yourself.

Use short pilot runs and read `condor_bank jobs` for actual running charges.
The pilot has a 25-minute process timeout and a 30-minute Condor hold backstop.
At price 35, 30 minutes is about 525 credits plus submission fee (an estimate,
not a hard monetary cap). A timeout or other error holds the job for inspection.

## Exact inspection commands used

```bash
condor_status -help
condor_q -submitter $USER
condor_bank balance
condor_bank jobs
cluster_quota
condor_free_gpus
condor_nodes
condor_status -limit 1 -long
condor_status -af:h PartitionableSlot CUDADeviceName CUDAGlobalMemoryMb TotalGPUs GPUs MinRunningPrice -constraint PartitionableSlot
```

`condor_nodes` lists installed resources and remaining capacity. `condor_free_gpus`
filters this to free GPUs. These are site wrappers, with cached snapshots; free
counts are not a reservation or proof a job will match. The wrapper's markers:
`*` = a Special condition in START, `+` = full-node jobs, `-` = all child slots
retiring. In particular, do not count one node's eight free A100s as normal capacity.

Observed GPU families: Quadro RTX 6000 (24023 MiB), A100 40GB (40441 MiB),
A100 80GB (81153 MiB), H100 (95320 MiB), H100 80GB (81079 MiB), B200
(about 182632 MiB). Minimum bids: 15, 15, 25, 35, 35, 55 respectively;
full-node A100 nodes advertise 35. Memory figures come from Condor ads.

Also inspected local scripts for `condor_submit`, `condor_submit_bid`,
`condor_change_bid`, `condor_nodes`, `condor_free`, `condor_free_gpus`,
`condor_bank`, and `cluster_quota`, and Condor submission configuration.
`condor_submit_bid --help` is NOT a help command: it attempted to parse a
submission and failed with no queue statement; no job was submitted.
Bare `condor_bank` prints its two supported commands; `--help` is unsupported.

## Setup and launch

Dedicated environment: `~/prefill-venv`. Repository: `~/prefill-awareness`, a
Git checkout of https://github.com/ydankner/training-prefill-awareness. Commit and push
locally, then `git pull` on the login node; do not copy files by hand.
Data, logs, run outputs and `tmp/` there are ignored by Git.
Dependencies: torch 2.8.0, transformers 4.56.2, accelerate, datasets, peft.
Installation initially exceeded the login node's temporary-directory quota.
Use shared-home temporary space and avoid pip's redundant download cache:

```bash
mkdir -p ~/prefill-awareness/tmp
TMPDIR="$HOME/prefill-awareness/tmp" ~/prefill-venv/bin/pip install --no-cache-dir torch==2.8.0 transformers==4.56.2 accelerate datasets peft
```

Model download (started on a login node):

```bash
nohup ~/prefill-venv/bin/hf download Qwen/Qwen3-32B > ~/prefill-awareness/logs/download.log 2>&1 < /dev/null &
```

Never instantiate the model on the login node to download it. This uses the Hub
file downloader only. Cache: `~/.cache/huggingface`; the compute script sets
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` and uses local prompt JSONL.

Once installation and download finish:

```bash
cd ~/prefill-awareness
PYTHONPATH=src ~/prefill-venv/bin/python condor/prepare_prompts.py
/usr/bin/condor_submit -dry-run logs/pilot-dryrun.ads -append priority=-965 condor/pilot.sub
condor_submit_bid 35 condor/pilot.sub
condor_q -submitter $USER
condor_bank jobs
```

The dry run was validated successfully. The pilot requests 32 examples from
Dolly brainstorming, batch size 1, greedy decoding, 200-word instruction and a
512-token cap. This is a throughput/correctness pilot, not the final dataset.
Qwen3 non-thinking mode is explicit; each record stores that mode and model.
Surprisal scoring reads the record's thinking mode to reconstruct its prefix.
Training/evaluation chat-template handling still needs Qwen3-specific validation
before SFT; generation readiness does not imply training readiness.

## First launch (2026-09-16)

All 17 weight shards passed the offline preflight. Cached model revision:
`9216db5781bf21249d130ec9da846c4624c16137`.
The Python environment was installed and frozen in `logs/environment.txt`.
The first submission, 17566668, exited immediately because HOME was missing
from the job environment; it was removed with `condor_rm 17566668`. The submit
file now explicitly passes HOME. The corrected pilot is **17566671**, which
started on a compute node with an H100 (about 95,830 MiB reported by nvidia-smi).
The first job's bank entry confirmed bid 35, running price 35, slot weight 30.

Exact additional launch/inspection commands:

```bash
HF_HUB_OFFLINE=1 ~/prefill-venv/bin/python condor/preflight.py
~/prefill-venv/bin/pip freeze > logs/environment.txt
condor_submit_bid 35 condor/pilot.sub
condor_q 17566668 -af JobStatus RemoteHost HoldReason
condor_bank jobs
condor_rm 17566668
condor_submit_bid 35 condor/pilot.sub
condor_q 17566671 -af JobStatus RemoteHost HoldReason
```

### Pilot result

Job 17566671 completed: 32 records (16 modified, 16 clean), no truncated
responses, no `<think>` blocks, all planted probabilities inside (1e-4, 0.01).
Model loading took 2m53s and generation 6m39s at batch size 1. The records are
committed as `data/qwen3-32b-pilot-17566671.0-1789561019.jsonl` (force-added
past the `data/` ignore) so they can be inspected without cluster access.

## Training dataset

`condor/generate.sub` generates from all 1,751 usable Dolly brainstorming
prompts (1,766 in the category, before the length filter and deduplication),
with the pilot's settings at batch size 32. Prepare the prompts once on the
login node and commit the file:

```bash
PYTHONPATH=src ~/prefill-venv/bin/python condor/prepare_prompts.py --limit 100000 --out data/dolly-brainstorming-prompts.jsonl
condor_submit_bid 35 condor/generate.sub
```

The process timeout is 3 hours and the hold 3h15m. Records are flushed as they
are written, so watch the first chunks in `logs/generate-*.err` for throughput.

## Fine-tuning

`condor/train.sub` runs LoRA SFT of Qwen3-32B on both Dolly datasets
(`--data` takes several files): the default LoRA and optimiser settings (r=16,
lr 1e-4, 2 epochs, effective batch 16), no KL anchor, gradient checkpointing,
and in-loop evaluation on 128 test examples every 50 steps, including the
surprisal profile. The adapter, `eval_log.jsonl` and `meta.json` land in
`runs/sft-qwen3-32b-dolly-all-*`. The first run (job 17567334) trained on the
brainstorming dataset alone.

After each epoch a resumable `checkpoint-N` (adapter, optimizer, scheduler and
RNG state, about 1.6 GB) is written into the run directory; the first one sits
exactly at the end of epoch 1. The learning rate follows a cosine over both
epochs, so that checkpoint is not the same as a finished one-epoch run: its
learning rate is still about two thirds of the peak. If the job hits its limit (5
hours; hold at 5h15m), commit a `train.sh` change that passes the old run
directory as `--out` and its last checkpoint as `--resume-from`, then resubmit.

```bash
condor_submit_bid 35 condor/train.sub
```

## Checkpoint evaluation

### Ordinary-answer retention

`condor/capability_eval.sub` compares the base model and both checkpoints on
128 **existing held-out Dolly prompts**, sampled once from the two generated
datasets. The manifest contains the original user instruction (including the
200-word cap) and the stored context needed for 32 task-leakage probes. Each
model answers the ordinary instruction, then on those 32 examples answers the
tamper question and an ordinary follow-up. All generations use the recorded
non-thinking mode and greedy decoding. Results are written incrementally under
`runs/capability-dolly-128/`, so a held or interrupted job can resume without
rerunning completed examples. `summary.json` reports narrow task-leak flags;
`review-ordinary.csv` and `review-recovery.csv` randomize model identities per
question. Fill `score_0_to_2` blind: 0 = wrong, irrelevant, or stuck in the
tamper task; 1 = partly useful; 2 = correct and helpful. Keep `review-key.jsonl`
closed until scoring is finished. Then run
`python -m prefill_awareness.capability_eval score --out runs/capability-dolly-128`.
It averages only questions scored for all three models. A bare `none` count
alone is not a quality metric.

```bash
condor_submit_bid 35 condor/capability_eval.sub
```

### GSM8K retention

Prepare a fixed 200-question sample from the GSM8K test split on the **login
node**, where Hugging Face access works:

```bash
PYTHONPATH=src ~/prefill-venv/bin/python -m prefill_awareness.gsm8k_eval prepare \
  --out data/gsm8k-test-200.jsonl --n 200 --seed 0
condor_submit_bid 35 condor/gsm8k_eval.sub
```

The offline GPU job compares base, `checkpoint-554`, and `checkpoint-1108` on
identical zero-shot prompts with non-thinking mode and greedy decoding. It
writes raw answers incrementally, then `summary.json` with numeric final-answer
accuracy, unparsed counts, and truncation counts. The scoring parser requires
the requested `#### <number>` line; inspect unparsed answers before interpreting
an accuracy gap. `details.jsonl` retains every answer for paired error analysis.

`condor/eval_run.sub` runs `prefill_awareness.eval_run` on every checkpoint of
the second SFT run with a single model load: generation on the whole test split
and on an equally large random train sample (the gap measures memorisation), and
the surprisal profile of every modified test record with the adapter on and off,
including per-record plant ranks and the tokens that outrank the plant. Outputs
and an `eval-meta-*.json` provenance file land in the run directory. Timeout 2
hours; hold at 2h15m.

```bash
condor_submit_bid 35 condor/eval_run.sub
```

`condor/answer_probs.sub` runs the same module with `--no-generate
--no-surprisal --answer-probs`: for every test record, the teacher-forced
log-probability of the gold answer (and of `none` for modified records), the
log-probability and vocabulary rank of its first token, for both checkpoints and
the base model. Timeout 1 hour; hold at 1h15m.

```bash
condor_submit_bid 35 condor/answer_probs.sub
```

`condor/token_stats.sub` runs it with `--token-stats`: for every token of every
test answer, the surprisal, the entropy of the predictive distribution, the
probability of the most likely token and the rank of the actual token, for both
checkpoints and the base model. It tests whether fine-tuning sharpens the
model's predictions on the prefilled turn. Timeout 1 hour; hold at 1h15m.

```bash
condor_submit_bid 35 condor/token_stats.sub
```

## Hidden states

`condor/hidden_states.sub` runs `prefill_awareness.hidden_states` for the
mechanistic analysis. It stores the residual stream after the embedding and after
each of the 64 decoder layers, at every token of the tampered turn and at the
answer's generation header, for all 997 test records. It does this for the base
model and for `checkpoint-1108` of the second SFT run. The conversation is encoded
exactly as in training. Per-token surprisal, entropy and rank are recomputed in
that context, because Qwen3's template drops the empty `<think>` block from the
earlier turn. The output is `runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306/hidden-<job>/`,
about 124 GiB per model (200k rows × 65 states × 5,120 × 2 bytes). It stays on
the cluster; read it with `hidden_states.load_layer` from a CPU job. Timeout
2 hours; hold at 2h15m.

```bash
condor_submit_bid 35 condor/hidden_states.sub
```

`condor/hidden_pairs.sub` adds minimal pairs to that dump (`--pairs`). For each
of the 491 modified test records it cuts the conversation right after the plant
and runs it twice in one batch: once as written and once with the original token
put back. Everything before the plant is identical, so the two states at that
position differ only in whether the token is the one the model wrote. It stores
every layer for both, for both models, in `hidden-17569245.0/pairs/` (~1.3 GB).
Timeout 30 minutes; hold at 45 minutes.

```bash
condor_submit_bid 35 condor/hidden_pairs.sub
```

`condor/directions.sub` runs `prefill_awareness.directions` on the dump. It is a
CPU job with 4 cores and 48 GB (4 computing units). A bid of 5 was refused as
below the minimum; at 25 it costs 100 credits per hour. At every layer it computes difference-of-means plant directions
for both models: plant vs. the other tokens, one and two positions after the plant,
and the minimal pairs. Each direction is estimated on half of the records and
tested on the other half. It reports cosines, separation (AUROC) and correlation
with surprisal, and where the fine-tuned model's wrongly named tokens rank.
Timeout 3 hours; hold at 3h15m.

```bash
condor_submit_bid 25 condor/directions.sub
```

`condor/steer.sub` runs `prefill_awareness.steer` on `checkpoint-1108` with the
directions of job 17569599. It picks 3 clean test answers the model called
unchanged, and one plant-like word token in each. At that token only, and at each
of the 65 stored states in turn, it adds 0 to 32 times a direction:
- plant vs. rest, from the base model and from the fine-tuned model,
- the minimal pair, from each model,
- a random direction as long as the fine-tuned plant-vs-rest one.

It records the greedy answer and the teacher-forced log-probabilities of naming
the token and of `none`. Timeout 2 hours; hold at 2h15m.

```bash
condor_submit_bid 35 condor/steer.sub
```

## Attention

`condor/attention.sub` runs `prefill_awareness.attention` on `checkpoint-1108`
and the base model. All attention goes through one attention function registered
with transformers. It matches eager attention exactly, and it can record weights
and block individual heads from individual positions. The job has three parts:

- **Attention to the plant.** For all 997 test records, how much each head
  attends to the plant from the reply header's last position, and averaged over
  every position after the answer. On clean answers, the same for the token the
  model wrongly named.
- **Geometry, for every head.** The attention score from the last header
  position to the plant position, with keys built from the stored minimal-pair
  states: original token, plant, original plus only the plant's shift along the
  fine-tuned minimal-pair direction, plant minus that shift, and three random
  directions of the same length.
- **Attention under steering.** Attention to the steered token for the 3
  steering answers, at layers 32–63 and strengths 0–8.

It needs the `pairs/` and `directions-17569599.0.pt` of the hidden-state dump,
not its per-layer states. Those were deleted on 2026-09-17 to free 248 GiB;
`condor/hidden_states.sub` regenerates them in about 7 minutes. Timeout 2 hours;
hold at 2h15m.

```bash
condor_submit_bid 35 condor/attention.sub
```

`condor/knockout.sub` runs `prefill_awareness.knockout` on the 491 modified test
records of `checkpoint-1108`. It blocks attention to the plant in 37 settings:
- every head in layers >= L (12 cutoffs), from all later positions or only from
  the positions after the answer,
- the top 1-32 plant-attending heads of attention job 17570820,
- random control heads from the same layers.

For each setting it scores the gold answer and `none` teacher-forced. Timeout
3 hours; hold at 3h15m.

```bash
condor_submit_bid 35 condor/knockout.sub
```

## Ablations of the pipeline

Two variations on SFT run 2 (job 17568019). Each changes one thing and keeps
everything else: LoRA r16, lr 1e-4, 2 epochs, both Dolly datasets.

- **Keep-think.** Same data, but the tampered turn keeps its empty `<think>`
  block. Qwen3's template drops that block from an assistant turn that is
  followed by a user turn, so in run 2 the model read the turn in a different
  context from the one it was written in (see Hidden states). With the block
  kept, the tampered turn follows exactly its generation prefix
  (`dataset.render(..., keep_think=True)`, four more tokens).
- **Sampled.** New datasets: the clean answer and the re-completion after the
  plant are sampled at temperature 1 from the full distribution
  (`--top-p 1.0 --sample-top-k 0`) instead of greedily. `top_k` must be given:
  Qwen3's `generation_config` sets `top_k=20`, which would otherwise apply.
  Every record stores `surprisal_check`: the plant's rank among the turn's
  tokens by surprisal, scored once after the generation prefix and once as the
  turn is read in the detection chat (think block dropped). `rank_eligible`
  counts only tokens from `--min-pos` on, since the first tokens of a turn read
  without its think block are surprising for unrelated reasons. All records
  are used for training, whatever the plant's rank.
- **Plant-orig.** Same data, but the reply also names the token that was
  there originally: `changed -> original` (`--answer-format plant-orig`,
  `formats.apply_answer_format`; the question is rewritten to ask for it, and
  clean records still answer `none`). In greedy data the original is the
  model's own top prediction at that position, so the extra target asks it to
  compare what it predicted with what is written. The planted token stays
  first, so `loc_acc` is scored as before; `orig_acc` / `orig_acc_hit` score
  the original. Question: does it strengthen the signal and speed up learning?
  Compare the in-loop learning curves (the same 128 test records as run 2).

These jobs run in the checkout they are submitted from (`cd "$(dirname "$0")/.."`,
no `initialdir`), so they can run from a separate clone without touching
`~/prefill-awareness`. They were run from `~/prefill-awareness-ablations`.
Submit each from the root of the checkout it should run in.

| file | what it runs | timeout / hold |
| --- | --- | --- |
| `keepthink_train.sub` | `train --keep-think` on run 2's datasets, into `runs/sft-qwen3-32b-dolly-all-keepthink-<job>-<time>/` (job 17607642) | 5h / 5h15m |
| `keepthink_eval.sub` | `eval_run` on both checkpoints of that run: whole test split, an equal train sample, surprisal in both contexts. `keep_think` is read from the run's `meta.json` | 3h / 3h15m |
| `sampled_generate.sub` | 3 processes (job 17607643): process 0 regenerates every brainstorming prompt as job 17567181 did; processes 1–2 regenerate the other no-context categories as job 17567560 did, as shards 0 and 1 of 2 | 4h / 4h15m |
| `sampled_train.sub` | `train` on the three sampled files, into `runs/sft-qwen3-32b-dolly-all-sampled-<job>-<time>/` | 5h / 5h15m |
| `sampled_eval.sub` | three `eval_run`s: the sampled run on its own data; the sampled run on run 2's greedy data (tag `-greedydata`); run 2 on the sampled data (tag `-sampleddata`) | 8h hold |
| `sampled_probs.sub` | `eval_run --no-generate --no-surprisal --answer-probs --token-stats` on the sampled run and its test split (as run 2's `answer_probs` and `token_stats` jobs) | 2h / 2h15m |
| `sampled_hidden.sub` | One job: `hidden_states`, `hidden_states --pairs` and `directions` on the sampled run (base and checkpoint-1108, test split; wrong names from eval job 17608692), as run 2's jobs 17569245, 17569542 and 17569599. The ~270 GB dump goes to the node's local scratch (`request_disk = 320GB`): it does not fit the 400 GiB home quota, and the shared fast storage wrote it at ~4 MB/s. Only index, per-token stats, pairs and directions are copied to `runs/sft-...-sampled-.../hidden-<job>/`; the per-layer states are discarded with the scratch directory | 1h + 1h + 2h / 4h |
| `run2_sampled_hidden.sub` | Control: the same analysis for run 2's checkpoint-1108 on the sampled test answers (wrong names from its cross-evaluation, job 17608692), so both runs are compared on identical text; outputs in run 2's `hidden-sampleddata-<job>/` | 1h + 1h + 2h / 4h |
| `plantorig_train.sub` | `train --answer-format plant-orig` on run 2's datasets, into `runs/sft-qwen3-32b-dolly-all-plantorig-<job>-<time>/` | 5h / 5h15m |
| `plantorig_eval.sub` | `eval_run` on both checkpoints of that run, format read from its `meta.json` | 3h / 3h15m |

`eval_run` now also profiles surprisal in the reading context
(`<checkpoint>_reading` and `base_reading` in `surprisal-<tag>.json`), next to
the generation context used before.

For the last evaluation, run 2's checkpoints are symlinked into this clone's
copy of the run directory, so the outputs are written there and not into
`~/prefill-awareness`:

```bash
r=runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306
for c in checkpoint-554 checkpoint-1108; do ln -sfn ~/prefill-awareness/$r/$c $r/$c; done
```

## Caps generalization test

Does prefill-detection training stop a behavior learned in prefilled history
from transferring to the model's own history? A LoRA is trained to reply in
ALL CAPS, only in two-turn conversations whose first assistant turn is
prefilled. It is trained once on the base model and once on the base model with
run 2's detection adapter (checkpoint-1108) merged in (`train.py
--init-adapter`). Every checkpoint is then scored with the same second prompt
under different first turns: the prefilled one it was trained on, the model's
own clean answer to the same prompt, and none. Two measures: the greedy caps
rate, and the teacher-forced log ratio of the caps vs normal version of the
model's own reply (`caps_eval`).

The datasets are built from run 2's greedy datasets with `caps_data`; each
records its command in `.meta.json`. The human variant reads the Dolly authors'
answers from the Hugging Face cache, so it is built on the login node (a few
seconds):

```bash
python -m prefill_awareness.caps_data --out data/caps-qwen3-32b-dolly.jsonl \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl
python -m prefill_awareness.caps_data --history human --out data/caps-human-qwen3-32b-dolly.jsonl \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl
```

Both submit files queue two jobs, `base` and `ft`; each job trains (80 steps,
run 2's LoRA and optimiser settings, adapter-only checkpoints) and then runs
`caps_eval` in the same allocation. `initialdir` is `~/prefill-awareness-caps`,
a separate clone; the `ft` job reads run 2's checkpoint from
`~/prefill-awareness`.

| file | first turn in training | checkpoints scored | histories scored | timeout / hold |
| --- | --- | --- | --- | --- |
| `caps.sub` | the model's answer with one planted token | every 5 steps (jobs 17612555, stopped after step 45) | planted, own, none | 1h30m train + 3h eval / 4h30m |
| `caps_human.sub` | the Dolly author's answer to the same prompt | 0, 10, 16–26 every 2, 30, 40 (jobs 17613014) | human, planted, own, none | same |

```bash
cd ~/prefill-awareness-caps
condor_submit_bid 65 condor/caps.sub
condor_submit_bid 65 condor/caps_human.sub
```

Outputs: `runs/caps[-human]-qwen3-32b-<base|ft>-<job>-<time>/` with
`caps-eval-<job>.json` (summary per checkpoint and history),
`caps-samples-<job>.jsonl` (every reply and log ratio), their meta, and one
checkpoint per save step (~0.5 GB each). Paired comparison of the two jobs:

```bash
python -m prefill_awareness.caps_compare <base>/caps-samples-<job>.jsonl \
  <ft>/caps-samples-<job>.jsonl --trained planted   # or --trained human
```

Results were summarised in a private report; the run outputs and
checkpoints were deleted afterwards, so rerun the jobs to regenerate them.

## Tampering-marker experiment (Qwen3.8-27B)

Can a behavior be made to route through prefill detection? Every reply that
follows a tampered assistant turn starts with the marker `¦`, whatever the user
asked, and the fine-tune is then used for the caps test again. Model:
`Qwen/Qwen3.8-27B`, non-thinking mode for generation and training. Its chat
template keeps the (empty) think block of earlier assistant turns, so a history
turn is read after exactly the tokens it was generated after.

- `tamper_gen`: one sampled answer per Dolly no-context prompt (temperature 0.7,
  top-p 0.8, top-k 20, no presence penalty). Half are tampered with 1–4 plants:
  1/2 one plant, 1/4 two, 1/8 three, 1/8 four. For 2+ plants, half are
  consecutive runs and half scattered; a scattered plant goes into the first
  1/(plants left) of the answer remaining after the previous plant. Each prompt
  asks for the largest plant count still below its quota; a partial attempt
  counts as fewer plants (0 plants → clean). Plants have probability between
  1e-4 and 1%, lie outside the top-0.8 nucleus of the temperature-0.7
  distribution (up to 3 redraws), and are whole-word tokens anywhere in the
  answer (no padding). Output is appended per record; a rerun resumes.
- `tamper_data`: each history is used once. 60% get the detection question
  (`tamper_formats`: reply `¦tampered: a, b` or `not tampered`), 40% a No Robots
  request (`data/no-robots-prompts.jsonl`, built with `condor/prepare_no_robots.py`)
  answered by the model itself in that conversation, with the marker prepended
  after a tampered turn.
- `train --task tamper`: loss averaged per example, the first reply token as
  its own term, weighted 3× after a clean turn (against false alarms). The whole
  test split and a signal check (plant surprisal, KL on the history turn,
  adapter on vs off) are evaluated every 100 steps into `eval_log.jsonl`.
- `tamper_general`: the model's own GSM8K answers, history turns from other
  sources, and base vs adapter scoring (detection question, follow-up,
  held-out wordings, scope probe).
- `tamper_caps` + `train --caps-stop-at` + `caps_eval --marker`: ALL-CAPS after
  tampered history only (target `¦` + uppercased own answer), stopped once 50%
  of held-out replies are in caps, then scored under every history. The
  fine-tuned variant is re-evaluated on the detection test set afterwards.

`condor/tamper.sh` runs every phase from the checkout it is submitted from (it
was run from `~/prefill-awareness-tampermark`). Qwen3.8's linear-attention layers
run on transformers' reference PyTorch code: flash-linear-attention installed with
`pip install --target ~/tampermark-pylib --no-deps flash-linear-attention fla-core einops`
failed on the compute nodes because Triton could not compile its CUDA helper
(jobs 17616988/17616991); `-append 'fla = 1'` would put it back on the path. Minimum bid; 4 CPUs, 64 GB,
1 H100 = 28 units.

```bash
condor_submit_bid 35 condor/tamper.sub -append 'phase = pilot'
condor_submit_bid 35 condor/tamper.sub -append 'phase = histories'
condor_submit_bid 35 condor/tamper.sub -append 'phase = gsm8k'
condor_submit_bid 35 condor/tamper.sub -append 'phase = conversations'
condor_submit_bid 35 condor/tamper.sub -append 'phase = train main'
condor_submit_bid 35 condor/tamper.sub -append 'phase = general main checkpoint-N'
condor_submit_bid 35 condor/tamper.sub -append 'phase = caps_train base main checkpoint-N'
condor_submit_bid 35 condor/tamper.sub -append 'phase = caps_train ft main checkpoint-N'
python3 condor/tamper_report.py data/tamper-qwen38/histories.jsonl
```

## In-context learning baseline

`condor/icl.sub` runs `prefill_awareness.icl`: can the untuned model do the task
from examples alone? For each shot count n in 0, 1, 2, 4, 8, 16, 32, 64, the same
50 test records of run 2's data (25 tampered, 25 clean) are asked with n random
train records shown in the first user turn, before the real request. Each shot shows
the request, the assistant's answer (tampered or clean, unmarked) and the correct
reply; the detection question is quoted once in the header. The shots are drawn
fresh for every test record (half tampered, half clean for n >= 2). The test
record is then asked as in the regular evaluation. Besides the generated reply,
every item gets the teacher-forced log-probability of the gold reply and of `none`.
64 shots are about 15k tokens, inside Qwen3's 32k native context. Output:
`runs/icl-qwen3-32b-<job>-<time>/` with `summary.json`, `items.jsonl` and
`meta.json`. Timeout 1h30m; hold at 1h45m.

```bash
condor_submit_bid 35 condor/icl.sub
```

With the shots in the first turn, the tampered answer follows a different prefix
than the one it was generated after, so the plant's surprisal is diluted.
`condor/icl_question.sub` runs the same thing with `--placement question`: the
shots go into the follow-up user turn, before the detection question, and
everything up to and including the tampered turn is exactly as in the regular
evaluation. Same seed, so the same test records and the same shots per record as
`icl.sub`. Output: `runs/icl-question-qwen3-32b-<job>-<time>/`.

```bash
condor_submit_bid 35 condor/icl_question.sub
```

## Baseline evaluation

`condor/baseline.sub` runs `prefill_awareness.evaluate` with the untuned model
on every record of the pilot dataset (`--split all`) and writes metrics plus all
generations to `runs/`. It has a 20-minute process timeout and a 25-minute hold.

```bash
condor_submit_bid 35 condor/baseline.sub
```

Useful commands after submission (replace JOB_ID with the returned number):

```bash
condor_q JOB_ID -better-analyze
condor_q JOB_ID -af JobStatus HoldReason
condor_change_bid 40 JOB_ID
condor_rm JOB_ID
```

Changing a bid does not change the current charge of a running job. Never pass
your username to a mutation command unless you intend to affect all your jobs.

Qwen's model card: https://huggingface.co/Qwen/Qwen3-32B
32.8B parameters imply about 65.6 GB decimal / 61.1 GiB of BF16 weights alone.
An 80GB GPU is a reasonable small-batch starting point; profile peak memory
before increasing batching. A single 40GB GPU cannot hold BF16 weights. The
generator currently places the full model on one GPU, so requesting two 40GB
GPUs does not automatically solve this.
