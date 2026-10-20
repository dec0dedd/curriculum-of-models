# Curriculum of Models

Reproduction code for the paper. An investigator LLM is trained with RL to
write prompts that elicit a prohibited behavior from a target LLM, scored by an
LLM judge. Three methods are reproduced:

| Method | Reward | What the target is |
|---|---|---|
| **CoM** (ours) | `curriculum` | a ladder of refusal-free LoRA adapters, hardest stage last |
| **Black-box** | `curriculum`, one stage | the base target only, from an SFT warm start |
| **PRBO** | `combined_estimator` | the base target, with a refusal-free proposal distribution |

Both targets from the paper are supported (Gemma-4-31B-it and gpt-oss-120b),
with reasoning on or off.

## Layout

Each directory under `packages/` is an **independent uv project** with its own
`pyproject.toml`, `uv.lock` and `.venv`. They share code through `common`,
wired in as an editable path dependency. There is deliberately no uv
workspace, which would force a single shared environment.

| Package | Stage | What it is |
|---|---|---|
| [`common`](packages/common) | — | behaviors, prompts, the judge, scoring, the target client |
| [`target_server`](packages/target_server) | 1 | serves the target and its LoRA ladder (vLLM, GPU) |
| [`investigator_trainer`](packages/investigator_trainer) | 2 | the RL training: CoM, Black-box, PRBO |
| [`cbrn_eval`](packages/cbrn_eval) | 3 | evaluates a checkpoint, exports results as CSV |

`data/behaviors/` is shared: the trainer and the eval both read it.

The refusal-free LoRA adapters themselves (Gemma and gpt-oss) are pre-trained
and not part of this repo; `target_server`'s ladder builder downloads them
from HF by default (`noahrossi/gemma-4-31b-it-refusal-free-v2` and
`noahrossi/gpt-oss-120b-refusal-r1-vllm`).

### Why the packages are separate

Not only because the stacks are heavy. They have a real conflict: vLLM 0.29
requires `transformers>=5.10.4`, and `tinker-cookbook` declares
`transformers<=5.5.4`. One environment for both can only be resolved with an
override that puts `tinker-cookbook` outside the range it declares supported.

Splitting along the boundary that already exists removes the conflict instead
of suppressing it: the trainer and the eval reach the target **over HTTP**, so
vLLM never has to be importable beside them. `target_server` holds vLLM and
the fork pin; every other package resolves `transformers 5.5.4` with no
override at all.

`common` therefore carries a `constraint-dependencies` ceiling matching what
its consumers resolve, so its own tests cannot pass against a version no stage
runs.

## Setup

```bash
cp .env.example .env     # OPENROUTER_API_KEY, TINKER_API_KEY, HF_TOKEN, WANDB_API_KEY
make sync                # every CPU package
make sync-server         # vLLM, Linux + CUDA only
make test                # 163 tests
```

Per package: `cd packages/<pkg> && uv sync && uv run pytest`. Don't run
`uv run pytest` from the repo root — there is no root project.

## Reproducing a run

All three stages work today.

**1. Serve the target** in its own shell. The script downloads the
pre-trained refusal-free adapter from HF and builds the ladder from it on
first run, then starts vLLM with every stage registered:

```bash
cd packages/target_server
uv sync --group vllm              # once; Linux + CUDA
./scripts/serve_gemma.sh          # or ./scripts/serve_gpt_oss.sh
```

Defaults are two GPUs (`TP=2 DP=1`); the paper's Gemma runs used four as
`TP=2 DP=2`. One GPU is `TP=1 CUDA_VISIBLE_DEVICES=0`. `REASONING=on` serves
Gemma with thinking enabled and raises the context accordingly — `MAX_MODEL_LEN`
has to cover the investigator prompt plus `target_max_tokens`, because vLLM
rejects a request that exceeds it rather than truncating.

To build the ladder by itself, or to build only the stages you need:

```bash
uv run build-ladder --scales 1.0,0.8,0.6,0.4,0.2,0.0
uv run build-ladder --source-repo noahrossi/gpt-oss-120b-refusal-r1-vllm \
    --output-dir scaled_target_loras_gptoss
```

**2. Train.** A run names one method and one target, and any `field=value`
after them overrides both:

```bash
cd packages/investigator_trainer
uv run com-train methods/com.yaml      targets/gemma.yaml   behavior_ids=transluce_tabun
uv run com-train methods/prbo.yaml     targets/gpt_oss.yaml behavior_ids=transluce_hmtd
uv run com-train methods/blackbox.yaml targets/gemma.yaml   behavior_ids=transluce_vx \
    load_checkpoint_path=tinker://dc6e07a3-5e75-5480-bd95-57ddd75a4394:train:0/weights/sft-final
```

Black-box needs `load_checkpoint_path`: the paper's investigator SFT warm start
above, which every Black-box run in the paper used.

A plain argument is a config file, resolved as given or else against
[`configs/`](configs/); anything with an `=` is a field override. The methods
are [`com`, `blackbox`, `prbo`, `prbo_plain`](configs/methods/) and the targets
[`gemma`, `gemma_reasoning`, `gpt_oss`](configs/targets/). Those files carry
the paper's per-method values (KL penalty, the truncation gate) and
per-target values (model, token budget, chat template),
each written in exactly one place. A later config beats an earlier one, so a
third file is the place for local settings like `vllm_base_url` — see
[`config_file.py`](packages/investigator_trainer/investigator_trainer/config_file.py).

A single-behavior run is one `behavior_ids`; leave it out to train on all 48.
`uv run com-train --help` lists the configs and every field.

Each run writes `runs/<name>/` with `run_metadata.json` (full config, git
state, the config files that produced it, command line), `rollouts.jsonl`,
and the curriculum stage files. While a run is going, or after it finishes:

```bash
cat runs/<name>/run_metadata.json | python -m json.tool | head -30   # what it was run with
tail -f runs/<name>/rollouts.jsonl | jq '.reward, .behavior_id'       # live reward per rollout
cat runs/<name>/curriculum_stage*.txt                                 # current CoM stage(s), if curriculum
```

**3. Evaluate** the resulting checkpoint, then export its scores:

```bash
cd packages/cbrn_eval
uv run com-eval sweep --investigator-model tinker://... --target-model google/gemma-4-31B-it
uv run com-eval export
```

`sweep` samples investigator prompts (or, with `--prompt-source=behaviors`,
sends each behavior verbatim as a direct-ask baseline), rolls them out
against the target, and judges them; `--behaviors-file`/`--behavior-ids`
mirror the trainer's. Results land in `eval_results/<run_id>_{full.parquet,
summary.json}`, where `run_id` hashes every result-affecting input, so a
completed run is not repeated and an interrupted one resumes. `export` then
writes one CSV per run plus an index, so the numbers can be tabulated with
whatever you like. Plotting lives in the paper repo, not here.

```bash
cat eval_results/exports/index.csv                    # one row per run: mean score, etc.
column -s, -t < eval_results/exports/<run_id>.csv | less -S   # the full per-prompt scores
```

## Useful commands

From the repo root:

```bash
make lint                          # ruff check, across every package
make format                        # ruff format, across every package
jq -r .behavior_id data/behaviors/transluce_cbrn.jsonl   # every behavior_id to pick from
```

Inside a package (`cd packages/<pkg>` first; each has its own `.venv`):

```bash
uv run pytest -k <name>            # a single test
uv run com-train --help            # investigator_trainer: methods/targets + every field
uv run com-eval sweep --help       # cbrn_eval: every sweep flag, with its default
uv run build-ladder --help         # target_server: ladder-builder flags
```

## Status

| Stage | Package | State |
|---|---|---|
| 1, serving | `target_server` | complete, 14 tests |
| 2, training | `investigator_trainer`, `common` | complete, 110 tests |
| 3, eval | `cbrn_eval` | complete, 39 tests |

## The paper's runs

For reference when reproducing: 157 CoM/Black-box runs, 77 PRBO (combined
estimator) and 5 plain PRBO, across both targets with reasoning on and off.
Single-behavior runs trained on one behavior, which here is
`behavior_ids=<one id>`. Evaluation sampled 256 investigator prompts per
behavior with one target completion each, with the prompt judge on. The
per-method values are in [`configs/methods/`](configs/methods/).
