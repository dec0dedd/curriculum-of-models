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
| [`common`](packages/common) | — | behaviors, prompts, the judge, scoring, refusal detection, the target client |
| [`sft_lora_pipeline`](packages/sft_lora_pipeline) | 1 | Gemma refusal-abliteration LoRA (trl + peft, GPU) |
| [`sft_investigator`](packages/sft_investigator) | 1 | investigator warm start, and the gpt-oss refusal LoRA (Tinker) |
| [`target_server`](packages/target_server) | 2 | serves the target and its LoRA ladder (vLLM, GPU) |
| [`investigator_trainer`](packages/investigator_trainer) | 3 | the RL training: CoM, Black-box, PRBO |
| [`cbrn_eval`](packages/cbrn_eval) | 4 | evaluates a checkpoint, exports results as CSV |

`data/behaviors/` is shared: the trainer and the eval both read it.

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
make test                # 54 tests
```

Per package: `cd packages/<pkg> && uv sync && uv run pytest`. Don't run
`uv run pytest` from the repo root — there is no root project.

## Reproducing a run

Stage 3 works today. Stages 1, 2 and 4 are still being ported, and their
commands below are the intended interface, not yet a working one. Until then,
train against a target you serve yourself with vLLM at `vllm_base_url`, with
the ladder adapters registered under the names `alpha_1_0`, `alpha_0_8`, ... .

**1. Serve the target** in its own shell, with the LoRA ladder loaded
(*not yet ported*):

```bash
cd packages/target_server
./scripts/serve_gemma.sh          # or ./scripts/serve_gpt_oss.sh
```

**2. Train.** A method preset and a target preset compose, and any
`field=value` after them overrides:

```bash
cd packages/investigator_trainer
uv run com-train preset=com,gemma  behavior_ids=transluce_tabun
uv run com-train preset=prbo,gpt_oss behavior_ids=transluce_hmtd
uv run com-train preset=blackbox,gemma behavior_ids=transluce_vx \
    load_checkpoint_path=tinker://...
```

Methods are `com`, `blackbox`, `prbo`, `prbo_plain`; targets are `gemma`,
`gemma_reasoning`, `gpt_oss`. The presets carry the paper's per-method values
(KL penalty, the judge's refusal skip, the truncation gate) — see
[`presets.py`](packages/investigator_trainer/investigator_trainer/presets.py).
A single-behavior run is one `behavior_ids`; leave it out to train on all 48.

Each run writes `runs/<name>/` with `run_metadata.json` (full config, git
state, command line), `rollouts.jsonl`, and the curriculum stage files.

**3. Evaluate** the resulting checkpoint, then export its scores
(*not yet ported*):

```bash
cd packages/cbrn_eval
uv run com-eval sweep --investigator-model tinker://... --target-model google/gemma-4-31B-it
uv run com-eval export
```

That writes one CSV per run plus an index, so the numbers can be tabulated
with whatever you like. Plotting lives in the paper repo, not here.

## Status

| Stage | Package | State |
|---|---|---|
| 3, training | `investigator_trainer`, `common` | complete, 74 tests |
| 1, SFT | `sft_investigator`, `sft_lora_pipeline` | packaged and pinned, not yet ported |
| 2, serving | `target_server` | packaged and pinned, not yet ported |
| 4, eval | `cbrn_eval` | config and run identity only |

## The paper's runs

For reference when reproducing: 157 CoM/Black-box runs, 77 PRBO (combined
estimator) and 5 plain PRBO, across both targets with reasoning on and off.
Single-behavior runs trained on one behavior, which here is
`behavior_ids=<one id>`. Evaluation sampled 256 investigator prompts per
behavior with one target completion each, with the prompt judge on. The
per-method values are in
[`presets.py`](packages/investigator_trainer/investigator_trainer/presets.py).
