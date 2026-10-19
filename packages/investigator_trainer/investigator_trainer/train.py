"""Train an investigator with CoM, Black-box, or PRBO.

    com-train preset=com,gemma behavior_ids=transluce_tabun
    com-train preset=prbo,gpt_oss behavior_ids=transluce_hmtd
    com-train preset=blackbox,gemma behavior_ids=transluce_vx load_checkpoint_path=tinker://...

Presets (``rl/presets.py``) apply in order, then every other ``field=value``
overrides them. A single-behavior run is ``behavior_ids=<one id>``; leave it
out to train on every behavior in ``behaviors_file``.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import chz
from chz.blueprint import EntrypointHelpException
from dotenv import load_dotenv
from tinker_cookbook import cli_utils
from tinker_cookbook.rl.train import Config, KLReferenceConfig
from tinker_cookbook.rl.train import main as tinker_train

from investigator_trainer.env import InvestigatorDatasetBuilder
from investigator_trainer.presets import METHODS, PRESETS, TARGETS
from investigator_trainer.rewards.prbo import DEFAULT_PROPOSAL_LORA
from investigator_trainer.rollout_log import save_run_metadata

RECIPE_NAME = "curriculum_of_models"
REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class TrainConfig:
    # ── Behaviors ─────────────────────────────────────────────────────────────
    # Set behavior_rubric (one inline behavior) or behaviors_file, not both.
    behavior_rubric: str | None = None
    behaviors_file: str | None = str(REPO_ROOT / "data/behaviors/transluce_cbrn.jsonl")
    # A subset of behaviors_file, by id. None = every row.
    behavior_ids: list[str] | None = None
    # Seeds the per-epoch behavior shuffle.
    behavior_seed: int = 0

    # ── Investigator (trained on Tinker) ──────────────────────────────────────
    investigator_model_name: str = "Qwen/Qwen3.5-9B"
    renderer_name: str = "qwen3_5_disable_thinking"
    # A /weights/ checkpoint to start from (the SFT warm start for Black-box).
    load_checkpoint_path: str | None = None
    # The KL reference. None = the starting checkpoint's /sampler_weights/
    # counterpart (or the base model when starting fresh).
    kl_reference_checkpoint_path: str | None = None

    # ── Target (served by vLLM, see scripts/serve_*.sh) ───────────────────────
    target_model_id: str = "google/gemma-4-31B-it"
    vllm_base_url: str = "http://localhost:8000"
    chat_template_kwargs: dict = chz.field(
        default_factory=lambda: {"enable_thinking": False}
    )
    target_max_tokens: int = 7000
    # Must be 1.0 for PRBO.
    target_temperature: float = 1.0
    # Target samples per investigator prompt; the reward averages over them.
    n_target_samples: int = 1

    # ── RL ────────────────────────────────────────────────────────────────────
    group_size: int = 64
    # Behaviors per step, one group each.
    batch_size: int = 1
    max_steps: int = 400
    learning_rate: float = 1e-5
    lora_rank: int = 32
    investigator_max_tokens: int = 512
    temperature: float = 1.0
    # Method-dependent; every method preset sets it.
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # ── Method (set by the method presets) ────────────────────────────────────
    reward_type: str = "curriculum"
    # Zero the reward of an investigator cut off at investigator_max_tokens.
    zero_reward_on_truncation: bool = False
    # Score regex-detected target refusals 0 without a judge call.
    judge_skip_refusals: bool = False
    curriculum_scales: list[float] = chz.field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    )
    curriculum_threshold: float = 60.0
    prbo_judge_scale: float = 1.0
    prbo_natural_response_threshold: float | None = None
    prbo_proposal_lora: str = DEFAULT_PROPOSAL_LORA

    # ── Logging and checkpoints ───────────────────────────────────────────────
    # Tinker checkpoints, metrics, rollouts.jsonl and run_metadata.json.
    # None = runs/<run_name>.
    log_path: str | None = None
    run_name: str | None = None
    wandb_project: str | None = None
    save_every: int = 50
    # Server-side checkpoint retention in seconds. None keeps them.
    ttl_seconds: int | None = None
    reasoning_max_chars: int = 5000
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def kl_reference_path(cfg: TrainConfig) -> str | None:
    """The KL reference must be a /sampler_weights/ path; tinker rejects /weights/."""
    if cfg.kl_reference_checkpoint_path is not None:
        return cfg.kl_reference_checkpoint_path
    if cfg.load_checkpoint_path is None:
        return None
    return cfg.load_checkpoint_path.replace("/weights/", "/sampler_weights/")


def validate(cfg: TrainConfig) -> None:
    if cfg.curriculum_scales == [0.0] and cfg.reward_type == "curriculum":
        if cfg.load_checkpoint_path is None:
            raise ValueError(
                "Black-box (curriculum_scales=[0.0]) starts from the SFT warm "
                "start: set load_checkpoint_path."
            )


async def train(cfg: TrainConfig) -> None:
    validate(cfg)
    run_name = cfg.run_name or (
        f"{cfg.reward_type}-{cfg.target_model_id.split('/')[-1]}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_path = cfg.log_path or str(REPO_ROOT / "runs" / run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    save_run_metadata(log_path, cfg)

    dataset_builder = InvestigatorDatasetBuilder(
        behavior_rubric=cfg.behavior_rubric,
        behaviors_file=None if cfg.behavior_rubric is not None else cfg.behaviors_file,
        behavior_ids=cfg.behavior_ids,
        behavior_seed=cfg.behavior_seed,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        max_steps=cfg.max_steps,
        investigator_model_name=cfg.investigator_model_name,
        renderer_name=cfg.renderer_name,
        investigator_max_tokens=cfg.investigator_max_tokens,
        target_model_id=cfg.target_model_id,
        vllm_base_url=cfg.vllm_base_url,
        target_max_tokens=cfg.target_max_tokens,
        target_temperature=cfg.target_temperature,
        chat_template_kwargs=cfg.chat_template_kwargs,
        n_target_samples=cfg.n_target_samples,
        judge_skip_refusals=cfg.judge_skip_refusals,
        reward_type=cfg.reward_type,
        zero_reward_on_truncation=cfg.zero_reward_on_truncation,
        curriculum_scales=cfg.curriculum_scales,
        curriculum_threshold=cfg.curriculum_threshold,
        prbo_judge_scale=cfg.prbo_judge_scale,
        prbo_natural_response_threshold=cfg.prbo_natural_response_threshold,
        prbo_proposal_lora=cfg.prbo_proposal_lora,
        log_dir=log_path,
        reasoning_max_chars=cfg.reasoning_max_chars,
    )
    config = Config(
        dataset_builder=dataset_builder,
        model_name=cfg.investigator_model_name,
        recipe_name=RECIPE_NAME,
        learning_rate=cfg.learning_rate,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.investigator_max_tokens,
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        # The reference is the policy's own starting point, so a warm-started
        # run measures drift from its warm start, not from the bare base model.
        kl_reference_config=(
            KLReferenceConfig(
                base_model=cfg.investigator_model_name,
                load_checkpoint_path=kl_reference_path(cfg),
            )
            if cfg.kl_penalty_coef > 0
            else None
        ),
        num_substeps=cfg.num_substeps,
        load_checkpoint_path=cfg.load_checkpoint_path,
        log_path=log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=run_name,
        save_every=cfg.save_every,
        ttl_seconds=cfg.ttl_seconds,
    )
    await tinker_train(config)


USAGE = f"""\
Train the investigator. Compose one method preset with one target preset:

    com-train preset=<method>,<target> [field=value ...]

  methods  {", ".join(METHODS)}
  targets  {", ".join(TARGETS)}

The presets set the paper's values for that combination; every field=value
after them overrides. Each field below can be set that way.

    com-train preset=com,gemma behavior_ids=transluce_tabun
    com-train preset=prbo,gpt_oss group_size=256 max_steps=200
"""


def parse_args(argv: list[str]) -> TrainConfig:
    """``preset=a,b`` applies presets a then b; every other arg overrides them."""
    blueprint = chz.Blueprint(TrainConfig)
    rest = []
    for arg in argv:
        if arg.startswith("preset="):
            for name in arg.removeprefix("preset=").split(","):
                if name not in PRESETS:
                    raise SystemExit(
                        f"Unknown preset {name!r}. Known: {', '.join(PRESETS)}"
                    )
                blueprint = blueprint.apply(PRESETS[name])
        else:
            rest.append(arg)
    return blueprint.apply_from_argv(rest).make()


def main() -> None:
    load_dotenv()
    try:
        cfg = parse_args(sys.argv[1:])
    except EntrypointHelpException as help_text:
        # chz prints its field table by raising. Lead with the preset usage,
        # which is the actual interface and is not a field.
        print(USAGE)
        print(help_text.args[0] if help_text.args else "")
        raise SystemExit(0) from None
    asyncio.run(train(cfg))


if __name__ == "__main__":
    main()
