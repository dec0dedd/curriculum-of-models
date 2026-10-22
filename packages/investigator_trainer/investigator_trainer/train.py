"""Train an investigator with CoM, Black-box, or PRBO.

    train methods/com.yaml targets/gemma.yaml behavior_ids=transluce_tabun

A run names one method and one target, resolved against ``configs/``. Those
files hold the values the paper's runs used; a later config beats an earlier
one, and every ``field=value`` after them beats all of them.

A single-behavior run is ``behavior_ids=<one id>``; leave it out to train on
every behavior in ``behaviors_file``.
"""

import asyncio
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import chz
from chz.blueprint import EntrypointHelpException
from dotenv import load_dotenv
from tinker_cookbook import cli_utils
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.train import Config, KLReferenceConfig
from tinker_cookbook.rl.train import main as tinker_train

from investigator_trainer.config_file import ConfigFileError, available, load_config
from investigator_trainer.env import InvestigatorDatasetBuilder
from investigator_trainer.rewards.prbo import DEFAULT_PROPOSAL_LORA
from investigator_trainer.rollout_log import save_run_metadata

RECIPE_NAME = "curriculum_of_models"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


@chz.chz
class TrainConfig:
    # behaviors
    # Set behavior_rubric (one inline behavior) or behaviors_file, not both.
    behavior_rubric: str | None = None

    behaviors_file: str | None = str(REPO_ROOT / "data/behaviors/transluce_cbrn.jsonl")

    # A subset of behaviors_file, by id. None = every row.
    behavior_ids: list[str] | None = None

    # Seeds the per-epoch behavior shuffle.
    behavior_seed: int = 0

    # investigator (trained on tinker)
    investigator_model_name: str = "Qwen/Qwen3.5-9B"

    renderer_name: str = "qwen3_5_disable_thinking"

    # A /weights/ checkpoint to start from (the SFT warm start for Black-box).
    load_checkpoint_path: str | None = None

    # KL reference. None = the start checkpoint's /sampler_weights/ path, or
    # the base model if starting fresh.
    kl_reference_checkpoint_path: str | None = None

    # target (served by vllm, see target_server/scripts/serve_*.sh)
    target_model_id: str = "google/gemma-4-31B-it"

    vllm_base_url: str = "http://localhost:8000"

    chat_template_kwargs: dict = chz.field(
        default_factory=lambda: {"enable_thinking": False}
    )

    target_max_tokens: int = 7000

    # rl
    group_size: int = 64

    # Behaviors per step, one group each.
    batch_size: int = 1

    max_steps: int = 400

    learning_rate: float = 1e-5

    lora_rank: int = 32

    investigator_max_tokens: int = 512

    temperature: float = 1.0

    # Method-dependent; every config in configs/methods/ sets it.
    kl_penalty_coef: float = 0.0

    num_substeps: int = 1

    # method (set by configs/methods/*.yaml)
    reward_type: str = "curriculum"

    curriculum_scales: list[float] = chz.field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    )

    curriculum_threshold: float = 60.0

    prbo_judge_scale: float = 1.0

    prbo_natural_response_threshold: float | None = None

    prbo_proposal_lora: str = DEFAULT_PROPOSAL_LORA

    # logging and checkpoints
    # Tinker checkpoints, metrics, rollouts.jsonl and run_metadata.json.
    # None = runs/<run_name>.
    log_path: str | None = None

    run_name: str | None = None

    wandb_project: str | None = None

    save_every: int = 50

    # Server-side checkpoint retention in seconds. None keeps them.
    ttl_seconds: int | None = None

    # A cheap crash-restore checkpoint every N steps (0 disables), so a crash
    # loses at most N steps rather than up to save_every. Only one is kept.
    rolling_save_every: int = 1

    # Retention for the rolling checkpoint: long enough to resume a crash
    # noticed the next day.
    rolling_ttl_seconds: int = 2 * 24 * 60 * 60

    # Failed rollouts (target error, unscorable PRBO sample) a group may retry,
    # in total, before the group is dropped from its step.
    rollout_max_retries: int = 3

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


async def train(cfg: TrainConfig, config_files: Sequence[str | Path] = ()) -> None:
    validate(cfg)
    run_name = cfg.run_name or (
        f"{cfg.reward_type}-{cfg.target_model_id.split('/')[-1]}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_path = cfg.log_path or str(REPO_ROOT / "runs" / run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    save_run_metadata(log_path, cfg, config_files=config_files)

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
        chat_template_kwargs=cfg.chat_template_kwargs,
        reward_type=cfg.reward_type,
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
        # Reference is where this run started, so a warm-started run measures
        # drift from its warm start rather than the base model.
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
        rolling_save_every=cfg.rolling_save_every,
        rolling_ttl_seconds=cfg.rolling_ttl_seconds,
        # A failed rollout gets no stand-in reward: it is rerun with a fresh
        # env, and a group that runs out of retries is skipped.
        rollout_error_tolerance=RetryOnFailure(max_retries=cfg.rollout_max_retries),
    )
    await tinker_train(config)


def usage() -> str:
    methods = ", ".join(available(CONFIGS_DIR / "methods")) or "(none found)"
    targets = ", ".join(available(CONFIGS_DIR / "targets")) or "(none found)"
    return f"""\
Train the investigator. Name one method and one target, then any overrides:

    train methods/<method>.yaml targets/<target>.yaml [field=value ...]

  methods   {methods}
  targets   {targets}

A plain argument is a config file (resolved as given, else under {CONFIGS_DIR}).
A later config beats an earlier one, and every field=value beats them all; each
field listed below can be set that way.

Examples:

    train methods/com.yaml  targets/gemma.yaml   behavior_ids=transluce_tabun
    train methods/prbo.yaml targets/gpt_oss.yaml group_size=256
"""


def parse_args(argv: list[str], config_files: list[Path] | None = None) -> TrainConfig:
    """A plain argument is a config file; ``field=value`` and flags go to chz.

    If given, ``config_files`` collects each resolved config path, in the
    order applied, for ``run_metadata.json`` provenance.
    """
    blueprint = chz.Blueprint(TrainConfig)
    fields = set(TrainConfig.__chz_fields__)
    rest: list[str] = []
    for arg in argv:
        if arg.startswith("-") or "=" in arg:
            rest.append(arg)
            continue
        path = Path(arg)
        if not path.exists():
            path = CONFIGS_DIR / arg
        try:
            blueprint = blueprint.apply(load_config(path, fields))
        except ConfigFileError as exc:
            raise SystemExit(str(exc)) from None
        if config_files is not None:
            config_files.append(path.resolve())
    return blueprint.apply_from_argv(rest).make()


def main() -> None:
    load_dotenv()
    config_files: list[Path] = []
    try:
        cfg = parse_args(sys.argv[1:], config_files)
    except EntrypointHelpException as help_text:
        # chz signals --help by raising. Print our own usage first, then its
        # field table.
        print(usage())
        print(help_text.args[0] if help_text.args else "")
        raise SystemExit(0) from None
    asyncio.run(train(cfg, config_files))


if __name__ == "__main__":
    main()
