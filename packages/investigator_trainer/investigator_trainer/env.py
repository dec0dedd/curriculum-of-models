"""The tinker RL environment: one investigator rollout per Env, one behavior per group.

GRPO centres rewards within a group, so a group holds a single behavior;
across behaviors the advantage would measure how hard each behavior is, which
the policy cannot choose. A step covers ``batch_size`` behaviors, scheduled by
``behaviors_for_step``.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chz
import httpx
from tinker.types import ModelInput
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from common.behaviors import Behavior, behaviors_for_step, resolve_behaviors
from common.judge import Judge
from common.prompts import INVESTIGATOR_SYSTEM_PROMPT
from investigator_trainer.rewards import (
    CombinedEstimatorReward,
    CurriculumReward,
    PRBOReward,
    RewardFormula,
    RewardResult,
)
from investigator_trainer.rewards.prbo import DEFAULT_PROPOSAL_LORA
from investigator_trainer.rollout_log import RolloutLog
from common.target import VLLMTarget

logger = logging.getLogger(__name__)

REWARD_TYPES = ("curriculum", "prbo", "combined_estimator")
PRBO_REWARD_TYPES = ("prbo", "combined_estimator")

# A target response this close to target_max_tokens counts as clipped.
CLIP_MARGIN_TOKENS = 5


class InvestigatorEnv(Env):
    """Single-use environment for one investigator rollout."""

    def __init__(
        self,
        *,
        env_id: int,
        step: int,
        renderer: Renderer,
        behavior: Behavior,
        target: VLLMTarget,
        judge: Judge,
        reward_formula: RewardFormula,
        n_target_samples: int,
        investigator_max_tokens: int,
        zero_reward_on_truncation: bool,
        rollout_log: RolloutLog,
    ):
        self.env_id = env_id
        self.step_index = step
        self.renderer = renderer
        self.behavior = behavior
        self.target = target
        self.judge = judge
        self.reward_formula = reward_formula
        self.n_target_samples = n_target_samples
        self.investigator_max_tokens = investigator_max_tokens
        self.zero_reward_on_truncation = zero_reward_on_truncation
        self.rollout_log = rollout_log

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        system = INVESTIGATOR_SYSTEM_PROMPT.format(
            behavior_rubric=self.behavior.behavior
        )
        model_input = self.renderer.build_generation_prompt(
            [{"role": "system", "content": system}]
        )
        return model_input, self.renderer.get_stop_sequences()

    async def step(self, action: Action, **kwargs) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        clean = bool(termination.is_clean)
        investigator_prompt = message["content"]
        if isinstance(investigator_prompt, list):
            investigator_prompt = "".join(
                b.get("text", "")
                for b in investigator_prompt
                if b.get("type") == "text"
            )
        investigator_clipped = len(action) >= self.investigator_max_tokens - 1

        result: RewardResult | None = None
        error: str | None = None
        try:
            result = await self.reward_formula.compute_reward(
                investigator_prompt=investigator_prompt,
                behavior=self.behavior.behavior,
                target=self.target,
                judge=self.judge,
                n_target_samples=self.n_target_samples,
            )
        except Exception as e:
            # A failed target or judge call, or a PRBO sample that cannot be
            # scored token by token. Zero this rollout rather than fail the run.
            logger.exception(f"[step {self.step_index}] rollout {self.env_id} failed")
            error = repr(e)

        reward = result.reward if result is not None else 0.0
        # CoM's gate: an investigator cut off at its token budget earns nothing.
        if self.zero_reward_on_truncation and not clean:
            reward = 0.0

        self.rollout_log.write(
            step=self.step_index,
            env_id=self.env_id,
            behavior_id=self.behavior.behavior_id,
            investigator_prompt=investigator_prompt,
            reward=reward,
            result=result,
            investigator_clipped=investigator_clipped,
            error=error,
        )

        metrics: dict[str, float] = dict(result.metrics) if result is not None else {}
        metrics.update(
            {
                "reward/trained": reward,
                "target/rollout_failed": float(result is None),
                "target/is_clipped": self._clipped_fraction(result),
                "investigator/is_clipped": float(investigator_clipped),
                "investigator/clean_termination": float(clean),
                "investigator/completion_tokens": float(len(action)),
                "investigator/prompt_length_words": float(
                    len(investigator_prompt.split())
                ),
            }
        )
        # A copy suffixed with the behavior: tinker averages each key over the
        # rollouts that carry it, so `<key>/<behavior_id>` is that behavior's
        # own mean, at any batch size.
        metrics |= {
            f"{k}/{self.behavior.behavior_id}": v for k, v in list(metrics.items())
        }

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt([]),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )

    def _clipped_fraction(self, result: RewardResult | None) -> float:
        """Share of target samples that ran into target_max_tokens.

        The judge gives a cut-off response full credit, so this is the only
        sign that target_max_tokens is too low.
        """
        if result is None or not result.responses:
            return 0.0
        limit = self.target.max_tokens - CLIP_MARGIN_TOKENS
        clipped = sum(
            len(self.target.tokenizer.encode(r)) >= limit for r in result.responses
        )
        return clipped / len(result.responses)


@dataclass
class InvestigatorGroupBuilder(EnvGroupBuilder):
    step: int
    behavior: Behavior
    group_size: int
    renderer: Renderer
    target: VLLMTarget
    judge: Judge
    reward_formula: RewardFormula
    n_target_samples: int
    investigator_max_tokens: int
    zero_reward_on_truncation: bool
    rollout_log: RolloutLog

    async def make_envs(self) -> Sequence[InvestigatorEnv]:
        return [
            InvestigatorEnv(
                env_id=i,
                step=self.step,
                renderer=self.renderer,
                behavior=self.behavior,
                target=self.target,
                judge=self.judge,
                reward_formula=self.reward_formula,
                n_target_samples=self.n_target_samples,
                investigator_max_tokens=self.investigator_max_tokens,
                zero_reward_on_truncation=self.zero_reward_on_truncation,
                rollout_log=self.rollout_log,
            )
            for i in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        # tinker writes env/<tag>/<key> for every tag that is a strict subset
        # of the step's groups (so only at batch_size > 1).
        return [self.behavior.behavior_id, self.behavior.category or "uncategorized"]


@dataclass
class InvestigatorDataset(RLDataset):
    behaviors: list[Behavior]
    batch_size: int
    behavior_seed: int
    max_steps: int
    group_size: int
    renderer: Renderer
    target: VLLMTarget
    judge: Judge
    # Keyed by behavior_id. CoM keeps a ladder per behavior; the PRBO formulas
    # are stateless and share one instance.
    reward_formulas: dict[str, RewardFormula]
    n_target_samples: int
    investigator_max_tokens: int
    zero_reward_on_truncation: bool
    rollout_log: RolloutLog

    def get_batch(self, index: int) -> Sequence[InvestigatorGroupBuilder]:
        # Must depend on `index` only: tinker may call it from rollout workers.
        return [
            InvestigatorGroupBuilder(
                step=index + 1,
                behavior=behavior,
                group_size=self.group_size,
                renderer=self.renderer,
                target=self.target,
                judge=self.judge,
                reward_formula=self.reward_formulas[behavior.behavior_id],
                n_target_samples=self.n_target_samples,
                investigator_max_tokens=self.investigator_max_tokens,
                zero_reward_on_truncation=self.zero_reward_on_truncation,
                rollout_log=self.rollout_log,
            )
            for behavior in behaviors_for_step(
                self.behaviors,
                index=index,
                batch_size=self.batch_size,
                seed=self.behavior_seed,
            )
        ]

    def __len__(self) -> int:
        return self.max_steps


def build_reward_formulas(
    *,
    reward_type: str,
    behaviors: Sequence[Behavior],
    group_size: int,
    log_dir: str | None,
    curriculum_scales: Sequence[float],
    curriculum_threshold: float,
    prbo_judge_scale: float,
    prbo_natural_response_threshold: float | None,
    prbo_proposal_lora: str,
) -> dict[str, RewardFormula]:
    if reward_type == "curriculum":
        single = len(behaviors) == 1
        return {
            b.behavior_id: CurriculumReward(
                curriculum_scales,
                threshold=curriculum_threshold,
                rollouts_per_step=group_size,
                stage_file=(
                    Path(log_dir)
                    / (
                        "curriculum_stage.txt"
                        if single
                        else f"curriculum_stage_{b.behavior_id}.txt"
                    )
                    if log_dir
                    else None
                ),
                label=b.behavior_id,
            )
            for b in behaviors
        }
    if reward_type == "prbo":
        formula: RewardFormula = PRBOReward(
            judge_scale=prbo_judge_scale, proposal_lora=prbo_proposal_lora
        )
    elif reward_type == "combined_estimator":
        formula = CombinedEstimatorReward(
            judge_scale=prbo_judge_scale,
            natural_response_threshold=prbo_natural_response_threshold,
            proposal_lora=prbo_proposal_lora,
        )
    else:
        raise ValueError(
            f"Unknown reward_type {reward_type!r}; expected one of {REWARD_TYPES}"
        )
    return {b.behavior_id: formula for b in behaviors}


@chz.chz
class InvestigatorDatasetBuilder(RLDatasetBuilder):
    """Built inside tinker's training loop; see train.TrainConfig for each field."""

    behavior_rubric: str | None
    behaviors_file: str | None
    behavior_ids: list[str] | None
    behavior_seed: int
    batch_size: int
    group_size: int
    max_steps: int
    investigator_model_name: str
    renderer_name: str
    investigator_max_tokens: int
    target_model_id: str
    vllm_base_url: str
    target_max_tokens: int
    target_temperature: float
    chat_template_kwargs: dict
    n_target_samples: int
    reward_type: str
    zero_reward_on_truncation: bool
    curriculum_scales: list[float]
    curriculum_threshold: float
    prbo_judge_scale: float
    prbo_natural_response_threshold: float | None
    prbo_proposal_lora: str = DEFAULT_PROPOSAL_LORA
    log_dir: str | None = None
    reasoning_max_chars: int = 5000

    async def __call__(self) -> tuple[InvestigatorDataset, None]:
        if self.reward_type in PRBO_REWARD_TYPES and self.target_temperature != 1.0:
            raise ValueError(
                "PRBO needs target_temperature=1.0: q's sampled log-probs and p's "
                "replayed ones must both describe the unscaled distributions."
            )
        behaviors = resolve_behaviors(
            behavior_rubric=self.behavior_rubric,
            behaviors_file=self.behaviors_file,
            behavior_ids=self.behavior_ids,
        )
        logger.info(
            f"{self.reward_type}: {len(behaviors)} behavior(s), {self.batch_size} per "
            f"step, {self.group_size} rollouts each"
        )

        async with httpx.AsyncClient() as client:
            try:
                (
                    await client.get(f"{self.vllm_base_url}/health", timeout=60)
                ).raise_for_status()
            except Exception as e:
                raise RuntimeError(
                    f"vLLM server not reachable at {self.vllm_base_url}"
                ) from e

        renderer = get_renderer(
            name=self.renderer_name,
            tokenizer=get_tokenizer(self.investigator_model_name),
        )
        target = VLLMTarget(
            model_id=self.target_model_id,
            base_url=self.vllm_base_url,
            max_tokens=self.target_max_tokens,
            temperature=self.target_temperature,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        reward_formulas = build_reward_formulas(
            reward_type=self.reward_type,
            behaviors=behaviors,
            group_size=self.group_size,
            log_dir=self.log_dir,
            curriculum_scales=self.curriculum_scales,
            curriculum_threshold=self.curriculum_threshold,
            prbo_judge_scale=self.prbo_judge_scale,
            prbo_natural_response_threshold=self.prbo_natural_response_threshold,
            prbo_proposal_lora=self.prbo_proposal_lora,
        )
        dataset = InvestigatorDataset(
            behaviors=behaviors,
            batch_size=self.batch_size,
            behavior_seed=self.behavior_seed,
            max_steps=self.max_steps,
            group_size=self.group_size,
            renderer=renderer,
            target=target,
            judge=Judge(),
            reward_formulas=reward_formulas,
            n_target_samples=self.n_target_samples,
            investigator_max_tokens=self.investigator_max_tokens,
            zero_reward_on_truncation=self.zero_reward_on_truncation,
            rollout_log=RolloutLog(self.log_dir, self.reasoning_max_chars),
        )
        return dataset, None
