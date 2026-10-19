"""Curriculum of Models (CoM): the target climbs a ladder of refusal-free adapters.

Stage k samples the target under the adapter of ``scales[k]`` (1.0 = fully
refusal-free, 0.0 = the base model). Once a behavior's mean reward over one
training step reaches the threshold, that behavior moves to the next, harder
stage. Each behavior holds its own instance and stage file: behaviors differ
widely in difficulty, and a shared stage would advance on the easiest ones.

With ``scales=[0.0]`` there is one stage, the base model, which is the
Black-box baseline.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from common.judge import Judge
from common.refusal import is_investigator_refusal
from investigator_trainer.rewards.base import (
    RewardFormula,
    RewardResult,
    base_metrics,
    judge_rollout,
)
from common.target import VLLMTarget

logger = logging.getLogger(__name__)


def scale_to_adapter_name(scale: float) -> str | None:
    """The vLLM LoRA name served for a ladder scale; None is the base model.

    Must match scripts/build_lora_ladder.py: ``alpha_<scale>`` with ``.`` as
    ``_``, e.g. 0.6 -> ``alpha_0_6``. Rounding strips float noise.
    """
    if scale == 0.0:
        return None
    return f"alpha_{str(round(float(scale), 10)).replace('.', '_')}"


def validate_curriculum_scales(scales: Sequence[float]) -> None:
    """Non-empty, in [0, 1], strictly decreasing (easiest first)."""
    if not scales:
        raise ValueError("curriculum_scales must contain at least one scale.")
    if any(not 0.0 <= s <= 1.0 for s in scales):
        raise ValueError(
            f"Every curriculum scale must be in [0, 1]; got {list(scales)}."
        )
    if any(a <= b for a, b in zip(scales, scales[1:])):
        raise ValueError(
            "curriculum_scales must be strictly decreasing (easiest first, "
            f"hardest last); got {list(scales)}."
        )


class CurriculumReward(RewardFormula):
    def __init__(
        self,
        scales: Sequence[float],
        *,
        threshold: float,
        rollouts_per_step: int,
        stage_file: Path | None = None,
        label: str | None = None,
    ):
        validate_curriculum_scales(scales)
        self.scales = list(scales)
        self.adapters = [scale_to_adapter_name(s) for s in self.scales]
        self.threshold = threshold
        # A behavior contributes one group per step it is scheduled into, so
        # this is group_size: the stage decision is that group's mean.
        self.rollouts_per_step = rollouts_per_step
        self.stage_file = stage_file
        self.label = label
        self.batch_scores: list[float] = []
        self.current_stage = self._load_stage()

    def _load_stage(self) -> int:
        if self.stage_file is None or not self.stage_file.exists():
            return 0
        stage = int(self.stage_file.read_text().strip())
        logger.warning(f"Resuming curriculum [{self.label}] at stage {stage}.")
        return min(stage, len(self.scales) - 1)

    def _save_stage(self) -> None:
        if self.stage_file is not None:
            self.stage_file.parent.mkdir(parents=True, exist_ok=True)
            self.stage_file.write_text(str(self.current_stage))

    def _record(self, reward: float) -> None:
        self.batch_scores.append(reward)
        # `>=`, not `==`: a rollout whose IO fails returns before recording,
        # and a retried rollout records twice; either would desync `==`.
        if len(self.batch_scores) < self.rollouts_per_step:
            return
        mean = sum(self.batch_scores) / len(self.batch_scores)
        self.batch_scores.clear()
        if mean >= self.threshold and self.current_stage < len(self.scales) - 1:
            self.current_stage += 1
            self._save_stage()
            logger.info(
                f"Curriculum [{self.label}] advanced (mean={mean:.1f}) to stage "
                f"{self.current_stage}, scale {self.scales[self.current_stage]}"
            )

    async def compute_reward(
        self,
        *,
        investigator_prompt: str,
        behavior: str,
        target: VLLMTarget,
        judge: Judge,
        n_target_samples: int,
    ) -> RewardResult:
        # Read the stage before any await: every rollout of a step starts
        # before the first one finishes, so the whole group sees one stage.
        stage = self.current_stage
        messages = [{"role": "user", "content": investigator_prompt}]
        responses = await target.generate(
            messages, n=n_target_samples, lora_name=self.adapters[stage]
        )
        verdicts = await judge_rollout(
            judge,
            behavior=behavior,
            investigator_prompt=investigator_prompt,
            responses=responses,
        )

        response_score = sum(verdicts.response_scores) / len(verdicts.response_scores)
        # Hard gate: a refused or invalid prompt earns nothing.
        gated = (
            is_investigator_refusal(investigator_prompt) or not verdicts.prompt_passed
        )
        reward = 0.0 if gated else float(response_score)
        self._record(reward)

        return RewardResult(
            reward=reward,
            responses=responses,
            prompt_criteria=verdicts.prompt_criteria,
            response_criteria=verdicts.response_criteria,
            prompt_reasoning=verdicts.prompt_reasoning,
            response_reasonings=verdicts.response_reasonings,
            metrics={
                **base_metrics(
                    reward=reward,
                    investigator_prompt=investigator_prompt,
                    verdicts=verdicts,
                    responses=responses,
                    response_score=float(response_score),
                ),
                "curriculum/scale": self.scales[stage],
                "curriculum/stage": float(stage),
                "curriculum/threshold_success_rate": float(reward >= self.threshold),
            },
            log_extra={"curriculum_scale": self.scales[stage]},
        )
