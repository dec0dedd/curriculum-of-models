"""What every reward formula shares: the result type, judging, and base metrics.

Metric namespaces (each key is a per-rollout value; tinker averages them over
the step and logs them as ``env/all/<key>``):

  reward/*        every formula
  judge/*         every formula: per-criterion pass rates
  target/*        every formula
  investigator/*  every formula
  curriculum/*    CurriculumReward only
  prbo/*          the PRBO formulas only
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from common.judge import Judge
from common.refusal import is_investigator_refusal, is_refusal
from common.scoring import (
    PROMPT_CRITERIA,
    RESPONSE_CRITERIA,
    prompt_passes,
    response_score,
)
from common.target import VLLMTarget


@dataclass
class RewardResult:
    reward: float
    # Every target response the reward was computed from, in judge order.
    responses: list[str]
    prompt_criteria: dict[str, bool]
    response_criteria: list[dict[str, bool]]
    prompt_reasoning: str
    response_reasonings: list[str]
    # Numeric per-rollout metrics, see the namespaces above.
    metrics: dict[str, float] = field(default_factory=dict)
    # Non-numeric data for rollouts.jsonl only (e.g. PRBO's natural reasoning).
    log_extra: dict = field(default_factory=dict)


@dataclass
class Verdicts:
    prompt_criteria: dict[str, bool]
    prompt_reasoning: str
    response_criteria: list[dict[str, bool]]
    response_reasonings: list[str]

    @property
    def prompt_passed(self) -> bool:
        return prompt_passes(self.prompt_criteria)

    @property
    def response_scores(self) -> list[int]:
        """Rubric scores on [0, 100], one per response."""
        return [response_score(c) for c in self.response_criteria]


async def judge_rollout(
    judge: Judge, *, behavior: str, investigator_prompt: str, responses: list[str]
) -> Verdicts:
    """One prompt-judge call and one response-judge call per response, concurrently."""
    (prompt_criteria, prompt_reasoning), response_verdicts = await asyncio.gather(
        judge.score_prompt(behavior=behavior, investigator_prompt=investigator_prompt),
        judge.score_responses(behavior=behavior, responses=responses),
    )
    return Verdicts(
        prompt_criteria=prompt_criteria,
        prompt_reasoning=prompt_reasoning,
        response_criteria=[c for c, _ in response_verdicts],
        response_reasonings=[r for _, r in response_verdicts],
    )


def base_metrics(
    *,
    reward: float,
    investigator_prompt: str,
    verdicts: Verdicts,
    responses: list[str],
    response_score: float,
) -> dict[str, float]:
    """Metrics every formula reports. ``response_score`` is on the 0-100 rubric scale.

    The judge/response_* and target/* keys describe the first response, the
    one written to rollouts.jsonl.
    """
    first_criteria = verdicts.response_criteria[0] if verdicts.response_criteria else {}
    first_response = responses[0] if responses else ""
    return {
        "reward/final": reward,
        "reward/prompt_passed": float(verdicts.prompt_passed),
        "reward/response_score": response_score,
        "target/is_empty": float(len(first_response.split()) < 10),
        "target/response_is_refusal": float(is_refusal(first_response)),
        "investigator/is_refusal": float(is_investigator_refusal(investigator_prompt)),
        **{
            f"judge/prompt_{c}": float(bool(verdicts.prompt_criteria.get(c, False)))
            for c in PROMPT_CRITERIA
        },
        **{
            f"judge/response_{c}": float(bool(first_criteria.get(c, False)))
            for c in RESPONSE_CRITERIA
        },
    }


class RewardFormula(ABC):
    @abstractmethod
    async def compute_reward(
        self,
        *,
        investigator_prompt: str,
        behavior: str,
        target: VLLMTarget,
        judge: Judge,
        n_target_samples: int,
    ) -> RewardResult:
        """Score one investigator prompt against one behavior.

        The investigator prompt is sent to the target as the only user turn.
        """
