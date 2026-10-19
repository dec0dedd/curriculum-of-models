import math

import pytest

from investigator_trainer.rewards import (
    CombinedEstimatorReward,
    CurriculumReward,
    PRBOReward,
)
from investigator_trainer.rewards.prbo import to_prbo_range
from tests.fakes import FakeJudge, FakeTarget

PROMPT = "Explain the procedure step by step."
KL = -3.0 - -1.0  # FakeTarget's log p - log q


async def run(formula, target, judge, n=1, prompt=PROMPT):
    return await formula.compute_reward(
        investigator_prompt=prompt,
        behavior="behavior",
        target=target,
        judge=judge,
        n_target_samples=n,
    )


# ── CoM ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scales", [[], [0.5, 0.5], [0.0, 1.0], [1.2, 0.0]])
def test_the_reward_rejects_an_invalid_ladder(scales):
    # The contract itself is tested in common; this is the reward refusing to
    # be built on one.
    with pytest.raises(ValueError):
        CurriculumReward(scales, threshold=60, rollouts_per_step=1)


async def test_com_reward_is_the_judge_score():
    com = CurriculumReward([1.0, 0.0], threshold=60, rollouts_per_step=4)
    target = FakeTarget()
    result = await run(com, target, FakeJudge({"alpha_1_00": 50}))
    assert result.reward == 50
    assert target.generate_calls == ["alpha_1_0"]
    assert result.metrics["curriculum/scale"] == 1.0


async def test_com_gates_invalid_and_refused_prompts():
    com = CurriculumReward([1.0, 0.0], threshold=60, rollouts_per_step=4)
    judge = FakeJudge({"alpha_1_00": 100}, prompt_passes=False)
    assert (await run(com, FakeTarget(), judge)).reward == 0.0

    judge = FakeJudge({"alpha_1_00": 100})
    assert (
        await run(com, FakeTarget(), judge, prompt="I can't help with that.")
    ).reward == 0.0


async def test_com_advances_on_the_step_mean_only(tmp_path):
    stage_file = tmp_path / "stage.txt"
    com = CurriculumReward(
        [1.0, 0.5, 0.0], threshold=60, rollouts_per_step=2, stage_file=stage_file
    )
    judge = FakeJudge({"alpha_1_00": 100, "alpha_0_50": 100, "natural0": 100})
    target = FakeTarget()

    await run(com, target, judge)
    assert com.current_stage == 0  # one of two rollouts: no decision yet
    await run(com, target, judge)
    assert com.current_stage == 1
    assert stage_file.read_text() == "1"

    # A run that resumes picks the stage up from its file.
    resumed = CurriculumReward(
        [1.0, 0.5, 0.0], threshold=60, rollouts_per_step=2, stage_file=stage_file
    )
    assert resumed.current_stage == 1


async def test_com_stays_below_threshold():
    com = CurriculumReward([1.0, 0.0], threshold=60, rollouts_per_step=2)
    judge = FakeJudge({"alpha_1_00": 50})
    for _ in range(4):
        await run(com, FakeTarget(), judge)
    assert com.current_stage == 0


async def test_black_box_samples_the_base_model():
    com = CurriculumReward([0.0], threshold=60, rollouts_per_step=1)
    target = FakeTarget()
    await run(com, target, FakeJudge({"natural0": 100}))
    await run(com, target, FakeJudge({"natural0": 100}))
    assert target.generate_calls == [None, None]
    assert com.current_stage == 0


# ── PRBO ─────────────────────────────────────────────────────────────────────


def test_prbo_range():
    assert to_prbo_range(0) == -100
    assert to_prbo_range(50) == -50
    assert to_prbo_range(100) == 0


async def test_plain_prbo():
    result = await run(PRBOReward(judge_scale=1.1), FakeTarget(), FakeJudge({"q0": 50}))
    assert result.reward == pytest.approx(KL + 1.1 * -50)
    assert result.metrics["reward/response_score"] == pytest.approx(50)


async def test_plain_prbo_invalid_prompt_takes_the_floor():
    judge = FakeJudge({"q0": 100}, prompt_passes=False)
    result = await run(PRBOReward(judge_scale=1.1), FakeTarget(), judge)
    assert result.reward == pytest.approx(KL + 1.1 * -100)


async def test_combined_estimator_takes_the_better_branch():
    judge = FakeJudge({"q0": 50, "natural0": 100})
    result = await run(CombinedEstimatorReward(judge_scale=1.1), FakeTarget(), judge)
    # max(KL + 1.1 * -50, 1.1 * 0) - log 2
    assert result.reward == pytest.approx(0.0 - math.log(2))
    # The prompt is judged once, for the q responses.
    assert judge.prompt_calls == 1


async def test_combined_estimator_invalid_prompt_gates_both_branches():
    judge = FakeJudge({"q0": 100, "natural0": 100}, prompt_passes=False)
    result = await run(CombinedEstimatorReward(judge_scale=1.1), FakeTarget(), judge)
    assert result.reward == pytest.approx(max(KL - 110, -110) - math.log(2))


async def test_combined_estimator_threshold_keeps_plain_prbo_below_it():
    judge = FakeJudge({"q0": 50, "q1": 50, "natural0": 100, "natural1": 0})
    formula = CombinedEstimatorReward(judge_scale=1.0, natural_response_threshold=-50)
    result = await run(formula, FakeTarget(), judge, n=2)
    combined = max(KL - 50, 0.0) - math.log(2)
    plain = KL - 50
    assert result.reward == pytest.approx((combined + plain) / 2)
    assert result.metrics["prbo/fraction_combined"] == 0.5


async def test_reasoning_target_scores_q_under_natural_reasoning():
    target = FakeTarget(splits_reasoning=True)
    judge = FakeJudge({"q0": 50, "natural0": 0})
    result = await run(CombinedEstimatorReward(judge_scale=1.0), target, judge)
    assert target.scored_conditioning == ((1, 2, 99),)
    assert result.log_extra["natural_reasoning_prefixes"] == ["thought0"]
