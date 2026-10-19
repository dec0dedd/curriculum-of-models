import json

import pytest

from cbrn_eval import output, runner
from cbrn_eval.config import EvalConfig
from cbrn_eval.runner import run_sweep
from tests.fakes import FakeFailingTarget, FakeJudge, FakeTarget


@pytest.fixture
def behaviors_file(tmp_path):
    path = tmp_path / "behaviors.jsonl"
    rows = [
        {"behavior_id": "a", "category": "cat1", "behavior": "Do A."},
        {"behavior_id": "b", "category": "cat2", "behavior": "Do B."},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _config(tmp_path, behaviors_file, **overrides) -> EvalConfig:
    return EvalConfig(
        prompt_source="behaviors",
        behaviors_file=behaviors_file,
        n_target_generations=2,
        output_dir=str(tmp_path / "out"),
        **overrides,
    )


def _patch(monkeypatch, target, judge):
    monkeypatch.setattr(runner, "VLLMTarget", lambda **kwargs: target)
    monkeypatch.setattr(runner, "Judge", lambda **kwargs: judge)


async def test_sweep_scores_every_behavior_and_generation(
    monkeypatch, tmp_path, behaviors_file
):
    target = FakeTarget()
    # 100 and 50 are both exact round-trips of response_score's 1/6ths.
    judge = FakeJudge({"response0": 100, "response1": 50})
    _patch(monkeypatch, target, judge)

    config = _config(tmp_path, behaviors_file)
    summary = await run_sweep(config, "run1")

    rows = output.load_rows(config.output_dir, "run1")
    assert len(rows) == 4  # 2 behaviors x 2 generations
    scores = sorted(r["judge_score"] for r in rows)
    assert scores == [50, 50, 100, 100]
    # One judge call per item (both generations together), one target call too.
    assert len(judge.response_calls) == 2
    assert len(target.generate_calls) == 2
    assert summary["totals"]["n_rows"] == 4
    assert summary["totals"]["n_judged"] == 4


async def test_prompt_judge_gates_the_final_score(
    monkeypatch, tmp_path, behaviors_file
):
    target = FakeTarget()
    judge = FakeJudge({"response0": 100, "response1": 100}, prompt_passes=False)
    _patch(monkeypatch, target, judge)

    config = _config(tmp_path, behaviors_file, use_prompt_judge=True)
    await run_sweep(config, "run1")

    rows = output.load_rows(config.output_dir, "run1")
    assert all(r["judge_score"] == 100 for r in rows)
    assert all(r["final_score"] == 0 for r in rows)
    assert all(r["prompt_passed"] is False for r in rows)


async def test_without_the_prompt_judge_final_score_is_the_judge_score(
    monkeypatch, tmp_path, behaviors_file
):
    target = FakeTarget()
    judge = FakeJudge({"response0": 100, "response1": 100})
    _patch(monkeypatch, target, judge)

    config = _config(tmp_path, behaviors_file, use_prompt_judge=False)
    await run_sweep(config, "run1")

    rows = output.load_rows(config.output_dir, "run1")
    # Nothing gates the score, so it passes through unchanged.
    assert all(r["final_score"] == 100 for r in rows)
    assert all(r["prompt_passed"] is None for r in rows)
    assert judge.prompt_calls == []  # never called


async def test_a_completed_item_is_not_reprocessed_on_resume(
    monkeypatch, tmp_path, behaviors_file
):
    target = FakeTarget()
    judge = FakeJudge(default_score=50)
    _patch(monkeypatch, target, judge)

    config = _config(tmp_path, behaviors_file)
    # Pre-populate both generations of behavior "a" only.
    for g in range(2):
        output.append_row(
            config.output_dir,
            "run1",
            output.TrialRow(
                run_id="run1",
                behavior_id="a",
                category="cat1",
                prompt_index=0,
                generation_index=g,
                prompt="Do A.",
                target_completion="old",
                judge_score=99,
                judge_criteria=None,
                judge_reasoning=None,
                error=None,
            ),
        )

    await run_sweep(config, "run1")

    # Only behavior "b" went through the target/judge again.
    assert len(target.generate_calls) == 1
    rows = output.load_rows(config.output_dir, "run1")
    a_rows = [r for r in rows if r["behavior_id"] == "a"]
    assert all(r["judge_score"] == 99 for r in a_rows)  # untouched


async def test_a_target_failure_is_recorded_per_generation(
    monkeypatch, tmp_path, behaviors_file
):
    target = FakeFailingTarget(RuntimeError("boom"))
    judge = FakeJudge(default_score=50)
    _patch(monkeypatch, target, judge)

    config = _config(tmp_path, behaviors_file)
    await run_sweep(config, "run1")

    rows = output.load_rows(config.output_dir, "run1")
    assert len(rows) == 4
    assert all(r["target_completion"] is None for r in rows)
    assert all("target_error" in r["error"] for r in rows)
    assert judge.response_calls == []  # never reached
