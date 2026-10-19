import json

from click.testing import CliRunner

from cbrn_eval import cli, output
from cbrn_eval.config import EvalConfig


def _behaviors_file(tmp_path) -> str:
    path = tmp_path / "behaviors.jsonl"
    path.write_text(
        json.dumps({"behavior_id": "x", "category": None, "behavior": "do x"}) + "\n"
    )
    return str(path)


def _fake_summary(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "behaviors": {
            "x": {
                "n_rows": 1,
                "n_errors": 0,
                "n_judged": 1,
                "mean_judge_score": 50.0,
                "max_judge_score": 50,
                "n_prompt_judged": 0,
                "n_prompt_failed": 0,
            }
        },
        "totals": {"n_rows": 1},
    }


async def _fake_run_sweep(config, run_id):
    return _fake_summary(run_id)


def test_sweep_prints_the_run_id_and_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "run_sweep", _fake_run_sweep)
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "sweep",
            "--prompt-source",
            "behaviors",
            "--behaviors-file",
            _behaviors_file(tmp_path),
            "--target-model",
            "m",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Run ID" in result.output
    assert "Done." in result.output


def test_sweep_skips_an_already_complete_run(monkeypatch, tmp_path):
    called = False

    async def _should_not_run(config, run_id):
        nonlocal called
        called = True
        return _fake_summary(run_id)

    monkeypatch.setattr(cli, "run_sweep", _should_not_run)
    runner = CliRunner()
    out_dir = tmp_path / "out"
    behaviors_file = _behaviors_file(tmp_path)

    config = EvalConfig(
        prompt_source="behaviors",
        behaviors_file=behaviors_file,
        target_model="m",
        output_dir=str(out_dir),
    )
    output.save_summary(str(out_dir), config.run_id(), {"run_id": config.run_id()})

    result = runner.invoke(
        cli.main,
        [
            "sweep",
            "--prompt-source",
            "behaviors",
            "--behaviors-file",
            behaviors_file,
            "--target-model",
            "m",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Already complete" in result.output
    assert not called


def test_force_rerun_discards_and_reruns(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "run_sweep", _fake_run_sweep)
    runner = CliRunner()
    out_dir = tmp_path / "out"
    behaviors_file = _behaviors_file(tmp_path)

    config = EvalConfig(
        prompt_source="behaviors",
        behaviors_file=behaviors_file,
        target_model="m",
        output_dir=str(out_dir),
    )
    output.save_summary(str(out_dir), config.run_id(), {"run_id": config.run_id()})

    result = runner.invoke(
        cli.main,
        [
            "sweep",
            "--prompt-source",
            "behaviors",
            "--behaviors-file",
            behaviors_file,
            "--target-model",
            "m",
            "--output-dir",
            str(out_dir),
            "--force-rerun",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "discarding previous results" in result.output
    assert "Done." in result.output


def test_export_with_no_runs_says_so(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli.main, ["export", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "No completed runs found" in result.output


def test_export_writes_csvs(tmp_path):
    from cbrn_eval.output import TrialRow

    out_dir = tmp_path / "out"
    output.append_row(
        str(out_dir),
        "run1",
        TrialRow(
            run_id="run1",
            behavior_id="x",
            category=None,
            prompt_index=0,
            generation_index=0,
            prompt="p",
            target_completion="c",
            judge_score=50,
            judge_criteria=None,
            judge_reasoning=None,
            error=None,
        ),
    )
    output.save_summary(
        str(out_dir),
        "run1",
        {
            "run_id": "run1",
            "config": {"target_model": "m", "prompt_source": "behaviors"},
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli.main, ["export", "--output-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert "Exported 1 run" in result.output
    assert (out_dir / "exports" / "run1.csv").exists()
    assert (out_dir / "exports" / "index.csv").exists()
