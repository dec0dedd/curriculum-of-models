from cbrn_eval import output
from cbrn_eval.export import (
    INDEX_COLUMNS,
    RESULT_COLUMNS,
    export_all,
    export_results,
    export_run,
    local_run_ids,
)
from cbrn_eval.output import TrialRow


def _row(**overrides) -> TrialRow:
    base = dict(
        run_id="r",
        behavior_id="b",
        category=None,
        prompt_index=0,
        generation_index=0,
        prompt="p",
        investigator_refusal=False,
        target_completion="c",
        target_refusal=False,
        judge_score=50,
        judge_criteria=None,
        judge_reasoning=None,
        error=None,
        prompt_passed=True,
        final_score=50,
    )
    base.update(overrides)
    return TrialRow(**base)


def test_export_results_schema_and_values():
    from dataclasses import asdict

    rows = [asdict(_row(judge_score=83, prompt_passed=True, final_score=83))]
    out = export_results("run1", rows)
    assert out.columns == RESULT_COLUMNS
    assert out["run_id"].to_list() == ["run1"]
    assert out["judge_score"].to_list() == [83]
    assert out["final_score"].to_list() == [83]


def test_export_results_with_no_gate_in_effect_passes_the_judge_score_through():
    from dataclasses import asdict

    # The runner always writes final_score via output.gated_score; with no
    # prompt judge prompt_passed is None, which is not a gate.
    row = asdict(_row(judge_score=40, prompt_passed=None, final_score=40))
    out = export_results("run1", [row])
    assert out["final_score"].to_list() == [40]


def test_export_run_writes_a_csv(tmp_path):
    output_dir = tmp_path / "out"
    output.append_row(str(output_dir), "run1", _row())
    export_dir = tmp_path / "exports"

    df = export_run(output_dir, "run1", export_dir)
    csv_path = export_dir / "run1.csv"
    assert csv_path.exists()
    assert df.height == 1
    assert "run1" in csv_path.read_text()


def test_local_run_ids_lists_completed_runs_only(tmp_path):
    output_dir = tmp_path
    output.append_row(str(output_dir), "incomplete", _row())  # no summary written
    output.append_row(str(output_dir), "complete", _row())
    output.save_summary(str(output_dir), "complete", {"run_id": "complete"})

    assert local_run_ids(output_dir) == ["complete"]


def test_export_all_writes_one_csv_per_run_plus_an_index(tmp_path):
    output_dir = tmp_path / "out"
    export_dir = tmp_path / "exports"

    for run_id, score in (("run_a", 20), ("run_b", 80)):
        output.append_row(
            str(output_dir), run_id, _row(judge_score=score, final_score=score)
        )
        output.save_summary(
            str(output_dir),
            run_id,
            {
                "run_id": run_id,
                "started_at": "t0",
                "finished_at": "t1",
                "config": {"target_model": "m", "prompt_source": "investigator"},
            },
        )

    n = export_all(output_dir, export_dir)
    assert n == 2
    assert (export_dir / "run_a.csv").exists()
    assert (export_dir / "run_b.csv").exists()

    import csv

    with (export_dir / "index.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["run_id"] for r in rows} == {"run_a", "run_b"}
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["run_a"]["mean_judge_score"] == "20.0"
    assert by_id["run_b"]["mean_judge_score"] == "80.0"
    assert list(rows[0]) == INDEX_COLUMNS


def test_export_all_on_an_empty_dir_writes_nothing(tmp_path):
    assert export_all(tmp_path, tmp_path / "exports") == 0
    assert not (tmp_path / "exports").exists()
