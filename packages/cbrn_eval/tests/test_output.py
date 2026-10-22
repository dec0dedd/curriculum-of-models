import json

from cbrn_eval import output
from cbrn_eval.output import TrialRow, gated_score


def _row(**overrides) -> TrialRow:
    base = dict(
        run_id="r",
        behavior_id="b",
        category=None,
        prompt_index=0,
        generation_index=0,
        prompt="p",
        target_completion="c",
        judge_score=50,
        judge_criteria={"addresses_exact_target": True},
        judge_reasoning="reasoning",
        error=None,
    )
    base.update(overrides)
    return TrialRow(**base)


def test_gated_score():
    assert gated_score(judge_score=None, prompt_passed=True) is None
    assert gated_score(judge_score=50, prompt_passed=False) == 0
    assert gated_score(judge_score=50, prompt_passed=True) == 50
    assert gated_score(judge_score=50, prompt_passed=None) == 50  # no gate in effect


def test_key_is_behavior_prompt_and_generation():
    assert _row(generation_index=0).key() == "b:0:0"
    assert _row(generation_index=3).key() == "b:0:3"


def test_append_and_load_round_trips(tmp_path):
    output_dir = str(tmp_path)
    output.append_row(output_dir, "run1", _row(prompt_index=0))
    output.append_row(output_dir, "run1", _row(prompt_index=1, generation_index=1))

    rows = output.load_rows(output_dir, "run1")
    assert {r["prompt_index"] for r in rows} == {0, 1}
    assert output.load_completed_keys(output_dir, "run1") == {"b:0:0", "b:1:1"}


def test_resumed_rows_dedupe_keeping_the_last_write(tmp_path):
    output_dir = str(tmp_path)
    output.append_row(output_dir, "run1", _row(judge_score=10))
    output.append_row(
        output_dir, "run1", _row(judge_score=90)
    )  # a later, fresher write

    rows = output.load_rows(output_dir, "run1")
    assert len(rows) == 1
    assert rows[0]["judge_score"] == 90

    # The undeduped view (used for resume-skip bookkeeping) still sees both.
    assert len(output.load_rows(output_dir, "run1", deduped=False)) == 2


def test_compact_moves_jsonl_to_parquet(tmp_path):
    output_dir = str(tmp_path)
    output.append_row(output_dir, "run1", _row())

    assert output.full_path(output_dir, "run1").exists()
    assert output.compact_full(output_dir, "run1") == output.full_parquet_path(
        output_dir, "run1"
    )
    assert not output.full_path(output_dir, "run1").exists()
    assert output.full_parquet_path(output_dir, "run1").exists()
    assert output.load_rows(output_dir, "run1")[0]["judge_score"] == 50

    # Compacting an already-compacted run is a no-op.
    assert output.compact_full(output_dir, "run1") is None


def test_summarize_aggregates_per_behavior_and_totals():
    rows = [
        {
            "behavior_id": "a",
            "category": "cat",
            "error": None,
            "judge_score": 80,
            "final_score": 80,
            "prompt_passed": True,
        },
        {
            "behavior_id": "a",
            "category": "cat",
            "error": None,
            "judge_score": None,
            "final_score": None,
            "prompt_passed": None,
        },
        {
            "behavior_id": "b",
            "category": None,
            "error": "target_error: boom",
            "judge_score": None,
            "final_score": None,
            "prompt_passed": False,
        },
    ]
    summary = output.summarize(rows)
    assert summary["behaviors"]["a"]["n_rows"] == 2
    assert summary["behaviors"]["a"]["mean_judge_score"] == 80
    assert summary["behaviors"]["a"]["mean_final_score"] == 80
    assert summary["behaviors"]["b"]["n_errors"] == 1
    assert summary["behaviors"]["b"]["n_prompt_judged"] == 1
    assert summary["behaviors"]["b"]["n_prompt_failed"] == 1
    assert summary["behaviors"]["b"]["mean_judge_score"] is None
    assert summary["totals"]["n_rows"] == 3
    assert summary["totals"]["n_errors"] == 1


def test_write_summary_writes_to_disk(tmp_path):
    output_dir = str(tmp_path)
    output.append_row(output_dir, "run1", _row())
    rows = output.load_rows(output_dir, "run1")
    summary = output.write_summary(
        output_dir, "run1", {"target_model": "x"}, "t0", "t1", rows
    )
    assert summary["run_id"] == "run1"
    written = json.loads(output.summary_path(output_dir, "run1").read_text())
    assert written == summary


def test_remove_run_outputs_deletes_everything(tmp_path):
    output_dir = str(tmp_path)
    output.append_row(output_dir, "run1", _row())
    output.compact_full(output_dir, "run1")
    output.save_summary(output_dir, "run1", {"run_id": "run1"})

    output.remove_run_outputs(output_dir, "run1")
    assert not output.full_parquet_path(output_dir, "run1").exists()
    assert not output.summary_path(output_dir, "run1").exists()


def test_compact_survives_more_than_100_leading_null_rows(tmp_path):
    # polars infers a schema from the first 100 rows by default; a sweep whose
    # first rows are all target errors would then fail on the first score.
    output_dir = str(tmp_path)
    for i in range(150):
        output.append_row(
            output_dir,
            "run1",
            _row(
                prompt_index=i,
                target_completion=None,
                judge_score=None,
                judge_criteria=None,
                error="target_error: boom",
            ),
        )
    output.append_row(output_dir, "run1", _row(prompt_index=150))

    output.compact_full(output_dir, "run1")

    rows = output.load_rows(output_dir, "run1")
    assert len(rows) == 151
    scored = [r for r in rows if r["judge_score"] is not None]
    assert scored[0]["judge_criteria"] == {"addresses_exact_target": True}
