import json

from cbrn_eval.config import EvalConfig


def _behaviors_file(tmp_path, text="a", name="behaviors.jsonl") -> str:
    path = tmp_path / name
    path.write_text(
        json.dumps({"behavior_id": "x", "category": None, "behavior": text}) + "\n"
    )
    return str(path)


def test_run_id_is_stable_for_the_same_config(tmp_path):
    bf = _behaviors_file(tmp_path)
    a = EvalConfig(behaviors_file=bf, target_model="m")
    b = EvalConfig(behaviors_file=bf, target_model="m")
    assert a.run_id() == b.run_id()


def test_run_id_changes_with_a_result_affecting_field(tmp_path):
    bf = _behaviors_file(tmp_path)
    a = EvalConfig(behaviors_file=bf, target_model="m", judge_model="x")
    b = EvalConfig(behaviors_file=bf, target_model="m", judge_model="y")
    assert a.run_id() != b.run_id()


def test_run_id_changes_with_behaviors_file_contents(tmp_path):
    a_file = _behaviors_file(tmp_path, "do a", name="a.jsonl")
    b_file = _behaviors_file(tmp_path, "do b", name="b.jsonl")
    a = EvalConfig(behaviors_file=a_file, target_model="m")
    b = EvalConfig(behaviors_file=b_file, target_model="m")
    assert a.run_id() != b.run_id()


def test_run_id_ignores_execution_knobs(tmp_path):
    bf = _behaviors_file(tmp_path)
    a = EvalConfig(behaviors_file=bf, target_model="m", target_concurrency=1)
    b = EvalConfig(behaviors_file=bf, target_model="m", target_concurrency=99)
    assert a.run_id() == b.run_id()


def test_run_id_is_slugified_from_the_target_model(tmp_path):
    bf = _behaviors_file(tmp_path)
    config = EvalConfig(behaviors_file=bf, target_model="google/gemma-4-31B-it")
    assert config.run_id().startswith("google-gemma-4-31B-it_")


def test_to_dict_round_trips_every_field(tmp_path):
    bf = _behaviors_file(tmp_path)
    config = EvalConfig(behaviors_file=bf, target_model="m", behavior_ids=("x",))
    as_dict = config.to_dict()
    assert as_dict["target_model"] == "m"
    assert as_dict["behavior_ids"] == ("x",)
