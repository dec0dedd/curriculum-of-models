"""run_metadata.json: what a run writes down about how it was launched."""

import json

import chz

from investigator_trainer.rollout_log import save_run_metadata


@chz.chz
class _FakeConfig:
    reward_type: str = "curriculum"
    kl_penalty_coef: float = 0.5


def test_records_config_files_as_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "com.yaml").write_text("reward_type: curriculum\n")

    save_run_metadata(
        tmp_path / "runs" / "r1", _FakeConfig(), config_files=["configs/com.yaml"]
    )

    metadata = json.loads((tmp_path / "runs" / "r1" / "run_metadata.json").read_text())
    assert metadata["config_files"] == [str(tmp_path / "configs" / "com.yaml")]
    # The resolved config is recorded too, so the files and the values they
    # produced can be checked against each other.
    assert metadata["config"]["reward_type"] == "curriculum"


def test_no_config_files_is_the_default():
    # Older call sites, or a run launched without any file, still produce
    # valid metadata.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        save_run_metadata(Path(tmp) / "r2", _FakeConfig())
        metadata = json.loads((Path(tmp) / "r2" / "run_metadata.json").read_text())
    assert metadata["config_files"] == []
