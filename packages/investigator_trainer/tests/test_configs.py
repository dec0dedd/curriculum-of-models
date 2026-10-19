"""The config files and the loader: the interface every paper run is launched through."""

import pytest

from investigator_trainer.config_file import ConfigFileError, available, load_config
from investigator_trainer.train import (
    CONFIGS_DIR,
    TrainConfig,
    kl_reference_path,
    parse_args,
    validate,
)

METHODS_DIR = CONFIGS_DIR / "methods"
TARGETS_DIR = CONFIGS_DIR / "targets"

METHODS = available(METHODS_DIR)
TARGETS = available(TARGETS_DIR)

# A method config must pin each of these. They differ per method, so a method
# that left one at its default would silently train with another's setting.
METHOD_FIELDS = (
    "reward_type",
    "kl_penalty_coef",
    "judge_skip_refusals",
    "zero_reward_on_truncation",
)
# A target config must pin each of these.
TARGET_FIELDS = ("target_model_id", "chat_template_kwargs", "target_max_tokens")

# Stands in for the investigator SFT warm start, which readers build themselves.
WARM_START = "tinker://run-id:train:0/weights/sft-final"


def args_for(method: str, target: str) -> list[str]:
    return [f"methods/{method}.yaml", f"targets/{target}.yaml"]


def test_the_papers_methods_and_targets_all_ship():
    assert METHODS == ["blackbox", "com", "prbo", "prbo_plain"]
    assert TARGETS == ["gemma", "gemma_reasoning", "gpt_oss"]


@pytest.mark.parametrize("method", METHODS)
def test_every_method_pins_the_values_that_must_not_drift(method):
    values = load_config(
        METHODS_DIR / f"{method}.yaml", set(TrainConfig.__chz_fields__)
    )
    for field in METHOD_FIELDS:
        assert field in values, f"{method} does not set {field}"


@pytest.mark.parametrize("target", TARGETS)
def test_every_target_pins_the_target(target):
    values = load_config(
        TARGETS_DIR / f"{target}.yaml", set(TrainConfig.__chz_fields__)
    )
    for field in TARGET_FIELDS:
        assert field in values, f"{target} does not set {field}"


@pytest.mark.parametrize("method", METHODS)
def test_a_method_never_sets_the_target(method):
    # The two kinds of config are independent, so neither may reach into the
    # other: a target value hidden in a method file would apply to every
    # target it is paired with.
    values = load_config(METHODS_DIR / f"{method}.yaml")
    assert not set(values) & set(TARGET_FIELDS)


@pytest.mark.parametrize("target", TARGETS)
def test_a_target_never_sets_the_method(target):
    values = load_config(TARGETS_DIR / f"{target}.yaml")
    assert not set(values) & set(METHOD_FIELDS)


def test_the_papers_method_values():
    cfg = parse_args(args_for("com", "gemma"))
    assert cfg.reward_type == "curriculum"
    assert cfg.kl_penalty_coef == 0.5
    assert cfg.curriculum_scales == [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    assert cfg.curriculum_threshold == 60.0
    # Black-box is the curriculum reward with only the base target.
    blackbox = parse_args(
        [*args_for("blackbox", "gemma"), f"load_checkpoint_path={WARM_START}"]
    )
    assert blackbox.curriculum_scales == [0.0]
    assert blackbox.kl_penalty_coef == 0.5
    prbo = parse_args(args_for("prbo", "gemma"))
    assert prbo.reward_type == "combined_estimator"
    assert prbo.kl_penalty_coef == 0.1
    assert prbo.prbo_judge_scale == 1.1
    # The paper put no threshold on the natural-response term, so the
    # config omits it and the default must stay None.
    assert prbo.prbo_natural_response_threshold is None
    # The judge's refusal shortcut was on for PRBO runs and off for CoM.
    assert prbo.judge_skip_refusals is True
    assert cfg.judge_skip_refusals is False


def test_the_papers_target_values():
    cfg = parse_args(args_for("com", "gpt_oss"))
    assert cfg.target_model_id == "openai/gpt-oss-120b"
    assert cfg.chat_template_kwargs == {}
    gemma = parse_args(args_for("com", "gemma"))
    assert gemma.target_model_id == "google/gemma-4-31B-it"
    assert gemma.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", TARGETS)
def test_every_pairing_builds(method, target):
    extra = [f"load_checkpoint_path={WARM_START}"] if method == "blackbox" else []
    cfg = parse_args([*args_for(method, target), *extra])
    validate(cfg)


def test_reasoning_targets_raise_the_token_budget():
    plain = parse_args(args_for("prbo", "gemma"))
    reasoning = parse_args(args_for("prbo", "gemma_reasoning"))
    assert reasoning.chat_template_kwargs == {"enable_thinking": True}
    assert reasoning.target_max_tokens > plain.target_max_tokens


def test_a_field_on_the_command_line_overrides_the_files():
    cfg = parse_args(
        [*args_for("prbo", "gpt_oss"), "kl_penalty_coef=0.3", "group_size=256"]
    )
    assert cfg.kl_penalty_coef == 0.3
    assert cfg.group_size == 256
    assert cfg.reward_type == "combined_estimator"


def test_the_last_config_wins():
    # Both set reward_type, so a reader can tell which method a command runs
    # from the last method file on it.
    com, prbo = "methods/com.yaml", "methods/prbo.yaml"
    assert parse_args([com, prbo]).reward_type == "combined_estimator"
    assert parse_args([prbo, com]).reward_type == "curriculum"


def test_a_later_config_replaces_a_table_rather_than_merging_it():
    # gemma sets enable_thinking, gpt_oss sets an empty table; the empty one
    # must win outright, not leave the earlier key behind.
    cfg = parse_args(["targets/gemma.yaml", "targets/gpt_oss.yaml"])
    assert cfg.chat_template_kwargs == {}


def test_black_box_demands_a_warm_start():
    # Its whole point is starting from the SFT checkpoint; without one it is
    # just CoM with a one-stage ladder.
    with pytest.raises(ValueError, match="load_checkpoint_path"):
        validate(parse_args(args_for("blackbox", "gemma")))


def test_the_kl_reference_is_the_runs_own_starting_point():
    # tinker only accepts a sampler_weights path for the reference, and a
    # warm-started run must measure drift from its warm start, not the base.
    cfg = parse_args(
        [*args_for("blackbox", "gemma"), f"load_checkpoint_path={WARM_START}"]
    )
    assert kl_reference_path(cfg) == WARM_START.replace(
        "/weights/", "/sampler_weights/"
    )
    assert kl_reference_path(parse_args(args_for("com", "gemma"))) is None


def test_an_unknown_field_names_the_file_that_sets_it(tmp_path):
    bad = tmp_path / "typo.yaml"
    bad.write_text("reward_typo: curriculum\n")
    with pytest.raises(ConfigFileError, match="reward_typo"):
        load_config(bad, set(TrainConfig.__chz_fields__))
    # And the trainer surfaces it rather than starting a run.
    with pytest.raises(SystemExit, match="reward_typo"):
        parse_args([str(bad)])


def test_a_missing_config_is_rejected_by_path(tmp_path):
    with pytest.raises(SystemExit, match="cannot read config file"):
        parse_args([str(tmp_path / "absent.yaml")])


def test_malformed_yaml_is_rejected(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("reward_type: [curriculum\n")
    with pytest.raises(SystemExit, match="not valid YAML"):
        parse_args([str(bad)])


def test_a_config_can_reset_a_field_to_null(tmp_path):
    # Unlike TOML, YAML has a null, so a later config can reset a field an
    # earlier one set, not just leave it unmentioned.
    sets = tmp_path / "sets.yaml"
    sets.write_text("kl_reference_checkpoint_path: tinker://some-run/sampler_weights\n")
    resets = tmp_path / "resets.yaml"
    resets.write_text("kl_reference_checkpoint_path: null\n")

    cfg = parse_args([str(sets)])
    assert cfg.kl_reference_checkpoint_path == "tinker://some-run/sampler_weights"

    cfg = parse_args([str(sets), str(resets)])
    assert cfg.kl_reference_checkpoint_path is None


def test_a_config_that_is_not_a_mapping_is_rejected(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- curriculum\n- prbo\n")
    with pytest.raises(SystemExit, match="must be a mapping"):
        parse_args([str(bad)])


def test_a_bare_name_resolves_against_the_configs_directory(monkeypatch, tmp_path):
    # The commands in the README are run from inside a package directory, so a
    # bare name must not depend on the working directory.
    monkeypatch.chdir(tmp_path)
    assert parse_args(["methods/com.yaml"]).reward_type == "curriculum"


def test_an_explicit_path_wins_over_the_configs_directory(tmp_path):
    local = tmp_path / "methods"
    local.mkdir()
    (local / "com.yaml").write_text("reward_type: prbo\n")
    assert parse_args([str(local / "com.yaml")]).reward_type == "prbo"


def test_parse_args_collects_the_resolved_config_paths():
    # run_metadata.json needs these to trace a run back to the files that
    # produced it, separately from the resolved values it already records.
    seen: list = []
    parse_args(args_for("com", "gemma"), seen)
    assert [p.name for p in seen] == ["com.yaml", "gemma.yaml"]
    assert all(p.is_absolute() for p in seen)


def test_an_override_on_the_command_line_is_not_collected_as_a_config():
    seen: list = []
    parse_args([*args_for("com", "gemma"), "group_size=256"], seen)
    assert len(seen) == 2


def test_help_exits_cleanly_and_names_the_configs(capsys, monkeypatch):
    # chz signals help by raising, so --help used to print its field table and
    # then die with a traceback. It is the first command a reader runs.
    from investigator_trainer.train import main

    monkeypatch.setattr("sys.argv", ["com-train", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    for name in (*METHODS, *TARGETS):
        assert name in printed
