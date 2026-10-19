"""Preset composition: the interface every paper run is launched through."""

import pytest

from investigator_trainer.presets import METHODS, PAPER_WARM_START, PRESETS, TARGETS
from investigator_trainer.train import TrainConfig, parse_args, validate


def test_methods_and_targets_do_not_collide():
    # They share one namespace on the command line.
    assert not set(METHODS) & set(TARGETS)
    assert set(PRESETS) == set(METHODS) | set(TARGETS)


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_sets_only_real_fields(name):
    fields = set(TrainConfig.__chz_fields__)
    assert set(PRESETS[name]) <= fields, f"{name} sets unknown field(s)"


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_pins_the_values_that_must_not_drift(method):
    # These differ per method, so a method that left one at its default would
    # silently train with another method's setting.
    for key in (
        "reward_type",
        "kl_penalty_coef",
        "judge_skip_refusals",
        "zero_reward_on_truncation",
    ):
        assert key in METHODS[method], f"{method} does not set {key}"


def test_the_papers_method_values():
    assert METHODS["com"]["kl_penalty_coef"] == 0.5
    assert METHODS["blackbox"]["kl_penalty_coef"] == 0.5
    assert METHODS["prbo"]["kl_penalty_coef"] == 0.1
    # The judge's refusal shortcut was on for PRBO runs and off for CoM.
    assert METHODS["prbo"]["judge_skip_refusals"] is True
    assert METHODS["com"]["judge_skip_refusals"] is False
    # Black-box is the curriculum reward with only the base target.
    assert METHODS["blackbox"]["curriculum_scales"] == [0.0]


def test_a_method_and_a_target_compose():
    cfg = parse_args(["preset=com,gemma"])
    assert cfg.reward_type == "curriculum"
    assert cfg.kl_penalty_coef == 0.5
    assert cfg.target_model_id == "google/gemma-4-31B-it"
    assert cfg.chat_template_kwargs == {"enable_thinking": False}


def test_a_later_field_overrides_the_presets():
    cfg = parse_args(["preset=prbo,gpt_oss", "kl_penalty_coef=0.3", "group_size=256"])
    assert cfg.kl_penalty_coef == 0.3
    assert cfg.group_size == 256
    assert cfg.reward_type == "combined_estimator"


def test_preset_order_decides_the_winner():
    # Both set reward_type; the one applied last wins.
    assert parse_args(["preset=com,prbo"]).reward_type == "combined_estimator"
    assert parse_args(["preset=prbo,com"]).reward_type == "curriculum"


def test_an_unknown_preset_is_rejected_by_name():
    with pytest.raises(SystemExit, match="nope"):
        parse_args(["preset=nope"])


def test_reasoning_targets_raise_the_token_budget():
    plain = parse_args(["preset=prbo,gemma"])
    reasoning = parse_args(["preset=prbo,gemma_reasoning"])
    assert reasoning.chat_template_kwargs == {"enable_thinking": True}
    assert reasoning.target_max_tokens > plain.target_max_tokens


def test_black_box_demands_a_warm_start():
    # Its whole point is starting from the SFT checkpoint; without one it is
    # just CoM with a one-stage ladder.
    with pytest.raises(ValueError, match="load_checkpoint_path"):
        validate(parse_args(["preset=blackbox,gemma"]))
    validate(
        parse_args(
            ["preset=blackbox,gemma", f"load_checkpoint_path={PAPER_WARM_START}"]
        )
    )


def test_the_kl_reference_is_the_runs_own_starting_point():
    from investigator_trainer.train import kl_reference_path

    # tinker only accepts a sampler_weights path for the reference, and a
    # warm-started run must measure drift from its warm start, not the base.
    cfg = parse_args(
        ["preset=blackbox,gemma", f"load_checkpoint_path={PAPER_WARM_START}"]
    )
    assert kl_reference_path(cfg) == PAPER_WARM_START.replace(
        "/weights/", "/sampler_weights/"
    )
    assert kl_reference_path(parse_args(["preset=com,gemma"])) is None
