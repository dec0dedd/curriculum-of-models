"""The adapter-naming contract between the ladder builder and the trainer."""

import pytest

from common.adapters import (
    FULL_STRENGTH_ADAPTER,
    adapter_name_for_scale,
    lora_name_for_scale,
    validate_ladder,
)


def test_names_are_the_scale_with_a_dot_as_underscore():
    assert adapter_name_for_scale(1.0) == "alpha_1_0"
    assert adapter_name_for_scale(0.6) == "alpha_0_6"
    assert adapter_name_for_scale(0.05) == "alpha_0_05"
    assert adapter_name_for_scale(0.0) == "alpha_0_0"


def test_float_noise_does_not_change_a_name():
    # A ladder built by repeated addition must land on the served name.
    assert adapter_name_for_scale(3 * 0.2) == "alpha_0_6"
    assert adapter_name_for_scale(1 - 0.9) == "alpha_0_1"


def test_full_strength_is_the_proposal_adapter():
    assert FULL_STRENGTH_ADAPTER == adapter_name_for_scale(1.0)


def test_scale_zero_means_no_adapter_for_the_trainer():
    # The base model is served by passing no adapter, but the builder still
    # writes an alpha_0_0 directory as an eval control.
    assert lora_name_for_scale(0.0) is None
    assert adapter_name_for_scale(0.0) == "alpha_0_0"
    assert lora_name_for_scale(0.4) == "alpha_0_4"


@pytest.mark.parametrize("scale", [-0.1, 1.1, 2.0])
def test_a_scale_outside_the_unit_range_is_rejected(scale):
    with pytest.raises(ValueError):
        adapter_name_for_scale(scale)


def test_the_papers_ladder_is_valid():
    validate_ladder([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    validate_ladder([0.0])  # Black-box: the base target only


@pytest.mark.parametrize(
    "scales", [[], [0.5, 0.5], [0.0, 1.0], [1.2, 0.0], [0.2, 0.4, 0.6]]
)
def test_invalid_ladders_are_rejected(scales):
    with pytest.raises(ValueError):
        validate_ladder(scales)
