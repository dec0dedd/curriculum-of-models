"""Behavior loading, selection, and the per-step schedule.

The schedule has to be reproducible: the same seed must replay the same
behavior order, or per-seed runs cannot be compared.
"""

from collections import Counter
from pathlib import Path

import pytest

from common.behaviors import (
    CUSTOM_BEHAVIOR_ID,
    Behavior,
    behaviors_for_step,
    load_behaviors,
    resolve_behaviors,
)

BEHAVIORS_FILE = (
    Path(__file__).resolve().parents[3] / "data/behaviors/transluce_cbrn.jsonl"
)


@pytest.fixture
def behaviors():
    return load_behaviors(BEHAVIORS_FILE)


def test_the_shipped_behaviors_file_loads(behaviors):
    assert len(behaviors) == 48
    assert all(b.behavior_id and b.behavior for b in behaviors)
    # Ids are the join key for per-behavior metrics and eval results.
    assert len({b.behavior_id for b in behaviors}) == len(behaviors)


def test_an_inline_rubric_is_a_one_behavior_run():
    resolved = resolve_behaviors(
        behavior_rubric="Do the thing.", behaviors_file=None, behavior_ids=None
    )
    assert resolved == [Behavior(CUSTOM_BEHAVIOR_ID, None, "Do the thing.")]


def test_exactly_one_source_is_required():
    with pytest.raises(ValueError):
        resolve_behaviors(behavior_rubric=None, behaviors_file=None, behavior_ids=None)
    with pytest.raises(ValueError):
        resolve_behaviors(
            behavior_rubric="x", behaviors_file=str(BEHAVIORS_FILE), behavior_ids=None
        )


def test_behavior_ids_select_a_subset_in_order():
    resolved = resolve_behaviors(
        behavior_rubric=None,
        behaviors_file=str(BEHAVIORS_FILE),
        behavior_ids=["transluce_vx", "transluce_mdma", "transluce_vx"],
    )
    # Deduped, order preserved. A single id is how single-behavior runs are set up.
    assert [b.behavior_id for b in resolved] == ["transluce_vx", "transluce_mdma"]


def test_an_unknown_behavior_id_fails_loudly():
    with pytest.raises(ValueError, match="not in"):
        resolve_behaviors(
            behavior_rubric=None,
            behaviors_file=str(BEHAVIORS_FILE),
            behavior_ids=["nope"],
        )


def test_the_schedule_is_reproducible(behaviors):
    for seed in (0, 1, 7):
        runs = [
            [
                b.behavior_id
                for b in behaviors_for_step(behaviors, index=i, batch_size=4, seed=seed)
            ]
            for i in range(20)
        ]
        again = [
            [
                b.behavior_id
                for b in behaviors_for_step(behaviors, index=i, batch_size=4, seed=seed)
            ]
            for i in range(20)
        ]
        assert runs == again


def test_different_seeds_give_different_orders(behaviors):
    a = [
        b.behavior_id
        for b in behaviors_for_step(behaviors, index=0, batch_size=8, seed=0)
    ]
    b = [
        b.behavior_id
        for b in behaviors_for_step(behaviors, index=0, batch_size=8, seed=1)
    ]
    assert a != b


def test_a_step_never_repeats_a_behavior(behaviors):
    for i in range(60):
        step = behaviors_for_step(behaviors, index=i, batch_size=7, seed=0)
        assert len(step) == 7
        assert len({b.behavior_id for b in step}) == 7


def test_every_behavior_appears_once_per_epoch(behaviors):
    # 48 behaviors, 4 per step -> 12 steps is one epoch.
    counts = Counter(
        b.behavior_id
        for i in range(12)
        for b in behaviors_for_step(behaviors, index=i, batch_size=4, seed=0)
    )
    assert set(counts.values()) == {1}
    assert len(counts) == len(behaviors)


def test_batch_size_larger_than_the_pool_is_rejected(behaviors):
    with pytest.raises(ValueError, match="larger than"):
        behaviors_for_step(behaviors[:2], index=0, batch_size=3, seed=0)
