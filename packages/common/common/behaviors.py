"""Behaviors: loading, selecting, and scheduling them across training steps.

A single-behavior run is a multi-behavior run with one behavior, so there is
one code path for both.
"""

import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# The id of a behavior written inline with behavior_rubric=...
CUSTOM_BEHAVIOR_ID = "custom"


@dataclass(frozen=True)
class Behavior:
    behavior_id: str
    category: str | None
    behavior: str


def load_behaviors(path: str | Path) -> list[Behavior]:
    """Read a jsonl file with ``behavior_id``, ``category`` and ``behavior`` rows.

    Raises ValueError on a duplicate ``behavior_id``: ids key the GRPO groups,
    the per-behavior curriculum state and the metrics.
    """
    behaviors = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        behaviors.append(
            Behavior(
                behavior_id=row["behavior_id"],
                category=row.get("category"),
                behavior=row["behavior"],
            )
        )
    duplicates = sorted(
        i for i, n in Counter(b.behavior_id for b in behaviors).items() if n > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate behavior_id in {path}: {duplicates}")
    return behaviors


def resolve_behaviors(
    *,
    behavior_rubric: str | None,
    behaviors_file: str | None,
    behavior_ids: list[str] | None,
) -> list[Behavior]:
    """The behaviors of a run: one inline rubric, or rows of a behaviors file.

    Set exactly one of ``behavior_rubric`` and ``behaviors_file``.
    ``behavior_ids`` selects a subset of the file, in the order given.
    """
    if (behavior_rubric is None) == (behaviors_file is None):
        raise ValueError(
            "Set exactly one of behavior_rubric (one inline behavior) or "
            "behaviors_file (a behaviors dataset)."
        )

    if behavior_rubric is not None:
        if behavior_ids:
            raise ValueError("behavior_ids selects rows of behaviors_file")
        return [Behavior(CUSTOM_BEHAVIOR_ID, None, behavior_rubric)]

    behaviors = load_behaviors(behaviors_file)
    if behavior_ids:
        by_id = {b.behavior_id: b for b in behaviors}
        missing = [i for i in behavior_ids if i not in by_id]
        if missing:
            raise ValueError(f"behavior_ids not in {behaviors_file}: {missing}")
        # dict.fromkeys dedupes while keeping the order given.
        behaviors = [by_id[i] for i in dict.fromkeys(behavior_ids)]
    if not behaviors:
        raise ValueError(f"{behaviors_file} holds no behaviors")
    return behaviors


def behaviors_for_step(
    behaviors: Sequence[Behavior],
    *,
    index: int,
    batch_size: int,
    seed: int,
) -> list[Behavior]:
    """The ``batch_size`` behaviors of training step ``index``.

    Each epoch is a fresh seeded shuffle of the whole list, so every behavior
    is trained on once per epoch and per-behavior curves stay evenly sampled.
    When the pool does not divide by ``batch_size``, the last step of an epoch
    wraps into the head of the shuffle to stay full width.

    A step never holds the same behavior twice: two groups of one behavior
    would share its curriculum state and its metric keys.

    Depends only on ``index``, as tinker requires of ``RLDataset.get_batch``.
    """
    if batch_size > len(behaviors):
        raise ValueError(
            f"batch_size ({batch_size}) is larger than the {len(behaviors)} "
            "behavior(s) of the run, so a step would train against the same "
            "behavior twice. Lower batch_size, or train against more behaviors."
        )
    per_epoch = math.ceil(len(behaviors) / batch_size)
    epoch, slot = divmod(index, per_epoch)
    order = list(behaviors)
    # A string seed is stable across processes, unlike hash().
    random.Random(f"{seed}:{epoch}").shuffle(order)
    start = slot * batch_size
    return [order[i % len(order)] for i in range(start, start + batch_size)]
