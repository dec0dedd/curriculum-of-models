"""The naming contract for the refusal-free adapter ladder.

The ladder is one adapter served at several strengths: 1.0 is fully
refusal-free (easiest to elicit from), decreasing toward 0.0, the unmodified
target. vLLM registers each one under the name this module computes.

Both sides of the contract import it: the builder writes directories whose
names map here, and the trainer asks vLLM for exactly these names. If the two
ever disagreed, a run would die partway up the ladder on an unknown adapter.
"""

from collections.abc import Sequence


def adapter_name_for_scale(scale: float) -> str:
    """``0.6 -> "alpha_0_6"``. Rounding strips float noise, so 3 * 0.2 works."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"A ladder scale must be in [0, 1]; got {scale}")
    return f"alpha_{str(round(float(scale), 10)).replace('.', '_')}"


def lora_name_for_scale(scale: float) -> str | None:
    """The adapter name for a scale, or None for the base model at scale 0.0."""
    return None if scale == 0.0 else adapter_name_for_scale(scale)


def validate_ladder(scales: Sequence[float]) -> None:
    """Non-empty, each in [0, 1], strictly decreasing (easiest stage first)."""
    if not scales:
        raise ValueError("The ladder must have at least one scale.")
    if any(not 0.0 <= s <= 1.0 for s in scales):
        raise ValueError(f"Every ladder scale must be in [0, 1]; got {list(scales)}.")
    if any(a <= b for a, b in zip(scales, scales[1:])):
        raise ValueError(
            "The ladder must be strictly decreasing (easiest first, hardest "
            f"last); got {list(scales)}."
        )
