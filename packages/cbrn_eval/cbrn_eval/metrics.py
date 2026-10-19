"""Evaluation metrics shared across the cbrn_eval notebooks and code.

The canonical home for ``pass_at_k`` so notebooks import one definition rather
than each carrying a copy.
"""

from math import prod

__all__ = ["pass_at_k"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate that k samples drawn from n contain at least one pass.

    Given ``n`` rollouts of which ``c`` passed, this is the probability that a
    random draw of ``k`` of them contains at least one pass -- the pass@k of
    Chen et al. (2021, "Evaluating Large Language Models Trained on Code"),
    computed in the numerically stable product form

        pass@k = 1 - prod_{i=n-c+1}^{n} (1 - k / i)

    rather than the ratio of binomial coefficients, which overflows.

    Args:
        n: total rollouts sampled.
        c: how many of them passed (0 <= c <= n).
        k: draw size; clamped to n, since you cannot draw more than you have.

    Returns:
        The pass@k probability in [0, 1]. With k == 1 this is the empirical
        pass rate c / n; with c == 0 it is 0.0; with c == n it is 1.0.
    """
    n, c, k = int(n), int(c), int(k)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, {n}], got {c}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, n)
    if n - c < k:  # fewer than k failures -> every draw of k hits a pass
        return 1.0
    return 1.0 - prod(1.0 - k / i for i in range(n - c + 1, n + 1))
