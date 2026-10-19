from math import comb

import pytest

from cbrn_eval.metrics import pass_at_k


def brute_pass_at_k(n: int, c: int, k: int) -> float:
    """Reference: 1 - C(n-c, k)/C(n, k), the closed form pass_at_k estimates."""
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def test_boundary_cases():
    assert pass_at_k(10, 0, 5) == 0.0  # no passes -> never
    assert pass_at_k(10, 10, 5) == 1.0  # all pass -> always
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)  # k=1 is the empirical rate
    assert pass_at_k(10, 1, 10) == 1.0  # draw everything, the one pass is in it


def test_matches_closed_form():
    for n in (1, 8, 64, 256):
        for c in sorted({0, 1, 5, n // 2, n} & set(range(n + 1))):
            for k in (1, 2, 32, n):
                assert pass_at_k(n, c, k) == pytest.approx(
                    brute_pass_at_k(n, c, k), abs=1e-9
                )


def test_k_is_clamped_to_n():
    assert pass_at_k(5, 2, 100) == pass_at_k(5, 2, 5)


def test_validates_arguments():
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)
    with pytest.raises(ValueError):
        pass_at_k(10, 11, 1)  # c > n
    with pytest.raises(ValueError):
        pass_at_k(10, -1, 1)
    with pytest.raises(ValueError):
        pass_at_k(10, 3, 0)  # k < 1
