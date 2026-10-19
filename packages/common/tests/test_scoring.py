"""The judge's criteria -> the numbers. Training and eval share these."""

import pytest

from common.scoring import (
    CONTENT_CRITERIA,
    PROMPT_CRITERIA,
    RESPONSE_GATES,
    prompt_passes,
    response_score,
    to_bool,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        # The judge sometimes emits strings or numbers rather than JSON
        # booleans, and bool("false") is True.
        ("true", True),
        ("True", True),
        ("false", False),
        ("no", False),
        ("yes", True),
        ("1", True),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_to_bool_coerces_what_judges_actually_emit(value, expected):
    assert to_bool(value) is expected


def test_prompt_passes_needs_every_criterion():
    assert prompt_passes({c: True for c in PROMPT_CRITERIA})
    for missing in PROMPT_CRITERIA:
        criteria = {c: True for c in PROMPT_CRITERIA}
        criteria[missing] = False
        assert not prompt_passes(criteria)


def test_a_missing_criterion_fails_the_prompt_gate():
    # A judge call that failed returns {}, which must not pass.
    assert not prompt_passes({})
    assert not prompt_passes({PROMPT_CRITERIA[0]: True})


def test_response_score_is_zero_unless_every_gate_holds():
    full_content = {c: True for c in CONTENT_CRITERIA}
    for failed_gate in RESPONSE_GATES:
        criteria = {**{g: True for g in RESPONSE_GATES}, **full_content}
        criteria[failed_gate] = False
        assert response_score(criteria) == 0


def test_response_score_counts_content_criteria():
    gates = {g: True for g in RESPONSE_GATES}
    assert response_score(gates) == 0
    assert response_score({**gates, **{c: True for c in CONTENT_CRITERIA}}) == 100
    half = dict(
        list({c: True for c in CONTENT_CRITERIA}.items())[: len(CONTENT_CRITERIA) // 2]
    )
    assert response_score({**gates, **half}) == 50


def test_response_score_of_an_empty_verdict_is_zero():
    assert response_score({}) == 0
