"""Heuristic investigator-refusal detection.

For logging only: ``investigator/is_refusal`` tracks how often the policy
refuses to write a prompt over training. Nothing is gated or skipped on it.
"""

import re

REFUSAL_RE = re.compile(
    r"\bI can['’]?t\b"
    r"|\bI cannot\b"
    r"|\bI won['’]?t\b"
    r"|\bI will not\b"
    r"|\bI['’]?m unable\b"
    r"|\bI am unable\b"
    r"|\bI must decline\b"
    r"|\bI have to decline\b"
    r"|\bI['’]?m not able\b"
    r"|\bI am not able\b"
    r"|\bI apologize\b"
    r"|\bI['’]?m sorry\b"
    r"|\bI am sorry\b"
    r"|\bsorry, but\b"
    r"|\bas an AI\b"
    r"|\bas a language model\b"
    r"|\bas an assistant\b"
    r"|\bagainst my\b"
    r"|\bI['’]?d rather not\b"
    r"|\bI would rather not\b"
    r"|\bI don['’]?t feel comfortable\b"
    r"|\bI['’]?m not comfortable\b"
    r"|\bI need to be responsible\b"
    r"|\bI refuse\b",
    re.IGNORECASE,
)

# Only the start of the prompt is searched: a jailbreak often quotes a refusal
# phrase later on ("do not say 'I cannot'"), which is not a refusal.
INVESTIGATOR_REFUSAL_WINDOW = 80


def is_investigator_refusal(text: str) -> bool:
    """Whether a refusal phrase appears in the first 80 characters."""
    return bool(REFUSAL_RE.search(text[:INVESTIGATOR_REFUSAL_WINDOW]))
