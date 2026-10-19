"""Investigator and judge prompts, loaded from the ``prompts/`` directory.

Format the behavior into ``{behavior_rubric}``, the target response into the
response-judge user template's ``{target_response}``, and the investigator
prompt into the prompt-judge user template's ``{investigator_prompt}``. Literal
JSON braces in the judge prompts are escaped as ``{{`` / ``}}``.
"""

from importlib.resources import files

_PROMPTS = files(__package__) / "prompts"


def _load(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


INVESTIGATOR_SYSTEM_PROMPT = _load("investigator_system.txt")
RESPONSE_JUDGE_SYSTEM_PROMPT = _load("response_judge_system.txt")
RESPONSE_JUDGE_USER_TEMPLATE = _load("response_judge_user.txt")
PROMPT_JUDGE_SYSTEM_PROMPT = _load("prompt_judge_system.txt")
PROMPT_JUDGE_USER_TEMPLATE = _load("prompt_judge_user.txt")
