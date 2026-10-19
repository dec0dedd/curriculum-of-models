"""The LLM judge: one call scores the investigator prompt, one per target response.

By default every target response is judged. ``skip_refusals=True`` instead
scores a regex-detected refusal 0 without a judge call. The paper's PRBO runs
used it and its CoM runs did not, so the method presets set it to match. It is
off by default because it can only produce false zeros, e.g. on hedged
compliance ("While I cannot X, I can provide Y...") followed by the content.
"""

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from common.prompts import (
    PROMPT_JUDGE_SYSTEM_PROMPT,
    PROMPT_JUDGE_USER_TEMPLATE,
    RESPONSE_JUDGE_SYSTEM_PROMPT,
    RESPONSE_JUDGE_USER_TEMPLATE,
)
from common.refusal import is_refusal
from common.scoring import PROMPT_CRITERIA, RESPONSE_CRITERIA, to_bool

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The cheapest gpt-oss-120b providers on OpenRouter, in order. Pinning them also
# keeps cross-provider quantization differences out of the reward.
JUDGE_PROVIDERS = ["wandb", "deepinfra", "dekallm"]

# A stalled socket never raises, so without a per-attempt timeout one hung call
# holds up a whole training step. The deadline bounds one judge call across all
# retries; a call that exhausts it returns an empty verdict (every criterion
# false, score 0) and is counted in Judge.failed_calls.
REQUEST_TIMEOUT_S = 120.0
TOTAL_DEADLINE_S = 600.0
MAX_ATTEMPTS = 10

# The reasoning returned with the empty verdict of a call that ran out of retries.
JUDGE_FAILED = "[JUDGE FAILED]"


def extract_json_object(content: str) -> dict:
    """The first JSON object in a reply, tolerating code fences and prose."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    content = content.strip()
    if "{" in content and "}" not in content:
        content += "\n}"  # a reply truncated before its closing brace
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in judge reply: {content[:200]!r}")
    return json.loads(content[start : end + 1])


def extract_reasoning(message: Any) -> str | None:
    """The reasoning trace of a chat message, wherever the provider put it."""
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = getattr(message, key, None) or extra.get(key)
        if value:
            return value.strip() if isinstance(value, str) else str(value)
    details = extra.get("reasoning_details") or []
    text = " ".join(
        block.get("text", "")
        for block in details
        if block.get("type") == "reasoning.text"
    ).strip()
    return text or None


def parse_criteria(content: str, criteria: Sequence[str]) -> dict[str, bool]:
    """Every expected criterion as a bool. A missing one raises, so it is retried."""
    parsed = extract_json_object(content)
    missing = [c for c in criteria if c not in parsed]
    if missing:
        raise ValueError(f"Judge reply missing criteria {missing}: {content[:200]!r}")
    return {c: to_bool(parsed[c]) for c in criteria}


class Judge:
    """Scores investigator prompts and target responses with an LLM judge.

    The defaults are the paper's judge configuration. Changing any of them
    changes the reward signal.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        max_tokens: int = 10000,
        reasoning_effort: str | None = "medium",
        providers: list[str] | None = None,
        skip_refusals: bool = False,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        # max_retries=0: tenacity below is the only retry layer, so the timeout
        # and deadline mean what they say.
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.providers = JUDGE_PROVIDERS if providers is None else providers
        self.skip_refusals = skip_refusals

        self.total_calls = 0
        self.failed_calls = 0
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.skipped_calls = 0

    async def _call(
        self, system_prompt: str, user_message: str, criteria: Sequence[str]
    ) -> tuple[dict[str, bool], str]:
        """One judge call: (criteria, reasoning). ({}, reason) once retries run out."""

        def _count_retry(_state) -> None:
            self.retries += 1

        @retry(
            stop=stop_after_attempt(MAX_ATTEMPTS) | stop_after_delay(TOTAL_DEADLINE_S),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=_count_retry,
            reraise=True,
        )
        async def _attempt() -> tuple[dict[str, bool], str]:
            extra_body: dict[str, Any] = {"usage": {"include": True}}
            if self.reasoning_effort:
                extra_body["reasoning"] = {"effort": self.reasoning_effort}
            if self.providers:
                extra_body["provider"] = {
                    "order": self.providers,
                    "allow_fallbacks": True,
                }
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                timeout=REQUEST_TIMEOUT_S,
                extra_body=extra_body,
            )
            self.total_calls += 1
            usage = response.usage
            if usage is not None:
                self.prompt_tokens += usage.prompt_tokens or 0
                self.completion_tokens += usage.completion_tokens or 0
                cost = getattr(usage, "cost", None) or (usage.model_extra or {}).get(
                    "cost"
                )
                if cost is not None:
                    self.cost_usd += float(cost)

            message = response.choices[0].message
            content = (message.content or "").strip()
            if not content:
                raise ValueError(
                    f"Empty judge reply (finish_reason={response.choices[0].finish_reason!r})"
                )
            reasoning = extract_reasoning(message) or ""
            return parse_criteria(content, criteria), reasoning

        try:
            return await _attempt()
        except Exception:
            self.failed_calls += 1
            logger.exception(
                "Judge call failed after retries; returning an empty verdict."
            )
            return {}, JUDGE_FAILED

    async def score_prompt(
        self, *, behavior: str, investigator_prompt: str
    ) -> tuple[dict[str, bool], str]:
        """Judge an investigator prompt: (criteria, reasoning)."""
        return await self._call(
            PROMPT_JUDGE_SYSTEM_PROMPT.format(behavior_rubric=behavior),
            PROMPT_JUDGE_USER_TEMPLATE.format(investigator_prompt=investigator_prompt),
            PROMPT_CRITERIA,
        )

    async def score_responses(
        self, *, behavior: str, responses: Sequence[str]
    ) -> list[tuple[dict[str, bool], str]]:
        """Judge each target response, concurrently: one (criteria, reasoning) each."""
        system_prompt = RESPONSE_JUDGE_SYSTEM_PROMPT.format(behavior_rubric=behavior)

        async def _judge(response: str) -> tuple[dict[str, bool], str]:
            if self.skip_refusals and is_refusal(response):
                self.skipped_calls += 1
                return (
                    {},
                    "[SKIPPED: target response classified as a refusal, scored 0]",
                )
            return await self._call(
                system_prompt,
                RESPONSE_JUDGE_USER_TEMPLATE.format(
                    target_response=(
                        response
                        if response.strip()
                        else "[NO RESPONSE GENERATED / EMPTY]"
                    )
                ),
                RESPONSE_CRITERIA,
            )

        return list(await asyncio.gather(*(_judge(r) for r in responses)))

    def stats(self) -> dict[str, float]:
        """Cumulative counters, for logging."""
        return {
            "judge/total_calls": float(self.total_calls),
            "judge/failed_calls": float(self.failed_calls),
            "judge/skipped_calls": float(self.skipped_calls),
            "judge/retries": float(self.retries),
            "judge/prompt_tokens": float(self.prompt_tokens),
            "judge/completion_tokens": float(self.completion_tokens),
            "judge/cost_usd": self.cost_usd,
        }
