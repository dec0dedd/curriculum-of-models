"""The target model, served by vLLM with the refusal-free LoRA ladder loaded.

Two kinds of call:

* ``generate`` -- chat sampling under a named adapter (or the base weights).
  CoM samples its current curriculum stage this way, the eval samples every
  response this way, and PRBO samples a non-reasoning target's natural
  responses this way.
* Token-exact PRBO: ``generate_lora_prbo_completions`` samples q (the
  refusal-free adapter) and records each response's exact token IDs and
  log-probability, and ``score_prbo_completions`` replays those same IDs under
  the base weights to get log p. Working in token IDs, not text, keeps p and q
  over exactly the same tokens.

For a reasoning target (gpt-oss, or Gemma 4 with ``enable_thinking``), PRBO
scores the final response tokens only: each q sample is split into reasoning
(conditioning) and response, and p conditions on reasoning sampled from the
base weights rather than on the adapter's reasoning.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import httpx
import torch
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from common.reasoning_format import (
    NaturalCompletionSplitError,
    ReasoningFormatter,
    get_reasoning_formatter,
    reasoning_is_enabled,
)

logger = logging.getLogger(__name__)

# A natural reasoning sample that never reaches its final response is resampled
# up to this many times.
REASONING_SPLIT_MAX_RETRIES = 3

# A long generation under a busy server takes minutes; retrying is safe because
# q is resampled fresh and the p replay is deterministic.
TARGET_TIMEOUT = httpx.Timeout(1200.0, connect=10.0)
TARGET_MAX_RETRIES = 2


class PRBOCompletionError(RuntimeError):
    """A token-exact PRBO completion could not be sampled or scored."""


@dataclass(frozen=True)
class PRBOCompletion:
    """A response with its exact conditioning and sampled token IDs."""

    conditioning_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response: str
    # Sum of the sampled tokens' log-probabilities under the sampling weights.
    sample_log_prob: float


@dataclass(frozen=True)
class NaturalCompletion:
    """A base-weights reasoning prefix and the final response that followed it."""

    conditioning_token_ids: tuple[int, ...]
    response: str
    reasoning_prefix: str


class VLLMTarget:
    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000",
        max_tokens: int = 7000,
        temperature: float = 1.0,
        chat_template_kwargs: dict | None = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.client = AsyncOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            timeout=TARGET_TIMEOUT,
            max_retries=TARGET_MAX_RETRIES,
        )

    # ── Chat sampling (CoM stages, PRBO natural responses) ─────────────────────

    async def generate(
        self, messages: list[dict], n: int = 1, lora_name: str | None = None
    ) -> list[str]:
        """Sample ``n`` responses. ``lora_name=None`` is the base model.

        Served with a reasoning parser, vLLM returns the reasoning separately,
        so ``content`` is the final response only.
        """
        response = await self.client.chat.completions.create(
            model=lora_name or self.model_id,
            messages=messages,
            n=n,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body={"chat_template_kwargs": self.chat_template_kwargs},
        )
        texts = [choice.message.content or "" for choice in response.choices]
        if any(not t for t in texts):
            logger.warning("Target generated an empty response.")
        return texts

    # ── Token-exact PRBO ───────────────────────────────────────────────────────

    @cached_property
    def tokenizer(self) -> Any:
        """Loaded on first use: ``generate`` needs none, so a served adapter name
        such as ``alpha_1_0``, which is not an HF repo, works as ``model_id``."""
        return AutoTokenizer.from_pretrained(self.model_id)

    @cached_property
    def splits_reasoning(self) -> bool:
        """Whether this target reasons first, so PRBO scores the response only."""
        return reasoning_is_enabled(self.model_id, self.chat_template_kwargs)

    @cached_property
    def _formatter(self) -> ReasoningFormatter:
        return get_reasoning_formatter(self.model_id, self.tokenizer)

    def render_prompt_token_ids(self, messages: list[dict]) -> list[int]:
        """The chat prompt as token IDs, under the same template kwargs as ``generate``.

        These must match what the target saw: PRBO replays ``conditioning +
        response`` IDs, and a prompt rendered under different settings (Gemma 4's
        template changes with ``enable_thinking``) silently corrupts log p.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if not isinstance(encoded, Sequence) or any(
            not isinstance(t, int) for t in encoded
        ):
            raise TypeError("Chat template did not return a flat token-ID sequence")
        return list(encoded)

    async def generate_lora_prbo_completions(
        self, messages: list[dict], *, n: int, lora_name: str
    ) -> list[PRBOCompletion]:
        """Sample ``n`` q completions from the refusal-free adapter, token-exact."""
        prompt_ids = self.render_prompt_token_ids(messages)
        choices = await self._token_completions(
            model=lora_name, prompt_token_ids=prompt_ids, n=n, logprobs=True
        )
        if self.splits_reasoning:
            return [self._split_reasoning_proposal(c, prompt_ids) for c in choices]
        return [self._plain_proposal(c, prompt_ids) for c in choices]

    async def sample_natural_completions(
        self, messages: list[dict], *, n: int
    ) -> list[NaturalCompletion]:
        """For a reasoning target: ``n`` base-weights reasoning prefixes and responses.

        p scores each q response after reasoning the unsteered target would
        itself have written, not after the adapter's reasoning.
        """
        base_prompt_ids = tuple(self.render_prompt_token_ids(messages))
        return list(
            await asyncio.gather(
                *(self._natural_completion(i, base_prompt_ids) for i in range(n))
            )
        )

    async def score_prbo_completions(
        self,
        completions: Sequence[PRBOCompletion],
        *,
        conditioning_token_ids: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor:
        """log p: replay each response's exact IDs under the base weights and sum.

        ``conditioning_token_ids`` replaces each completion's own prefix (the
        natural reasoning, for a reasoning target).
        """
        if conditioning_token_ids is None:
            conditioning_token_ids = [c.conditioning_token_ids for c in completions]
        if len(conditioning_token_ids) != len(completions):
            raise ValueError("Each PRBO completion needs one conditioning prefix")
        if not completions:
            return torch.empty(0, dtype=torch.float32)

        replay_ids = [
            list(prefix) + list(c.response_token_ids)
            for prefix, c in zip(conditioning_token_ids, completions, strict=True)
        ]
        response = await self.client.completions.create(
            model=self.model_id,
            prompt=replay_ids,
            max_tokens=1,
            logprobs=1,
            echo=True,
            extra_body={"add_special_tokens": False, "return_token_ids": True},
        )
        if len(response.choices) != len(completions):
            raise PRBOCompletionError(
                "vLLM returned the wrong number of replay choices"
            )

        log_probs = []
        for choice, prefix, completion, ids in zip(
            response.choices,
            conditioning_token_ids,
            completions,
            replay_ids,
            strict=True,
        ):
            returned = _choice_field(choice, "prompt_token_ids")
            if returned is not None and list(returned) != ids:
                raise PRBOCompletionError(
                    "vLLM changed token IDs while replaying a response"
                )
            log_probs.append(
                _sum_span(
                    _choice_log_probs(choice),
                    start=len(prefix),
                    count=len(completion.response_token_ids),
                    operation="PRBO replay",
                )
            )
        return torch.tensor(log_probs, dtype=torch.float32)

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _token_completions(
        self,
        *,
        model: str,
        prompt_token_ids: Sequence[int],
        n: int,
        logprobs: bool,
    ) -> list[Any]:
        request: dict[str, Any] = {
            "model": model,
            "prompt": list(prompt_token_ids),
            "n": n,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_body": {
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "return_token_ids": True,
            },
        }
        if logprobs:
            request["logprobs"] = 1
        response = await self.client.completions.create(**request)
        choices = list(response.choices)
        if len(choices) != n:
            raise PRBOCompletionError(f"vLLM returned {len(choices)} choices for n={n}")
        return choices

    def _plain_proposal(self, choice: Any, prompt_ids: Sequence[int]) -> PRBOCompletion:
        generated = _choice_token_ids(choice)
        response_ids = list(generated)
        eos = self.tokenizer.eos_token_id
        while response_ids and isinstance(eos, int) and response_ids[-1] == eos:
            response_ids.pop()
        if not response_ids:
            raise PRBOCompletionError("Proposal generated an empty response")
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        if not text:
            raise PRBOCompletionError("Proposal response was empty after decoding")
        return PRBOCompletion(
            conditioning_token_ids=tuple(prompt_ids),
            response_token_ids=tuple(response_ids),
            response=text,
            sample_log_prob=_sum_span(
                _choice_log_probs(choice),
                start=0,
                count=len(response_ids),
                operation="proposal sampling",
            ),
        )

    def _split_reasoning_proposal(
        self, choice: Any, prompt_ids: Sequence[int]
    ) -> PRBOCompletion:
        generated = _choice_token_ids(choice)
        try:
            parts = self._formatter.split_natural_completion(prompt_ids, generated)
        except NaturalCompletionSplitError as error:
            raise PRBOCompletionError(
                f"Proposal could not be split: {error}"
            ) from error
        # The sampled log-probs cover every generated token; the reasoning ones
        # condition q rather than being scored under it.
        response_start = len(parts.conditioning_token_ids) - len(prompt_ids)
        return PRBOCompletion(
            conditioning_token_ids=parts.conditioning_token_ids,
            response_token_ids=parts.response_token_ids,
            response=self._formatter.decode_response(parts.response_token_ids),
            sample_log_prob=_sum_span(
                _choice_log_probs(choice),
                start=response_start,
                count=len(parts.response_token_ids),
                operation="proposal sampling",
            ),
        )

    async def _natural_completion(
        self, sample_index: int, base_prompt_ids: tuple[int, ...]
    ) -> NaturalCompletion:
        last_error: NaturalCompletionSplitError | None = None
        for attempt in range(REASONING_SPLIT_MAX_RETRIES + 1):
            (choice,) = await self._token_completions(
                model=self.model_id,
                prompt_token_ids=base_prompt_ids,
                n=1,
                logprobs=False,
            )
            try:
                parts = self._formatter.split_natural_completion(
                    base_prompt_ids, _choice_token_ids(choice)
                )
            except NaturalCompletionSplitError as error:
                last_error = error
                logger.warning(
                    "Natural sample %d attempt %d could not be split; retrying: %s",
                    sample_index,
                    attempt + 1,
                    error,
                )
                continue
            return NaturalCompletion(
                conditioning_token_ids=parts.conditioning_token_ids,
                response=self._formatter.decode_response(parts.response_token_ids),
                reasoning_prefix=parts.reasoning_prefix,
            )
        raise PRBOCompletionError(
            f"Natural reasoning sample {sample_index} could not be split after "
            f"{REASONING_SPLIT_MAX_RETRIES + 1} attempts: {last_error}"
        ) from last_error


def _choice_field(choice: Any, name: str) -> Any:
    value = getattr(choice, name, None)
    if value is not None:
        return value
    extra = getattr(choice, "model_extra", None)
    return extra.get(name) if isinstance(extra, Mapping) else None


def _choice_token_ids(choice: Any) -> tuple[int, ...]:
    token_ids = _choice_field(choice, "token_ids")
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise PRBOCompletionError(
            "vLLM did not return generated token IDs (needs return_token_ids support)"
        )
    if any(not isinstance(t, int) for t in token_ids):
        raise PRBOCompletionError("vLLM returned invalid generated token IDs")
    return tuple(token_ids)


def _choice_log_probs(choice: Any) -> tuple[float | None, ...]:
    values = getattr(getattr(choice, "logprobs", None), "token_logprobs", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PRBOCompletionError("vLLM did not return token log-probabilities")
    return tuple(None if v is None else float(v) for v in values)


def _sum_span(
    values: Sequence[float | None], *, start: int, count: int, operation: str
) -> float:
    span = values[start : start + count]
    if len(span) != count:
        raise PRBOCompletionError(
            f"{operation} returned {len(span)} log-probabilities for {count} tokens"
        )
    missing = [i for i, v in enumerate(span) if v is None]
    if missing:
        raise PRBOCompletionError(
            f"{operation} is missing log-probabilities at response positions {missing}"
        )
    return float(sum(span))
