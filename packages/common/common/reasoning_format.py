"""Splitting a reasoning target's completion into reasoning and final response.

PRBO scores the final response tokens only, so each completion of a reasoning
target is cut at the marker that opens its final response: everything before it
(prompt + reasoning) conditions the score, everything after is scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from transformers import PreTrainedTokenizerBase

# Both sizes share the harmony template and tokenizer.
GPT_OSS_MODEL_IDS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
# Gemma 4 reasons only when its template is rendered with enable_thinking.
GEMMA4_MODEL_PREFIX = "google/gemma-4"


class NaturalCompletionSplitError(ValueError):
    """A completion has no usable final response to split off."""


@dataclass(frozen=True)
class CompletionParts:
    conditioning_token_ids: tuple[int, ...]  # prompt + reasoning + final marker
    response_token_ids: tuple[int, ...]
    reasoning_prefix: str  # decoded with the channel markers, for the logs


class ReasoningFormatter(Protocol):
    def split_natural_completion(
        self, base_prompt_token_ids: Sequence[int], generated_token_ids: Sequence[int]
    ) -> CompletionParts: ...

    def decode_response(self, token_ids: Sequence[int]) -> str: ...


def _last_subsequence_index(values: Sequence[int], sub: Sequence[int]) -> int:
    if not sub or len(sub) > len(values):
        return -1
    for i in range(len(values) - len(sub), -1, -1):
        if tuple(values[i : i + len(sub)]) == tuple(sub):
            return i
    return -1


class _MarkerSplitFormatter:
    """Split at the last occurrence of a final-response marker."""

    final_response_marker: str
    stop_tokens: tuple[str, ...]

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer
        self._marker_ids = self._encode(self.final_response_marker)
        stop_ids: set[int] = set()
        if isinstance(tokenizer.eos_token_id, int):
            stop_ids.add(tokenizer.eos_token_id)
        for token in self.stop_tokens:
            ids = self._encode(token)
            if len(ids) == 1:
                stop_ids.add(ids[0])
        self._stop_ids = frozenset(stop_ids)

    def _encode(self, text: str) -> tuple[int, ...]:
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        ids = tuple(encoded)
        if not ids:
            raise ValueError(f"Structural marker {text!r} encoded to no tokens")
        return ids

    def _decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def split_natural_completion(
        self, base_prompt_token_ids: Sequence[int], generated_token_ids: Sequence[int]
    ) -> CompletionParts:
        marker_index = _last_subsequence_index(generated_token_ids, self._marker_ids)
        if marker_index < 0:
            raise NaturalCompletionSplitError(
                "Completion never reached its final response"
            )
        response_start = marker_index + len(self._marker_ids)
        response_ids = list(generated_token_ids[response_start:])
        while response_ids and response_ids[-1] in self._stop_ids:
            response_ids.pop()
        if not response_ids:
            raise NaturalCompletionSplitError("Completion has an empty final response")
        prefix = tuple(generated_token_ids[:response_start])
        return CompletionParts(
            conditioning_token_ids=tuple(base_prompt_token_ids) + prefix,
            response_token_ids=tuple(response_ids),
            reasoning_prefix=self._decode(prefix, skip_special_tokens=False),
        )

    def decode_response(self, token_ids: Sequence[int]) -> str:
        return self._decode(token_ids, skip_special_tokens=True)


class GPTOSSReasoningFormatter(_MarkerSplitFormatter):
    """The stock gpt-oss (harmony) template: analysis channel, then final."""

    final_response_marker = "<|end|><|start|>assistant<|channel|>final<|message|>"
    stop_tokens = ("<|return|>", "<|endoftext|>")


class Gemma4ReasoningFormatter(_MarkerSplitFormatter):
    """The stock Gemma 4 template with enable_thinking.

    The model opens ``<|channel>thought ... <channel|>`` itself, so the final
    response starts after the single ``<channel|>`` token that closes it.
    """

    final_response_marker = "<channel|>"
    stop_tokens = ("<turn|>", "<|turn>")


def _is_gpt_oss(model_id: str) -> bool:
    return model_id.casefold() in {m.casefold() for m in GPT_OSS_MODEL_IDS}


def reasoning_is_enabled(model_id: str, chat_template_kwargs: Mapping | None) -> bool:
    """gpt-oss always reasons; Gemma 4 only with ``enable_thinking``."""
    if _is_gpt_oss(model_id):
        return True
    if model_id.casefold().startswith(GEMMA4_MODEL_PREFIX):
        return bool((chat_template_kwargs or {}).get("enable_thinking"))
    return False


def get_reasoning_formatter(
    model_id: str, tokenizer: PreTrainedTokenizerBase
) -> ReasoningFormatter:
    if _is_gpt_oss(model_id):
        return GPTOSSReasoningFormatter(tokenizer)
    if model_id.casefold().startswith(GEMMA4_MODEL_PREFIX):
        return Gemma4ReasoningFormatter(tokenizer)
    raise ValueError(
        f"No reasoning formatter for {model_id!r}. Supported: "
        f"{', '.join(GPT_OSS_MODEL_IDS)} and {GEMMA4_MODEL_PREFIX}* models."
    )
