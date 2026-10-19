"""The investigator client: native Tinker sampling from a trained checkpoint.

Bypasses the OpenAI-compatible REST shim, which rate-limits aggressively at
the concurrency a sweep uses; the native gRPC path does not, and is what the
trainer itself samples with.
"""

import logging

import tinker
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from common.prompts import INVESTIGATOR_SYSTEM_PROMPT

from .config import EvalConfig

logger = logging.getLogger(__name__)


class TinkerInvestigatorBackend:
    """Samples investigator prompts from a tinker:// sampler_weights checkpoint."""

    def __init__(self, sampling_client, tokenizer, renderer, params):
        self._client = sampling_client
        self._tokenizer = tokenizer
        self._renderer = renderer
        self._params = params

    @classmethod
    async def create(cls, config: EvalConfig) -> "TinkerInvestigatorBackend":
        tokenizer = get_tokenizer(config.investigator_base_model)
        renderer = get_renderer(config.investigator_renderer, tokenizer)
        service = tinker.ServiceClient()
        sampling_client = await service.create_sampling_client_async(
            model_path=config.investigator_model,
            base_model=config.investigator_base_model,
        )
        params = tinker.SamplingParams(
            max_tokens=config.investigator_max_tokens,
            temperature=config.investigator_temperature,
            stop=renderer.get_stop_sequences(),
        )
        return cls(sampling_client, tokenizer, renderer, params)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def generate(self, behavior: str) -> str:
        system = INVESTIGATOR_SYSTEM_PROMPT.format(behavior_rubric=behavior)
        model_input = self._renderer.build_generation_prompt(
            [{"role": "system", "content": system}]
        )
        result = await self._client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=self._params
        )
        # Tinker includes the stop token in the returned sequence;
        # skip_special_tokens drops it so prompts carry no stray control tokens.
        return self._tokenizer.decode(
            result.sequences[0].tokens, skip_special_tokens=True
        ).strip()
