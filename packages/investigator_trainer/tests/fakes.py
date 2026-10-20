"""Stand-ins for the vLLM target and the LLM judge: no network, fixed outputs."""

import torch

from common.scoring import CONTENT_CRITERIA, PROMPT_CRITERIA, RESPONSE_GATES
from common.target import NaturalCompletion, PRBOCompletion


def criteria_for_score(score: int) -> dict[str, bool]:
    """Response criteria that response_score maps to ``score`` (a multiple of 100/6)."""
    if score == 0:
        return {g: False for g in RESPONSE_GATES}
    met = round(score / 100 * len(CONTENT_CRITERIA))
    return {
        **{g: True for g in RESPONSE_GATES},
        **{c: i < met for i, c in enumerate(CONTENT_CRITERIA)},
    }


class FakeJudge:
    def __init__(self, scores: dict[str, int], prompt_passes: bool = True):
        # Response text -> rubric score.
        self.scores = scores
        self.prompt_passes = prompt_passes
        self.prompt_calls = 0
        self.response_calls: list[list[str]] = []

    async def score_prompt(self, *, behavior, investigator_prompt):
        self.prompt_calls += 1
        return {c: self.prompt_passes for c in PROMPT_CRITERIA}, "prompt reasoning"

    async def score_responses(self, *, behavior, responses):
        self.response_calls.append(list(responses))
        return [(criteria_for_score(self.scores[r]), f"judged {r}") for r in responses]


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeTarget:
    """q samples "q<i>", the base model "natural<i>", the stage adapter "<lora><i>"."""

    max_tokens = 100
    tokenizer = FakeTokenizer()

    def __init__(self, *, q_log_prob=-1.0, p_log_prob=-3.0, splits_reasoning=False):
        self.q_log_prob = q_log_prob
        self.p_log_prob = p_log_prob
        self.splits_reasoning = splits_reasoning
        self.generate_calls: list[str | None] = []
        self.scored_conditioning: list | None = None

    async def generate(self, messages, n=1, lora_name=None):
        self.generate_calls.append(lora_name)
        prefix = lora_name or "natural"
        return [f"{prefix}{i}" for i in range(n)]

    async def generate_lora_prbo_completions(self, messages, *, n, lora_name):
        return [
            PRBOCompletion(
                conditioning_token_ids=(1, 2),
                response_token_ids=(10 + i,),
                response=f"q{i}",
                sample_log_prob=self.q_log_prob,
            )
            for i in range(n)
        ]

    async def sample_natural_completions(self, messages, *, n):
        return [
            NaturalCompletion(
                conditioning_token_ids=(1, 2, 99),
                response=f"natural{i}",
                reasoning_prefix=f"thought{i}",
            )
            for i in range(n)
        ]

    async def score_prbo_completions(self, completions, *, conditioning_token_ids=None):
        self.scored_conditioning = conditioning_token_ids
        return torch.full((len(completions),), self.p_log_prob)
