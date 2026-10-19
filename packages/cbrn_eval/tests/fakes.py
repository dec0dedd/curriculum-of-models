"""Stand-ins for the vLLM target and the LLM judge: no network, fixed outputs."""

from common.scoring import CONTENT_CRITERIA, PROMPT_CRITERIA, RESPONSE_GATES


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
    """Scores responses from a response-text -> score table, with a fallback."""

    def __init__(
        self,
        scores: dict[str, int] | None = None,
        prompt_passes: bool = True,
        default_score: int | None = None,
    ):
        self.scores = scores or {}
        self.prompt_passes = prompt_passes
        self.default_score = default_score
        self.prompt_calls: list[str] = []
        self.response_calls: list[list[str]] = []

    async def score_prompt(self, *, behavior, investigator_prompt):
        self.prompt_calls.append(investigator_prompt)
        return {c: self.prompt_passes for c in PROMPT_CRITERIA}, "prompt reasoning"

    async def score_responses(self, *, behavior, responses):
        self.response_calls.append(list(responses))
        return [
            (criteria_for_score(self.scores.get(r, self.default_score)), f"judged {r}")
            for r in responses
        ]


class FakeFailingJudge:
    """A judge that has exhausted its retries, as common.judge.Judge itself would."""

    async def score_prompt(self, *, behavior, investigator_prompt):
        return {}, "[JUDGE FAILED]"

    async def score_responses(self, *, behavior, responses):
        return [({}, "[JUDGE FAILED]") for _ in responses]


class FakeTarget:
    """Generates deterministic, numbered responses -- no network."""

    def __init__(self, prefix: str = "response"):
        self.prefix = prefix
        self.generate_calls: list[tuple[list[dict], int]] = []

    async def generate(self, messages, n=1, lora_name=None):
        self.generate_calls.append((messages, n))
        return [f"{self.prefix}{i}" for i in range(n)]


class FakeFailingTarget:
    def __init__(self, error: Exception):
        self.error = error

    async def generate(self, messages, n=1, lora_name=None):
        raise self.error
