"""Prompt items: the (behavior, prompt) pairs a sweep evaluates.

Both prompt sources produce the same shape, so the runner does not care which
one is in use.
"""

import asyncio
from dataclasses import dataclass

from common.behaviors import Behavior, resolve_behaviors
from common.refusal import is_investigator_refusal

from .clients import TinkerInvestigatorBackend
from .config import EvalConfig


@dataclass(frozen=True)
class PromptItem:
    behavior_id: str
    category: str | None
    behavior: str
    prompt_index: int
    prompt: str
    investigator_refusal: bool


def load_eval_behaviors(config: EvalConfig) -> list[Behavior]:
    return resolve_behaviors(
        behavior_rubric=None,
        behaviors_file=config.behaviors_file,
        behavior_ids=list(config.behavior_ids) if config.behavior_ids else None,
    )


def expected_prompt_total(config: EvalConfig) -> int:
    """How many prompt items a sweep over ``config`` will produce."""
    n_behaviors = len(load_eval_behaviors(config))
    if config.prompt_source == "behaviors":
        return n_behaviors
    if config.prompt_source == "investigator":
        return n_behaviors * config.n_investigator_generations
    raise ValueError(f"Unknown prompt_source: {config.prompt_source!r}")


def _item(b: Behavior, idx: int, prompt: str) -> PromptItem:
    return PromptItem(
        behavior_id=b.behavior_id,
        category=b.category,
        behavior=b.behavior,
        prompt_index=idx,
        prompt=prompt,
        investigator_refusal=is_investigator_refusal(prompt),
    )


async def build_prompt_items(config: EvalConfig) -> list[PromptItem]:
    """Every prompt item the sweep evaluates.

    For ``prompt_source="behaviors"`` each behavior's own text is the prompt
    (the direct-ask baseline). For ``"investigator"`` the investigator model
    samples ``n_investigator_generations`` prompts per behavior, with a
    bounded worker pool rather than one task per prompt up front -- at the
    paper's generation counts that would otherwise materialize tens of
    thousands of tasks simultaneously.
    """
    behaviors = load_eval_behaviors(config)

    if config.prompt_source == "behaviors":
        return [_item(b, 0, b.behavior) for b in behaviors]
    if config.prompt_source != "investigator":
        raise ValueError(f"Unknown prompt_source: {config.prompt_source!r}")

    backend = await TinkerInvestigatorBackend.create(config)
    jobs: asyncio.Queue[tuple[Behavior, int]] = asyncio.Queue()
    for b in behaviors:
        for idx in range(config.n_investigator_generations):
            jobs.put_nowait((b, idx))

    items: list[PromptItem] = []
    lock = asyncio.Lock()

    async def _worker() -> None:
        while True:
            try:
                b, idx = jobs.get_nowait()
            except asyncio.QueueEmpty:
                return
            prompt = await backend.generate(b.behavior)
            async with lock:
                items.append(_item(b, idx, prompt))

    n_workers = min(config.investigator_concurrency, jobs.qsize())
    await asyncio.gather(*(_worker() for _ in range(n_workers)))
    items.sort(key=lambda it: (it.behavior_id, it.prompt_index))
    return items
