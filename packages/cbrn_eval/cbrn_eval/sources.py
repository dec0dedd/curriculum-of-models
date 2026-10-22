"""Prompt items: the (behavior, prompt) pairs a sweep evaluates.

Both prompt sources produce the same shape, so the runner does not care which
one is in use.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from common.behaviors import Behavior, resolve_behaviors

from .clients import TinkerInvestigatorBackend
from .config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptItem:
    behavior_id: str
    category: str | None
    behavior: str
    prompt_index: int
    prompt: str


def load_eval_behaviors(config: EvalConfig) -> list[Behavior]:
    return resolve_behaviors(
        behavior_rubric=None,
        behaviors_file=config.behaviors_file,
        behavior_ids=list(config.behavior_ids) if config.behavior_ids else None,
    )


def _item(b: Behavior, idx: int, prompt: str) -> PromptItem:
    return PromptItem(
        behavior_id=b.behavior_id,
        category=b.category,
        behavior=b.behavior,
        prompt_index=idx,
        prompt=prompt,
    )


def _load_saved_prompts(path: Path | None) -> dict[tuple[str, int], str]:
    if path is None or not path.exists():
        return {}
    saved = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                saved[(r["behavior_id"], r["prompt_index"])] = r["prompt"]
    return saved


def _save_prompt(path: Path | None, item: PromptItem) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "behavior_id": item.behavior_id,
        "prompt_index": item.prompt_index,
        "prompt": item.prompt,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


async def build_prompt_items(
    config: EvalConfig, prompts_file: Path | None = None
) -> list[PromptItem]:
    """Every prompt item the sweep evaluates.

    For ``prompt_source="behaviors"`` each behavior's own text is the prompt
    (the direct-ask baseline). For ``"investigator"`` the investigator model
    samples ``n_investigator_generations`` prompts per behavior, with a
    bounded worker pool rather than one task per prompt up front -- at the
    paper's generation counts that would otherwise materialize tens of
    thousands of tasks simultaneously.

    With ``prompts_file``, each investigator prompt is appended to it as it is
    sampled, and prompts already in it are reused: a resumed sweep samples
    only the ones it is missing.
    """
    behaviors = load_eval_behaviors(config)

    if config.prompt_source == "behaviors":
        return [_item(b, 0, b.behavior) for b in behaviors]
    if config.prompt_source != "investigator":
        raise ValueError(f"Unknown prompt_source: {config.prompt_source!r}")

    saved = _load_saved_prompts(prompts_file)
    items: list[PromptItem] = []
    jobs: asyncio.Queue[tuple[Behavior, int]] = asyncio.Queue()
    for b in behaviors:
        for idx in range(config.n_investigator_generations):
            prompt = saved.get((b.behavior_id, idx))
            if prompt is None:
                jobs.put_nowait((b, idx))
            else:
                items.append(_item(b, idx, prompt))
    if items:
        logger.info(f"Reusing {len(items)} saved investigator prompt(s)")

    if not jobs.empty():
        backend = await TinkerInvestigatorBackend.create(config)
        lock = asyncio.Lock()

        async def _worker() -> None:
            while True:
                try:
                    b, idx = jobs.get_nowait()
                except asyncio.QueueEmpty:
                    return
                item = _item(b, idx, await backend.generate(b.behavior))
                async with lock:
                    items.append(item)
                    _save_prompt(prompts_file, item)

        n_workers = min(config.investigator_concurrency, jobs.qsize())
        await asyncio.gather(*(_worker() for _ in range(n_workers)))

    items.sort(key=lambda it: (it.behavior_id, it.prompt_index))
    return items
