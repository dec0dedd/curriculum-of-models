"""Sweep orchestration: prompt items -> target -> judge -> jsonl rows.

Per-stage semaphores bound target and judge concurrency independently, so the
two overlap. Rows are appended to full.jsonl as items complete; an item whose
every generation is already recorded is skipped, which is what makes a sweep
resumable. An item resumed with only some of its generations done is
reprocessed whole rather than topped up -- output.load_rows dedupes by key, so
the fresher rows simply replace the stale ones.
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timezone

from common.judge import Judge
from common.scoring import prompt_passes, response_score
from common.target import VLLMTarget

from . import output
from .config import EvalConfig
from .sources import PromptItem, build_prompt_items

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_S = 15


def _target_base_url(config: EvalConfig) -> str:
    """Strip a trailing /v1, however the caller spelled it: VLLMTarget adds it."""
    return config.target_base_url.removesuffix("/v1").rstrip("/")


def _expected_keys(config: EvalConfig, item: PromptItem) -> set[str]:
    return {
        f"{item.behavior_id}:{item.prompt_index}:{g}"
        for g in range(config.n_target_generations)
    }


def _row(
    run_id: str,
    item: PromptItem,
    *,
    generation_index: int | None,
    target_completion: str | None = None,
    judge_score: int | None = None,
    judge_criteria: dict | None = None,
    judge_reasoning: str | None = None,
    error: str | None = None,
) -> output.TrialRow:
    return output.TrialRow(
        run_id=run_id,
        behavior_id=item.behavior_id,
        category=item.category,
        prompt_index=item.prompt_index,
        generation_index=generation_index,
        prompt=item.prompt,
        target_completion=target_completion,
        judge_score=judge_score,
        judge_criteria=judge_criteria,
        judge_reasoning=judge_reasoning,
        error=error,
    )


async def _run_item(
    run_id: str,
    config: EvalConfig,
    item: PromptItem,
    target: VLLMTarget,
    judge: Judge,
    target_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
) -> list[output.TrialRow]:
    messages = [{"role": "user", "content": item.prompt}]
    try:
        async with target_sem:
            responses = await target.generate(messages, n=config.n_target_generations)
    except Exception as e:
        logger.warning(
            f"target failed for {item.behavior_id}:{item.prompt_index}: {e!r}"
        )
        return [
            _row(run_id, item, generation_index=g, error=f"target_error: {e!r}")
            for g in range(config.n_target_generations)
        ]

    async with judge_sem:
        verdicts = await judge.score_responses(
            behavior=item.behavior, responses=responses
        )

    rows = []
    for g, (completion, (criteria, reasoning)) in enumerate(
        zip(responses, verdicts, strict=True)
    ):
        judge_failed = not criteria  # Judge already retried; {} means it gave up
        rows.append(
            _row(
                run_id,
                item,
                generation_index=g,
                target_completion=completion,
                judge_score=None if judge_failed else response_score(criteria),
                judge_criteria=criteria or None,
                judge_reasoning=reasoning,
                error=f"judge_error: {reasoning}" if judge_failed else None,
            )
        )
    return rows


async def run_sweep(config: EvalConfig, run_id: str) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()

    done = output.load_completed_keys(config.output_dir, run_id)
    if done:
        logger.info(f"Resuming: {len(done)} trial(s) already completed")

    logger.info("Building prompt items...")
    items = await build_prompt_items(config)
    n_behaviors = len({i.behavior_id for i in items})
    logger.info(f"{len(items)} prompt item(s) across {n_behaviors} behavior(s)")

    target = VLLMTarget(
        model_id=config.target_model,
        base_url=_target_base_url(config),
        max_tokens=config.target_max_tokens,
        temperature=config.target_temperature,
        chat_template_kwargs={"enable_thinking": not config.target_disable_reasoning},
    )
    judge = Judge(model=config.judge_model, max_tokens=config.judge_max_tokens)
    target_sem = asyncio.Semaphore(config.target_concurrency)
    judge_sem = asyncio.Semaphore(config.judge_concurrency)
    write_lock = asyncio.Lock()

    n_done = 0
    n_total = len(items)

    async def _process(item: PromptItem) -> None:
        nonlocal n_done
        if _expected_keys(config, item) <= done:
            n_done += 1
            return

        prompt_verdict = (
            asyncio.create_task(
                judge.score_prompt(
                    behavior=item.behavior, investigator_prompt=item.prompt
                )
            )
            if config.use_prompt_judge
            else None
        )
        rows = await _run_item(
            run_id, config, item, target, judge, target_sem, judge_sem
        )

        prompt_criteria: dict | None = None
        prompt_passed: bool | None = None
        if prompt_verdict is not None:
            raw_criteria, _reasoning = await prompt_verdict
            prompt_passed = prompt_passes(raw_criteria)
            prompt_criteria = raw_criteria or None

        async with write_lock:
            for row in rows:
                row.prompt_criteria = prompt_criteria
                row.prompt_passed = prompt_passed
                # With no prompt judge, prompt_passed is None and gated_score
                # passes the judge score through unchanged -- nothing gates it.
                row.final_score = output.gated_score(
                    judge_score=row.judge_score, prompt_passed=prompt_passed
                )
                output.append_row(config.output_dir, run_id, row)
        n_done += 1

    async def _report_progress() -> None:
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL_S)
            logger.info(f"{n_done}/{n_total} prompt item(s) complete")

    reporter = asyncio.create_task(_report_progress())
    try:
        await asyncio.gather(*(_process(item) for item in items))
    finally:
        reporter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reporter

    logger.info(f"{n_done}/{n_total} prompt item(s) complete")
    rows = output.load_rows(config.output_dir, run_id)
    summary = output.write_summary(
        config.output_dir,
        run_id,
        config.to_dict(),
        started_at,
        datetime.now(timezone.utc).isoformat(),
        rows,
    )
    # Run complete: compact the streamed jsonl to a much smaller zstd parquet.
    output.compact_full(config.output_dir, run_id)
    return summary
