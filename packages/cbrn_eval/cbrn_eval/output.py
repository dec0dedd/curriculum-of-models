"""Output files: <output_dir>/<run_id>_full.{jsonl,parquet}, _prompts.jsonl and
_summary.json.

full.jsonl gets one row per trial, appended as trials complete -- its
presence without a summary marks a resumable run. prompts.jsonl holds the
sampled investigator prompts, so a resume reuses them rather than sampling
again. summary.json is written
once at the end and marks the run complete (the cache hit signal).

On completion the streamed full.jsonl is compacted to full.parquet (zstd) and
the jsonl removed, which is far smaller on disk. Loaders read whichever
exists, preferring the jsonl (the live, resumable form) when both are
present, and always dedupe by trial key -- a resumed run can re-process an
item that had only some of its generations done, which appends a row whose
key is already on disk; the later write wins.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl


@dataclass
class TrialRow:
    run_id: str
    behavior_id: str
    category: str | None
    prompt_index: int
    generation_index: int
    prompt: str
    target_completion: str | None
    judge_score: int | None  # computed from judge_criteria (common.scoring)
    judge_criteria: dict | None  # per-criterion booleans from the judge
    judge_reasoning: str | None
    error: str | None  # None | "target_error: ..." | "judge_error: [JUDGE FAILED]"
    # Prompt judge. None when it is disabled (--no-prompt-judge); a prompt-judge
    # call that failed counts as a failed prompt, as in training.
    prompt_passed: bool | None = None
    prompt_criteria: dict | None = None
    # judge_score gated by the prompt judge, mirroring how the trainer gates
    # reward: judge_score unchanged with no gate in effect, None only when
    # there is no judge_score to gate.
    final_score: int | None = None

    def key(self) -> str:
        return f"{self.behavior_id}:{self.prompt_index}:{self.generation_index}"


def gated_score(*, judge_score: int | None, prompt_passed: bool | None) -> int | None:
    """The response score after the prompt-judge gate, or None if ungradeable.

    Mirrors the prompt-judge gate of the reward formulas the trainer trains
    with, which is their only gate on the investigator prompt.
    """
    if judge_score is None:
        return None
    return 0 if prompt_passed is False else judge_score


def full_path(output_dir: str, run_id: str) -> Path:
    return Path(output_dir) / f"{run_id}_full.jsonl"


def full_parquet_path(output_dir: str, run_id: str) -> Path:
    return Path(output_dir) / f"{run_id}_full.parquet"


def summary_path(output_dir: str, run_id: str) -> Path:
    return Path(output_dir) / f"{run_id}_summary.json"


def prompts_path(output_dir: str, run_id: str) -> Path:
    """Investigator prompts, appended as sampled, so a resume reuses them."""
    return Path(output_dir) / f"{run_id}_prompts.jsonl"


def _key(r: dict) -> str:
    return f"{r['behavior_id']}:{r['prompt_index']}:{r['generation_index']}"


def _dedupe(rows: list[dict]) -> list[dict]:
    """One row per trial key, keeping the last (the freshest, on a resume)."""
    by_key = {_key(r): r for r in rows}
    return list(by_key.values())


def load_completed_keys(output_dir: str, run_id: str) -> set[str]:
    """Trial keys already present on disk, for resume."""
    return {_key(r) for r in load_rows(output_dir, run_id, deduped=False)}


def append_row(output_dir: str, run_id: str, row: TrialRow) -> None:
    path = full_path(output_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(row)) + "\n")


def load_rows(output_dir: str, run_id: str, *, deduped: bool = True) -> list[dict]:
    """Rows from the live jsonl if present, else the compacted parquet."""
    jsonl = full_path(output_dir, run_id)
    if jsonl.exists():
        rows = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        parquet = full_parquet_path(output_dir, run_id)
        rows = pl.read_parquet(parquet).to_dicts() if parquet.exists() else []
    return _dedupe(rows) if deduped else rows


def write_full_parquet(output_dir: str, run_id: str, rows: list[dict]) -> Path:
    """Write rows to the compacted parquet, replacing any existing one.

    Atomic: a crash mid-write leaves the previous parquet intact rather than a
    truncated file that no reader can open.
    """
    parquet = full_parquet_path(output_dir, run_id)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(".parquet.tmp")
    # Infer the schema from every row, not polars' default first 100: those
    # can all be target errors with null scores and criteria.
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(tmp, compression="zstd")
    tmp.replace(parquet)
    return parquet


def compact_full(output_dir: str, run_id: str) -> Path | None:
    """Convert full.jsonl to a zstd parquet and delete the jsonl.

    No-op (returns None) if the jsonl is already gone -- e.g. a run that was
    completed and compacted on an earlier invocation.
    """
    jsonl = full_path(output_dir, run_id)
    if not jsonl.exists():
        return None
    parquet = write_full_parquet(output_dir, run_id, load_rows(output_dir, run_id))
    jsonl.unlink()
    return parquet


def remove_run_outputs(output_dir: str, run_id: str) -> None:
    """Delete every on-disk artifact for a run (for --force-rerun)."""
    for path in (
        summary_path(output_dir, run_id),
        full_path(output_dir, run_id),
        full_parquet_path(output_dir, run_id),
        prompts_path(output_dir, run_id),
    ):
        path.unlink(missing_ok=True)


def summarize(rows: list[dict]) -> dict:
    """Per-behavior and run-level aggregates over a run's (deduped) rows."""
    by_behavior: dict[str, list[dict]] = {}
    for r in rows:
        by_behavior.setdefault(r["behavior_id"], []).append(r)

    behaviors = {}
    for behavior_id, rs in sorted(by_behavior.items()):
        scored = [r["judge_score"] for r in rs if r["judge_score"] is not None]
        gated = [r["final_score"] for r in rs if r.get("final_score") is not None]
        behaviors[behavior_id] = {
            "category": rs[0]["category"],
            "n_rows": len(rs),
            "n_errors": sum(1 for r in rs if r["error"]),
            "n_judged": len(scored),
            "mean_judge_score": sum(scored) / len(scored) if scored else None,
            "max_judge_score": max(scored) if scored else None,
            # Prompt-judge-gated view. With the prompt judge off, n_prompt_judged
            # is 0 and the final scores equal the judge scores.
            "n_prompt_judged": sum(1 for r in rs if r.get("prompt_passed") is not None),
            "n_prompt_failed": sum(1 for r in rs if r.get("prompt_passed") is False),
            "mean_final_score": sum(gated) / len(gated) if gated else None,
            "max_final_score": max(gated) if gated else None,
        }

    totals = {
        key: sum(b[key] for b in behaviors.values())
        for key in (
            "n_rows",
            "n_errors",
            "n_judged",
            "n_prompt_judged",
            "n_prompt_failed",
        )
    }
    return {"behaviors": behaviors, "totals": totals}


def write_summary(
    output_dir: str,
    run_id: str,
    config_dict: dict,
    started_at: str,
    finished_at: str,
    rows: list[dict],
) -> dict:
    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": config_dict,
        **summarize(rows),
    }
    save_summary(output_dir, run_id, summary)
    return summary


def save_summary(output_dir: str, run_id: str, summary: dict) -> Path:
    """Write a summary dict atomically."""
    path = summary_path(output_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(path)
    return path
