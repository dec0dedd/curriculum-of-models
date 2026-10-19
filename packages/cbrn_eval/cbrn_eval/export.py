"""Export completed sweeps to CSV: one file per run, plus an index.

The schema rules live here so they can be unit tested independently of the
CLI around them (cli.export).
"""

import csv
from pathlib import Path

import polars as pl

from . import output

RESULT_COLUMNS = [
    "run_id",
    "behavior_id",
    "prompt_index",
    "generation_index",
    "judge_score",
    "final_score",
    "prompt_passed",
]

INDEX_COLUMNS = [
    "run_id",
    "target_model",
    "investigator_model",
    "prompt_source",
    "n_rows",
    "mean_judge_score",
    "mean_final_score",
    "started_at",
    "finished_at",
]


def local_run_ids(output_dir: Path) -> list[str]:
    """Completed runs on disk -- a summary.json is the completion marker."""
    return sorted(
        p.name.removesuffix("_summary.json") for p in output_dir.glob("*_summary.json")
    )


def export_results(run_id: str, rows: list[dict]) -> pl.DataFrame:
    """The results schema applied to a run's (already-deduped) rows."""
    df = pl.DataFrame(rows)
    return (
        df.with_columns(pl.lit(run_id).alias("run_id"))
        .select(
            pl.col("run_id"),
            pl.col("behavior_id"),
            pl.col("prompt_index").cast(pl.Int64),
            pl.col("generation_index").cast(pl.Int64),
            pl.col("judge_score").cast(pl.Int64),
            pl.col("final_score").cast(pl.Int64),
            pl.col("prompt_passed").cast(pl.Boolean),
        )
        .sort(["behavior_id", "prompt_index", "generation_index"])
    )


def export_run(output_dir: Path, run_id: str, export_dir: Path) -> pl.DataFrame:
    """Write <export_dir>/<run_id>.csv and return the frame it was built from."""
    rows = output.load_rows(str(output_dir), run_id)
    df = export_results(run_id, rows)
    export_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(export_dir / f"{run_id}.csv")
    return df


def _index_row(run_id: str, summary: dict, df: pl.DataFrame) -> dict:
    config = summary.get("config", {})
    means = df.select(
        pl.col("judge_score").mean().alias("mean_judge_score"),
        pl.col("final_score").mean().alias("mean_final_score"),
    ).row(0, named=True)
    return {
        "run_id": run_id,
        "target_model": config.get("target_model"),
        "investigator_model": config.get("investigator_model"),
        "prompt_source": config.get("prompt_source"),
        "n_rows": df.height,
        "mean_judge_score": means["mean_judge_score"],
        "mean_final_score": means["mean_final_score"],
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
    }


def write_index(export_dir: Path, rows: list[dict]) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / "index.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_all(output_dir: Path, export_dir: Path) -> int:
    """Export every completed run in ``output_dir``. Returns how many."""
    run_ids = local_run_ids(output_dir)
    index_rows = []
    for run_id in run_ids:
        df = export_run(output_dir, run_id, export_dir)
        summary = output.load_summary(str(output_dir), run_id)
        index_rows.append(_index_row(run_id, summary, df))
    if index_rows:
        write_index(export_dir, index_rows)
    return len(run_ids)
