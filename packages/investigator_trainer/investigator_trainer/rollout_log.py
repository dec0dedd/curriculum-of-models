"""What a run writes to its log_dir: run_metadata.json and rollouts.jsonl.

rollouts.jsonl holds one line per rollout: the prompt, the target responses,
every judge verdict and the reward. tinker writes its own per-step averages to
metrics.jsonl beside it; rollouts.jsonl is the per-rollout record behind them,
so reward and curriculum_scale per step give the learning and ladder curves,
and investigator_refused the refusal rate over training.
"""

import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import chz

from investigator_trainer.rewards.base import RewardResult


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def save_run_metadata(
    log_dir: str | Path,
    config: object,
    config_files: Sequence[str | Path] = (),
) -> None:
    """The full config, code version and command line, for tracing a run back.

    ``config_files`` are the method/target YAML files that produced ``config``,
    in the order they were applied.
    """
    path = Path(log_dir) / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": chz.asdict(config),
        "config_files": [str(Path(p).resolve()) for p in config_files],
        "git_commit": _git("rev-parse", "HEAD"),
        "versions": {
            "python": sys.version,
            **{
                p: _version(p)
                for p in ("investigator-trainer", "common", "tinker", "tinker-cookbook")
            },
        },
        "command": " ".join(sys.argv),
    }
    path.write_text(json.dumps(metadata, indent=2, default=str))


class RolloutLog:
    def __init__(self, log_dir: str | Path | None, reasoning_max_chars: int = 5000):
        self.path = Path(log_dir) / "rollouts.jsonl" if log_dir else None
        self.reasoning_max_chars = reasoning_max_chars
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        step: int,
        env_id: int,
        behavior_id: str,
        investigator_prompt: str,
        result: RewardResult | None,
        investigator_refused: bool,
        investigator_clipped: bool,
        error: str | None = None,
    ) -> None:
        if self.path is None:
            return
        cut = self.reasoning_max_chars
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "env_id": env_id,
            "behavior_id": behavior_id,
            "investigator_prompt": investigator_prompt,
            # Null for a failed rollout: it is never trained on, but retried
            # or its group dropped; `error` says why.
            "reward": result.reward if result else None,
            "investigator_refused": investigator_refused,
            "investigator_clipped": investigator_clipped,
            "error": error,
        }
        if result is not None:
            record |= {
                "target_responses": result.responses,
                "prompt_criteria": result.prompt_criteria,
                "response_criteria": result.response_criteria,
                "prompt_reasoning": result.prompt_reasoning[:cut],
                "response_reasonings": [r[:cut] for r in result.response_reasonings],
                "metrics": result.metrics,
                **result.log_extra,
            }
        # One process, one event loop: appends never interleave.
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
