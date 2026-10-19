"""What a run writes to its log_dir: run_metadata.json and rollouts.jsonl.

rollouts.jsonl holds one line per rollout. It is the only record of a run's
training curve: reward and curriculum_scale per step give the learning and
ladder curves, investigator_refused the refusal rate over training.
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

    ``config_files`` are the method/target TOML files that produced ``config``,
    in the order they were applied (see ``train.parse_args``). The resolved
    config is already in ``config``, so this is for the question the resolved
    values alone cannot answer: which files on disk, as of this run, are
    responsible for it. ``command`` records the same paths as typed, which may
    differ (relative, or resolved against configs/) from what is listed here.
    """
    path = Path(log_dir) / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": chz.asdict(config),
        "config_files": [str(Path(p).resolve()) for p in config_files],
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "is_dirty": bool(_git("status", "--porcelain")),
        },
        "versions": {
            "python": sys.version,
            **{
                p: _version(p)
                for p in ("curriculum-of-models", "tinker", "tinker-cookbook")
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
        reward: float,
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
            # The reward the policy was trained on; formula_reward is before
            # the env's truncation gate.
            "reward": reward,
            "formula_reward": result.reward if result else None,
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
