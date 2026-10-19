"""Eval configuration and its content-addressed run id."""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from common.prompts import (
    INVESTIGATOR_SYSTEM_PROMPT,
    PROMPT_JUDGE_SYSTEM_PROMPT,
    RESPONSE_JUDGE_SYSTEM_PROMPT,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _not_hashed(default):
    """A field that does not affect results, so it stays out of the run id."""
    return field(default=default, metadata={"hashed": False})


@dataclass(frozen=True)
class EvalConfig:
    # "investigator": sample prompts from investigator_model.
    # "behaviors": send each behavior's text verbatim (the direct-ask baseline).
    prompt_source: str = "investigator"
    behaviors_file: str = str(REPO_ROOT / "data/behaviors/transluce_cbrn.jsonl")
    # A subset of behaviors_file by id. None = every row.
    behavior_ids: tuple[str, ...] | None = None
    n_investigator_generations: int = 256

    # Investigator: a tinker:// sampler_weights path.
    investigator_model: str = ""
    investigator_base_model: str = "Qwen/Qwen3.5-9B"
    investigator_renderer: str = "qwen3_5_disable_thinking"
    investigator_max_tokens: int = 512
    investigator_temperature: float = 1.0

    # Target, served by vLLM. target_model is the served name: the base model
    # id, or an adapter such as alpha_1_0 for the refusal-weakened ceiling.
    target_base_url: str = "http://localhost:8000/v1"
    target_model: str = "google/gemma-4-31B-it"
    target_max_tokens: int = 7000
    target_temperature: float = 1.0
    target_disable_reasoning: bool = True
    n_target_generations: int = 1

    # Send an investigator prompt that looks like a refusal to the target
    # anyway (False), or record it as a 0 without calls (True). The paper's
    # later evals used False, earlier ones True; the prompt judge fails nearly
    # every refusal, so both give the same final scores.
    skip_investigator_refusals: bool = False
    # Gate the score on the prompt judge, as training does.
    use_prompt_judge: bool = True

    judge_model: str = "openai/gpt-oss-120b"
    judge_max_tokens: int = 10000

    investigator_concurrency: int = _not_hashed(128)
    target_concurrency: int = _not_hashed(64)
    judge_concurrency: int = _not_hashed(150)
    output_dir: str = _not_hashed(str(REPO_ROOT / "eval_results"))

    def hash_payload(self) -> dict:
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("hashed", True)
        }
        payload["behaviors_file_sha"] = _sha(Path(self.behaviors_file).read_bytes())
        payload["prompt_shas"] = [
            _sha(p.encode())
            for p in (
                INVESTIGATOR_SYSTEM_PROMPT,
                RESPONSE_JUDGE_SYSTEM_PROMPT,
                PROMPT_JUDGE_SYSTEM_PROMPT,
            )
        ]
        return payload

    def run_id(self) -> str:
        """A slug of the target plus a hash of every result-affecting input.

        Includes the contents of the behaviors file and the prompts, so editing
        any of them gives a new id instead of resuming a stale run.
        """
        digest = _sha(json.dumps(self.hash_payload(), sort_keys=True).encode())[:10]
        slug = "".join(
            c if c.isalnum() or c in "._-" else "-" for c in self.target_model
        )
        return f"{slug}_{digest}"

    def to_dict(self) -> dict:
        return asdict(self)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
