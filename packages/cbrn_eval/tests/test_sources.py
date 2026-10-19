import json

import pytest

from cbrn_eval import sources
from cbrn_eval.config import EvalConfig
from cbrn_eval.sources import build_prompt_items, expected_prompt_total


@pytest.fixture
def behaviors_file(tmp_path):
    path = tmp_path / "behaviors.jsonl"
    rows = [
        {"behavior_id": "a", "category": "cat1", "behavior": "Do A."},
        {"behavior_id": "b", "category": "cat2", "behavior": "Do B."},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_behaviors_source_is_one_item_per_behavior(behaviors_file):
    config = EvalConfig(prompt_source="behaviors", behaviors_file=behaviors_file)
    assert expected_prompt_total(config) == 2


async def test_behaviors_source_sends_the_rubric_verbatim(behaviors_file):
    config = EvalConfig(prompt_source="behaviors", behaviors_file=behaviors_file)
    items = await build_prompt_items(config)
    assert sorted((i.behavior_id, i.prompt) for i in items) == [
        ("a", "Do A."),
        ("b", "Do B."),
    ]
    assert all(i.prompt_index == 0 for i in items)


def test_investigator_source_total_scales_with_generations(behaviors_file):
    config = EvalConfig(
        prompt_source="investigator",
        behaviors_file=behaviors_file,
        n_investigator_generations=5,
    )
    assert expected_prompt_total(config) == 10


def test_unknown_prompt_source_is_rejected(behaviors_file):
    config = EvalConfig(prompt_source="nonsense", behaviors_file=behaviors_file)
    with pytest.raises(ValueError, match="Unknown prompt_source"):
        expected_prompt_total(config)


class _FakeBackend:
    def __init__(self):
        self.calls: list[str] = []

    async def generate(self, behavior: str) -> str:
        self.calls.append(behavior)
        return f"prompt for {behavior} #{len(self.calls)}"

    @classmethod
    async def create(cls, config):
        return cls()


async def test_investigator_source_generates_n_per_behavior(
    monkeypatch, behaviors_file
):
    monkeypatch.setattr(sources, "TinkerInvestigatorBackend", _FakeBackend)
    config = EvalConfig(
        prompt_source="investigator",
        behaviors_file=behaviors_file,
        n_investigator_generations=3,
        investigator_concurrency=2,
    )
    items = await build_prompt_items(config)
    assert len(items) == 6
    by_behavior = {}
    for item in items:
        by_behavior.setdefault(item.behavior_id, []).append(item.prompt_index)
    assert by_behavior == {"a": [0, 1, 2], "b": [0, 1, 2]}
