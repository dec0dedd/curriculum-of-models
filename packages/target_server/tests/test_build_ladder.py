"""Building the ladder: the names, the scaling, and the symlinked weights."""

import json

import pytest
from click.testing import CliRunner
from common.adapters import adapter_name_for_scale

from target_server.build_ladder import build_ladder, main, parse_scales

# A trained adapter is identified by r and lora_alpha; vLLM applies
# lora_alpha / r to the update, so that ratio is the strength to rescale.
SOURCE_CONFIG = {
    "r": 1,
    "lora_alpha": 2,
    "peft_type": "LORA",
    "target_modules": ["o_proj"],
}


@pytest.fixture
def source(tmp_path):
    d = tmp_path / "trained_adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(SOURCE_CONFIG))
    (d / "adapter_model.safetensors").write_bytes(b"fake weights")
    return d


def _strength(path):
    config = json.loads((path / "adapter_config.json").read_text())
    return config["lora_alpha"] / config["r"]


def test_full_strength_reproduces_the_adapter_as_trained(source, tmp_path):
    ((_, _, path),) = build_ladder(source, tmp_path / "ladder", [1.0])
    assert _strength(path) == pytest.approx(
        SOURCE_CONFIG["lora_alpha"] / SOURCE_CONFIG["r"]
    )


def test_zero_strength_reproduces_the_base_model(source, tmp_path):
    ((_, _, path),) = build_ladder(source, tmp_path / "ladder", [0.0])
    assert _strength(path) == 0.0


def test_strength_is_proportional_to_the_scale(source, tmp_path):
    built = build_ladder(source, tmp_path / "ladder", [1.0, 0.5, 0.25])
    full, half, quarter = (_strength(p) for _, _, p in built)
    assert half == pytest.approx(full / 2)
    assert quarter == pytest.approx(full / 4)


def test_directories_are_named_by_the_shared_contract(source, tmp_path):
    scales = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    built = build_ladder(source, tmp_path / "ladder", scales)
    # The trainer asks vLLM for exactly these names.
    assert [name for _, name, _ in built] == [adapter_name_for_scale(s) for s in scales]
    assert [p.name for _, _, p in built] == [adapter_name_for_scale(s) for s in scales]


def test_weights_are_symlinked_not_copied(source, tmp_path):
    built = build_ladder(source, tmp_path / "ladder", [1.0, 0.5])
    for _, _, path in built:
        weights = path / "adapter_model.safetensors"
        assert weights.is_symlink()
        assert weights.resolve() == (source / "adapter_model.safetensors").resolve()
        assert weights.read_bytes() == b"fake weights"


def test_copy_weights_writes_real_files(source, tmp_path):
    ((_, _, path),) = build_ladder(
        source, tmp_path / "ladder", [1.0], copy_weights=True
    )
    weights = path / "adapter_model.safetensors"
    assert not weights.is_symlink()
    assert weights.read_bytes() == b"fake weights"


def test_rebuilding_over_an_existing_ladder_is_safe(source, tmp_path):
    out = tmp_path / "ladder"
    build_ladder(source, out, [1.0, 0.5])
    build_ladder(source, out, [1.0, 0.5])  # a serve script may re-run this
    assert _strength(out / "alpha_1_0") == pytest.approx(2.0)


def test_the_cli_builds_the_requested_scales(source, tmp_path):
    out = tmp_path / "ladder"
    result = CliRunner().invoke(
        main,
        [
            "--source-repo",
            str(source),
            "--output-dir",
            str(out),
            "--scales",
            "1.0,0.5,0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == [
        "alpha_0_0",
        "alpha_0_5",
        "alpha_1_0",
    ]


def test_the_default_grid_covers_every_fifth(source):
    scales = parse_scales(None)
    assert len(scales) == 21
    assert scales[0] == 0.0 and scales[-1] == 1.0
    # Every scale the paper's ladders use must be on the grid.
    assert all(s in scales for s in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])


@pytest.mark.parametrize("raw", ["1.5", "-0.2", "abc", "1.0,2.0"])
def test_scales_outside_the_unit_range_are_rejected(raw):
    with pytest.raises(Exception):
        parse_scales(raw)


def test_a_source_without_weights_fails_clearly(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(SOURCE_CONFIG))
    with pytest.raises(Exception, match="adapter_model.safetensors"):
        build_ladder(d, tmp_path / "ladder", [1.0])
