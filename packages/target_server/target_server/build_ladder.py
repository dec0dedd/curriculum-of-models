"""Build the refusal-free adapter ladder that the target is served with.

One trained adapter becomes many, each a copy of the same weights with
``lora_alpha`` rescaled, so stage k of the curriculum is the same direction in
weight space applied at strength k. The weights are symlinked rather than
copied, so the ladder does not hold a copy of identical tensors per strength.

    build-ladder                       # the 0.05 grid, 0.00 to 1.00
    build-ladder --scales 1.0,0.8,0.6,0.4,0.2,0.0   # just the paper's CoM ladder
    build-ladder --source-repo <hf repo or local dir> --output-dir <dir>

The serve scripts pass each directory to vLLM under the name
``common.adapters.adapter_name_for_scale`` gives it, which is the same
function the trainer uses to ask for a stage.
"""

import json
import shutil
from pathlib import Path

import click
from common.adapters import adapter_name_for_scale
from huggingface_hub import snapshot_download

# The adapters the paper's targets were served with.
GEMMA_ADAPTER_REPO = "noahrossi/gemma-4-31b-it-refusal-free-v2"
GPT_OSS_ADAPTER_REPO = "noahrossi/gpt-oss-120b-refusal-r1-vllm"

# The full grid. A ladder only needs the stages its run uses, but the grid
# costs nothing to write and lets any ladder be served without rebuilding.
DEFAULT_SCALES = [i / 20 for i in range(21)]

WEIGHTS_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"


def resolve_source(repo_or_path: str) -> Path:
    """A local adapter directory, or one downloaded from the Hub."""
    path = Path(repo_or_path)
    if path.is_dir():
        return path
    click.echo(f"Downloading {repo_or_path} ...")
    return Path(
        snapshot_download(repo_or_path, allow_patterns=[f"*{WEIGHTS_FILE}", "*.json"])
    )


def build_ladder(
    source: Path, output_dir: Path, scales: list[float], *, copy_weights: bool = False
) -> list[tuple[float, str, Path]]:
    """Write one adapter directory per scale. Returns (scale, name, path) each."""
    config = json.loads((source / CONFIG_FILE).read_text())
    weights = (source / WEIGHTS_FILE).resolve()
    if not weights.exists():
        raise click.ClickException(f"No {WEIGHTS_FILE} in {source}")

    # lora_alpha / r is the scaling vLLM applies to B @ A, so rescaling
    # lora_alpha scales the whole update. Expressed relative to the source's
    # own strength, so scale 1.0 reproduces the adapter as trained.
    source_strength = config["lora_alpha"] / config["r"]
    built = []
    for scale in scales:
        name = adapter_name_for_scale(scale)
        out = output_dir / name
        out.mkdir(parents=True, exist_ok=True)

        target = out / WEIGHTS_FILE
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy_weights:
            shutil.copy2(weights, target)
        else:
            target.symlink_to(weights)

        (out / CONFIG_FILE).write_text(
            json.dumps(
                {**config, "lora_alpha": scale * source_strength * config["r"]},
                indent=2,
            )
        )
        built.append((scale, name, out))
    return built


def parse_scales(raw: str | None) -> list[float]:
    if not raw:
        return DEFAULT_SCALES
    try:
        scales = [float(s) for s in raw.replace(" ", "").split(",") if s]
    except ValueError as e:
        raise click.BadParameter(
            f"--scales takes a comma-separated list of numbers: {e}"
        ) from e
    for s in scales:
        if not 0.0 <= s <= 1.0:
            raise click.BadParameter(f"every scale must be in [0, 1]; got {s}")
    return scales


@click.command()
@click.option(
    "--source-repo",
    default=GEMMA_ADAPTER_REPO,
    show_default=True,
    help="HF repo or local directory of the trained adapter to scale.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("scaled_target_loras"),
    show_default=True,
    help="Where to write the ladder. The serve scripts read this.",
)
@click.option(
    "--scales",
    "raw_scales",
    help="Comma-separated, e.g. 1.0,0.8,0.6. Default: the 0.05 grid.",
)
@click.option(
    "--copy-weights",
    is_flag=True,
    help="Copy the weights into each directory instead of symlinking. Needed "
    "only where symlinks do not survive, e.g. some container mounts.",
)
def main(
    source_repo: str, output_dir: Path, raw_scales: str | None, copy_weights: bool
) -> None:
    """Scale one trained adapter into the ladder the curriculum climbs."""
    scales = parse_scales(raw_scales)
    source = resolve_source(source_repo)
    built = build_ladder(source, output_dir, scales, copy_weights=copy_weights)

    click.echo(f"\n{len(built)} adapter(s) in {output_dir}:")
    for scale, name, _ in built:
        click.echo(f"  {name:14} strength {scale:.2f}")
    click.echo(
        "\nScale 1.00 is the adapter as trained (fully refusal-free) and 0.00 "
        "reproduces the base model.\nServe with scripts/serve_gemma.sh or "
        "scripts/serve_gpt_oss.sh, which register these names with vLLM."
    )


if __name__ == "__main__":
    main()
