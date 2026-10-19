"""Run configurations, as YAML files.

A config is a flat mapping of ``TrainConfig`` field names:

    reward_type: curriculum
    kl_penalty_coef: 0.5

A run names one method and one target, each defined in exactly one file:

    com-train methods/com.yaml targets/gemma.yaml

They apply in the order given, the later one winning, and every
``field=value`` on the command line beats all of them. A third file overrides
both, which is where a local ``vllm_base_url`` or ``wandb_project`` belongs
rather than in the files that hold the paper's values.

Merging is per top-level key, so a config that sets ``chat_template_kwargs``
replaces that whole mapping rather than merging into it.

Each file is checked against the real field names as it is read, which is the
point of keeping the paper's values here: a typo'd key is an error naming the
file that holds it, not a value silently left at its default.

YAML has a null (``~`` or the bare ``key:``), unlike TOML, so a config can
reset a field to None as well as omit it.
"""

from collections.abc import Collection
from pathlib import Path
from typing import Any

import yaml


class ConfigFileError(Exception):
    """A config is missing, malformed, or names a field that does not exist."""


def load_config(
    path: str | Path, fields: Collection[str] | None = None
) -> dict[str, Any]:
    """Read one config into a dict of field values.

    ``fields`` are the field names it may set; None skips that check.
    """
    path = Path(path).expanduser()
    try:
        table = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        raise ConfigFileError(
            f"cannot read config file {path}: {exc.strerror}"
        ) from None
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"{path} is not valid YAML: {exc}") from None

    if not isinstance(table, dict):
        raise ConfigFileError(
            f"{path} must be a mapping of field: value, got {type(table).__name__}"
        )
    if fields is not None:
        unknown = sorted(set(table) - set(fields))
        if unknown:
            raise ConfigFileError(
                f"{path} sets unknown field(s): {', '.join(unknown)}.\n"
                f"Known fields: {', '.join(sorted(fields))}"
            )
    return table


def available(configs_dir: str | Path) -> list[str]:
    """The config names in one directory, e.g. the methods or the targets."""
    directory = Path(configs_dir)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))
