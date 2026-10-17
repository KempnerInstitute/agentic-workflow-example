"""Folders, experiment settings, and the validation of round configurations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import yaml

EXPERIMENT_PATH = Path("experiment.yaml")
SCHEMA_PATH = Path("configs/round.schema.json")
SPLIT_PATH = Path("data/split.json")
BASELINE_MODEL = Path("results/baseline/model.pt")


class ConfigError(ValueError):
    """A round configuration that breaks the schema or the extra rule."""


@dataclass(frozen=True)
class Paths:
    """The folders of one job. Each job starts in the project folder."""

    project: Path
    run_root: Path
    data_dir: str
    state: Path

    @property
    def logs(self) -> Path:
        return self.project / "logs"


def paths_from_env() -> Paths:
    """Read the folders from the variables that cluster.env sets."""
    missing = [name for name in ("RUN_ROOT", "DATA_DIR") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"Set {' and '.join(missing)}: run 'source cluster.env' in the project folder.")
    project = Path(os.environ.get("PROJECT_DIR", Path.cwd())).resolve()
    return Paths(
        project=project,
        run_root=Path(os.environ["RUN_ROOT"]),
        data_dir=os.environ["DATA_DIR"],
        state=project / os.environ.get("STATE_DIR", "state"),
    )


def experiment_path(project: Path) -> Path:
    """Return the experiment file: EXPERIMENT_FILE if it is set, else experiment.yaml.

    To test the loop with small limits, set EXPERIMENT_FILE to an edited copy outside Git control.
    """
    return project / os.environ.get("EXPERIMENT_FILE", str(EXPERIMENT_PATH))


def load_experiment(project: Path) -> dict:
    with open(experiment_path(project)) as f:
        return yaml.safe_load(f)


def load_schema(project: Path) -> dict:
    with open(project / SCHEMA_PATH) as f:
        return json.load(f)


def validate_round_config(round_config: dict, schema: dict) -> dict:
    """Return the configuration if it is valid. Raise ConfigError if it is not valid."""
    try:
        jsonschema.validate(round_config, schema)
    except jsonschema.ValidationError as error:
        raise ConfigError(f"invalid configuration: {error.message}") from None
    if round_config["finetune_epochs"] < round_config["steps"]:
        raise ConfigError("invalid configuration: finetune_epochs must be at least steps")
    return round_config


def load_round_config(path: Path, schema: dict) -> dict:
    with open(path) as f:
        return validate_round_config(json.load(f), schema)
