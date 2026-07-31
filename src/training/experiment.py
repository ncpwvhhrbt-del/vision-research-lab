"""Top-level experiment metadata and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import TrainingConfig


@dataclass(frozen=True)
class ExperimentSpec:
    """Coordinates existing data, model, and training configurations."""

    name: str
    seed: int
    data_config: Path
    model_config: Path
    output_dir: Path
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as file:
            raw: Mapping[str, Any] = yaml.safe_load(file)

        # Expected location: <project>/configs/experiment/<name>.yaml
        project_root = config_path.parents[2]

        def project_path(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else project_root / candidate

        spec = cls(
            name=str(raw.get("experiment", "")).strip(),
            seed=int(raw.get("seed", 42)),
            data_config=project_path(str(raw["data_config"])),
            model_config=project_path(str(raw["model_config"])),
            output_dir=project_path(str(raw["output_dir"])),
            training=TrainingConfig.from_experiment_yaml(config_path),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.name:
            raise ValueError("experiment name cannot be empty.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if not self.data_config.is_file():
            raise FileNotFoundError(f"Data config not found: {self.data_config}")
        if not self.model_config.is_file():
            raise FileNotFoundError(f"Model config not found: {self.model_config}")
        self.training.validate()
