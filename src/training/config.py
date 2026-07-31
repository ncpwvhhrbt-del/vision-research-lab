"""Validated training settings loaded from an experiment YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    device: str = "auto"
    optimizer_name: str = "sgd"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler_name: str = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_min_lr: float = 0.0

    @classmethod
    def from_experiment_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw: Mapping[str, Any] = yaml.safe_load(file)

        training = raw.get("training", {})
        optimizer = training.get("optimizer", {})
        scheduler = training.get("scheduler", {})
        config = cls(
            epochs=int(training.get("epochs", 30)),
            device=str(training.get("device", "auto")).lower(),
            optimizer_name=str(optimizer.get("name", "sgd")).lower(),
            learning_rate=float(optimizer.get("learning_rate", 0.1)),
            momentum=float(optimizer.get("momentum", 0.9)),
            weight_decay=float(optimizer.get("weight_decay", 5e-4)),
            scheduler_name=str(scheduler.get("name", "none")).lower(),
            scheduler_step_size=int(scheduler.get("step_size", 10)),
            scheduler_gamma=float(scheduler.get("gamma", 0.1)),
            scheduler_min_lr=float(scheduler.get("min_lr", 0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if self.optimizer_name not in {"sgd", "adamw"}:
            raise ValueError("optimizer name must be sgd or adamw.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.scheduler_name not in {"none", "step", "cosine"}:
            raise ValueError("scheduler name must be none, step, or cosine.")
        if self.scheduler_step_size <= 0:
            raise ValueError("scheduler step_size must be positive.")
        if not 0.0 < self.scheduler_gamma <= 1.0:
            raise ValueError("scheduler gamma must be in (0, 1].")
        if self.scheduler_min_lr < 0:
            raise ValueError("scheduler min_lr cannot be negative.")
