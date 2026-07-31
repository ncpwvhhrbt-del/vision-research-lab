"""Tests for scheduler settings before scheduler construction is implemented."""

import unittest

from src.training.config import TrainingConfig


class SchedulerConfigTests(unittest.TestCase):
    def test_reads_step_scheduler(self) -> None:
        config = TrainingConfig.from_experiment_yaml(
            "configs/experiment/simple_cnn_step_lr.yaml"
        )
        self.assertEqual(config.scheduler_name, "step")
        self.assertEqual(config.scheduler_step_size, 10)

    def test_reads_cosine_scheduler(self) -> None:
        config = TrainingConfig.from_experiment_yaml(
            "configs/experiment/simple_cnn_cosine.yaml"
        )
        self.assertEqual(config.scheduler_name, "cosine")
        self.assertAlmostEqual(config.scheduler_min_lr, 0.0001)

    def test_rejects_unknown_scheduler(self) -> None:
        config = TrainingConfig(scheduler_name="unknown")
        with self.assertRaisesRegex(ValueError, "none, step, or cosine"):
            config.validate()
