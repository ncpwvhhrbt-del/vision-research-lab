"""Tests for training configuration."""

import unittest

from src.training.config import TrainingConfig


class TrainingConfigTests(unittest.TestCase):
    def test_loads_baseline_experiment(self) -> None:
        config = TrainingConfig.from_experiment_yaml(
            "configs/experiment/baseline_simple_cnn.yaml"
        )
        self.assertEqual(config.optimizer_name, "sgd")
        self.assertAlmostEqual(config.learning_rate, 0.1)

    def test_rejects_nonpositive_learning_rate(self) -> None:
        config = TrainingConfig(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
