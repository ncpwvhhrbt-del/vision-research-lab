"""Tests for multi-seed summary math without launching training processes."""

import unittest

from scripts.sweep_schedulers import aggregate


class SweepAggregationTests(unittest.TestCase):
    def test_mean_std_and_extremes(self) -> None:
        records = [
            {"scheduler": "none", "seed": 1, "test_accuracy": 0.7, "samples": 3, "output_dir": "a"},
            {"scheduler": "none", "seed": 2, "test_accuracy": 0.9, "samples": 3, "output_dir": "b"},
            {"scheduler": "step", "seed": 1, "test_accuracy": 0.8, "samples": 3, "output_dir": "c"},
        ]

        summary = aggregate(records)
        fixed = next(item for item in summary if item["scheduler"] == "none")

        self.assertAlmostEqual(fixed["mean_test_accuracy"], 0.8)
        self.assertAlmostEqual(fixed["std_test_accuracy"], 0.1)
        self.assertEqual(fixed["best_seed"], 2)
        self.assertEqual(fixed["worst_seed"], 1)


if __name__ == "__main__":
    unittest.main()
