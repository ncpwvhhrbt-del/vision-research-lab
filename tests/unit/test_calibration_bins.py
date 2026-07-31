import unittest

import torch

from src.evaluation.calibration import calibration_bin_stats


class CalibrationBinTests(unittest.TestCase):
    def test_returns_accuracy_and_confidence_per_nonempty_bin(self):
        confidences = torch.tensor([0.25, 0.25, 0.75, 0.75])
        correctness = torch.tensor([False, False, True, True])

        result = calibration_bin_stats(confidences, correctness, num_bins=2)

        self.assertEqual(
            result,
            [
                {"bin": 0, "count": 2, "confidence": 0.25, "accuracy": 0.0},
                {"bin": 1, "count": 2, "confidence": 0.75, "accuracy": 1.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
