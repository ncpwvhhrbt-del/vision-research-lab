import unittest

import torch

from src.evaluation.calibration import expected_calibration_error


class CalibrationTests(unittest.TestCase):
    def test_perfectly_calibrated_two_bins(self):
        confidences = torch.tensor([
        0.25, 0.25, 0.25, 0.25,
        0.75, 0.75, 0.75, 0.75,
        ])
        correctness = torch.tensor([
        False, False, False, True,
        True, True, True, False,
        ])
        self.assertAlmostEqual(
            expected_calibration_error(confidences, correctness, num_bins=2),
            0.0,
        )

    def test_detects_overconfidence(self):
        confidences = torch.tensor([0.9, 0.9])
        correctness = torch.tensor([True, False])
        self.assertAlmostEqual(
            expected_calibration_error(confidences, correctness, num_bins=10),
            0.4,
        )

    def test_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            expected_calibration_error(torch.tensor([0.5]), torch.tensor([True, False]))


if __name__ == "__main__":
    unittest.main()
