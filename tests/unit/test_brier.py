import unittest

import torch

from src.evaluation.brier import multiclass_brier_score


class BrierScoreTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_score(self):
        probabilities = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        targets = torch.tensor([0, 1])
        self.assertAlmostEqual(multiclass_brier_score(probabilities, targets), 0.0)

    def test_uniform_three_class_prediction(self):
        probabilities = torch.full((1, 3), 1 / 3)
        targets = torch.tensor([2])
        expected = (1 / 3) ** 2 + (1 / 3) ** 2 + (2 / 3) ** 2
        self.assertAlmostEqual(
            multiclass_brier_score(probabilities, targets),
            expected,
        )

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            multiclass_brier_score(torch.ones(2, 3) / 3, torch.tensor([0]))

    def test_rejects_probability_out_of_range(self):
        with self.assertRaises(ValueError):
            multiclass_brier_score(torch.tensor([[1.2, -0.2]]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
