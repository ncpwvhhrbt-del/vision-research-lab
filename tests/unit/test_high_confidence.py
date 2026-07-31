import unittest

import torch

from src.evaluation.high_confidence import rank_high_confidence_errors


class HighConfidenceErrorTests(unittest.TestCase):
    def test_returns_wrong_predictions_by_confidence(self):
        logits = torch.tensor([
            [0.0, 4.0, 0.0],  # true 0, confidently predicts 1
            [3.0, 0.0, 0.0],  # true 0, correct
            [0.0, 0.0, 2.0],  # true 1, less confident wrong prediction 2
        ])
        targets = torch.tensor([0, 0, 1])

        result = rank_high_confidence_errors(
            logits,
            targets,
            ("cat", "dog", "bird"),
            top_k=2,
        )

        self.assertEqual(result[0]["index"], 0)
        self.assertEqual(result[0]["true_class"], "cat")
        self.assertEqual(result[0]["predicted_class"], "dog")
        self.assertGreater(result[0]["confidence"], result[1]["confidence"])

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            rank_high_confidence_errors(
                torch.ones(1, 2), torch.tensor([0]), ("a", "b"), top_k=0
            )


if __name__ == "__main__":
    unittest.main()
