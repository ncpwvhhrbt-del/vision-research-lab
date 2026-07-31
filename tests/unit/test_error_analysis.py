import unittest

import torch

from src.evaluation.error_analysis import rank_confusions


class ErrorAnalysisTests(unittest.TestCase):
    def test_ranks_off_diagonal_errors_deterministically(self):
        matrix = torch.tensor([
            [8, 3, 1],
            [2, 7, 4],
            [0, 5, 9],
        ])

        result = rank_confusions(matrix, ("cat", "dog", "bird"), top_k=3)

        self.assertEqual(
            result,
            [
                {"true_class": "bird", "predicted_class": "dog", "count": 5},
                {"true_class": "dog", "predicted_class": "bird", "count": 4},
                {"true_class": "cat", "predicted_class": "dog", "count": 3},
            ],
        )

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            rank_confusions(torch.eye(2, dtype=torch.int64), ("a", "b"), top_k=0)


if __name__ == "__main__":
    unittest.main()
