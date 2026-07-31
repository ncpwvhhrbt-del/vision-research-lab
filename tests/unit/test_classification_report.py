"""Behavioral specification for confusion-matrix reporting."""

import unittest

import torch

from src.evaluation.classification import (
    report_from_confusion_matrix,
    update_confusion_matrix,
)


class ClassificationReportTests(unittest.TestCase):
    def test_vectorized_confusion_matrix_update(self) -> None:
        matrix = torch.zeros((2, 2), dtype=torch.int64)
        logits = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [0.0, 2.0]],
        )
        targets = torch.tensor([0, 0, 1])

        update_confusion_matrix(matrix, logits, targets)

        expected = torch.tensor([[1, 1], [0, 1]])
        self.assertTrue(torch.equal(matrix, expected))

    def test_report_computes_overall_and_per_class_accuracy(self) -> None:
        matrix = torch.tensor([[2, 1], [1, 1]])

        report = report_from_confusion_matrix(matrix, ("zero", "one"))

        self.assertAlmostEqual(report.overall_accuracy, 3 / 5)
        self.assertAlmostEqual(report.per_class_accuracy[0], 2 / 3)
        self.assertAlmostEqual(report.per_class_accuracy[1], 1 / 2)
        self.assertEqual(report.samples, 5)


if __name__ == "__main__":
    unittest.main()
