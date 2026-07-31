"""Unit tests for model selection."""

import unittest

from src.models.factory import read_model_name


class ModelFactoryTests(unittest.TestCase):
    def test_reads_simple_cnn_selector(self) -> None:
        self.assertEqual(
            read_model_name("configs/model/simple_cnn.yaml"),
            "simple_cnn",
        )

    def test_reads_resnet18_selector(self) -> None:
        self.assertEqual(
            read_model_name("configs/model/resnet18.yaml"),
            "resnet18",
        )


if __name__ == "__main__":
    unittest.main()
