"""Smoke-test both selectable architectures with CIFAR tensor shapes."""

import unittest

import torch

from src.models.factory import build_model_from_yaml


class ModelForwardSmokeTests(unittest.TestCase):
    def test_all_model_configs_return_ten_logits(self) -> None:
        images = torch.randn(2, 3, 32, 32)
        for config_path in (
            "configs/model/simple_cnn.yaml",
            "configs/model/resnet18.yaml",
        ):
            with self.subTest(config=config_path):
                model = build_model_from_yaml(config_path)
                self.assertEqual(tuple(model(images).shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
