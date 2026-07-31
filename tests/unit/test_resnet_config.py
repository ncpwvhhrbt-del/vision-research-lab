"""Unit tests for ResNet-18 configuration."""

import unittest

from src.models.resnet import ResNet18Config


class ResNet18ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        ResNet18Config().validate()

    def test_first_baseline_rejects_pretrained_weights(self) -> None:
        config = ResNet18Config(weights="imagenet")
        with self.assertRaisesRegex(ValueError, "from scratch"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
