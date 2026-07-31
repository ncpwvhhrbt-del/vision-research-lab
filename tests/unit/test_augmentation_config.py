"""Ensure each augmentation experiment points to its intended data policy."""

import unittest

from src.data.cifar import CIFARDataConfig


class AugmentationConfigTests(unittest.TestCase):
    def test_named_data_configs_load_expected_policies(self) -> None:
        expected = {
            "configs/data/cifar10_no_augmentation.yaml": "none",
            "configs/data/cifar10_crop_only.yaml": "crop",
            "configs/data/cifar10.yaml": "crop_flip",
        }
        for path, policy in expected.items():
            with self.subTest(path=path):
                self.assertEqual(CIFARDataConfig.from_yaml(path).augmentation_name, policy)
