"""Behavioral specification for named augmentation policies."""

import unittest

from src.data.augmentations import build_train_augmentation


class AugmentationPolicyTests(unittest.TestCase):
    def test_none_has_no_stochastic_operations(self) -> None:
        self.assertEqual(
            build_train_augmentation(
                name="none",
                image_size=32,
                random_crop_padding=4,
                horizontal_flip_probability=0.5,
            ),
            [],
        )

    def test_crop_flip_has_crop_then_flip(self) -> None:
        operations = build_train_augmentation(
            name="crop_flip",
            image_size=32,
            random_crop_padding=4,
            horizontal_flip_probability=0.5,
        )
        self.assertEqual(
            [operation.__class__.__name__ for operation in operations],
            ["RandomCrop", "RandomHorizontalFlip"],
        )

    def test_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "none, crop, or crop_flip"):
            build_train_augmentation(
                name="unknown",
                image_size=32,
                random_crop_padding=4,
                horizontal_flip_probability=0.5,
            )
