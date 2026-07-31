"""Named CIFAR-10 training augmentation policies."""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def build_train_augmentation(
    *,
    name: str,
    image_size: int,
    random_crop_padding: int,
    horizontal_flip_probability: float,
) -> list[Any]:
    """Build only the stochastic image operations used for training.

    Responsibility:
        Select a named policy while leaving tensor conversion and normalization
        to ``build_transforms``.
    Policies:
        ``none``: no stochastic augmentation.
        ``crop``: random crop only.
        ``crop_flip``: random crop followed by horizontal flip.
    Output:
        A list passed before ``ToTensor`` in ``transforms.Compose``.
    """
    if name=="none":
        return []
    crop=transforms.RandomCrop(image_size,padding=random_crop_padding)
    if name=="crop":
        return [crop]
    if name=="crop_flip":
        return[crop,
               transforms.RandomHorizontalFlip(horizontal_flip_probability),]

    raise ValueError("none, crop, or crop_flip")
