import unittest

import torch

from src.evaluation.robustness import add_gaussian_noise


class RobustnessTests(unittest.TestCase):
    def test_zero_noise_preserves_images(self):
        images = torch.ones(2, 3, 4, 4)
        perturbed = add_gaussian_noise(images, 0.0)
        self.assertTrue(torch.equal(images, perturbed))

    def test_rejects_negative_noise(self):
        with self.assertRaises(ValueError):
            add_gaussian_noise(torch.zeros(1, 3, 4, 4), -0.1)


if __name__ == "__main__":
    unittest.main()
