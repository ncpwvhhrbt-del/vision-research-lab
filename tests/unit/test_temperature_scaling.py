import unittest

import torch

from src.evaluation.temperature_scaling import scale_logits


class TemperatureScalingTests(unittest.TestCase):
    def test_scales_logits(self):
        logits = torch.tensor([[2.0, 0.0]])
        self.assertTrue(torch.equal(scale_logits(logits, 2.0), torch.tensor([[1.0, 0.0]])))

    def test_rejects_nonpositive_temperature(self):
        with self.assertRaises(ValueError):
            scale_logits(torch.ones(1, 2), 0.0)

    def test_rejects_wrong_logit_shape(self):
        with self.assertRaises(ValueError):
            scale_logits(torch.ones(2), 1.0)


if __name__ == "__main__":
    unittest.main()
