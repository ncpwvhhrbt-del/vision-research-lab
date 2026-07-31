"""Fast structural tests for the SimpleCNN baseline."""

import unittest

from src.models.simple_cnn import (
    SimpleCNNConfig,
    build_simple_cnn,
    count_trainable_parameters,
)


class SimpleCNNTests(unittest.TestCase):
    def test_builds_with_positive_trainable_parameter_count(self) -> None:
        model = build_simple_cnn(SimpleCNNConfig())
        self.assertGreater(count_trainable_parameters(model), 0)

    def test_rejects_wrong_number_of_channel_stages(self) -> None:
        config = SimpleCNNConfig(channels=(32, 64))
        with self.assertRaisesRegex(ValueError, "three positive"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
