"""Unit tests for edge runtime utilities."""

import unittest

from src.utils.runtime import resolve_device


class RuntimeUtilsTests(unittest.TestCase):
    def test_cpu_device_is_explicit(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_rejects_unknown_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto, cpu, or cuda"):
            resolve_device("tpu")


if __name__ == "__main__":
    unittest.main()
