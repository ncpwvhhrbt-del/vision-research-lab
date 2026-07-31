"""Unit tests for configuration logic that do not require dataset loading."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data.cifar import CIFARDataConfig, inspect_cifar10


class CIFARDataConfigTests(unittest.TestCase):
    def test_rejects_invalid_validation_ratio(self) -> None:
        config = CIFARDataConfig(root=Path("data"), val_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "val_ratio"):
            config.validate()

    def test_inspection_lists_missing_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            report = inspect_cifar10(temporary_directory)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["missing_files"]), 7)


if __name__ == "__main__":
    unittest.main()
