"""Dataset configuration and DataLoader construction."""

from .cifar import CIFARDataConfig, DataBundle, inspect_cifar10

__all__ = ["CIFARDataConfig", "DataBundle", "inspect_cifar10"]
