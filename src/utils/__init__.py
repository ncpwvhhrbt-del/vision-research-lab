"""Runtime utilities shared by command-line experiment entrypoints."""

from .runtime import resolve_device, seed_everything

__all__ = ["resolve_device", "seed_everything"]
