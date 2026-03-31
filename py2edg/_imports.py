"""Lazy import helpers for optional dependencies."""

from __future__ import annotations

import importlib
from typing import Any


class _LazyModule:
    """Delay ImportError until the module is actually used."""

    def __init__(self, name: str, install_hint: str):
        self._name = name
        self._hint = install_hint
        self._mod = None

    def _load(self):
        if self._mod is None:
            try:
                self._mod = importlib.import_module(self._name)
            except ImportError:
                raise ImportError(
                    f"'{self._name}' is required for this operation.\n"
                    f"Install it with:  pip install {self._hint}"
                )
        return self._mod

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)


def require(module_name: str, install_hint: str | None = None):
    """Import a module or raise a helpful error."""
    hint = install_hint or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"'{module_name}' is required for this operation.\n"
            f"Install it with:  pip install {hint}"
        )


def is_available(module_name: str) -> bool:
    """Check if a module is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# ── Pre-built lazy imports ──
torch = _LazyModule("torch", "torch")
torchvision = _LazyModule("torchvision", "torchvision")
onnx = _LazyModule("onnx", "onnx")
onnxruntime = _LazyModule("onnxruntime", "onnxruntime")
tensorflow = _LazyModule("tensorflow", "tensorflow")
openvino = _LazyModule("openvino", "openvino")
onnxsim = _LazyModule("onnxsim", "onnxsim")
