"""
╔══════════════════════════════════════════════════════════════════════╗
║                        Py2Edg v0.1.0                              ║
║   One-line edge deployment for Computer Vision models              ║
║   by Mouissat Rabah Abderrahmane                                   ║
╚══════════════════════════════════════════════════════════════════════╝

Convert, quantize, optimize, and benchmark CV models for edge devices
with a single function call.

Quick Start:
    >>> import py2edg as rcv
    >>>
    >>> # Convert PyTorch model to optimized ONNX
    >>> result = rcv.convert("model.pt", target="onnx", quantize="fp16")
    >>>
    >>> # Full deployment pipeline: convert + optimize + benchmark
    >>> report = rcv.deploy("model.pt", device="rpi4", quantize="int8")
    >>>
    >>> # Benchmark any model
    >>> stats = rcv.benchmark("model.onnx", input_shape=(1, 3, 640, 640))
    >>>
    >>> # Compare original vs optimized
    >>> rcv.compare("model.pt", "model_opt.onnx", input_shape=(1, 3, 640, 640))
"""

__version__ = "0.1.0"
__author__ = "Mouissat Rabah Abderrahmane"
__license__ = "MIT"

# ── High-level API (the magic) ──
from py2edg.api import (
    convert,
    deploy,
    benchmark,
    compare,
    profile,
    validate,
    inspect_model,
)

# ── Device Profiles ──
from py2edg.devices import (
    DeviceProfile,
    get_device,
    list_devices,
    register_device,
)

# ── Deployment Recipe ──
from py2edg.recipe import (
    DeployRecipe,
    load_recipe,
    save_recipe,
)

# ── Report ──
from py2edg.report import DeployReport

__all__ = [
    # API
    "convert",
    "deploy",
    "benchmark",
    "compare",
    "profile",
    "validate",
    "inspect_model",
    # Devices
    "DeviceProfile",
    "get_device",
    "list_devices",
    "register_device",
    # Recipe
    "DeployRecipe",
    "load_recipe",
    "save_recipe",
    # Report
    "DeployReport",
]
