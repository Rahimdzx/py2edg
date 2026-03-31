"""Edge device profiles with hardware-specific optimization presets.

Each device profile encodes the optimal conversion settings for a target
hardware platform — things like quantization type, operator compatibility,
memory limits, and runtime preferences.

Built-in profiles:
    rpi4, rpi5, jetson_nano, jetson_orin, orange_pi, coral_tpu,
    android_cpu, android_gpu, ios_coreml, generic_arm, generic_x86

Custom profiles:
    >>> rcv.register_device(DeviceProfile(
    ...     name="my_board",
    ...     compute="cpu",
    ...     memory_mb=512,
    ...     preferred_format="tflite",
    ...     quantize="int8",
    ... ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


TargetFormat = Literal["onnx", "tflite", "openvino", "coreml"]
QuantizeMode = Literal["fp32", "fp16", "int8", "uint8", "dynamic"]
ComputeType = Literal["cpu", "gpu", "npu", "tpu"]


@dataclass
class DeviceProfile:
    """Hardware profile for an edge deployment target.

    Attributes:
        name: Unique device identifier (e.g., 'rpi4', 'jetson_nano').
        display_name: Human-readable name.
        compute: Primary compute unit.
        memory_mb: Available RAM in megabytes.
        preferred_format: Best model format for this device.
        quantize: Recommended quantization mode.
        max_threads: Optimal number of inference threads.
        supported_ops: List of supported ONNX operator sets or TFLite ops.
        onnx_opset: Recommended ONNX opset version.
        optimization_level: ORT optimization level (0-3).
        notes: Extra deployment tips.
        extra: Additional key-value settings for custom logic.
    """

    name: str
    display_name: str = ""
    compute: ComputeType = "cpu"
    memory_mb: int = 1024
    preferred_format: TargetFormat = "onnx"
    quantize: QuantizeMode = "fp16"
    max_threads: int = 4
    supported_ops: List[str] = field(default_factory=list)
    onnx_opset: int = 17
    optimization_level: int = 3  # ORT: 0=disabled, 1=basic, 2=extended, 3=all
    input_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 640, 640]])
    notes: str = ""
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "compute": self.compute,
            "memory_mb": self.memory_mb,
            "preferred_format": self.preferred_format,
            "quantize": self.quantize,
            "max_threads": self.max_threads,
            "onnx_opset": self.onnx_opset,
            "optimization_level": self.optimization_level,
            "input_sizes": self.input_sizes,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeviceProfile":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# Built-in Device Registry
# ═══════════════════════════════════════════════════════════════

_REGISTRY: Dict[str, DeviceProfile] = {}


def _register_builtin(profile: DeviceProfile):
    _REGISTRY[profile.name] = profile


# ── Raspberry Pi ──
_register_builtin(DeviceProfile(
    name="rpi4",
    display_name="Raspberry Pi 4 (ARM Cortex-A72)",
    compute="cpu",
    memory_mb=2048,
    preferred_format="tflite",
    quantize="int8",
    max_threads=4,
    onnx_opset=13,
    optimization_level=3,
    input_sizes=[[1, 3, 320, 320], [1, 3, 416, 416]],
    notes="INT8 TFLite gives best perf. Avoid models >50MB. Use 320px input for real-time.",
))

_register_builtin(DeviceProfile(
    name="rpi5",
    display_name="Raspberry Pi 5 (ARM Cortex-A76)",
    compute="cpu",
    memory_mb=4096,
    preferred_format="tflite",
    quantize="int8",
    max_threads=4,
    onnx_opset=17,
    optimization_level=3,
    input_sizes=[[1, 3, 416, 416], [1, 3, 640, 640]],
    notes="2x faster than RPi4. Can handle 640px YOLO at ~8 FPS with INT8.",
))

# ── NVIDIA Jetson ──
_register_builtin(DeviceProfile(
    name="jetson_nano",
    display_name="NVIDIA Jetson Nano (Maxwell 128-core)",
    compute="gpu",
    memory_mb=4096,
    preferred_format="onnx",
    quantize="fp16",
    max_threads=4,
    onnx_opset=17,
    optimization_level=3,
    input_sizes=[[1, 3, 416, 416], [1, 3, 640, 640]],
    notes="Use FP16 with CUDA EP. TensorRT gives additional 2-3x speedup.",
    extra={"execution_provider": "CUDAExecutionProvider", "tensorrt": True},
))

_register_builtin(DeviceProfile(
    name="jetson_orin",
    display_name="NVIDIA Jetson Orin (Ampere)",
    compute="gpu",
    memory_mb=8192,
    preferred_format="onnx",
    quantize="fp16",
    max_threads=8,
    onnx_opset=17,
    optimization_level=3,
    input_sizes=[[1, 3, 640, 640], [1, 3, 1280, 1280]],
    notes="Extremely powerful. FP16 is the sweet spot. INT8 if latency-critical.",
    extra={"execution_provider": "CUDAExecutionProvider", "tensorrt": True},
))

# ── Orange Pi ──
_register_builtin(DeviceProfile(
    name="orange_pi",
    display_name="Orange Pi 5 (RK3588 / Mali-G610)",
    compute="npu",
    memory_mb=4096,
    preferred_format="onnx",
    quantize="int8",
    max_threads=4,
    onnx_opset=13,
    optimization_level=2,
    input_sizes=[[1, 3, 320, 320], [1, 3, 416, 416]],
    notes="RK3588 NPU supports INT8 RKNN format. Convert ONNX→RKNN for max perf.",
    extra={"npu_runtime": "rknn", "rknn_target": "rk3588"},
))

# ── Google Coral ──
_register_builtin(DeviceProfile(
    name="coral_tpu",
    display_name="Google Coral Edge TPU",
    compute="tpu",
    memory_mb=1024,
    preferred_format="tflite",
    quantize="uint8",
    max_threads=1,
    onnx_opset=13,
    optimization_level=3,
    input_sizes=[[1, 3, 320, 320]],
    notes="Requires full uint8 quantization. Only specific ops supported. Use edgetpu_compiler.",
    extra={"edgetpu_compiler": True},
))

# ── Mobile ──
_register_builtin(DeviceProfile(
    name="android_cpu",
    display_name="Android (CPU / ARM)",
    compute="cpu",
    memory_mb=2048,
    preferred_format="tflite",
    quantize="int8",
    max_threads=4,
    onnx_opset=13,
    optimization_level=3,
    input_sizes=[[1, 3, 320, 320], [1, 3, 416, 416]],
    notes="TFLite with XNNPACK delegate gives best CPU performance on Android.",
    extra={"delegate": "xnnpack"},
))

_register_builtin(DeviceProfile(
    name="android_gpu",
    display_name="Android (GPU / Adreno or Mali)",
    compute="gpu",
    memory_mb=2048,
    preferred_format="tflite",
    quantize="fp16",
    max_threads=1,
    onnx_opset=13,
    optimization_level=3,
    input_sizes=[[1, 3, 416, 416]],
    notes="Use GPU delegate. FP16 preferred. Not all ops supported on GPU.",
    extra={"delegate": "gpu"},
))

_register_builtin(DeviceProfile(
    name="ios_coreml",
    display_name="iOS (Core ML / Apple Neural Engine)",
    compute="npu",
    memory_mb=4096,
    preferred_format="coreml",
    quantize="fp16",
    max_threads=4,
    onnx_opset=17,
    optimization_level=3,
    input_sizes=[[1, 3, 640, 640]],
    notes="Convert ONNX → CoreML with coremltools. ANE supports FP16 natively.",
    extra={"compute_units": "ALL"},
))

# ── Generic ──
_register_builtin(DeviceProfile(
    name="generic_arm",
    display_name="Generic ARM (Cortex-A series)",
    compute="cpu",
    memory_mb=1024,
    preferred_format="onnx",
    quantize="int8",
    max_threads=4,
    onnx_opset=13,
    optimization_level=3,
    notes="Safe defaults for any ARM board. INT8 for best CPU inference.",
))

_register_builtin(DeviceProfile(
    name="generic_x86",
    display_name="Generic x86_64 (Intel / AMD)",
    compute="cpu",
    memory_mb=8192,
    preferred_format="onnx",
    quantize="fp32",
    max_threads=8,
    onnx_opset=17,
    optimization_level=3,
    notes="ONNX Runtime with OpenVINO EP for Intel. Use FP32 or dynamic quantization.",
))

_register_builtin(DeviceProfile(
    name="server_gpu",
    display_name="Server GPU (NVIDIA T4/A10/A100)",
    compute="gpu",
    memory_mb=16384,
    preferred_format="onnx",
    quantize="fp16",
    max_threads=1,
    onnx_opset=17,
    optimization_level=3,
    notes="TensorRT via ONNX Runtime gives best throughput. FP16 is standard.",
    extra={"execution_provider": "TensorrtExecutionProvider"},
))


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def get_device(name: str) -> DeviceProfile:
    """Get a device profile by name.

    Args:
        name: Device identifier (e.g., 'rpi4', 'jetson_nano').

    Raises:
        KeyError: If device is not found in registry.

    Example:
        >>> device = rcv.get_device("rpi4")
        >>> print(device.quantize)  # 'int8'
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Device '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_devices() -> List[DeviceProfile]:
    """List all registered device profiles.

    Example:
        >>> for dev in rcv.list_devices():
        ...     print(f"{dev.name}: {dev.display_name} ({dev.quantize})")
    """
    return list(_REGISTRY.values())


def register_device(profile: DeviceProfile) -> None:
    """Register a custom device profile.

    Example:
        >>> rcv.register_device(DeviceProfile(
        ...     name="my_fpga",
        ...     display_name="Custom FPGA Board",
        ...     compute="npu",
        ...     memory_mb=256,
        ...     preferred_format="onnx",
        ...     quantize="int8",
        ... ))
    """
    _REGISTRY[profile.name] = profile
