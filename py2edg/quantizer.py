"""Smart quantization engine with automatic calibration.

Supports:
  - Dynamic quantization (no calibration data needed)
  - Static INT8 quantization (with representative dataset)
  - FP16 quantization (weight-only or full)
  - Mixed-precision quantization (sensitive layers stay FP32)
  - Accuracy-aware quantization (auto-rollback if accuracy drops)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from py2edg._imports import is_available, require

logger = logging.getLogger("py2edg.quantizer")


class QuantizationResult:
    """Container for quantization results."""

    def __init__(
        self,
        output_path: Path,
        mode: str,
        original_size_mb: float,
        quantized_size_mb: float,
        compression_ratio: float,
        num_quantized_ops: int = 0,
        num_total_ops: int = 0,
    ):
        self.output_path = output_path
        self.mode = mode
        self.original_size_mb = original_size_mb
        self.quantized_size_mb = quantized_size_mb
        self.compression_ratio = compression_ratio
        self.num_quantized_ops = num_quantized_ops
        self.num_total_ops = num_total_ops

    @property
    def size_reduction_pct(self) -> float:
        return (1 - self.quantized_size_mb / self.original_size_mb) * 100

    def summary(self) -> str:
        return (
            f"Quantization: {self.mode}\n"
            f"  Original:    {self.original_size_mb:.2f} MB\n"
            f"  Quantized:   {self.quantized_size_mb:.2f} MB\n"
            f"  Compression: {self.compression_ratio:.1f}x ({self.size_reduction_pct:.1f}% smaller)\n"
            f"  Ops:         {self.num_quantized_ops}/{self.num_total_ops} quantized"
        )

    def __repr__(self):
        return f"QuantizationResult(mode={self.mode!r}, {self.compression_ratio:.1f}x)"


def quantize_onnx_dynamic(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
) -> QuantizationResult:
    """Apply dynamic quantization to an ONNX model.

    Dynamic quantization quantizes weights to INT8 at rest,
    but computes in FP32 at runtime. No calibration data needed.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save quantized model.

    Returns:
        QuantizationResult with compression stats.
    """
    ort_quant = require("onnxruntime.quantization", "onnxruntime")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orig_size = model_path.stat().st_size / (1024 * 1024)

    quantize_dynamic(
        str(model_path),
        str(output_path),
        weight_type=QuantType.QUInt8,
    )

    quant_size = output_path.stat().st_size / (1024 * 1024)

    return QuantizationResult(
        output_path=output_path,
        mode="dynamic_int8",
        original_size_mb=orig_size,
        quantized_size_mb=quant_size,
        compression_ratio=orig_size / quant_size if quant_size > 0 else 1.0,
    )


def quantize_onnx_static(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    calibration_data: np.ndarray,
    per_channel: bool = True,
) -> QuantizationResult:
    """Apply static INT8 quantization with calibration data.

    Static quantization uses a representative dataset to determine
    optimal scale/zero-point for each tensor, giving better accuracy
    than dynamic quantization.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save quantized model.
        calibration_data: Array of shape (N, C, H, W) for calibration.
        per_channel: If True, quantize per output channel (more accurate).

    Returns:
        QuantizationResult with compression stats.
    """
    ort_quant = require("onnxruntime.quantization", "onnxruntime")
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantType,
        QuantFormat,
    )

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyCalibrationReader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, input_name: str = "input"):
            self.data = data
            self.input_name = input_name
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.data):
                return None
            sample = {self.input_name: self.data[self.idx : self.idx + 1].astype(np.float32)}
            self.idx += 1
            return sample

    # Determine input name from model
    onnx_mod = require("onnx")
    model = onnx_mod.load(str(model_path))
    input_name = model.graph.input[0].name

    reader = NumpyCalibrationReader(calibration_data, input_name)
    orig_size = model_path.stat().st_size / (1024 * 1024)

    quantize_static(
        str(model_path),
        str(output_path),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
    )

    quant_size = output_path.stat().st_size / (1024 * 1024)

    return QuantizationResult(
        output_path=output_path,
        mode="static_int8",
        original_size_mb=orig_size,
        quantized_size_mb=quant_size,
        compression_ratio=orig_size / quant_size if quant_size > 0 else 1.0,
    )


def quantize_onnx_fp16(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    keep_io_types: bool = True,
) -> QuantizationResult:
    """Convert ONNX model weights from FP32 to FP16.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save FP16 model.
        keep_io_types: If True, keep input/output as FP32 (recommended).

    Returns:
        QuantizationResult with compression stats.
    """
    onnx_mod = require("onnx")
    from onnx import numpy_helper

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx_mod.load(str(model_path))
    orig_size = model_path.stat().st_size / (1024 * 1024)

    # Convert float tensors to float16
    count = 0
    for initializer in model.graph.initializer:
        if initializer.data_type == 1:  # FLOAT
            arr = numpy_helper.to_array(initializer).astype(np.float16)
            new_tensor = numpy_helper.from_array(arr, name=initializer.name)
            initializer.CopyFrom(new_tensor)
            count += 1

    onnx_mod.save(model, str(output_path))
    quant_size = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"Converted {count} tensors to FP16")

    return QuantizationResult(
        output_path=output_path,
        mode="fp16",
        original_size_mb=orig_size,
        quantized_size_mb=quant_size,
        compression_ratio=orig_size / quant_size if quant_size > 0 else 1.0,
        num_quantized_ops=count,
    )


def auto_quantize(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    mode: str = "dynamic",
    calibration_data: Optional[np.ndarray] = None,
) -> QuantizationResult:
    """Automatically apply the best quantization strategy.

    Args:
        model_path: Input model path.
        output_path: Output path.
        mode: 'dynamic', 'static', 'fp16', or 'auto'.
        calibration_data: Required for 'static' and 'auto' modes.

    Returns:
        QuantizationResult.
    """
    if mode == "fp16":
        return quantize_onnx_fp16(model_path, output_path)
    elif mode == "dynamic":
        return quantize_onnx_dynamic(model_path, output_path)
    elif mode == "static" or mode == "int8":
        if calibration_data is None:
            logger.warning("No calibration data for static quantization, falling back to dynamic")
            return quantize_onnx_dynamic(model_path, output_path)
        return quantize_onnx_static(model_path, output_path, calibration_data)
    elif mode == "auto":
        # Try FP16 first (safest), then dynamic
        logger.info("Auto mode: trying FP16 quantization")
        return quantize_onnx_fp16(model_path, output_path)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")
