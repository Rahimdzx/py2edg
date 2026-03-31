"""Model conversion engine: PyTorch / TensorFlow → ONNX / TFLite / OpenVINO.

Handles the full conversion pipeline including:
  - Format detection (PyTorch .pt/.pth, TF SavedModel, Keras .h5, ONNX .onnx)
  - ONNX export with proper opset and dynamic axes
  - ONNX → TFLite conversion
  - ONNX → OpenVINO conversion
  - Automatic input shape inference
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from py2edg._imports import is_available, require
from py2edg.devices import DeviceProfile

logger = logging.getLogger("py2edg.converter")


# ═══════════════════════════════════════════════════════════════
# Format Detection
# ═══════════════════════════════════════════════════════════════

def detect_format(model_path: Union[str, Path]) -> str:
    """Detect the format of a saved model.

    Returns:
        One of: 'pytorch', 'onnx', 'tflite', 'tensorflow', 'keras', 'unknown'.
    """
    path = Path(model_path)

    if path.suffix in (".pt", ".pth", ".torchscript"):
        return "pytorch"
    elif path.suffix == ".onnx":
        return "onnx"
    elif path.suffix == ".tflite":
        return "tflite"
    elif path.suffix in (".h5", ".keras"):
        return "keras"
    elif path.is_dir() and (path / "saved_model.pb").exists():
        return "tensorflow"
    elif path.suffix == ".xml" and (path.with_suffix(".bin")).exists():
        return "openvino"
    else:
        return "unknown"


# ═══════════════════════════════════════════════════════════════
# PyTorch → ONNX
# ═══════════════════════════════════════════════════════════════

def pytorch_to_onnx(
    model: Any,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    opset: int = 17,
    dynamic_axes: Optional[Dict] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    simplify: bool = True,
) -> Path:
    """Convert a PyTorch model to ONNX format.

    Args:
        model: PyTorch model (nn.Module) or path to .pt/.pth file.
        output_path: Where to save the .onnx file.
        input_shape: Input tensor shape (batch, channels, height, width).
        opset: ONNX operator set version.
        dynamic_axes: Dynamic axis configuration for variable batch/input sizes.
        input_names: Names for input tensors.
        output_names: Names for output tensors.
        simplify: If True, run onnx-simplifier after export.

    Returns:
        Path to the saved ONNX model.
    """
    torch = require("torch", "torch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model if path given
    if isinstance(model, (str, Path)):
        model_path = Path(model)
        if model_path.suffix == ".torchscript":
            model = torch.jit.load(str(model_path))
        else:
            model = torch.load(str(model_path), map_location="cpu", weights_only=False)
            if isinstance(model, dict):
                # Common pattern: checkpoint dict with 'model' key
                for key in ("model", "state_dict", "net"):
                    if key in model:
                        logger.info(f"Extracting '{key}' from checkpoint dict")
                        model = model[key]
                        break

    if hasattr(model, "eval"):
        model.eval()

    # Create dummy input
    dummy = torch.randn(*input_shape)

    # Default dynamic axes: batch dimension
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    logger.info(f"Exporting PyTorch → ONNX (opset={opset}, shape={input_shape})")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Simplify
    if simplify and is_available("onnxsim"):
        try:
            import onnx as onnx_mod
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx_mod.load(str(output_path))
            simplified, ok = onnx_simplify(onnx_model)
            if ok:
                onnx_mod.save(simplified, str(output_path))
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification failed, keeping original")
        except Exception as e:
            logger.warning(f"ONNX simplification error: {e}")

    logger.info(f"Saved ONNX model to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
# ONNX → TFLite
# ═══════════════════════════════════════════════════════════════

def onnx_to_tflite(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    quantize: str = "fp32",
    calibration_data: Optional[np.ndarray] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Path:
    """Convert an ONNX model to TFLite format.

    Args:
        onnx_path: Path to ONNX model.
        output_path: Where to save .tflite file.
        quantize: Quantization mode ('fp32', 'fp16', 'int8', 'dynamic').
        calibration_data: Representative dataset for INT8 calibration (N, C, H, W).
        input_shape: Override input shape.

    Returns:
        Path to the saved TFLite model.
    """
    tf = require("tensorflow", "tensorflow")
    import onnx as onnx_mod

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ONNX → TF SavedModel via onnx-tf or tf2onnx reverse
    try:
        from onnx_tf.backend import prepare
        onnx_model = onnx_mod.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tmp_dir = tempfile.mkdtemp()
        tf_rep.export_graph(tmp_dir)
    except ImportError:
        raise ImportError(
            "ONNX → TFLite requires 'onnx-tf'.\n"
            "Install with: pip install onnx-tf"
        )

    # Convert TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)

    if quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if calibration_data is not None:
            def representative_gen():
                for i in range(min(len(calibration_data), 200)):
                    sample = calibration_data[i:i+1].astype(np.float32)
                    # ONNX is NCHW, TFLite expects NHWC
                    if sample.ndim == 4 and sample.shape[1] in (1, 3, 4):
                        sample = np.transpose(sample, (0, 2, 3, 1))
                    yield [sample]
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    elif quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Saved TFLite model to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
# ONNX → OpenVINO
# ═══════════════════════════════════════════════════════════════

def onnx_to_openvino(
    onnx_path: Union[str, Path],
    output_dir: Union[str, Path],
    quantize: str = "fp32",
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Path:
    """Convert an ONNX model to OpenVINO IR format.

    Args:
        onnx_path: Path to ONNX model.
        output_dir: Directory to save .xml and .bin files.
        quantize: 'fp32', 'fp16', or 'int8'.
        input_shape: Override input shape.

    Returns:
        Path to the .xml file.
    """
    ov = require("openvino", "openvino")

    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    model = core.read_model(str(onnx_path))

    if input_shape is not None:
        model.reshape({model.input().get_any_name(): list(input_shape)})

    compress_to_fp16 = quantize == "fp16"

    xml_path = output_dir / f"{onnx_path.stem}.xml"
    ov.save_model(model, str(xml_path), compress_to_fp16=compress_to_fp16)

    logger.info(f"Saved OpenVINO IR to {xml_path}")
    return xml_path


# ═══════════════════════════════════════════════════════════════
# Universal Converter
# ═══════════════════════════════════════════════════════════════

def convert_model(
    model: Any,
    output_path: Union[str, Path],
    target: str = "onnx",
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    quantize: str = "fp32",
    opset: int = 17,
    simplify: bool = True,
    calibration_data: Optional[np.ndarray] = None,
    device: Optional[DeviceProfile] = None,
) -> Path:
    """Universal model converter.

    Automatically detects source format and converts to target.

    Args:
        model: Model object or path to model file.
        output_path: Output file path.
        target: Target format ('onnx', 'tflite', 'openvino').
        input_shape: Model input shape.
        quantize: Quantization mode ('fp32', 'fp16', 'int8', 'dynamic').
        opset: ONNX opset version.
        simplify: Simplify ONNX graph.
        calibration_data: Data for INT8 calibration.
        device: Device profile to auto-configure settings.

    Returns:
        Path to converted model.
    """
    output_path = Path(output_path)

    # Apply device profile overrides
    if device is not None:
        target = device.preferred_format if target == "onnx" else target
        quantize = device.quantize
        opset = min(opset, device.onnx_opset)
        if device.input_sizes:
            input_shape = tuple(device.input_sizes[0])

    # Detect source format
    src_format = "pytorch"  # default for model objects
    if isinstance(model, (str, Path)):
        src_format = detect_format(model)

    logger.info(f"Converting {src_format} → {target} (quantize={quantize})")

    # Step 1: Get to ONNX first
    if src_format == "onnx":
        onnx_path = Path(model)
    elif src_format == "pytorch":
        if target == "onnx" and quantize in ("fp32", "fp16"):
            return pytorch_to_onnx(
                model, output_path, input_shape=input_shape,
                opset=opset, simplify=simplify,
            )
        else:
            # Need intermediate ONNX
            onnx_path = output_path.with_suffix(".onnx")
            pytorch_to_onnx(model, onnx_path, input_shape=input_shape, opset=opset, simplify=simplify)
    else:
        raise ValueError(f"Unsupported source format: {src_format}")

    # Step 2: ONNX → target
    if target == "onnx":
        if onnx_path != output_path:
            shutil.copy2(onnx_path, output_path)
        return output_path
    elif target == "tflite":
        return onnx_to_tflite(
            onnx_path, output_path, quantize=quantize,
            calibration_data=calibration_data,
        )
    elif target == "openvino":
        return onnx_to_openvino(
            onnx_path, output_path.parent, quantize=quantize,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unsupported target format: {target}")
