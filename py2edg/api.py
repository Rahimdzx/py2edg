"""High-level API: the magic one-liners.

This module provides the top-level functions that users interact with.
Each function orchestrates the lower-level modules (converter, quantizer,
optimizer, benchmark) into a cohesive pipeline.

Examples:
    >>> import py2edg as rcv
    >>>
    >>> # Simple conversion
    >>> rcv.convert("model.pt", target="onnx", quantize="fp16")
    >>>
    >>> # Full pipeline for Raspberry Pi 4
    >>> report = rcv.deploy("model.pt", device="rpi4")
    >>> report.print()
    >>>
    >>> # Benchmark
    >>> stats = rcv.benchmark("model.onnx", input_shape=(1, 3, 640, 640))
    >>> print(stats.summary())
    >>>
    >>> # Compare original vs optimized
    >>> rcv.compare("model.pt", "model_opt.onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from py2edg import __version__
from py2edg._imports import is_available, require
from py2edg.benchmark import (
    BenchmarkResult,
    benchmark_onnx,
    benchmark_pytorch,
    compare_models,
)
from py2edg.converter import convert_model, detect_format
from py2edg.devices import DeviceProfile, get_device
from py2edg.optimizer import OptimizationResult, inspect_onnx, optimize_onnx
from py2edg.quantizer import QuantizationResult, auto_quantize
from py2edg.recipe import DeployRecipe
from py2edg.report import DeployReport

logger = logging.getLogger("py2edg")


# ═══════════════════════════════════════════════════════════════
# convert() — Format conversion with optional quantization
# ═══════════════════════════════════════════════════════════════

def convert(
    model: Any,
    output: Optional[str] = None,
    target: str = "onnx",
    quantize: str = "fp32",
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: Optional[str] = None,
    opset: int = 17,
    simplify: bool = True,
    calibration_data: Optional[np.ndarray] = None,
) -> Path:
    """Convert a model to an edge-friendly format.

    This is the core conversion function. Give it any model, tell it
    where you want to go, and it handles the rest.

    Args:
        model: Model object (nn.Module) or path to model file.
        output: Output file path. Auto-generated if None.
        target: Target format ('onnx', 'tflite', 'openvino').
        quantize: Quantization mode ('fp32', 'fp16', 'int8', 'dynamic').
        input_shape: Input tensor shape.
        device: Device name to auto-configure (e.g., 'rpi4', 'jetson_nano').
        opset: ONNX opset version.
        simplify: Simplify ONNX graph.
        calibration_data: Calibration data for INT8 quantization.

    Returns:
        Path to the converted model file.

    Examples:
        >>> rcv.convert("yolov8n.pt", target="onnx")
        >>> rcv.convert("model.pt", target="tflite", quantize="int8", device="rpi4")
        >>> rcv.convert("model.onnx", target="openvino", quantize="fp16")
    """
    # Resolve device profile
    device_profile = None
    if device:
        device_profile = get_device(device)
        if target == "onnx":
            target = device_profile.preferred_format
        quantize = device_profile.quantize
        input_shape = tuple(device_profile.input_sizes[0]) if device_profile.input_sizes else input_shape

    # Auto-generate output path
    if output is None:
        model_name = Path(model).stem if isinstance(model, (str, Path)) else "model"
        ext_map = {"onnx": ".onnx", "tflite": ".tflite", "openvino": ".xml", "coreml": ".mlmodel"}
        suffix = f"_{quantize}" if quantize != "fp32" else ""
        output = f"{model_name}{suffix}{ext_map.get(target, '.onnx')}"

    result = convert_model(
        model=model,
        output_path=output,
        target=target,
        input_shape=input_shape,
        quantize=quantize,
        opset=opset,
        simplify=simplify,
        calibration_data=calibration_data,
        device=device_profile,
    )

    logger.info(f"✅ Converted to {result}")
    return result


# ═══════════════════════════════════════════════════════════════
# deploy() — Full pipeline: convert + optimize + quantize + benchmark
# ═══════════════════════════════════════════════════════════════

def deploy(
    model: Any = None,
    output_dir: str = "./deploy_output",
    device: str = "generic_arm",
    target: Optional[str] = None,
    quantize: Optional[str] = None,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    opset: int = 17,
    optimize_level: int = 3,
    benchmark_runs: int = 100,
    calibration_data: Optional[np.ndarray] = None,
    recipe: Optional[DeployRecipe] = None,
    verbose: bool = True,
) -> DeployReport:
    """Full deployment pipeline: convert → optimize → quantize → benchmark.

    This is the main entry point. Give it a model and a target device,
    and it produces an optimized, benchmarked deployment artifact.

    Args:
        model: Model object or path to model file.
        output_dir: Directory for all output artifacts.
        device: Target device name (e.g., 'rpi4', 'jetson_orin').
        target: Override target format (uses device default if None).
        quantize: Override quantization mode (uses device default if None).
        input_shape: Model input shape.
        opset: ONNX opset version.
        optimize_level: Graph optimization level (0-3).
        benchmark_runs: Number of benchmark iterations.
        calibration_data: Data for INT8 calibration.
        recipe: Use a DeployRecipe instead of individual args.
        verbose: Print progress and report.

    Returns:
        DeployReport with all results and artifacts.

    Examples:
        >>> report = rcv.deploy("yolov8n.pt", device="rpi4")
        >>> report = rcv.deploy("model.pt", device="jetson_nano", quantize="fp16")
        >>> report = rcv.deploy(recipe=rcv.load_recipe("deploy.yaml"))
    """
    # Apply recipe if provided
    if recipe is not None:
        model = model or recipe.model
        output_dir = recipe.output_dir
        device = recipe.device
        target = target or recipe.target
        quantize = quantize or recipe.quantize
        input_shape = tuple(recipe.input_shape)
        opset = recipe.opset
        optimize_level = recipe.optimize_level
        benchmark_runs = recipe.benchmark_runs

    # Resolve device
    device_profile = get_device(device)
    target = target or device_profile.preferred_format
    quantize = quantize or device_profile.quantize

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model).stem if isinstance(model, (str, Path)) else "model"

    report = DeployReport(
        model_name=model_name,
        py2edg_version=__version__,
        device=device_profile,
        target_format=target,
        output_dir=out_dir,
    )

    if verbose:
        print(f"\n🚀 Py2Edg Deploy Pipeline")
        print(f"   Model:  {model_name}")
        print(f"   Device: {device_profile.display_name}")
        print(f"   Target: {target} ({quantize})")
        print()

    # ── Step 1: Convert to ONNX ──
    src_format = detect_format(model) if isinstance(model, (str, Path)) else "pytorch"
    onnx_path = out_dir / f"{model_name}.onnx"

    if src_format != "onnx":
        if verbose:
            print("📦 Step 1/4: Converting to ONNX...")
        convert_model(
            model, onnx_path,
            target="onnx",
            input_shape=input_shape,
            opset=opset,
            simplify=True,
        )
    else:
        onnx_path = Path(model)

    # ── Step 2: Optimize graph ──
    opt_path = out_dir / f"{model_name}_optimized.onnx"
    opt_result = None

    if optimize_level > 0 and is_available("onnxruntime"):
        if verbose:
            print("⚡ Step 2/4: Optimizing graph...")
        try:
            opt_result = optimize_onnx(
                onnx_path, opt_path,
                level=optimize_level,
                simplify=True,
            )
            report.optimization = opt_result
            onnx_path = opt_path  # Use optimized for next steps
            if verbose:
                print(f"   Ops: {opt_result.original_ops} → {opt_result.optimized_ops}")
        except Exception as e:
            report.warnings.append(f"Optimization failed: {e}")
            if verbose:
                print(f"   ⚠ Optimization skipped: {e}")

    # ── Step 3: Quantize ──
    quant_result = None
    final_path = onnx_path

    if quantize not in ("fp32", None):
        if verbose:
            print(f"🔧 Step 3/4: Quantizing ({quantize})...")
        try:
            quant_path = out_dir / f"{model_name}_{quantize}.onnx"
            quant_result = auto_quantize(
                onnx_path, quant_path,
                mode=quantize,
                calibration_data=calibration_data,
            )
            report.quantization = quant_result
            final_path = quant_path
            if verbose:
                print(f"   Size: {quant_result.original_size_mb:.1f} → {quant_result.quantized_size_mb:.1f} MB")
        except Exception as e:
            report.warnings.append(f"Quantization failed: {e}")
            if verbose:
                print(f"   ⚠ Quantization skipped: {e}")

    # ── Step 4: Convert to final target format (if not ONNX) ──
    if target != "onnx":
        if verbose:
            print(f"🔄 Converting ONNX → {target}...")
        try:
            ext_map = {"tflite": ".tflite", "openvino": ".xml"}
            target_path = out_dir / f"{model_name}{ext_map.get(target, '')}"
            convert_model(
                str(final_path), target_path,
                target=target,
                input_shape=input_shape,
                quantize=quantize,
                calibration_data=calibration_data,
            )
            final_path = target_path
        except Exception as e:
            report.warnings.append(f"Target conversion failed: {e}, keeping ONNX")
            if verbose:
                print(f"   ⚠ Falling back to ONNX: {e}")

    report.conversion_output = final_path

    # ── Step 5: Benchmark ──
    if is_available("onnxruntime") and final_path.suffix == ".onnx":
        if verbose:
            print("📊 Step 4/4: Benchmarking...")
        try:
            ep = device_profile.extra.get("execution_provider", "CPUExecutionProvider")
            bench = benchmark_onnx(
                final_path,
                input_shape=input_shape,
                num_runs=benchmark_runs,
                warmup_runs=min(benchmark_runs // 5, 20),
                num_threads=device_profile.max_threads,
                execution_provider=ep,
            )
            report.benchmark_after = bench
            if verbose:
                print(f"   Latency: {bench.latency_mean_ms:.1f} ms ({bench.throughput_fps:.0f} FPS)")
        except Exception as e:
            report.warnings.append(f"Benchmark failed: {e}")

    report.artifacts = [str(p) for p in out_dir.glob("*") if p.is_file()]

    if verbose:
        report.print()

    return report


# ═══════════════════════════════════════════════════════════════
# benchmark() — Standalone benchmarking
# ═══════════════════════════════════════════════════════════════

def benchmark(
    model: Any,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_runs: int = 100,
    warmup_runs: int = 10,
    num_threads: int = 4,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark any model (ONNX or PyTorch).

    Args:
        model: Path to ONNX model or PyTorch nn.Module.
        input_shape: Input tensor shape.
        num_runs: Number of benchmark iterations.
        warmup_runs: Warm-up iterations.
        num_threads: Number of inference threads.
        device: 'cpu' or 'cuda' (for PyTorch models).

    Returns:
        BenchmarkResult with detailed statistics.

    Examples:
        >>> stats = rcv.benchmark("model.onnx")
        >>> print(stats.summary())
        >>> print(f"FPS: {stats.throughput_fps:.1f}")
    """
    if isinstance(model, (str, Path)):
        fmt = detect_format(model)
        if fmt == "onnx":
            return benchmark_onnx(
                model, input_shape=input_shape,
                num_runs=num_runs, warmup_runs=warmup_runs,
                num_threads=num_threads,
            )
        else:
            raise ValueError(f"Direct benchmarking of '{fmt}' files not yet supported. Convert to ONNX first.")
    else:
        # Assume PyTorch model
        return benchmark_pytorch(
            model, input_shape=input_shape,
            num_runs=num_runs, warmup_runs=warmup_runs,
            device=device,
        )


# ═══════════════════════════════════════════════════════════════
# compare() — Multi-model comparison
# ═══════════════════════════════════════════════════════════════

def compare(
    *models: Any,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_runs: int = 50,
    num_threads: int = 4,
) -> str:
    """Compare multiple models side by side.

    Args:
        *models: Model paths (ONNX) or PyTorch modules.
        input_shape: Input shape for all models.
        num_runs: Benchmark runs per model.
        num_threads: Inference threads.

    Returns:
        Formatted comparison table string.

    Examples:
        >>> table = rcv.compare("model.onnx", "model_fp16.onnx", "model_int8.onnx")
        >>> print(table)
    """
    results = []
    for m in models:
        try:
            result = benchmark(m, input_shape=input_shape, num_runs=num_runs, num_threads=num_threads)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to benchmark {m}: {e}")

    table = compare_models(results)
    print(table)
    return table


# ═══════════════════════════════════════════════════════════════
# profile() — Layer-level profiling
# ═══════════════════════════════════════════════════════════════

def profile(
    model_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_threads: int = 4,
) -> dict:
    """Profile an ONNX model layer by layer.

    Returns per-node timing and memory information using ORT profiling.

    Args:
        model_path: Path to ONNX model.
        input_shape: Input tensor shape.
        num_threads: Inference threads.

    Returns:
        Dictionary with per-node profiling data.

    Example:
        >>> prof = rcv.profile("model.onnx")
        >>> for node in prof['nodes'][:5]:
        ...     print(f"{node['name']}: {node['time_ms']:.2f}ms")
    """
    ort = require("onnxruntime", "onnxruntime")

    model_path = Path(model_path)

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.intra_op_num_threads = num_threads

    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Run inference to generate profile
    for _ in range(5):
        session.run(None, {input_name: dummy})

    profile_path = session.end_profiling()

    # Parse profile JSON
    import json
    with open(profile_path) as f:
        profile_data = json.load(f)

    nodes = []
    for item in profile_data:
        if item.get("cat") == "Node":
            nodes.append({
                "name": item.get("name", ""),
                "op_type": item.get("args", {}).get("op_name", ""),
                "time_us": item.get("dur", 0),
                "time_ms": item.get("dur", 0) / 1000.0,
            })

    # Sort by time
    nodes.sort(key=lambda x: x["time_us"], reverse=True)

    # Clean up profile file
    Path(profile_path).unlink(missing_ok=True)

    return {
        "model": str(model_path),
        "total_time_ms": sum(n["time_ms"] for n in nodes),
        "num_nodes": len(nodes),
        "nodes": nodes,
        "top_5_bottlenecks": nodes[:5],
    }


# ═══════════════════════════════════════════════════════════════
# validate() — Output consistency check
# ═══════════════════════════════════════════════════════════════

def validate(
    original_model: Any,
    converted_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_samples: int = 10,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict:
    """Validate that a converted model produces similar outputs.

    Compares the outputs of the original and converted models
    across multiple random inputs.

    Args:
        original_model: Original PyTorch model or ONNX path.
        converted_path: Path to converted model (ONNX).
        input_shape: Input tensor shape.
        num_samples: Number of test inputs.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dictionary with validation results.

    Example:
        >>> result = rcv.validate(pytorch_model, "model_fp16.onnx")
        >>> print(f"Max diff: {result['max_diff']:.6f}")
        >>> print(f"Pass: {result['passed']}")
    """
    ort = require("onnxruntime", "onnxruntime")

    converted_path = Path(converted_path)

    # Setup converted model
    session = ort.InferenceSession(str(converted_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    diffs = []
    passed = True

    for i in range(num_samples):
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Get original output
        if isinstance(original_model, (str, Path)):
            orig_session = ort.InferenceSession(str(original_model), providers=["CPUExecutionProvider"])
            orig_name = orig_session.get_inputs()[0].name
            orig_out = orig_session.run(None, {orig_name: test_input})[0]
        else:
            # PyTorch
            torch = require("torch", "torch")
            original_model.eval()
            with torch.no_grad():
                orig_out = original_model(torch.from_numpy(test_input)).numpy()

        # Get converted output
        conv_out = session.run(None, {input_name: test_input})[0]

        max_diff = float(np.max(np.abs(orig_out - conv_out)))
        mean_diff = float(np.mean(np.abs(orig_out - conv_out)))
        diffs.append({"sample": i, "max_diff": max_diff, "mean_diff": mean_diff})

        if not np.allclose(orig_out, conv_out, atol=atol, rtol=rtol):
            passed = False

    overall_max = max(d["max_diff"] for d in diffs)
    overall_mean = np.mean([d["mean_diff"] for d in diffs])

    result = {
        "passed": passed,
        "max_diff": overall_max,
        "mean_diff": float(overall_mean),
        "num_samples": num_samples,
        "atol": atol,
        "rtol": rtol,
        "per_sample": diffs,
    }

    status = "✅ PASSED" if passed else "❌ FAILED"
    logger.info(f"Validation {status}: max_diff={overall_max:.6f}, mean_diff={overall_mean:.6f}")

    return result


# ═══════════════════════════════════════════════════════════════
# inspect_model() — Quick model info
# ═══════════════════════════════════════════════════════════════

def inspect_model(model_path: Union[str, Path]) -> dict:
    """Quick inspection of any model file.

    Args:
        model_path: Path to model file.

    Returns:
        Dictionary with model metadata.

    Example:
        >>> info = rcv.inspect_model("model.onnx")
        >>> print(f"Ops: {info['num_ops']}, Size: {info['size_mb']:.1f}MB")
    """
    model_path = Path(model_path)
    fmt = detect_format(model_path)

    if fmt == "onnx":
        return inspect_onnx(model_path)
    else:
        return {
            "path": str(model_path),
            "format": fmt,
            "size_mb": model_path.stat().st_size / (1024 * 1024),
        }
