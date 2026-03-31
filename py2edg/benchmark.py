"""Model benchmarking: latency, throughput, memory, and comparison.

Provides hardware-aware benchmarking with warm-up, statistical analysis,
and multi-model comparison in a single call.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from py2edg._imports import is_available, require

logger = logging.getLogger("py2edg.benchmark")


@dataclass
class BenchmarkResult:
    """Results from benchmarking a model."""

    model_name: str
    model_path: str
    model_format: str
    model_size_mb: float

    # Latency (milliseconds)
    latency_mean_ms: float
    latency_median_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float

    # Throughput
    throughput_fps: float

    # Memory
    peak_memory_mb: float = 0.0

    # Config
    input_shape: Tuple = ()
    num_runs: int = 0
    warmup_runs: int = 0
    num_threads: int = 1

    # Raw timings for custom analysis
    raw_timings_ms: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"╔═══ Benchmark: {self.model_name} ═══",
            f"║ Format:     {self.model_format}",
            f"║ Size:       {self.model_size_mb:.2f} MB",
            f"║ Input:      {self.input_shape}",
            f"║ Runs:       {self.num_runs} (+ {self.warmup_runs} warmup)",
            f"╠═══ Latency ═══",
            f"║ Mean:       {self.latency_mean_ms:.2f} ms",
            f"║ Median:     {self.latency_median_ms:.2f} ms",
            f"║ P90:        {self.latency_p90_ms:.2f} ms",
            f"║ P95:        {self.latency_p95_ms:.2f} ms",
            f"║ P99:        {self.latency_p99_ms:.2f} ms",
            f"║ Min/Max:    {self.latency_min_ms:.2f} / {self.latency_max_ms:.2f} ms",
            f"║ Std:        {self.latency_std_ms:.2f} ms",
            f"╠═══ Throughput ═══",
            f"║ FPS:        {self.throughput_fps:.1f}",
        ]
        if self.peak_memory_mb > 0:
            lines.append(f"╠═══ Memory ═══")
            lines.append(f"║ Peak:       {self.peak_memory_mb:.2f} MB")
        lines.append(f"╚{'═' * 40}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_format": self.model_format,
            "model_size_mb": self.model_size_mb,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_median_ms": self.latency_median_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "throughput_fps": self.throughput_fps,
            "peak_memory_mb": self.peak_memory_mb,
            "input_shape": list(self.input_shape),
        }

    def __repr__(self):
        return (
            f"BenchmarkResult({self.model_name}: "
            f"{self.latency_mean_ms:.1f}ms, {self.throughput_fps:.0f}fps)"
        )


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB (Linux)
    except Exception:
        return 0.0


def benchmark_onnx(
    model_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_runs: int = 100,
    warmup_runs: int = 10,
    num_threads: int = 4,
    execution_provider: str = "CPUExecutionProvider",
) -> BenchmarkResult:
    """Benchmark an ONNX model with ONNX Runtime.

    Args:
        model_path: Path to .onnx model.
        input_shape: Input tensor shape.
        num_runs: Number of inference runs for timing.
        warmup_runs: Warm-up runs (excluded from stats).
        num_threads: Number of inference threads.
        execution_provider: ORT execution provider.

    Returns:
        BenchmarkResult with detailed statistics.
    """
    ort = require("onnxruntime", "onnxruntime")

    model_path = Path(model_path)
    model_size = model_path.stat().st_size / (1024 * 1024)

    # Configure session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = num_threads
    opts.intra_op_num_threads = num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [execution_provider]
    if execution_provider == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in ort.get_available_providers():
        logger.warning("CUDA not available, falling back to CPU")
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), opts, providers=providers)

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    gc.collect()
    mem_before = _get_memory_mb()
    timings = []

    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)

    mem_after = _get_memory_mb()
    timings_arr = np.array(timings)

    return BenchmarkResult(
        model_name=model_path.stem,
        model_path=str(model_path),
        model_format="onnx",
        model_size_mb=model_size,
        latency_mean_ms=float(np.mean(timings_arr)),
        latency_median_ms=float(np.median(timings_arr)),
        latency_p90_ms=float(np.percentile(timings_arr, 90)),
        latency_p95_ms=float(np.percentile(timings_arr, 95)),
        latency_p99_ms=float(np.percentile(timings_arr, 99)),
        latency_min_ms=float(np.min(timings_arr)),
        latency_max_ms=float(np.max(timings_arr)),
        latency_std_ms=float(np.std(timings_arr)),
        throughput_fps=1000.0 / float(np.mean(timings_arr)),
        peak_memory_mb=max(mem_after - mem_before, 0),
        input_shape=input_shape,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        num_threads=num_threads,
        raw_timings_ms=timings,
    )


def benchmark_pytorch(
    model: Any,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark a PyTorch model.

    Args:
        model: PyTorch nn.Module.
        input_shape: Input tensor shape.
        num_runs: Number of inference runs.
        warmup_runs: Warm-up iterations.
        device: 'cpu' or 'cuda'.

    Returns:
        BenchmarkResult.
    """
    torch = require("torch", "torch")

    model.eval()
    model.to(device)

    dummy = torch.randn(*input_shape, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            timings.append(elapsed)

    timings_arr = np.array(timings)

    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return BenchmarkResult(
        model_name=model.__class__.__name__,
        model_path="<in-memory>",
        model_format="pytorch",
        model_size_mb=param_size,
        latency_mean_ms=float(np.mean(timings_arr)),
        latency_median_ms=float(np.median(timings_arr)),
        latency_p90_ms=float(np.percentile(timings_arr, 90)),
        latency_p95_ms=float(np.percentile(timings_arr, 95)),
        latency_p99_ms=float(np.percentile(timings_arr, 99)),
        latency_min_ms=float(np.min(timings_arr)),
        latency_max_ms=float(np.max(timings_arr)),
        latency_std_ms=float(np.std(timings_arr)),
        throughput_fps=1000.0 / float(np.mean(timings_arr)),
        input_shape=input_shape,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        raw_timings_ms=timings,
    )


def compare_models(
    results: List[BenchmarkResult],
) -> str:
    """Generate a comparison table from multiple benchmark results.

    Args:
        results: List of BenchmarkResult objects to compare.

    Returns:
        Formatted comparison string.
    """
    if not results:
        return "No results to compare."

    baseline = results[0]

    lines = [
        "┌─────────────────────────────────────────────────────────────────┐",
        "│                    Model Comparison Report                     │",
        "├───────────────────┬──────────┬───────────┬─────────┬───────────┤",
        "│ Model             │ Size(MB) │ Mean(ms)  │   FPS   │ vs Base   │",
        "├───────────────────┼──────────┼───────────┼─────────┼───────────┤",
    ]

    for i, r in enumerate(results):
        name = r.model_name[:17]
        speedup = baseline.latency_mean_ms / r.latency_mean_ms if r.latency_mean_ms > 0 else 0
        vs = f"{speedup:.2f}x" if i > 0 else "baseline"
        lines.append(
            f"│ {name:<17} │ {r.model_size_mb:>8.2f} │ {r.latency_mean_ms:>9.2f} │ {r.throughput_fps:>7.1f} │ {vs:>9} │"
        )

    lines.append("└───────────────────┴──────────┴───────────┴─────────┴───────────┘")

    winner = min(results, key=lambda r: r.latency_mean_ms)
    smallest = min(results, key=lambda r: r.model_size_mb)

    lines.append(f"\n🏆 Fastest: {winner.model_name} ({winner.latency_mean_ms:.1f}ms)")
    lines.append(f"📦 Smallest: {smallest.model_name} ({smallest.model_size_mb:.1f}MB)")

    return "\n".join(lines)
