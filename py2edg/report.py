"""Deployment report: comprehensive summary of the entire pipeline.

Aggregates conversion, quantization, optimization, and benchmark
results into a single, shareable report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from py2edg.benchmark import BenchmarkResult
from py2edg.devices import DeviceProfile
from py2edg.optimizer import OptimizationResult
from py2edg.quantizer import QuantizationResult

logger = logging.getLogger("py2edg.report")


@dataclass
class DeployReport:
    """Full deployment pipeline report.

    Contains all artifacts and measurements from a deploy() call.
    """

    # Identity
    model_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    py2edg_version: str = ""

    # Device
    device: Optional[DeviceProfile] = None
    target_format: str = ""

    # Pipeline results
    conversion_output: Optional[Path] = None
    optimization: Optional[OptimizationResult] = None
    quantization: Optional[QuantizationResult] = None
    benchmark_before: Optional[BenchmarkResult] = None
    benchmark_after: Optional[BenchmarkResult] = None

    # Paths
    output_dir: Optional[Path] = None
    artifacts: List[str] = field(default_factory=list)

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def speedup(self) -> float:
        """Speedup ratio (before/after benchmark)."""
        if self.benchmark_before and self.benchmark_after:
            if self.benchmark_after.latency_mean_ms > 0:
                return self.benchmark_before.latency_mean_ms / self.benchmark_after.latency_mean_ms
        return 1.0

    @property
    def compression(self) -> float:
        """Model size compression ratio."""
        if self.quantization:
            return self.quantization.compression_ratio
        return 1.0

    def summary(self) -> str:
        """Generate a human-readable deployment summary."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║              Py2Edg Deployment Report                     ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  Model:      {self.model_name}",
            f"  Target:     {self.target_format}",
            f"  Device:     {self.device.display_name if self.device else 'N/A'}",
            f"  Timestamp:  {self.timestamp}",
        ]

        if self.optimization:
            lines.extend([
                "",
                "  ── Optimization ──",
                f"  Ops:        {self.optimization.original_ops} → {self.optimization.optimized_ops} "
                f"(-{self.optimization.removed_ops})",
                f"  Applied:    {', '.join(self.optimization.optimizations_applied[:5])}",
            ])

        if self.quantization:
            lines.extend([
                "",
                "  ── Quantization ──",
                f"  Mode:       {self.quantization.mode}",
                f"  Size:       {self.quantization.original_size_mb:.1f} → "
                f"{self.quantization.quantized_size_mb:.1f} MB "
                f"({self.quantization.compression_ratio:.1f}x)",
            ])

        if self.benchmark_after:
            b = self.benchmark_after
            lines.extend([
                "",
                "  ── Performance ──",
                f"  Latency:    {b.latency_mean_ms:.1f} ms (P95: {b.latency_p95_ms:.1f} ms)",
                f"  Throughput: {b.throughput_fps:.1f} FPS",
            ])

        if self.benchmark_before and self.benchmark_after:
            lines.append(f"  Speedup:    {self.speedup:.2f}x vs original")

        if self.conversion_output:
            lines.extend([
                "",
                "  ── Output ──",
                f"  File:       {self.conversion_output}",
            ])

        if self.warnings:
            lines.extend(["", "  ⚠ Warnings:"])
            for w in self.warnings:
                lines.append(f"    - {w}")

        if self.device and self.device.notes:
            lines.extend(["", f"  💡 Tip: {self.device.notes}"])

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize report to dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "target_format": self.target_format,
            "device": self.device.to_dict() if self.device else None,
            "speedup": self.speedup,
            "compression": self.compression,
            "benchmark_after": self.benchmark_after.to_dict() if self.benchmark_after else None,
            "benchmark_before": self.benchmark_before.to_dict() if self.benchmark_before else None,
            "output": str(self.conversion_output) if self.conversion_output else None,
            "warnings": self.warnings,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {path}")
        return path

    def print(self) -> None:
        """Print the report summary."""
        print(self.summary())

    def __repr__(self):
        return (
            f"DeployReport({self.model_name}, "
            f"{self.speedup:.1f}x speedup, "
            f"{self.compression:.1f}x compression)"
        )
