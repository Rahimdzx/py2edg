"""ONNX graph optimization: fusion, pruning, and simplification.

Optimizes model graphs for faster inference by:
  - Fusing common operator patterns (Conv+BN, MatMul+Add, etc.)
  - Removing redundant operations (identity, reshape chains)
  - Constant folding
  - Shape inference and propagation
  - Dead code elimination
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from py2edg._imports import is_available, require

logger = logging.getLogger("py2edg.optimizer")


@dataclass
class OptimizationResult:
    """Result of graph optimization."""

    output_path: Path
    original_ops: int
    optimized_ops: int
    removed_ops: int
    original_size_mb: float
    optimized_size_mb: float
    optimizations_applied: List[str]

    @property
    def ops_reduction_pct(self) -> float:
        if self.original_ops == 0:
            return 0.0
        return (self.removed_ops / self.original_ops) * 100

    def summary(self) -> str:
        return (
            f"Optimization Results:\n"
            f"  Ops:    {self.original_ops} → {self.optimized_ops} "
            f"(-{self.removed_ops}, {self.ops_reduction_pct:.1f}% reduction)\n"
            f"  Size:   {self.original_size_mb:.2f} → {self.optimized_size_mb:.2f} MB\n"
            f"  Applied: {', '.join(self.optimizations_applied)}"
        )


def count_ops(model_path: Union[str, Path]) -> int:
    """Count the number of operations in an ONNX model."""
    onnx_mod = require("onnx")
    model = onnx_mod.load(str(model_path))
    return len(model.graph.node)


def optimize_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    level: int = 3,
    simplify: bool = True,
    fold_constants: bool = True,
    fuse_ops: bool = True,
    eliminate_dead_code: bool = True,
    infer_shapes: bool = True,
) -> OptimizationResult:
    """Optimize an ONNX model graph.

    Args:
        model_path: Input ONNX model path.
        output_path: Output optimized model path.
        level: Optimization aggressiveness (1-3).
        simplify: Run onnxsim simplification.
        fold_constants: Fold constant expressions.
        fuse_ops: Fuse operator patterns.
        eliminate_dead_code: Remove unused operations.
        infer_shapes: Run shape inference.

    Returns:
        OptimizationResult with before/after statistics.
    """
    onnx_mod = require("onnx")

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx_mod.load(str(model_path))
    original_ops = len(model.graph.node)
    original_size = model_path.stat().st_size / (1024 * 1024)

    optimizations = []

    # 1. Shape inference
    if infer_shapes:
        try:
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
            optimizations.append("shape_inference")
            logger.info("Applied shape inference")
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}")

    # 2. ONNX optimizer passes
    if fuse_ops or eliminate_dead_code or fold_constants:
        if is_available("onnxoptimizer"):
            try:
                import onnxoptimizer

                passes = []
                if fold_constants:
                    passes.extend([
                        "eliminate_nop_dropout",
                        "eliminate_nop_pad",
                        "fuse_consecutive_squeezes",
                        "fuse_consecutive_transposes",
                        "eliminate_unused_initializer",
                    ])
                if fuse_ops and level >= 2:
                    passes.extend([
                        "fuse_add_bias_into_conv",
                        "fuse_bn_into_conv",
                        "fuse_matmul_add_bias_into_gemm",
                        "fuse_consecutive_concats",
                    ])
                if eliminate_dead_code:
                    passes.extend([
                        "eliminate_deadend",
                        "eliminate_identity",
                        "eliminate_nop_monotone_argmax",
                    ])

                model = onnxoptimizer.optimize(model, passes)
                optimizations.extend(passes)
                logger.info(f"Applied {len(passes)} onnxoptimizer passes")
            except Exception as e:
                logger.warning(f"onnxoptimizer failed: {e}")
        else:
            logger.info("onnxoptimizer not installed, skipping graph passes")

    # 3. ONNX Simplifier
    if simplify and is_available("onnxsim"):
        try:
            from onnxsim import simplify as onnx_simplify
            model, ok = onnx_simplify(model)
            if ok:
                optimizations.append("onnxsim")
                logger.info("Applied onnxsim simplification")
        except Exception as e:
            logger.warning(f"onnxsim failed: {e}")

    # 4. ORT optimization (most aggressive)
    if level >= 3 and is_available("onnxruntime"):
        try:
            import onnxruntime as ort

            # Save intermediate
            temp_path = output_path.with_suffix(".pre_ort.onnx")
            onnx_mod.save(model, str(temp_path))

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.optimized_model_filepath = str(output_path)

            # Run session to trigger optimization and save
            ort.InferenceSession(str(temp_path), opts, providers=["CPUExecutionProvider"])

            optimizations.append("ort_graph_optimization")
            logger.info("Applied ORT graph optimization")
            temp_path.unlink(missing_ok=True)

            # Reload the ORT-optimized model for final stats
            optimized_ops = count_ops(output_path)
            optimized_size = output_path.stat().st_size / (1024 * 1024)

            return OptimizationResult(
                output_path=output_path,
                original_ops=original_ops,
                optimized_ops=optimized_ops,
                removed_ops=original_ops - optimized_ops,
                original_size_mb=original_size,
                optimized_size_mb=optimized_size,
                optimizations_applied=optimizations,
            )
        except Exception as e:
            logger.warning(f"ORT optimization failed: {e}")

    # Save result
    onnx_mod.save(model, str(output_path))

    optimized_ops = len(model.graph.node)
    optimized_size = output_path.stat().st_size / (1024 * 1024)

    return OptimizationResult(
        output_path=output_path,
        original_ops=original_ops,
        optimized_ops=optimized_ops,
        removed_ops=original_ops - optimized_ops,
        original_size_mb=original_size,
        optimized_size_mb=optimized_size,
        optimizations_applied=optimizations,
    )


def inspect_onnx(model_path: Union[str, Path]) -> dict:
    """Inspect an ONNX model and return detailed metadata.

    Returns:
        Dictionary with model info: inputs, outputs, ops, size, etc.
    """
    onnx_mod = require("onnx")
    from collections import Counter

    model_path = Path(model_path)
    model = onnx_mod.load(str(model_path))

    # Count op types
    op_counts = Counter(node.op_type for node in model.graph.node)

    # Get inputs
    inputs = []
    for inp in model.graph.input:
        shape = []
        if inp.type.tensor_type.HasField("shape"):
            for dim in inp.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param or "?")
        inputs.append({"name": inp.name, "shape": shape})

    # Get outputs
    outputs = []
    for out in model.graph.output:
        shape = []
        if out.type.tensor_type.HasField("shape"):
            for dim in out.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param or "?")
        outputs.append({"name": out.name, "shape": shape})

    return {
        "path": str(model_path),
        "size_mb": model_path.stat().st_size / (1024 * 1024),
        "opset": model.opset_import[0].version if model.opset_import else None,
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "num_ops": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
        "inputs": inputs,
        "outputs": outputs,
        "op_types": dict(op_counts.most_common()),
    }
