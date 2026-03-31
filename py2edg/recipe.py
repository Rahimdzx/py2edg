"""Deployment recipes: reproducible, shareable deployment configurations.

A recipe encodes every decision needed to go from a trained model
to an edge-deployable artifact: target format, quantization, optimization
passes, input shapes, and device constraints.

Example recipe (YAML):
    name: yolov8n-rpi4
    model: yolov8n.pt
    device: rpi4
    target: tflite
    quantize: int8
    input_shape: [1, 3, 320, 320]
    optimize:
      level: 3
      simplify: true
    benchmark:
      num_runs: 50
      warmup: 10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from py2edg.devices import DeviceProfile, get_device

logger = logging.getLogger("py2edg.recipe")


@dataclass
class DeployRecipe:
    """A reproducible deployment configuration.

    Attributes:
        name: Recipe name / identifier.
        model: Path to source model file.
        device: Target device name (e.g., 'rpi4').
        target: Output format ('onnx', 'tflite', 'openvino').
        quantize: Quantization mode.
        input_shape: Model input shape.
        optimize_level: Graph optimization level (0-3).
        simplify: Run ONNX simplifier.
        opset: ONNX opset version.
        benchmark_runs: Number of benchmark iterations.
        benchmark_warmup: Number of warmup iterations.
        output_dir: Output directory for all artifacts.
        calibration_data_path: Path to calibration data (.npy).
        notes: Optional notes about this recipe.
        extra: Additional custom configuration.
    """

    name: str = "default"
    model: str = ""
    device: str = "generic_arm"
    target: str = "onnx"
    quantize: str = "fp16"
    input_shape: List[int] = field(default_factory=lambda: [1, 3, 640, 640])
    optimize_level: int = 3
    simplify: bool = True
    opset: int = 17
    benchmark_runs: int = 100
    benchmark_warmup: int = 10
    output_dir: str = "./deploy_output"
    calibration_data_path: Optional[str] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_device_profile(self) -> DeviceProfile:
        """Get the device profile for this recipe."""
        return get_device(self.device)

    def apply_device_defaults(self) -> None:
        """Override recipe settings with device profile recommendations."""
        try:
            profile = self.get_device_profile()
            self.target = profile.preferred_format
            self.quantize = profile.quantize
            self.opset = min(self.opset, profile.onnx_opset)
            if profile.input_sizes:
                self.input_shape = profile.input_sizes[0]
            logger.info(f"Applied device defaults from '{self.device}'")
        except KeyError:
            logger.warning(f"Device '{self.device}' not found, keeping current settings")

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "device": self.device,
            "target": self.target,
            "quantize": self.quantize,
            "input_shape": self.input_shape,
            "optimize": {
                "level": self.optimize_level,
                "simplify": self.simplify,
            },
            "opset": self.opset,
            "benchmark": {
                "num_runs": self.benchmark_runs,
                "warmup": self.benchmark_warmup,
            },
            "output_dir": self.output_dir,
            "calibration_data_path": self.calibration_data_path,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeployRecipe":
        """Deserialize from dictionary."""
        opt = data.get("optimize", {})
        bench = data.get("benchmark", {})

        return cls(
            name=data.get("name", "default"),
            model=data.get("model", ""),
            device=data.get("device", "generic_arm"),
            target=data.get("target", "onnx"),
            quantize=data.get("quantize", "fp16"),
            input_shape=data.get("input_shape", [1, 3, 640, 640]),
            optimize_level=opt.get("level", 3),
            simplify=opt.get("simplify", True),
            opset=data.get("opset", 17),
            benchmark_runs=bench.get("num_runs", 100),
            benchmark_warmup=bench.get("warmup", 10),
            output_dir=data.get("output_dir", "./deploy_output"),
            calibration_data_path=data.get("calibration_data_path"),
            notes=data.get("notes", ""),
            extra=data.get("extra", {}),
        )


def save_recipe(recipe: DeployRecipe, path: Union[str, Path]) -> Path:
    """Save a deployment recipe to a YAML file.

    Args:
        recipe: DeployRecipe to save.
        path: Output YAML file path.

    Returns:
        Path to saved file.

    Example:
        >>> recipe = rcv.DeployRecipe(name="yolo-rpi4", device="rpi4", model="best.pt")
        >>> rcv.save_recipe(recipe, "deploy.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(recipe.to_dict(), f, default_flow_style=False, sort_keys=False)
    logger.info(f"Recipe saved to {path}")
    return path


def load_recipe(path: Union[str, Path]) -> DeployRecipe:
    """Load a deployment recipe from a YAML file.

    Args:
        path: Path to YAML recipe file.

    Returns:
        DeployRecipe object.

    Example:
        >>> recipe = rcv.load_recipe("deploy.yaml")
        >>> report = rcv.deploy(recipe=recipe)
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    recipe = DeployRecipe.from_dict(data)
    logger.info(f"Recipe loaded from {path}")
    return recipe
