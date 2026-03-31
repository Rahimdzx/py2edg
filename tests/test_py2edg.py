"""Tests for Py2Edg core modules (no heavy ML dependencies needed)."""

import sys
import tempfile
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import numpy as np
import py2edg as rcv
from py2edg.devices import DeviceProfile, get_device, list_devices, register_device
from py2edg.recipe import DeployRecipe, save_recipe, load_recipe
from py2edg.report import DeployReport
from py2edg.benchmark import BenchmarkResult, compare_models
from py2edg.converter import detect_format
from py2edg._imports import is_available


def test_version():
    assert rcv.__version__ == "0.1.0"
    assert rcv.__author__ == "Mouissat Rabah Abderrahmane"
    print("✅ Version and author")


def test_device_profiles():
    devices = list_devices()
    assert len(devices) >= 10
    names = [d.name for d in devices]
    assert "rpi4" in names
    assert "jetson_nano" in names
    assert "orange_pi" in names
    assert "coral_tpu" in names
    print(f"✅ {len(devices)} device profiles loaded")


def test_get_device():
    rpi4 = get_device("rpi4")
    assert rpi4.quantize == "int8"
    assert rpi4.preferred_format == "tflite"
    assert rpi4.compute == "cpu"
    assert rpi4.memory_mb == 2048

    jetson = get_device("jetson_nano")
    assert jetson.quantize == "fp16"
    assert jetson.compute == "gpu"

    orange = get_device("orange_pi")
    assert orange.compute == "npu"
    assert "rknn" in orange.extra.get("npu_runtime", "")

    print("✅ Device profiles correct")


def test_register_custom_device():
    custom = DeviceProfile(
        name="test_board",
        display_name="Test Board",
        compute="npu",
        memory_mb=256,
        quantize="int8",
    )
    register_device(custom)
    retrieved = get_device("test_board")
    assert retrieved.name == "test_board"
    assert retrieved.memory_mb == 256
    print("✅ Custom device registration")


def test_device_serialization():
    rpi4 = get_device("rpi4")
    d = rpi4.to_dict()
    restored = DeviceProfile.from_dict(d)
    assert restored.name == rpi4.name
    assert restored.quantize == rpi4.quantize
    print("✅ Device serialization")


def test_recipe_create():
    recipe = DeployRecipe(
        name="test-deploy",
        model="model.pt",
        device="rpi4",
    )
    recipe.apply_device_defaults()
    assert recipe.target == "tflite"
    assert recipe.quantize == "int8"
    assert recipe.input_shape == [1, 3, 320, 320]
    print("✅ Recipe creation with device defaults")


def test_recipe_save_load(tmp_path=None):
    if tmp_path is None:
        tmp_path = tempfile.gettempdir()
    recipe = DeployRecipe(
        name="test-recipe",
        model="yolov8n.pt",
        device="jetson_nano",
    )
    recipe.apply_device_defaults()

    path = f"{tmp_path}/test_recipe.yaml"
    save_recipe(recipe, path)
    loaded = load_recipe(path)

    assert loaded.name == recipe.name
    assert loaded.device == recipe.device
    assert loaded.quantize == recipe.quantize
    print("✅ Recipe save/load")


def test_benchmark_result():
    result = BenchmarkResult(
        model_name="test_model",
        model_path="test.onnx",
        model_format="onnx",
        model_size_mb=25.0,
        latency_mean_ms=15.0,
        latency_median_ms=14.5,
        latency_p90_ms=18.0,
        latency_p95_ms=20.0,
        latency_p99_ms=25.0,
        latency_min_ms=12.0,
        latency_max_ms=30.0,
        latency_std_ms=3.5,
        throughput_fps=66.7,
        input_shape=(1, 3, 640, 640),
        num_runs=100,
    )
    summary = result.summary()
    assert "test_model" in summary
    assert "15.00" in summary
    assert "66.7" in summary
    print("✅ BenchmarkResult summary")


def test_compare_models():
    results = [
        BenchmarkResult("model_fp32", "a.onnx", "onnx", 25.0, 18.0, 17.5, 22.0, 24.0, 28.0, 14.0, 30.0, 4.0, 55.6),
        BenchmarkResult("model_fp16", "b.onnx", "onnx", 12.5, 12.0, 11.5, 15.0, 16.0, 18.0, 10.0, 20.0, 2.5, 83.3),
        BenchmarkResult("model_int8", "c.onnx", "onnx", 6.5, 8.0, 7.8, 10.0, 11.0, 13.0, 6.0, 15.0, 2.0, 125.0),
    ]
    table = compare_models(results)
    assert "model_fp32" in table
    assert "model_int8" in table
    assert "Fastest" in table
    assert "Smallest" in table
    print("✅ Model comparison table")
    print(table)


def test_deploy_report():
    from py2edg.quantizer import QuantizationResult

    report = DeployReport(
        model_name="yolov8n",
        device=get_device("rpi4"),
        target_format="tflite",
    )
    report.quantization = QuantizationResult(
        output_path=None,
        mode="int8",
        original_size_mb=12.0,
        quantized_size_mb=3.5,
        compression_ratio=3.4,
    )
    report.benchmark_after = BenchmarkResult(
        "yolov8n_int8", "out.onnx", "onnx", 3.5,
        45.0, 44.0, 55.0, 58.0, 65.0, 40.0, 70.0, 5.0, 22.2,
    )

    summary = report.summary()
    assert "yolov8n" in summary
    assert "Raspberry Pi 4" in summary
    assert "int8" in summary
    print("✅ DeployReport generation")
    print(summary)


def test_detect_format():
    assert detect_format("model.pt") == "pytorch"
    assert detect_format("model.pth") == "pytorch"
    assert detect_format("model.onnx") == "onnx"
    assert detect_format("model.tflite") == "tflite"
    assert detect_format("model.h5") == "keras"
    print("✅ Format detection")


def test_imports_check():
    # These should work without error (lazy imports)
    from py2edg._imports import is_available
    assert isinstance(is_available("numpy"), bool)
    assert is_available("numpy") == True
    assert is_available("nonexistent_package_xyz") == False
    print("✅ Import checking")


if __name__ == "__main__":
    test_version()
    test_device_profiles()
    test_get_device()
    test_register_custom_device()
    test_device_serialization()
    test_recipe_create()
    test_recipe_save_load()
    test_benchmark_result()
    test_compare_models()
    test_deploy_report()
    test_detect_format()
    test_imports_check()
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
