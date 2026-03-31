<p align="center">
  <h1 align="center">🚀 Py2Edg</h1>
  <p align="center"><strong>One-line edge deployment for Computer Vision models</strong></p>
  <p align="center">
    Convert, quantize, optimize, and benchmark CV models for any edge device — in a single function call.
  </p>
  <p align="center">
    <em>Built by <a href="https://github.com/Rahimdzx">Mouissat Rabah Abderrahmane</a></em>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/py2edg/"><img src="https://img.shields.io/pypi/v/py2edg?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/py2edg/"><img src="https://img.shields.io/pypi/pyversions/py2edg" alt="Python"></a>
  <a href="https://github.com/Rahimdzx/py2edg/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

## The Problem

Deploying a CV model to edge devices is painful. You need to:
1. Export PyTorch → ONNX (handle dynamic axes, opsets, tracing issues)
2. Optimize the graph (constant folding, operator fusion, dead code elimination)
3. Quantize (FP16? INT8? Dynamic? Static with calibration?)
4. Convert to target format (TFLite? OpenVINO? CoreML?)
5. Benchmark (latency, throughput, memory)
6. Repeat for each device...

**Py2Edg does all of this in one line.**

## Quick Start

```bash
pip install py2edg
```

```python
import py2edg as rcv

# Deploy to Raspberry Pi 4 — converts, optimizes, quantizes, benchmarks
report = rcv.deploy("yolov8n.pt", device="rpi4")
```

That's it. Py2Edg automatically:
- Converts your PyTorch model to ONNX
- Optimizes the computation graph
- Quantizes to INT8 (optimal for RPi4)
- Converts to TFLite (best format for RPi4)
- Benchmarks and generates a deployment report

## Core API

### `rcv.deploy()` — Full Pipeline
```python
# Raspberry Pi 4
report = rcv.deploy("model.pt", device="rpi4")

# NVIDIA Jetson Nano
report = rcv.deploy("model.pt", device="jetson_nano")

# Orange Pi 5
report = rcv.deploy("model.pt", device="orange_pi")

# Custom configuration
report = rcv.deploy(
    "model.pt",
    device="rpi4",
    quantize="int8",
    input_shape=(1, 3, 320, 320),
    benchmark_runs=200,
)
report.print()
```

### `rcv.convert()` — Format Conversion
```python
# PyTorch → ONNX
rcv.convert("model.pt", target="onnx")

# ONNX with FP16 quantization
rcv.convert("model.pt", target="onnx", quantize="fp16")

# Auto-configure for device
rcv.convert("model.pt", device="jetson_nano")

# To TFLite with INT8
rcv.convert("model.onnx", target="tflite", quantize="int8")
```

### `rcv.benchmark()` — Performance Measurement
```python
stats = rcv.benchmark("model.onnx", input_shape=(1, 3, 640, 640))
print(f"Latency: {stats.latency_mean_ms:.1f} ms")
print(f"Throughput: {stats.throughput_fps:.0f} FPS")
print(f"P95: {stats.latency_p95_ms:.1f} ms")
```

### `rcv.compare()` — Side-by-Side Comparison
```python
rcv.compare(
    "model.onnx",
    "model_fp16.onnx",
    "model_int8.onnx",
    input_shape=(1, 3, 640, 640),
)
# ┌───────────────────┬──────────┬───────────┬─────────┬───────────┐
# │ Model             │ Size(MB) │ Mean(ms)  │   FPS   │ vs Base   │
# ├───────────────────┼──────────┼───────────┼─────────┼───────────┤
# │ model             │    25.30 │     18.42 │    54.3 │  baseline │
# │ model_fp16        │    12.70 │     12.15 │    82.3 │     1.52x │
# │ model_int8        │     6.80 │      8.91 │   112.2 │     2.07x │
# └───────────────────┴──────────┴───────────┴─────────┴───────────┘
```

### `rcv.validate()` — Output Accuracy Check
```python
result = rcv.validate(pytorch_model, "model_fp16.onnx")
print(f"Max diff: {result['max_diff']:.6f}")
print(f"Passed: {result['passed']}")
```

## Built-in Device Profiles

| Device | Format | Quantize | Notes |
|--------|--------|----------|-------|
| `rpi4` | TFLite | INT8 | Best for real-time on RPi4 |
| `rpi5` | TFLite | INT8 | 2x faster than RPi4 |
| `jetson_nano` | ONNX | FP16 | CUDA + TensorRT |
| `jetson_orin` | ONNX | FP16 | Extremely powerful |
| `orange_pi` | ONNX | INT8 | RK3588 NPU support |
| `coral_tpu` | TFLite | UINT8 | Edge TPU compiler |
| `android_cpu` | TFLite | INT8 | XNNPACK delegate |
| `android_gpu` | TFLite | FP16 | GPU delegate |
| `ios_coreml` | CoreML | FP16 | Apple Neural Engine |
| `server_gpu` | ONNX | FP16 | TensorRT provider |

### Custom Devices
```python
rcv.register_device(rcv.DeviceProfile(
    name="my_fpga",
    display_name="Custom FPGA Board",
    compute="npu",
    memory_mb=256,
    preferred_format="onnx",
    quantize="int8",
))
```

## Deployment Recipes

Save and share reproducible deployment configs:

```python
recipe = rcv.DeployRecipe(
    name="yolo-rpi4",
    model="yolov8n.pt",
    device="rpi4",
)
recipe.apply_device_defaults()
rcv.save_recipe(recipe, "deploy.yaml")

# Later, or on another machine:
recipe = rcv.load_recipe("deploy.yaml")
report = rcv.deploy(recipe=recipe)
```

## CLI

```bash
# Full deployment
py2edg deploy model.pt --device rpi4

# Convert only
py2edg convert model.pt --target onnx --quantize fp16

# Benchmark
py2edg benchmark model.onnx --runs 200

# Inspect model
py2edg inspect model.onnx

# List devices
py2edg devices

# Create recipe
py2edg recipe create --device jetson_nano --model model.pt -o deploy.yaml
```

## Architecture

```
py2edg/
├── api.py          # High-level one-liner API (convert, deploy, benchmark, ...)
├── converter.py    # Format conversion engine (PyTorch→ONNX→TFLite/OpenVINO)
├── quantizer.py    # Quantization (FP16, INT8 static/dynamic)
├── optimizer.py    # Graph optimization (fusion, pruning, simplification)
├── benchmark.py    # Speed/memory benchmarking with statistics
├── devices.py      # Edge device profiles (RPi, Jetson, Coral, mobile, ...)
├── recipe.py       # YAML-based deployment recipes
├── report.py       # Deployment report generation
├── cli.py          # Command-line interface
└── _imports.py     # Lazy optional dependency management
```

## Installation Options

```bash
pip install py2edg                    # Core (ONNX Runtime)
pip install py2edg[torch]             # + PyTorch support
pip install py2edg[tflite]            # + TFLite conversion
pip install py2edg[openvino]          # + OpenVINO support
pip install py2edg[full]              # Everything
```

## Contributing

```bash
git clone https://github.com/Rahimdzx/py2edg.git
cd py2edg
pip install -e ".[dev]"
pytest
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Mouissat Rabah Abderrahmane**
- AI & Robotics Engineer | MSc from Saint Petersburg State University
- GitHub: [@Rahimdzx](https://github.com/Rahimdzx)

---

<p align="center">Made with ❤️ 🇩🇿</p>
