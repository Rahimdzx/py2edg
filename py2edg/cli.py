"""Py2Edg command-line interface.

Usage:
    py2edg deploy model.pt --device rpi4
    py2edg convert model.pt --target onnx --quantize fp16
    py2edg benchmark model.onnx --input-shape 1,3,640,640
    py2edg inspect model.onnx
    py2edg devices
    py2edg recipe create --device rpi4 --model model.pt -o deploy.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="py2edg",
        description="🚀 Py2Edg — One-line edge deployment for CV models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── deploy ──
    deploy_p = subparsers.add_parser("deploy", help="Full deployment pipeline")
    deploy_p.add_argument("model", help="Path to model file")
    deploy_p.add_argument("--device", "-d", default="generic_arm", help="Target device (e.g., rpi4, jetson_nano)")
    deploy_p.add_argument("--target", "-t", help="Output format (onnx, tflite, openvino)")
    deploy_p.add_argument("--quantize", "-q", help="Quantization mode (fp32, fp16, int8, dynamic)")
    deploy_p.add_argument("--input-shape", default="1,3,640,640", help="Input shape (comma-separated)")
    deploy_p.add_argument("--output-dir", "-o", default="./deploy_output", help="Output directory")
    deploy_p.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    deploy_p.add_argument("--recipe", "-r", help="Path to recipe YAML")

    # ── convert ──
    convert_p = subparsers.add_parser("convert", help="Convert model format")
    convert_p.add_argument("model", help="Path to model file")
    convert_p.add_argument("--target", "-t", default="onnx", help="Target format")
    convert_p.add_argument("--quantize", "-q", default="fp32", help="Quantization mode")
    convert_p.add_argument("--output", "-o", help="Output file path")
    convert_p.add_argument("--input-shape", default="1,3,640,640", help="Input shape")
    convert_p.add_argument("--device", "-d", help="Device profile to use")

    # ── benchmark ──
    bench_p = subparsers.add_parser("benchmark", help="Benchmark a model")
    bench_p.add_argument("model", help="Path to ONNX model")
    bench_p.add_argument("--input-shape", default="1,3,640,640", help="Input shape")
    bench_p.add_argument("--runs", type=int, default=100, help="Number of runs")
    bench_p.add_argument("--threads", type=int, default=4, help="Number of threads")

    # ── inspect ──
    inspect_p = subparsers.add_parser("inspect", help="Inspect model metadata")
    inspect_p.add_argument("model", help="Path to model file")

    # ── devices ──
    subparsers.add_parser("devices", help="List available device profiles")

    # ── recipe ──
    recipe_p = subparsers.add_parser("recipe", help="Manage deployment recipes")
    recipe_sub = recipe_p.add_subparsers(dest="recipe_command")
    create_p = recipe_sub.add_parser("create", help="Create a new recipe")
    create_p.add_argument("--device", "-d", default="generic_arm")
    create_p.add_argument("--model", "-m", required=True)
    create_p.add_argument("--output", "-o", default="deploy.yaml")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    _parse_shape = lambda s: tuple(int(x) for x in s.split(","))

    if args.command == "deploy":
        from py2edg.api import deploy
        from py2edg.recipe import load_recipe

        recipe = load_recipe(args.recipe) if args.recipe else None
        deploy(
            model=args.model,
            output_dir=args.output_dir,
            device=args.device,
            target=args.target,
            quantize=args.quantize,
            input_shape=_parse_shape(args.input_shape),
            benchmark_runs=args.runs,
            recipe=recipe,
            verbose=True,
        )

    elif args.command == "convert":
        from py2edg.api import convert
        result = convert(
            model=args.model,
            output=args.output,
            target=args.target,
            quantize=args.quantize,
            input_shape=_parse_shape(args.input_shape),
            device=args.device,
        )
        print(f"✅ Saved: {result}")

    elif args.command == "benchmark":
        from py2edg.api import benchmark
        stats = benchmark(
            model=args.model,
            input_shape=_parse_shape(args.input_shape),
            num_runs=args.runs,
            num_threads=args.threads,
        )
        print(stats.summary())

    elif args.command == "inspect":
        from py2edg.api import inspect_model
        import json
        info = inspect_model(args.model)
        print(json.dumps(info, indent=2, default=str))

    elif args.command == "devices":
        from py2edg.devices import list_devices
        print("\n📱 Available Device Profiles:\n")
        for dev in list_devices():
            print(f"  {dev.name:<16} {dev.display_name:<40} {dev.quantize:<6} {dev.preferred_format}")
        print()

    elif args.command == "recipe":
        if args.recipe_command == "create":
            from py2edg.recipe import DeployRecipe, save_recipe
            recipe = DeployRecipe(
                name=Path(args.model).stem,
                model=args.model,
                device=args.device,
            )
            recipe.apply_device_defaults()
            save_recipe(recipe, args.output)
            print(f"✅ Recipe saved to {args.output}")


if __name__ == "__main__":
    main()
