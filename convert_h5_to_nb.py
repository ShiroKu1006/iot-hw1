from __future__ import annotations

import argparse
import importlib
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


DEFAULT_H5_PATH = Path("model") / "flower_cnn_final.h5"
DEFAULT_CALIBRATION_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("export_nb")
IMAGE_SIZE = (160, 160)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Keras .h5 model into quantized TFLite and, when available, VeriSilicon .nb output."
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5_PATH, help="Path to the input Keras .h5 model.")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory of images used as representative data for quantization.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for conversion outputs.")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=IMAGE_SIZE,
        help="Image size used for representative preprocessing.",
    )
    parser.add_argument(
        "--num-calibration-images",
        type=int,
        default=32,
        help="Maximum number of images to use for representative quantization data.",
    )
    parser.add_argument(
        "--viv-sdk",
        type=Path,
        default=None,
        help="Optional Vivante SDK root for Acuitylite export.",
    )
    parser.add_argument(
        "--licence",
        type=Path,
        default=None,
        help="Optional Acuitylite licence/device-target text file.",
    )
    return parser.parse_args()


def collect_image_paths(root_dir: Path, limit: int) -> list[Path]:
    if not root_dir.exists():
        raise SystemExit(f"Calibration directory not found: {root_dir}")

    image_paths: list[Path] = []
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(path)
        if len(image_paths) >= limit:
            break

    if not image_paths:
        raise SystemExit(f"No calibration images found under: {root_dir}")

    return image_paths


def load_image_as_float32(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(image_size)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Failed to load image: {image_path}") from exc

    return np.expand_dims(image_array, axis=0)


def representative_dataset(image_paths: list[Path], image_size: tuple[int, int]) -> Iterable[list[np.ndarray]]:
    for image_path in image_paths:
        yield [load_image_as_float32(image_path, image_size)]


def convert_h5_to_tflite(
    h5_path: Path,
    tflite_path: Path,
    calibration_images: list[Path],
    image_size: tuple[int, int],
) -> None:
    if not h5_path.exists():
        raise SystemExit(f"H5 model not found: {h5_path}")

    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_images, image_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def load_acuitylite_modules():
    importer = importlib.import_module("acuitylib.interface.importer")
    exporter = importlib.import_module("acuitylib.interface.exporter")
    return importer, exporter


def convert_tflite_to_nb(
    tflite_path: Path,
    output_prefix: Path,
    viv_sdk: Path | None,
    licence: Path | None,
) -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "NB export was skipped because Acuitylite officially documents Linux-only support "
            "(Ubuntu 20.04/22.04/24.04)."
        )

    try:
        importer, exporter = load_acuitylite_modules()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NB export was skipped because `acuitylite` / `acuitylib` is not installed.\n"
            "Official install path:\n"
            "  1. Use the Acuitylite Linux environment or Docker\n"
            "  2. `pip install acuitylite --no-deps`\n"
            "Then rerun this script on Linux."
        ) from exc

    TFLiteLoader = importer.TFLiteLoader
    OvxlibExporter = exporter.OvxlibExporter

    quant_model = TFLiteLoader(str(tflite_path)).load()
    export_kwargs: dict[str, object] = {"pack_nbg_only": True}

    if viv_sdk is not None:
        export_kwargs["viv_sdk"] = str(viv_sdk)
    if licence is not None:
        export_kwargs["licence"] = str(licence)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    OvxlibExporter(quant_model).export(str(output_prefix), **export_kwargs)


def main() -> None:
    args = parse_args()
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    calibration_images = collect_image_paths(args.calibration_dir, args.num_calibration_images)

    output_dir = args.output_dir
    tflite_path = output_dir / "flower_uint8.tflite"
    nb_prefix = output_dir / "nb" / "flower"

    print(f"Input H5: {args.h5}")
    print(f"Calibration dir: {args.calibration_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Image size: {image_size}")
    print(f"Calibration images used: {len(calibration_images)}")

    convert_h5_to_tflite(args.h5, tflite_path, calibration_images, image_size)
    print(f"Created quantized TFLite: {tflite_path}")

    try:
        convert_tflite_to_nb(tflite_path, nb_prefix, args.viv_sdk, args.licence)
    except SystemExit as exc:
        print(str(exc))
        print("NB export was not completed in this environment.")
        return

    print(f"NB export completed under prefix: {nb_prefix}")


if __name__ == "__main__":
    main()
