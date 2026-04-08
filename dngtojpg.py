from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image


def convert_dng_files(input_dir: Path, output_dir: Path | None = None) -> int:
    dng_files = list(iter_dng_files(input_dir))
    if not dng_files:
        print(f"No DNG files found in: {input_dir}")
        return 0

    save_dir = output_dir or input_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    for dng_path in dng_files:
        target_path = save_dir / f"{dng_path.stem}.jpg"

        if is_jpeg_file(dng_path):
            shutil.copy2(dng_path, target_path)
        else:
            import rawpy

            with rawpy.imread(str(dng_path)) as raw:
                rgb = raw.postprocess()

            image = Image.fromarray(rgb)
            image.save(target_path, format="JPEG", quality=95)

        converted_count += 1
        print(f"Converted: {dng_path.name} -> {target_path.name}")

    return converted_count


def iter_dng_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.DNG"))
    yield from sorted(input_dir.glob("*.dng"))


def find_default_input_dir() -> Path:
    data_dir = Path("data")
    matches = sorted(
        path for path in data_dir.iterdir() if path.is_dir() and "紅花玉芙蓉" in path.name
    )
    if not matches:
        raise SystemExit(
            "Could not auto-detect the target folder. Please pass the input folder path explicitly."
        )
    return matches[0]


def is_jpeg_file(path: Path) -> bool:
    with path.open("rb") as file:
        header = file.read(3)
    return header == b"\xFF\xD8\xFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert DNG files to JPG.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Input folder. Default: auto-detect the folder containing `紅花玉芙蓉` under data/",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Defaults to the input folder.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else find_default_input_dir()
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    needs_rawpy = any(not is_jpeg_file(path) for path in iter_dng_files(input_dir))
    if needs_rawpy:
        try:
            import rawpy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard for runtime only
            raise SystemExit(
                "Missing dependency: install with `pip install rawpy pillow`"
            ) from exc

    converted_count = convert_dng_files(input_dir=input_dir, output_dir=output_dir)
    print(f"Done. Converted {converted_count} file(s).")


if __name__ == "__main__":
    main()
