from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image

try:
    from pillow_heif import register_heif_opener
except ImportError as exc:  # pragma: no cover - import guard for runtime only
    raise SystemExit(
        "Missing dependency: install with `pip install pillow pillow-heif`"
    ) from exc


register_heif_opener()


def convert_heic_files(
    input_dir: Path,
    output_format: str = "jpg",
    output_dir: Path | None = None,
) -> int:
    output_format = output_format.lower()
    heic_files = list(iter_heic_files(input_dir))
    if not heic_files:
        print(f"No HEIC files found in: {input_dir}")
        return 0

    save_dir = output_dir or input_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    for heic_path in heic_files:
        target_path = save_dir / f"{heic_path.stem}.{output_format}"

        with Image.open(heic_path) as image:
            save_format = "JPEG" if output_format in {"jpg", "jpeg"} else "PNG"
            converted = image.convert("RGB") if save_format == "JPEG" else image
            save_kwargs = {"quality": 95} if save_format == "JPEG" else {}
            converted.save(target_path, format=save_format, **save_kwargs)

        converted_count += 1
        print(f"Converted: {heic_path.name} -> {target_path.name}")

    return converted_count


def iter_heic_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.HEIC"))
    yield from sorted(input_dir.glob("*.heic"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert HEIC files to JPG or PNG.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Input folder. Default: auto-detect the folder containing `水黃皮` under data/",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "jpeg", "png"),
        default="jpg",
        help="Output format. Default: jpg",
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

    converted_count = convert_heic_files(
        input_dir=input_dir,
        output_format=args.format,
        output_dir=output_dir,
    )
    print(f"Done. Converted {converted_count} file(s).")


def find_default_input_dir() -> Path:
    data_dir = Path("data")
    matches = sorted(path for path in data_dir.iterdir() if path.is_dir() and "水黃皮" in path.name)
    if not matches:
        raise SystemExit(
            "Could not auto-detect the target folder. Please pass the input folder path explicitly."
        )
    return matches[0]


if __name__ == "__main__":
    main()
