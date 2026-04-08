from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "hw1-inaturalist-downloader/1.0"
VALID_SIZES = {"square", "small", "medium", "large", "original"}
SPECIES_FALLBACK = {
    "Millettia pinnata": "(1) 水黃皮 Millettia pinnata",
    "Ficus microcarpa": "(2) 正榕 Ficus microcarpa",
    "Podocarpus costalis": "(3) 蘭嶼羅漢松 Podocarpus costalis",
    "Hibiscus rosa-sinensis": "(4) 朱槿 Hibiscus rosa-sinensis",
    "Leucophyllum frutescens": "(5) 紅花玉芙蓉 Leucophyllum frutescens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download flower images from the iNaturalist API.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root folder containing the five class folders. Default: data",
    )
    parser.add_argument(
        "--per-species",
        type=int,
        default=1000,
        help="Maximum number of images to download per species. Default: 100",
    )
    parser.add_argument(
        "--size",
        choices=sorted(VALID_SIZES),
        default="large",
        help="Photo size variant from iNaturalist. Default: large",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay in seconds between image downloads. Default: 0.2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    species_targets = discover_species_targets(data_dir)
    if not species_targets:
        raise SystemExit("No class folders were found under the data directory.")

    for species_name, target_dir in species_targets:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {species_name} -> {target_dir}")
        downloaded = download_species_images(
            species_name=species_name,
            target_dir=target_dir,
            max_images=args.per_species,
            size=args.size,
            sleep_seconds=args.sleep,
        )
        print(f"Finished {species_name}: downloaded {downloaded} image(s).")


def discover_species_targets(data_dir: Path) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        species_name = extract_species_name(folder.name)
        if species_name:
            targets.append((species_name, folder))

    if targets:
        return targets

    return [(species, data_dir / folder_name) for species, folder_name in SPECIES_FALLBACK.items()]


def extract_species_name(folder_name: str) -> str | None:
    match = re.search(r"([A-Z][a-z-]+ [a-z-]+)$", folder_name)
    if match:
        return match.group(1)
    return None


def download_species_images(
    species_name: str,
    target_dir: Path,
    max_images: int,
    size: str,
    sleep_seconds: float,
) -> int:
    existing_count = count_existing_jpgs(target_dir)
    if existing_count >= max_images:
        print(f"Already have {existing_count} jpg/jpeg files, skipping.")
        return 0

    downloaded = 0
    page = 1
    seen_photo_ids: set[str] = set()
    target_total = max_images - existing_count

    while downloaded < target_total:
        payload = fetch_observations(species_name=species_name, page=page)
        results = payload.get("results", [])
        if not results:
            break

        for observation in results:
            for photo in iter_photos(observation):
                photo_id = str(photo.get("id", ""))
                if not photo_id or photo_id in seen_photo_ids:
                    continue
                seen_photo_ids.add(photo_id)

                image_url = build_image_url(photo, size=size)
                if not image_url:
                    continue

                file_path = target_dir / f"inat_{species_name.replace(' ', '_')}_{photo_id}.jpg"
                if file_path.exists():
                    continue

                try:
                    download_file(image_url, file_path)
                except Exception as exc:
                    print(f"  Failed to download photo {photo_id}: {exc}")
                    if file_path.exists():
                        file_path.unlink(missing_ok=True)
                    continue

                downloaded += 1
                print(f"  Downloaded {downloaded}/{target_total}: {file_path.name}")
                time.sleep(sleep_seconds)

                if downloaded >= target_total:
                    break
            if downloaded >= target_total:
                break

        page += 1

    return downloaded


def count_existing_jpgs(target_dir: Path) -> int:
    return sum(1 for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})


def fetch_observations(species_name: str, page: int) -> dict:
    params = {
        "taxon_name": species_name,
        "quality_grade": "research",
        "photos": "true",
        "per_page": 100,
        "page": page,
        "order_by": "votes",
        "order": "desc",
    }
    url = f"{API_BASE}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def iter_photos(observation: dict) -> Iterable[dict]:
    for photo in observation.get("photos", []):
        if isinstance(photo, dict):
            yield photo

    for observation_photo in observation.get("observation_photos", []):
        photo = observation_photo.get("photo", {})
        if isinstance(photo, dict):
            yield photo


def build_image_url(photo: dict, size: str) -> str | None:
    url = photo.get("url")
    if not url:
        return None
    for source_size in ("square", "small", "medium", "large", "original"):
        token = f"/{source_size}."
        if token in url:
            return url.replace(token, f"/{size}.")
    return url


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        data = response.read()
    destination.write_bytes(data)


if __name__ == "__main__":
    main()
