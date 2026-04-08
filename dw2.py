import asyncio
import aiohttp
import aiofiles
import os
import json
import re
from pathlib import Path
from tqdm import tqdm

SPECIES_NAME = "Hibiscus rosa-sinensis"
TARGET_IMAGES = 1000
OUTPUT_DIR = Path("dataset/Hibiscus rosa-sinensis")
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"

API_URL = "https://api.inaturalist.org/v1/observations"

# API 查詢參數
PER_PAGE = 200
MAX_API_CONCURRENCY = 2

# 圖片下載參數
MAX_DOWNLOAD_CONCURRENCY = 16
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
RETRY_TIMES = 3

# 可選：只抓研究等級
QUALITY_GRADE = "research"

# 可選：只抓有照片的 observation
PHOTOS_ONLY = True


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def convert_photo_url(url: str, size: str = "large") -> str:
    return url.replace("/square.", f"/{size}.")


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict, semaphore: asyncio.Semaphore):
    async with semaphore:
        for attempt in range(RETRY_TIMES):
            try:
                async with session.get(url, params=params, timeout=REQUEST_TIMEOUT) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception:
                if attempt == RETRY_TIMES - 1:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))


async def collect_photo_records():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params_base = {
        "taxon_name": SPECIES_NAME,
        "per_page": PER_PAGE,
        "page": 1,
        "order_by": "created_at",
        "order": "desc",
    }

    if QUALITY_GRADE:
        params_base["quality_grade"] = QUALITY_GRADE
    if PHOTOS_ONLY:
        params_base["photos"] = "true"

    api_semaphore = asyncio.Semaphore(MAX_API_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=32, ssl=False)

    photo_records = []
    seen_urls = set()

    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "dataset-downloader/1.0"}) as session:
        page = 1
        total_results = None

        while len(photo_records) < TARGET_IMAGES:
            params = params_base.copy()
            params["page"] = page

            data = await fetch_json(session, API_URL, params, api_semaphore)

            if total_results is None:
                total_results = data.get("total_results", 0)

            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                obs_id = obs.get("id")
                species_guess = obs.get("species_guess")
                observed_on = obs.get("observed_on")
                quality_grade = obs.get("quality_grade")
                uri = obs.get("uri")
                license_code = obs.get("license_code")

                for photo in obs.get("photos", []):
                    photo_url = photo.get("url")
                    if not photo_url:
                        continue

                    large_url = convert_photo_url(photo_url, "large")
                    original_like_url = convert_photo_url(photo_url, "original")

                    if large_url in seen_urls:
                        continue
                    seen_urls.add(large_url)

                    record = {
                        "observation_id": obs_id,
                        "species_name": SPECIES_NAME,
                        "species_guess": species_guess,
                        "observed_on": observed_on,
                        "quality_grade": quality_grade,
                        "observation_uri": uri,
                        "observation_license_code": license_code,
                        "photo_id": photo.get("id"),
                        "photo_license_code": photo.get("license_code"),
                        "attribution": photo.get("attribution"),
                        "large_url": large_url,
                        "original_like_url": original_like_url,
                    }
                    photo_records.append(record)

                    if len(photo_records) >= TARGET_IMAGES:
                        break

                if len(photo_records) >= TARGET_IMAGES:
                    break

            print(f"已收集 {len(photo_records)} / {TARGET_IMAGES} 張，page={page}")
            page += 1

    return photo_records[:TARGET_IMAGES]


async def download_one(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, record: dict, index: int):
    async with semaphore:
        url_candidates = [record["original_like_url"], record["large_url"]]

        ext = ".jpg"
        filename = f"{index:04d}_obs{record['observation_id']}_photo{record['photo_id']}{ext}"
        filename = sanitize_filename(filename)
        file_path = OUTPUT_DIR / filename

        if file_path.exists() and file_path.stat().st_size > 0:
            return {"ok": True, "file": str(file_path), "record": record}

        last_error = None

        for url in url_candidates:
            for attempt in range(RETRY_TIMES):
                try:
                    async with session.get(url, timeout=REQUEST_TIMEOUT) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}")

                        content_type = resp.headers.get("Content-Type", "").lower()
                        if "png" in content_type:
                            final_path = file_path.with_suffix(".png")
                        elif "jpeg" in content_type or "jpg" in content_type:
                            final_path = file_path.with_suffix(".jpg")
                        elif "webp" in content_type:
                            final_path = file_path.with_suffix(".webp")
                        else:
                            final_path = file_path

                        async with aiofiles.open(final_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(1024 * 64):
                                await f.write(chunk)

                        return {"ok": True, "file": str(final_path), "record": record}

                except Exception as e:
                    last_error = e
                    if attempt < RETRY_TIMES - 1:
                        await asyncio.sleep(1.2 * (attempt + 1))

        return {"ok": False, "error": str(last_error), "record": record}


async def save_metadata_line(record: dict):
    async with aiofiles.open(METADATA_FILE, "a", encoding="utf-8") as f:
        await f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main():
    records = await collect_photo_records()
    print(f"準備下載 {len(records)} 張圖片")

    download_semaphore = asyncio.Semaphore(MAX_DOWNLOAD_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=MAX_DOWNLOAD_CONCURRENCY + 8, ssl=False)

    success = 0
    fail = 0

    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "dataset-downloader/1.0"}) as session:
        tasks = [download_one(session, download_semaphore, record, i + 1) for i, record in enumerate(records)]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="下載中"):
            result = await coro
            if result["ok"]:
                record = result["record"]
                record["saved_file"] = result["file"]
                await save_metadata_line(record)
                success += 1
            else:
                fail += 1

    print(f"完成：成功 {success} 張，失敗 {fail} 張")
    print(f"圖片資料夾：{OUTPUT_DIR}")
    print(f"Metadata：{METADATA_FILE}")

if __name__ == "__main__":
    asyncio.run(main())