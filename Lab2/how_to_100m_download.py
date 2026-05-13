# ============================================================
# OPTIMIZED HOWTO100M HEALTH DOWNLOADER FOR VS CODE
# Goal: download at least 1000 real HEALTH-related .mp4 videos
#
# Features:
# - Health-only filtering
# - Parallel yt-dlp downloads
# - Hard timeout for each video
# - Saves successful downloads
# - Saves failed video_ids and skips them on next runs
# - Logs real yt-dlp errors to ytdlp_errors.log
# - Resumes after Ctrl+C or crash
# ============================================================

import os
import sys
import csv
import ssl
import shutil
import zipfile
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("howto100m_project")
META_DIR = BASE_DIR / "metadata"
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"

META_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DOWNLOADED_VIDEOS = 1000
MAX_CLASSES = 10

MAX_VIDEOS_PER_CLASS = 1000

START_METADATA_ROW = 0
MAX_METADATA_ROWS_TO_SCAN = 20_000_000

DOWNLOAD_WORKERS = 2
YTDLP_TIMEOUT_SECONDS = 60

YTDLP_FORMAT = "worst[ext=mp4]/worst"

HOWTO100M_URL = "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/HowTo100M.zip"

ZIP_PATH = META_DIR / "HowTo100M.zip"
CSV_PATH = META_DIR / "HowTo100M_v1.csv"
TASK_IDS_PATH = META_DIR / "task_ids.csv"

DOWNLOADED_RECORDS_CSV = OUTPUT_DIR / "downloaded_records.csv"
FAILED_VIDEO_IDS_TXT = OUTPUT_DIR / "failed_video_ids.txt"
LABEL_ENCODER_PKL = OUTPUT_DIR / "howto100m_label_encoder.pkl"
YTDLP_ERRORS_LOG = OUTPUT_DIR / "ytdlp_errors.log"


# ============================================================
# METADATA
# ============================================================

def is_valid_zip(path: Path) -> bool:
    if not path.exists():
        return False

    if path.stat().st_size < 1_000_000:
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception:
        return False


def download_file(url: str, output_path: Path):
    print(f"Downloading metadata archive:\n{url}\n-> {output_path}")

    try:
        import certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_context = ssl.create_default_context()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, context=ssl_context) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024

        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)

                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    print(f"\rProgress: {percent:6.2f}%", end="")
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\rDownloaded: {mb:.1f} MB", end="")

    print("\nMetadata archive downloaded.")


def prepare_metadata():
    print("\n========== METADATA ==========")

    if not is_valid_zip(ZIP_PATH):
        print("HowTo100M.zip is missing or corrupted.")

        if ZIP_PATH.exists():
            ZIP_PATH.unlink()

        try:
            download_file(HOWTO100M_URL, ZIP_PATH)
        except Exception as e:
            raise RuntimeError(
                "Could not download HowTo100M.zip automatically.\n"
                "Download it manually in browser:\n"
                f"{HOWTO100M_URL}\n\n"
                "Then put it here:\n"
                f"{META_DIR.resolve()}\n\n"
                f"Original error: {e}"
            )

    if not is_valid_zip(ZIP_PATH):
        raise RuntimeError("HowTo100M.zip exists, but it is not a valid ZIP.")

    print("Extracting needed metadata files...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()

        for needed in ["HowTo100M_v1.csv", "task_ids.csv"]:
            target_path = META_DIR / needed

            if target_path.exists() and target_path.stat().st_size > 0:
                print("Already extracted:", target_path)
                continue

            matches = [name for name in names if Path(name).name == needed]

            if not matches:
                raise RuntimeError(f"{needed} was not found inside ZIP archive.")

            with zf.open(matches[0]) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

            print("Extracted:", target_path)

    if not CSV_PATH.exists():
        raise RuntimeError("HowTo100M_v1.csv was not extracted.")

    if not TASK_IDS_PATH.exists():
        raise RuntimeError("task_ids.csv was not extracted.")


def load_task_descriptions():
    task_id_to_description = {}

    with open(TASK_IDS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) >= 2:
                task_id_to_description[str(row[0])] = str(row[1])

    print("Loaded task descriptions:", len(task_id_to_description))
    return task_id_to_description


# ============================================================
# HEALTH LABEL LOGIC
# ============================================================

health_keywords = [
    "exercise", "workout", "fitness", "gym",
    "yoga", "pilates", "cardio",
    "massage", "meditation",
    "stretch", "stretching", "warm up", "warm-up",
    "therapy", "rehab", "health",
    "body", "muscle", "strength",
    "weight loss", "abs", "back pain",
    "physical therapy",
    "running", "nutrition", "diet",
]


def is_health_text(text):
    text = str(text).lower()
    return any(keyword in text for keyword in health_keywords)


def make_label(text):
    text = str(text).lower()

    if "yoga" in text:
        return "yoga"

    elif "pilates" in text:
        return "pilates"

    elif "cardio" in text or "running" in text:
        return "cardio"

    elif "meditation" in text:
        return "meditation"

    elif "massage" in text:
        return "massage"

    elif (
        "stretch" in text
        or "stretching" in text
        or "warm up" in text
        or "warm-up" in text
    ):
        return "stretch"

    elif (
        "rehab" in text
        or "physical therapy" in text
        or "back pain" in text
    ):
        return "rehab"

    elif (
        "strength" in text
        or "muscle" in text
        or "abs" in text
    ):
        return "strength"

    elif (
        "nutrition" in text
        or "diet" in text
        or "weight loss" in text
    ):
        return "nutrition"

    elif (
        "workout" in text
        or "fitness" in text
        or "gym" in text
        or "exercise" in text
        or "health" in text
        or "body" in text
    ):
        return "fitness"

    else:
        return "fitness"


def get_allowed_labels():
    return [
        "yoga",
        "pilates",
        "cardio",
        "meditation",
        "massage",
        "stretch",
        "rehab",
        "strength",
        "nutrition",
        "fitness",
    ][:MAX_CLASSES]


# ============================================================
# SAVE / LOAD PROGRESS
# ============================================================

def save_records(records):
    df = pd.DataFrame(records)

    if len(df) > 0 and "label" in df.columns:
        label_encoder = LabelEncoder()
        df["label_id"] = label_encoder.fit_transform(df["label"])
        joblib.dump(label_encoder, LABEL_ENCODER_PKL)

    df.to_csv(DOWNLOADED_RECORDS_CSV, index=False)


def load_existing_records():
    records = []
    by_label = defaultdict(int)
    seen = set()

    if not DOWNLOADED_RECORDS_CSV.exists():
        return records, by_label, seen

    print("Existing downloaded_records.csv found. Loading...")

    df = pd.read_csv(DOWNLOADED_RECORDS_CSV)

    for _, row in df.iterrows():
        video_path = Path(str(row["video_path"]))

        if video_path.exists() and video_path.stat().st_size > 50_000:
            item = row.to_dict()

            if "label_id" in item:
                item.pop("label_id", None)

            records.append(item)
            by_label[str(item["label"])] += 1
            seen.add(str(item["video_id"]))

    print("Loaded valid existing videos:", len(records))
    print("Existing by class:", dict(by_label))

    return records, by_label, seen


def load_failed_video_ids():
    if not FAILED_VIDEO_IDS_TXT.exists():
        return set()

    with open(FAILED_VIDEO_IDS_TXT, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_failed_video_id(video_id):
    with open(FAILED_VIDEO_IDS_TXT, "a", encoding="utf-8") as f:
        f.write(str(video_id) + "\n")


def log_ytdlp_error(header, video_id, youtube_url, text=""):
    with open(YTDLP_ERRORS_LOG, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{header}\n")
        f.write(f"video_id={video_id}\n")
        f.write(f"url={youtube_url}\n")
        if text:
            f.write(text[:5000] + "\n")


# ============================================================
# DOWNLOAD LOGIC
# ============================================================

def download_worker(item):
    output_path = str(item["video_path"])
    video_id = str(item["video_id"])
    youtube_url = item["youtube_url"]

    if os.path.exists(output_path) and os.path.getsize(output_path) > 50_000:
        return item

    command = [
        sys.executable,
        "-m",
        "yt_dlp",

        "--no-warnings",
        "--ignore-errors",
        "--no-playlist",

        "--cookies",
        "cookies.txt",

        "--socket-timeout", "10",
        "--retries", "2",
        "--fragment-retries", "2",

        "-f",
        YTDLP_FORMAT,

        "--merge-output-format", "mp4",
        "-o",
        output_path,
        youtube_url,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=YTDLP_TIMEOUT_SECONDS,
        )

    except subprocess.TimeoutExpired:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass

        log_ytdlp_error(
            "TIMEOUT",
            video_id,
            youtube_url,
            f"Timeout after {YTDLP_TIMEOUT_SECONDS} seconds.",
        )

        return {"failed_video_id": video_id}

    except Exception as e:
        log_ytdlp_error(
            "EXCEPTION",
            video_id,
            youtube_url,
            str(e),
        )

        return {"failed_video_id": video_id}

    if os.path.exists(output_path) and os.path.getsize(output_path) > 50_000:
        return item

    error_text = result.stderr.strip()
    output_text = result.stdout.strip()

    log_text = f"returncode={result.returncode}\n"

    if error_text:
        log_text += "\nSTDERR:\n" + error_text + "\n"

    if output_text:
        log_text += "\nSTDOUT:\n" + output_text + "\n"

    log_ytdlp_error(
        "FAILED",
        video_id,
        youtube_url,
        log_text,
    )

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    return {"failed_video_id": video_id}


def handle_done_futures(done, pending, records, by_label, allowed_labels):
    new_successes = 0
    new_failures = 0

    for future in done:
        pending.remove(future)

        try:
            item = future.result()
        except Exception:
            item = None

        if item is None:
            continue

        if isinstance(item, dict) and "failed_video_id" in item:
            save_failed_video_id(item["failed_video_id"])
            new_failures += 1
            continue

        label = str(item["label"])

        if by_label[label] >= MAX_VIDEOS_PER_CLASS:
            path = Path(item["video_path"])
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
            continue

        if len(records) >= TARGET_DOWNLOADED_VIDEOS:
            path = Path(item["video_path"])
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
            continue

        records.append(item)
        by_label[label] += 1
        new_successes += 1

        if len(records) % 10 == 0:
            save_records(records)
            print("\nDownloaded:", len(records))
            print("Downloaded by class:", {label: by_label[label] for label in allowed_labels})

    if new_successes > 0:
        save_records(records)

    return new_successes, new_failures


def download_1000_videos():
    print("\n========== OPTIMIZED PARALLEL HEALTH DOWNLOAD ==========")

    if MAX_CLASSES * MAX_VIDEOS_PER_CLASS < TARGET_DOWNLOADED_VIDEOS:
        raise RuntimeError("Class settings are too small for target video count.")

    allowed_labels = get_allowed_labels()
    task_id_to_description = load_task_descriptions()

    records, by_label, seen = load_existing_records()

    failed_video_ids = load_failed_video_ids()
    seen.update(failed_video_ids)

    print("Loaded failed video ids:", len(failed_video_ids))

    if len(records) >= TARGET_DOWNLOADED_VIDEOS:
        print("Target already reached.")
        return records

    print("Target:", TARGET_DOWNLOADED_VIDEOS)
    print("Start metadata row:", START_METADATA_ROW)
    print("Max metadata rows to scan after start:", MAX_METADATA_ROWS_TO_SCAN)
    print("Stop metadata row:", START_METADATA_ROW + MAX_METADATA_ROWS_TO_SCAN)
    print("Workers:", DOWNLOAD_WORKERS)
    print("Timeout per video:", YTDLP_TIMEOUT_SECONDS)
    print("Format:", YTDLP_FORMAT)
    print("Errors log:", YTDLP_ERRORS_LOG.resolve())

    rows_scanned = 0
    health_candidates = 0
    submitted = 0
    total_failures = 0

    pending = set()
    max_pending = DOWNLOAD_WORKERS * 3

    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                total_for_progress = START_METADATA_ROW + MAX_METADATA_ROWS_TO_SCAN

                for row_idx, row in enumerate(tqdm(reader, total=total_for_progress)):
                    if row_idx < START_METADATA_ROW:
                        continue

                    rows_scanned += 1

                    if row_idx >= START_METADATA_ROW + MAX_METADATA_ROWS_TO_SCAN:
                        break

                    if rows_scanned % 10_000 == 0:
                        print(
                            f"\nCurrent metadata row: {row_idx}, "
                            f"rows scanned after start: {rows_scanned}, "
                            f"downloaded: {len(records)}, "
                            f"pending: {len(pending)}, "
                            f"health_candidates: {health_candidates}, "
                            f"submitted: {submitted}, "
                            f"failures this run: {total_failures}"
                        )

                    while len(pending) >= max_pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        _, failures = handle_done_futures(
                            done,
                            pending,
                            records,
                            by_label,
                            allowed_labels,
                        )
                        total_failures += failures

                        if len(records) >= TARGET_DOWNLOADED_VIDEOS:
                            break

                    if len(records) >= TARGET_DOWNLOADED_VIDEOS:
                        print("Target downloaded videos reached.")
                        break

                    video_id = str(row.get("video_id", "")).strip()

                    if not video_id or video_id in seen:
                        continue

                    category_1 = str(row.get("category_1", "")).strip()
                    category_2 = str(row.get("category_2", "")).strip()
                    task_id = str(row.get("task_id", "")).strip()
                    rank = str(row.get("rank", "")).strip()

                    task_description = task_id_to_description.get(task_id, "")

                    combined_text = " ".join([
                        category_1,
                        category_2,
                        task_description,
                    ])

                    if not is_health_text(combined_text):
                        continue

                    label = make_label(combined_text)

                    if label not in allowed_labels:
                        continue

                    if by_label[label] >= MAX_VIDEOS_PER_CLASS:
                        continue

                    seen.add(video_id)
                    health_candidates += 1

                    item = {
                        "video_id": video_id,
                        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                        "video_path": str(VIDEO_DIR / f"{video_id}.mp4"),
                        "category_1": category_1,
                        "category_2": category_2,
                        "task_id": task_id,
                        "task_description": task_description,
                        "rank": rank,
                        "label": label,
                    }

                    future = executor.submit(download_worker, item)
                    pending.add(future)
                    submitted += 1

                while pending and len(records) < TARGET_DOWNLOADED_VIDEOS:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    _, failures = handle_done_futures(
                        done,
                        pending,
                        records,
                        by_label,
                        allowed_labels,
                    )
                    total_failures += failures

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        save_records(records)
        raise

    save_records(records)

    print("\n========== FINAL DOWNLOAD SUMMARY ==========")
    print("Start metadata row:", START_METADATA_ROW)
    print("Rows scanned after start:", rows_scanned)
    print("Health candidates:", health_candidates)
    print("Submitted download attempts:", submitted)
    print("New failed attempts this run:", total_failures)
    print("Successfully downloaded:", len(records))
    print("Downloaded by class:")

    for label in allowed_labels:
        print(label, ":", by_label[label])

    print("\nSaved records:")
    print(DOWNLOADED_RECORDS_CSV.resolve())

    print("\nFailed IDs saved to:")
    print(FAILED_VIDEO_IDS_TXT.resolve())

    print("\nyt-dlp errors log:")
    print(YTDLP_ERRORS_LOG.resolve())

    print("\nVideos folder:")
    print(VIDEO_DIR.resolve())

    if len(records) < TARGET_DOWNLOADED_VIDEOS:
        raise RuntimeError(
            f"Only {len(records)} videos were downloaded, "
            f"but {TARGET_DOWNLOADED_VIDEOS} are required.\n\n"
            "Open ytdlp_errors.log to see the real reason.\n"
            "Possible fixes:\n"
            "1. Use a full-system VPN, not browser-only VPN.\n"
            "2. Update yt-dlp: python -m pip install --upgrade yt-dlp\n"
            "3. Delete failed_video_ids.txt if it was created with too small timeout.\n"
            "4. Increase MAX_METADATA_ROWS_TO_SCAN.\n"
            "5. Keep YTDLP_TIMEOUT_SECONDS around 60."
        )

    return records


def main():
    print("Base directory:", BASE_DIR.resolve())
    print("Metadata directory:", META_DIR.resolve())
    print("Videos directory:", VIDEO_DIR.resolve())
    print("Outputs directory:", OUTPUT_DIR.resolve())

    prepare_metadata()
    download_1000_videos()


if __name__ == "__main__":
    main()
