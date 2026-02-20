import os
import time
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
DL_DIR = Path("data/raw/chbmit")

# List of subjects to download (all 24 subjects)
SUBJECTS = [f"chb{i:02d}" for i in range(1, 25)]

MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds between retries


def get_file_list(subject: str) -> list[str]:
    """Get list of .edf files for a subject from PhysioNet."""
    url = f"{BASE_URL}/{subject}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    files = []
    for line in resp.text.split('"'):
        if line.endswith(".edf"):
            files.append(line.split("/")[-1])
    return files


def get_remote_size(subject: str, filename: str) -> int | None:
    """Get the expected file size from the server via a HEAD request."""
    url = f"{BASE_URL}/{subject}/{filename}"
    try:
        resp = requests.head(url, timeout=30)
        resp.raise_for_status()
        size = resp.headers.get("content-length")
        return int(size) if size else None
    except requests.RequestException:
        return None


def download_file(subject: str, filename: str) -> None:
    """Download a single file from PhysioNet with progress bar and retries."""
    url = f"{BASE_URL}/{subject}/{filename}"
    dest = DL_DIR / subject / filename

    # If file exists, verify its size matches the remote size
    if dest.exists():
        remote_size = get_remote_size(subject, filename)
        local_size = dest.stat().st_size
        if remote_size and local_size == remote_size:
            print(f"  Skipping (verified): {filename} ({local_size / 1024 / 1024:.1f} MB)")
            return
        elif remote_size:
            print(f"  Re-downloading (incomplete): {filename} "
                  f"({local_size / 1024 / 1024:.1f} MB / {remote_size / 1024 / 1024:.1f} MB)")
            dest.unlink()
        else:
            # Can't verify size — re-download to be safe
            print(f"  Re-downloading (unverified): {filename}")
            dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            tmp_dest = dest.with_suffix(".edf.tmp")

            with (
                open(tmp_dest, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {filename}",
                    leave=True,
                ) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Verify downloaded size before renaming
            if total_size and tmp_dest.stat().st_size != total_size:
                tmp_dest.unlink()
                raise IOError(
                    f"Size mismatch: expected {total_size}, "
                    f"got {tmp_dest.stat().st_size}"
                )

            # Atomic rename: only move to final name if download is complete
            tmp_dest.rename(dest)
            return  # Success

        except (requests.RequestException, IOError, ConnectionError) as e:
            # Clean up partial temp file
            tmp_dest = dest.with_suffix(".edf.tmp")
            if tmp_dest.exists():
                tmp_dest.unlink()

            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"\n  ⚠ Attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  ✗ Failed after {MAX_RETRIES} attempts: {filename}")
                raise


def download_summary(subject: str) -> None:
    """Download the summary file for a subject."""
    summary_name = f"{subject}-summary.txt"
    url = f"{BASE_URL}/{subject}/{summary_name}"
    dest = DL_DIR / subject / summary_name

    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(resp.text)
    except requests.HTTPError:
        print(f"  Warning: No summary file for {subject}")


if __name__ == "__main__":
    print("Starting CHB-MIT download...")
    print(f"Saving to: {DL_DIR.resolve()}\n")

    for subject in SUBJECTS:
        print(f"[{subject}]")
        download_summary(subject)

        edf_files = get_file_list(subject)
        print(f"  Found {len(edf_files)} EDF files")

        for edf in edf_files:
            download_file(subject, edf)

        print()

    print("Download complete.")