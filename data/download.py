import os
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
DL_DIR = Path("data/raw/chbmit")

# List of subjects to download (all 24 subjects)
SUBJECTS = [f"chb{i:02d}" for i in range(1, 25)]


def get_file_list(subject: str) -> list[str]:
    """Get list of .edf files for a subject from PhysioNet."""
    url = f"{BASE_URL}/{subject}/"
    resp = requests.get(url)
    resp.raise_for_status()
    # Parse simple directory listing for .edf files
    files = []
    for line in resp.text.split('"'):
        if line.endswith(".edf"):
            files.append(line.split("/")[-1])
    return files


def download_file(subject: str, filename: str) -> None:
    """Download a single file from PhysioNet with a progress bar."""
    url = f"{BASE_URL}/{subject}/{filename}"
    dest = DL_DIR / subject / filename

    if dest.exists():
        print(f"  Skipping (exists): {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
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


def download_summary(subject: str) -> None:
    """Download the summary file for a subject."""
    summary_name = f"{subject}-summary.txt"
    url = f"{BASE_URL}/{subject}/{summary_name}"
    dest = DL_DIR / subject / summary_name

    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url)
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