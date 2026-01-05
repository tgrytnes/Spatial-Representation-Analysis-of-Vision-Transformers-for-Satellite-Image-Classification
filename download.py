#!/usr/bin/env python3
import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# EuroSAT RGB URL
URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"{dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with open(dest_path, "wb") as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def extract_file(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # EuroSAT.zip extracts to a folder named '2750'
    extracted_folder = extract_to / "2750"
    target_folder = extract_to / "eurosat"

    if extracted_folder.exists():
        if target_folder.exists():
            print(f"Removing existing {target_folder}...")
            shutil.rmtree(target_folder)
        extracted_folder.rename(target_folder)
        print(f"Renamed {extracted_folder} to {target_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract EuroSAT dataset."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Directory to store data."
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.data_dir / "EuroSAT.zip"

    download_file(URL, zip_path)
    extract_file(zip_path, args.data_dir)
    print("Done!")


if __name__ == "__main__":
    main()
