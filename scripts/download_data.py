"""Download SciFact and BEIR SciFact into data/raw and verify both archives."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "scifact": (
        "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "data.tar.gz",
        "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
    ),
    "beir": (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "scifact.zip",
        "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    ),
}


def fetch(url: str, destination: Path, expected_sha256: str) -> None:
    if not destination.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(f"{destination}: sha256 {digest} does not match the pinned value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    for name, (url, filename, sha256) in SOURCES.items():
        directory = args.root / name
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / filename
        fetch(url, archive, sha256)
        if filename.endswith(".tar.gz"):
            with tarfile.open(archive) as tar:
                tar.extractall(directory, filter="data")
        else:
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(directory)
        print(f"{name}: ready in {directory}")


if __name__ == "__main__":
    main()
