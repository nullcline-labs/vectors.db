#!/usr/bin/env python3
"""Download ann-benchmarks HDF5 datasets and convert to .bin format.

Binary format:
  vectors: u32 count, u32 dim, then count*dim f32 values (little-endian)
  neighbors: u32 count, u32 k, then count*k i32 values (little-endian)

Usage:
  python3 benchmarks/convert_hdf5.py sift-128-euclidean
  python3 benchmarks/convert_hdf5.py glove-100-angular
"""

import sys
import os
import struct
import urllib.request
import numpy as np
import h5py

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_URL = "https://ann-benchmarks.com"


def download(name: str) -> str:
    """Download HDF5 file if not already present."""
    filename = f"{name}.hdf5"
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"  Already exists: {path}")
        return path
    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {url} ...")
    # Use custom opener with User-Agent to avoid 403
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"  Saved to {path}")
    return path


def convert(name: str, prefix: str):
    """Convert HDF5 to .bin files."""
    hdf5_path = download(name)

    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)

    count_train, dim = train.shape
    count_test, _ = test.shape
    _, k = neighbors.shape

    print(f"  Train: {count_train} x {dim}d")
    print(f"  Test:  {count_test} x {dim}d")
    print(f"  Neighbors: {count_test} x top-{k}")

    # Write train vectors
    train_path = os.path.join(DATA_DIR, f"{prefix}_train.bin")
    with open(train_path, "wb") as out:
        out.write(struct.pack("<II", count_train, dim))
        out.write(train.tobytes())
    print(f"  Wrote {train_path}")

    # Write test vectors
    test_path = os.path.join(DATA_DIR, f"{prefix}_test.bin")
    with open(test_path, "wb") as out:
        out.write(struct.pack("<II", count_test, dim))
        out.write(test.tobytes())
    print(f"  Wrote {test_path}")

    # Write ground truth neighbors
    neighbors_path = os.path.join(DATA_DIR, f"{prefix}_neighbors.bin")
    with open(neighbors_path, "wb") as out:
        out.write(struct.pack("<II", count_test, k))
        out.write(neighbors.tobytes())
    print(f"  Wrote {neighbors_path}")


if __name__ == "__main__":
    datasets = {
        "sift-128-euclidean": "sift128",
        "glove-100-angular": "glove100",
        "glove-25-angular": "glove25",
    }

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(datasets.keys())

    for name in names:
        prefix = datasets.get(name, name.replace("-", ""))
        print(f"\n=== {name} -> {prefix} ===")
        convert(name, prefix)

    print("\nDone.")
