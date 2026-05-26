#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
from pathlib import Path

PACKAGES = [
    "numpy",
    "astropy",
    "speclite",
    "h5py",
    "zarr",
    "numcodecs",
    "lenstronomy",
    "slsim",
]


def main() -> int:
    missing = []
    for package in PACKAGES:
        try:
            importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - diagnostic script
            missing.append((package, exc))

    if missing:
        print("Missing or broken imports:")
        for package, exc in missing:
            print(f"  {package}: {type(exc).__name__}: {exc}")
        return 1

    hst_path = os.environ.get("HST_COSMOS_PATH")
    if hst_path:
        path = Path(hst_path).expanduser()
        print(f"HST_COSMOS_PATH={path} exists={path.exists()}")
    else:
        print("HST_COSMOS_PATH is not set")

    euclid_filter = os.environ.get("EUCLID_VIS_FILTER_PATH")
    if euclid_filter:
        path = Path(euclid_filter).expanduser()
        print(f"EUCLID_VIS_FILTER_PATH={path} exists={path.exists()}")
    else:
        print("EUCLID_VIS_FILTER_PATH is not set; speclite must already know Euclid-VIS")

    print("Environment import check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
