#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="${ROOT:-data/synthetic_smoke}"
INPUT_DIR="${ROOT}/input"
OUT="${ROOT}/survey_render.zarr"
mkdir -p "${INPUT_DIR}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import numpy as np
from astropy.io import fits

out = Path("data/synthetic_smoke/input")
out.mkdir(parents=True, exist_ok=True)
y, x = np.mgrid[:160, :160]
for idx, amp in enumerate([0.12, 0.02, 0.0005]):
    image = amp * np.exp(-((x - 80) ** 2 + (y - 80) ** 2) / (2 * 8.0 ** 2))
    image += 0.001 * np.random.default_rng(idx).normal(size=image.shape)
    fits.writeto(out / f"hst_cutout_{idx:03d}.fits", image.astype("f4"), overwrite=True)
PY

"${PYTHON_BIN}" scripts/render_hst_to_surveys.py \
  --input-glob "${INPUT_DIR}/*.fits" \
  --out "${OUT}" \
  --overwrite \
  --target-size 83 \
  --min-snr 5 \
  --require-detected-in all \
  --write-clean
