#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-data/smoke}"
COMMON_ARGS=(
  --enable-hst
  --num-runs 1
  --batch-size 1
  --num-workers 1
  --storage zarr
  --dtype float16
  --zarr-clevel 3
  --zarr-chunk-samples 1
  --write-batch 1
  --log-every 1
  --euclid-bands VIS
  --lsst-bands all
  --roman-bands F106,F129,F158
  --num-pix 83
  --lens-field-num-pix 83
  --field-galaxy-area-arcsec2 50
  --overwrite
  --disable-jit
)

if [[ -n "${HST_COSMOS_PATH:-}" ]]; then
  COMMON_ARGS+=(--hst-cosmos-path "${HST_COSMOS_PATH}")
fi

"${PYTHON_BIN}" LRE_gg_lens.py   "${COMMON_ARGS[@]}"   --lens-sky-area "${LENS_SKY_AREA:-8}"   --lens-sky-area-galaxies "${LENS_SKY_AREA_GALAXIES:-8}"   --data-root "${OUT_ROOT}/lens"

"${PYTHON_BIN}" LRE_gg_nonlens.py   "${COMMON_ARGS[@]}"   --nonlens-sky-area "${NONLENS_SKY_AREA:-8}"   --nonlens-sky-area-full "${NONLENS_SKY_AREA_FULL:-8}"   --data-root "${OUT_ROOT}/nonlens"
