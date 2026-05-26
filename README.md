# HST Anomaly Multi-Survey Generator

Standalone source-only generator for HST COSMOS-backed strong-lensing and false-positive simulations rendered into Euclid, LSST, and Roman image products.

The code uses SLSim for the lens/source populations and HST COSMOS catalog-source morphology support, then writes aligned multi-survey tensors and component products to zarr or HDF5.

## What It Generates

Entrypoints:

- `LRE_gg_lens.py`: positive strong-lens systems.
- `LRE_gg_nonlens.py`: false-positive / non-lens systems.
- `run_hst_100k_production.sh`: production wrapper for HST-enabled lens and non-lens generation.

Default HST production bands:

- Euclid: `VIS`
- LSST: `g,r,i,z,y` via `--lsst-bands all`
- Roman: `F106,F129,F158`

Each sample contains observed images, clean/no-PSF components, source/deflector components, noise realization, and lens-field maps such as kappa, deflection, shear, magnification, and potential.

## Attribution

This repository depends on and should credit SLSim:

- SLSim: https://github.com/LSST-strong-lensing/slsim
- Documentation: https://slsim.readthedocs.io
- License: MIT

SLSim originates from the LSST Strong Lensing Science Collaboration and LSST DESC ecosystem. If you use this generator for research, acknowledge/cite SLSim and the HST COSMOS source catalog data used for the morphology prior.

## Install

Create an environment with Python 3.10+ and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your local SLSim setup is custom, install that first instead of the GitHub dependency in `requirements.txt`.

## Required External Data

HST mode needs a COSMOS 23.5 training sample directory containing at least:

- `real_galaxy_catalog_23.5.fits`
- `real_galaxy_catalog_23.5_fits.fits`
- either `test_galaxy_images_23.5.fits` or `real_galaxy_images_23.5_n*.fits`

Set either:

```bash
export HST_COSMOS_PATH=/path/to/COSMOS_23.5_training_sample
```

or pass:

```bash
--hst-cosmos-path /path/to/COSMOS_23.5_training_sample
```

The Euclid VIS filter must be available to `speclite`. If it is not installed globally, set:

```bash
export EUCLID_VIS_FILTER_PATH=/path/to/Euclid-VIS.ecsv
```

or place `Euclid-VIS.ecsv` in the repository root.

## Smoke Run

```bash
export HST_COSMOS_PATH=/path/to/COSMOS_23.5_training_sample
export EUCLID_VIS_FILTER_PATH=/path/to/Euclid-VIS.ecsv  # only if needed
bash examples/run_smoke.sh
```

Manual lens-only example:

```bash
python LRE_gg_lens.py \
  --enable-hst \
  --num-runs 1 \
  --batch-size 1 \
  --num-workers 1 \
  --storage zarr \
  --dtype float16 \
  --euclid-bands VIS \
  --lsst-bands all \
  --roman-bands F106,F129,F158 \
  --num-pix 83 \
  --lens-field-num-pix 83 \
  --lens-sky-area 8 \
  --lens-sky-area-galaxies 8 \
  --data-root data/smoke/lens \
  --overwrite \
  --disable-jit
```

## Production Run

```bash
export HST_COSMOS_PATH=/path/to/COSMOS_23.5_training_sample
TARGET_PER_CLASS=100000 bash run_hst_100k_production.sh
```

Common overrides:

```bash
PYTHON_BIN=/path/to/python \
OUTPUT_ROOT=/path/to/output \
TARGET_PER_CLASS=100000 \
LENS_WORKERS=3 \
NONLENS_WORKERS=5 \
bash run_hst_100k_production.sh
```

## Output Layout

A zarr shard contains top-level arrays for sample IDs, labels, metadata, and per-product groups such as:

- `images/{euclid,lsst,roman}`
- `full_clean_psf/{euclid,lsst,roman}`
- `full_clean_nopsf/{euclid,lsst,roman}`
- `deflector_obs/{euclid,lsst,roman}`
- `source_lensed_obs/{euclid,lsst,roman}`
- `source_unlensed_clean_nopsf/{euclid,lsst,roman}`
- `lens_fields/{potential_map,lens_map,alpha_x,alpha_y,gamma1_map,gamma2_map,mu_map}`

Generated datasets are intentionally ignored by git.
