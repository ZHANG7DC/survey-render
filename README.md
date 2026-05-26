# Survey Render

Direct renderer for converting HST cutout images into survey-like Euclid, LSST, and Roman image tensors.

This repository is intentionally **not** a strong-lens population generator. It does not create deflectors, sources, lens fields, labels, or component maps. It takes existing HST image cutouts and applies survey-like degradation:

1. optional background subtraction,
2. flux-conserving pixel-scale resampling,
3. PSF broadening from the HST PSF to the target survey PSF,
4. optional per-band flux scaling,
5. target survey background/noise injection,
6. post-render brightness/SNR filtering.

## Outputs

The main output is a zarr group with:

- `id`: kept sample IDs,
- `source_path`: original HST file paths,
- `images/{euclid,lsst,roman}`: noisy survey-like renderings,
- `clean_images/{euclid,lsst,roman}`: optional pre-noise renderings when `--write-clean` is used.

The renderer also writes:

- `<out>.metadata.csv`: kept sample metrics,
- `<out>.rejected.csv`: rejected samples and reasons,
- `<out>.manifest.json`: run configuration and target survey settings.

## Filtering

Post-render filtering is enabled by default:

```bash
--min-snr 5 --require-detected-in all
```

A target band passes when the rendered clean image satisfies:

- integrated positive-mask flux >= `--min-flux`,
- integrated SNR >= `--min-snr`,
- peak SNR >= `--min-peak-snr`.

`--require-detected-in` can be:

- `all`: every selected target band must pass,
- `any`: at least one selected target band must pass,
- `euclid,lsst,roman`: at least one band in each listed survey must pass,
- explicit target keys such as `euclid/VIS,lsst/i,roman/F158`.

This is the step that removes HST cutouts that become too faint after rendering into the selected survey-like observations.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test

```bash
bash examples/run_synthetic_smoke.sh
```

## Render Real HST Cutouts

```bash
python scripts/render_hst_to_surveys.py \
  --input-glob "/path/to/hst_cutouts/*.fits" \
  --out data/hst_to_surveys.zarr \
  --overwrite \
  --hst-pixel-scale 0.05 \
  --hst-psf-fwhm 0.09 \
  --target-size 83 \
  --surveys euclid,lsst,roman \
  --euclid-bands VIS \
  --lsst-bands g,r,i,z,y \
  --roman-bands F106,F129,F158 \
  --min-snr 5 \
  --require-detected-in all
```

For single-band HST inputs, band colors are not physically inferred. Use `--flux-scale` to provide empirical or catalog-derived per-band scaling, for example:

```bash
--flux-scale euclid/VIS:1.0,lsst/g:0.7,lsst/r:0.9,lsst/i:1.0,roman/F158:1.1
```

Noise, PSF, and pixel-scale defaults are conservative approximations and should be overridden for a specific survey simulation campaign when calibrated values are available:

```bash
--noise-sigma euclid/VIS:0.008,lsst:0.02,roman:0.006
--psf-fwhm euclid/VIS:0.18,lsst/i:0.74,roman/F158:0.18
--pixel-scale euclid/VIS:0.101,lsst:0.2,roman:0.11
```

## Attribution

This renderer is source-only infrastructure for survey-like preprocessing. It does not vendor SLSim. The broader simulation work that motivated this repo uses and should credit SLSim, the LSST Strong Lensing Simulation pipeline:

- https://github.com/LSST-strong-lensing/slsim
- https://slsim.readthedocs.io

If you compare these rendered HST cutouts with SLSim-generated data, acknowledge SLSim and the original HST/COSMOS data products used as inputs.
