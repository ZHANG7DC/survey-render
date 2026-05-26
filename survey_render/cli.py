from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np

from .config import apply_overrides, select_targets
from .filters import compute_metrics, requirement_passed
from .image_ops import read_image, render_to_target, sanitize_image, subtract_background
from .writer import ZarrSurveyWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render HST cutout images into Euclid/LSST/Roman-like survey images."
    )
    parser.add_argument("--input-glob", action="append", default=[], help="Input glob. Repeatable.")
    parser.add_argument("--input-list", type=Path, help="Text file with one input path per line.")
    parser.add_argument("--out", type=Path, required=True, help="Output zarr path.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--input-hdu", type=int, default=None, help="FITS HDU. Defaults to first 2D HDU.")
    parser.add_argument("--npz-key", default=None)
    parser.add_argument("--surveys", default="euclid,lsst,roman")
    parser.add_argument("--euclid-bands", default="VIS")
    parser.add_argument("--lsst-bands", default="g,r,i,z,y")
    parser.add_argument("--roman-bands", default="F106,F129,F158")
    parser.add_argument("--hst-pixel-scale", type=float, default=0.05, help="HST input arcsec/pixel.")
    parser.add_argument("--hst-psf-fwhm", type=float, default=0.09, help="HST input PSF FWHM in arcsec.")
    parser.add_argument("--target-size", type=int, default=83)
    parser.add_argument("--interpolation-order", type=int, default=3, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--background-subtract", choices=["border", "median", "none"], default="border")
    parser.add_argument("--background-border", type=int, default=10)
    parser.add_argument("--noise-sigma", default=None, help="Overrides like euclid/VIS:0.01,lsst:0.02")
    parser.add_argument("--background", default=None, help="Overrides like euclid/VIS:0.0,roman:0.001")
    parser.add_argument("--flux-scale", default=None, help="Band flux scales like lsst/g:0.8,roman/F158:1.2")
    parser.add_argument("--psf-fwhm", default=None, help="Target PSF overrides in arcsec.")
    parser.add_argument("--pixel-scale", default=None, help="Target pixel-scale overrides in arcsec/pixel.")
    parser.add_argument("--min-flux", type=float, default=0.0)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--min-peak-snr", type=float, default=0.0)
    parser.add_argument("--signal-mask-fraction", type=float, default=0.02)
    parser.add_argument(
        "--require-detected-in",
        default="all",
        help="any, all, comma-separated instruments, or target keys like euclid/VIS,lsst/i.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--write-clean", action="store_true", help="Also store clean pre-noise survey images.")
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.input_glob:
        paths.extend(Path(item) for item in glob.glob(pattern))
    if args.input_list is not None:
        for line in args.input_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    unique.sort()
    if args.max_samples is not None:
        unique = unique[: int(args.max_samples)]
    if not unique:
        raise ValueError("No input files found. Use --input-glob or --input-list.")
    return unique


def metric_columns(targets) -> list[str]:
    columns = []
    for target in targets:
        prefix = target.key.replace("/", "_")
        columns.extend(
            [
                f"{prefix}_flux",
                f"{prefix}_peak",
                f"{prefix}_npix",
                f"{prefix}_snr",
                f"{prefix}_peak_snr",
                f"{prefix}_passed",
            ]
        )
    return columns


def metrics_to_row(metrics_by_key) -> dict[str, float | int]:
    row = {}
    for key, metrics in metrics_by_key.items():
        prefix = key.replace("/", "_")
        row[f"{prefix}_flux"] = metrics.flux
        row[f"{prefix}_peak"] = metrics.peak
        row[f"{prefix}_npix"] = metrics.npix
        row[f"{prefix}_snr"] = metrics.snr
        row[f"{prefix}_peak_snr"] = metrics.peak_snr
        row[f"{prefix}_passed"] = int(metrics.passed)
    return row


def json_safe_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = time.time()
    rng = np.random.default_rng(args.seed)
    targets = select_targets(args.surveys, args.euclid_bands, args.lsst_bands, args.roman_bands)
    targets = apply_overrides(
        targets,
        noise_sigma=args.noise_sigma,
        background=args.background,
        flux_scale=args.flux_scale,
        psf_fwhm=args.psf_fwhm,
        pixel_scale=args.pixel_scale,
    )
    inputs = collect_inputs(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out.with_suffix(".metadata.csv")
    rejected_path = args.out.with_suffix(".rejected.csv")
    manifest_path = args.out.with_suffix(".manifest.json")

    writer = ZarrSurveyWriter(
        args.out,
        targets,
        image_size=args.target_size,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        write_clean=args.write_clean,
        attrs={
            "renderer": "direct_hst_to_survey",
            "hst_pixel_scale": float(args.hst_pixel_scale),
            "hst_psf_fwhm": float(args.hst_psf_fwhm),
            "target_size": int(args.target_size),
            "require_detected_in": args.require_detected_in,
            "min_flux": float(args.min_flux),
            "min_snr": float(args.min_snr),
            "min_peak_snr": float(args.min_peak_snr),
        },
    )

    base_columns = ["sample_id", "source_path", "kept", "reject_reason", "background"]
    columns = base_columns + metric_columns(targets)
    kept = 0
    rejected = 0
    with metadata_path.open("w", newline="", encoding="utf-8") as meta_handle, rejected_path.open(
        "w", newline="", encoding="utf-8"
    ) as reject_handle:
        meta_writer = csv.DictWriter(meta_handle, fieldnames=columns)
        reject_writer = csv.DictWriter(reject_handle, fieldnames=columns)
        meta_writer.writeheader()
        reject_writer.writeheader()
        for index, path in enumerate(inputs):
            sample_id = path.stem
            try:
                image = read_image(path, hdu=args.input_hdu, npz_key=args.npz_key)
                image = sanitize_image(image)
                image, bg = subtract_background(
                    image, mode=args.background_subtract, border=args.background_border
                )
                clean_by_key = {}
                observed_by_key = {}
                metrics_by_key = {}
                passed_by_key = {}
                for target in targets:
                    clean, observed = render_to_target(
                        image,
                        target,
                        hst_pixel_scale=args.hst_pixel_scale,
                        hst_psf_fwhm=args.hst_psf_fwhm,
                        output_size=args.target_size,
                        interpolation_order=args.interpolation_order,
                        rng=rng,
                    )
                    metrics = compute_metrics(
                        clean,
                        noise_sigma=target.noise_sigma,
                        min_flux=args.min_flux,
                        min_snr=args.min_snr,
                        min_peak_snr=args.min_peak_snr,
                        signal_mask_fraction=args.signal_mask_fraction,
                    )
                    clean_by_key[target.key] = clean
                    observed_by_key[target.key] = observed
                    metrics_by_key[target.key] = metrics
                    passed_by_key[target.key] = metrics.passed
                sample_kept, reason = requirement_passed(passed_by_key, targets, args.require_detected_in)
            except Exception as exc:
                sample_kept = False
                reason = f"error:{type(exc).__name__}:{exc}"
                bg = float("nan")
                metrics_by_key = {}
                observed_by_key = {}
                clean_by_key = {}

            row = {
                "sample_id": sample_id,
                "source_path": str(path),
                "kept": int(sample_kept),
                "reject_reason": reason,
                "background": bg,
            }
            row.update(metrics_to_row(metrics_by_key))
            for column in columns:
                row.setdefault(column, "")
            if sample_kept:
                writer.append(
                    sample_id=sample_id,
                    source_path=str(path),
                    observed_by_key=observed_by_key,
                    clean_by_key=clean_by_key,
                )
                meta_writer.writerow(row)
                kept += 1
            else:
                reject_writer.writerow(row)
                rejected += 1
            if (index + 1) % 100 == 0:
                print(f"processed={index + 1} kept={kept} rejected={rejected}", flush=True)

    manifest = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "input_total": len(inputs),
        "kept": kept,
        "rejected": rejected,
        "output": str(args.out),
        "metadata_csv": str(metadata_path),
        "rejected_csv": str(rejected_path),
        "targets": [target.__dict__ for target in targets],
        "args": json_safe_args(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"done input={len(inputs)} kept={kept} rejected={rejected} output={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
