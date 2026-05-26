from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TargetBand


@dataclass(frozen=True)
class DetectionMetrics:
    flux: float
    peak: float
    npix: int
    snr: float
    peak_snr: float
    passed: bool


def compute_metrics(
    clean_image: np.ndarray,
    *,
    noise_sigma: float,
    min_flux: float,
    min_snr: float,
    min_peak_snr: float,
    signal_mask_fraction: float,
) -> DetectionMetrics:
    clean = np.asarray(clean_image, dtype=np.float32)
    peak = float(np.max(clean)) if clean.size else 0.0
    if peak > 0:
        threshold = max(0.0, peak * float(signal_mask_fraction))
        mask = clean > threshold
    else:
        mask = np.zeros(clean.shape, dtype=bool)
    npix = int(mask.sum())
    flux = float(clean[mask].sum()) if npix else 0.0
    if noise_sigma > 0 and npix > 0:
        snr = flux / (float(noise_sigma) * np.sqrt(npix))
        peak_snr = peak / float(noise_sigma)
    elif flux > 0:
        snr = float("inf")
        peak_snr = float("inf")
    else:
        snr = 0.0
        peak_snr = 0.0
    passed = flux >= min_flux and snr >= min_snr and peak_snr >= min_peak_snr
    return DetectionMetrics(
        flux=flux,
        peak=peak,
        npix=npix,
        snr=float(snr),
        peak_snr=float(peak_snr),
        passed=bool(passed),
    )


def normalize_target_key(value: str) -> str:
    instrument, band = value.split("/", 1)
    instrument = instrument.strip().lower()
    band = band.strip().lower() if instrument == "lsst" else band.strip().upper()
    return f"{instrument}/{band}"


def requirement_passed(
    passed_by_key: dict[str, bool],
    targets: list[TargetBand],
    requirement: str,
) -> tuple[bool, str]:
    requirement_raw = requirement.strip()
    requirement_mode = requirement_raw.lower()
    if requirement_mode == "any":
        ok = any(passed_by_key.values())
        return ok, "" if ok else "no_target_detected"
    if requirement_mode == "all":
        missing = [target.key for target in targets if not passed_by_key.get(target.key, False)]
        return not missing, "" if not missing else "faint:" + ";".join(missing)

    required = [item.strip() for item in requirement_raw.split(",") if item.strip()]
    if not required:
        return True, ""
    missing = []
    for item in required:
        if "/" in item:
            key = normalize_target_key(item)
            if not passed_by_key.get(key, False):
                missing.append(key)
        else:
            instrument = item.lower()
            instrument_targets = [target for target in targets if target.instrument == instrument]
            if not instrument_targets:
                missing.append(instrument)
            elif not any(passed_by_key.get(target.key, False) for target in instrument_targets):
                missing.append(instrument)
    return not missing, "" if not missing else "faint:" + ";".join(missing)
