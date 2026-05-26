from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TargetBand:
    instrument: str
    band: str
    pixel_scale: float
    psf_fwhm: float
    noise_sigma: float
    background: float = 0.0
    flux_scale: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.instrument}/{self.band}"


DEFAULT_TARGETS: dict[str, TargetBand] = {
    "euclid/VIS": TargetBand("euclid", "VIS", pixel_scale=0.101, psf_fwhm=0.18, noise_sigma=0.010),
    "lsst/g": TargetBand("lsst", "g", pixel_scale=0.200, psf_fwhm=0.81, noise_sigma=0.020),
    "lsst/r": TargetBand("lsst", "r", pixel_scale=0.200, psf_fwhm=0.77, noise_sigma=0.018),
    "lsst/i": TargetBand("lsst", "i", pixel_scale=0.200, psf_fwhm=0.74, noise_sigma=0.017),
    "lsst/z": TargetBand("lsst", "z", pixel_scale=0.200, psf_fwhm=0.72, noise_sigma=0.019),
    "lsst/y": TargetBand("lsst", "y", pixel_scale=0.200, psf_fwhm=0.70, noise_sigma=0.022),
    "roman/F106": TargetBand("roman", "F106", pixel_scale=0.110, psf_fwhm=0.14, noise_sigma=0.008),
    "roman/F129": TargetBand("roman", "F129", pixel_scale=0.110, psf_fwhm=0.16, noise_sigma=0.008),
    "roman/F158": TargetBand("roman", "F158", pixel_scale=0.110, psf_fwhm=0.18, noise_sigma=0.009),
}

SUPPORTED_BANDS = {
    "euclid": ["VIS"],
    "lsst": ["g", "r", "i", "z", "y"],
    "roman": ["F106", "F129", "F158"],
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bands(value: str, instrument: str) -> list[str]:
    instrument = instrument.lower()
    if value.lower() == "all":
        return list(SUPPORTED_BANDS[instrument])
    bands = parse_csv(value)
    normalized = []
    for band in bands:
        normalized.append(band.lower() if instrument == "lsst" else band.upper())
    unknown = [band for band in normalized if band not in SUPPORTED_BANDS[instrument]]
    if unknown:
        raise ValueError(f"Unsupported {instrument} bands: {unknown}")
    return normalized


def select_targets(
    surveys: str,
    euclid_bands: str,
    lsst_bands: str,
    roman_bands: str,
) -> list[TargetBand]:
    selected_surveys = [survey.lower() for survey in parse_csv(surveys)]
    unknown = sorted(set(selected_surveys) - set(SUPPORTED_BANDS))
    if unknown:
        raise ValueError(f"Unsupported surveys: {unknown}")

    bands_by_survey = {
        "euclid": parse_bands(euclid_bands, "euclid"),
        "lsst": parse_bands(lsst_bands, "lsst"),
        "roman": parse_bands(roman_bands, "roman"),
    }
    targets = []
    for survey in selected_surveys:
        for band in bands_by_survey[survey]:
            key = f"{survey}/{band}"
            targets.append(DEFAULT_TARGETS[key])
    if not targets:
        raise ValueError("No target bands selected.")
    return targets


def parse_overrides(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    overrides = {}
    for item in parse_csv(value):
        if ":" not in item:
            raise ValueError(f"Expected key:value override, got {item!r}")
        key, raw = item.split(":", 1)
        overrides[key.strip()] = float(raw)
    return overrides


def value_for_target(overrides: dict[str, float], target: TargetBand, default: float) -> float:
    if target.key in overrides:
        return overrides[target.key]
    if target.instrument in overrides:
        return overrides[target.instrument]
    return default


def apply_overrides(
    targets: list[TargetBand],
    *,
    noise_sigma: str | None = None,
    background: str | None = None,
    flux_scale: str | None = None,
    psf_fwhm: str | None = None,
    pixel_scale: str | None = None,
) -> list[TargetBand]:
    noise = parse_overrides(noise_sigma)
    bg = parse_overrides(background)
    scale = parse_overrides(flux_scale)
    fwhm = parse_overrides(psf_fwhm)
    pix = parse_overrides(pixel_scale)
    out = []
    for target in targets:
        out.append(
            replace(
                target,
                noise_sigma=value_for_target(noise, target, target.noise_sigma),
                background=value_for_target(bg, target, target.background),
                flux_scale=value_for_target(scale, target, target.flux_scale),
                psf_fwhm=value_for_target(fwhm, target, target.psf_fwhm),
                pixel_scale=value_for_target(pix, target, target.pixel_scale),
            )
        )
    return out


def group_targets_by_instrument(targets: list[TargetBand]) -> dict[str, list[TargetBand]]:
    grouped: dict[str, list[TargetBand]] = {}
    for target in targets:
        grouped.setdefault(target.instrument, []).append(target)
    return grouped
