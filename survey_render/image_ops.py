from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .config import TargetBand


def read_image(path: Path, *, hdu: int | None = None, npz_key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fz"} or path.name.lower().endswith(".fits.gz"):
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            if hdu is not None:
                data = hdul[hdu].data
            else:
                data = None
                for item in hdul:
                    if item.data is not None and np.asarray(item.data).ndim >= 2:
                        data = item.data
                        break
        if data is None:
            raise ValueError(f"No image data found in FITS file: {path}")
        return ensure_2d(data)
    if suffix == ".npy":
        return ensure_2d(np.load(path))
    if suffix == ".npz":
        archive = np.load(path)
        key = npz_key or next(iter(archive.files))
        return ensure_2d(archive[key])
    raise ValueError(f"Unsupported input format: {path}")


def ensure_2d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {array.shape}")
    return np.asarray(array, dtype=np.float32)


def sanitize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if finite.all():
        return image
    fill = float(np.nanmedian(image[finite])) if finite.any() else 0.0
    return np.where(finite, image, fill).astype(np.float32)


def subtract_background(image: np.ndarray, mode: str = "border", border: int = 10) -> tuple[np.ndarray, float]:
    if mode == "none":
        return image.astype(np.float32, copy=False), 0.0
    if mode == "median":
        background = float(np.median(image))
    elif mode == "border":
        h, w = image.shape
        b = max(1, min(int(border), h // 2, w // 2))
        pixels = np.concatenate(
            [
                image[:b, :].ravel(),
                image[-b:, :].ravel(),
                image[:, :b].ravel(),
                image[:, -b:].ravel(),
            ]
        )
        background = float(np.median(pixels))
    else:
        raise ValueError(f"Unknown background subtraction mode: {mode}")
    return (image - background).astype(np.float32), background


def center_crop_or_pad(image: np.ndarray, size: int) -> np.ndarray:
    size = int(size)
    h, w = image.shape
    out = np.zeros((size, size), dtype=np.float32)
    src_y0 = max(0, (h - size) // 2)
    src_x0 = max(0, (w - size) // 2)
    src_y1 = min(h, src_y0 + size)
    src_x1 = min(w, src_x0 + size)
    crop = image[src_y0:src_y1, src_x0:src_x1]
    dst_y0 = max(0, (size - crop.shape[0]) // 2)
    dst_x0 = max(0, (size - crop.shape[1]) // 2)
    out[dst_y0 : dst_y0 + crop.shape[0], dst_x0 : dst_x0 + crop.shape[1]] = crop
    return out


def resample_flux_conserving(
    image: np.ndarray,
    *,
    source_pixel_scale: float,
    target_pixel_scale: float,
    output_size: int,
    order: int = 3,
) -> np.ndarray:
    if source_pixel_scale <= 0 or target_pixel_scale <= 0:
        raise ValueError("Pixel scales must be positive.")
    factor = float(source_pixel_scale) / float(target_pixel_scale)
    if not math.isclose(factor, 1.0):
        resampled = zoom(image, zoom=factor, order=order, mode="nearest", prefilter=(order > 1))
        resampled = resampled * (target_pixel_scale / source_pixel_scale) ** 2
    else:
        resampled = np.asarray(image, dtype=np.float32)
    return center_crop_or_pad(np.asarray(resampled, dtype=np.float32), output_size)


def convolve_to_target_psf(
    image: np.ndarray,
    *,
    hst_psf_fwhm: float,
    target_psf_fwhm: float,
    target_pixel_scale: float,
) -> np.ndarray:
    extra_fwhm = math.sqrt(max(float(target_psf_fwhm) ** 2 - float(hst_psf_fwhm) ** 2, 0.0))
    if extra_fwhm <= 0:
        return image.astype(np.float32, copy=False)
    sigma_pix = extra_fwhm / (2.354820045 * float(target_pixel_scale))
    if sigma_pix <= 0:
        return image.astype(np.float32, copy=False)
    return gaussian_filter(image, sigma=sigma_pix, mode="nearest").astype(np.float32)


def render_to_target(
    hst_image: np.ndarray,
    target: TargetBand,
    *,
    hst_pixel_scale: float,
    hst_psf_fwhm: float,
    output_size: int,
    interpolation_order: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resampled = resample_flux_conserving(
        hst_image,
        source_pixel_scale=hst_pixel_scale,
        target_pixel_scale=target.pixel_scale,
        output_size=output_size,
        order=interpolation_order,
    )
    clean = convolve_to_target_psf(
        resampled,
        hst_psf_fwhm=hst_psf_fwhm,
        target_psf_fwhm=target.psf_fwhm,
        target_pixel_scale=target.pixel_scale,
    )
    clean = clean * float(target.flux_scale) + float(target.background)
    if rng is None:
        rng = np.random.default_rng()
    if target.noise_sigma > 0:
        noise = rng.normal(0.0, float(target.noise_sigma), size=clean.shape).astype(np.float32)
        observed = clean + noise
    else:
        observed = clean.copy()
    return clean.astype(np.float32), observed.astype(np.float32)
