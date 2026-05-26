#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent

# Avoid known OpenMP / numba issues unless the user overrides them.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")


META_COLUMNS = [
    "sigma_v",
    "logMstar",
    "theta_E",
    "R_eff_s",
    "R_eff_l",
    "z_l",
    "z_s",
    "m_source",
    "m_lensed_source",
    "m_lens",
    "mu",
    "source_z",
    "deflector_z",
    "reff_s",
    "reff_d",
    "ls_mag",
    "e1_deflector_light",
    "e2_deflector_light",
    "q_deflector_light",
    "pa_deflector_light_deg",
    "e1_deflector_mass",
    "e2_deflector_mass",
    "q_deflector_mass",
    "pa_deflector_mass_deg",
    "e1_source",
    "e2_source",
    "q_source",
    "pa_source_deg",
    "kappa_ext",
    "gamma1_ext",
    "gamma2_ext",
    "gamma_ext",
    "pa_ext_deg",
]

OBSERVATORY_NAMES = {
    "euclid": "Euclid",
    "lsst": "LSST",
    "roman": "Roman",
}

SUPPORTED_BANDS = {
    "euclid": ["VIS"],
    "lsst": ["g", "r", "i", "z", "y"],
    "roman": ["F062", "F087", "F106", "F129", "F146", "F158", "F184"],
}

ROMAN_BAND_SURVEY_MODE = {
    "F062": "wide_area",
    "F106": "wide_area",
    "F129": "wide_area",
    "F158": "wide_area",
    "F184": "wide_area",
    "F087": "microlensing",
    "F146": "microlensing",
}

PIXEL_PRODUCT_GROUPS = [
    "images",
    "full_clean_psf",
    "full_clean_nopsf",
    "deflector_obs",
    "deflector_clean_psf",
    "deflector_clean_nopsf",
    "deflector_recon",
    "source_lensed_obs",
    "source_lensed_clean_psf",
    "source_lensed_clean_nopsf",
    "source_unlensed_clean_nopsf",
    "source_recon",
    "noise_realization",
]

LENS_FIELD_PRODUCTS = [
    "potential_map",
    "lens_map",
    "alpha_x",
    "alpha_y",
    "gamma1_map",
    "gamma2_map",
    "mu_map",
]

RECOMMENDED_CENTER_CROP_NUM_PIX_BY_INSTRUMENT = {
    "euclid": 83,
    "lsst": 42,
    "roman": 76,
}

NATIVE_PIXEL_SCALE_ARCSEC_BY_INSTRUMENT = {
    "euclid": 0.101,
    "lsst": 0.2,
    "roman": 0.11,
}

_HST_COSMOS_REQUIRED_FILES = (
    "real_galaxy_catalog_23.5.fits",
    "real_galaxy_catalog_23.5_fits.fits",
    "test_galaxy_images_23.5.fits",
)


@dataclass(frozen=True)
class BatchTask:
    run_id: int
    shard_index: int
    requested_count: int
    start_index: int
    output_path: str
    seed: int | None


def _parser_for_category(category: str) -> argparse.ArgumentParser:
    default_data_root = _REPO_ROOT / "data"
    parser = argparse.ArgumentParser(
        description=f"Generate {category} shards from the updated LRE gg notebooks."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of independent notebook-style runs (one shard per run).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of requested systems per run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Output root.",
    )
    parser.add_argument(
        "--storage",
        choices=["zarr", "hdf5"],
        default="zarr",
        help="Shard storage format.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression method (ignored for zarr).",
    )
    parser.add_argument(
        "--zarr-compressor",
        choices=["blosc", "none"],
        default="blosc",
        help="Zarr compressor.",
    )
    parser.add_argument(
        "--zarr-clevel",
        type=int,
        default=3,
        help="Zarr Blosc compression level (0-9).",
    )
    parser.add_argument(
        "--zarr-chunk-samples",
        type=int,
        default=64,
        help="Chunk size along the sample axis for zarr datasets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--write-batch",
        type=int,
        default=64,
        help="Buffered sample count before flushing to disk.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Progress logging cadence in written samples.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Optional base seed. Each run uses seed_base + run_id.",
    )
    parser.add_argument(
        "--run-id-offset",
        type=int,
        default=0,
        help="Offset added to generated run/shard ids.",
    )
    parser.add_argument(
        "--start-index-offset",
        type=int,
        default=0,
        help="Offset added to nonlens global sample indices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files.",
    )
    parser.add_argument(
        "--instruments",
        default="euclid,lsst,roman",
        help="Comma-separated instruments: euclid,lsst,roman.",
    )
    parser.add_argument(
        "--euclid-bands",
        default="VIS",
        help="Comma-separated Euclid bands or 'all'.",
    )
    parser.add_argument(
        "--lsst-bands",
        default="g,r,i",
        help="Comma-separated LSST bands or 'all'.",
    )
    parser.add_argument(
        "--roman-bands",
        default="F106,F129,F158",
        help="Comma-separated Roman bands or 'all'.",
    )
    parser.add_argument(
        "--num-pix",
        type=int,
        default=83,
        help="Uniform cutout size used for all instruments.",
    )
    parser.add_argument(
        "--lens-field-num-pix",
        type=int,
        default=None,
        help="Lens-field map size. Defaults to --num-pix.",
    )
    parser.add_argument(
        "--field-galaxy-area-arcsec2",
        type=float,
        default=50.0,
        help="Area used when drawing field galaxies to add to each system.",
    )
    parser.add_argument(
        "--hst-cosmos-path",
        type=Path,
        default=None,
        help="Path to the COSMOS_23.5_training_sample directory.",
    )
    hst_group = parser.add_mutually_exclusive_group()
    hst_group.add_argument(
        "--enable-hst",
        dest="use_hst",
        action="store_true",
        help="Use HST COSMOS catalog-source morphologies for source galaxies.",
    )
    hst_group.add_argument(
        "--disable-hst",
        dest="use_hst",
        action="store_false",
        help="Disable HST COSMOS source morphologies and fall back to analytic Sersic sources.",
    )
    parser.set_defaults(use_hst=True)
    parser.add_argument(
        "--max-draws",
        type=int,
        default=50,
        help="Maximum repeated lens-population draws per run before aborting.",
    )
    parser.add_argument(
        "--lens-sky-area",
        type=float,
        default=5.0,
        help="Lens population sky area in deg^2.",
    )
    parser.add_argument(
        "--lens-sky-area-galaxies",
        type=float,
        default=5.0,
        help="Lens galaxy catalog sky area in deg^2.",
    )
    parser.add_argument(
        "--nonlens-sky-area",
        type=float,
        default=5.0,
        help="Nonlens source population sky area in deg^2.",
    )
    parser.add_argument(
        "--nonlens-sky-area-full",
        type=float,
        default=5.0,
        help="Nonlens central galaxy population sky area in deg^2.",
    )
    parser.add_argument(
        "--mp-start",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="Multiprocessing start method.",
    )
    parser.add_argument(
        "--disable-jit",
        action="store_true",
        help="Disable numba JIT for stability.",
    )
    return parser


def _configure_env(disable_jit: bool) -> None:
    if disable_jit:
        os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _jaxtronomy_available() -> bool:
    return importlib.util.find_spec("jaxtronomy") is not None


def _repo_hst_cosmos_fallback_path() -> Path:
    return _REPO_ROOT / "data" / "COSMOS_23.5_training_sample"


def _euclid_filter_candidates() -> list[Path]:
    candidates = [
        _REPO_ROOT / "Euclid-VIS.ecsv",
    ]
    env_path = os.environ.get("EUCLID_VIS_FILTER_PATH")
    if env_path:
        candidates.insert(0, Path(env_path).expanduser())
    return candidates


def _configure_filters() -> None:
    import speclite.filters as speclite_filters
    from slsim.Pipelines.roman_speclite import configure_roman_filters, filter_names

    roman_filters = filter_names()
    if not all(Path(path).exists() for path in roman_filters):
        configure_roman_filters()
    speclite_filters.load_filters(*roman_filters)
    try:
        speclite_filters.load_filters("Euclid-VIS")
        return
    except Exception:
        pass

    for candidate in _euclid_filter_candidates():
        if candidate.exists():
            speclite_filters.load_filter(str(candidate))
            return

    raise RuntimeError(
        "Euclid-VIS filter not found. Set EUCLID_VIS_FILTER_PATH or place "
        "Euclid-VIS.ecsv at the repository root."
    )


def _is_valid_hst_cosmos_path(path: Path) -> bool:
    required_catalogs = (
        path / "real_galaxy_catalog_23.5.fits",
        path / "real_galaxy_catalog_23.5_fits.fits",
    )
    if not all(file_path.exists() for file_path in required_catalogs):
        return False
    if (path / "test_galaxy_images_23.5.fits").exists():
        return True
    return any(path.glob("real_galaxy_images_23.5_n*.fits"))


def _resolve_hst_cosmos_path(explicit_path: Path | None) -> Path:
    fallback = _repo_hst_cosmos_fallback_path()
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not _is_valid_hst_cosmos_path(path):
            raise FileNotFoundError(
                f"--hst-cosmos-path does not point to a valid COSMOS sample: {path}"
            )
        return path

    env_path = os.environ.get("HST_COSMOS_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not _is_valid_hst_cosmos_path(path):
            raise FileNotFoundError(
                f"HST_COSMOS_PATH does not point to a valid COSMOS sample: {path}"
            )
        return path

    if _is_valid_hst_cosmos_path(fallback):
        return fallback.resolve()

    raise FileNotFoundError(
        "No HST COSMOS catalog found. Pass --hst-cosmos-path or set HST_COSMOS_PATH."
    )


def parse_instruments(value: str) -> list[str]:
    allowed = {"euclid", "lsst", "roman"}
    instruments = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not instruments:
        raise ValueError("No instruments provided.")
    unknown = sorted(set(instruments) - allowed)
    if unknown:
        raise ValueError(f"Unknown instruments: {unknown}")
    return instruments


def _normalize_band_token(instrument: str, band: str) -> str:
    token = band.strip()
    if not token:
        return ""
    return token.lower() if instrument == "lsst" else token.upper()


def parse_bands(value: str, instrument: str) -> list[str]:
    supported = SUPPORTED_BANDS[instrument]
    if value.strip().lower() == "all":
        return list(supported)
    bands = [_normalize_band_token(instrument, token) for token in value.split(",")]
    bands = [band for band in bands if band]
    if not bands:
        raise ValueError(f"No bands provided for {instrument}.")
    unknown = sorted(set(bands) - set(supported))
    if unknown:
        raise ValueError(
            f"Unsupported {instrument} bands: {unknown}. Supported: {supported}"
        )
    return bands


def resolve_num_pix_config(
    args: argparse.Namespace, instruments: list[str]
) -> tuple[dict[str, int], int]:
    num_pix_by_instrument = {instrument: int(args.num_pix) for instrument in instruments}
    lens_field_num_pix = int(args.lens_field_num_pix or args.num_pix)
    return num_pix_by_instrument, lens_field_num_pix


def make_output_dirs(data_root: Path) -> dict[str, Path]:
    shards_root = data_root / "shards"
    lens_root = shards_root / "lens"
    nonlens_root = shards_root / "nonlens"
    lens_root.mkdir(parents=True, exist_ok=True)
    nonlens_root.mkdir(parents=True, exist_ok=True)
    return {
        "shards_root": shards_root,
        "lens_root": lens_root,
        "nonlens_root": nonlens_root,
    }


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m{rem:04.1f}s"


def _print_progress(done: int, total: int, start_time: float) -> None:
    width = 30
    ratio = 0 if total == 0 else done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = _format_seconds(time.time() - start_time)
    line = f"\r[{bar}] {done}/{total} ({ratio * 100:5.1f}%) elapsed {elapsed}"
    sys.stdout.write(line)
    sys.stdout.flush()


def _resource_snapshot(state: dict) -> str:
    now = time.time()
    proc_cpu = time.process_time()
    cpu_proc_pct = None
    if "proc_wall" in state and "proc_cpu" in state:
        d_wall = now - state["proc_wall"]
        d_cpu = proc_cpu - state["proc_cpu"]
        if d_wall > 0:
            cpu_proc_pct = 100.0 * d_cpu / d_wall
    state["proc_wall"] = now
    state["proc_cpu"] = proc_cpu

    rss_mb = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass

    if rss_mb is not None:
        state["peak_rss_mb"] = max(float(state.get("peak_rss_mb", 0.0)), rss_mb)

    proc_cpu_text = "n/a" if cpu_proc_pct is None else f"{cpu_proc_pct:.1f}%"
    rss_text = "n/a" if rss_mb is None else f"{rss_mb:.1f}MB"
    return f"cpu_proc={proc_cpu_text} rss={rss_text}"


def _resource_peak_text(state: dict) -> str:
    peak_rss = state.get("peak_rss_mb")
    if peak_rss is None:
        return "peak_rss=n/a"
    return f"peak_rss={float(peak_rss):.1f}MB"


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _instrument_kwargs(instrument: str, band: str) -> dict:
    if instrument != "roman":
        return {}
    survey_mode = ROMAN_BAND_SURVEY_MODE.get(band)
    if survey_mode is None:
        raise ValueError(f"Roman band {band} is not supported in observation config.")
    return {"survey_mode": survey_mode}


def _instrument_single_band_config(instrument: str, band: str) -> dict:
    from slsim.ImageSimulation import image_quality_lenstronomy

    kwargs = _instrument_kwargs(instrument, band)
    return image_quality_lenstronomy.kwargs_single_band(
        observatory=OBSERVATORY_NAMES[instrument], band=band, **kwargs
    )


def _band_config_payload(instruments, bands_by_instrument):
    key_order = [
        "pixel_scale",
        "magnitude_zero_point",
        "exposure_time",
        "seeing",
        "psf_type",
        "num_exposures",
        "background_noise",
        "read_noise",
        "sky_brightness",
        "ccd_gain",
    ]
    out = {}
    for instrument in instruments:
        rows = []
        for band in bands_by_instrument[instrument]:
            cfg = _instrument_single_band_config(instrument, band)
            row = {"band": band}
            for key in key_order:
                if key in cfg:
                    row[key] = _json_safe(cfg[key])
            rows.append(row)
        out[instrument] = rows
    return out


def _build_skypy_pipeline(cosmo, sky_area):
    import slsim
    import slsim.Pipelines as pipelines

    path = os.path.dirname(slsim.__file__)
    module_path, _ = os.path.split(path)
    skypy_config = os.path.join(module_path, "data/SkyPy/euclid-roman-lsst-like.yml")
    return pipelines.SkyPyPipeline(
        skypy_config=skypy_config, sky_area=sky_area, filters=None, cosmo=cosmo
    )


def _catalog_source_kwargs(config: dict) -> dict:
    if not config.get("use_hst", True):
        return {}
    kwargs = {
        "catalog_type": "HST_COSMOS",
        "catalog_path": config["hst_cosmos_path"],
    }
    if config.get("catalog_source_sersic_fallback", False):
        kwargs["sersic_fallback"] = True
    return kwargs


def _build_lens_context(config: dict) -> dict:
    from astropy.cosmology import FlatLambdaCDM
    from astropy.table import vstack
    from astropy.units import Quantity
    import slsim.Deflectors as deflectors
    from slsim.Lenses.lens_pop import LensPop
    import slsim.Sources as sources

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=config["lens_sky_area"], unit="deg2")
    sky_area_galaxies = Quantity(value=config["lens_sky_area_galaxies"], unit="deg2")
    kwargs_deflector_cut = {"band": "i", "band_max": 24, "z_min": 0.1, "z_max": 2}
    kwargs_source_cut = {"band": "i", "band_max": 26, "z_min": 0.1, "z_max": 5}

    pipeline = _build_skypy_pipeline(cosmo=cosmo, sky_area=sky_area_galaxies)
    red_gal = pipeline.red_galaxies
    blue_gal = pipeline.blue_galaxies
    all_galaxy_catalog = vstack([blue_gal, red_gal])

    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=red_gal,
        blue_galaxy_list=blue_gal,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=None,
        cosmo=cosmo,
        sky_area=sky_area_galaxies,
    )
    extended_source_type = "catalog_source" if config.get("use_hst", True) else "single_sersic"
    source_galaxies = sources.Galaxies(
        galaxy_list=blue_gal,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area_galaxies,
        catalog_type="skypy",
        source_size=None,
        extended_source_type=extended_source_type,
        extended_source_kwargs=_catalog_source_kwargs(config),
    )
    field_galaxy_pop = sources.Galaxies(
        galaxy_list=all_galaxy_catalog,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area_galaxies,
        catalog_type="skypy",
    )
    lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=bool(config["use_jaxtronomy"]),
    )
    kwargs_lens_cut = {
        "min_image_separation": 0.1,
        "max_image_separation": 8,
        "second_brightest_image_cut": {"i": 25},
    }
    return {
        "lens_pop": lens_pop,
        "field_galaxy_pop": field_galaxy_pop,
        "kwargs_lens_cut": kwargs_lens_cut,
    }


def _build_nonlens_context(config: dict) -> dict:
    from astropy.cosmology import FlatLambdaCDM
    from astropy.table import vstack
    from astropy.units import Quantity
    import slsim.Deflectors as deflectors
    from slsim.FalsePositives.false_positive_pop import FalsePositivePop
    import slsim.Sources as sources

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area_galaxy = Quantity(value=config["nonlens_sky_area"], unit="deg2")
    sky_area_full = Quantity(value=config["nonlens_sky_area_full"], unit="deg2")
    kwargs_deflector_cut = {"band": "i", "band_max": 24, "z_min": 0.01, "z_max": 2}
    kwargs_source_cut = {"band": "i", "band_max": 26, "z_min": 0.01, "z_max": 5.0}

    pipeline = _build_skypy_pipeline(cosmo=cosmo, sky_area=sky_area_galaxy)
    red_gal = pipeline.red_galaxies
    blue_gal = pipeline.blue_galaxies
    all_galaxy_catalog = vstack([red_gal, blue_gal])

    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=red_gal,
        blue_galaxy_list=blue_gal,
        gamma_pl={"mean": 2.0, "std_dev": 0.16},
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=None,
        cosmo=cosmo,
        sky_area=sky_area_full,
    )
    extended_source_type = "catalog_source" if config.get("use_hst", True) else "single_sersic"
    source_galaxies = sources.Galaxies(
        galaxy_list=blue_gal,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area_galaxy,
        catalog_type="skypy",
        source_size=None,
        extended_source_type=extended_source_type,
        extended_source_kwargs=_catalog_source_kwargs(config),
    )
    field_galaxy_pop = sources.Galaxies(
        galaxy_list=all_galaxy_catalog,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area_galaxy,
        catalog_type="skypy",
    )
    false_positive_pop = FalsePositivePop(
        central_galaxy_population=lens_galaxies,
        intruder_populations=source_galaxies,
        intruder_number_choices=[1, 2],
        cosmo=cosmo,
        include_central_galaxy_light=True,
        test_area_factor=1,
    )
    return {
        "false_positive_pop": false_positive_pop,
        "field_galaxy_pop": field_galaxy_pop,
    }


def _iter_lens_objects(lens_pop, kwargs_lens_cut, max_draws):
    draws = 0
    while draws < max_draws:
        population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_cut)
        draws += 1
        if not population:
            continue
        for item in population:
            yield item
    raise RuntimeError(
        "Not enough lens candidates; increase --lens-sky-area or --max-draws."
    )


def _normalize_population(population) -> list:
    if population is None:
        return []
    if isinstance(population, list):
        return population
    if isinstance(population, tuple):
        return list(population)
    return [population]


def _augment_with_field_galaxies(gg_lens, field_galaxy_pop, area_arcsec2: float) -> None:
    from astropy.units import Quantity

    field_galaxies = field_galaxy_pop.draw_galaxies(
        area=Quantity(area_arcsec2, "arcsec2")
    )
    gg_lens.add_field_galaxies(field_galaxies=field_galaxies)


def _meta_row(gg_lens):
    def _scalarize(value):
        array = np.asarray(value)
        if array.shape == ():
            return float(array)
        if array.size == 0:
            return float("nan")
        return _scalarize(array.reshape(-1)[0])

    def _e1e2_to_q_pa_deg(e1, e2):
        e1 = _scalarize(e1)
        e2 = _scalarize(e2)
        e = float(np.hypot(e1, e2))
        e = min(max(e, 0.0), 0.999999)
        q = (1.0 - e) / (1.0 + e)
        pa_deg = np.degrees(0.5 * np.arctan2(e2, e1))
        return q, pa_deg

    vel_disp = _scalarize(gg_lens.deflector_velocity_dispersion())
    m_star = _scalarize(gg_lens.deflector_stellar_mass())
    theta_e = _scalarize(gg_lens.einstein_radius)
    zl = _scalarize(gg_lens.deflector_redshift)
    zs = _scalarize(gg_lens.source_redshift_list)
    source_mag = _scalarize(gg_lens.extended_source_magnitude(band="g"))
    lensed_source_mag = _scalarize(
        gg_lens.extended_source_magnitude(band="g", lensed=True)
    )
    deflector_mag = _scalarize(gg_lens.deflector_magnitude(band="g"))
    reff = _scalarize(gg_lens._source[0].angular_size)
    reff_l = _scalarize(gg_lens.deflector.angular_size_light)
    magnification = _scalarize(gg_lens.extended_source_magnification)
    e1_light, e2_light, e1_mass, e2_mass = gg_lens.deflector_ellipticity()
    q_light, pa_light = _e1e2_to_q_pa_deg(e1_light, e2_light)
    q_mass, pa_mass = _e1e2_to_q_pa_deg(e1_mass, e2_mass)
    e1_source, e2_source = gg_lens._source[0].ellipticity
    q_source, pa_source = _e1e2_to_q_pa_deg(e1_source, e2_source)
    kappa_ext, gamma1_ext, gamma2_ext = gg_lens.los_linear_distortions
    e1_light = _scalarize(e1_light)
    e2_light = _scalarize(e2_light)
    e1_mass = _scalarize(e1_mass)
    e2_mass = _scalarize(e2_mass)
    e1_source = _scalarize(e1_source)
    e2_source = _scalarize(e2_source)
    kappa_ext = _scalarize(kappa_ext)
    gamma1_ext = _scalarize(gamma1_ext)
    gamma2_ext = _scalarize(gamma2_ext)
    gamma_ext = float(np.hypot(gamma1_ext, gamma2_ext))
    pa_ext = np.degrees(0.5 * np.arctan2(gamma2_ext, gamma1_ext))

    return np.array(
        [
            vel_disp,
            np.log10(m_star),
            theta_e,
            reff,
            reff_l,
            zl,
            zs,
            source_mag,
            lensed_source_mag,
            deflector_mag,
            magnification,
            zs,
            zl,
            reff,
            reff_l,
            lensed_source_mag,
            e1_light,
            e2_light,
            q_light,
            pa_light,
            e1_mass,
            e2_mass,
            q_mass,
            pa_mass,
            e1_source,
            e2_source,
            q_source,
            pa_source,
            kappa_ext,
            gamma1_ext,
            gamma2_ext,
            gamma_ext,
            pa_ext,
        ],
        dtype=np.float32,
    )


def _source_unlensed_map(kwargs_model, kwargs_source_amp, num_pix, delta_pix):
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Util import util

    source_model_list = kwargs_model.get("source_light_model_list", [])
    if not source_model_list or not kwargs_source_amp:
        return np.zeros((num_pix, num_pix), dtype=np.float32)
    light_model = LightModel(light_model_list=source_model_list)
    x, y = util.make_grid(numPix=num_pix, deltapix=delta_pix)
    image = light_model.surface_brightness(x, y, kwargs_source_amp)
    return image.reshape((num_pix, num_pix))


def _lens_field_pixel_scale(instruments, bands_by_instrument):
    for instrument in ("euclid", "lsst", "roman"):
        if instrument not in instruments:
            continue
        bands = bands_by_instrument.get(instrument, [])
        if not bands:
            continue
        cfg = _instrument_single_band_config(instrument, bands[0])
        return float(cfg["pixel_scale"])
    raise RuntimeError("No instrument/band available to set lens-field pixel scale.")


def _simulate_products(
    gg_lens,
    instruments,
    bands_by_instrument,
    num_pix_by_instrument,
    lens_field_num_pix,
    dtype,
):
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    from lenstronomy.Util import util

    products = {group_name: {} for group_name in PIXEL_PRODUCT_GROUPS}

    for instrument in instruments:
        num_pix = int(num_pix_by_instrument[instrument])
        band_products = {group_name: [] for group_name in PIXEL_PRODUCT_GROUPS}

        for band in bands_by_instrument[instrument]:
            kwargs_model, kwargs_params = gg_lens.lenstronomy_kwargs(band=band)
            obs_cfg = _instrument_single_band_config(instrument, band)
            sim_api = SimAPI(
                numpix=num_pix,
                kwargs_single_band=obs_cfg,
                kwargs_model=kwargs_model,
            )
            kwargs_lens_light, kwargs_source, kwargs_ps = sim_api.magnitude2amplitude(
                kwargs_lens_light_mag=kwargs_params.get("kwargs_lens_light", None),
                kwargs_source_mag=kwargs_params.get("kwargs_source", None),
                kwargs_ps_mag=kwargs_params.get("kwargs_ps", None),
            )
            image_model = sim_api.image_model_class(
                {"point_source_supersampling_factor": 1, "supersampling_factor": 3}
            )
            kwargs_lens = kwargs_params.get("kwargs_lens", None)

            full_clean_psf = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=kwargs_ps,
                unconvolved=False,
                source_add=True,
                lens_light_add=True,
                point_source_add=True,
            )
            deflector_clean_psf = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=kwargs_ps,
                unconvolved=False,
                source_add=False,
                lens_light_add=True,
                point_source_add=False,
            )
            source_lensed_clean_psf = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=kwargs_ps,
                unconvolved=False,
                source_add=True,
                lens_light_add=False,
                point_source_add=True,
            )
            full_clean_nopsf = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=kwargs_ps,
                unconvolved=True,
                source_add=True,
                lens_light_add=True,
                point_source_add=True,
            )
            deflector_clean_nopsf = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=kwargs_ps,
                unconvolved=True,
                source_add=False,
                lens_light_add=True,
                point_source_add=False,
            )
            source_lensed_clean_nopsf = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=kwargs_ps,
                unconvolved=True,
                source_add=True,
                lens_light_add=False,
                point_source_add=True,
            )

            noise_full = sim_api.noise_for_model(model=full_clean_psf)
            noise_deflector = sim_api.noise_for_model(model=deflector_clean_psf)
            noise_source = sim_api.noise_for_model(model=source_lensed_clean_psf)

            full_obs = full_clean_psf + noise_full
            deflector_obs = deflector_clean_psf + noise_deflector
            source_lensed_obs = source_lensed_clean_psf + noise_source
            noise_realization = full_obs - full_clean_psf
            source_unlensed = _source_unlensed_map(
                kwargs_model=kwargs_model,
                kwargs_source_amp=kwargs_source,
                num_pix=num_pix,
                delta_pix=float(obs_cfg["pixel_scale"]),
            )

            band_products["images"].append(np.asarray(full_obs, dtype=dtype))
            band_products["full_clean_psf"].append(np.asarray(full_clean_psf, dtype=dtype))
            band_products["full_clean_nopsf"].append(
                np.asarray(full_clean_nopsf, dtype=dtype)
            )
            band_products["deflector_obs"].append(np.asarray(deflector_obs, dtype=dtype))
            band_products["deflector_clean_psf"].append(
                np.asarray(deflector_clean_psf, dtype=dtype)
            )
            band_products["deflector_clean_nopsf"].append(
                np.asarray(deflector_clean_nopsf, dtype=dtype)
            )
            band_products["deflector_recon"].append(
                np.asarray(deflector_clean_nopsf, dtype=dtype)
            )
            band_products["source_lensed_obs"].append(
                np.asarray(source_lensed_obs, dtype=dtype)
            )
            band_products["source_lensed_clean_psf"].append(
                np.asarray(source_lensed_clean_psf, dtype=dtype)
            )
            band_products["source_lensed_clean_nopsf"].append(
                np.asarray(source_lensed_clean_nopsf, dtype=dtype)
            )
            band_products["source_unlensed_clean_nopsf"].append(
                np.asarray(source_unlensed, dtype=dtype)
            )
            band_products["source_recon"].append(np.asarray(source_unlensed, dtype=dtype))
            band_products["noise_realization"].append(
                np.asarray(noise_realization, dtype=dtype)
            )

        for group_name in PIXEL_PRODUCT_GROUPS:
            products[group_name][instrument] = np.stack(
                band_products[group_name], axis=-1
            )

    lens_model, kwargs_lens = gg_lens.deflector_mass_model_lenstronomy(source_index=0)
    lens_pixel_scale = _lens_field_pixel_scale(instruments, bands_by_instrument)
    x, y = util.make_grid(numPix=lens_field_num_pix, deltapix=lens_pixel_scale)

    potential = lens_model.potential(x, y, kwargs_lens)
    kappa = lens_model.kappa(x, y, kwargs_lens)
    alpha_x, alpha_y = lens_model.alpha(x, y, kwargs_lens)
    gamma1, gamma2 = lens_model.gamma(x, y, kwargs_lens)
    det_a = (1.0 - kappa) ** 2 - gamma1**2 - gamma2**2
    eps = 1e-6
    det_a_safe = np.where(np.abs(det_a) < eps, np.where(det_a >= 0, eps, -eps), det_a)
    mu = 1.0 / det_a_safe

    products["potential_map"] = np.asarray(
        potential.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    products["lens_map"] = np.asarray(
        kappa.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    products["alpha_x"] = np.asarray(
        alpha_x.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    products["alpha_y"] = np.asarray(
        alpha_y.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    products["gamma1_map"] = np.asarray(
        gamma1.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    products["gamma2_map"] = np.asarray(
        gamma2.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    products["mu_map"] = np.asarray(
        mu.reshape((lens_field_num_pix, lens_field_num_pix)), dtype=dtype
    )
    return products


def _all_finite_products(products, instruments):
    for group_name in PIXEL_PRODUCT_GROUPS:
        for instrument in instruments:
            if not np.isfinite(products[group_name][instrument]).all():
                return False
    for field_name in LENS_FIELD_PRODUCTS:
        if not np.isfinite(products[field_name]).all():
            return False
    return True


def _prepare_output_path(output_path: str, overwrite: bool) -> None:
    path = Path(output_path)
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _store_common_attrs(
    attrs,
    *,
    count,
    bands_by_instrument,
    band_config_payload,
    num_pix_by_instrument,
    lens_field_num_pix,
    config_json,
    config,
) -> None:
    attrs["config"] = config_json
    attrs["count"] = int(count)
    attrs["written"] = 0
    attrs["bands"] = json.dumps(bands_by_instrument, sort_keys=True)
    attrs["band_config"] = json.dumps(band_config_payload, sort_keys=True)
    attrs["num_pix_by_instrument"] = json.dumps(num_pix_by_instrument, sort_keys=True)
    attrs["lens_field_num_pix"] = int(lens_field_num_pix)
    attrs["recommended_center_crop_num_pix_by_instrument"] = json.dumps(
        config["recommended_center_crop_num_pix_by_instrument"], sort_keys=True
    )
    attrs["native_pixel_scale_arcsec_by_instrument"] = json.dumps(
        config["native_pixel_scale_arcsec_by_instrument"], sort_keys=True
    )
    attrs["field_galaxy_area_arcsec2"] = float(config["field_galaxy_area_arcsec2"])
    attrs["hst_cosmos_path"] = str(config["hst_cosmos_path"])


def _instrument_dataset_attrs(dataset, instrument: str, bands, band_config_payload, config):
    dataset.attrs["bands"] = list(bands)
    dataset.attrs["band_config"] = json.dumps(
        band_config_payload[instrument], sort_keys=True
    )
    dataset.attrs["recommended_center_crop_num_pix"] = int(
        config["recommended_center_crop_num_pix_by_instrument"][instrument]
    )
    dataset.attrs["native_pixel_scale_arcsec"] = float(
        config["native_pixel_scale_arcsec_by_instrument"][instrument]
    )
    dataset.attrs["num_pix"] = int(config["num_pix_by_instrument"][instrument])


def _init_hdf5(
    output_path,
    count,
    num_pix_by_instrument,
    lens_field_num_pix,
    instruments,
    bands_by_instrument,
    dtype,
    meta_columns,
    compression,
    chunk_samples,
    config_json,
    config,
):
    import h5py

    compression = None if compression == "none" else compression
    handle = h5py.File(output_path, "w")
    band_config_payload = _band_config_payload(instruments, bands_by_instrument)
    _store_common_attrs(
        handle.attrs,
        count=count,
        bands_by_instrument=bands_by_instrument,
        band_config_payload=band_config_payload,
        num_pix_by_instrument=num_pix_by_instrument,
        lens_field_num_pix=lens_field_num_pix,
        config_json=config_json,
        config=config,
    )
    chunk_n = min(max(1, int(chunk_samples)), count)

    out = {
        "id": handle.create_dataset(
            "id",
            shape=(count,),
            dtype=h5py.string_dtype("ascii", 32),
        ),
        "index": handle.create_dataset("index", shape=(count,), dtype=np.int64),
        "label": handle.create_dataset("label", shape=(count,), dtype=np.uint8),
        "meta": handle.create_dataset(
            "meta",
            shape=(count, len(meta_columns)),
            dtype=np.float32,
            compression=compression,
        ),
        **{group_name: {} for group_name in PIXEL_PRODUCT_GROUPS},
    }
    out["meta"].attrs["columns"] = np.array(meta_columns, dtype="S")

    for group_name in PIXEL_PRODUCT_GROUPS:
        group = handle.create_group(group_name)
        for instrument in instruments:
            bands = bands_by_instrument[instrument]
            num_pix = int(num_pix_by_instrument[instrument])
            dataset = group.create_dataset(
                instrument,
                shape=(count, num_pix, num_pix, len(bands)),
                dtype=dtype,
                compression=compression,
                chunks=(chunk_n, num_pix, num_pix, len(bands)),
            )
            _instrument_dataset_attrs(dataset, instrument, bands, band_config_payload, config)
            out[group_name][instrument] = dataset

    lens_group = handle.create_group("lens_fields")
    lens_group.attrs["num_pix"] = int(lens_field_num_pix)
    lens_group.attrs["pixel_scale_arcsec"] = float(
        _lens_field_pixel_scale(instruments, bands_by_instrument)
    )
    for field_name in LENS_FIELD_PRODUCTS:
        out[field_name] = lens_group.create_dataset(
            field_name,
            shape=(count, lens_field_num_pix, lens_field_num_pix),
            dtype=dtype,
            compression=compression,
            chunks=(chunk_n, lens_field_num_pix, lens_field_num_pix),
        )
    return handle, out


def _zarr_compressor(config):
    if config["zarr_compressor"] == "none":
        return None
    import numcodecs

    clevel = max(0, min(9, int(config["zarr_clevel"])))
    return numcodecs.Blosc(
        cname="zstd",
        clevel=clevel,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )


def _init_zarr(
    output_path,
    count,
    num_pix_by_instrument,
    lens_field_num_pix,
    instruments,
    bands_by_instrument,
    dtype,
    meta_columns,
    chunk_samples,
    config_json,
    config,
):
    import numcodecs
    import zarr

    if os.path.exists(output_path) and config["overwrite"]:
        shutil.rmtree(output_path)

    handle = zarr.open_group(output_path, mode="w")
    band_config_payload = _band_config_payload(instruments, bands_by_instrument)
    _store_common_attrs(
        handle.attrs,
        count=count,
        bands_by_instrument=bands_by_instrument,
        band_config_payload=band_config_payload,
        num_pix_by_instrument=num_pix_by_instrument,
        lens_field_num_pix=lens_field_num_pix,
        config_json=config_json,
        config=config,
    )
    chunk_n = min(max(1, int(chunk_samples)), count)
    compressor = _zarr_compressor(config)

    out = {
        "id": handle.create_dataset(
            "id",
            shape=(count,),
            chunks=(chunk_n,),
            dtype=object,
            object_codec=numcodecs.VLenUTF8(),
        ),
        "index": handle.create_dataset(
            "index",
            shape=(count,),
            chunks=(chunk_n,),
            dtype="i8",
            compressor=compressor,
        ),
        "label": handle.create_dataset(
            "label",
            shape=(count,),
            chunks=(chunk_n,),
            dtype="u1",
            compressor=compressor,
        ),
        "meta": handle.create_dataset(
            "meta",
            shape=(count, len(meta_columns)),
            chunks=(chunk_n, len(meta_columns)),
            dtype="f4",
            compressor=compressor,
        ),
        **{group_name: {} for group_name in PIXEL_PRODUCT_GROUPS},
    }
    out["meta"].attrs["columns"] = list(meta_columns)

    for group_name in PIXEL_PRODUCT_GROUPS:
        group = handle.create_group(group_name)
        for instrument in instruments:
            bands = bands_by_instrument[instrument]
            num_pix = int(num_pix_by_instrument[instrument])
            dataset = group.create_dataset(
                instrument,
                shape=(count, num_pix, num_pix, len(bands)),
                chunks=(chunk_n, num_pix, num_pix, len(bands)),
                dtype=dtype,
                compressor=compressor,
            )
            _instrument_dataset_attrs(dataset, instrument, bands, band_config_payload, config)
            out[group_name][instrument] = dataset

    lens_group = handle.create_group("lens_fields")
    lens_group.attrs["num_pix"] = int(lens_field_num_pix)
    lens_group.attrs["pixel_scale_arcsec"] = float(
        _lens_field_pixel_scale(instruments, bands_by_instrument)
    )
    for field_name in LENS_FIELD_PRODUCTS:
        out[field_name] = lens_group.create_dataset(
            field_name,
            shape=(count, lens_field_num_pix, lens_field_num_pix),
            chunks=(chunk_n, lens_field_num_pix, lens_field_num_pix),
            dtype=dtype,
            compressor=compressor,
        )
    return handle, out


def _init_store(
    output_path,
    count,
    num_pix_by_instrument,
    lens_field_num_pix,
    instruments,
    bands_by_instrument,
    dtype,
    meta_columns,
    config,
    config_json,
):
    if config["storage"] == "hdf5":
        return _init_hdf5(
            output_path=output_path,
            count=count,
            num_pix_by_instrument=num_pix_by_instrument,
            lens_field_num_pix=lens_field_num_pix,
            instruments=instruments,
            bands_by_instrument=bands_by_instrument,
            dtype=dtype,
            meta_columns=meta_columns,
            compression=config["compression"],
            chunk_samples=config["chunk_samples"],
            config_json=config_json,
            config=config,
        )
    return _init_zarr(
        output_path=output_path,
        count=count,
        num_pix_by_instrument=num_pix_by_instrument,
        lens_field_num_pix=lens_field_num_pix,
        instruments=instruments,
        bands_by_instrument=bands_by_instrument,
        dtype=dtype,
        meta_columns=meta_columns,
        chunk_samples=config["chunk_samples"],
        config_json=config_json,
        config=config,
    )


def _append_meta_csv_rows(
    csv_path: Path,
    ids,
    indices,
    *,
    label_value: int,
    run_id: int,
    shard_index: int,
    worker_pid: int,
    meta,
) -> None:
    need_header = not csv_path.exists()
    header = [
        "id",
        "index",
        "label",
        "run_id",
        "shard_id",
        "worker_pid",
        *list(META_COLUMNS),
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if need_header:
            writer.writerow(header)
        for idx in range(len(ids)):
            row = [
                str(ids[idx]),
                int(indices[idx]),
                int(label_value),
                int(run_id),
                int(shard_index),
                int(worker_pid),
            ]
            row.extend(float(value) for value in meta[idx].tolist())
            writer.writerow(row)


def _build_context(config: dict) -> dict:
    if config["category"] == "lens":
        return _build_lens_context(config)
    return _build_nonlens_context(config)


def _set_system_use_jax(gg_lens, use_jax: bool) -> None:
    gg_lens._use_jax = bool(use_jax)
    for attr_name in (
        "_theta_E_infinity",
        "_theta_E_list",
        "_kwargs_lens",
        "_lens_mass_model_list",
    ):
        if hasattr(gg_lens, attr_name):
            delattr(gg_lens, attr_name)


def _lens_global_index(run_id: int, local_index: int) -> int:
    return (int(run_id) << 32) + int(local_index)


def _lens_sample_id(run_id: int, local_index: int) -> str:
    return f"Lens_{run_id:06d}_{local_index:07d}"


def _nonlens_global_index(start_index: int, local_index: int) -> int:
    return int(start_index) + int(local_index)


def _nonlens_sample_id(global_index: int) -> str:
    return f"Nonlens_{global_index:07d}"


def _draw_population_for_task(task: BatchTask, config: dict, context: dict) -> list:
    if config["category"] == "nonlens":
        population = _normalize_population(
            context["false_positive_pop"].draw_false_positive(int(task.requested_count))
        )
        for gg_lens in population:
            _set_system_use_jax(gg_lens, config["use_jaxtronomy"])
            _augment_with_field_galaxies(
                gg_lens,
                context["field_galaxy_pop"],
                config["field_galaxy_area_arcsec2"],
            )
        return population

    population = _normalize_population(
        context["lens_pop"].draw_population(
            kwargs_lens_cuts=context["kwargs_lens_cut"]
        )
    )
    for gg_lens in population:
        _set_system_use_jax(gg_lens, config["use_jaxtronomy"])
        _augment_with_field_galaxies(
            gg_lens,
            context["field_galaxy_pop"],
            config["field_galaxy_area_arcsec2"],
        )
    return population


def _run_batch(task: BatchTask, config: dict) -> dict:
    _configure_env(config["disable_jit"])
    worker_pid = os.getpid()
    resource_state = {}
    category = config["category"]
    label_value = 1 if category == "lens" else 0
    log_prefix = f"[{category} run={task.run_id:06d}]"
    requested_label = (
        f"{task.requested_count}" if category == "nonlens" else "sky_area"
    )

    if os.path.exists(task.output_path) and not config["overwrite"]:
        return {
            "status": "skipped",
            "path": task.output_path,
            "count": 0,
            "requested_count": task.requested_count,
            "run_id": task.run_id,
            "shard_index": task.shard_index,
            "worker_pid": worker_pid,
        }

    if task.seed is None:
        np.random.seed(None)
        random.seed(None)
    else:
        seed = int(task.seed) & 0xFFFFFFFF
        np.random.seed(seed)
        random.seed(seed)

    _configure_filters()
    print(
        f"{log_prefix} pid={worker_pid} step=init requested={requested_label} "
        f"path={task.output_path} {_resource_snapshot(resource_state)}",
        flush=True,
    )
    print(
        f"{log_prefix} pid={worker_pid} step=build_population "
        f"{_resource_snapshot(resource_state)}",
        flush=True,
    )

    context = _build_context(config)
    population = _draw_population_for_task(task, config, context)
    found = len(population)
    if found <= 0:
        print(
            f"{log_prefix} pid={worker_pid} step=finished wrote=0/{requested_label} "
            f"found=0 {_resource_snapshot(resource_state)} {_resource_peak_text(resource_state)}",
            flush=True,
        )
        return {
            "status": "skipped",
            "path": task.output_path,
            "count": 0,
            "requested_count": int(task.requested_count),
            "run_id": task.run_id,
            "shard_index": task.shard_index,
            "worker_pid": worker_pid,
            "population_candidates_total": 0,
            "candidates_seen": 0,
            "sim_failures": 0,
            "peak_rss_mb": resource_state.get("peak_rss_mb"),
        }

    print(
        f"{log_prefix} pid={worker_pid} step=simulate_prepare found={found} "
        f"{_resource_snapshot(resource_state)}",
        flush=True,
    )

    _prepare_output_path(task.output_path, config["overwrite"])
    shard_started_at = time.time()
    config_for_json = dict(config)
    config_for_json.pop("start_time", None)
    config_json = json.dumps(config_for_json, sort_keys=True)
    dtype = np.dtype(config["dtype"])
    handle = None
    written = 0
    failures = 0
    last_log = 0
    start_time = config.get("start_time", time.time())
    write_batch = max(1, int(config["write_batch"]))
    log_every = int(config["log_every"])
    meta_csv_path = Path(config["shards_root"]) / f"{category}_meta.csv"

    buffer_meta = []
    buffer_products = {
        group_name: {name: [] for name in config["instruments"]}
        for group_name in PIXEL_PRODUCT_GROUPS
    }
    buffer_fields = {field_name: [] for field_name in LENS_FIELD_PRODUCTS}

    try:
        handle, datasets = _init_store(
            output_path=task.output_path,
            count=found,
            num_pix_by_instrument=config["num_pix_by_instrument"],
            lens_field_num_pix=config["lens_field_num_pix"],
            instruments=config["instruments"],
            bands_by_instrument=config["bands_by_instrument"],
            dtype=dtype,
            meta_columns=META_COLUMNS,
            config=config,
            config_json=config_json,
        )
        datasets["label"][:] = np.full(found, label_value, dtype=np.uint8)

        def _flush() -> None:
            nonlocal written, last_log
            if not buffer_meta:
                return
            n = len(buffer_meta)
            start = written
            end = written + n
            local_offset = np.arange(start, end, dtype=np.int64)
            if category == "lens":
                indices_array = np.asarray(
                    [_lens_global_index(task.run_id, int(offset)) for offset in local_offset],
                    dtype=np.int64,
                )
                ids_array = np.asarray(
                    [_lens_sample_id(task.run_id, int(offset)) for offset in local_offset],
                    dtype="U32",
                )
            else:
                indices_array = np.arange(
                    task.start_index + start, task.start_index + end, dtype=np.int64
                )
                ids_array = np.asarray(
                    [_nonlens_sample_id(int(index)) for index in indices_array],
                    dtype="U32",
                )
            if config["storage"] == "hdf5":
                datasets["id"][start:end] = ids_array.astype("S32")
            else:
                datasets["id"][start:end] = ids_array
            datasets["index"][start:end] = indices_array
            meta_array = np.stack(buffer_meta).astype(np.float32)
            datasets["meta"][start:end] = meta_array
            for group_name in PIXEL_PRODUCT_GROUPS:
                for name in config["instruments"]:
                    datasets[group_name][name][start:end] = np.stack(
                        buffer_products[group_name][name]
                    ).astype(dtype, copy=False)
            for field_name in LENS_FIELD_PRODUCTS:
                datasets[field_name][start:end] = np.stack(
                    buffer_fields[field_name]
                ).astype(dtype, copy=False)

            handle.attrs["written"] = end
            if hasattr(handle, "flush"):
                handle.flush()
            _append_meta_csv_rows(
                csv_path=meta_csv_path,
                ids=ids_array,
                indices=indices_array,
                label_value=label_value,
                run_id=task.run_id,
                shard_index=task.shard_index,
                worker_pid=worker_pid,
                meta=meta_array,
            )
            written = end
            buffer_meta.clear()
            for group_name in PIXEL_PRODUCT_GROUPS:
                for items in buffer_products[group_name].values():
                    items.clear()
            for items in buffer_fields.values():
                items.clear()

            if log_every > 0 and written - last_log >= log_every:
                elapsed = _format_seconds(time.time() - start_time)
                print(
                    f"{log_prefix} pid={worker_pid} step=simulate_write {written}/{found} "
                    f"(elapsed {elapsed}) {_resource_snapshot(resource_state)}",
                    flush=True,
                )
                last_log = written

        for gg_lens in population:
            try:
                products = _simulate_products(
                    gg_lens=gg_lens,
                    instruments=config["instruments"],
                    bands_by_instrument=config["bands_by_instrument"],
                    num_pix_by_instrument=config["num_pix_by_instrument"],
                    lens_field_num_pix=config["lens_field_num_pix"],
                    dtype=dtype,
                )
                meta_row = _meta_row(gg_lens)
                if not _all_finite_products(products, config["instruments"]):
                    raise RuntimeError("Non-finite values in generated products.")
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    print(
                        f"{log_prefix} pid={worker_pid} simulation failure #{failures}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if failures > max(50, found):
                    raise RuntimeError("Too many simulation failures; aborting run.")
                continue

            buffer_meta.append(meta_row)
            for group_name in PIXEL_PRODUCT_GROUPS:
                for name in config["instruments"]:
                    buffer_products[group_name][name].append(products[group_name][name])
            for field_name in LENS_FIELD_PRODUCTS:
                buffer_fields[field_name].append(products[field_name])

            if len(buffer_meta) >= write_batch:
                _flush()

        if buffer_meta:
            _flush()

        status = "ok" if written == found else "partial"
        shard_elapsed = time.time() - shard_started_at
        print(
            f"{log_prefix} pid={worker_pid} step=finished wrote={written}/{found} "
            f"requested={requested_label} sim_failures={failures} "
            f"elapsed={_format_seconds(shard_elapsed)} "
            f"{_resource_snapshot(resource_state)} {_resource_peak_text(resource_state)}",
            flush=True,
        )
        return {
            "status": status,
            "path": task.output_path,
            "count": int(written),
            "requested_count": int(task.requested_count),
            "drawn_count": int(found),
            "elapsed_seconds": float(shard_elapsed),
            "category": category,
            "start_index": int(task.start_index),
            "shard_index": int(task.shard_index),
            "run_id": int(task.run_id),
            "worker_pid": int(worker_pid),
            "population_candidates_total": int(found),
            "candidates_seen": int(found),
            "sim_failures": int(failures),
            "peak_rss_mb": resource_state.get("peak_rss_mb"),
        }
    except Exception:
        if handle is not None and hasattr(handle, "close"):
            handle.close()
        if os.path.isdir(task.output_path):
            shutil.rmtree(task.output_path, ignore_errors=True)
        elif os.path.exists(task.output_path):
            try:
                os.remove(task.output_path)
            except FileNotFoundError:
                pass
        raise
    finally:
        if handle is not None and hasattr(handle, "close"):
            handle.close()


def run_main(category: str) -> int:
    if category not in {"lens", "nonlens"}:
        raise ValueError(f"Unsupported category: {category}")

    parser = _parser_for_category(category)
    args = parser.parse_args()
    if args.num_runs <= 0:
        print("--num-runs must be > 0.", file=sys.stderr)
        return 1
    if args.batch_size <= 0:
        print("--batch-size must be > 0.", file=sys.stderr)
        return 1

    instruments = parse_instruments(args.instruments)
    bands_by_instrument = {}
    if "euclid" in instruments:
        bands_by_instrument["euclid"] = parse_bands(args.euclid_bands, "euclid")
    if "lsst" in instruments:
        bands_by_instrument["lsst"] = parse_bands(args.lsst_bands, "lsst")
    if "roman" in instruments:
        bands_by_instrument["roman"] = parse_bands(args.roman_bands, "roman")

    num_pix_by_instrument, lens_field_num_pix = resolve_num_pix_config(args, instruments)
    if args.use_hst:
        hst_cosmos_path = _resolve_hst_cosmos_path(args.hst_cosmos_path)
        using_repo_hst_cosmos_fallback = (
            hst_cosmos_path.resolve() == _repo_hst_cosmos_fallback_path().resolve()
        )
    else:
        hst_cosmos_path = None
        using_repo_hst_cosmos_fallback = False
    data_root = args.data_root.resolve()
    paths = make_output_dirs(data_root)
    manifest_name = f"manifest_lre_{category}.json"

    config = {
        "category": category,
        "num_pix_by_instrument": num_pix_by_instrument,
        "lens_field_num_pix": lens_field_num_pix,
        "instruments": instruments,
        "bands_by_instrument": bands_by_instrument,
        "storage": args.storage,
        "dtype": args.dtype,
        "compression": args.compression,
        "zarr_compressor": args.zarr_compressor,
        "zarr_clevel": args.zarr_clevel,
        "chunk_samples": args.zarr_chunk_samples,
        "overwrite": args.overwrite,
        "disable_jit": args.disable_jit,
        "write_batch": args.write_batch,
        "log_every": args.log_every,
        "field_galaxy_area_arcsec2": args.field_galaxy_area_arcsec2,
        "hst_cosmos_path": None if hst_cosmos_path is None else str(hst_cosmos_path),
        "use_hst": bool(args.use_hst),
        "recommended_center_crop_num_pix_by_instrument": dict(
            RECOMMENDED_CENTER_CROP_NUM_PIX_BY_INSTRUMENT
        ),
        "native_pixel_scale_arcsec_by_instrument": dict(
            NATIVE_PIXEL_SCALE_ARCSEC_BY_INSTRUMENT
        ),
        "max_draws": args.max_draws,
        "lens_sky_area": args.lens_sky_area,
        "lens_sky_area_galaxies": args.lens_sky_area_galaxies,
        "nonlens_sky_area": args.nonlens_sky_area,
        "nonlens_sky_area_full": args.nonlens_sky_area_full,
        "run_id_offset": args.run_id_offset,
        "start_index_offset": args.start_index_offset,
        "shards_root": str(paths["shards_root"]),
        "manifest_name": manifest_name,
        "use_jaxtronomy": _jaxtronomy_available(),
        "catalog_source_sersic_fallback": using_repo_hst_cosmos_fallback,
    }

    _configure_env(args.disable_jit)
    try:
        _configure_filters()
    except Exception as exc:
        print(f"Filter setup failed: {exc}", file=sys.stderr)
        return 1

    tasks = []
    suffix = ".zarr" if args.storage == "zarr" else ".h5"
    category_root = paths[f"{category}_root"]
    for run_id in range(args.num_runs):
        task_run_id = int(args.run_id_offset) + run_id
        seed = None if args.seed_base is None else int(args.seed_base) + task_run_id
        output_path = category_root / f"{category}_{task_run_id:06d}{suffix}"
        tasks.append(
            BatchTask(
                run_id=task_run_id,
                shard_index=task_run_id,
                requested_count=int(args.batch_size),
                start_index=int(args.start_index_offset) + run_id * int(args.batch_size),
                output_path=str(output_path),
                seed=seed,
            )
        )

    main_resource_state = {}
    print(f"HST enabled: {config['use_hst']}", flush=True)
    print(
        "HST COSMOS path: "
        f"{hst_cosmos_path if hst_cosmos_path is not None else 'disabled'}",
        flush=True,
    )
    print(
        f"Catalog-source Sersic fallback: {config['catalog_source_sersic_fallback']}",
        flush=True,
    )
    print(f"Bands: {bands_by_instrument}", flush=True)
    print(
        f"Image sizes: {num_pix_by_instrument}, lens_fields={lens_field_num_pix}",
        flush=True,
    )
    print(
        "Crop guidance: "
        f"{RECOMMENDED_CENTER_CROP_NUM_PIX_BY_INSTRUMENT} "
        f"pixel_scales={NATIVE_PIXEL_SCALE_ARCSEC_BY_INSTRUMENT}",
        flush=True,
    )
    print(f"JAXtronomy enabled: {config['use_jaxtronomy']}", flush=True)
    print(
        f"Submitting {len(tasks)} notebook-style {category} runs with {args.num_workers} workers "
        f"(storage={args.storage}, requested_mode={'count' if category == 'nonlens' else 'sky_area'}, "
        f"batch_size={args.batch_size}, dtype={args.dtype}, "
        f"seed_base={'none' if args.seed_base is None else args.seed_base}) "
        f"{_resource_snapshot(main_resource_state)}.",
        flush=True,
    )

    started = time.time()
    config["start_time"] = started
    results = []
    ok = 0
    partial = 0
    skipped = 0
    written_total = 0
    peak_worker_rss = 0.0
    per_worker = {}
    done = 0
    interrupted = False

    def _record_finished(task: BatchTask, result: dict) -> None:
        nonlocal done, ok, partial, skipped, written_total, peak_worker_rss
        results.append(result)
        done += 1
        status = result.get("status")
        if status == "ok":
            ok += 1
        elif status == "partial":
            partial += 1
        else:
            skipped += 1
        count = int(result.get("count", 0))
        written_total += count
        peak_worker_rss = max(peak_worker_rss, float(result.get("peak_rss_mb") or 0.0))
        pid = int(result.get("worker_pid", -1))
        stats = per_worker.setdefault(
            pid,
            {"runs": 0, "drawn": 0, "written": 0, "sim_failures": 0},
        )
        stats["runs"] += 1
        stats["drawn"] += int(result.get("drawn_count", 0))
        stats["written"] += count
        stats["sim_failures"] += int(result.get("sim_failures", 0))
        if status == "skipped":
            print(f"\n[skipped] {task.output_path} (0 samples)", file=sys.stderr)
        elif status == "partial":
            requested_text = (
                str(task.requested_count) if category == "nonlens" else "sky_area"
            )
            print(
                f"\n[partial] {task.output_path} wrote={count}/{result.get('drawn_count', 0)} "
                f"(requested={requested_text})",
                file=sys.stderr,
            )

    if args.num_workers <= 1:
        _print_progress(done, len(tasks), started)
        for task in tasks:
            result = _run_batch(task, config)
            _record_finished(task, result)
            _print_progress(done, len(tasks), started)
        print()
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=mp.get_context(args.mp_start),
        ) as executor:
            futures = {executor.submit(_run_batch, task, config): task for task in tasks}
            _print_progress(done, len(tasks), started)
            try:
                while futures:
                    done_set, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                    for future in done_set:
                        task = futures.pop(future)
                        result = future.result()
                        _record_finished(task, result)
                        _print_progress(done, len(tasks), started)
            except KeyboardInterrupt:
                interrupted = True
                print(
                    "\nKeyboardInterrupt received: waiting for running runs to finish...",
                    file=sys.stderr,
                )
                for future in futures:
                    future.cancel()
                raise
            print()

    if per_worker:
        print("Per-worker run summary:")
        for pid in sorted(per_worker):
            stats = per_worker[pid]
            keep = 100.0 * stats["written"] / max(1, stats["drawn"])
            print(
                f"  pid={pid} runs={stats['runs']} drawn={stats['drawn']} "
                f"written={stats['written']} keep={keep:.1f}% "
                f"sim_failures={stats['sim_failures']}"
            )

    results.sort(key=lambda item: item.get("run_id", 0))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "summary": {
            "ok": ok,
            "partial": partial,
            "skipped": skipped,
            "num_runs": args.num_runs,
            "batch_size": args.batch_size,
            "written_total": written_total,
            "interrupted": interrupted,
        },
        "results": results,
    }
    manifest_path = paths["shards_root"] / manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    elapsed = time.time() - started
    throughput = written_total / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {_format_seconds(elapsed)}. ok={ok} partial={partial} skipped={skipped} "
        f"runs={len(tasks)} written_total={written_total} throughput={throughput:.2f} samples/s. "
        f"peak_worker_rss={peak_worker_rss:.1f}MB Manifest: {manifest_path} "
        f"{_resource_snapshot(main_resource_state)} {_resource_peak_text(main_resource_state)}",
        flush=True,
    )
    return 0
