"""Pinned biological dataset fetchers and leakage-free preprocessing for MemGRN."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import requests
from scipy.io import loadmat

_LOGGER = logging.getLogger(__name__)

MSN2_PROMOTERS: tuple[str, ...] = (
    "ALD3",
    "DCS2",
    "DDR2",
    "HXK1",
    "RTN2",
    "SIP18",
    "TKL2",
)
MSN2_GIT_REPO = "zechnerlab/PromoterManifest"
MSN2_GIT_REVISION = "9d22cc6f5d06f85543be2cfbdbb13fb7e57c1c70"
MSN2_TRAIN_MINUTES: tuple[int, ...] = (10, 20, 40, 50)
MSN2_TRAIN_DOSES_NM: tuple[int, ...] = (100, 275, 690, 3000)
MSN2_HELD_SINGLE_PULSE: tuple[int, int] = (30, 690)

NFKB_GIT_REPO = "biomystery/duration_decode_manuscript"
NFKB_GIT_REVISION = "37be90e56f6e4a9c726990c09e0adc114dcc8efc"
NFKB_LIGANDS: tuple[str, ...] = ("IL1", "TNF", "LPS")


@dataclass(frozen=True, kw_only=True)
class Msn2SplitData:
    """Preprocessed single-pulse training and held-out Msn2 arrays."""

    promoters: tuple[str, ...]
    obs_minutes: np.ndarray
    tf_grid_minutes: np.ndarray
    train_pulses: list[tuple[int, int]]
    held_pulse: tuple[int, int]
    train_y_scaled: np.ndarray
    held_y_scaled: np.ndarray
    target_scales: np.ndarray
    tf_drivers_scaled: np.ndarray
    tf_scale: float
    held_cell_counts: dict[str, int]


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _msn2_single_filename(gene: str, pulse: tuple[int, int]) -> str:
    minutes, dose = pulse
    dose_str = "3uM" if dose == 3000 else f"{dose}nM"
    return f"{gene}_DM_{minutes}min_{dose_str}_size.mat"


def fetch_msn2_raw(
    raw_dir: Path,
    include_repeated: bool = True,
    repeated_schedules: list[dict[str, float | int | str]] | None = None,
) -> dict[str, str]:
    """Download and verify pinned Hansen & Zechner (2021) single-cell .mat files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tree_url = (
        f"https://api.github.com/repos/{MSN2_GIT_REPO}/git/trees/"
        f"{MSN2_GIT_REVISION}?recursive=1"
    )
    resp = requests.get(tree_url, timeout=45)
    resp.raise_for_status()
    blob_shas = {item["path"]: item["sha"] for item in resp.json()["tree"]}

    pulses = [
        (m, d) for m in MSN2_TRAIN_MINUTES for d in MSN2_TRAIN_DOSES_NM
    ] + [MSN2_HELD_SINGLE_PULSE]
    jobs: list[tuple[str, str]] = [
        (g, _msn2_single_filename(g, p)) for p in pulses for g in MSN2_PROMOTERS
    ]
    if include_repeated and repeated_schedules is not None:
        for g in MSN2_PROMOTERS:
            for sched in repeated_schedules:
                jobs.append((g, f"{g}_{sched['id']}_size.mat"))

    def _download_one(job: tuple[str, str]) -> tuple[str, str]:
        gene, fname = job
        repo_key = f"Data/MSN2/{gene}/{fname}"
        dest = raw_dir / fname
        for attempt in range(4):
            try:
                if dest.exists():
                    blob = dest.read_bytes()
                else:
                    raw_url = (
                        f"https://raw.githubusercontent.com/{MSN2_GIT_REPO}/"
                        f"{MSN2_GIT_REVISION}/{repo_key}"
                    )
                    r = requests.get(raw_url, timeout=60)
                    r.raise_for_status()
                    blob = r.content
                git_sha1 = hashlib.sha1(
                    f"blob {len(blob)}\0".encode() + blob
                ).hexdigest()
                if git_sha1 != blob_shas.get(repo_key):
                    raise ValueError(f"Git blob SHA-1 mismatch for {repo_key}")
                if not dest.exists():
                    dest.write_bytes(blob)
                return fname, hashlib.sha256(blob).hexdigest()
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"Unreachable download retry state for {fname}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = dict(pool.map(_download_one, jobs))
    _LOGGER.info("Verified %d Msn2 raw microscopy files in %s", len(manifest), raw_dir)
    return manifest


def load_msn2_population_trace(
    mat_path: Path,
    field: str,
    target_times: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Extract baseline-subtracted population mean fluorescence trace from a raw .mat file."""
    mat = loadmat(
        mat_path,
        variable_names=["time", "pulse_parameters", "inhibitor_conc", field],
        squeeze_me=True,
    )
    time_grid = np.asarray(mat["time"], dtype=np.float64)
    cells = np.asarray(mat[field], dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != len(time_grid):
        raise ValueError(f"Unexpected matrix shape {cells.shape} in {mat_path}")
    if not np.any(time_grid < 0.0):
        raise ValueError(f"Missing pre-onset baseline frames in {mat_path}")

    idx = np.searchsorted(time_grid, target_times)
    if not np.allclose(time_grid[idx], target_times):
        raise ValueError(f"Requested times not aligned with {mat_path}")

    pop_mean = np.nanmean(cells, axis=0)
    centered = pop_mean[idx] - float(np.mean(pop_mean[time_grid < 0.0]))
    if not np.all(np.isfinite(centered)):
        raise ValueError(f"Non-finite centered trace in {mat_path}")
    return centered, int(cells.shape[0])


def prepare_msn2_single_pulse_splits(raw_dir: Path) -> Msn2SplitData:
    """Compute leakage-free Msn2 single-pulse training and 30-min held-out arrays."""
    tf_grid = np.arange(0.0, 120.01, 2.5)
    obs_grid = np.arange(5.0, 120.01, 5.0)
    train_pulses = [
        (m, d) for m in MSN2_TRAIN_MINUTES for d in MSN2_TRAIN_DOSES_NM
    ]

    def _yfp_panel(pulse: tuple[int, int]) -> tuple[np.ndarray, dict[str, int]]:
        traces = []
        counts: dict[str, int] = {}
        for g in MSN2_PROMOTERS:
            tr, n_cells = load_msn2_population_trace(
                raw_dir / _msn2_single_filename(g, pulse), "YFP_molecules", obs_grid
            )
            traces.append(tr)
            counts[g] = n_cells
        return np.stack(traces, axis=1), counts

    def _rfp_driver(pulse: tuple[int, int]) -> np.ndarray:
        strain_means = [
            load_msn2_population_trace(
                raw_dir / _msn2_single_filename(g, pulse), "MSN2_RFP", tf_grid
            )[0]
            for g in MSN2_PROMOTERS
        ]
        return np.mean(strain_means, axis=0)

    train_y_raw = np.stack([_yfp_panel(p)[0] for p in train_pulses], axis=0)
    held_y_raw, held_counts = _yfp_panel(MSN2_HELD_SINGLE_PULSE)

    target_scales = np.maximum(
        1.0, np.sqrt(np.mean(train_y_raw**2, axis=(0, 1)))
    )
    all_drivers = np.stack(
        [_rfp_driver(p) for p in [*train_pulses, MSN2_HELD_SINGLE_PULSE]], axis=0
    )
    tf_scale = float(np.max(np.abs(all_drivers[:-1])))
    if tf_scale <= 0.0:
        raise ValueError("Training Msn2-RFP scale must be strictly positive")

    return Msn2SplitData(
        promoters=MSN2_PROMOTERS,
        obs_minutes=obs_grid,
        tf_grid_minutes=tf_grid,
        train_pulses=train_pulses,
        held_pulse=MSN2_HELD_SINGLE_PULSE,
        train_y_scaled=train_y_raw / target_scales[None, None, :],
        held_y_scaled=held_y_raw / target_scales[None, :],
        target_scales=target_scales,
        tf_drivers_scaled=all_drivers / tf_scale,
        tf_scale=tf_scale,
        held_cell_counts=held_counts,
    )


def build_rk4_stage_grid(
    drivers: np.ndarray,
    substeps_per_knot: int = 8,
) -> np.ndarray:
    """Interpolate 2.5-minute Msn2 driver knots onto RK4 (start, mid, end) stage grids."""
    frac = np.arange(substeps_per_knot, dtype=np.float64) / substeps_per_knot
    lo = drivers[:, :-1, None]
    delta = np.diff(drivers, axis=1)[:, :, None]

    def _interp(offset: float) -> np.ndarray:
        vals = lo + delta * (frac + offset / substeps_per_knot)
        return vals.reshape(len(drivers), -1).T

    # Shape: (n_total_steps, n_schedules, 1, 3) for single-TF Msn2
    stages = np.stack([_interp(0.0), _interp(0.5), _interp(1.0)], axis=-1)
    return stages[:, :, None, :]


def load_archived_json(results_root: Path, relative_path: str) -> dict:
    """Load a JSON artifact from results/stage3."""
    path = results_root / relative_path
    return json.loads(path.read_text(encoding="utf-8"))
