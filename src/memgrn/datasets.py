"""Pinned biological dataset fetchers and leakage-free preprocessing for MemGRN."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator
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
    expected_sha256: dict[str, str] | None = None,
    only_held_and_repeated: bool = False,
) -> dict[str, str]:
    """Download and verify pinned Hansen & Zechner (2021) single-cell .mat files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pulses = (
        [MSN2_HELD_SINGLE_PULSE]
        if only_held_and_repeated
        else [(m, d) for m in MSN2_TRAIN_MINUTES for d in MSN2_TRAIN_DOSES_NM]
        + [MSN2_HELD_SINGLE_PULSE]
    )
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
                digest = hashlib.sha256(blob).hexdigest()
                if expected_sha256 is not None and fname in expected_sha256:
                    if digest != expected_sha256[fname]:
                        raise ValueError(f"SHA-256 mismatch for {repo_key}")
                if not dest.exists():
                    dest.write_bytes(blob)
                return fname, digest
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"Unreachable download retry state for {fname}")

    with ThreadPoolExecutor(max_workers=12) as pool:
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


MACROPHAGE_GIT_REPO = "signalingsystemslab/ResponseSpecificity"
MACROPHAGE_GIT_REVISION = "5336be7e3fe121a29bee1c8e4f39fe3ecf667a85"
MACROPHAGE_REGULATORS: tuple[str, ...] = ("NFkB", "IRF", "AP1", "p38")
MACROPHAGE_STIMULI: tuple[str, ...] = (
    "PAM",
    "PIC",
    "LPS",
    "FLA",
    "FSL",
    "R848",
    "CPG",
    "TNF",
    "IFNB",
    "MCMV",
    "VSV",
    "STREP",
    "LM",
    "tPIC",
)


@dataclass(frozen=True, kw_only=True)
class MacrophageMultiTFData:
    """Preprocessed 4-TF, 14-stimulus, 477-gene macrophage innate-immune dataset."""

    regulators: tuple[str, ...]
    stimuli: tuple[str, ...]
    genes: tuple[str, ...]
    grs_labels: tuple[str, ...]
    causal_regulator_mask: np.ndarray
    obs_hours: np.ndarray
    dt_hours: float
    obs_stride: int
    stage_inputs: np.ndarray
    expression_by_stimulus: np.ndarray
    file_sha256: dict[str, str]


def fetch_and_prepare_macrophage_multitf(raw_dir: Path) -> MacrophageMultiTFData:
    """Fetch and prepare pinned Cheng et al. (2017) & Sheu et al. (2023) 4-TF BMDM data.

    Retrieves measured NF-kB, IRF, AP-1, and p38 activation trajectories across 14 innate
    immune stimuli alongside paired BMDM RNA-seq time courses (0, 1, 3, 8 h) and curated
    Gene Regulatory Strategy (GRS) ground-truth annotations for 477 induced target genes.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    media_base = (
        f"https://media.githubusercontent.com/media/{MACROPHAGE_GIT_REPO}/"
        f"{MACROPHAGE_GIT_REVISION}"
    )
    raw_base = (
        f"https://raw.githubusercontent.com/{MACROPHAGE_GIT_REPO}/"
        f"{MACROPHAGE_GIT_REVISION}"
    )

    files_to_fetch = {
        "TableS2_NFkBactivationBMDM.txt": f"{media_base}/genes2GRS/Cheng2017_TableS2inputs/TableS2_NFkBactivationBMDM.txt",
        "TableS2_IRFactivationBMDM.txt": f"{media_base}/genes2GRS/Cheng2017_TableS2inputs/TableS2_IRFactivationBMDM.txt",
        "TableS2_AP1activationBMDM.txt": f"{media_base}/genes2GRS/Cheng2017_TableS2inputs/TableS2_AP1activationBMDM.txt",
        "Table_p38activationBMDM.txt": f"{media_base}/genes2GRS/Cheng2017_TableS2inputs/Table_p38activationBMDM.txt",
        "Cheng2017_induced.txt": f"{media_base}/output/Cheng2017_induced.txt",
        "TableS3_genes2GRS.xlsx": f"{raw_base}/SuppTables/TableS3_genes2GRS.xlsx",
    }

    hashes: dict[str, str] = {}
    for fname, url in files_to_fetch.items():
        dest = raw_dir / fname
        if not dest.exists():
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        hashes[fname] = sha256_file(dest)

    # Parse 4 TF/kinase activation tables (basal = 0.05 subtracted so unstimulated = 0.0)
    tf_files = (
        "TableS2_NFkBactivationBMDM.txt",
        "TableS2_IRFactivationBMDM.txt",
        "TableS2_AP1activationBMDM.txt",
        "Table_p38activationBMDM.txt",
    )
    fine_hours = np.linspace(0.0, 8.0, 65)  # 0.125 h (7.5 min) RK4 intervals -> stride 8 per hour
    mid_hours = 0.5 * (fine_hours[:-1] + fine_hours[1:])

    tf_stage_list = []
    for tf_file in tf_files:
        lines = (raw_dir / tf_file).read_text(encoding="utf-8").splitlines()
        t_hours = np.array([float(x) / 60.0 for x in lines[0].split("\t")[1:]], dtype=np.float64)
        stim_rows: dict[str, np.ndarray] = {}
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split("\t")
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            stim_rows[parts[0]] = np.maximum(0.0, vals - vals[0])

        stim_interp_0 = []
        stim_interp_m = []
        stim_interp_1 = []
        for stim in MACROPHAGE_STIMULI:
            interp = PchipInterpolator(t_hours, stim_rows[stim])
            stim_interp_0.append(np.clip(interp(fine_hours[:-1]), 0.0, 1.5))
            stim_interp_m.append(np.clip(interp(mid_hours), 0.0, 1.5))
            stim_interp_1.append(np.clip(interp(fine_hours[1:]), 0.0, 1.5))
        # Shape per TF: (n_steps, n_stimuli, 3)
        tf_stage_list.append(
            np.stack(
                [
                    np.stack(stim_interp_0, axis=1),
                    np.stack(stim_interp_m, axis=1),
                    np.stack(stim_interp_1, axis=1),
                ],
                axis=-1,
            )
        )
    # Combine to (n_steps=64, n_stimuli=14, n_sources=4, 3)
    stage_inputs = np.stack(tf_stage_list, axis=2)

    # Parse TableS3 GRS ground-truth annotations
    with zipfile.ZipFile(io.BytesIO((raw_dir / "TableS3_genes2GRS.xlsx").read_bytes())) as zf:
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sst = [
            "".join(si.itertext())
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", ns)
        ]
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        grs_map: dict[str, str] = {}
        for row in sheet.findall(".//m:row", ns)[1:]:
            vals = []
            for cell in row.findall("m:c", ns):
                v_node = cell.find("m:v", ns)
                vals.append(
                    ""
                    if v_node is None
                    else (sst[int(v_node.text)] if cell.attrib.get("t") == "s" else v_node.text)
                )
            if len(vals) >= 3 and vals[0]:
                grs_map[vals[0]] = vals[2]

    # Parse Cheng2017_induced.txt log2 expression across 14 stimuli at (0, 1, 3, 8) h
    ind_lines = (raw_dir / "Cheng2017_induced.txt").read_text(encoding="utf-8").splitlines()
    header = ind_lines[0].split("\t")
    ut_cols = [i for i, c in enumerate(header) if c.startswith("BMDM_UT_")]
    col_name_prefix = {s: ("Trans_PIC" if s == "tPIC" else s) for s in MACROPHAGE_STIMULI}
    stim_col_idx = {
        s: [header.index(f"BMDM_{col_name_prefix[s]}_{hr}") for hr in (1, 3, 8)]
        for s in MACROPHAGE_STIMULI
    }

    expr_rows = []
    kept_genes: list[str] = []
    kept_grs: list[str] = []
    for ln in ind_lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        gene = parts[0]
        if gene not in grs_map:
            continue
        num = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        ut_base = float(np.mean([ float(parts[c]) for c in ut_cols ]))
        if not np.all(np.isfinite(num)):
            continue
        # Trajectory (n_stimuli, 4) centered at unstimulated baseline t=0 -> [0, y(1)-y(0), y(3)-y(0), y(8)-y(0)]
        mat = np.zeros((len(MACROPHAGE_STIMULI), 4), dtype=np.float64)
        for s_i, stim in enumerate(MACROPHAGE_STIMULI):
            idxs = stim_col_idx[stim]
            mat[s_i, 1:] = [float(parts[c]) - ut_base for c in idxs]
        kept_genes.append(gene)
        kept_grs.append(grs_map[gene])
        expr_rows.append(mat)

    # Shape: (n_stimuli=14, 4, n_genes=477) at t=(0, 1, 3, 8) h
    expr_arr = np.transpose(np.stack(expr_rows, axis=0), (1, 2, 0))

    # Ground-truth causal regulator mask of shape (4, n_genes) for (NFkB, IRF, AP1, p38)
    reg_mask = np.zeros((4, len(kept_genes)), dtype=bool)
    for g_i, label in enumerate(kept_grs):
        if "NFkB" in label:
            reg_mask[0, g_i] = True
        if "IRF" in label:
            reg_mask[1, g_i] = True
        if "AP1" in label:
            reg_mask[2, g_i] = True
        if "p38" in label:
            reg_mask[3, g_i] = True

    return MacrophageMultiTFData(
        regulators=MACROPHAGE_REGULATORS,
        stimuli=MACROPHAGE_STIMULI,
        genes=tuple(kept_genes),
        grs_labels=tuple(kept_grs),
        causal_regulator_mask=reg_mask,
        obs_hours=np.array([0.0, 1.0, 3.0, 8.0], dtype=np.float64),
        dt_hours=0.125,
        obs_stride=8,
        stage_inputs=stage_inputs,
        expression_by_stimulus=expr_arr,
        file_sha256=hashes,
    )


HIPSC_RENGE_REPO = "masastat/RENGE"
HIPSC_RENGE_REVISION = "473b4f9b33072d84bbae3b730d3998434370714b"
HIPSC_PINNED_SHA256: dict[str, str] = {
    "X_renge_d2_80.csv": "149b1f59c8f1e81190fc296862674dce3a0f84983e03a0119813a83d08c02498",
    "E_renge_d2_80.csv": "d4282200b36e03bf4af90a786727b4ca44ecbdbb1a65ee74b7a8fb932b9152d7",
    "A_renge.csv": "dd6bf4ad99858a0ee70ded522957ee1464b5c6329a9e1eff377de523b9715ba9",
}
HIPSC_CHIP_ATLAS_TFS: tuple[str, ...] = (
    "ETS2",
    "ETV4",
    "FOXH1",
    "ID1",
    "JARID2",
    "MYC",
    "MYCN",
    "NANOG",
    "NR5A2",
    "POU5F1",
    "PRDM14",
    "RUNX1T1",
    "SOX2",
    "TRIM24",
    "TRIM25",
    "ZIC2",
    "ZNF398",
)


@dataclass(frozen=True, kw_only=True)
class HiPSCPerturbSeqData:
    """Longitudinal human iPSC CRISPR Perturb-seq dataset (GSE213069; Ishikawa et al., 2023)."""

    genes: tuple[str, ...]
    ko_genes: tuple[str, ...]
    ko_gene_indices: np.ndarray
    obs_days: np.ndarray
    basal_expression: np.ndarray
    ko_delta_trajectories: np.ndarray
    ko_binary_masks: np.ndarray
    eval_target_masks: np.ndarray
    cell_counts: np.ndarray
    renge_reference_graph: np.ndarray
    chip_psc_tf_indices: np.ndarray
    chip_psc_edge_mask: np.ndarray
    chip_any_tf_indices: np.ndarray
    chip_any_edge_mask: np.ndarray
    file_sha256: dict[str, str]


def fetch_and_prepare_hipsc_perturbseq(raw_dir: Path) -> HiPSCPerturbSeqData:
    """Fetch and prepare pinned GSE213069 longitudinal hiPSC scCRISPR + ChIP-Atlas PSC targets."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = raw_dir / "chip_atlas"
    chip_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/{HIPSC_RENGE_REPO}/{HIPSC_RENGE_REVISION}"

    hashes: dict[str, str] = {}
    for fname, expected in HIPSC_PINNED_SHA256.items():
        dest = raw_dir / fname
        if not dest.exists():
            resp = requests.get(f"{base_url}/examples/data/{fname}", timeout=90)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        digest = sha256_file(dest)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch on {fname}: {digest} != {expected}")
        hashes[fname] = digest

    x_df = pd.read_csv(raw_dir / "X_renge_d2_80.csv", index_col=0)
    e_df = pd.read_csv(raw_dir / "E_renge_d2_80.csv", index_col=0)
    a_ref_df = pd.read_csv(raw_dir / "A_renge.csv", index_col=0)

    genes = tuple(sorted(e_df.columns))
    e_df = e_df.reindex(list(genes), axis=1)
    x_genes = x_df.reindex(list(genes), axis=1)
    a_ref = a_ref_df.reindex(index=list(genes), columns=list(genes)).fillna(0.0).to_numpy(dtype=np.float64)
    # Note: in RENGE, A[target, regulator] has rows=targets, cols=regulators; transpose to [regulator, target]
    renge_graph = np.abs(a_ref.T) * (1.0 - np.eye(len(genes)))

    days = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    t_col = x_df["t"].to_numpy(dtype=np.float64)
    x_mat = x_genes.to_numpy(dtype=np.float64)
    e_mat = e_df.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(e_mat)):
        raise ValueError("Non-finite expression entry in GSE213069 E matrix")

    ctrl_mask = np.sum(x_mat != 0.0, axis=1) == 0
    ctrl_by_day = np.stack(
        [np.mean(e_mat[ctrl_mask & (t_col == d)], axis=0) for d in days], axis=0
    )  # (4, 103)
    basal = np.mean(ctrl_by_day, axis=0)  # (103,)

    ko_indices = np.where(np.any(x_mat != 0.0, axis=0))[0]
    ko_genes = tuple(genes[i] for i in ko_indices)
    n_ko = len(ko_indices)
    n_genes = len(genes)

    ko_deltas = np.zeros((n_ko, len(days), n_genes), dtype=np.float64)
    ko_masks = np.zeros((n_ko, n_genes), dtype=np.float64)
    eval_masks = np.ones((n_ko, len(days), n_genes), dtype=np.float64)
    counts = np.zeros((n_ko, len(days)), dtype=np.int64)

    for k_pos, g_idx in enumerate(ko_indices):
        ko_masks[k_pos, g_idx] = 1.0
        eval_masks[k_pos, :, g_idx] = 0.0
        for d_i, d_val in enumerate(days):
            sel = (x_mat[:, g_idx] == 1.0) & (t_col == d_val)
            if not np.any(sel):
                raise ValueError(f"Missing cells for KO {genes[g_idx]} at day {d_val}")
            counts[k_pos, d_i] = int(np.sum(sel))
            ko_deltas[k_pos, d_i] = np.mean(e_mat[sel], axis=0) - ctrl_by_day[d_i]

    # Parse ChIP-Atlas hg38 human PSC and pan-cell target binding matrices
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    chip_psc_edge = np.zeros((n_genes, n_genes), dtype=bool)
    chip_any_edge = np.zeros((n_genes, n_genes), dtype=bool)
    psc_tf_idx_list: list[int] = []
    any_tf_idx_list: list[int] = []
    psc_keywords = ("ES", "iPS", "H1", "H9", "hESC", "hiPSC", "PSC", "NT2", "NCCIT")

    for tf in HIPSC_CHIP_ATLAS_TFS:
        tsv_path = chip_dir / f"{tf}.1.tsv"
        if not tsv_path.exists():
            url = f"https://chip-atlas.dbcls.jp/data/hg38/target/{tf}.1.tsv"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            tsv_path.write_bytes(r.content)
        hashes[f"chip_atlas/{tf}.1.tsv"] = sha256_file(tsv_path)

        tf_i = gene_to_idx[tf]
        lines = tsv_path.read_text(encoding="utf-8").splitlines()
        hdr = lines[0].split("\t")
        psc_col_idxs = [
            c_i for c_i, c_name in enumerate(hdr[2:], start=2) if any(kw in c_name for kw in psc_keywords)
        ]
        for ln in lines[1:]:
            parts = ln.split("\t")
            tgt = parts[0]
            if tgt not in gene_to_idx or tgt == tf:
                continue
            tgt_j = gene_to_idx[tgt]
            # ChIP-Atlas MACS2 score >= 200 corresponds to stringent q < 10^-20 promoter binding
            if float(parts[1]) >= 200.0:
                chip_any_edge[tf_i, tgt_j] = True
            if psc_col_idxs and max(float(parts[c_i]) for c_i in psc_col_idxs) >= 200.0:
                chip_psc_edge[tf_i, tgt_j] = True

        if 3 <= int(np.sum(chip_any_edge[tf_i])) <= n_genes - 4:
            any_tf_idx_list.append(tf_i)
        if 3 <= int(np.sum(chip_psc_edge[tf_i])) <= n_genes - 4:
            psc_tf_idx_list.append(tf_i)

    return HiPSCPerturbSeqData(
        genes=genes,
        ko_genes=ko_genes,
        ko_gene_indices=ko_indices,
        obs_days=days,
        basal_expression=basal,
        ko_delta_trajectories=ko_deltas,
        ko_binary_masks=ko_masks,
        eval_target_masks=eval_masks,
        cell_counts=counts,
        renge_reference_graph=renge_graph,
        chip_psc_tf_indices=np.array(psc_tf_idx_list, dtype=np.int64),
        chip_psc_edge_mask=chip_psc_edge,
        chip_any_tf_indices=np.array(any_tf_idx_list, dtype=np.int64),
        chip_any_edge_mask=chip_any_edge,
        file_sha256=hashes,
    )


NFKB_PINNED_SHA256: dict[str, str] = {
    "mRNA.nfkbgene.rpkm.csv": "a4b91c3411a4fd9fe1ca0ec6d4dbe5db6e2dc7e3d87d039cdfb5665eb9f9724d",
    "Final_EMSA_ifnar.csv": "2257cf343a9f398fa584a9abe73239ca2c170a93b6ffc02ee04d10ead9ee86c9",
    "Final_EMSA_ifnarikba.csv": "d72cd809992c07a4a2c3878f4561fe8c66e6d03aea9cc0780e46afbc98876be8",
}


@dataclass(frozen=True, kw_only=True)
class NfkbWashoutData:
    """Sen et al. (2020) 177-gene mouse fibroblast NF-kB 30-min pulse washout dataset."""

    ligands: tuple[str, ...]
    genotypes: tuple[str, ...]
    genes: tuple[str, ...]
    obs_hours: np.ndarray
    dt_hours: float
    obs_step_indices: tuple[int, ...]
    emsa_basal_subtracted: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    rna_responses: dict[tuple[str, str, int], np.ndarray]
    file_sha256: dict[str, str]


def fetch_and_prepare_nfkb_washout(raw_dir: Path) -> NfkbWashoutData:
    """Fetch and SHA-256 verify Sen et al. (2020) fibroblast EMSA + RNA-seq washout arrays."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/{NFKB_GIT_REPO}/{NFKB_GIT_REVISION}/data"

    hashes: dict[str, str] = {}
    for fname, expected in NFKB_PINNED_SHA256.items():
        dest = raw_dir / fname
        if not dest.exists():
            resp = requests.get(f"{base_url}/{fname}", timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        digest = sha256_file(dest)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch on {fname}: {digest} != {expected}")
        hashes[fname] = digest

    emsa_curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for geno, fn in (("ifnr", "Final_EMSA_ifnar.csv"), ("ifnrikba", "Final_EMSA_ifnarikba.csv")):
        df_e = pd.read_csv(raw_dir / fn)
        t_h = df_e.iloc[:, 0].to_numpy(dtype=np.float64) / 60.0
        for lig in NFKB_LIGANDS:
            raw_v = df_e[lig].to_numpy(dtype=np.float64)
            emsa_curves[(lig, geno)] = (t_h, np.maximum(0.0, raw_v - raw_v[0]))

    rpkm_df = pd.read_csv(raw_dir / "mRNA.nfkbgene.rpkm.csv", index_col=0)
    genes = tuple(str(g) for g in rpkm_df.index)
    obs_h = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0], dtype=np.float64)
    time_strs = ("0", "0.5", "1", "2", "3", "5", "8")
    lig_code = {"LPS": ".l", "TNF": ".t", "IL1": ""}

    rna_resp: dict[tuple[str, str, int], np.ndarray] = {}
    for lig in NFKB_LIGANDS:
        for geno in ("ifnr", "ifnrikba"):
            for rep in (1, 2):
                cols = [f"X{ts}.{geno}{lig_code[lig]}.{rep}" for ts in time_strs]
                vals = np.log2(1.0 + rpkm_df[cols].to_numpy(dtype=np.float64))
                # Zero-origin response: (7, 177)
                rna_resp[(lig, geno, rep)] = (vals - vals[:, 0:1]).T

    # dt = 0.05 h (3.0 min) -> 160 RK4 steps over [0, 8] h; obs indices at [0, 10, 20, 40, 60, 100, 160]
    obs_idx = tuple(int(round(h / 0.05)) for h in obs_h)
    return NfkbWashoutData(
        ligands=NFKB_LIGANDS,
        genotypes=("ifnr", "ifnrikba"),
        genes=genes,
        obs_hours=obs_h,
        dt_hours=0.05,
        obs_step_indices=obs_idx,
        emsa_basal_subtracted=emsa_curves,
        rna_responses=rna_resp,
        file_sha256=hashes,
    )


# =============================================================================
# Cook & Vanderhyden (2020) Human EMT Multi-Ligand Induction & Withdrawal scRNA-seq
# Accession: GSE147405 (Nature Communications 11, 2142, doi:10.1038/s41467-020-16066-2)
# =============================================================================

EMT_GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147405/suppl"
EMT_CELL_LINES: tuple[str, ...] = ("A549", "DU145", "MCF7", "OVCA420")
EMT_LIGANDS: tuple[str, ...] = ("TGFB1", "EGF", "TNF")
EMT_TIME_LABELS: tuple[str, ...] = ("0d", "8h", "1d", "3d", "7d", "8h_rm", "1d_rm", "3d_rm")
EMT_TIME_DAYS: tuple[float, ...] = (0.0, 1.0 / 3.0, 1.0, 3.0, 7.0, 7.0 + 1.0 / 3.0, 8.0, 10.0)

EMT_TFS: tuple[str, ...] = (
    "SMAD3", "SMAD4", "SMAD7", "SNAI1", "JUN", "JUNB", "JUND", "FOS", "FOSL1",
    "ATF3", "EGR1", "ETS2", "RELA", "RELB", "NFKB1", "NFKB2", "STAT1", "STAT3",
    "IRF1", "MYC", "KLF4", "KLF6", "HIF1A", "SOX4", "ID1", "ID2", "ID3", "CEBPB",
    "BHLHE40", "MAFF", "MAFK", "NFIL3", "ZFP36", "CSRNP1", "NR4A1", "HES1",
)
EMT_TARGETS: tuple[str, ...] = (
    "VIM", "FN1", "CDH1", "SERPINE1", "PMEPA1", "TGFBI", "ITGB1", "MMP9", "COL1A1",
    "EPCAM", "KRT8", "KRT18", "KRT19", "LAMC2", "LAMB3", "CXCL8", "CXCL1", "CXCL2",
    "CCL2", "ICAM1", "TNFAIP3", "NFKBIA", "BIRC3", "SOD2", "CTGF", "CYR61", "TPM1",
    "TAGLN", "MYL9", "PLAU", "PLAUR", "THBS1", "TIMP1", "CD44", "S100A4", "S100A6",
    "ANXA1", "ANXA2", "GADD45B", "IER3", "DUSP1", "DUSP4", "DUSP5", "DUSP6", "ERRFI1",
    "AREG", "HBEGF", "PHLDA1", "ETV4", "ETV5", "LIF", "IL32", "OAS1", "ISG15", "IFIT1",
    "LGALS1", "LOXL2", "CLDN4", "OCLN", "DSP", "MUC1", "KRT7", "S100A2", "ITGA2",
    "ITGA5", "ITGAV", "TGM2", "FST", "NDRG1", "VEGFA", "LDHA", "SLC2A1", "CDKN1A",
    "CCND1", "MKI67", "TOP2A",
)
EMT_PINNED_SHA256: dict[str, str] = {
    "GSE147405_A549_EGF_TimeCourse_UMI_matrix.csv.gz": "5a31be01fa8f856b35bc46c70e757f10be039c32075db2501282a1a357552e60",
    "GSE147405_A549_EGF_TimeCourse_metadata.csv.gz": "a376240d132b9330aefb82ad0361f61f965b1315b690437dddb7315475394c36",
    "GSE147405_A549_TGFB1_TimeCourse_UMI_matrix.csv.gz": "8c8d8bc4480b6e12b8e7982bf71e43ff7cb0b1158f48dfe6932af84f9aab9a31",
    "GSE147405_A549_TGFB1_TimeCourse_metadata.csv.gz": "e253a4fd1ee5b6479a719419b22f156638a633eb2992f9ba0290c6a3163e3100",
    "GSE147405_A549_TNF_TimeCourse_UMI_matrix.csv.gz": "b1466e9271b956ce0bdf0a2f8c72adbf79de018106c97052c36c7f3699e6b88c",
    "GSE147405_A549_TNF_TimeCourse_metadata.csv.gz": "f3424dc76e386e929cf4cd8413a45319be90ed4f09b83aa0befc0a0694ef319f",
    "GSE147405_DU145_EGF_TimeCourse_UMI_matrix.csv.gz": "d1a98372b02e77dd864ca5b61b1aea026278d13b9475b838c91de15075c07576",
    "GSE147405_DU145_EGF_TimeCourse_metadata.csv.gz": "513db590e0d4dcb70956f6657bd7b4297091f56ab46b3784122c6ceb0182be2b",
    "GSE147405_DU145_TGFB1_TimeCourse_UMI_matrix.csv.gz": "a8d9c18319616bf04a48648de9e0cbcee86e6579919bf625132a9cd12e88719e",
    "GSE147405_DU145_TGFB1_TimeCourse_metadata.csv.gz": "79434ebea01b1d8b46587ea16106e44e813bfb127de5ddb4e265cd29aa19a2be",
    "GSE147405_DU145_TNF_TimeCourse_UMI_matrix.csv.gz": "e2282c40e03177342dce38897a4c18757bff3c18f2dcbc1d0722394c385939f2",
    "GSE147405_DU145_TNF_TimeCourse_metadata.csv.gz": "ba0a2bd247abf1aa5c5e05c30cb9f548a3ef4abd5c421fd09f2440d3fc4d602e",
    "GSE147405_MCF7_EGF_TimeCourse_UMI_matrix.csv.gz": "0f78930fb1b8e7c9409114bcbfcb3dc9e040eb77041a8a0d32a12cb11e8ee1fb",
    "GSE147405_MCF7_EGF_TimeCourse_metadata.csv.gz": "29cc6d2bc5f7b8daa26ea7283aa77fa78e8dc33ef000d8c90e029d21ffc8a0b3",
    "GSE147405_MCF7_TGFB1_TimeCourse_UMI_matrix.csv.gz": "e48bd278c2eb0eb8efcbf1858fe4f91e392b9881f2d6a35baef7adb46c24c910",
    "GSE147405_MCF7_TGFB1_TimeCourse_metadata.csv.gz": "bedc881a7ad9ebaed0c00f05c0524b85cf63fc6c58c35acb7b4f86bc2474ba58",
    "GSE147405_MCF7_TNF_TimeCourse_UMI_matrix.csv.gz": "a485a6d3d86bba8ff26038b08e659d21b27da66ea5de9c8407a7f20acd0ff0d0",
    "GSE147405_MCF7_TNF_TimeCourse_metadata.csv.gz": "086bb76700d0354bc1a18c99e30db4b457a4bb4fc9606d9f0505af9587a9b91a",
    "GSE147405_OVCA420_EGF_TimeCourse_UMI_matrix.csv.gz": "6da0febf3820456cc9afdce381dc7a87f8c2a2aaa2e8296831df64213ec987a9",
    "GSE147405_OVCA420_EGF_TimeCourse_metadata.csv.gz": "6be1e9fdc5aeea24b19c94ddd5ad82c74ef34384b988e3cfa33bb31e16e03b6e",
    "GSE147405_OVCA420_TGFB1_TimeCourse_UMI_matrix.csv.gz": "51c22a51691b618110c80c51e039171f34e9464fdef7a0d3c3e44c58cfb0511e",
    "GSE147405_OVCA420_TGFB1_TimeCourse_metadata.csv.gz": "9026a5d52d5a6ec92de98a9c80d234daa7ccf74b62d702fa4f4e7f2e0f6ae5a5",
    "GSE147405_OVCA420_TNF_TimeCourse_UMI_matrix.csv.gz": "48ab105e8d53b7d52622161507e28a4c2cf5aac155269b1f9f26d3f041b20330",
    "GSE147405_OVCA420_TNF_TimeCourse_metadata.csv.gz": "0595779b67a285f76300680723fc88498b17accc99cc6044e346452e1a85a57c",
}


@dataclass(frozen=True, kw_only=True)
class EMTWithdrawalData:
    """Cook & Vanderhyden (2020) GSE147405 multi-ligand EMT induction + withdrawal scRNA-seq."""

    cell_lines: tuple[str, ...]
    ligands: tuple[str, ...]
    conditions: tuple[tuple[str, str], ...]
    genes: tuple[str, ...]
    tf_indices: np.ndarray
    time_labels: tuple[str, ...]
    obs_days: np.ndarray
    delta_trajectories: np.ndarray  # (12, 8, 112) baseline-subtracted log1p(CP10k)
    basal_expression: np.ndarray  # (12, 112) 0d expression per condition
    ligand_one_hot: np.ndarray  # (12, 3) binary indicator of TGFB1/EGF/TNF
    cell_counts: np.ndarray  # (12, 8) QC singlet counts per (condition, timepoint)
    total_singlet_cells: int
    file_sha256: dict[str, str]


def fetch_and_prepare_emt_withdrawal(raw_dir: Path) -> EMTWithdrawalData:
    """Fetch, SHA-256 verify, and pseudobulk GSE147405 (53,290 human singlets, 112 EMT/TF genes)."""
    import gzip

    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for fname, expected in EMT_PINNED_SHA256.items():
        dest = raw_dir / fname
        if not dest.exists():
            resp = requests.get(f"{EMT_GEO_BASE}/{fname}", timeout=180)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        digest = sha256_file(dest)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch on {fname}: {digest} != {expected}")
        hashes[fname] = digest

    all_genes = EMT_TFS + EMT_TARGETS
    tf_idx = np.arange(len(EMT_TFS), dtype=np.int64)
    conditions = tuple((cl, lig) for cl in EMT_CELL_LINES for lig in EMT_LIGANDS)
    cache_npz = raw_dir / "emt_gse147405_112genes_cache.npz"

    if cache_npz.exists():
        z = np.load(cache_npz)
        delta_arr = np.asarray(z["delta_trajectories"], dtype=np.float64)
        basal_arr = np.asarray(z["basal_expression"], dtype=np.float64)
        counts_arr = np.asarray(z["cell_counts"], dtype=np.int64)
    else:
        sel = set(all_genes)
        delta_list = []
        basal_list = []
        counts_list = []
        for cl, lig in conditions:
            meta = pd.read_csv(
                raw_dir / f"GSE147405_{cl}_{lig}_TimeCourse_metadata.csv.gz", index_col=0
            )
            if "Doublet" in meta.columns:
                meta = meta[meta["Doublet"] == "Singlet"]
            umi_p = raw_dir / f"GSE147405_{cl}_{lig}_TimeCourse_UMI_matrix.csv.gz"
            with gzip.open(umi_p, "rt") as f:
                cols = [c.strip('"') for c in f.readline().strip().split(",")]
                if cols[0] == "":
                    cols = cols[1:]
                col_to_idx = {c: i for i, c in enumerate(cols)}
                keep_cells = [c for c in meta.index if c in col_to_idx]
                keep_idx = np.array([col_to_idx[c] for c in keep_cells], dtype=np.int64)
                lib_size = meta.loc[keep_cells, "nCount_RNA"].to_numpy(dtype=np.float64)
                times = meta.loc[keep_cells, "Time"].to_numpy()
                t_masks = [times == t for t in EMT_TIME_LABELS]
                t_cnts = [int(np.sum(m)) for m in t_masks]
                if min(t_cnts) <= 0:
                    raise ValueError(f"Zero cells for {cl}/{lig}: {t_cnts}")
                counts_list.append(t_cnts)
                gene_rows: dict[str, list[float]] = {}
                for line in f:
                    gname, rest = line.split(",", 1)
                    gname = gname.strip('"')
                    if gname in sel:
                        vals = np.fromstring(rest, sep=",", dtype=np.float64)[keep_idx]
                        log_cp10k = np.log1p(vals / np.maximum(1.0, lib_size) * 1e4)
                        gene_rows[gname] = [float(np.mean(log_cp10k[m])) for m in t_masks]
            mat = np.array([gene_rows[g] for g in all_genes], dtype=np.float64).T  # (8, 112)
            if not np.all(np.isfinite(mat)):
                raise ValueError(f"Non-finite pseudobulk in {cl}/{lig}")
            basal_list.append(mat[0])
            delta_list.append(mat - mat[0:1])
        delta_arr = np.stack(delta_list, axis=0)
        basal_arr = np.stack(basal_list, axis=0)
        counts_arr = np.asarray(counts_list, dtype=np.int64)
        np.savez(
            cache_npz,
            delta_trajectories=delta_arr,
            basal_expression=basal_arr,
            cell_counts=counts_arr,
        )

    lig_oh = np.zeros((len(conditions), len(EMT_LIGANDS)), dtype=np.float64)
    for c_i, (_cl, lig) in enumerate(conditions):
        lig_oh[c_i, EMT_LIGANDS.index(lig)] = 1.0

    return EMTWithdrawalData(
        cell_lines=EMT_CELL_LINES,
        ligands=EMT_LIGANDS,
        conditions=conditions,
        genes=all_genes,
        tf_indices=tf_idx,
        time_labels=EMT_TIME_LABELS,
        obs_days=np.asarray(EMT_TIME_DAYS, dtype=np.float64),
        delta_trajectories=delta_arr,
        basal_expression=basal_arr,
        ligand_one_hot=lig_oh,
        cell_counts=counts_arr,
        total_singlet_cells=int(np.sum(counts_arr)),
        file_sha256=hashes,
    )
