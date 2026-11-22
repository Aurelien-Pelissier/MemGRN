"""Large-Scale Closed-Loop Primary Human CD4+ T-Cell Perturb-seq Benchmark (GSE314342).

Evaluates autonomous closed-loop regulatory dynamics (zero future TF traces supplied) at real
large scale (S = 128 clean on-target human TFs, N = 256 genes = 128 TFs + 128 immune effector
targets; 221,398 primary human CD4+ T cells from Zhu, Dann, Pritchard, Marson et al., Cell 2026 /
bioRxiv Dec 2025, GSE314342) across three longitudinal activation states (Rest 0h, Stim8hr,
Stim48hr):
  1. Task 1 (5-Fold Unseen Human TF-Knockdown Autonomous 8h -> 48h Cascade Forecasting):
     Hold out 25-26 completely unseen knocked-down human TFs per fold across all 128 TFs.
     Warm-start intracellular memory (z, h) via observed Rest -> Stim8hr states (clamp_steps=4),
     then release y(t) autonomously in closed loop over 40 hours (8h -> 48h, zero future traces)
     to forecast all 256 genes (edited gene j == k masked) at Stim48hr.
  2. Task 2 (External Curated Human CollecTRI Causal GRN Recovery across 32,640 Directed Pairs):
     Evaluate the 5-fold averaged early-time counterfactual knockout sensitivity matrix
     (128 TFs x 256 genes, off-diagonal) against 715 curated human CollecTRI TF->target edges.
  3. Task 3 (Real-Human S = 128 Parameter & Runtime Scaling):
     Contrast O(SR) memory scaling of Closed-Loop MemGRN against O(S^2 R^2) coupled baselines.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_env_var] = "1"
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    ClosedLoopModelFamily,
    ClosedLoopSpec,
    fit_closed_loop_network,
)
from memgrn.datasets import fetch_and_prepare_marson_cd4t_perturbseq  # noqa: E402

OUT_DIR = ROOT / "results" / "cd4t_closed_loop"
RAW_DIR = ROOT / "data" / "raw" / "marson_cd4t"

N_TFS = 128
N_TARGETS = 128
DT = 0.25  # 2.5 h per step (in 10-h units): steps 0..2 = 0->Rest, 2..4 = Rest->8h, 4..20 = 8h->48h (40h)
N_STEPS = 20
CLAMP_STEPS = 4
OBS_STEP_INDICES = (0, 2, 4, 20)

CLOSED_LOOP_FAMILIES: tuple[ClosedLoopModelFamily, ...] = (
    ClosedLoopModelFamily.MEMGRN,
    ClosedLoopModelFamily.LOCAL_CONST,
    ClosedLoopModelFamily.LTC_LOCAL,
    ClosedLoopModelFamily.SELECTIVE_SSM,
    ClosedLoopModelFamily.CFC,
    ClosedLoopModelFamily.COUPLED_ADAPTIVE,
    ClosedLoopModelFamily.LTC_COUPLED,
    ClosedLoopModelFamily.PERTURB_ODE,
    ClosedLoopModelFamily.LATENT_ODE,
)


def _auroc_and_ap(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores)
    y_true = labels[order].astype(np.float64)
    n_pos = float(np.sum(y_true))
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5, 0.0
    cum_pos = np.cumsum(y_true)
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    ap = float(np.sum((cum_pos / ranks) * y_true) / n_pos)
    auroc = float(np.sum((1.0 - (ranks - cum_pos) / n_neg) * y_true) / n_pos)
    return auroc, ap


def _fit_renge_baseline(
    train_ko: np.ndarray,
    train_y3: np.ndarray,
    basal: np.ndarray,
    reg_indices: np.ndarray,
    n_powers: int = 3,
) -> tuple[np.ndarray, callable]:
    """Static knockout power-series propagation RENGE baseline."""
    n_tr, n_t, n_g = train_y3.shape
    x_ko = np.zeros_like(train_ko)
    x_ko[:, reg_indices] = -train_ko[:, reg_indices] * basal[None, reg_indices]
    y_mean = np.mean(train_y3, axis=1)
    ridge = Ridge(alpha=5.0, fit_intercept=False)
    ridge.fit(x_ko, y_mean)
    a_mat = ridge.coef_.copy() * (1.0 - np.eye(n_g))
    norm_a = np.linalg.norm(a_mat, ord=2)
    if norm_a > 0.85:
        a_mat = a_mat * (0.85 / norm_a)

    powers = []
    cur = x_ko @ a_mat.T
    for _ in range(n_powers):
        powers.append(cur)
        cur = (cur * (1.0 - train_ko)) @ a_mat.T
    p_stack = np.stack(powers, axis=-1)
    feat = p_stack.reshape(n_tr * n_g, n_powers)
    tgt48 = train_y3[:, 2, :].reshape(n_tr * n_g)
    w48, _, _, _ = np.linalg.lstsq(
        feat.T @ feat + 1e-2 * np.eye(n_powers), feat.T @ tgt48, rcond=None
    )

    def _predict_48(ko_test: np.ndarray) -> np.ndarray:
        xk = np.zeros_like(ko_test)
        xk[:, reg_indices] = -ko_test[:, reg_indices] * basal[None, reg_indices]
        pl = []
        c = xk @ a_mat.T
        for _ in range(n_powers):
            pl.append(c)
            c = (c * (1.0 - ko_test)) @ a_mat.T
        return np.einsum("bgm,m->bg", np.stack(pl, axis=-1), w48)

    return np.abs(a_mat.T[: len(reg_indices), :]), _predict_48


def _run_fold_job(job: tuple[int, str, list[list[int]]]) -> tuple[int, str, dict]:
    fold_i, fam_val, folds = job
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    data = fetch_and_prepare_marson_cd4t_perturbseq(RAW_DIR, N_TFS, N_TARGETS)
    n_ko = len(data.ko_genes)
    n_genes = len(data.genes)
    reg_tuple = tuple(int(i) for i in data.ko_gene_indices)

    test_idx = folds[fold_i]
    val_idx = folds[(fold_i + 1) % len(folds)]
    train_idx = [i for i in range(n_ko) if i not in test_idx and i not in val_idx]

    tr_raw = data.ko_delta_trajectories[train_idx]
    gene_scales = np.maximum(0.10, np.sqrt(np.mean(tr_raw**2, axis=(0, 1))))
    basal_scaled = data.basal_expression / gene_scales

    # Top-50 dynamic targets identified strictly on training TFs (zero test leakage)
    tr_rms_g = np.sqrt(np.mean(tr_raw[:, 2, :] ** 2, axis=0))
    top50_idx = np.argsort(-tr_rms_g)[:50]

    def _pack(idxs: list[int]):
        ko_b = data.ko_binary_masks[idxs]
        y3 = data.ko_delta_trajectories[idxs] / gene_scales[None, None, :]  # (B, 3, 256)
        y_obs = np.concatenate([np.zeros((len(idxs), 1, n_genes)), y3], axis=1)
        w_obs = np.zeros_like(y_obs)
        w_obs[:, 3, :] = data.eval_target_masks[idxs, 2, :]  # score strictly on Stim48hr off-diagonal
        y_cl = np.zeros((N_STEPS, len(idxs), n_genes), dtype=np.float64)
        for s in range(2):
            y_cl[s] = ((s + 1) / 2.0) * y3[:, 0, :]
        for s in range(2, 4):
            f = (s - 1) / 2.0
            y_cl[s] = (1.0 - f) * y3[:, 0, :] + f * y3[:, 1, :]
        return ko_b, y3, y_obs, w_obs, y_cl

    tr_ko, tr_y3, tr_y, tr_w, tr_cl = _pack(train_idx)
    va_ko, _, va_y, va_w, va_cl = _pack(val_idx)
    te_ko, te_y3, te_y, te_w, te_cl = _pack(test_idx)

    t0 = time.perf_counter()
    if fam_val == "zero":
        pred48 = np.zeros_like(te_y[:, 3, :])
        n_params, mem_params = 0, 0
        graph = np.zeros((N_TFS, n_genes), dtype=np.float64)
    elif fam_val == "persistence":
        pred48 = te_y3[:, 1, :].copy()  # naive copy y(48h) = y(8h)
        n_params, mem_params = 0, 0
        graph = np.zeros((N_TFS, n_genes), dtype=np.float64)
    elif fam_val == "renge":
        graph, pred_fn = _fit_renge_baseline(
            tr_ko, tr_y3, basal_scaled, data.ko_gene_indices
        )
        pred48 = pred_fn(te_ko)
        n_params = N_TFS * n_genes + 3
        mem_params = 0
    else:
        fam = ClosedLoopModelFamily(fam_val)
        spec = ClosedLoopSpec(
            family=fam,
            n_genes=n_genes,
            regulator_indices=reg_tuple,
            hollow=True,
            use_cooperativity=(fam == ClosedLoopModelFamily.MEMGRN),
            coop_rank=8,
            readout_rank=(
                24
                if fam in (ClosedLoopModelFamily.MEMGRN, ClosedLoopModelFamily.LOCAL_CONST)
                else None
            ),
            base_rates=(0.4, 1.2, 3.5),
            n_channels=3,
            hidden_width=5,
            module_dim=48,
            latent_dim=48,
        )
        fitted = fit_closed_loop_network(
            spec=spec,
            basal_levels=basal_scaled,
            train_ko=tr_ko,
            train_y=tr_y,
            train_weights=tr_w,
            train_y_clamp=tr_cl,
            val_ko=va_ko,
            val_y=va_y,
            val_weights=va_w,
            val_y_clamp=va_cl,
            dt=DT,
            n_steps=N_STEPS,
            obs_step_indices=OBS_STEP_INDICES,
            clamp_steps=CLAMP_STEPS,
            seeds=(101, 202),
            reg_grid=(1e-3, 1e-2, 5e-2),
            group_l21_scale=(0.01 if fam == ClosedLoopModelFamily.MEMGRN else 0.0),
            total_updates=450,
            lr_max=0.005,
        )
        pred48 = fitted.predict_ko_fn(te_ko, None, None, te_cl)[:, 3, :]
        n_params = fitted.n_params
        # Count memory-module parameters separately from explicit direct/readout weights
        from memgrn.baselines import _init_closed_loop_tree

        tree0 = _init_closed_loop_tree(spec, seed=0)
        readout_keys = {
            "direct",
            "delayed",
            "delayed_u",
            "delayed_v",
            "readout",
            "decay",
            "in_gain",
            "coop_u",
            "coop_v",
            "coop_w",
        }
        mem_params = int(sum(int(v.size) for k, v in tree0.items() if k not in readout_keys))
        graph = fitted.interaction_matrix[:N_TFS, :]
    fold_sec = time.perf_counter() - t0

    true48 = te_y[:, 3, :]
    mask48 = te_w[:, 3, :]
    sig_mask48 = mask48 * data.sig_de_masks[test_idx, 2, :].astype(np.float64)
    top50_mask = np.zeros_like(mask48)
    top50_mask[:, top50_idx] = mask48[:, top50_idx]

    per_tf_mse = []
    per_tf_top50 = []
    per_tf_sig = []
    per_tf_r = []
    for b_i in range(len(test_idx)):
        m = mask48[b_i] > 0.5
        e2 = (pred48[b_i] - true48[b_i]) ** 2
        per_tf_mse.append(float(np.mean(e2[m])))
        mt = top50_mask[b_i] > 0.5
        per_tf_top50.append(float(np.mean(e2[mt])))
        ms = sig_mask48[b_i] > 0.5
        if np.any(ms):
            per_tf_sig.append(float(np.mean(e2[ms])))
        if np.std(pred48[b_i, m]) > 1e-8 and np.std(true48[b_i, m]) > 1e-8:
            per_tf_r.append(float(np.corrcoef(pred48[b_i, m], true48[b_i, m])[0, 1]))
        else:
            per_tf_r.append(0.0)

    fold_mse = float(np.sum(((pred48 - true48) ** 2) * mask48) / np.sum(mask48))
    fold_var = float(np.sum((true48**2) * mask48) / np.sum(mask48)) + 1e-8
    fold_r2 = 1.0 - fold_mse / fold_var
    fold_sig = float(
        np.sum(((pred48 - true48) ** 2) * sig_mask48) / max(1.0, np.sum(sig_mask48))
    )
    fold_top50 = float(
        np.sum(((pred48 - true48) ** 2) * top50_mask) / max(1.0, np.sum(top50_mask))
    )

    return (
        fold_i,
        fam_val,
        {
            "n_params": n_params,
            "mem_params": mem_params,
            "fold_seconds": fold_sec,
            "mse": fold_mse,
            "r2": fold_r2,
            "top50_mse": fold_top50,
            "sig_de_mse": fold_sig,
            "pearson_r": float(np.mean(per_tf_r)),
            "test_tf_indices": test_idx,
            "per_tf_mse": per_tf_mse,
            "per_tf_top50": per_tf_top50,
            "per_tf_sig": per_tf_sig,
            "graph": graph.tolist(),
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_marson_cd4t_perturbseq(RAW_DIR, N_TFS, N_TARGETS)
    n_ko = len(data.ko_genes)
    folds = [list(range(f, n_ko, 5)) for f in range(5)]
    all_models = [*[m.value for m in CLOSED_LOOP_FAMILIES], "renge", "persistence", "zero"]

    design = {
        "benchmark": "primary_human_cd4t_closed_loop_perturbseq_gse314342",
        "accession": "GSE314342",
        "reference": "Zhu, Dann, Pritchard, Marson et al., Cell (2026) / bioRxiv (Dec 2025)",
        "total_cells_in_panel": int(np.sum(data.cell_counts)),
        "n_tfs": N_TFS,
        "n_genes": len(data.genes),
        "obs_hours": data.obs_hours.tolist(),
        "n_collectri_edges": int(np.sum(data.collectri_edge_mask)),
        "folds": folds,
        "models": all_models,
        "future_tf_traces_supplied": False,
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    jobs = [(f_i, mk, folds) for f_i in range(5) for mk in all_models]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=24, mp_context=ctx) as pool:
        raw_out = list(pool.map(_run_fold_job, jobs))
    elapsed = time.time() - t0

    by_model: dict[str, list[dict]] = {m: [] for m in all_models}
    graphs_by_model: dict[str, list[np.ndarray]] = {m: [] for m in all_models}
    for _, mk, res in raw_out:
        graphs_by_model[mk].append(np.asarray(res.pop("graph"), dtype=np.float64))
        by_model[mk].append(res)

    off_mask = np.ones((N_TFS, len(data.genes)), dtype=bool)
    for i in range(N_TFS):
        off_mask[i, i] = False
    col_labels = data.collectri_edge_mask[off_mask]
    col_base_rate = float(np.mean(col_labels))

    summary: dict[str, dict] = {}
    per_tf_mse_map: dict[str, np.ndarray] = {}
    for mk in all_models:
        fl = by_model[mk]
        # Align per-TF MSE vector across all 128 held-out human TFs
        tf_mse = np.zeros(n_ko, dtype=np.float64)
        tf_t50 = np.zeros(n_ko, dtype=np.float64)
        for fd in fl:
            for idx_tf, v_m, v_50 in zip(fd["test_tf_indices"], fd["per_tf_mse"], fd["per_tf_top50"]):
                tf_mse[idx_tf] = v_m
                tf_t50[idx_tf] = v_50
        per_tf_mse_map[mk] = tf_mse

        # 5-fold averaged directed causal graph -> CollecTRI regulon discovery
        mean_g = np.mean(np.stack(graphs_by_model[mk], axis=0), axis=0)
        if np.max(mean_g) > 1e-12:
            col_norm = mean_g / np.maximum(1e-8, np.max(mean_g, axis=0, keepdims=True))
            auroc_n, ap_n = _auroc_and_ap(col_norm[off_mask], col_labels)
            auroc_r, ap_r = _auroc_and_ap(mean_g[off_mask], col_labels)
        else:
            auroc_n, ap_n, auroc_r, ap_r = 0.5, col_base_rate, 0.5, col_base_rate

        summary[mk] = {
            "n_params": fl[0]["n_params"],
            "mem_params": fl[0]["mem_params"],
            "mean_fold_seconds": float(np.mean([x["fold_seconds"] for x in fl])),
            "unseen_tf_48h_mse": float(np.mean(tf_mse)),
            "unseen_tf_48h_mse_se": float(np.std(tf_mse, ddof=1) / np.sqrt(n_ko)),
            "unseen_tf_48h_r2": float(np.mean([x["r2"] for x in fl])),
            "unseen_tf_48h_pearson_r": float(np.mean([x["pearson_r"] for x in fl])),
            "unseen_tf_top50_mse": float(np.mean(tf_t50)),
            "unseen_tf_top50_mse_se": float(np.std(tf_t50, ddof=1) / np.sqrt(n_ko)),
            "unseen_tf_sig_de_mse": float(np.mean([x["sig_de_mse"] for x in fl])),
            "collectri_auroc": auroc_n,
            "collectri_ap": ap_n,
            "collectri_ap_over_random": float(ap_n / max(1e-8, col_base_rate)),
            "collectri_raw_auroc": auroc_r,
            "collectri_raw_ap_over_random": float(ap_r / max(1e-8, col_base_rate)),
        }

    # Paired Wilcoxon across ALL 128 held-out primary human TFs vs cl_memgrn!
    mem_vec = per_tf_mse_map["cl_memgrn"]
    wilcoxon_128: dict[str, dict] = {}
    for mk in all_models:
        if mk == "cl_memgrn":
            continue
        oth = per_tf_mse_map[mk]
        p_val = float(wilcoxon(mem_vec, oth, alternative="less").pvalue) if np.any(mem_vec != oth) else 1.0
        wilcoxon_128[mk] = {
            "pct_reduction": float(100.0 * (1.0 - np.mean(mem_vec) / max(1e-12, np.mean(oth)))),
            "wins_128": int(np.sum(mem_vec < oth)),
            "wilcoxon_p": p_val,
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "wilcoxon_128_tfs_vs_memgrn": wilcoxon_128,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": round(elapsed, 1), "summary": summary, "wilcoxon": wilcoxon_128}, indent=2))


if __name__ == "__main__":
    main()
