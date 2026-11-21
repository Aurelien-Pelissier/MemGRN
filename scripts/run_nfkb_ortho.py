"""Orthogonalized Semimechanistic Cofactor Benchmark on Fibroblast NF-kB Washout (177 Genes).

Evaluates whether MemGRN-Ortho (source-isolated asymmetric NF-kB channels + L2(P_train)
projection-orthogonalized low-rank cofactor module) resolves unmeasured secondary pathways
(such as LPS-TRIF-IRF3) on Sen et al. (2020) cross-ligand holdouts while preserving pure
observed NF-kB counterfactual attribution.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.baselines import (  # noqa: E402
    DynamicalModelFamily,
    ModelSpec,
    counterfactual_source_attribution,
    fit_dynamical_model,
    fit_two_stage_memgrn_ortho,
)
from memgrn.datasets import fetch_and_prepare_nfkb_washout  # noqa: E402

OUT_DIR = ROOT / "results" / "nfkb_ortho"
RAW_DIR = ROOT / "data" / "raw" / "nfkb"

BASE_MODELS: tuple[DynamicalModelFamily, ...] = (
    DynamicalModelFamily.LOCAL_CONST,
    DynamicalModelFamily.MEMGRN,
    DynamicalModelFamily.PERTURB_ODE,
    DynamicalModelFamily.LATENT_ODE,
)


def _build_emsa_stage_grid(
    t_hours: np.ndarray,
    vals: np.ndarray,
    scale: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Linearly interpolate basal-subtracted EMSA onto RK4 (start, mid, end) knots."""
    s0 = np.arange(n_steps, dtype=np.float64) * dt
    sm = (np.arange(n_steps, dtype=np.float64) + 0.5) * dt
    s1 = (np.arange(n_steps, dtype=np.float64) + 1.0) * dt
    v = vals / scale
    return np.stack(
        [np.interp(s0, t_hours, v), np.interp(sm, t_hours, v), np.interp(s1, t_hours, v)],
        axis=-1,
    )[:, None, :]  # (n_steps, 1, 3)


def _run_outer_ligand(held_ligand: str) -> tuple[str, dict]:
    data = fetch_and_prepare_nfkb_washout(RAW_DIR)
    source_ligands = [lig for lig in data.ligands if lig != held_ligand]
    n_genes = len(data.genes)
    n_steps = 160  # 8.0 h / 0.05 h
    obs_stride = 10  # every 0.5 h -> 17 grid knots [0, 0.5, 1.0, ..., 8.0] h
    # Observed RNA times [0, 0.5, 1, 2, 3, 5, 8] h map to stride-0.5h indices [0, 1, 2, 4, 6, 10, 16]
    knot_positions = (0, 1, 2, 4, 6, 10, 16)
    post_washout_knots = (2, 4, 6, 10, 16)  # strictly post-washout: 1, 2, 3, 5, 8 h

    # Source-only input normalization max across the two fit ligands
    src_max = max(
        float(np.max(data.emsa_basal_subtracted[(lig, geno)][1]))
        for lig in source_ligands
        for geno in data.genotypes
    )

    def _pack_ligand_set(lig_list: list[str], post_only_mask: bool = False):
        u_list = []
        y_list = []
        m_list = []
        for lig in lig_list:
            for geno in data.genotypes:
                t_e, v_e = data.emsa_basal_subtracted[(lig, geno)]
                u_curve = _build_emsa_stage_grid(t_e, v_e, src_max, data.dt_hours, n_steps)
                for rep in (1, 2):
                    u_list.append(u_curve)
                    rna_7 = data.rna_responses[(lig, geno, rep)]  # (7, 177)
                    y_17 = np.zeros((17, n_genes), dtype=np.float64)
                    m_17 = np.zeros((17, n_genes), dtype=np.float64)
                    for r_idx, k_pos in enumerate(knot_positions):
                        y_17[k_pos] = rna_7[r_idx]
                        if post_only_mask:
                            if k_pos in post_washout_knots:
                                m_17[k_pos] = 1.0
                        elif k_pos > 0:
                            m_17[k_pos] = 1.0
                    y_list.append(y_17)
                    m_list.append(m_17)
        return (
            np.stack(u_list, axis=1),  # (160, B, 1, 3)
            np.stack(y_list, axis=0),  # (B, 17, 177)
            np.stack(m_list, axis=0),
        )

    # Inner cross-ligand split: train on source_ligands[0], validate post-washout on source_ligands[1]
    # and vice versa to select regularization & architecture hyperparameters cleanly on source data,
    # then fit on pooled source ligands with last-replicate validation holdout.
    tr_u_a, tr_y_a, tr_m_a = _pack_ligand_set([source_ligands[0]], post_only_mask=False)
    va_u_b, va_y_b, va_m_b = _pack_ligand_set([source_ligands[1]], post_only_mask=True)

    src_u_all, src_y_all, src_m_all = _pack_ligand_set(source_ligands, post_only_mask=False)
    _, _, src_post_m = _pack_ligand_set(source_ligands, post_only_mask=True)
    # Replicate-1 (indices 0,2,4,6) vs Replicate-2 (indices 1,3,5,7) across both source ligands & genotypes
    rep1_idx = [0, 2, 4, 6]
    rep2_idx = [1, 3, 5, 7]

    te_u, te_y, _ = _pack_ligand_set([held_ligand], post_only_mask=True)

    fold_res: dict[str, dict] = {}
    fitted_map = {}

    for fam in BASE_MODELS:
        spec = ModelSpec(
            family=fam,
            n_sources=1,
            n_targets=n_genes,
            n_channels=3,
            hidden_width=4,
            latent_dim=14,
            ortho_dim=6,
            module_dim=18,
        )
        reg_candidates = (1e-5, 1e-4, 1e-3, 5e-3)
        cv_fit = fit_dynamical_model(
            spec=spec,
            train_stage_u=tr_u_a,
            train_y=tr_y_a,
            train_mask=tr_m_a,
            val_stage_u=va_u_b,
            val_y=va_y_b,
            val_mask=va_m_b,
            dt=data.dt_hours,
            obs_stride=obs_stride,
            seeds=(416, 819),
            reg_grid=reg_candidates,
            total_updates=1200,
            lr_max=0.008,
            gate_shrinkage_grid=(1.0, 8.0),
            stimulus_window_steps=10,
        )
        best_reg = cv_fit.chosen_reg

        final_fit = fit_dynamical_model(
            spec=spec,
            train_stage_u=src_u_all[:, rep1_idx],
            train_y=src_y_all[rep1_idx],
            train_mask=src_m_all[rep1_idx],
            val_stage_u=src_u_all[:, rep2_idx],
            val_y=src_y_all[rep2_idx],
            val_mask=src_post_m[rep2_idx],
            dt=data.dt_hours,
            obs_stride=obs_stride,
            seeds=(416, 819),
            reg_grid=(best_reg,),
            total_updates=1400,
            lr_max=0.008,
            gate_shrinkage_grid=(1.0, 8.0),
            stimulus_window_steps=10,
        )
        fitted_map[fam.value] = final_fit

    # Fit Proposition 1(iii) two-stage orthogonalized cofactor module atop frozen source MemGRN
    fitted_map["memgrn_ortho"] = fit_two_stage_memgrn_ortho(
        base_memgrn=fitted_map["memgrn"],
        train_stage_u=src_u_all[:, rep1_idx],
        train_y=src_y_all[rep1_idx],
        train_mask=src_m_all[rep1_idx],
        val_stage_u=src_u_all[:, rep2_idx],
        val_y=src_y_all[rep2_idx],
        val_mask=src_post_m[rep2_idx],
        dt=data.dt_hours,
        obs_stride=obs_stride,
        latent_dim=14,
        seeds=(416, 819),
        reg_grid=(1e-5, 1e-4, 1e-3, 1e-2),
        total_updates=1400,
        lr_max=0.008,
    )

    for m_key, final_fit in fitted_map.items():
        pred_te = final_fit.predict_fn(te_u, None)
        post_pred = pred_te[:, list(post_washout_knots), :]
        post_true = te_y[:, list(post_washout_knots), :]
        post_mse = float(np.mean((post_pred - post_true) ** 2))
        late_mse = float(np.mean((pred_te[:, [10, 16], :] - te_y[:, [10, 16], :]) ** 2))

        if final_fit.predict_observed_only_fn is not None:
            attr_vec = counterfactual_source_attribution(
                final_fit.predict_observed_only_fn, te_u, None
            )[0, :, list(post_washout_knots), :]
        else:
            attr_vec = counterfactual_source_attribution(final_fit.predict_fn, te_u, None)[
                0, :, list(post_washout_knots), :
            ]

        fold_res[m_key] = {
            "n_params": final_fit.n_params,
            "chosen_reg": final_fit.chosen_reg,
            "post_washout_mse": post_mse,
            "late_5_8h_mse": late_mse,
            "_attr_flat": attr_vec.ravel(),
        }

    # Measure cosine agreement of observed NF-kB attribution with pure source-isolated MemGRN
    ref_attr = fold_res["memgrn"]["_attr_flat"]
    ref_norm = float(np.linalg.norm(ref_attr)) + 1e-12
    for m_k, d_m in fold_res.items():
        v = d_m.pop("_attr_flat")
        cos_sim = float(np.dot(v, ref_attr) / (np.linalg.norm(v) * ref_norm + 1e-12))
        rel_l2_theft = float(np.linalg.norm(v - ref_attr) / ref_norm)
        d_m["nfkb_attribution_cosine_vs_pure"] = cos_sim
        d_m["nfkb_attribution_rel_drift"] = rel_l2_theft

    return held_ligand, fold_res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_and_prepare_nfkb_washout(RAW_DIR)

    all_model_keys = ("local_const", "memgrn", "memgrn_ortho", "perturb_ode", "latent_ode")
    design = {
        "benchmark": "nfkb_washout_orthogonal_cofactor_177genes",
        "repo": "biomystery/duration_decode_manuscript",
        "commit": "37be90e56f6e4a9c726990c09e0adc114dcc8efc",
        "ligands": list(data.ligands),
        "genotypes": list(data.genotypes),
        "n_genes": len(data.genes),
        "washout_hours": 0.5,
        "post_washout_eval_hours": [1.0, 2.0, 3.0, 5.0, 8.0],
        "models": list(all_model_keys),
        "file_sha256": data.file_sha256,
    }
    design_bytes = (json.dumps(design, indent=2) + "\n").encode("utf-8")
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    (OUT_DIR / "design_lock.json").write_bytes(design_bytes)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        by_ligand = dict(pool.map(_run_outer_ligand, list(data.ligands)))
    elapsed = time.time() - t0

    summary: dict[str, dict] = {}
    for mk in all_model_keys:
        summary[mk] = {
            "n_params": by_ligand["IL1"][mk]["n_params"],
            "IL1_post_mse": by_ligand["IL1"][mk]["post_washout_mse"],
            "TNF_post_mse": by_ligand["TNF"][mk]["post_washout_mse"],
            "LPS_post_mse": by_ligand["LPS"][mk]["post_washout_mse"],
            "mean_post_mse": float(
                np.mean([by_ligand[lig][mk]["post_washout_mse"] for lig in data.ligands])
            ),
            "mean_late_mse": float(
                np.mean([by_ligand[lig][mk]["late_5_8h_mse"] for lig in data.ligands])
            ),
            "mean_nfkb_attr_cosine": float(
                np.mean(
                    [by_ligand[lig][mk]["nfkb_attribution_cosine_vs_pure"] for lig in data.ligands]
                )
            ),
            "mean_nfkb_attr_drift": float(
                np.mean([by_ligand[lig][mk]["nfkb_attribution_rel_drift"] for lig in data.ligands])
            ),
        }

    out = {
        "design_sha256": design_sha,
        "elapsed_seconds": elapsed,
        "by_ligand": by_ligand,
        "summary": summary,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
