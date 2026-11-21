"""Audit archived SHA-256 receipts, replay live Msn2 ODE trajectories, and print summary metrics."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgrn.datasets import build_rk4_stage_grid, load_archived_json  # noqa: E402
from memgrn.dynamics import (  # noqa: E402
    MemGRNConfig,
    MemGRNParams,
    RateInputMode,
    rollout_memgrn_midpoint,
    rollout_memgrn_rk4,
)


def sha256_variants(path: Path) -> set[str]:
    """Return SHA-256 digests for LF and CRLF line-ending normalizations."""
    lf_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(lf_bytes).hexdigest(),
        hashlib.sha256(crlf_bytes).hexdigest(),
    }


def verify_msn2_frozen_readout_replay(stage3: Path) -> float:
    """Independently replay all 48 selected 6-promoter frozen-readout RK4 trajectories."""
    six_scores = load_archived_json(stage3, "msn2_six_promoter_rates/scores.json")
    audit = load_archived_json(stage3, "msn2_six_promoter_rates/audit.json")
    scores_path = stage3 / "msn2_six_promoter_rates" / "scores.json"
    if audit["scores"] not in sha256_variants(scores_path):
        raise AssertionError("SHA-256 mismatch on msn2_six_promoter_rates/scores.json")

    with np.load(stage3 / "msn2_repeated_pulses" / "predictions.npz") as rep_f:
        drivers = rep_f["drivers"]
    with np.load(
        stage3 / "neural_rate_msn2_transfer" / "final_neural_rate_gw.npz"
    ) as base_f:
        base_weights = {s: base_f[f"weights_{s}"] for s in (416, 819)}
    with np.load(stage3 / "msn2_six_promoter_rates" / "predictions.npz") as pred_f:
        archived_preds = dict(pred_f)

    all_genes = ["ALD3", "DCS2", "DDR2", "HXK1", "RTN2", "SIP18", "TKL2"]
    stage_grid = build_rk4_stage_grid(drivers, substeps_per_knot=8)
    dt = 2.5 / (60.0 * 8)

    # Unpack archived joint weights layout:
    # bias(4), decay(7), direct(1,7), input(3,4), mix(3,7), on(3), output(3,4), rate(3)
    def _unpack_joint(vec: np.ndarray) -> dict[str, np.ndarray]:
        shapes = [
            ("bias", (4,)),
            ("decay", (7,)),
            ("direct", (1, 7)),
            ("input", (3, 4)),
            ("mix", (3, 7)),
            ("on", (3,)),
            ("output", (3, 4)),
            ("rate", (3,)),
        ]
        out: dict[str, np.ndarray] = {}
        idx = 0
        for name, sh in shapes:
            sz = int(np.prod(sh))
            out[name] = vec[idx : idx + sz].reshape(sh)
            idx += sz
        return out

    def _unpack_rate(vec: np.ndarray, mode: RateInputMode) -> dict[str, np.ndarray]:
        if mode == RateInputMode.CONSTANT:
            return {"rate": vec.reshape(1, 3)}
        n_in = 1 if mode == RateInputMode.ACTIVITY else (2 if mode == RateInputMode.MEMORY else 3)
        shapes = [
            ("bias", (4,)),
            ("input", (n_in, 4)),
            ("output", (1, 3, 4)),
            ("rate", (1, 3)),
        ]
        out: dict[str, np.ndarray] = {}
        idx = 0
        for name, sh in shapes:
            sz = int(np.prod(sh))
            out[name] = vec[idx : idx + sz].reshape(sh)
            idx += sz
        return out

    max_diff = 0.0
    for gene in six_scores["genes"]:
        g_idx = all_genes.index(gene)
        for mode_str in ("constant", "activity", "memory", "both"):
            mode = RateInputMode(mode_str)
            cfg = MemGRNConfig(
                n_sources=1,
                n_targets=1,
                n_channels=3,
                hidden_width=4,
                rate_mode=mode,
                use_activity_switch=False,
            )
            for seed in (416, 819):
                joint = _unpack_joint(base_weights[seed])
                fit_dir = stage3 / "msn2_six_promoter_rates" / "fits"
                meta = json.loads(
                    (fit_dir / f"{gene}_{mode_str}_{seed}.json").read_text()
                )
                chosen = int(meta["chosen"])
                with np.load(fit_dir / f"{gene}_{mode_str}_{seed}.npz") as rf:
                    rate_vec = rf["rates"][chosen]
                r_dict = _unpack_rate(rate_vec, mode)

                params = MemGRNParams(
                    direct_weights=np.array([[float(joint["direct"][0, g_idx])]]),
                    delayed_weights=joint["mix"][:, g_idx].reshape(1, 3, 1),
                    relaxation=np.array(
                        [0.1 + float(np.logaddexp(0.0, joint["decay"][g_idx]))]
                    ),
                    rate_logits=r_dict["rate"],
                    input_proj=r_dict.get("input"),
                    hidden_bias=r_dict.get("bias"),
                    output_proj=r_dict.get("output"),
                )
                _, _, y_sim = rollout_memgrn_rk4(stage_grid, dt, params, cfg)
                # 5-minute observations correspond to every 16 RK4 steps starting at step 16
                pred_obs = y_sim[16::16, :, 0].T
                archived = archived_preds[f"{gene}_{mode_str}_{seed}_8"][chosen]
                max_diff = max(max_diff, float(np.max(np.abs(pred_obs - archived))))

    if max_diff > 5e-6:
        raise AssertionError(f"Double-precision RK4 replay drifted by {max_diff}")
    return max_diff


def verify_single_pulse_midpoint_replay(stage3: Path) -> float:
    """Independently replay single-pulse 30-min Msn2 predictions from frozen weights."""
    with np.load(stage3 / "neural_rate_msn2_transfer" / "training.npz") as tf_f:
        held_driver = tf_f["tf"][-1]
    with np.load(
        stage3 / "neural_rate_msn2_transfer" / "final_neural_rate_gw.npz"
    ) as wf:
        w416, w819 = wf["weights_416"], wf["weights_819"]
    with np.load(stage3 / "neural_rate_msn2_transfer" / "predictions.npz") as pf:
        sealed_ens = pf["neural_rate_gw"]

    dt = 2.5 / 60.0
    grid_h = np.arange(0.0, 120.01, 2.5) / 60.0
    starts = np.arange(round(2.0 / dt)) * dt
    mid_times = np.stack([starts, starts + 0.5 * dt], axis=-1)
    mid_u = np.interp(mid_times, grid_h, held_driver)[:, None, :]

    cfg = MemGRNConfig(
        n_sources=1,
        n_targets=7,
        n_channels=3,
        hidden_width=4,
        rate_mode=RateInputMode.BOTH,
        use_activity_switch=True,
        switch_threshold=0.1,
    )

    preds = []
    for vec in (w416, w819):
        shapes = [
            ("bias", (4,)),
            ("decay", (7,)),
            ("direct", (1, 7)),
            ("input", (3, 4)),
            ("mix", (1, 3, 7)),
            ("on", (1, 3)),
            ("output", (1, 3, 4)),
            ("rate", (1, 3)),
        ]
        d: dict[str, np.ndarray] = {}
        idx = 0
        for k, sh in shapes:
            sz = int(np.prod(sh))
            d[k] = vec[idx : idx + sz].reshape(sh)
            idx += sz
        params = MemGRNParams(
            direct_weights=d["direct"],
            delayed_weights=d["mix"],
            relaxation=0.1 + np.logaddexp(0.0, d["decay"]),
            rate_logits=d["rate"],
            switch_offsets=d["on"],
            input_proj=d["input"],
            hidden_bias=d["bias"],
            output_proj=d["output"],
        )
        _, _, y_traj = rollout_memgrn_midpoint(mid_u, dt, params, cfg)
        preds.append(y_traj[2::2])

    ens = np.mean(preds, axis=0)
    err = float(np.max(np.abs(ens - sealed_ens)))
    if err > 5e-6:
        raise AssertionError(f"Single-pulse midpoint replay drifted by {err}")
    return err


def main() -> None:
    stage3 = ROOT / "results" / "stage3"
    rk4_diff = verify_msn2_frozen_readout_replay(stage3)
    mid_diff = verify_single_pulse_midpoint_replay(stage3)

    six = load_archived_json(stage3, "msn2_six_promoter_rates/scores.json")
    buf = load_archived_json(stage3, "neural_rate_buffer/test_scores.json")
    occ = load_archived_json(stage3, "neural_rate_pulse/test_scores.json")
    swap = load_archived_json(stage3, "strength_swap_replication/scores.json")
    mtf = json.loads((ROOT / "results" / "multitf_factorial" / "scores.json").read_text())
    msn2_joint = json.loads(
        (ROOT / "results" / "msn2_joint_and_cross_readout" / "scores.json").read_text()
    )
    mac = json.loads((ROOT / "results" / "macrophage_multitf" / "scores.json").read_text())
    hipsc = json.loads((ROOT / "results" / "hipsc_closed_loop" / "scores.json").read_text())
    emt = json.loads((ROOT / "results" / "emt_ligand_shift" / "scores.json").read_text())
    po = json.loads((ROOT / "results" / "partial_obs_and_ltc_sep" / "scores.json").read_text())
    interp = json.loads(
        (ROOT / "results" / "interpretability_and_scaling" / "scores.json").read_text()
    )
    nfkb_ortho = json.loads((ROOT / "results" / "nfkb_ortho" / "scores.json").read_text())

    six_means = {
        m: float(
            np.mean([six["results"][g][m]["mse"] for g in six["genes"]])
        )
        for m in six["models"]
    }
    report = {
        "msn2_frozen_readout_rk4_max_replay_diff": rk4_diff,
        "msn2_single_pulse_midpoint_max_replay_diff": mid_diff,
        "six_promoter_pooled_mse": six_means,
        "memory_vs_constant_mse_reduction_pct": 100.0
        * (1.0 - six_means["memory"] / six_means["constant"]),
        "combined_vs_activity_mse_reduction_pct": 100.0
        * (1.0 - six_means["both"] / six_means["activity"]),
        "occupancy_tail_mse_reduction_pct": 100.0
        * (1.0 - occ["comparisons"][1][1]["effect"]),
        "buffer_tail_mse_reduction_pct": 100.0
        * (1.0 - buf["comparisons"][1][1]["effect"]),
        "swap_vs_repeat_gain_pp": 100.0 * swap["primary"]["swap_vs_repeat"]["mean"],
        "multitf_additive_memgrn_vs_coupled_mse_reduction_pct": mtf["summary"]["additive"][
            "comparisons"
        ]["memgrn_vs_coupled_adaptive_mse"]["pct_reduction_vs_b"],
        "multitf_cooperative_memgrn_vs_ltc_mse_reduction_pct": mtf["summary"][
            "cooperative"
        ]["comparisons"]["memgrn_vs_ltc_coupled_mse"]["pct_reduction_vs_b"],
        "msn2_joint_memgrn_10train_mse": msn2_joint["joint_training"]["memgrn"][
            "ten_trains_post_mse"
        ],
        "msn2_const_readout_memory_reduction_pct": 100.0
        * (
            1.0
            - msn2_joint["cross_readout_2x2"]["local_const"]["memory"]
            / msn2_joint["cross_readout_2x2"]["local_const"]["constant"]
        ),
        "macrophage_memgrn_auroc": mac["summary"]["memgrn"][
            "regulator_attribution_auroc"
        ],
        "hipsc_closed_loop_memgrn_unseen_ko_mse": hipsc["summary"]["cl_memgrn"][
            "unseen_ko_mse"
        ],
        "hipsc_closed_loop_memgrn_chip_psc_auroc": hipsc["summary"]["cl_memgrn"][
            "chip_psc_pooled_auroc"
        ],
        "emt_cross_ligand_memgrn_top30_withdrawal_mse": emt["summary"]["memgrn"][
            "top30_withdrawal_mse"
        ],
        "partial_obs_6of8_memgrn_ortho_ood_mse": po["part_a_partial_obs_6of8_summary"][
            "memgrn_ortho"
        ]["test_mse"],
        "vs_local_ltc_peak_mse_ratio": (
            po["part_b_ltc_separation"]["ltc_local_matched"]["delayed_peak_post_mse"]
            / po["part_b_ltc_separation"]["memgrn_matched"]["delayed_peak_post_mse"]
        ),
        "yeast_slow_over_fast_chromatin_tau_ratio": interp["chromatin_summary"][
            "slow_over_fast_tau_ratio"
        ],
        "nfkb_memgrn_ortho_mean_mse": nfkb_ortho["summary"]["memgrn_ortho"][
            "mean_post_mse"
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
