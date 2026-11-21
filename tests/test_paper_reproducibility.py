"""End-to-end reproducibility tests verifying archived SHA-256 receipts and live replays."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_all import (  # noqa: E402
    sha256_variants,
    verify_msn2_frozen_readout_replay,
    verify_single_pulse_midpoint_replay,
)


def test_archived_msn2_and_synthetic_numbers() -> None:
    stage3 = ROOT / "results" / "stage3"
    six = json.loads((stage3 / "msn2_six_promoter_rates" / "scores.json").read_text())
    audit = json.loads((stage3 / "msn2_six_promoter_rates" / "audit.json").read_text())
    assert audit["scores"] in sha256_variants(
        stage3 / "msn2_six_promoter_rates" / "scores.json"
    )

    means = [
        np.mean([six["results"][g][m]["mse"] for g in six["genes"]])
        for m in ("constant", "activity", "memory", "both")
    ]
    np.testing.assert_allclose(
        means, [0.264209, 0.072538, 0.022801, 0.035050], rtol=5e-5
    )

    assert verify_msn2_frozen_readout_replay(stage3) < 5e-6
    assert verify_single_pulse_midpoint_replay(stage3) < 5e-6

    # Verify pre-locked multi-TF factorial, joint Msn2, and macrophage results
    mtf_dir = ROOT / "results" / "multitf_factorial"
    mtf = json.loads((mtf_dir / "scores.json").read_text())
    assert mtf["design_sha256"] in sha256_variants(mtf_dir / "design_lock.json")
    assert (
        mtf["summary"]["additive"]["means"]["memgrn"]["test_mse"]
        < mtf["summary"]["additive"]["means"]["coupled_adaptive"]["test_mse"]
    )
    assert (
        mtf["summary"]["cooperative"]["means"]["memgrn"]["attribution_rmse"]
        < mtf["summary"]["cooperative"]["means"]["ltc_coupled"]["attribution_rmse"]
    )

    j_dir = ROOT / "results" / "msn2_joint_and_cross_readout"
    j_res = json.loads((j_dir / "scores.json").read_text())
    assert j_res["design_sha256"] in sha256_variants(j_dir / "design_lock.json")
    assert (
        j_res["cross_readout_2x2"]["local_const"]["memory"]
        < 0.25 * j_res["cross_readout_2x2"]["local_const"]["constant"]
    )

    mac_dir = ROOT / "results" / "macrophage_multitf"
    mac = json.loads((mac_dir / "scores.json").read_text())
    assert mac["design_sha256"] in sha256_variants(mac_dir / "design_lock.json")
    assert (
        mac["summary"]["memgrn"]["regulator_attribution_auroc"]
        > mac["summary"]["ltc_coupled"]["regulator_attribution_auroc"]
    )

    hipsc_dir = ROOT / "results" / "hipsc_closed_loop"
    hipsc = json.loads((hipsc_dir / "scores.json").read_text())
    assert hipsc["design_sha256"] in sha256_variants(hipsc_dir / "design_lock.json")
    assert (
        hipsc["summary"]["cl_memgrn"]["unseen_ko_mse"]
        < hipsc["summary"]["renge"]["unseen_ko_mse"]
    )

    ortho_dir = ROOT / "results" / "nfkb_ortho"
    ortho = json.loads((ortho_dir / "scores.json").read_text())
    assert ortho["design_sha256"] in sha256_variants(ortho_dir / "design_lock.json")
    assert (
        ortho["summary"]["memgrn_ortho"]["mean_post_mse"]
        < ortho["summary"]["memgrn"]["mean_post_mse"]
    )

    emt_dir = ROOT / "results" / "emt_ligand_shift"
    emt = json.loads((emt_dir / "scores.json").read_text())
    assert emt["design_sha256"] in sha256_variants(emt_dir / "design_lock.json")
    assert (
        emt["summary"]["memgrn"]["top30_withdrawal_mse"]
        < emt["summary"]["ltc_coupled"]["top30_withdrawal_mse"]
    )
    assert (
        emt["summary"]["memgrn"]["top30_withdrawal_mse"]
        < emt["summary"]["local_const"]["top30_withdrawal_mse"]
    )

    po_dir = ROOT / "results" / "partial_obs_and_ltc_sep"
    po = json.loads((po_dir / "scores.json").read_text())
    assert po["design_sha256"] in sha256_variants(po_dir / "design_lock.json")
    assert (
        po["part_a_partial_obs_6of8_summary"]["memgrn_ortho"]["test_mse"]
        < po["part_a_partial_obs_6of8_summary"]["memgrn"]["test_mse"]
    )
    assert (
        po["part_b_ltc_separation"]["memgrn_matched"]["delayed_peak_post_mse"]
        < 0.5 * po["part_b_ltc_separation"]["ltc_local_matched"]["delayed_peak_post_mse"]
    )

    interp_dir = ROOT / "results" / "interpretability_and_scaling"
    interp = json.loads((interp_dir / "scores.json").read_text())
    assert interp["design_sha256"] in sha256_variants(interp_dir / "design_lock.json")
    assert interp["chromatin_summary"]["slow_over_fast_tau_ratio"] > 1.4
