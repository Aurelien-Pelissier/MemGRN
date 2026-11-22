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

    emt_dir = ROOT / "results" / "emt_withdrawal_forecast"
    emt = json.loads((emt_dir / "scores.json").read_text())
    assert emt["design_sha256"] in sha256_variants(emt_dir / "design_lock.json")
    assert (
        emt["summary"]["memgrn"]["top30_withdrawal_mse"]
        < emt["summary"]["local_const"]["top30_withdrawal_mse"]
    )
    assert (
        emt["summary"]["memgrn"]["top30_withdrawal_mse"]
        < emt["summary"]["coupled_adaptive"]["top30_withdrawal_mse"]
    )
    assert (
        emt["summary"]["memgrn"]["top30_withdrawal_mse"]
        < emt["summary"]["latent_ode"]["top30_withdrawal_mse"]
    )
    assert emt["summary"]["memgrn_ortho"]["obs_tf_attr_drift_vs_memgrn"] == 0.0
    assert emt["summary"]["memgrn"]["collectri_ap_over_random"] > 1.2

    cyc_dir = ROOT / "results" / "cyclic_double_ko"
    cyc = json.loads((cyc_dir / "scores.json").read_text())
    assert cyc["design_sha256"] in sha256_variants(cyc_dir / "design_lock.json")
    assert (
        cyc["summary"]["cl_memgrn"]["double_ko_mse"]
        < 0.35 * cyc["summary"]["cl_selective_ssm"]["double_ko_mse"]
    )
    assert cyc["summary"]["cl_memgrn"]["graph_ap"] > 0.88

    sens_dir = ROOT / "results" / "sensitivity_ablations"
    sens = json.loads((sens_dir / "scores.json").read_text())
    assert sens["design_sha256"] in sha256_variants(sens_dir / "design_lock.json")
    assert (
        sens["summary"]["channels"]["R=5"]["memgrn"]["test_mse"]
        < sens["summary"]["channels"]["R=1"]["memgrn"]["test_mse"]
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

    # Verify primary human CD4+ T-cell Perturb-seq (GSE314342, S=128 TFs)
    cd4t_dir = ROOT / "results" / "cd4t_closed_loop"
    cd4t = json.loads((cd4t_dir / "scores.json").read_text())
    assert cd4t["design_sha256"] in sha256_variants(cd4t_dir / "design_lock.json")
    assert cd4t["summary"]["cl_memgrn"]["unseen_tf_48h_mse"] < cd4t["summary"]["renge"]["unseen_tf_48h_mse"]
    assert cd4t["summary"]["cl_memgrn"]["mem_params"] < 0.1 * cd4t["summary"]["cl_coupled_adaptive"]["mem_params"]
    assert cd4t["summary"]["cl_memgrn"]["collectri_ap_over_random"] > 1.2

    # Verify cross-domain non-biology benchmarks (Beijing AQ, METR-LA, IEEE 39-Bus, Binance, Reddit)
    bq_dir = ROOT / "results" / "beijing_air_quality"
    bq = json.loads((bq_dir / "scores.json").read_text())
    assert bq["design_sha256"] in sha256_variants(bq_dir / "design_lock.json")
    for rival in ("local_const", "ltc_local", "coupled_adaptive", "ltc_coupled", "selective_ssm", "cfc", "neural_cde", "perturb_ode", "latent_ode"):
        assert bq["full_8driver_loso"]["memgrn"]["loso_mse"] < bq["full_8driver_loso"][rival]["loso_mse"]
        assert bq["full_wilcoxon_vs_memgrn"][rival]["wilcoxon_p"] < 0.05

    metr_dir = ROOT / "results" / "metr_la_traffic"
    metr = json.loads((metr_dir / "scores.json").read_text())
    assert metr["design_sha256"] in sha256_variants(metr_dir / "design_lock.json")
    for rival in ("cl_local_const", "cl_ltc_local", "cl_selective_ssm", "cl_cfc", "cl_coupled_adaptive", "cl_ltc_coupled", "cl_perturb_ode", "cl_latent_ode", "persistence"):
        assert metr["summary"]["cl_memgrn"]["test_mse"] < metr["summary"][rival]["test_mse"]
        assert metr["summary"]["cl_memgrn"]["road_graph_colnorm_ap_over_random"] > metr["summary"][rival]["road_graph_colnorm_ap_over_random"]

    pg_dir = ROOT / "results" / "powergrid_n2"
    pg = json.loads((pg_dir / "scores.json").read_text())
    assert pg["design_sha256"] in sha256_variants(pg_dir / "design_lock.json")
    assert pg["summary"]["cl_memgrn"]["double_n2_mse"] < pg["summary"]["cl_selective_ssm"]["double_n2_mse"]
    assert pg["summary"]["cl_memgrn"]["double_n2_mse"] < pg["summary"]["cl_ltc_local"]["double_n2_mse"]
    assert pg["summary"]["cl_memgrn"]["graph_auroc"] > 0.84

    fin_dir = ROOT / "results" / "financial_contagion"
    fin = json.loads((fin_dir / "scores.json").read_text())
    assert fin["design_sha256"] in sha256_variants(fin_dir / "design_lock.json")
    assert fin["partial_14alt_hidden_btc_eth"]["memgrn_ortho"]["obs_asset_attr_drift_vs_memgrn"] == 0.0
    assert fin["partial_14alt_hidden_btc_eth"]["memgrn_ortho"]["aug_test_mse"] < fin["partial_14alt_hidden_btc_eth"]["latent_ode"]["aug_test_mse"]
    assert fin["partial_14alt_hidden_btc_eth"]["memgrn_ortho"]["idio_shock_mse"] < fin["partial_14alt_hidden_btc_eth"]["latent_ode"]["idio_shock_mse"]

    red_dir = ROOT / "results" / "reddit_temporal"
    red = json.loads((red_dir / "scores.json").read_text())
    assert red["design_sha256"] in sha256_variants(red_dir / "design_lock.json")
    assert red["summary"]["cl_memgrn"]["test_mse"] < red["summary"]["cl_local_const"]["test_mse"]
    assert red["summary"]["cl_memgrn"]["test_mse"] < red["summary"]["cl_cfc"]["test_mse"]
