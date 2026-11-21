"""Build all publication-ready vector figures and LaTeX tables from verified artifacts.

Typography and layout rules enforced across all assets:
- Figure width = 5.50 inches (1:1 with ICLR \\textwidth -> zero LaTeX rescaling).
- Figure text sizes = 7.8--8.5 pt (just slightly below the 9.0 pt figure captions).
- Editable TrueType vector fonts (pdf.fonttype = 42, svg.fonttype = 'none').
- Every LaTeX table has <= 6 columns and uses \\memgrntable for uniform 8.43 pt font size.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "results" / "stage3"
OUT_FIG_REPO = ROOT / "figures"

SKILL_SCRIPTS = Path(
    "/usr/local/google/home/apelissier/.gemini/config/skills/nature-figure/scripts"
)
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    require_matplotlib_panel_alignment = None

TEXTWIDTH_IN = 5.50

PALETTE = {
    "navy": "#1D4E89",
    "teal": "#008B72",
    "amber": "#D97706",
    "crimson": "#B8334A",
    "violet": "#6D4C94",
    "slate": "#5A6577",
    "steel": "#2F6EA6",
    "light_blue": "#EBF2FA",
    "light_teal": "#E6F5F1",
    "light_amber": "#FEF3E2",
    "light_violet": "#F2EEF8",
    "light_gray": "#F4F6F8",
    "border": "#334155",
    "ink": "#1E293B",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.2,
        "axes.titlesize": 8.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.9,
        "ytick.labelsize": 7.9,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

SOURCE_HASHES: dict[str, str] = {}


def load_stage3(rel_dir: str, fname: str = "scores.json") -> dict:
    p = STAGE3 / rel_dir / fname
    raw = p.read_bytes()
    SOURCE_HASHES[f"stage3/{rel_dir}/{fname}"] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def load_new_result(suite: str, fname: str = "scores.json") -> dict:
    p = ROOT / "results" / suite / fname
    raw = p.read_bytes()
    SOURCE_HASHES[f"{suite}/{fname}"] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def add_panel_tag(
    ax: plt.Axes, tag: str, x_pt: float = -30.0, y_pt: float = 6.0
) -> None:
    ax.annotate(
        tag,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(x_pt, y_pt),
        textcoords="offset points",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="bottom",
    )


def export_figure(
    fig: plt.Figure,
    stem: str,
    out_latex_fig: Path,
    audit_alignment: bool = True,
    exclude_axes: list[plt.Axes] | None = None,
    exemptions: list[dict] | None = None,
) -> None:
    fig.canvas.draw()
    if audit_alignment and require_matplotlib_panel_alignment is not None:
        qa_dir = ROOT / "output" / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(qa_dir / f"{stem}.alignment.json"),
            overlay_svg=str(qa_dir / f"{stem}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            exclude_axes=exclude_axes or [],
            exemptions=exemptions or [],
            strict=True,
        )

    for folder in (OUT_FIG_REPO, out_latex_fig):
        fig.savefig(folder / f"{stem}.pdf")
        fig.savefig(folder / f"{stem}.svg")
        fig.savefig(folder / f"{stem}.png", dpi=300)
    plt.close(fig)


def write_tex_table(
    out_tab_dir: Path, fname: str, col_spec: str, header: list[str], rows: list[list[str]]
) -> None:
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        if row and row[0] == "\\midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (out_tab_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fig1_architecture(out_latex_fig: Path) -> None:
    """Figure 1: Closed- and open-loop MemGRN architecture + theoretical pillars."""
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.58))
    ax = fig.add_axes([0.012, 0.02, 0.976, 0.96])
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 5.24)
    ax.axis("off")

    def draw_box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        fc: str,
        ec: str = PALETTE["border"],
        lw: float = 0.9,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.10",
                fc=fc,
                ec=ec,
                lw=lw,
            )
        )
        ax.text(
            x + 0.5 * w,
            y + 0.5 * h,
            text,
            ha="center",
            va="center",
            fontsize=7.9,
            color=PALETTE["ink"],
            linespacing=1.18,
        )

    def draw_arrow(
        p0: tuple[float, float],
        p1: tuple[float, float],
        color: str = PALETTE["ink"],
        lw: float = 1.0,
        ls: str = "-",
    ) -> None:
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops={
                "arrowstyle": "-|>,head_length=0.34,head_width=0.19",
                "lw": lw,
                "ls": ls,
                "color": color,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    ax.text(
        0.05,
        5.00,
        "a   Closed- and open-loop source-isolated memory with enhancer readout",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )

    draw_box(
        0.04,
        2.74,
        1.84,
        1.02,
        "Source drive\naᵢ = tanh(gᵢ)",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
        lw=1.0,
    )
    draw_box(
        2.08,
        2.74,
        1.90,
        1.02,
        "Loading zᵢᵣ\nżᵢᵣ = λ↑ᵢᵣ(aᵢ−zᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        4.18,
        2.74,
        1.90,
        1.02,
        "Clearance hᵢᵣ\nḣᵢᵣ = λ↓ᵢᵣ(zᵢᵣ−hᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        6.28,
        2.74,
        2.66,
        1.02,
        "Enhancer readout\nDᵢⱼaᵢ + Bᵢᵣⱼhᵢᵣ + Γh̄ᵢh̄ₖ",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        9.14,
        2.74,
        1.82,
        1.02,
        "Gene state\nẏⱼ = pⱼ − δⱼyⱼ",
        PALETTE["light_gray"],
        ec=PALETTE["border"],
    )

    draw_arrow((1.92, 3.25), (2.04, 3.25))
    draw_arrow((4.02, 3.25), (4.14, 3.25))
    draw_arrow((6.12, 3.25), (6.24, 3.25))
    draw_arrow((8.98, 3.25), (9.10, 3.25))

    ax.plot(
        [10.05, 10.05, 0.96],
        [3.82, 4.30, 4.30],
        color=PALETTE["navy"],
        lw=0.95,
    )
    draw_arrow((0.96, 4.30), (0.96, 3.82), color=PALETTE["navy"], lw=0.95)
    ax.text(
        5.50,
        4.40,
        "Closed-loop feedback  gᵢ = (1−koᵢ)γᵢyᵢ − koᵢγᵢyᵢ,₀   +   Direct path  Dᵢⱼaᵢ",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color=PALETTE["navy"],
    )

    draw_box(
        1.14,
        1.56,
        5.80,
        0.74,
        "Isolated rates  (λ↑ᵢᵣ, λ↓ᵢᵣ) : ∂(zᵢᵣ, hᵢᵣ)/∂gₖ ≡ 0  (k ≠ i)",
        PALETTE["light_amber"],
        ec=PALETTE["amber"],
    )
    draw_arrow((3.03, 2.36), (3.03, 2.68), color=PALETTE["amber"])
    draw_arrow((5.13, 2.36), (5.13, 2.68), color=PALETTE["amber"])
    ax.plot(
        [0.96, 0.96, 1.09],
        [2.68, 1.93, 1.93],
        color=PALETTE["amber"],
        lw=0.85,
        ls="--",
    )

    draw_box(
        7.20,
        1.56,
        3.76,
        0.74,
        "Ortho cofactor  Ψ⊥ ⊥ span(A, H)",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
        lw=0.85,
    )
    draw_arrow((8.35, 2.36), (8.35, 2.68), color=PALETTE["violet"], lw=0.85, ls="--")

    ax.text(
        0.05,
        1.18,
        "b   Three structural guarantees across closed- and open-loop benchmarks",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )
    draw_box(
        0.04,
        0.08,
        3.44,
        0.84,
        "1. S-TF & closed-loop ID\nZero cross-TF leakage",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        3.78,
        0.08,
        3.44,
        0.84,
        "2. Unit DC gain vs. LTC\n2nd-order delay peaks",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        7.52,
        0.08,
        3.44,
        0.84,
        "3. Iso-expr. & Ortho\nO(SR) genome scaling",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
    )

    export_figure(fig, "fig1_memgrn_architecture", out_latex_fig, audit_alignment=False)


def build_fig2_human_longitudinal(out_latex_fig: Path) -> None:
    """Figure 2: Human iPSC closed-loop Perturb-seq (GSE213069) & EMT withdrawal (GSE147405)."""
    hipsc = load_new_result("hipsc_closed_loop")["summary"]
    emt = load_new_result("emt_ligand_shift")["summary"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.22))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.105,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.42,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    keys_h = [
        "renge",
        "cl_ltc_coupled",
        "cl_perturb_ode",
        "cl_coupled_adaptive",
        "cl_ltc_local",
        "cl_local_const",
        "cl_memgrn",
    ]
    lbls_h = ["RENGE", "LTC-Cpl", "PertODE", "Cpl-Adapt", "LTC-Loc", "Loc-Const", "MemGRN"]
    cols_h = [
        PALETTE["slate"],
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["teal"],
        "#7B8A9E",
        PALETTE["navy"],
    ]
    x_h = np.arange(len(keys_h))

    # Panel a: hiPSC unseen CRISPR KO MSE
    ko_vals = [hipsc[k]["unseen_ko_mse"] for k in keys_h]
    ax0.bar(x_h, ko_vals, width=0.64, color=cols_h, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(x_h)
    ax0.set_xticklabels(lbls_h, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, len(keys_h) - 0.4)
    ax0.set_ylim(0.0, float(max(ko_vals) * 1.15))
    ax0.set_ylabel("Unseen-KO MSE")
    ax0.set_title("hiPSC CRISPR KOs")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: hiPSC Day-5 horizon extrapolation MSE
    d5_vals = [hipsc[k]["day5_extrapolation_mse"] for k in keys_h]
    ax1.bar(x_h, d5_vals, width=0.64, color=cols_h, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax1.set_xticks(x_h)
    ax1.set_xticklabels(lbls_h, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, len(keys_h) - 0.4)
    ax1.set_ylim(0.0, float(max(d5_vals) * 1.15))
    ax1.set_ylabel("Day-5 holdout MSE")
    ax1.set_title("hiPSC Day-5")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Human EMT cross-ligand top-30 withdrawal excess MSE (%) over MemGRN (GSE147405)
    keys_e = [
        "ltc_coupled",
        "coupled_adaptive",
        "local_const",
        "latent_ode",
        "perturb_ode",
        "ltc_local",
        "memgrn",
    ]
    lbls_e = ["LTC-Cpl", "Cpl-Adapt", "Loc-Const", "Lat-ODE", "PertODE", "LTC-Loc", "MemGRN"]
    cols_e = [
        PALETTE["crimson"],
        PALETTE["amber"],
        PALETTE["slate"],
        "#8C6D9B",
        PALETTE["violet"],
        PALETTE["teal"],
        PALETTE["navy"],
    ]
    ref_e = emt["memgrn"]["top30_withdrawal_mse"]
    exc_e = [100.0 * (emt[k]["top30_withdrawal_mse"] / ref_e - 1.0) for k in keys_e]
    x_e = np.arange(len(keys_e))
    ax2.bar(x_e, exc_e, width=0.64, color=cols_e, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax2.set_xticks(x_e)
    ax2.set_xticklabels(lbls_e, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.6, len(keys_e) - 0.4)
    ax2.set_ylim(0.0, float(max(exc_e) * 1.22))
    ax2.set_ylabel("Excess wdr. MSE (%)")
    ax2.set_title("Human EMT wdr.")
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(fig, "fig2_human_longitudinal", out_latex_fig)


def build_fig3_multitf_ltc_ortho(out_latex_fig: Path) -> None:
    """Figure 3: Multi-TF cooperative OOD, Source-Local LTC separation, & 6-of-8 MemGRN-Ortho."""
    mtf = load_new_result("multitf_factorial")["summary"]
    po = load_new_result("partial_obs_and_ltc_sep")
    coop_m = mtf["cooperative"]["means"]
    part_b = po["part_b_ltc_separation"]
    part_a = po["part_a_partial_obs_6of8_summary"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.24))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.12, 0.92, 1.06],
        left=0.105,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.44,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: 24-world cooperative OOD MSE
    k_c = [
        "ltc_coupled",
        "perturb_ode",
        "local_const",
        "coupled_adaptive",
        "ltc_local",
        "memgrn_additive",
        "memgrn",
    ]
    l_c = ["LTC-Cpl", "PertODE", "Loc-Const", "Cpl-Adapt", "LTC-Loc", "Mem-Add", "MemGRN"]
    c_c = [
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["steel"],
        PALETTE["navy"],
    ]
    xc = np.arange(len(k_c))
    vc = [coop_m[k]["test_mse"] for k in k_c]
    ax0.bar(xc, vc, width=0.64, color=c_c, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(xc)
    ax0.set_xticklabels(l_c, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, len(k_c) - 0.4)
    ax0.set_ylim(0.0, float(max(vc) * 1.16))
    ax0.set_ylabel("OOD test MSE")
    ax0.set_title("Cooperative (8-TF)")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Parameter-matched MemGRN (252) vs Source-Local LTC (288) on Theorem 1(iii)
    xb = np.arange(2)
    bw = 0.34
    ltc_vals = [
        1e5 * part_b["ltc_local_matched"]["delayed_peak_post_mse"],
        100.0 * part_b["ltc_local_matched"]["steady_state_amp_invariance_error"],
    ]
    mem_vals = [
        1e5 * part_b["memgrn_matched"]["delayed_peak_post_mse"],
        100.0 * part_b["memgrn_matched"]["steady_state_amp_invariance_error"],
    ]
    ax1.bar(
        xb - 0.5 * bw,
        ltc_vals,
        width=bw * 0.92,
        color=PALETTE["teal"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="LTC-Loc",
    )
    ax1.bar(
        xb + 0.5 * bw,
        mem_vals,
        width=bw * 0.92,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="MemGRN",
    )
    ax1.set_xticks(xb)
    ax1.set_xticklabels(["Peak MSE\n(×10⁻⁵)", "Amp. shift\ndist. (%)"], rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, 1.6)
    ax1.set_ylim(0.0, float(max(*ltc_vals, *mem_vals) * 1.92))
    ax1.set_ylabel("Error / distortion")
    ax1.set_title("vs. Local LTC")
    ax1.legend(loc="upper left", ncol=1, handlelength=1.0, labelspacing=0.14)
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    # Panel c: Ground-truth 6-of-8 partially observed multi-TF OOD MSE
    k_p = ["perturb_ode", "latent_ode", "local_const", "memgrn_unproj", "memgrn", "memgrn_ortho"]
    l_p = ["PertODE", "Lat-ODE", "Loc-Const", "Unproj", "MemGRN", "Mem-Ortho"]
    c_p = [
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["slate"],
        PALETTE["crimson"],
        PALETTE["steel"],
        PALETTE["navy"],
    ]
    xp = np.arange(len(k_p))
    vp = [part_a[k]["test_mse"] for k in k_p]
    ax2.bar(xp, vp, width=0.64, color=c_p, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(l_p, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.6, len(k_p) - 0.4)
    ax2.set_ylim(0.0, float(max(vp) * 1.16))
    ax2.set_ylabel("6-of-8 OOD MSE")
    ax2.set_title("Partial obs. (GT)")
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(
        fig,
        "fig3_multitf_ltc_ortho",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios match 7-bar, 2-group, and 6-bar panel densities",
            }
        ],
    )


def build_fig4_yeast_chromatin_portraits(out_latex_fig: Path) -> None:
    """Figure 4: Live yeast Msn2 joint fits, cross-readout swap, & chromatin memory portraits."""
    joint = load_new_result("msn2_joint_and_cross_readout")
    interp = load_new_result("interpretability_and_scaling")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.24))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.04, 1.00, 1.12],
        left=0.100,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Joint source-CV fits
    j_keys = ["local_const", "perturb_ode", "latent_ode", "ltc_local", "memgrn_sym", "memgrn"]
    j_lbls = ["Const gW", "PertODE", "Lat-ODE", "LTC-Loc", "Mem-Sym", "MemGRN"]
    j_cols = [
        PALETTE["slate"],
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["steel"],
        PALETTE["navy"],
    ]
    j_vals = [joint["joint_training"][k]["ten_trains_post_mse"] for k in j_keys]
    jx = np.arange(len(j_keys))
    ax0.bar(jx, j_vals, width=0.62, color=j_cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(jx)
    ax0.set_xticklabels(j_lbls, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, len(j_keys) - 0.4)
    ax0.set_ylim(0.0, float(max(j_vals) * 1.15))
    ax0.set_ylabel("10-train MSE")
    ax0.set_title("Joint source-CV")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Symmetric 2x2 cross-readout swap
    r_modes = ["constant", "activity", "memory", "both"]
    r_lbls = ["Const", "Activity", "Memory", "Act+Mem"]
    rx = np.arange(4)
    bw = 0.34
    const_w_vals = [joint["cross_readout_2x2"]["local_const"][m] for m in r_modes]
    mem_w_vals = [joint["cross_readout_2x2"]["memgrn"][m] for m in r_modes]
    ax1.bar(
        rx - 0.5 * bw,
        const_w_vals,
        width=bw * 0.92,
        color=PALETTE["amber"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="Const-gW W",
    )
    ax1.bar(
        rx + 0.5 * bw,
        mem_w_vals,
        width=bw * 0.92,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="MemGRN W",
    )
    ax1.set_xticks(rx)
    ax1.set_xticklabels(r_lbls, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.65, 3.65)
    ax1.set_ylim(0.0, float(max(*const_w_vals, *mem_w_vals) * 1.32))
    ax1.set_ylabel("10-train MSE")
    ax1.set_title("Cross-readout")
    ax1.legend(loc="upper right", ncol=1, handlelength=1.1, labelspacing=0.16)
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    # Panel c: Learned chromatin-memory washout timescale tau_off (min) ordered by chromatin class
    portraits = interp["yeast_portraits"]
    p_order = ["HXK1", "DCS2", "DDR2", "RTN2", "ALD3", "TKL2", "SIP18"]
    taus = [portraits[g]["effective_washout_tau_min"] for g in p_order]
    p_cols = [
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["navy"],
        PALETTE["amber"],
        PALETTE["navy"],
        PALETTE["navy"],
        PALETTE["navy"],
    ]
    px = np.arange(len(p_order))
    ax2.bar(px, taus, width=0.64, color=p_cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    # Legend proxies
    ax2.bar([-5], [0], color=PALETTE["teal"], edgecolor=PALETTE["ink"], linewidth=0.55, label="Fast (open)")
    ax2.bar([-5], [0], color=PALETTE["navy"], edgecolor=PALETTE["ink"], linewidth=0.55, label="Slow (remod.)")
    ax2.set_xticks(px)
    ax2.set_xticklabels(p_order, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.6, len(p_order) - 0.4)
    ax2.set_ylim(0.0, 82.0)
    ax2.set_yticks([0, 20, 40, 60])
    ax2.set_ylabel("Washout τ (min)")
    ax2.set_title("Learned chromatin")
    ax2.legend(loc="upper left", ncol=1, handlelength=1.0, labelspacing=0.14)
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(
        fig,
        "fig4_yeast_chromatin_portraits",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios accommodate 6-bar, 4-pair, and 7-promoter panels",
            }
        ],
    )


def build_fig5_mammalian_and_scaling(out_latex_fig: Path) -> None:
    """Figure 5: 4-TF Macrophage GRS, Fibroblast MemGRN-Ortho, & Genome-Scale O(SR) Scaling."""
    mac = load_new_result("macrophage_multitf")["summary"]
    ortho = load_new_result("nfkb_ortho")["summary"]
    scaling = load_new_result("interpretability_and_scaling")["scaling_benchmark"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.24))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.00, 1.12],
        left=0.105,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: 4-TF Macrophage GRS attribution AUROC
    m_keys = [
        "perturb_ode",
        "ltc_coupled",
        "ltc_local",
        "local_const",
        "coupled_adaptive",
        "memgrn",
    ]
    m_lbls = ["PertODE", "LTC-Cpl", "LTC-Loc", "Loc-Const", "Cpl-Adapt", "MemGRN"]
    m_cols = [
        PALETTE["violet"],
        PALETTE["crimson"],
        PALETTE["teal"],
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["navy"],
    ]
    mx = np.arange(len(m_keys))
    auroc_vals = np.array([mac[k]["regulator_attribution_auroc"] for k in m_keys])
    ax0.bar(
        mx,
        auroc_vals - 0.5,
        bottom=0.5,
        width=0.62,
        color=m_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
    )
    ax0.set_xticks(mx)
    ax0.set_xticklabels(m_lbls, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, len(m_keys) - 0.4)
    ax0.set_ylim(0.5, float(max(auroc_vals) + 0.05))
    ax0.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax0.set_ylabel("Regulator AUROC")
    ax0.set_title("Macrophage GRS")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: 177-gene fibroblast NF-kB washout 3-ligand MSE
    o_keys = ["local_const", "memgrn", "memgrn_ortho", "perturb_ode", "latent_ode"]
    o_lbls = ["Const gW", "MemGRN", "Mem-Ortho", "PertODE", "Lat-ODE"]
    o_cols = [
        PALETTE["slate"],
        PALETTE["steel"],
        PALETTE["navy"],
        PALETTE["violet"],
        PALETTE["amber"],
    ]
    ox = np.arange(len(o_keys))
    o_vals = [ortho[k]["mean_post_mse"] for k in o_keys]
    ax1.bar(ox, o_vals, width=0.60, color=o_cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax1.set_xticks(ox)
    ax1.set_xticklabels(o_lbls, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, len(o_keys) - 0.4)
    ax1.set_ylim(0.0, float(max(o_vals) * 1.15))
    ax1.set_ylabel("3-ligand MSE")
    ax1.set_title("Fibroblast Ortho")
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    # Panel c: Genome-scale memory parameter scaling S = 8 -> 512 TFs (up to 2,048 genes)
    # Note: plot S=8..256 for matched-state Coupled LTC (M=3S) and S=8..512 for MemGRN & Coupled Adaptive
    s_all = np.array([r["n_tfs"] for r in scaling], dtype=np.float64)
    mem_p = np.array([r["memgrn_mem_params"] for r in scaling], dtype=np.float64)
    cpl_p = np.array([r["coupled_adaptive_mem_params"] for r in scaling], dtype=np.float64)
    s_ltc = s_all[:6]
    ltc_p = np.array([r["coupled_ltc_mem_params"] for r in scaling[:6]], dtype=np.float64)

    ax2.loglog(
        s_ltc,
        ltc_p,
        marker="^",
        ms=3.8,
        lw=1.3,
        color=PALETTE["crimson"],
        label="LTC-Cpl",
    )
    ax2.loglog(
        s_all,
        cpl_p,
        marker="s",
        ms=3.6,
        lw=1.3,
        color=PALETTE["amber"],
        label="Cpl-Ad.",
    )
    ax2.loglog(
        s_all,
        mem_p,
        marker="o",
        ms=4.0,
        lw=1.45,
        color=PALETTE["navy"],
        label="MemGRN",
    )
    ax2.set_xticks([8, 32, 128, 512])
    ax2.set_xticklabels(["8", "32", "128", "512"])
    ax2.minorticks_off()
    ax2.set_xlim(6.5, 680.0)
    ax2.set_ylim(100.0, 1.2e8)
    ax2.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax2.set_yticklabels(["100", "1k", "10k", "100k", "1M", "10M"])
    ax2.set_xlabel("Regulators S (N=4S)")
    ax2.set_ylabel("Memory params")
    ax2.set_title("Genome scaling")
    ax2.legend(loc="upper left", ncol=1, handlelength=1.0, labelspacing=0.14)
    add_panel_tag(ax2, "c", x_pt=-32.0, y_pt=6.0)

    export_figure(
        fig,
        "fig5_mammalian_and_scaling",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios accommodate 6-bar, 5-bar, and log-log scaling panels",
            }
        ],
    )


def build_all_tables(out_tab_dir: Path) -> None:
    """Write all 5 main-text tables and 8 appendix tables."""
    hipsc = load_new_result("hipsc_closed_loop")["summary"]
    emt = load_new_result("emt_ligand_shift")
    mtf = load_new_result("multitf_factorial")["summary"]
    po = load_new_result("partial_obs_and_ltc_sep")
    joint = load_new_result("msn2_joint_and_cross_readout")
    mac = load_new_result("macrophage_multitf")["summary"]
    ortho = load_new_result("nfkb_ortho")["summary"]
    interp = load_new_result("interpretability_and_scaling")
    occ = load_stage3("neural_rate_pulse", "test_scores.json")
    buf = load_stage3("neural_rate_buffer", "test_scores.json")
    six = load_stage3("msn2_six_promoter_rates")
    swap = load_stage3("strength_swap_replication")
    cert = load_stage3("paired_time_certificate", "test_scores.json")
    cert_meta = load_stage3("paired_time_certificate", "certificate.json")

    # =========================================================================
    # Table 1: Closed-Loop Longitudinal Human iPSC CRISPR Perturb-seq (GSE213069)
    # =========================================================================
    t1_order = [
        ("cl_memgrn", "\\textbf{Closed-Loop MemGRN}"),
        ("cl_local_const", "Closed-Loop Local Const."),
        ("cl_ltc_local", "Closed-Loop Local LTC"),
        ("cl_coupled_adaptive", "Closed-Loop Coupled Adapt."),
        ("cl_ltc_coupled", "Closed-Loop Coupled LTC"),
        ("cl_perturb_ode", "PerturbODE"),
        ("cl_latent_ode", "Latent Neural ODE"),
        ("renge", "RENGE"),
    ]
    best_ko = min(hipsc[k]["unseen_ko_mse"] for k, _ in t1_order)
    best_t25 = min(hipsc[k]["unseen_ko_top25_mse"] for k, _ in t1_order)
    best_r = max(hipsc[k]["unseen_ko_pearson_r"] for k, _ in t1_order)
    best_d5 = min(hipsc[k]["day5_extrapolation_mse"] for k, _ in t1_order)
    best_ap = max(hipsc[k]["chip_psc_ap_over_random"] for k, _ in t1_order)
    t1_rows: list[list[str]] = []
    for k, lbl in t1_order:
        d = hipsc[k]
        s_ko = f"{d['unseen_ko_mse']:.4f}"
        s_t25 = f"{d['unseen_ko_top25_mse']:.4f}"
        s_r = f"{d['unseen_ko_pearson_r']:.3f}"
        s_d5 = f"{d['day5_extrapolation_mse']:.4f}"
        s_ap = f"{d['chip_psc_ap_over_random']:.2f}$\\times$"
        if np.isclose(d["unseen_ko_mse"], best_ko):
            s_ko = f"\\textbf{{{s_ko}}}"
        if np.isclose(d["unseen_ko_top25_mse"], best_t25):
            s_t25 = f"\\textbf{{{s_t25}}}"
        if np.isclose(d["unseen_ko_pearson_r"], best_r):
            s_r = f"\\textbf{{{s_r}}}"
        if np.isclose(d["day5_extrapolation_mse"], best_d5):
            s_d5 = f"\\textbf{{{s_d5}}}"
        if np.isclose(d["chip_psc_ap_over_random"], best_ap):
            s_ap = f"\\textbf{{{s_ap}}}"
        t1_rows.append([lbl, s_ko, s_t25, s_r, s_d5, s_ap])
    write_tex_table(
        out_tab_dir,
        "tab1_hipsc_closed_loop.tex",
        "lrrrrr",
        [
            "Closed-loop model (103 genes)",
            r"Unseen KO $\downarrow$",
            r"Top-25 DE $\downarrow$",
            r"KO $r$ $\uparrow$",
            r"Day-5 MSE $\downarrow$",
            r"ChIP AP/rand $\uparrow$",
        ],
        t1_rows,
    )

    # =========================================================================
    # Table 2: Human EMT Cross-Ligand Induction & Withdrawal (GSE147405, 53,290 cells)
    # =========================================================================
    emt_s = emt["summary"]
    emt_l = emt["by_ligand"]
    t2_order = [
        ("memgrn", "\\textbf{MemGRN (ours)}"),
        ("ltc_local", "Source-Local LTC"),
        ("coupled_const", "Coupled Constant (SSM)"),
        ("perturb_ode", "PerturbODE"),
        ("latent_ode", "Latent Neural ODE"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("local_const", "Local Constant ($gW$)"),
        ("ltc_coupled", "Coupled LTC"),
    ]
    b_f = min(emt_s[k]["full_10d_mse"] for k, _ in t2_order)
    b_w = min(emt_s[k]["withdrawal_mse"] for k, _ in t2_order)
    b_tw = min(emt_s[k]["top30_withdrawal_mse"] for k, _ in t2_order)
    b_pk = min(emt_s[k]["post_withdrawal_peak_mse"] for k, _ in t2_order)
    b_tg = min(emt_l[k]["TGFB1"]["top30_withdrawal_mse"] for k, _ in t2_order)
    t2_rows: list[list[str]] = []
    for k, lbl in t2_order:
        d = emt_s[k]
        tg_v = emt_l[k]["TGFB1"]["top30_withdrawal_mse"]
        sf = f"{d['full_10d_mse']:.3f}"
        sw = f"{d['withdrawal_mse']:.3f}"
        stw = f"{d['top30_withdrawal_mse']:.3f}"
        spk = f"{d['post_withdrawal_peak_mse']:.3f}"
        stg = f"{tg_v:.2f}"
        if np.isclose(d["full_10d_mse"], b_f):
            sf = f"\\textbf{{{sf}}}"
        if np.isclose(d["withdrawal_mse"], b_w):
            sw = f"\\textbf{{{sw}}}"
        if np.isclose(d["top30_withdrawal_mse"], b_tw):
            stw = f"\\textbf{{{stw}}}"
        if np.isclose(d["post_withdrawal_peak_mse"], b_pk):
            spk = f"\\textbf{{{spk}}}"
        if np.isclose(tg_v, b_tg):
            stg = f"\\textbf{{{stg}}}"
        t2_rows.append([lbl, sf, sw, stw, spk, stg])
    write_tex_table(
        out_tab_dir,
        "tab2_emt_ligand_shift.tex",
        "lrrrrr",
        [
            "Architecture (76 EMT targets)",
            r"Full 10d $\downarrow$",
            r"Wdr.\ MSE $\downarrow$",
            r"Top-30 wdr.\ $\downarrow$",
            r"Peak MSE $\downarrow$",
            r"TGF$\beta$1 wdr.\ $\downarrow$",
        ],
        t2_rows,
    )

    # =========================================================================
    # Table 3: Locked 24-World Multi-TF Factorial + Parameter-Matched Local-LTC Separation
    # =========================================================================
    mtf_order = [
        ("local_const", "Local Constant ($gW$)"),
        ("coupled_const", "Coupled Constant (SSM)"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("ltc_coupled", "Coupled LTC"),
        ("perturb_ode", "PerturbODE"),
        ("ltc_local", "Source-Local LTC"),
        ("memgrn_additive", "MemGRN ($\\Gamma = 0$ ablation)"),
        ("memgrn", "\\textbf{MemGRN (ours)}"),
    ]
    add_m = mtf["additive"]["means"]
    coop_m = mtf["cooperative"]["means"]
    b_add = min(add_m[k]["test_mse"] for k, _ in mtf_order)
    b_coo = min(coop_m[k]["test_mse"] for k, _ in mtf_order)
    b_att = min(0.5 * (add_m[k]["attribution_rmse"] + coop_m[k]["attribution_rmse"]) for k, _ in mtf_order)
    b_eap = max(0.5 * (add_m[k]["edge_ap"] + coop_m[k]["edge_ap"]) for k, _ in mtf_order)
    t3_rows: list[list[str]] = []
    for k, lbl in mtf_order:
        a_v = add_m[k]["test_mse"]
        c_v = coop_m[k]["test_mse"]
        at_v = 0.5 * (add_m[k]["attribution_rmse"] + coop_m[k]["attribution_rmse"])
        ap_v = 0.5 * (add_m[k]["edge_ap"] + coop_m[k]["edge_ap"])
        sa = f"\\textbf{{{a_v:.4f}}}" if np.isclose(a_v, b_add) else f"{a_v:.4f}"
        sc = f"\\textbf{{{c_v:.4f}}}" if np.isclose(c_v, b_coo) else f"{c_v:.4f}"
        sat = f"\\textbf{{{at_v:.4f}}}" if np.isclose(at_v, b_att) else f"{at_v:.4f}"
        sap = f"\\textbf{{{ap_v:.3f}}}" if np.isclose(ap_v, b_eap) else f"{ap_v:.3f}"
        t3_rows.append([lbl, str(add_m[k]["n_params"]), sa, sc, sat, sap])

    # Bottom panel of Table 3: Parameter-matched MemGRN vs Source-Local LTC (Theorem 1(iii))
    t3_rows.append(["\\midrule"])
    pb = po["part_b_ltc_separation"]
    for k, lbl in [
        ("memgrn_matched", "\\textbf{MemGRN (Thm.~1(iii))}"),
        ("ltc_local_matched", "Source-Local LTC (matched)"),
        ("local_const_matched", "Local Constant (matched)"),
    ]:
        dv = pb[k]
        pk_s = f"{1e5 * dv['delayed_peak_post_mse']:.2f}$\\times 10^{{-5}}$"
        amp_s = f"{100.0 * dv['steady_state_amp_invariance_error']:.2f}\\%"
        if k == "memgrn_matched":
            pk_s = f"\\textbf{{{pk_s}}}"
        t3_rows.append([lbl, str(dv["n_params"]), pk_s, amp_s, "--", "--"])
    write_tex_table(
        out_tab_dir,
        "tab3_multitf_and_ltc_sep.tex",
        "lrrrrr",
        [
            "Architecture (24 worlds / Thm.~1(iii))",
            "Params",
            r"Add.\ / Peak $\downarrow$",
            r"Coop.\ / Amp.\ $\downarrow$",
            r"Attr.\ RMSE $\downarrow$",
            r"Edge AP $\uparrow$",
        ],
        t3_rows,
    )

    # =========================================================================
    # Table 4: Yeast Msn2 Fair Source-CV Joint Training
    # =========================================================================
    jt = joint["joint_training"]
    t4_rows: list[list[str]] = []
    j_order = [
        ("memgrn", "\\textbf{MemGRN (asymmetric, ours)}"),
        ("memgrn_sym", "MemGRN (symmetric rate)"),
        ("ltc_local", "Source-Local LTC"),
        ("latent_ode", "Latent Neural ODE"),
        ("perturb_ode", "PerturbODE"),
        ("local_const", "Constant $gW$"),
    ]
    best_30 = min(jt[k]["held_30min_post_mse"] for k, _ in j_order)
    best_10 = min(jt[k]["ten_trains_post_mse"] for k, _ in j_order)
    best_40 = min(jt[k]["ten_trains_first40_mse"] for k, _ in j_order)
    best_ns = min(jt[k]["ten_trains_no_sip18_mse"] for k, _ in j_order)
    for k, lbl in j_order:
        d = jt[k]
        s30 = f"{d['held_30min_post_mse']:.4f}"
        s10 = f"{d['ten_trains_post_mse']:.4f}"
        s40 = f"{d['ten_trains_first40_mse']:.4f}"
        sns = f"{d['ten_trains_no_sip18_mse']:.4f}"
        if np.isclose(d["held_30min_post_mse"], best_30):
            s30 = f"\\textbf{{{s30}}}"
        if np.isclose(d["ten_trains_post_mse"], best_10):
            s10 = f"\\textbf{{{s10}}}"
        if np.isclose(d["ten_trains_first40_mse"], best_40):
            s40 = f"\\textbf{{{s40}}}"
        if np.isclose(d["ten_trains_no_sip18_mse"], best_ns):
            sns = f"\\textbf{{{sns}}}"
        t4_rows.append([lbl, str(d["n_params"]), s30, s10, s40, sns])
    write_tex_table(
        out_tab_dir,
        "tab4_msn2_joint_and_cross.tex",
        "lrrrrr",
        [
            "Joint source-CV model",
            "Params",
            r"30-min $\downarrow$",
            r"10 trains $\downarrow$",
            r"First 40m $\downarrow$",
            r"No \texttt{SIP18} $\downarrow$",
        ],
        t4_rows,
    )

    # =========================================================================
    # Table 5: Partially Observed Cofactors (6-of-8 Ground Truth + Fibroblast 177 Genes)
    # =========================================================================
    pa = po["part_a_partial_obs_6of8_summary"]
    t5_rows: list[list[str]] = []
    pa_order = [
        ("memgrn_ortho", "\\textbf{MemGRN-Ortho (ours)}"),
        ("memgrn", "MemGRN (6 obs.\\ TFs)"),
        ("memgrn_unproj", "MemGRN + Unproj."),
        ("local_const", "Local Const.\\ ($gW$)"),
        ("latent_ode", "Latent Neural ODE"),
        ("perturb_ode", "PerturbODE"),
    ]
    b_pmse = min(pa[k]["test_mse"] for k, _ in pa_order)
    b_patr = min(pa[k]["gt_obs_attr_rmse"] for k, _ in pa_order)
    b_pau = max(pa[k]["gt_obs_edge_auroc"] for k, _ in pa_order)
    for i_p, (k, lbl) in enumerate(pa_order):
        d = pa[k]
        sm = f"\\textbf{{{d['test_mse']:.4f}}}" if np.isclose(d["test_mse"], b_pmse) else f"{d['test_mse']:.4f}"
        sa = f"\\textbf{{{d['gt_obs_attr_rmse']:.4f}}}" if np.isclose(d["gt_obs_attr_rmse"], b_patr) else f"{d['gt_obs_attr_rmse']:.4f}"
        su = f"\\textbf{{{d['gt_obs_edge_auroc']:.3f}}}" if np.isclose(d["gt_obs_edge_auroc"], b_pau) else f"{d['gt_obs_edge_auroc']:.3f}"
        sd = f"{100.0 * d['obs_attr_drift_vs_memgrn']:.1f}\\%"
        t5_rows.append(["6/8 GT" if i_p == 0 else "", lbl, sm, sa, su, sd])

    t5_rows.append(["\\midrule"])
    o_order = [
        ("memgrn_ortho", "\\textbf{MemGRN-Ortho (ours)}"),
        ("latent_ode", "Latent Neural ODE"),
        ("memgrn", "MemGRN (NF-$\\kappa$B)"),
        ("perturb_ode", "PerturbODE"),
        ("local_const", "Constant $gW$"),
    ]
    b_il = min(ortho[k]["IL1_post_mse"] for k, _ in o_order)
    b_tn = min(ortho[k]["TNF_post_mse"] for k, _ in o_order)
    b_mn = min(ortho[k]["mean_post_mse"] for k, _ in o_order)
    for i_o, (k, lbl) in enumerate(o_order):
        d = ortho[k]
        s_mn = f"\\textbf{{{d['mean_post_mse']:.3f}}}" if np.isclose(d["mean_post_mse"], b_mn) else f"{d['mean_post_mse']:.3f}"
        s_il = f"\\textbf{{{d['IL1_post_mse']:.3f}}}" if np.isclose(d["IL1_post_mse"], b_il) else f"{d['IL1_post_mse']:.3f}"
        s_tn = f"\\textbf{{{d['TNF_post_mse']:.3f}}}" if np.isclose(d["TNF_post_mse"], b_tn) else f"{d['TNF_post_mse']:.3f}"
        s_dr = f"{100.0 * d['mean_nfkb_attr_drift']:.1f}\\%"
        t5_rows.append(["Fibro." if i_o == 0 else "", lbl, s_mn, s_il, s_tn, s_dr])
    write_tex_table(
        out_tab_dir,
        "tab5_ortho_ground_truth_and_fibroblast.tex",
        "llrrrr",
        [
            "Suite",
            "Model",
            r"MSE $\downarrow$",
            r"RMSE / IL-1 $\downarrow$",
            r"AUROC / TNF",
            r"Drift $\downarrow$",
        ],
        t5_rows,
    )

    # =========================================================================
    # Appendix Table A1: 4-TF Mammalian Macrophage Benchmark (14 stimuli x 477 GRS genes)
    # =========================================================================
    ta1_rows: list[list[str]] = []
    mac_order = [
        ("memgrn", "\\textbf{MemGRN (ours)}"),
        ("memgrn_additive", "MemGRN ($\\Gamma = 0$ ablation)"),
        ("local_const", "Local Constant ($gW$)"),
        ("ltc_local", "Source-Local LTC"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("coupled_const", "Coupled Constant (SSM)"),
        ("ltc_coupled", "Coupled LTC"),
        ("perturb_ode", "PerturbODE"),
    ]
    best_auroc = max(mac[k]["regulator_attribution_auroc"] for k, _ in mac_order)
    best_top1 = max(mac[k]["top1_regulator_accuracy"] for k, _ in mac_order)
    best_sel = min(mac[k]["selective_decorrelated_mse"] for k, _ in mac_order)
    best_all = min(mac[k]["all_14_stimuli_mse"] for k, _ in mac_order)
    for k, lbl in mac_order:
        d = mac[k]
        s_au = f"\\textbf{{{d['regulator_attribution_auroc']:.3f}}}" if np.isclose(d["regulator_attribution_auroc"], best_auroc) else f"{d['regulator_attribution_auroc']:.3f}"
        s_t1 = f"\\textbf{{{100.0 * d['top1_regulator_accuracy']:.1f}\\%}}" if np.isclose(d["top1_regulator_accuracy"], best_top1) else f"{100.0 * d['top1_regulator_accuracy']:.1f}\\%"
        s_se = f"\\textbf{{{d['selective_decorrelated_mse']:.3f}}}" if np.isclose(d["selective_decorrelated_mse"], best_sel) else f"{d['selective_decorrelated_mse']:.3f}"
        s_al = f"\\textbf{{{d['all_14_stimuli_mse']:.3f}}}" if np.isclose(d["all_14_stimuli_mse"], best_all) else f"{d['all_14_stimuli_mse']:.3f}"
        ta1_rows.append([lbl, s_au, s_t1, s_se, s_al])
    write_tex_table(
        out_tab_dir,
        "tab_a1_macrophage_4tf.tex",
        "lrrrr",
        [
            "Architecture (4 TFs, 477 genes)",
            r"GRS AUROC $\uparrow$",
            r"Top-1 TF $\uparrow$",
            r"De-corr.\ MSE $\downarrow$",
            r"14-stim.\ MSE $\downarrow$",
        ],
        ta1_rows,
    )

    # =========================================================================
    # Appendix Table A2: Yeast Chromatin-Memory Portraits & S=8..512 Scaling Benchmark
    # =========================================================================
    portraits = interp["yeast_portraits"]
    scaling = interp["scaling_benchmark"]
    ta2_rows: list[list[str]] = []
    for g in ("HXK1", "DCS2", "RTN2", "DDR2", "ALD3", "TKL2", "SIP18"):
        pg = portraits[g]
        cls_short = "Fast (open)" if g in ("HXK1", "DCS2") else ("Interm." if g == "RTN2" else "Slow (remod.)")
        ta2_rows.append(
            [
                f"\\texttt{{{g}}}",
                cls_short,
                f"{pg['effective_washout_tau_min']:.1f}\\,min",
                f"{pg['asymmetry_rise_over_recovery']:.2f}$\\times$",
                f"{pg['ten_train_post_mse']:.4f}",
            ]
        )
    ta2_rows.append(["\\midrule"])
    for r in scaling:
        s_n = f"$S={r['n_tfs']}$ ({r['n_genes']})"
        ltc_mp = f"{r['coupled_ltc_mem_params']:,}" if r["n_tfs"] <= 256 else "--"
        ta2_rows.append(
            [
                s_n,
                f"{r['memgrn_mem_params']:,} ({r['memgrn_ms_per_step']:.0f}\\,ms)",
                f"{r['coupled_adaptive_mem_params']:,} ({r['coupled_adaptive_ms_per_step']:.0f}\\,ms)",
                ltc_mp,
                f"{r['coupled_adaptive_mem_params'] / r['memgrn_mem_params']:.1f}$\\times$",
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a2_portraits_and_scaling.tex",
        "llrrr",
        [
            r"Promoter / $S(N)$",
            "Class / MemGRN",
            r"$\tau_{\mathrm{off}}$ / Cpl-Ad.",
            r"Asym.\ / Cpl-LTC",
            r"MSE / Ratio",
        ],
        ta2_rows,
    )

    # =========================================================================
    # Appendix Tables A3-A8: Six-Promoter, LOPO, Synthetic, Multiscale, RTN2, Swap
    # =========================================================================
    modes = ["constant", "activity", "memory", "both"]
    ta3_rows: list[list[str]] = []
    for r_orig, r_lbl in [
        ("local_const", "Joint Constant-$gW$ readout (7 genes)"),
        ("memgrn", "Joint MemGRN readout (7 genes)"),
    ]:
        vals = [joint["cross_readout_2x2"][r_orig][m] for m in modes]
        b_i = int(np.argmin(vals))
        cells = [
            f"\\textbf{{{vals[i]:.4f}}}" if i == b_i else f"{vals[i]:.4f}" for i in range(4)
        ]
        ta3_rows.append([r_lbl, *cells])
    ta3_rows.append(["\\midrule"])
    genes = six["genes"]
    arr = np.array([[six["results"][g][m]["mse"] for m in modes] for g in genes])
    for g_i, g in enumerate(genes):
        best_m = int(np.argmin(arr[g_i]))
        cells = [
            f"\\textbf{{{arr[g_i, m_i]:.4f}}}"
            if m_i == best_m
            else f"{arr[g_i, m_i]:.4f}"
            for m_i in range(4)
        ]
        ta3_rows.append([f"\\texttt{{{g}}} (archived single-pulse $W$)", *cells])
    means = arr.mean(axis=0)
    best_mean = int(np.argmin(means))
    ta3_rows.append(
        [
            "Six-promoter mean",
            *[
                f"\\textbf{{{means[m_i]:.4f}}}" if m_i == best_mean else f"{means[m_i]:.4f}"
                for m_i in range(4)
            ],
        ]
    )
    write_tex_table(
        out_tab_dir,
        "tab_a3_six_promoters.tex",
        "lrrrr",
        [
            "Readout origin / promoter",
            r"Constant $\downarrow$",
            r"Activity only $\downarrow$",
            r"Memory only $\downarrow$",
            r"Activity + memory $\downarrow$",
        ],
        ta3_rows,
    )

    lopo = joint["shared_filter_lopo_transfer"]["by_promoter"]
    ta4_rows: list[list[str]] = []
    for g_name, g_vals in lopo.items():
        m_v = g_vals["shared_memgrn_filter_mse"]
        c_v = g_vals["shared_const_filter_mse"]
        m_s = f"\\textbf{{{m_v:.4f}}}" if m_v < c_v else f"{m_v:.4f}"
        c_s = f"\\textbf{{{c_v:.4f}}}" if c_v < m_v else f"{c_v:.4f}"
        ta4_rows.append(
            [f"\\texttt{{{g_name}}}", c_s, m_s, f"{100.0 * (1.0 - m_v / c_v):+.1f}\\%"]
        )
    ta4_rows.append(["\\midrule"])
    m_mean = joint["shared_filter_lopo_transfer"]["mean_memgrn_shared"]
    c_mean = joint["shared_filter_lopo_transfer"]["mean_const_shared"]
    ta4_rows.append(
        [
            "7-promoter LOPO mean",
            f"{c_mean:.4f}",
            f"\\textbf{{{m_mean:.4f}}}",
            f"{100.0 * (1.0 - m_mean / c_mean):+.1f}\\%",
        ]
    )
    write_tex_table(
        out_tab_dir,
        "tab_a4_lopo_shared_filter.tex",
        "lrrr",
        [
            "Held-out promoter",
            r"Shared Const.\ filter $\downarrow$",
            r"Shared MemGRN filter $\downarrow$",
            r"MSE reduction $\uparrow$",
        ],
        ta4_rows,
    )

    ta5_rows: list[list[str]] = []
    params_map = [168, 186, 274, 1158]
    model_names = [
        "Constant $gW$",
        "Two-rate $gW$",
        "\\textbf{MemGRN (ours)}",
        "Latent Neural ODE",
    ]
    for idx, tag, data_s, r_idx in [
        (0, "Occupancy-gated ($n=8$)", occ, 1),
        (1, "Mass-action buffer ($n=12$)", buf, 1),
        (2, "Constant control ($n=12$)", buf, 0),
    ]:
        if idx > 0:
            ta5_rows.append(["\\midrule"])
        means_s = data_s["means"][r_idx]
        best_full = int(np.argmin([m[0] for m in means_s[:4]]))
        best_tail = int(np.argmin([m[1] for m in means_s[:4]]))
        best_rec = int(np.argmax([m[3] for m in means_s[:4]]))
        for m_i in range(4):
            vals = means_s[m_i]
            f_str = f"{1e6 * vals[0]:.2f}"
            t_str = f"{1e6 * vals[1]:.2f}"
            r_str = f"{100.0 * vals[3]:.2f}\\%"
            if m_i == best_full:
                f_str = f"\\textbf{{{f_str}}}"
            if m_i == best_tail:
                t_str = f"\\textbf{{{t_str}}}"
            if m_i == best_rec:
                r_str = f"\\textbf{{{r_str}}}"
            ta5_rows.append(
                [
                    tag if m_i == 0 else "",
                    model_names[m_i],
                    str(params_map[m_i]),
                    f_str,
                    t_str,
                    r_str,
                ]
            )
    write_tex_table(
        out_tab_dir,
        "tab_a5_synthetic_buffering.tex",
        "llrrrr",
        [
            "Kinetic regime",
            "Model",
            "Params",
            r"Full MSE $\downarrow$",
            r"Tail MSE $\downarrow$",
            r"Pair ID $\uparrow$",
        ],
        ta5_rows,
    )

    multi = load_stage3("multiscale_gw")
    mixed = load_stage3("mixed_directed_memory")
    ta6_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Multiscale MemGRN}"),
        ("history_basis", "History time basis"),
        ("current_hybrid", "Single-rate hybrid"),
        ("state_space", "Linear state space"),
    ]:
        c_dat = multi["classes"][k]["biphasic"]
        m_all = multi["means"][k]
        ta6_rows.append(
            [
                "Multiscale suite" if k == "learned_gw" else "",
                lbl,
                f"{100.0 * c_dat['error']:.2f}\\%",
                f"{100.0 * m_all['reversal']:.1f}\\%",
                f"{m_all['delayed_ap']:.3f}",
            ]
        )
    ta6_rows.append(["\\midrule"])
    for k, lbl in [
        ("memory_driver", "\\textbf{MemGRN + oscillator}"),
        ("latent_ODE", "Latent Neural ODE"),
        ("state_space", "Linear state space"),
        ("directed_memory", "MemGRN (no oscillator)"),
        ("instant_driver", "Memoryless + oscillator"),
    ]:
        m_dat = mixed["means"][k]
        ta6_rows.append(
            [
                "Shared driver" if k == "memory_driver" else "",
                lbl,
                f"{100.0 * m_dat['tail_error_ratio']:.2f}\\%",
                f"{100.0 * m_dat['tail_sign']:.1f}\\%",
                f"{m_dat['delayed_ap']:.3f}",
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a6_multiscale_driver.tex",
        "llrrr",
        [
            "Benchmark",
            "Model",
            r"Tail error $\downarrow$",
            r"Sign acc. $\uparrow$",
            r"Edge AP $\uparrow$",
        ],
        ta6_rows,
    )

    rtn_sw = load_stage3("msn2_rtn2_switch_neural")
    reset = load_stage3("msn2_memory_reset")
    ta7_rows: list[list[str]] = []
    for idx_r, (k, lbl) in enumerate(
        [
            ("constant", "Constant rates"),
            ("switch_only", "Discrete switch only"),
            ("neural_only", "Continuous neural rate"),
            ("full", "Switch + neural rate"),
        ]
    ):
        d_sw = rtn_sw["models"][k]
        ta7_rows.append(
            [
                "\\texttt{RTN2} switch" if idx_r == 0 else "",
                lbl,
                f"{d_sw['primary']:.4f}",
                f"{d_sw['half_step']:.4f}",
                f"{d_sw['first40']:.4f}",
            ]
        )
    ta7_rows.append(["\\midrule"])
    for tr_id, tr_lbl in [
        ("FM_6_5min_690nM", "6 pulses (intact / reset)"),
        ("FM_8_5min_690nM", "8 pulses (intact / reset)"),
    ]:
        p_all = reset["results"][tr_id]["panels"]["all"]["mse"]
        p_rtn = reset["results"][tr_id]["promoters"]["RTN2"]
        ta7_rows.append(
            [
                "Filter reset" if tr_id == "FM_6_5min_690nM" else "",
                tr_lbl,
                f"{p_all['intact']:.4f} / {p_all['reset']:.4f}",
                f"+{p_rtn['shift_attributable_to_reporter_at_off']:.3f}",
                f"+{p_rtn['shift_attributable_to_production_after_off']:.3f}",
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a7_rtn2_and_reset.tex",
        "llrrr",
        [
            "Study",
            "Condition / model",
            r"Primary MSE $\downarrow$",
            r"Refined / Carryover",
            r"First 40m / Post-off",
        ],
        ta7_rows,
    )

    ta8_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Learned-memory $gW$ (all $96$)}"),
        ("fixed_gw", "Fixed-rate $gW$ (all $96$)"),
        ("history_basis", "History time basis (all $96$)"),
        ("oracle", "Oracle templates (all $96$)"),
    ]:
        rep_v = 100.0 * swap["overall"][k]["repeat"]
        swp_v = 100.0 * swap["overall"][k]["swap"]
        orc_v = 100.0 * swap["overall"]["oracle"]["swap"]
        ta8_rows.append(
            [lbl, f"{rep_v:.2f}\\%", f"\\textbf{{{swp_v:.2f}\\%}}", f"+{swp_v - rep_v:.2f}", f"{orc_v:.2f}\\%"]
        )
    ta8_rows.append(["\\midrule"])
    for sg_key, sg_lbl in [
        ("aligned", "Subgroup: Aligned signs ($48$)"),
        ("opposing", "Subgroup: Opposing signs ($48$)"),
        ("fast/fast opposing", "Subgroup: Fast opposing ($11$)"),
        ("biphasic/biphasic", "Subgroup: Biphasic pairs ($16$)"),
    ]:
        sg_m = swap["subgroups"][sg_key]["means"]["learned_gw"]
        sr = 100.0 * sg_m["repeat"]
        ss = 100.0 * sg_m["swap"]
        so = 100.0 * swap["subgroups"][sg_key]["means"]["oracle"]["swap"]
        ta8_rows.append(
            [sg_lbl, f"{sr:.2f}\\%", f"\\textbf{{{ss:.2f}\\%}}", f"+{ss - sr:.2f}", f"{so:.2f}\\%"]
        )
    ta8_rows.append(["\\midrule"])
    name_map = {
        "learned_gw": "5-pt AR(1): Learned $gW$",
        "fixed_gw": "5-pt AR(1): Fixed $gW$",
        "history_basis": "5-pt AR(1): History basis",
    }
    for idx_c, r_cert in enumerate(cert["rows"]):
        p_meta = cert_meta["panels"][idx_c]
        ci_lo = 100.0 * p_meta["ci"][0]
        ci_hi = 100.0 * p_meta["ci"][1]
        ta8_rows.append(
            [
                name_map[r_cert["model"]],
                f"{100.0 * r_cert['full_window']:.2f}\\%",
                f"\\textbf{{{100.0 * r_cert['progression']:.2f}\\%}}",
                f"+{100.0 * r_cert['secondary_gap']:.2f}",
                f"[{ci_lo:+.2f}, {ci_hi:+.2f}]",
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a8_swap_and_cert.tex",
        "lrrrr",
        [
            "Template / subgroup / 5-pt policy",
            r"Baseline $\uparrow$",
            r"Swap / Selected $\uparrow$",
            r"Gain (pp) $\uparrow$",
            r"Oracle / 98.5\% CI",
        ],
        ta8_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latex-dir",
        type=Path,
        default=ROOT.parent / "memgrn-latex",
        help="Path to the companion LaTeX repository",
    )
    args = parser.parse_args()
    out_latex_fig = args.latex_dir / "figures"
    out_latex_tab = args.latex_dir / "tables"
    OUT_FIG_REPO.mkdir(parents=True, exist_ok=True)
    out_latex_fig.mkdir(parents=True, exist_ok=True)
    out_latex_tab.mkdir(parents=True, exist_ok=True)

    build_fig1_architecture(out_latex_fig)
    build_fig2_human_longitudinal(out_latex_fig)
    build_fig3_multitf_ltc_ortho(out_latex_fig)
    build_fig4_yeast_chromatin_portraits(out_latex_fig)
    build_fig5_mammalian_and_scaling(out_latex_fig)
    build_all_tables(out_latex_tab)

    ledger_path = ROOT / "results" / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps({"source_sha256": SOURCE_HASHES}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built 5 figures and 13 tables from {len(SOURCE_HASHES)} verified artifacts.")


if __name__ == "__main__":
    main()
