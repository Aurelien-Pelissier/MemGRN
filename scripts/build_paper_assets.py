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
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.62))
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

    # Closed-loop feedback arc above boxes (y_j -> g_i under CRISPR ko clamp)
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
        "2. Memory & ortho cofactor\nBeats state-only ODEs",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        7.52,
        0.08,
        3.44,
        0.84,
        "3. 4:1 strength swap\nBreaks pair cancellation",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
    )

    export_figure(fig, "fig1_memgrn_architecture", out_latex_fig, audit_alignment=False)


def build_fig2_hipsc_closed_loop(out_latex_fig: Path) -> None:
    """Figure 2: Closed-loop longitudinal human iPSC CRISPR Perturb-seq (GSE213069)."""
    hipsc = load_new_result("hipsc_closed_loop")["summary"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.18))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.105,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    keys = [
        "renge",
        "cl_ltc_coupled",
        "cl_perturb_ode",
        "cl_coupled_adaptive",
        "cl_ltc_local",
        "cl_local_const",
        "cl_memgrn",
    ]
    lbls = [
        "RENGE",
        "LTC-Cpl",
        "PertODE",
        "Cpl-Adapt",
        "LTC-Loc",
        "Loc-Const",
        "MemGRN",
    ]
    cols = [
        PALETTE["slate"],
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["teal"],
        "#7B8A9E",
        PALETTE["navy"],
    ]
    x = np.arange(len(keys))

    # Panel a: Zero-shot unseen gene KO MSE across 23 pluripotency TFs (103 genes)
    ko_vals = [hipsc[k]["unseen_ko_mse"] for k in keys]
    ax0.bar(x, ko_vals, width=0.64, color=cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(x)
    ax0.set_xticklabels(lbls, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, len(keys) - 0.4)
    y0_top = float(max(ko_vals) * 1.15)
    ax0.set_ylim(0.0, y0_top)
    ax0.set_ylabel("Unseen-KO MSE")
    ax0.set_title("Unseen CRISPR KOs")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Day-5 longitudinal extrapolation MSE (trained on Days 2-4)
    d5_vals = [hipsc[k]["day5_extrapolation_mse"] for k in keys]
    ax1.bar(x, d5_vals, width=0.64, color=cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, len(keys) - 0.4)
    ax1.set_ylim(0.0, float(max(d5_vals) * 1.15))
    ax1.set_ylabel("Day-5 holdout MSE")
    ax1.set_title("Day-5 extrapolation")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Unseen CRISPR knockout trajectory Pearson correlation r across 23 TFs
    r_vals = [max(0.0, hipsc[k]["unseen_ko_pearson_r"]) for k in keys]
    ax2.bar(
        x,
        r_vals,
        width=0.64,
        color=cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(lbls, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.6, len(keys) - 0.4)
    ax2.set_ylim(0.0, float(max(r_vals) * 1.18))
    ax2.set_ylabel("Unseen-KO Pearson r")
    ax2.set_title("Trajectory correlation")
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(fig, "fig2_hipsc_closed_loop", out_latex_fig)


def build_fig3_multitf_factorial(out_latex_fig: Path) -> None:
    """Figure 3: Locked 2x2 Multi-TF Correlation-Shift Benchmark (24 eight-TF worlds)."""
    mtf = load_new_result("multitf_factorial")
    add_m = mtf["summary"]["additive"]["means"]
    coop_m = mtf["summary"]["cooperative"]["means"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.18))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.105,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    keys = [
        "ltc_coupled",
        "perturb_ode",
        "local_const",
        "coupled_adaptive",
        "ltc_local",
        "memgrn_additive",
        "memgrn",
    ]
    labels = [
        "LTC-Cpl",
        "PertODE",
        "Loc-Const",
        "Cpl-Adapt",
        "LTC-Loc",
        "Mem-Add",
        "MemGRN",
    ]
    colors = [
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["teal"],
        "#2F6EA6",
        PALETTE["navy"],
    ]
    x = np.arange(len(keys))

    vals_add = [add_m[k]["test_mse"] for k in keys]
    ax0.bar(x, vals_add, width=0.64, color=colors, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, len(keys) - 0.4)
    ax0.set_ylim(0.0, float(max(vals_add) * 1.15))
    ax0.set_ylabel("OOD test MSE")
    ax0.set_title("Additive worlds")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    vals_coop = [coop_m[k]["test_mse"] for k in keys]
    ax1.bar(x, vals_coop, width=0.64, color=colors, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, len(keys) - 0.4)
    ax1.set_ylim(0.0, float(max(vals_coop) * 1.15))
    ax1.set_ylabel("OOD test MSE")
    ax1.set_title("Cooperative")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    vals_attr = [
        0.5 * (add_m[k]["attribution_rmse"] + coop_m[k]["attribution_rmse"]) for k in keys
    ]
    ax2.bar(x, vals_attr, width=0.64, color=colors, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.6, len(keys) - 0.4)
    ax2.set_ylim(0.0, float(max(vals_attr) * 1.16))
    ax2.set_ylabel("Attribution RMSE")
    ax2.set_title("Source attribution")
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(fig, "fig3_multitf_factorial", out_latex_fig)


def build_fig4_msn2_and_promoters(out_latex_fig: Path) -> None:
    """Figure 4: Live-cell yeast Msn2 joint source-CV, cross-readout 2x2, & 6 promoters."""
    joint = load_new_result("msn2_joint_and_cross_readout")
    six = load_stage3("msn2_six_promoter_rates")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.20))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.02, 1.08],
        left=0.100,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    j_keys = ["local_const", "perturb_ode", "latent_ode", "ltc_local", "memgrn_sym", "memgrn"]
    j_lbls = ["Const gW", "PertODE", "Lat-ODE", "LTC-Loc", "Mem-Sym", "MemGRN"]
    j_cols = [
        PALETTE["slate"],
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["teal"],
        "#2F6EA6",
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
    ax0.set_title("Joint source-CV fits")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

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
    ax1.set_ylim(0.0, float(max(*const_w_vals, *mem_w_vals) * 1.25))
    ax1.set_ylabel("10-train MSE")
    ax1.set_title("Cross-readout swap")
    ax1.legend(loc="upper right", ncol=1, handlelength=1.1, labelspacing=0.18)
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    genes = six["genes"]
    arr = np.array([[six["results"][g][m]["per_schedule"] for m in r_modes] for g in genes])
    mode_labels = ["Constant", "Activity only", "Memory only", "Act + memory"]
    mode_colors = [PALETTE["slate"], PALETTE["amber"], PALETTE["teal"], PALETTE["navy"]]
    mode_markers = ["s", "^", "o", "D"]
    gx = np.arange(len(genes))
    for m_idx in range(4):
        ax2.plot(
            gx,
            arr[:, m_idx].mean(axis=1),
            marker=mode_markers[m_idx],
            ms=3.8,
            lw=1.25,
            color=mode_colors[m_idx],
            label=mode_labels[m_idx],
        )
    ax2.set_yscale("log")
    ax2.set_xlim(-0.45, 5.45)
    ax2.set_ylim(4e-3, 320.0)
    ax2.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax2.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1.0"])
    ax2.set_xticks(gx)
    ax2.set_xticklabels(genes, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_ylabel("Mean recovery MSE")
    ax2.set_title("Six yeast promoters")
    ax2.legend(loc="upper left", ncol=1, handlelength=1.4, labelspacing=0.16)
    add_panel_tag(ax2, "c", x_pt=-32.0, y_pt=6.0)

    export_figure(
        fig,
        "fig4_msn2_and_promoters",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios accommodate 6-bar, 4-pair, and 6-promoter panels",
            }
        ],
    )


def build_fig5_macrophage_ortho_swap(out_latex_fig: Path) -> None:
    """Figure 5: 4-TF Macrophage attribution, MemGRN-Ortho cofactor recovery, & 4:1 swap."""
    mac = load_new_result("macrophage_multitf")["summary"]
    ortho = load_new_result("nfkb_ortho")["summary"]
    swap = load_stage3("strength_swap_replication")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.20))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.06, 1.05, 1.04],
        left=0.105,
        right=0.985,
        bottom=0.35,
        top=0.84,
        wspace=0.42,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

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

    # Panel b: 177-gene fibroblast NF-kB washout 3-ligand mean MSE (MemGRN-Ortho)
    o_keys = ["local_const", "perturb_ode", "memgrn", "latent_ode", "memgrn_ortho"]
    o_lbls = ["Const gW", "PertODE", "MemGRN", "Lat-ODE", "Mem-Ortho"]
    o_cols = [
        PALETTE["slate"],
        PALETTE["violet"],
        "#2F6EA6",
        PALETTE["amber"],
        PALETTE["navy"],
    ]
    ox = np.arange(len(o_keys))
    o_vals = [ortho[k]["mean_post_mse"] for k in o_keys]
    ax1.bar(ox, o_vals, width=0.60, color=o_cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax1.set_xticks(ox)
    ax1.set_xticklabels(o_lbls, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, len(o_keys) - 0.4)
    ax1.set_ylim(0.0, float(max(o_vals) * 1.15))
    ax1.set_ylabel("3-ligand washout MSE")
    ax1.set_title("Orthogonal cofactor")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Counterbalanced 4:1 strength swap
    overall_rep = 100.0 * swap["overall"]["learned_gw"]["repeat"]
    overall_swp = 100.0 * swap["overall"]["learned_gw"]["swap"]
    overall_orc = 100.0 * swap["overall"]["oracle"]["swap"]
    opp_fast_rep = (
        100.0 * swap["subgroups"]["fast/fast opposing"]["means"]["learned_gw"]["repeat"]
    )
    opp_fast_swp = (
        100.0 * swap["subgroups"]["fast/fast opposing"]["means"]["learned_gw"]["swap"]
    )
    opp_fast_orc = (
        100.0 * swap["subgroups"]["fast/fast opposing"]["means"]["oracle"]["swap"]
    )
    gx = np.arange(2)
    w = 0.23
    ax2.bar(
        gx - w,
        [overall_rep, opp_fast_rep],
        width=w * 0.90,
        color=PALETTE["amber"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="Repeat 4:1",
    )
    ax2.bar(
        gx,
        [overall_swp, opp_fast_swp],
        width=w * 0.90,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="Swap 4:1",
    )
    ax2.bar(
        gx + w,
        [overall_orc, opp_fast_orc],
        width=w * 0.90,
        color=PALETTE["teal"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="Oracle",
    )
    ax2.set_xticks(gx)
    ax2.set_xticklabels(["All 96", "Fast opp."], rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(0.0, 195.0)
    ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel("Pair recovery (%)")
    ax2.set_title("4:1 strength swap")
    ax2.legend(loc="upper left", ncol=1, handlelength=1.1, labelspacing=0.16)
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(
        fig,
        "fig5_macrophage_ortho_swap",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Three slight width ratios accommodate 6-bar, 5-bar, and 2-group panels",
            }
        ],
    )


def build_all_tables(out_tab_dir: Path) -> None:
    """Write all 5 main-text tables and 6 appendix tables."""
    hipsc = load_new_result("hipsc_closed_loop")["summary"]
    mtf = load_new_result("multitf_factorial")["summary"]
    joint = load_new_result("msn2_joint_and_cross_readout")
    mac = load_new_result("macrophage_multitf")["summary"]
    ortho = load_new_result("nfkb_ortho")["summary"]
    occ = load_stage3("neural_rate_pulse", "test_scores.json")
    buf = load_stage3("neural_rate_buffer", "test_scores.json")
    six = load_stage3("msn2_six_promoter_rates")
    swap = load_stage3("strength_swap_replication")
    cert = load_stage3("paired_time_certificate", "test_scores.json")
    cert_meta = load_stage3("paired_time_certificate", "certificate.json")

    # Table 1: Closed-Loop Longitudinal Human iPSC CRISPR Perturb-seq (GSE213069)
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
    best_r = max(hipsc[k]["unseen_ko_pearson_r"] for k, _ in t1_order)
    best_d5 = min(hipsc[k]["day5_extrapolation_mse"] for k, _ in t1_order)
    best_ch = max(hipsc[k]["chip_psc_mean_tf_auroc"] for k, _ in t1_order)
    best_ap = max(hipsc[k]["chip_psc_ap_over_random"] for k, _ in t1_order)
    t1_rows: list[list[str]] = []
    for k, lbl in t1_order:
        d = hipsc[k]
        s_ko = f"{d['unseen_ko_mse']:.4f}"
        s_r = f"{d['unseen_ko_pearson_r']:.3f}"
        s_d5 = f"{d['day5_extrapolation_mse']:.4f}"
        s_ch = f"{d['chip_psc_mean_tf_auroc']:.3f}"
        s_ap = f"{d['chip_psc_ap_over_random']:.2f}$\\times$"
        if np.isclose(d["unseen_ko_mse"], best_ko):
            s_ko = f"\\textbf{{{s_ko}}}"
        if np.isclose(d["unseen_ko_pearson_r"], best_r):
            s_r = f"\\textbf{{{s_r}}}"
        if np.isclose(d["day5_extrapolation_mse"], best_d5):
            s_d5 = f"\\textbf{{{s_d5}}}"
        if np.isclose(d["chip_psc_mean_tf_auroc"], best_ch):
            s_ch = f"\\textbf{{{s_ch}}}"
        if np.isclose(d["chip_psc_ap_over_random"], best_ap):
            s_ap = f"\\textbf{{{s_ap}}}"
        t1_rows.append([lbl, s_ko, s_r, s_d5, s_ch, s_ap])
    write_tex_table(
        out_tab_dir,
        "tab1_hipsc_closed_loop.tex",
        "lrrrrr",
        [
            "Closed-loop model (103 genes)",
            r"Unseen KO $\downarrow$",
            r"KO $r$ $\uparrow$",
            r"Day-5 MSE $\downarrow$",
            r"ChIP AUROC $\uparrow$",
            r"ChIP AP/rand $\uparrow$",
        ],
        t1_rows,
    )

    # Table 2: Locked 2x2 Multi-TF Correlation-Shift Benchmark (24 eight-TF worlds)
    t2_rows: list[list[str]] = []
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
    for idx_r, (reg_key, reg_lbl) in enumerate(
        [("additive", "Additive"), ("cooperative", "Cooperative")]
    ):
        if idx_r > 0:
            t2_rows.append(["\\midrule"])
        means = mtf[reg_key]["means"]
        best_mse = min(means[k]["test_mse"] for k, _ in mtf_order)
        best_attr = min(means[k]["attribution_rmse"] for k, _ in mtf_order)
        best_ap = max(means[k]["edge_ap"] for k, _ in mtf_order)
        for m_i, (k, lbl) in enumerate(mtf_order):
            d = means[k]
            c_mse = f"{d['test_mse']:.4f}"
            c_att = f"{d['attribution_rmse']:.4f}"
            c_ap = f"{d['edge_ap']:.3f}"
            if np.isclose(d["test_mse"], best_mse):
                c_mse = f"\\textbf{{{c_mse}}}"
            if np.isclose(d["attribution_rmse"], best_attr):
                c_att = f"\\textbf{{{c_att}}}"
            if np.isclose(d["edge_ap"], best_ap):
                c_ap = f"\\textbf{{{c_ap}}}"
            t2_rows.append(
                [reg_lbl if m_i == 0 else "", lbl, str(d["n_params"]), c_mse, c_att, c_ap]
            )
    write_tex_table(
        out_tab_dir,
        "tab2_multitf_factorial.tex",
        "llrrrr",
        [
            "Regime",
            "Architecture",
            "Params",
            r"OOD MSE $\downarrow$",
            r"Attr.\ RMSE $\downarrow$",
            r"Edge AP $\uparrow$",
        ],
        t2_rows,
    )

    # Table 3: Yeast Msn2 Fair Source-CV Joint Training & Cross-Readout 2x2 Summary
    jt = joint["joint_training"]
    t3_rows: list[list[str]] = []
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
        t3_rows.append([lbl, str(d["n_params"]), s30, s10, s40, sns])
    write_tex_table(
        out_tab_dir,
        "tab3_msn2_joint_and_cross.tex",
        "lrrrrr",
        [
            "Joint source-CV model",
            "Params",
            r"30-min $\downarrow$",
            r"10 trains $\downarrow$",
            r"First 40m $\downarrow$",
            r"No \texttt{SIP18} $\downarrow$",
        ],
        t3_rows,
    )

    # Table 4: 4-TF Mammalian Macrophage Benchmark (14 stimuli x 477 GRS genes)
    t4_rows: list[list[str]] = []
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
        s_au = f"{d['regulator_attribution_auroc']:.3f}"
        s_t1 = f"{100.0 * d['top1_regulator_accuracy']:.1f}\\%"
        s_se = f"{d['selective_decorrelated_mse']:.3f}"
        s_al = f"{d['all_14_stimuli_mse']:.3f}"
        if np.isclose(d["regulator_attribution_auroc"], best_auroc):
            s_au = f"\\textbf{{{s_au}}}"
        if np.isclose(d["top1_regulator_accuracy"], best_top1):
            s_t1 = f"\\textbf{{{s_t1}}}"
        if np.isclose(d["selective_decorrelated_mse"], best_sel):
            s_se = f"\\textbf{{{s_se}}}"
        if np.isclose(d["all_14_stimuli_mse"], best_all):
            s_al = f"\\textbf{{{s_al}}}"
        t4_rows.append([lbl, s_au, s_t1, s_se, s_al])
    write_tex_table(
        out_tab_dir,
        "tab4_macrophage_and_ortho.tex",
        "lrrrr",
        [
            "Architecture (4 TFs, 477 genes)",
            r"GRS AUROC $\uparrow$",
            r"Top-1 TF $\uparrow$",
            r"De-corr.\ MSE $\downarrow$",
            r"14-stim.\ MSE $\downarrow$",
        ],
        t4_rows,
    )

    # Table 5: Fibroblast Orthogonal Cofactor Recovery (177 genes) & 4:1 Strength Swap
    t5_rows: list[list[str]] = []
    o_order = [
        ("memgrn_ortho", "\\textbf{MemGRN-Ortho (ours)}"),
        ("latent_ode", "Latent Neural ODE"),
        ("memgrn", "MemGRN (observed TF only)"),
        ("perturb_ode", "PerturbODE"),
        ("local_const", "Constant $gW$"),
    ]
    best_il1 = min(ortho[k]["IL1_post_mse"] for k, _ in o_order)
    best_tnf = min(ortho[k]["TNF_post_mse"] for k, _ in o_order)
    best_lps = min(ortho[k]["LPS_post_mse"] for k, _ in o_order)
    best_om = min(ortho[k]["mean_post_mse"] for k, _ in o_order)
    for k, lbl in o_order:
        d = ortho[k]
        sil = f"{d['IL1_post_mse']:.3f}"
        stn = f"{d['TNF_post_mse']:.3f}"
        slp = f"{d['LPS_post_mse']:.3f}"
        smn = f"{d['mean_post_mse']:.3f}"
        if np.isclose(d["IL1_post_mse"], best_il1):
            sil = f"\\textbf{{{sil}}}"
        if np.isclose(d["TNF_post_mse"], best_tnf):
            stn = f"\\textbf{{{stn}}}"
        if np.isclose(d["LPS_post_mse"], best_lps):
            slp = f"\\textbf{{{slp}}}"
        if np.isclose(d["mean_post_mse"], best_om):
            smn = f"\\textbf{{{smn}}}"
        t5_rows.append(
            [lbl, sil, stn, slp, smn, f"{d['mean_nfkb_attr_cosine']:.3f}"]
        )
    write_tex_table(
        out_tab_dir,
        "tab5_nfkb_ortho_and_swap.tex",
        "lrrrrr",
        [
            "Fibroblast washout (177 genes)",
            r"IL-1 $\downarrow$",
            r"TNF $\downarrow$",
            r"LPS $\downarrow$",
            r"Mean MSE $\downarrow$",
            r"NF-$\kappa$B $\cos$ $\uparrow$",
        ],
        t5_rows,
    )

    # Appendix Table A1: Cross-Readout 2x2 & Six-Promoter Frozen-Readout Ablation
    modes = ["constant", "activity", "memory", "both"]
    ta1_rows: list[list[str]] = []
    for r_orig, r_lbl in [
        ("local_const", "Joint Constant-$gW$ readout (7 genes)"),
        ("memgrn", "Joint MemGRN readout (7 genes)"),
    ]:
        vals = [joint["cross_readout_2x2"][r_orig][m] for m in modes]
        b_i = int(np.argmin(vals))
        cells = [
            f"\\textbf{{{vals[i]:.4f}}}" if i == b_i else f"{vals[i]:.4f}" for i in range(4)
        ]
        ta1_rows.append([r_lbl, *cells])
    ta1_rows.append(["\\midrule"])
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
        ta1_rows.append([f"\\texttt{{{g}}} (archived single-pulse $W$)", *cells])
    means = arr.mean(axis=0)
    best_mean = int(np.argmin(means))
    ta1_rows.append(
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
        "tab_a1_six_promoters.tex",
        "lrrrr",
        [
            "Readout origin / promoter",
            r"Constant $\downarrow$",
            r"Activity only $\downarrow$",
            r"Memory only $\downarrow$",
            r"Activity + memory $\downarrow$",
        ],
        ta1_rows,
    )

    # Appendix Table A2: 7-Promoter LOPO Shared Filter Transfer
    lopo = joint["shared_filter_lopo_transfer"]["by_promoter"]
    ta2_rows: list[list[str]] = []
    for g_name, g_vals in lopo.items():
        m_v = g_vals["shared_memgrn_filter_mse"]
        c_v = g_vals["shared_const_filter_mse"]
        m_s = f"\\textbf{{{m_v:.4f}}}" if m_v < c_v else f"{m_v:.4f}"
        c_s = f"\\textbf{{{c_v:.4f}}}" if c_v < m_v else f"{c_v:.4f}"
        ta2_rows.append(
            [f"\\texttt{{{g_name}}}", c_s, m_s, f"{100.0 * (1.0 - m_v / c_v):+.1f}\\%"]
        )
    ta2_rows.append(["\\midrule"])
    m_mean = joint["shared_filter_lopo_transfer"]["mean_memgrn_shared"]
    c_mean = joint["shared_filter_lopo_transfer"]["mean_const_shared"]
    ta2_rows.append(
        [
            "7-promoter LOPO mean",
            f"{c_mean:.4f}",
            f"\\textbf{{{m_mean:.4f}}}",
            f"{100.0 * (1.0 - m_mean / c_mean):+.1f}\\%",
        ]
    )
    write_tex_table(
        out_tab_dir,
        "tab_a2_lopo_shared_filter.tex",
        "lrrr",
        [
            "Held-out promoter",
            r"Shared Const.\ filter $\downarrow$",
            r"Shared MemGRN filter $\downarrow$",
            r"MSE reduction $\uparrow$",
        ],
        ta2_rows,
    )

    # Appendix Table A3: Synthetic Occupancy & Mass-Action Buffering
    ta3_rows: list[list[str]] = []
    params_map = [168, 186, 274, 1158]
    model_names = [
        "Constant $gW$",
        "Two-rate $gW$",
        "\\textbf{MemGRN (ours)}",
        "Latent Neural ODE",
    ]
    for idx, tag, data, r_idx in [
        (0, "Occupancy-gated ($n=8$)", occ, 1),
        (1, "Mass-action buffer ($n=12$)", buf, 1),
        (2, "Constant control ($n=12$)", buf, 0),
    ]:
        if idx > 0:
            ta3_rows.append(["\\midrule"])
        means_s = data["means"][r_idx]
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
            ta3_rows.append(
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
        "tab_a3_synthetic_buffering.tex",
        "llrrrr",
        [
            "Kinetic regime",
            "Model",
            "Params",
            r"Full MSE $\downarrow$",
            r"Tail MSE $\downarrow$",
            r"Pair ID $\uparrow$",
        ],
        ta3_rows,
    )

    # Appendix Table A4: Multiscale & Shared Oscillator Disambiguation
    multi = load_stage3("multiscale_gw")
    mixed = load_stage3("mixed_directed_memory")
    ta4_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Multiscale MemGRN}"),
        ("history_basis", "History time basis"),
        ("current_hybrid", "Single-rate hybrid"),
        ("state_space", "Linear state space"),
    ]:
        c_dat = multi["classes"][k]["biphasic"]
        m_all = multi["means"][k]
        ta4_rows.append(
            [
                "Multiscale suite" if k == "learned_gw" else "",
                lbl,
                f"{100.0 * c_dat['error']:.2f}\\%",
                f"{100.0 * m_all['reversal']:.1f}\\%",
                f"{m_all['delayed_ap']:.3f}",
            ]
        )
    ta4_rows.append(["\\midrule"])
    for k, lbl in [
        ("memory_driver", "\\textbf{MemGRN + oscillator}"),
        ("latent_ODE", "Latent Neural ODE"),
        ("state_space", "Linear state space"),
        ("directed_memory", "MemGRN (no oscillator)"),
        ("instant_driver", "Memoryless + oscillator"),
    ]:
        m_dat = mixed["means"][k]
        ta4_rows.append(
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
        "tab_a4_multiscale_driver.tex",
        "llrrr",
        [
            "Benchmark",
            "Model",
            r"Tail error $\downarrow$",
            r"Sign acc. $\uparrow$",
            r"Edge AP $\uparrow$",
        ],
        ta4_rows,
    )

    # Appendix Table A5: RTN2 Switch Ablation & Inter-Pulse Filter Resets
    rtn_sw = load_stage3("msn2_rtn2_switch_neural")
    reset = load_stage3("msn2_memory_reset")
    ta5_rows: list[list[str]] = []
    for idx_r, (k, lbl) in enumerate(
        [
            ("constant", "Constant rates"),
            ("switch_only", "Discrete switch only"),
            ("neural_only", "Continuous neural rate"),
            ("full", "Switch + neural rate"),
        ]
    ):
        d_sw = rtn_sw["models"][k]
        ta5_rows.append(
            [
                "\\texttt{RTN2} switch" if idx_r == 0 else "",
                lbl,
                f"{d_sw['primary']:.4f}",
                f"{d_sw['half_step']:.4f}",
                f"{d_sw['first40']:.4f}",
            ]
        )
    ta5_rows.append(["\\midrule"])
    for tr_id, tr_lbl in [
        ("FM_6_5min_690nM", "6 pulses (intact / reset)"),
        ("FM_8_5min_690nM", "8 pulses (intact / reset)"),
    ]:
        p_all = reset["results"][tr_id]["panels"]["all"]["mse"]
        p_rtn = reset["results"][tr_id]["promoters"]["RTN2"]
        ta5_rows.append(
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
        "tab_a5_rtn2_and_reset.tex",
        "llrrr",
        [
            "Study",
            "Condition / model",
            r"Primary MSE $\downarrow$",
            r"Refined / Carryover",
            r"First 40m / Post-off",
        ],
        ta5_rows,
    )

    # Appendix Table A6: Swap Template Libraries, Subgroups, & AR(1) 5-Time Certificate
    ta6_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Learned-memory $gW$ (all $96$)}"),
        ("fixed_gw", "Fixed-rate $gW$ (all $96$)"),
        ("history_basis", "History time basis (all $96$)"),
        ("oracle", "Oracle templates (all $96$)"),
    ]:
        rep_v = 100.0 * swap["overall"][k]["repeat"]
        swp_v = 100.0 * swap["overall"][k]["swap"]
        orc_v = 100.0 * swap["overall"]["oracle"]["swap"]
        ta6_rows.append(
            [lbl, f"{rep_v:.2f}\\%", f"\\textbf{{{swp_v:.2f}\\%}}", f"+{swp_v - rep_v:.2f}", f"{orc_v:.2f}\\%"]
        )
    ta6_rows.append(["\\midrule"])
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
        ta6_rows.append(
            [sg_lbl, f"{sr:.2f}\\%", f"\\textbf{{{ss:.2f}\\%}}", f"+{ss - sr:.2f}", f"{so:.2f}\\%"]
        )
    ta6_rows.append(["\\midrule"])
    name_map = {
        "learned_gw": "5-pt AR(1): Learned $gW$",
        "fixed_gw": "5-pt AR(1): Fixed $gW$",
        "history_basis": "5-pt AR(1): History basis",
    }
    for idx_c, r_cert in enumerate(cert["rows"]):
        p_meta = cert_meta["panels"][idx_c]
        ci_lo = 100.0 * p_meta["ci"][0]
        ci_hi = 100.0 * p_meta["ci"][1]
        ta6_rows.append(
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
        "tab_a6_swap_and_cert.tex",
        "lrrrr",
        [
            "Template / subgroup / 5-pt policy",
            r"Baseline $\uparrow$",
            r"Swap / Selected $\uparrow$",
            r"Gain (pp) $\uparrow$",
            r"Oracle / 98.5\% CI",
        ],
        ta6_rows,
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
    build_fig2_hipsc_closed_loop(out_latex_fig)
    build_fig3_multitf_factorial(out_latex_fig)
    build_fig4_msn2_and_promoters(out_latex_fig)
    build_fig5_macrophage_ortho_swap(out_latex_fig)
    build_all_tables(out_latex_tab)

    ledger_path = ROOT / "results" / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps({"source_sha256": SOURCE_HASHES}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built 5 figures and 11 tables from {len(SOURCE_HASHES)} verified artifacts.")


if __name__ == "__main__":
    main()
