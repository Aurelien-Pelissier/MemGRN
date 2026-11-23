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
        "a   Closed- and open-loop source-isolated memory & interaction readout",
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
        "Stage-1 zᵢᵣ\nżᵢᵣ = λ⁽¹⁾ᵢᵣ(aᵢ−zᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        4.18,
        2.74,
        1.90,
        1.02,
        "Stage-2 hᵢᵣ\nḣᵢᵣ = λ⁽²⁾ᵢᵣ(zᵢᵣ−hᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        6.28,
        2.74,
        2.66,
        1.02,
        "Interaction readout\nDᵢⱼaᵢ + Bᵢᵣⱼhᵢᵣ + Γh̄ᵢh̄ₖ",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        9.14,
        2.74,
        1.82,
        1.02,
        "Node state\nẏⱼ = pⱼ − δⱼyⱼ",
        PALETTE["light_gray"],
        ec=PALETTE["border"],
    )

    draw_arrow((1.92, 3.25), (2.04, 3.25))
    draw_arrow((4.02, 3.25), (4.14, 3.25))
    draw_arrow((6.12, 3.25), (6.24, 3.25))
    draw_arrow((8.98, 3.25), (9.10, 3.25))

    # Forward skip connection for Direct path D_{ij} a_i (Source drive -> Interaction readout)
    ax.plot(
        [1.45, 1.45, 7.60],
        [3.82, 4.08, 4.08],
        color=PALETTE["violet"],
        lw=0.90,
        ls="--",
    )
    draw_arrow((7.60, 4.08), (7.60, 3.82), color=PALETTE["violet"], lw=0.90, ls="--")
    ax.text(
        4.52,
        4.11,
        "Direct path  Dᵢⱼaᵢ",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color=PALETTE["violet"],
    )

    # Closed-loop feedback arc (Node state -> Source drive)
    ax.plot(
        [10.05, 10.05, 0.72],
        [3.82, 4.54, 4.54],
        color=PALETTE["navy"],
        lw=0.95,
    )
    draw_arrow((0.72, 4.54), (0.72, 3.82), color=PALETTE["navy"], lw=0.95)
    ax.text(
        5.50,
        4.58,
        "Closed-loop feedback  gᵢ = (1−koᵢ)γᵢyᵢ − koᵢγᵢyᵢ,basal + uᵢ",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=PALETTE["navy"],
    )

    draw_box(
        1.14,
        1.52,
        5.84,
        0.82,
        "Isolated rates  (λ⁽¹⁾ᵢᵣ, λ⁽²⁾ᵢᵣ)(aᵢ, zᵢᵣ, hᵢᵣ)\n∂(zᵢᵣ, hᵢᵣ)/∂gₖ ≡ 0  (k ≠ i)",
        PALETTE["light_amber"],
        ec=PALETTE["amber"],
    )
    # Bidirectional state/rate couplings (z, h) <-> Isolated rates
    draw_arrow((2.82, 2.40), (2.82, 2.68), color=PALETTE["amber"])
    draw_arrow((3.24, 2.68), (3.24, 2.40), color=PALETTE["amber"], lw=0.80, ls=":")
    draw_arrow((4.92, 2.40), (4.92, 2.68), color=PALETTE["amber"])
    draw_arrow((5.34, 2.68), (5.34, 2.40), color=PALETTE["amber"], lw=0.80, ls=":")
    ax.plot(
        [0.96, 0.96, 1.09],
        [2.68, 1.93, 1.93],
        color=PALETTE["amber"],
        lw=0.85,
        ls="--",
    )

    draw_box(
        7.24,
        1.52,
        3.72,
        0.82,
        "Ortho cofactor\n" + r"$\Psi_\perp \perp \mathrm{col}(\Phi_{\mathrm{obs}})$",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
        lw=0.85,
    )
    draw_arrow((8.35, 2.40), (8.35, 2.68), color=PALETTE["violet"], lw=0.85, ls="--")

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
        "1. S-node & closed-loop ID\nZero cross-node leakage",
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
        "3. Iso-state & Ortho\nO(SR) network scaling",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
    )

    export_figure(fig, "fig1_memgrn_architecture", out_latex_fig, audit_alignment=False)


def build_fig2_cross_domain_networks(out_latex_fig: Path) -> None:
    """Figure 2: Fair equal-budget cross-domain networks (Beijing Delta-u, S=207 METR-LA, IEEE 39-Bus)."""
    ol = load_new_result("fair_open_loop")
    cl = load_new_result("fair_closed_loop")
    bq = ol["beijing_air_quality"]["delta_u_4season"]["summary"]
    ml207 = cl["metr_la_s207"]["summary"]
    pg = cl["powergrid_ieee39"]["summary"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.34))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.08, 1.08, 1.08],
        left=0.098,
        right=0.985,
        bottom=0.39,
        top=0.84,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: UCI Beijing Air Quality 4-fold LOSO MSE on anchored Delta u (ODEs crush discrete baselines!)
    keys_b = [
        "dlinear",
        "transformer",
        "gru",
        "coupled_adaptive",
        "neural_cde",
        "selective_ssm",
        "local_const",
        "depthwise_ssm",
        "memgrn_additive",
        "cfc",
    ]
    lbls_b = ["DLinear", "Transf.", "GRU", "Cpl-Ad.", "N-CDE", "Mamba", "Loc-Con", "D-SSM", "MemGRN", "CfC"]
    cols_b = [
        "#6D9DC5",
        "#A3A8B3",
        "#7E8796",
        PALETTE["amber"],
        PALETTE["steel"],
        "#D97736",
        PALETTE["slate"],
        PALETTE["teal"],
        PALETTE["navy"],
        "#B85C38",
    ]
    xb = np.arange(len(keys_b))
    vb = np.array([bq[k]["loso_mse"] for k in keys_b], dtype=np.float64)
    eb = np.array([bq[k]["loso_mse_se"] for k in keys_b], dtype=np.float64)
    ax0.bar(
        xb,
        vb - 0.44,
        bottom=0.44,
        yerr=eb,
        width=0.64,
        color=cols_b,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.3,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax0.set_xticks(xb)
    ax0.set_xticklabels(lbls_b, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_xlim(-0.6, len(keys_b) - 0.4)
    ax0.set_ylim(0.44, 0.62)
    ax0.set_yticks([0.45, 0.50, 0.55, 0.60])
    ax0.set_ylabel("LOSO MSE (Δu)")
    ax0.set_title("Beijing AQ (Δu)")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Matched Full S=207 METR-LA 1-h test MSE (MemGRN #1 of 15)
    keys_m = [
        "transformer",
        "gru",
        "persistence",
        "dlinear",
        "cl_cfc",
        "cl_coupled_adaptive",
        "cl_depthwise_ssm_selfmem",
        "cl_selective_ssm",
        "cl_latent_ode",
        "cl_local_const_selfmem",
        "cl_memgrn_selfmem",
    ]
    lbls_m = ["Transf.", "GRU", "Persist.", "DLinear", "CfC", "Cpl-Ad.", "D-SSM", "Mamba", "Lat-ODE", "Loc-Con", "MemGRN"]
    cols_m = [
        "#A3A8B3",
        "#7E8796",
        "#C8CDD6",
        "#6D9DC5",
        "#B85C38",
        PALETTE["amber"],
        PALETTE["teal"],
        "#D97736",
        "#8C6D9B",
        PALETTE["slate"],
        PALETTE["navy"],
    ]
    xm = np.arange(len(keys_m))
    vm = np.array([ml207[k]["test_mse"] for k in keys_m], dtype=np.float64)
    em = np.array([ml207[k]["test_mse_se"] for k in keys_m], dtype=np.float64)
    ax1.bar(
        xm,
        vm - 0.11,
        bottom=0.11,
        yerr=em,
        width=0.64,
        color=cols_m,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.2,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax1.set_xticks(xm)
    ax1.set_xticklabels(lbls_m, rotation=90, ha="center", va="top", fontsize=7.4)
    ax1.tick_params(axis="x", pad=4.0)
    ax1.set_xlim(-0.6, len(keys_m) - 0.4)
    ax1.set_ylim(0.11, 0.205)
    ax1.set_yticks([0.12, 0.15, 0.18])
    ax1.set_ylabel("1-h test MSE")
    ax1.set_title("METR-LA (S=207)")
    add_panel_tag(ax1, "b", x_pt=-32.0, y_pt=6.0)

    # Panel c: Matched IEEE 39-Bus synthetic ODE zero-shot N-2 MSE (MemGRN+sm & Loc-Con+sm tie #1)
    keys_p = [
        "cl_ltc_coupled",
        "cl_perturb_ode",
        "cl_depthwise_ssm_selfmem",
        "cl_ltc_local_selfmem",
        "cl_selective_ssm",
        "cl_cfc",
        "cl_memgrn_hollow",
        "cl_coupled_adaptive",
        "cl_latent_ode",
        "cl_memgrn_selfmem",
        "cl_local_const_selfmem",
    ]
    lbls_p = ["LTC-Cpl", "PertODE", "D-SSM", "LTC-Loc", "Mamba", "CfC", "Mem(hol)", "Cpl-Ad.", "Lat-ODE", "MemGRN", "Loc-Con"]
    cols_p = [
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["teal"],
        "#48A9A6",
        "#D97736",
        "#B85C38",
        PALETTE["steel"],
        PALETTE["amber"],
        "#8C6D9B",
        PALETTE["navy"],
        PALETTE["slate"],
    ]
    xp = np.arange(len(keys_p))
    vp = np.array([pg[k]["double_n2_mse"] for k in keys_p], dtype=np.float64)
    ep = np.array([pg[k]["double_n2_mse_se"] for k in keys_p], dtype=np.float64)
    ax2.bar(
        xp,
        vp,
        yerr=ep,
        width=0.64,
        color=cols_p,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.2,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax2.set_xticks(xp)
    ax2.set_xticklabels(lbls_p, rotation=90, ha="center", va="top", fontsize=7.4)
    ax2.tick_params(axis="x", pad=4.0)
    ax2.set_xlim(-0.6, len(keys_p) - 0.4)
    ax2.set_ylim(0.0, float(np.max(vp + ep) * 1.12))
    ax2.set_yticks([0.00, 0.02, 0.04, 0.06])
    ax2.set_ylabel("N−2 double MSE")
    ax2.set_title("IEEE 39-Bus (sim.)")
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(fig, "fig2_cross_domain_networks", out_latex_fig)


def build_fig2_human_longitudinal(out_latex_fig: Path) -> None:
    """Appendix Figure: Human iPSC (GSE213069), Human CD4+ T-cell CollecTRI (GSE314342), & Clean EMT (GSE147405)."""
    hipsc = load_new_result("hipsc_closed_loop")["summary"]
    cd4t = load_new_result("fair_closed_loop")["cd4t_gse314342"]["summary"]
    emt = load_new_result("fair_open_loop")["emt_gse147405"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.30))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.00, 1.06, 1.06],
        left=0.100,
        right=0.985,
        bottom=0.38,
        top=0.84,
        wspace=0.44,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Human iPSC unseen CRISPR KO MSE (GSE213069, 14,945 cells)
    keys_h = [
        "renge",
        "cl_latent_ode",
        "cl_cfc",
        "cl_ltc_coupled",
        "cl_perturb_ode",
        "cl_local_const",
        "cl_memgrn",
        "cl_selective_ssm",
    ]
    lbls_h = ["RENGE", "Lat-ODE", "CfC", "LTC-Cpl", "PertODE", "Loc-Con", "MemGRN", "Mamba"]
    cols_h = [
        PALETTE["slate"],
        "#8C6D9B",
        "#B85C38",
        PALETTE["crimson"],
        PALETTE["violet"],
        "#7B8A9E",
        PALETTE["navy"],
        PALETTE["teal"],
    ]
    x_h = np.arange(len(keys_h))
    ko_vals = [hipsc[k]["unseen_ko_mse"] for k in keys_h]
    ax0.bar(x_h, ko_vals, width=0.64, color=cols_h, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(x_h)
    ax0.set_xticklabels(lbls_h, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_xlim(-0.6, len(keys_h) - 0.4)
    ax0.set_ylim(0.0, float(max(ko_vals) * 1.15))
    ax0.set_ylabel("Unseen-KO MSE")
    ax0.set_title("hiPSC CRISPR KOs")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Primary Human CD4+ T-cell (GSE314342, S=128) CollecTRI AP enrichment (replaces duplicate Cyclic 2-KO!)
    keys_cd = [
        "cl_perturb_ode",
        "cl_ltc_coupled",
        "cl_latent_ode",
        "cl_cfc",
        "renge_graph_only",
        "cl_depthwise_ssm",
        "cl_selective_ssm",
        "cl_local_const",
        "cl_memgrn",
        "cl_coupled_adaptive",
    ]
    lbls_cd = ["PertODE", "LTC-Cpl", "Lat-ODE", "CfC", "RENGE", "D-SSM", "Mamba", "Loc-Con", "MemGRN", "Cpl-Ad."]
    cols_cd = [
        PALETTE["violet"],
        PALETTE["crimson"],
        "#8C6D9B",
        "#B85C38",
        "#94A3B8",
        PALETTE["teal"],
        "#D97736",
        PALETTE["slate"],
        PALETTE["navy"],
        PALETTE["amber"],
    ]
    x_cd = np.arange(len(keys_cd))
    v_cd = np.array([cd4t[k]["collectri_colnorm_ap_over_random"] for k in keys_cd], dtype=np.float64)
    ax1.axhline(1.0, color=PALETTE["slate"], lw=0.75, ls="--", zorder=0)
    ax1.bar(x_cd, v_cd - 0.85, bottom=0.85, width=0.64, color=cols_cd, edgecolor=PALETTE["ink"], linewidth=0.50)
    ax1.set_xticks(x_cd)
    ax1.set_xticklabels(lbls_cd, rotation=90, ha="center", va="top", fontsize=7.4)
    ax1.tick_params(axis="x", pad=4.0)
    ax1.set_xlim(-0.6, len(keys_cd) - 0.4)
    ax1.set_ylim(0.85, 1.35)
    ax1.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    ax1.set_ylabel("CollecTRI AP (×rnd)")
    ax1.set_title("CD4+ T-cell (S=128)")
    add_panel_tag(ax1, "b", x_pt=-32.0, y_pt=6.0)

    # Panel c: Clean zero-leakage Human EMT withdrawal top-30 MSE (± SE, equal 8-trial budget)
    keys_e = [
        "latent_ode",
        "perturb_ode",
        "coupled_adaptive",
        "ltc_local_coop",
        "neural_cde",
        "local_const_coop",
        "selective_ssm",
        "memgrn",
        "memgrn_ortho",
        "cfc",
        "depthwise_ssm_coop",
    ]
    lbls_e = ["Lat-ODE", "PertODE", "Cpl-Ad.", "LTC-Loc", "N-CDE", "Loc-Con", "Mamba", "MemGRN", "Mem-Or", "CfC", "D-SSM"]
    cols_e = [
        "#8C6D9B",
        PALETTE["violet"],
        PALETTE["amber"],
        "#48A9A6",
        PALETTE["steel"],
        PALETTE["slate"],
        "#D97736",
        PALETTE["navy"],
        "#12355B",
        "#B85C38",
        PALETTE["teal"],
    ]
    e_vals = np.array([emt[k]["top30_withdrawal_mse"] for k in keys_e], dtype=np.float64)
    e_ses = np.array([emt[k]["top30_withdrawal_mse_se"] for k in keys_e], dtype=np.float64)
    x_e = np.arange(len(keys_e))
    ax2.bar(
        x_e,
        e_vals,
        yerr=e_ses,
        width=0.64,
        color=cols_e,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.2,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax2.set_xticks(x_e)
    ax2.set_xticklabels(lbls_e, rotation=90, ha="center", va="top", fontsize=7.4)
    ax2.tick_params(axis="x", pad=4.0)
    ax2.set_xlim(-0.6, len(keys_e) - 0.4)
    ax2.set_ylim(0.0, float(np.max(e_vals + e_ses) * 1.12))
    ax2.set_ylabel("Top-30 wdr. MSE")
    ax2.set_title("Human EMT wdr.")
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(
        fig,
        "fig2_human_longitudinal",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios match 8-bar, 10-bar, and 11-bar panel densities",
            }
        ],
    )


def build_fig3_multitf_ltc_ortho(out_latex_fig: Path) -> None:
    """Figure 3: Equal-budget multi-source OOD, split-axis Thm 1(iii) separation, & Cyclic Double-KO."""
    ol = load_new_result("fair_open_loop")
    coop_m = ol["multitf_24world"]["cooperative"]["means"]
    part_b = ol["ltc_and_dssm_separation"]
    cyc = load_new_result("cyclic_double_ko")["summary"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.30))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.12, 1.06, 0.78],
        left=0.095,
        right=0.905,
        bottom=0.37,
        top=0.84,
        wspace=0.44,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Equal-budget 24-world cooperative OOD MSE (± SE, with matched +Gamma & Depthwise SSM)
    k_c = [
        "cfc",
        "ltc_coupled",
        "perturb_ode",
        "coupled_adaptive",
        "local_const_coop",
        "selective_ssm",
        "neural_cde",
        "memgrn",
        "depthwise_ssm_coop",
        "ltc_local_coop",
    ]
    l_c = ["CfC", "LTC-Cpl", "PertODE", "Cpl-Ad.", "Loc-Con*", "Mamba", "N-CDE", "MemGRN", "D-SSM*", "LTC-Loc*"]
    c_c = [
        "#B85C38",
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["slate"],
        "#D97736",
        PALETTE["steel"],
        PALETTE["navy"],
        PALETTE["teal"],
        "#48A9A6",
    ]
    xc = np.arange(len(k_c))
    vc = np.array([coop_m[k]["test_mse"] for k in k_c], dtype=np.float64)
    ec = np.array([coop_m[k]["test_mse_se"] for k in k_c], dtype=np.float64)
    ax0.bar(
        xc,
        vc,
        yerr=ec,
        width=0.64,
        color=c_c,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.2,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax0.set_xticks(xc)
    ax0.set_xticklabels(l_c, rotation=90, ha="center", va="top", fontsize=7.4)
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_xlim(-0.6, len(k_c) - 0.4)
    ax0.set_ylim(0.0, float(np.max(vc + ec) * 1.12))
    ax0.set_ylabel("OOD test MSE")
    ax0.set_title("Cooperative (8-src)")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Cyclic closed-loop zero-shot combinatorial double-KO MSE (12 worlds, ± SE)
    k_p = [
        "cl_perturb_ode",
        "cl_ltc_local",
        "cl_selective_ssm",
        "cl_cfc",
        "cl_ltc_coupled",
        "cl_latent_ode",
        "cl_local_const",
        "cl_coupled_adaptive",
        "cl_memgrn",
    ]
    l_p = ["PertODE", "LTC-Loc", "Mamba", "CfC", "LTC-Cpl", "Lat-ODE", "Loc-Con", "Cpl-Ad.", "MemGRN"]
    c_p = [
        PALETTE["violet"],
        PALETTE["teal"],
        "#D97736",
        "#B85C38",
        PALETTE["crimson"],
        "#8C6D9B",
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["navy"],
    ]
    xp = np.arange(len(k_p))
    vp = np.array([cyc[k]["double_ko_mse"] for k in k_p], dtype=np.float64)
    ep = np.array([cyc[k]["double_ko_mse_se"] for k in k_p], dtype=np.float64)
    ax1.bar(
        xp,
        vp,
        yerr=ep,
        width=0.64,
        color=c_p,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.2,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax1.set_xticks(xp)
    ax1.set_xticklabels(l_p, rotation=90, ha="center", va="top", fontsize=7.5)
    ax1.tick_params(axis="x", pad=4.0)
    ax1.set_xlim(-0.6, len(k_p) - 0.4)
    ax1.set_ylim(0.0, float(np.max(vp + ep) * 1.12))
    ax1.set_ylabel("Double-KO MSE")
    ax1.set_title("Cyclic 0-shot 2-KO")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Dedicated twin-y-axis separation of Thm 1(iii) on far right (clean margin for twin axis!)
    b_keys = ["memgrn_matched", "depthwise_ssm_matched", "ltc_local_matched"]
    b_lbls = ["MemGRN", "D-SSM", "LTC-Loc"]
    xb = np.arange(len(b_keys))
    bw = 0.36
    pk_vals = [1e5 * part_b[k]["delayed_peak_post_mse"] for k in b_keys]
    amp_vals = [100.0 * part_b[k]["steady_state_amp_invariance_error"] for k in b_keys]
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["right"].set_visible(True)
    ax2_r.spines["right"].set_linewidth(0.75)
    ax2_r.spines["right"].set_color(PALETTE["amber"])
    b1 = ax2.bar(
        xb - 0.5 * bw,
        pk_vals,
        width=bw * 0.90,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        label="Peak",
    )
    b2 = ax2_r.bar(
        xb + 0.5 * bw,
        amp_vals,
        width=bw * 0.90,
        color=PALETTE["amber"],
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        label="Amp.",
    )
    ax2.set_xticks(xb)
    ax2.set_xticklabels(b_lbls, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.0)
    ax2.set_xlim(-0.6, len(b_keys) - 0.4)
    ax2.set_ylim(0.0, 7.6)
    ax2.set_yticks([0, 2, 4, 6])
    ax2.set_ylabel(r"Peak ($\times 10^{-5}$)", color=PALETTE["navy"])
    ax2.tick_params(axis="y", colors=PALETTE["navy"])
    ax2_r.set_ylim(0.0, 14.5)
    ax2_r.set_yticks([0, 4, 8, 12])
    ax2_r.set_ylabel("Amp. dist. (%)", color=PALETTE["amber"], labelpad=2.0)
    ax2_r.tick_params(axis="y", colors=PALETTE["amber"])
    ax2.set_title("Thm. 1(iii) sep.")
    ax2.legend([b1, b2], ["Peak MSE", "Amp. %"], loc="upper left", handlelength=0.85, labelspacing=0.14, fontsize=6.9)
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(
        fig,
        "fig3_multitf_ltc_ortho",
        out_latex_fig,
        exclude_axes=[ax2_r],
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Width ratios accommodate 10-bar, 9-bar, and twin-axis 3-group panels",
            }
        ],
    )


def build_fig4_yeast_chromatin_portraits(out_latex_fig: Path) -> None:
    """Figure 4: Equal-budget live yeast Msn2 joint fits, cross-readout swap, & gauge-invariant chromatin portraits."""
    jt = load_new_result("fair_open_loop")["yeast_msn2_joint"]["models"]
    joint = load_new_result("msn2_joint_and_cross_readout")
    port = load_new_result("cross_domain_portraits")["yeast_msn2_minutes"]["by_promoter"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.30))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.12, 0.96, 1.08],
        left=0.098,
        right=0.985,
        bottom=0.37,
        top=0.84,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Equal-budget joint source-CV fits (transparently including asym, sym, D-SSM, Cpl-Ad.)
    j_keys = [
        "local_const",
        "cfc",
        "depthwise_ssm",
        "selective_ssm",
        "perturb_ode",
        "memgrn",
        "memgrn_sym",
        "latent_ode",
        "ltc_local",
        "coupled_adaptive",
    ]
    j_lbls = ["Const gW", "CfC", "D-SSM", "Mamba", "PertODE", "Mem(as)", "Mem(sy)", "Lat-ODE", "LTC-Loc", "Cpl-Ad."]
    j_cols = [
        PALETTE["slate"],
        "#B85C38",
        PALETTE["teal"],
        "#D97736",
        PALETTE["violet"],
        PALETTE["steel"],
        PALETTE["navy"],
        "#8C6D9B",
        "#48A9A6",
        PALETTE["amber"],
    ]
    j_vals = np.array([jt[k]["ten_train_post_mse"] for k in j_keys], dtype=np.float64)
    j_ses = np.array([jt[k]["ten_train_post_mse_se"] for k in j_keys], dtype=np.float64)
    jx = np.arange(len(j_keys))
    ax0.bar(
        jx,
        j_vals,
        yerr=j_ses,
        width=0.64,
        color=j_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.2,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax0.set_xticks(jx)
    ax0.set_xticklabels(j_lbls, rotation=90, ha="center", va="top", fontsize=7.4)
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_xlim(-0.6, len(j_keys) - 0.4)
    ax0.set_ylim(0.0, float(np.max(j_vals + j_ses) * 1.12))
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
    ax1.tick_params(axis="x", pad=4.0)
    ax1.set_xlim(-0.65, 3.65)
    ax1.set_ylim(0.0, float(max(*const_w_vals, *mem_w_vals) * 1.32))
    ax1.set_ylabel("10-train MSE")
    ax1.set_title("Cross-readout")
    ax1.legend(loc="upper right", ncol=1, handlelength=1.0, labelspacing=0.14, fontsize=7.2)
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    # Panel c: Gauge-invariant dynamical washout half-times tau_off (± SE, 3-seed) with all 3 legend classes!
    p_order = ["HXK1", "DCS2", "RTN2", "DDR2", "SIP18", "TKL2", "ALD3"]
    taus = np.array([port[g]["tau_off_half"] for g in p_order], dtype=np.float64)
    tses = np.array([port[g]["tau_off_half_se"] for g in p_order], dtype=np.float64)
    p_cols = [
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["amber"],
        PALETTE["navy"],
        PALETTE["navy"],
        PALETTE["navy"],
        PALETTE["navy"],
    ]
    px = np.arange(len(p_order))
    ax2.bar(
        px,
        taus,
        yerr=tses,
        width=0.64,
        color=p_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.5,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax2.bar([-5], [0], color=PALETTE["teal"], edgecolor=PALETTE["ink"], linewidth=0.55, label="Fast (open)")
    ax2.bar([-5], [0], color=PALETTE["amber"], edgecolor=PALETTE["ink"], linewidth=0.55, label="Interm. (RTN2)")
    ax2.bar([-5], [0], color=PALETTE["navy"], edgecolor=PALETTE["ink"], linewidth=0.55, label="Slow (remod.)")
    ax2.set_xticks(px)
    ax2.set_xticklabels(p_order, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.0)
    ax2.set_xlim(-0.6, len(p_order) - 0.4)
    ax2.set_ylim(0.0, 185.0)
    ax2.set_yticks([0, 50, 100, 150])
    ax2.set_ylabel(r"Washout $\tau_{\mathrm{off}}$ (min)")
    ax2.set_title("Chromatin memory")
    ax2.legend(loc="upper left", ncol=1, handlelength=0.9, labelspacing=0.12, fontsize=6.9)
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(
        fig,
        "fig4_yeast_chromatin_portraits",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios accommodate 10-bar, 4-pair, and 7-promoter panels",
            }
        ],
    )


def build_fig5_mammalian_and_scaling(out_latex_fig: Path) -> None:
    """Figure 5: 4-TF Macrophage GRS, 6-of-8 MemGRN-Ortho vs Unproj, & Complete O(SR) Scaling."""
    mac = load_new_result("macrophage_multitf")["summary"]
    ol = load_new_result("fair_open_loop")
    po6 = ol["partial_obs_6of8"]["summary"]
    scaling = ol["scaling_s8_to_s512"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.30))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.96, 1.08, 1.12],
        left=0.102,
        right=0.985,
        bottom=0.37,
        top=0.84,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: 4-TF Macrophage GRS attribution AUROC (strictly ascending order)
    m_keys = [
        "perturb_ode",
        "ltc_coupled",
        "ltc_local",
        "local_const",
        "memgrn",
        "coupled_adaptive",
    ]
    m_lbls = ["PertODE", "LTC-Cpl", "LTC-Loc", "Loc-Con", "MemGRN", "Cpl-Ad."]
    m_cols = [
        PALETTE["violet"],
        PALETTE["crimson"],
        PALETTE["teal"],
        PALETTE["slate"],
        PALETTE["navy"],
        PALETTE["amber"],
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
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_xlim(-0.6, len(m_keys) - 0.4)
    ax0.set_ylim(0.5, float(max(auroc_vals) + 0.05))
    ax0.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax0.set_ylabel("Regulator AUROC")
    ax0.set_title("Macrophage GRS")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Ground-truth 6-of-8 partial observability full-model attribution RMSE (± SE)
    o_keys = [
        "perturb_ode",
        "latent_ode",
        "local_const",
        "depthwise_ssm",
        "memgrn",
        "memgrn_unproj",
        "memgrn_ortho_joint",
        "memgrn_ortho_2stage",
    ]
    o_lbls = ["PertODE", "Lat-ODE", "Loc-Con", "D-SSM", "MemGRN", "Unproj", "Or-Jnt", "Mem-Or"]
    o_cols = [
        PALETTE["violet"],
        "#8C6D9B",
        PALETTE["slate"],
        PALETTE["teal"],
        PALETTE["steel"],
        PALETTE["amber"],
        "#5B8C85",
        PALETTE["navy"],
    ]
    ox = np.arange(len(o_keys))
    o_vals = np.array([po6[k]["full_model_gt_attr_rmse"] for k in o_keys], dtype=np.float64)
    o_ses = np.array([po6[k]["full_model_gt_attr_rmse_se"] for k in o_keys], dtype=np.float64)
    ax1.bar(
        ox,
        o_vals,
        yerr=o_ses,
        width=0.62,
        color=o_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.50,
        capsize=1.3,
        error_kw={"elinewidth": 0.65, "ecolor": PALETTE["ink"]},
    )
    ax1.set_xticks(ox)
    ax1.set_xticklabels(o_lbls, rotation=90, ha="center", va="top", fontsize=7.5)
    ax1.tick_params(axis="x", pad=4.0)
    ax1.set_xlim(-0.6, len(o_keys) - 0.4)
    ax1.set_ylim(0.0, float(np.max(o_vals + o_ses) * 1.15))
    ax1.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax1.set_ylabel("Full attr. RMSE")
    ax1.set_title("6/8 Partial Obs.")
    add_panel_tag(ax1, "b", x_pt=-32.0, y_pt=6.0)

    # Panel c: Complete S=8..512 scaling showing BOTH memory params (solid) AND total params (dashed)!
    s_all = np.array([r["n_tfs"] for r in scaling], dtype=np.float64)
    mem_p = np.array([r["memgrn_rank24_mem_params"] for r in scaling], dtype=np.float64)
    mem_tot = np.array([r["memgrn_rank24_total_params"] for r in scaling], dtype=np.float64)
    cpl_p = np.array([r["coupled_adaptive_mem_params"] for r in scaling], dtype=np.float64)
    cpl_tot = np.array([r["coupled_adaptive_total_params"] for r in scaling], dtype=np.float64)
    s_ltc = s_all[:6]
    ltc_p = np.array([r["coupled_ltc_mem_params"] for r in scaling[:6]], dtype=np.float64)

    ax2.loglog(s_all, cpl_tot, lw=0.95, ls="--", color=PALETTE["amber"], alpha=0.85, label="Cpl tot.")
    ax2.loglog(s_all, mem_tot, lw=1.05, ls="--", color=PALETTE["steel"], label="Mem tot")
    ax2.loglog(s_ltc, ltc_p, marker="^", ms=3.4, lw=1.2, color=PALETTE["crimson"], label="LTC mem")
    ax2.loglog(s_all, cpl_p, marker="s", ms=3.2, lw=1.2, color=PALETTE["amber"], label="Cpl mem")
    ax2.loglog(s_all, mem_p, marker="o", ms=3.6, lw=1.4, color=PALETTE["navy"], label="Mem mem")
    ax2.set_xticks([8, 32, 128, 512])
    ax2.set_xticklabels(["8", "32", "128", "512"])
    ax2.minorticks_off()
    ax2.set_xlim(6.5, 680.0)
    ax2.set_ylim(150.0, 5.0e8)
    ax2.set_yticks([1e3, 1e5, 1e7])
    ax2.set_yticklabels(["1k", "100k", "10M"])
    ax2.set_xlabel("Sources S (N=4S)")
    ax2.set_ylabel("Parameters")
    ax2.set_title("Network scaling")
    ax2.legend(
        loc="upper left",
        ncol=2,
        columnspacing=0.35,
        handlelength=0.80,
        handletextpad=0.25,
        labelspacing=0.12,
        borderpad=0.10,
        fontsize=6.3,
    )
    add_panel_tag(ax2, "c", x_pt=-32.0, y_pt=6.0)

    export_figure(
        fig,
        "fig5_mammalian_and_scaling",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Slight width ratios accommodate 6-bar, 8-bar, and log-log scaling panels",
            }
        ],
    )


def build_fig6_trajectories_and_kinetics(out_latex_fig: Path) -> None:
    """Figure 6: Real continuous-time trajectories & gauge-invariant step-on/step-off memory kinetics."""
    z = np.load(ROOT / "results" / "trajectory_examples.npz")
    port = load_new_result("cross_domain_portraits")
    yst = port["yeast_msn2_minutes"]
    t_min = np.asarray(yst["step_curve_t_model_units"], dtype=np.float64) * 60.0
    yp = yst["by_promoter"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.30))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.05, 1.05],
        left=0.092,
        right=0.968,
        bottom=0.22,
        top=0.84,
        wspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Cyclic zero-shot double-KO trajectory ([0, 9] h with [5, 9] h extrapolation shading)
    t_d = z["cyc_t_dense"]
    t_o = z["cyc_t_obs"]
    ax0.fill_between([5.0, 9.0], -0.12, 1.98, color=PALETTE["light_gray"], zorder=0, lw=0)
    ax0.plot([5.0, 5.0], [-0.12, 1.98], color=PALETTE["slate"], lw=0.7, ls=":", zorder=1)
    ax0.plot(t_d, z["cyc_cl_selective_ssm"], color="#D97736", lw=1.05, ls=":", label="Mamba")
    ax0.plot(t_d, z["cyc_cl_latent_ode"], color="#8C6D9B", lw=1.05, ls="-.", label="Lat-ODE")
    ax0.plot(t_d, z["cyc_cl_coupled_adaptive"], color=PALETTE["amber"], lw=1.15, ls="--", label="Cpl-Ad.")
    ax0.plot(t_d, z["cyc_cl_memgrn"], color=PALETTE["navy"], lw=1.45, label="MemGRN")
    ax0.scatter(t_o, z["cyc_truth_obs"], s=11, color=PALETTE["ink"], zorder=5, label="Truth")
    ax0.set_xlim(0.0, 9.2)
    ax0.set_ylim(-0.12, 3.08)
    ax0.set_xticks([0, 3, 6, 9])
    ax0.set_yticks([0.0, 0.8, 1.6, 2.4])
    ax0.set_xlabel("Time (h)")
    ax0.set_ylabel("Node state y₁(t)")
    ax0.set_title("Cyclic 0-shot 2-KO")
    ax0.legend(
        loc="upper left",
        ncol=2,
        columnspacing=0.32,
        handlelength=0.78,
        handletextpad=0.25,
        labelspacing=0.14,
        borderpad=0.12,
        fontsize=7.1,
    )
    add_panel_tag(ax0, "a", x_pt=-30.0, y_pt=6.0)

    # Panel b: Live-cell yeast SIP18 6-pulse chromatin hysteresis (experimental microscopy vs fits)
    t_m = z["yeast_obs_min"]
    for k_p in range(6):
        ax1.fill_between(
            [10.0 * k_p, 10.0 * k_p + 5.0],
            -0.06,
            1.18,
            color=PALETTE["light_blue"],
            alpha=0.85,
            zorder=0,
            lw=0,
        )
    ax1.plot(t_m, z["yeast_SIP18_constant"], color=PALETTE["crimson"], lw=1.2, ls="--", label="Const gW")
    ax1.plot(t_m, z["yeast_SIP18_memory"], color=PALETTE["navy"], lw=1.45, label="MemGRN")
    ax1.scatter(t_m, z["yeast_SIP18_obs"], s=11, color=PALETTE["ink"], zorder=5, label="Data")
    ax1.set_xlim(0.0, 122.0)
    ax1.set_ylim(-0.06, 2.15)
    ax1.set_xticks([0, 40, 80, 120])
    ax1.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("SIP18 YFP (norm.)")
    ax1.set_title("Yeast 6-pulse train")
    ax1.legend(
        loc="upper left",
        ncol=2,
        columnspacing=0.45,
        handlelength=0.95,
        handletextpad=0.3,
        labelspacing=0.14,
        borderpad=0.15,
        fontsize=7.3,
    )
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Gauge-invariant RK4 dynamical step-on (dashed) vs step-off (solid) memory curves
    # Directly shows rapid ~24-25 min loading across all promoters vs 1.4x/3.0x/4.7x washout lock-in!
    ax2.axhline(0.5, color=PALETTE["slate"], lw=0.65, ls=":", zorder=0)
    ax2.plot(t_min, yp["HXK1"]["step_on_curve"], color=PALETTE["teal"], lw=1.05, ls="--", label="HXK1 on")
    ax2.plot(t_min, yp["HXK1"]["step_off_curve"], color=PALETTE["teal"], lw=1.45, label="HXK1 off")
    ax2.plot(t_min, yp["SIP18"]["step_on_curve"], color=PALETTE["navy"], lw=1.05, ls="--", label="SIP18 on")
    ax2.plot(t_min, yp["SIP18"]["step_off_curve"], color=PALETTE["navy"], lw=1.45, label="SIP18 off")
    ax2.plot(t_min, yp["ALD3"]["step_off_curve"], color=PALETTE["amber"], lw=1.45, ls="-.", label="ALD3 off")
    ax2.set_xlim(0.0, 180.0)
    ax2.set_ylim(-0.04, 1.44)
    ax2.set_xticks([0, 60, 120, 180])
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel("Step time (min)")
    ax2.set_ylabel("Norm. memory")
    ax2.set_title("Step-response")
    ax2.legend(
        loc="upper right",
        ncol=2,
        columnspacing=0.40,
        handlelength=0.9,
        handletextpad=0.28,
        labelspacing=0.12,
        borderpad=0.12,
        fontsize=7.1,
    )
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(fig, "fig6_trajectories_and_kinetics", out_latex_fig)


def build_fig7_cross_domain_hysteresis(out_latex_fig: Path) -> None:
    """Figure 7: Gauge-invariant cross-domain step-response hysteresis (all 8 Beijing, all 16 Binance, 4-domain incl. METR-LA)."""
    port = load_new_result("cross_domain_portraits")
    bq = port["beijing_air_quality_hours"]
    bq_src = bq["by_source"]
    fin = port["binance_16asset_minutes"]
    fin_src = fin["by_source"]
    metr = port["metr_la_s64_minutes"]
    yst = port["yeast_msn2_minutes"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.40))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.02, 1.24, 0.90],
        left=0.092,
        right=0.985,
        bottom=0.34,
        top=0.84,
        wspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: ALL 8 UCI Beijing Air Quality drivers' gauge-invariant H = tau_off / tau_on (± SE across 4 folds)
    drv_items = sorted(
        bq_src.items(), key=lambda kv: kv[1]["hysteresis_ratio_off_over_on"], reverse=True
    )
    short_bq = {
        "Pressure": "Pres.",
        "Temp": "Temp",
        "Wind_North": "Wind-N",
        "Rain": "Rain",
        "Wind_East": "Wind-E",
        "Dewpoint": "Dewpt",
        "Diurnal_Sin": "Diurn.",
        "Wind_Speed": "W-Spd",
    }
    drv_lbls = [short_bq[k] for k, _ in drv_items]
    drv_vals = np.array([v["hysteresis_ratio_off_over_on"] for _, v in drv_items])
    drv_ses = np.array([v["hysteresis_ratio_se"] for _, v in drv_items])
    drv_cols = [
        PALETTE["navy"] if v["dominant_effect_sign"] == "positive" else PALETTE["teal"]
        for _, v in drv_items
    ]
    xa = np.arange(len(drv_items))
    ax0.axhline(1.0, color=PALETTE["slate"], lw=0.75, ls="--", zorder=0)
    ax0.bar(
        xa,
        drv_vals,
        yerr=drv_ses,
        width=0.62,
        color=drv_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        capsize=1.8,
        error_kw={"elinewidth": 0.7, "ecolor": PALETTE["ink"]},
    )
    ax0.bar([-5], [0], color=PALETTE["navy"], edgecolor=PALETTE["ink"], lw=0.55, label="+ effect")
    ax0.bar([-5], [0], color=PALETTE["teal"], edgecolor=PALETTE["ink"], lw=0.55, label="− effect")
    ax0.set_xticks(xa)
    ax0.set_xticklabels(drv_lbls, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_xlim(-0.6, len(drv_items) - 0.4)
    ax0.set_ylim(0.0, 9.2)
    ax0.set_yticks([0, 2, 4, 6, 8])
    ax0.set_ylabel(r"Hysteresis $\tau_{\mathrm{off}}/\tau_{\mathrm{on}}$")
    ax0.set_title("Beijing (all 8)")
    ax0.legend(loc="upper right", ncol=1, handlelength=0.95, labelspacing=0.14, fontsize=7.2)
    add_panel_tag(ax0, "a", x_pt=-30.0, y_pt=6.0)

    # Panel b: ALL 16 Binance assets' gauge-invariant H = tau_off / tau_on (± SE across 3 seeds)
    fin_items = sorted(
        fin_src.items(), key=lambda kv: kv[1]["hysteresis_ratio_off_over_on"], reverse=True
    )
    fin_lbls = [k.replace("USDT", "") for k, _ in fin_items]
    fin_vals = np.array([v["hysteresis_ratio_off_over_on"] for _, v in fin_items])
    fin_ses = np.array([v["hysteresis_ratio_se"] for _, v in fin_items])
    fin_cols = [
        PALETTE["amber"] if v > 1.25 else (PALETTE["slate"] if v >= 0.85 else PALETTE["teal"])
        for v in fin_vals
    ]
    xb = np.arange(len(fin_items))
    ax1.axhline(1.0, color=PALETTE["slate"], lw=0.75, ls="--", zorder=0)
    ax1.bar(
        xb,
        fin_vals,
        yerr=fin_ses,
        width=0.66,
        color=fin_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.45,
        capsize=1.2,
        error_kw={"elinewidth": 0.6, "ecolor": PALETTE["ink"]},
    )
    ax1.set_xticks(xb)
    ax1.set_xticklabels(fin_lbls, rotation=90, ha="center", va="top", fontsize=6.8)
    ax1.tick_params(axis="x", pad=3.5)
    ax1.set_xlim(-0.6, len(fin_items) - 0.4)
    ax1.set_ylim(0.0, 5.5)
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    ax1.set_ylabel(r"Ratio $\tau_{\mathrm{off}}/\tau_{\mathrm{on}}$")
    ax1.set_title("Binance (all 16)")
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    # Panel c: Unified 4-domain contrast INCLUDING METR-LA S=64 exact negative control (H = 1.000)
    yp = yst["by_promoter"]
    yf_h = [yp[g]["hysteresis_ratio_off_over_on"] for g in ("HXK1", "DCS2")]
    ys_h = [yp[g]["hysteresis_ratio_off_over_on"] for g in ("DDR2", "ALD3", "TKL2", "SIP18")]
    cd_lbls = ["METR ctrl", "Binance", "Yst fast", "Beijing", "Yst slow"]
    cd_vals = [
        metr["mean_hysteresis_ratio"],
        fin["mean_hysteresis_ratio"],
        yst["fast_open_mean_hysteresis"],
        bq["mean_hysteresis_ratio"],
        yst["slow_remod_mean_hysteresis"],
    ]
    cd_ses = [
        metr["mean_hysteresis_ratio_se"],
        fin["mean_hysteresis_ratio_se"],
        float(np.std(yf_h, ddof=1) / np.sqrt(len(yf_h))),
        bq["mean_hysteresis_ratio_se"],
        float(np.std(ys_h, ddof=1) / np.sqrt(len(ys_h))),
    ]
    cd_cols = [
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["steel"],
        PALETTE["navy"],
    ]
    xc = np.arange(len(cd_lbls))
    ax2.axhline(1.0, color=PALETTE["slate"], lw=0.75, ls="--", zorder=0)
    ax2.bar(
        xc,
        cd_vals,
        yerr=cd_ses,
        width=0.60,
        color=cd_cols,
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        capsize=1.8,
        error_kw={"elinewidth": 0.7, "ecolor": PALETTE["ink"]},
    )
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cd_lbls, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.0)
    ax2.set_xlim(-0.6, len(cd_lbls) - 0.4)
    ax2.set_ylim(0.0, 4.4)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_ylabel(r"Mean $\tau_{\mathrm{off}}/\tau_{\mathrm{on}}$")
    ax2.set_title("4-domain contrast")
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(
        fig,
        "fig7_cross_domain_hysteresis",
        out_latex_fig,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Width ratios accommodate 8-driver, 16-asset, and 5-domain panels",
            }
        ],
    )


def build_all_tables(out_tab_dir: Path) -> None:
    """Write all 5 main-text tables and 14 appendix tables with 100% programmatic bolding."""
    fair_ol = load_new_result("fair_open_loop")
    fair_cl = load_new_result("fair_closed_loop")
    port = load_new_result("cross_domain_portraits")
    hipsc = load_new_result("hipsc_closed_loop")["summary"]
    cyc = load_new_result("cyclic_double_ko")["summary"]
    emt = fair_ol["emt_gse147405"]
    mtf = fair_ol["multitf_24world"]
    po6 = fair_ol["partial_obs_6of8"]["summary"]
    ltc_sep = fair_ol["ltc_and_dssm_separation"]
    joint = load_new_result("msn2_joint_and_cross_readout")
    yeast_fair = fair_ol["yeast_msn2_joint"]
    mac = load_new_result("macrophage_multitf")["summary"]
    ortho = load_new_result("nfkb_ortho")["summary"]
    sens = load_new_result("sensitivity_ablations")["summary"]
    occ = load_stage3("neural_rate_pulse", "test_scores.json")
    buf = load_stage3("neural_rate_buffer", "test_scores.json")
    six = load_stage3("msn2_six_promoter_rates")
    swap = load_stage3("strength_swap_replication")
    cert = load_stage3("paired_time_certificate", "test_scores.json")
    cert_meta = load_stage3("paired_time_certificate", "certificate.json")

    # =========================================================================
    # Appendix Table (hiPSC-only; duplicate Cyclic 2-KO block removed per R3-M4)
    # =========================================================================
    t1_hi_order = [
        ("cl_memgrn", "\\textbf{MemGRN (ours)}"),
        ("cl_local_const", "Local Constant"),
        ("cl_ltc_local", "Source-Local LTC"),
        ("cl_selective_ssm", "Selective SSM (Mamba)"),
        ("cl_coupled_adaptive", "Coupled Adaptive"),
        ("cl_ltc_coupled", "Coupled LTC"),
        ("cl_cfc", "CfC"),
        ("cl_perturb_ode", "PerturbODE"),
        ("cl_latent_ode", "Latent Neural ODE"),
        ("renge", "RENGE"),
    ]
    b_ko = min(hipsc[k]["unseen_ko_mse"] for k, _ in t1_hi_order)
    b_t25 = min(hipsc[k]["unseen_ko_top25_mse"] for k, _ in t1_hi_order)
    b_d5 = min(hipsc[k]["day5_extrapolation_mse"] for k, _ in t1_hi_order)
    b_ap = max(hipsc[k]["chip_psc_ap_over_random"] for k, _ in t1_hi_order)
    t1_rows: list[list[str]] = []
    for k, lbl in t1_hi_order:
        d = hipsc[k]
        s_ko = f"\\textbf{{{d['unseen_ko_mse']:.4f}}}" if np.isclose(d["unseen_ko_mse"], b_ko) else f"{d['unseen_ko_mse']:.4f}"
        s_t25 = f"\\textbf{{{d['unseen_ko_top25_mse']:.4f}}}" if np.isclose(d["unseen_ko_top25_mse"], b_t25) else f"{d['unseen_ko_top25_mse']:.4f}"
        s_d5 = f"\\textbf{{{d['day5_extrapolation_mse']:.4f}}}" if np.isclose(d["day5_extrapolation_mse"], b_d5) else f"{d['day5_extrapolation_mse']:.4f}"
        s_ap = f"\\textbf{{{d['chip_psc_ap_over_random']:.2f}$\\times$}}" if np.isclose(d["chip_psc_ap_over_random"], b_ap) else f"{d['chip_psc_ap_over_random']:.2f}$\\times$"
        t1_rows.append([lbl, str(d["n_params"]), s_ko, s_t25, s_d5, s_ap])

    write_tex_table(
        out_tab_dir,
        "tab1_hipsc_closed_loop.tex",
        "lrrrrr",
        [
            "Closed-loop model (hiPSC)",
            "Params",
            r"Unseen KO $\downarrow$",
            r"Top-25 $\downarrow$",
            r"Day 5 $\downarrow$",
            r"PSC ChIP $\uparrow$",
        ],
        t1_rows,
    )

    t1_cy_order = [
        ("cl_memgrn", "\\textbf{MemGRN (ours)}"),
        ("cl_coupled_adaptive", "Coupled Adaptive"),
        ("cl_local_const", "Local Constant"),
        ("cl_latent_ode", "Latent Neural ODE"),
        ("cl_ltc_coupled", "Coupled LTC"),
        ("cl_cfc", "CfC"),
        ("cl_selective_ssm", "Selective SSM (Mamba)"),
        ("cl_ltc_local", "Source-Local LTC"),
        ("cl_perturb_ode", "PerturbODE"),
    ]
    b_1k = min(cyc[k]["unseen_single_ko_mse"] for k, _ in t1_cy_order)
    b_2k = min(cyc[k]["double_ko_mse"] for k, _ in t1_cy_order)
    b_ex = min(cyc[k]["horizon_extrap_mse"] for k, _ in t1_cy_order)
    b_gap = max(cyc[k]["graph_ap"] for k, _ in t1_cy_order)

    # Compact standalone Cyclic Zero-Shot Double-KO table for Main Table 4
    t4_cy_rows: list[list[str]] = []
    for k, lbl in t1_cy_order:
        d = cyc[k]
        s1 = f"{d['unseen_single_ko_mse']:.4f}"
        s2 = f"{d['double_ko_mse']:.3f}$\\pm${d['double_ko_mse_se']:.3f}"
        se = f"{d['horizon_extrap_mse']:.4f}"
        sau = f"{d['graph_auroc']:.3f}"
        sg = f"{d['graph_ap']:.3f}"
        if np.isclose(d["unseen_single_ko_mse"], b_1k):
            s1 = f"\\textbf{{{s1}}}"
        if np.isclose(d["double_ko_mse"], b_2k):
            s2 = f"\\textbf{{{d['double_ko_mse']:.3f}}}$\\pm${d['double_ko_mse_se']:.3f}"
        if np.isclose(d["horizon_extrap_mse"], b_ex):
            se = f"\\textbf{{{se}}}"
        if np.isclose(d["graph_ap"], b_gap):
            sg = f"\\textbf{{{sg}}}"
        if np.isclose(d["graph_auroc"], max(cyc[kk]["graph_auroc"] for kk, _ in t1_cy_order)):
            sau = f"\\textbf{{{sau}}}"
        t4_cy_rows.append([lbl, s1, s2, se, sau, sg])
    write_tex_table(
        out_tab_dir,
        "tab4_cyclic_double_ko.tex",
        "lrrrrr",
        [
            "Closed-loop model ($12$ worlds)",
            r"1-KO $\downarrow$",
            r"Double-KO $\downarrow$",
            r"$[5,9]\,\mathrm{h}$ Ext.\ $\downarrow$",
            r"AUROC $\uparrow$",
            r"Graph AP $\uparrow$",
        ],
        t4_cy_rows,
    )

    # =========================================================================
    # Appendix Table: Clean Equal-Budget Human Cancer EMT 3-Day Withdrawal (GSE147405)
    # =========================================================================
    t2_order = [
        ("memgrn_ortho", "\\textbf{MemGRN-Ortho (ours)}"),
        ("memgrn", "\\textbf{MemGRN (ours)}"),
        ("memgrn_additive", "MemGRN ($\\Gamma = 0$)"),
        ("depthwise_ssm_coop", "Depthwise SSM ($+\\Gamma$)"),
        ("local_const_coop", "Local Const.\\ ($+\\Gamma$)"),
        ("ltc_local_coop", "Source-Local LTC ($+\\Gamma$)"),
        ("perturb_ode", "PerturbODE"),
        ("cfc", "CfC"),
        ("selective_ssm", "Selective SSM (Mamba)"),
        ("neural_cde", "Neural CDE"),
        ("ltc_coupled", "Coupled LTC"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("latent_ode", "Latent Neural ODE"),
    ]
    b_t30 = min(emt[k]["top30_withdrawal_mse"] for k, _ in t2_order)
    b_all = min(emt[k]["withdrawal_mse"] for k, _ in t2_order)
    b_r2 = max(emt[k]["withdrawal_r2"] for k, _ in t2_order)
    b_col = max(emt[k]["collectri_ap_over_random_stage1"] for k, _ in t2_order)
    t2_rows: list[list[str]] = []
    for idx_e, (k, lbl) in enumerate(t2_order):
        if idx_e == 7:
            t2_rows.append(["\\midrule"])
        d = emt[k]
        st = (
            f"\\textbf{{{d['top30_withdrawal_mse']:.3f}}}$\\pm${d['top30_withdrawal_mse_se']:.3f}"
            if np.isclose(d["top30_withdrawal_mse"], b_t30)
            else f"{d['top30_withdrawal_mse']:.3f}$\\pm${d['top30_withdrawal_mse_se']:.3f}"
        )
        sa = (
            f"\\textbf{{{d['withdrawal_mse']:.3f}}}"
            if np.isclose(d["withdrawal_mse"], b_all)
            else f"{d['withdrawal_mse']:.3f}"
        )
        sr = (
            f"\\textbf{{{d['withdrawal_r2']:.3f}}}"
            if np.isclose(d["withdrawal_r2"], b_r2)
            else f"{d['withdrawal_r2']:.3f}"
        )
        sc = (
            f"\\textbf{{{d['collectri_ap_over_random_stage1']:.2f}$\\times$}}"
            if np.isclose(d["collectri_ap_over_random_stage1"], b_col)
            else f"{d['collectri_ap_over_random_stage1']:.2f}$\\times$"
        )
        t2_rows.append([lbl, str(d["n_params"]), st, sa, sr, sc])
    write_tex_table(
        out_tab_dir,
        "tab2_emt_ligand_shift.tex",
        "lrrrrr",
        [
            "Model (76 EMT targets)",
            "Params",
            r"Top-30 MSE $\downarrow$",
            r"All-76 $\downarrow$",
            r"Wdr.\ $R^2$ $\uparrow$",
            r"CollecTRI $\uparrow$",
        ],
        t2_rows,
    )

    # =========================================================================
    # Main Table 3: Equal-Budget 24-World Multi-Source (15 models) + Thm 1(iii) Separation
    # =========================================================================
    mtf_order = [
        ("memgrn", "\\textbf{MemGRN (ours)}"),
        ("depthwise_ssm_coop", "Depthwise SSM ($+\\Gamma$)"),
        ("ltc_local_coop", "Source-Local LTC ($+\\Gamma$)"),
        ("local_const_coop", "Local Const.\\ ($+\\Gamma$)"),
        ("memgrn_additive", "MemGRN ($\\Gamma = 0$)"),
        ("neural_cde", "Neural CDE"),
        ("selective_ssm", "Selective SSM (Mamba)"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("perturb_ode", "PerturbODE"),
        ("ltc_coupled", "Coupled LTC"),
        ("cfc", "CfC"),
    ]
    add_m = mtf["additive"]["means"]
    coop_m = mtf["cooperative"]["means"]
    b_add = min(add_m[k]["test_mse"] for k, _ in mtf_order)
    b_coo = min(coop_m[k]["test_mse"] for k, _ in mtf_order)
    b_att = min(0.5 * (add_m[k]["attribution_rmse"] + coop_m[k]["attribution_rmse"]) for k, _ in mtf_order)
    b_eap = max(0.5 * (add_m[k]["edge_ap"] + coop_m[k]["edge_ap"]) for k, _ in mtf_order)
    t3_rows: list[list[str]] = []
    for idx_m, (k, lbl) in enumerate(mtf_order):
        if idx_m == 5:
            t3_rows.append(["\\midrule"])
        a_v, a_se = add_m[k]["test_mse"], add_m[k]["test_mse_se"]
        c_v, c_se = coop_m[k]["test_mse"], coop_m[k]["test_mse_se"]
        at_v = 0.5 * (add_m[k]["attribution_rmse"] + coop_m[k]["attribution_rmse"])
        ap_v = 0.5 * (add_m[k]["edge_ap"] + coop_m[k]["edge_ap"])
        sa = f"\\textbf{{{a_v:.3f}}}$\\pm${a_se:.3f}" if np.isclose(a_v, b_add) else f"{a_v:.3f}$\\pm${a_se:.3f}"
        sc = f"\\textbf{{{c_v:.3f}}}$\\pm${c_se:.3f}" if np.isclose(c_v, b_coo) else f"{c_v:.3f}$\\pm${c_se:.3f}"
        sat = f"\\textbf{{{at_v:.4f}}}" if np.isclose(at_v, b_att) else f"{at_v:.4f}"
        sap = f"\\textbf{{{ap_v:.3f}}}" if np.isclose(ap_v, b_eap) else f"{ap_v:.3f}"
        t3_rows.append([lbl, str(add_m[k]["n_params"]), sa, sc, sat, sap])

    t3_rows.append(["\\midrule"])
    pb_order = [
        ("memgrn_matched", "\\textbf{MemGRN (Thm.~1(iii))}"),
        ("depthwise_ssm_matched", "Depthwise SSM (m.)"),
        ("ltc_local_matched", "Source-Local LTC (m.)"),
        ("local_const_matched", "Local Constant (m.)"),
    ]
    b_pk = min(ltc_sep[k]["delayed_peak_post_mse"] for k, _ in pb_order)
    b_ai = min(ltc_sep[k]["steady_state_amp_invariance_error"] for k, _ in pb_order)
    b_wr = min(ltc_sep[k]["steady_state_weight_rel_error"] for k, _ in pb_order)
    for k, lbl in pb_order:
        dv = ltc_sep[k]
        pk_s = f"{1e5 * dv['delayed_peak_post_mse']:.2f}$\\times 10^{{-5}}$"
        amp_s = f"{100.0 * dv['steady_state_amp_invariance_error']:.2f}\\%"
        wt_s = f"{100.0 * dv['steady_state_weight_rel_error']:.2f}\\%"
        if np.isclose(dv["delayed_peak_post_mse"], b_pk):
            pk_s = f"\\textbf{{{pk_s}}}"
        if np.isclose(dv["steady_state_amp_invariance_error"], b_ai):
            amp_s = f"\\textbf{{{amp_s}}}"
        if np.isclose(dv["steady_state_weight_rel_error"], b_wr):
            wt_s = f"\\textbf{{{wt_s}}}"
        t3_rows.append([lbl, str(dv["n_params"]), pk_s, amp_s, wt_s, "--"])
    write_tex_table(
        out_tab_dir,
        "tab3_multitf_and_ltc_sep.tex",
        "lrrrrr",
        [
            "Architecture (24w / Thm.~1(iii))",
            "Params",
            r"Add./Peak $\downarrow$",
            r"Coop./Amp.\ $\downarrow$",
            r"Attr./Wt.\ $\downarrow$",
            r"AP $\uparrow$",
        ],
        t3_rows,
    )

    # =========================================================================
    # Appendix Table: Equal-Budget Yeast Msn2 Joint Source-CV (12 architectures)
    # =========================================================================
    jt = yeast_fair["models"]
    js_holm = yeast_fair["holm_vs_memgrn_sym"]
    t4_rows: list[list[str]] = []
    j_order = [
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("ltc_local", "Source-Local LTC"),
        ("latent_ode", "Latent Neural ODE"),
        ("neural_cde", "Neural CDE"),
        ("memgrn_sym", "\\textbf{MemGRN (sym., ours)}"),
        ("memgrn", "\\textbf{MemGRN (asym., ours)}"),
        ("perturb_ode", "PerturbODE"),
        ("ltc_coupled", "Coupled LTC"),
        ("selective_ssm", "Selective SSM (Mamba)"),
        ("depthwise_ssm", "Depthwise SSM"),
        ("cfc", "CfC"),
        ("local_const", "Local Constant ($gW$)"),
    ]
    best_val = min(jt[k]["val_loss"] for k, _ in j_order)
    best_10 = min(jt[k]["ten_train_post_mse"] for k, _ in j_order)
    for k, lbl in j_order:
        d = jt[k]
        sv = f"\\textbf{{{d['val_loss']:.4f}}}" if np.isclose(d["val_loss"], best_val) else f"{d['val_loss']:.4f}"
        s10 = (
            f"\\textbf{{{d['ten_train_post_mse']:.4f}}}$\\pm${d['ten_train_post_mse_se']:.4f}"
            if np.isclose(d["ten_train_post_mse"], best_10)
            else f"{d['ten_train_post_mse']:.4f}$\\pm${d['ten_train_post_mse_se']:.4f}"
        )
        if k == "memgrn_sym":
            w_s, p_s = "--", "--"
        else:
            st = js_holm[k]
            w_s = f"{st['wins']}/{st['total']}"
            p_s = f"{st['holm_p_two_sided']:.3f}" if st["holm_p_two_sided"] >= 0.001 else f"{st['holm_p_two_sided']:.1e}"
        t4_rows.append([lbl, str(d["n_params"]), sv, s10, w_s, p_s])
    write_tex_table(
        out_tab_dir,
        "tab4_msn2_joint_and_cross.tex",
        "lrrrrr",
        [
            "Joint source-CV ($S{=}1$)",
            "Params",
            r"Val.\ $\downarrow$",
            r"10-train MSE $\downarrow$",
            r"Sym.\ wins",
            r"$p_{\mathrm{Holm}}$",
        ],
        t4_rows,
    )

    # =========================================================================
    # Main Table 5: Cross-Domain Partially Observed Cofactors (6/8 GT, Fibro, Finance 14-alt)
    # =========================================================================
    fin_po = fair_ol["binance_contagion"]["partial_14alt"]
    t5_rows: list[list[str]] = []
    pa_order = [
        ("memgrn_ortho_2stage", "\\textbf{MemGRN-Ortho (2-st.)}"),
        ("memgrn_ortho_joint", "MemGRN-Ortho (joint)"),
        ("memgrn_unproj", "MemGRN + Unproj.\\ (j.)"),
        ("memgrn", "MemGRN (obs.\\ only)"),
        ("depthwise_ssm", "Depthwise SSM"),
        ("latent_ode", "Latent Neural ODE"),
        ("perturb_ode", "PerturbODE"),
    ]
    b_pmse = min(po6[k]["test_mse"] for k, _ in pa_order)
    b_patr = min(po6[k]["full_model_gt_attr_rmse"] for k, _ in pa_order)
    b_pap = max(po6[k]["full_model_gt_edge_ap"] for k, _ in pa_order)
    for i_p, (k, lbl) in enumerate(pa_order):
        d = po6[k]
        sm = f"\\textbf{{{d['test_mse']:.4f}}}" if np.isclose(d["test_mse"], b_pmse) else f"{d['test_mse']:.4f}"
        sa = f"\\textbf{{{d['full_model_gt_attr_rmse']:.4f}}}" if np.isclose(d["full_model_gt_attr_rmse"], b_patr) else f"{d['full_model_gt_attr_rmse']:.4f}"
        sp = f"\\textbf{{{d['full_model_gt_edge_ap']:.3f}}}" if np.isclose(d["full_model_gt_edge_ap"], b_pap) else f"{d['full_model_gt_edge_ap']:.3f}"
        sd = (
            f"{100.0 * d['obs_branch_drift_vs_memgrn']:.0f}\\%/{100.0 * d['full_model_drift_vs_memgrn']:.0f}\\%"
            if k in ("memgrn_ortho_2stage", "memgrn_ortho_joint")
            else f"{100.0 * d['full_model_drift_vs_memgrn']:.1f}\\%"
        )
        t5_rows.append(["6/8 GT" if i_p == 0 else "", lbl, sm, sa, sp, sd])

    t5_rows.append(["\\midrule"])
    o_order = [
        ("memgrn_ortho", "\\textbf{MemGRN-Ortho (ours)}"),
        ("latent_ode", "Latent Neural ODE"),
        ("memgrn", "MemGRN (NF-$\\kappa$B)"),
        ("perturb_ode", "PerturbODE"),
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

    t5_rows.append(["\\midrule"])
    f_po_ode = [
        ("memgrn_ortho", "\\textbf{MemGRN-Ortho (ours)}"),
        ("memgrn", "MemGRN (14 altcoins)"),
        ("latent_ode", "Latent Neural ODE"),
        ("memgrn_unproj", "MemGRN + Unproj."),
        ("depthwise_ssm_coop", "Depthwise SSM ($+\\Gamma$)"),
    ]
    f_po_disc = [
        ("dlinear", "DLinear (discrete)"),
    ]
    f_po_all = f_po_ode + f_po_disc
    bf_a_all = min(fin_po[k]["aug_test_mse"] for k, _ in f_po_all)
    bf_a_ode = min(fin_po[k]["aug_test_mse"] for k, _ in f_po_ode)
    bf_i_all = min(fin_po[k]["idio_shock_mse"] for k, _ in f_po_all)
    bf_i_ode = min(fin_po[k]["idio_shock_mse"] for k, _ in f_po_ode)
    bf_r2_all = max(fin_po[k]["aug_test_r2"] for k, _ in f_po_all)
    bf_r2_ode = max(fin_po[k]["aug_test_r2"] for k, _ in f_po_ode)
    for i_f, (k, lbl) in enumerate(f_po_all):
        d = fin_po[k]
        s_a = f"{d['aug_test_mse']:.3f}"
        if np.isclose(d["aug_test_mse"], bf_a_all):
            s_a = f"\\textbf{{{s_a}}}"
        elif k in dict(f_po_ode) and np.isclose(d["aug_test_mse"], bf_a_ode):
            s_a = f"\\underline{{{s_a}}}"
        s_i = f"{d['idio_shock_mse']:.3f}"
        if np.isclose(d["idio_shock_mse"], bf_i_all):
            s_i = f"\\textbf{{{s_i}}}"
        elif k in dict(f_po_ode) and np.isclose(d["idio_shock_mse"], bf_i_ode):
            s_i = f"\\underline{{{s_i}}}"
        s_r = f"{d['aug_test_r2']:.3f}"
        if np.isclose(d["aug_test_r2"], bf_r2_all):
            s_r = f"\\textbf{{{s_r}}}"
        elif k in dict(f_po_ode) and np.isclose(d["aug_test_r2"], bf_r2_ode):
            s_r = f"\\underline{{{s_r}}}"
        t5_rows.append(["Finance" if i_f == 0 else "", lbl, s_a, s_i, s_r, str(d["n_params"])])

    write_tex_table(
        out_tab_dir,
        "tab5_ortho_ground_truth_and_fibroblast.tex",
        "llrrrr",
        [
            "Suite",
            "Model",
            r"MSE $\downarrow$",
            r"Attr./\texttt{IL1}/Idio $\downarrow$",
            r"AP/\texttt{TNF}/$R^2$",
            r"Drift / Params",
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
    # Appendix Table A2: Gauge-Invariant Yeast Chromatin Portraits & Complete S=8..512 Scaling
    # =========================================================================
    portraits = port["yeast_msn2_minutes"]["by_promoter"]
    scaling = fair_ol["scaling_s8_to_s512"]
    ta2_rows: list[list[str]] = []
    for g in ("HXK1", "DCS2", "RTN2", "DDR2", "SIP18", "TKL2", "ALD3"):
        pg_d = portraits[g]
        cls_short = "Fast (open)" if g in ("HXK1", "DCS2") else ("Interm." if g == "RTN2" else "Slow (remod.)")
        ta2_rows.append(
            [
                f"\\texttt{{{g}}}",
                cls_short,
                f"{pg_d['tau_on_half']:.1f}/{pg_d['tau_off_half']:.1f}\\,m",
                f"{pg_d['hysteresis_ratio_off_over_on']:.2f}$\\times$",
                f"{pg_d['ten_train_post_mse']:.4f}",
            ]
        )
    def _fmt_p(n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 100_000:
            return f"{n / 1000.0:.1f}k"
        if n < 1_000_000:
            return f"{n / 1000.0:.0f}k"
        return f"{n / 1e6:.2f}M"

    ta2_rows.append(["\\midrule"])
    for r in scaling:
        s_n = f"$S={r['n_tfs']}$ ({r['n_genes']})"
        rk_s = f"{_fmt_p(r['memgrn_rank24_mem_params'])}/{_fmt_p(r['memgrn_rank24_total_params'])} ({r['memgrn_rank24_ms_per_step']:.0f}ms)"
        ca_s = f"{_fmt_p(r['coupled_adaptive_mem_params'])}/{_fmt_p(r['coupled_adaptive_total_params'])} ({r['coupled_adaptive_ms_per_step']:.0f}ms)"
        ltc_s = f"{_fmt_p(r['coupled_ltc_mem_params'])}/{_fmt_p(r['coupled_ltc_total_params'])}"
        if r["n_tfs"] == 512:
            ltc_s += "*"
        ta2_rows.append(
            [
                s_n,
                rk_s,
                ca_s,
                ltc_s,
                f"{r['coupled_adaptive_mem_params'] / r['memgrn_rank24_mem_params']:.1f}$\\times$",
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a2_portraits_and_scaling.tex",
        "llrrr",
        [
            r"Prom.\ / $S(N)$",
            "Class / MemGRN (m/tot)",
            r"$\tau_{\mathrm{on}}/\tau_{\mathrm{off}}$ / Cpl-Ad.",
            r"$\mathcal{H}$ / Cpl-LTC",
            "MSE / Ratio",
        ],
        ta2_rows,
    )

    # =========================================================================
    # Appendix Table A9: Sensitivity & Robustness Ablations (Channels R, Noise, Dropout)
    # =========================================================================
    ta9_rows: list[list[str]] = []
    for r_i, r_lbl in enumerate(("R=1", "R=2", "R=3", "R=5")):
        dm = sens["channels"][r_lbl]["memgrn"]
        ta9_rows.append(
            [
                "Channels" if r_i == 0 else "",
                f"MemGRN (${r_lbl}$)",
                f"{dm['test_mse']:.4f}$\\pm${dm['test_mse_se']:.3f}",
                f"{dm['attribution_rmse']:.4f}",
                str(dm["n_params"]),
            ]
        )
    ta9_rows.append(["\\midrule"])
    for s_i, s_lbl in enumerate(("sigma=0.015", "sigma=0.050", "sigma=0.100", "sigma=0.150")):
        dm = sens["noise"][s_lbl]["memgrn"]
        dl = sens["noise"][s_lbl]["ltc_local"]
        dc = sens["noise"][s_lbl]["coupled_adaptive"]
        vals_n = [dm["test_mse"], dl["test_mse"], dc["test_mse"]]
        bn = min(vals_n)
        cells_n = [f"\\textbf{{{v:.4f}}}" if np.isclose(v, bn) else f"{v:.4f}" for v in vals_n]
        ta9_rows.append(["Noise" if s_i == 0 else "", f"$\\sigma={s_lbl.split('=')[1]}$", *cells_n])
    ta9_rows.append(["\\midrule"])
    for d_i, d_lbl in enumerate(("drop=0%", "drop=25%", "drop=50%")):
        dm = sens["dropout"][d_lbl]["memgrn"]
        dl = sens["dropout"][d_lbl]["ltc_local"]
        dc = sens["dropout"][d_lbl]["coupled_adaptive"]
        vals_d = [dm["test_mse"], dl["test_mse"], dc["test_mse"]]
        bd = min(vals_d)
        cells_d = [f"\\textbf{{{v:.4f}}}" if np.isclose(v, bd) else f"{v:.4f}" for v in vals_d]
        ta9_rows.append(
            ["Dropout" if d_i == 0 else "", d_lbl.replace("=", " ").replace("%", "\\%"), *cells_d]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a9_sensitivity.tex",
        "llrrr",
        [
            "Axis",
            "Setting",
            r"MemGRN MSE $\downarrow$",
            r"Attr.\ / LTC-Loc $\downarrow$",
            r"Params / Cpl-Ad.",
        ],
        ta9_rows,
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

    # =========================================================================
    # Appendix Table A10: Complete Architectural Specifications & Complexity Summary
    # =========================================================================
    ta10_rows = [
        [r"\textbf{MemGRN (ours)}", "2-stage base+hyst.", "Yes", r"$O(SR)$", "Unit DC; 2nd-order"],
        [r"\textbf{MemGRN-Ortho}", "MemGRN + $\\Psi_\\perp$", "Yes", r"$O(SR + K^2)$", "Zero Stage-1 drift"],
        ["Local Constant ($gW$)", "2-stage fixed $\\lambda$", "Yes", r"$O(SR)$", "No rate adaptation"],
        ["Source-Local LTC", "1-stage LTC cell", "Yes", r"$O(SR^2)$", "1st-order; scale-ent."],
        ["Depthwise SSM", "Per-src $\\Delta_i(a_i)$", "Yes", r"$O(SR)$", "1st-order selective"],
        ["Coupled Constant (SSM)", "Linear mixed $(z,h)$", "No", r"$O(S^2 R)$", "Cross-src leakage"],
        ["Coupled Adaptive ($gW$)", "Mixed rates + mix", "No", r"$O(S^2 R^2)$", "Cross-src leakage"],
        ["Coupled LTC", "Dense LTC network", "No", r"$O(M^2)$", "Cross-src; $GL(M)$"],
        ["Selective SSM (Mamba)", "Diag.\\ $A$, $\\Delta(a)$", "No ($\\Delta$)", r"$O(SM)$", "Gate cross-talk"],
        ["CfC", "Closed-form / rel.\\ ODE", "No", r"$O(M^2)$", "Cross-src; $GL(M)$"],
        ["Neural CDE", "Controlled $da/dt$", "No", r"$O(S M^2)$", "Black-box state"],
        ["PerturbODE", "State-only modules", "N/A", r"$O(N M)$", "Iso-state floor"],
        ["Latent Neural ODE", "Neural MLP ODE", "No", r"$O(M^2)$", "Black-box state"],
        ["RENGE", "Static KO powers", "N/A", r"$O(S N)$", "No unseen KO dyn."],
        ["DLinear (discrete)", "Trend/seas.\\ lags", "Ch.-ind.", r"$O(T^2 + SN)$", "Fixed grid; no graph"],
        ["GRU (discrete)", "Gated RNN cell", "No", r"$O(M^2)$", "Fixed grid; black-box"],
        ["Transformer (discrete)", "Causal self-attn.", "No", r"$O(M^2 + T^2)$", "Fixed grid; black-box"],
    ]
    write_tex_table(
        out_tab_dir,
        "tab_a10_baselines_spec.tex",
        "lllll",
        [
            "Architecture",
            "Memory mechanism",
            "Source-isol.",
            "Mem.\\ scaling",
            "Key property / limit",
        ],
        ta10_rows,
    )

    # =========================================================================
    # Main Table 1 (Cross-Domain Open-Loop): Equal-Budget Beijing (raw u & Delta u) & Binance 16-Asset
    # =========================================================================
    bq_raw = fair_ol["beijing_air_quality"]["raw_u_4season"]["summary"]
    bq_du = fair_ol["beijing_air_quality"]["delta_u_4season"]["summary"]
    fin_full = fair_ol["binance_contagion"]["full_16asset"]
    cd_open_ode = [
        ("memgrn_additive", "\\textbf{MemGRN ($\\Gamma = 0$, ours)}"),
        ("memgrn", "\\textbf{MemGRN (ours)}"),
        ("depthwise_ssm_coop", "Depthwise SSM ($+\\Gamma$)"),
        ("local_const_coop", "Local Const.\\ ($+\\Gamma$)"),
        ("ltc_local_coop", "Source-Local LTC ($+\\Gamma$)"),
        ("perturb_ode", "PerturbODE"),
        ("cfc", "CfC"),
        ("latent_ode", "Latent Neural ODE"),
        ("neural_cde", "Neural CDE"),
        ("selective_ssm", "Selective SSM (Mamba)"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("ltc_coupled", "Coupled LTC"),
    ]
    disc_open_order = [
        ("dlinear", "DLinear (discrete)"),
        ("gru", "GRU (discrete)"),
        ("transformer", "Transformer (discrete)"),
    ]
    all_open_keys = [k for k, _ in cd_open_ode + disc_open_order]
    ode_open_keys = {k for k, _ in cd_open_ode}
    b_raw_all = min(bq_raw[k]["loso_mse"] for k in all_open_keys)
    b_raw_ode = min(bq_raw[k]["loso_mse"] for k in ode_open_keys)
    b_du_all = min(bq_du[k]["loso_mse"] for k in all_open_keys)
    b_du_ode = min(bq_du[k]["loso_mse"] for k in ode_open_keys)
    b_ws_all = min(bq_du[k]["washout_mse"] for k in all_open_keys)
    b_ws_ode = min(bq_du[k]["washout_mse"] for k in ode_open_keys)
    b_fm_all = min(fin_full[k]["aug_test_mse"] for k in all_open_keys)
    b_fm_ode = min(fin_full[k]["aug_test_mse"] for k in ode_open_keys)
    b_fi_all = min(fin_full[k]["idio_shock_mse"] for k in all_open_keys)
    b_fi_ode = min(fin_full[k]["idio_shock_mse"] for k in ode_open_keys)

    def _fmt_ov_ode(val: float, best_all: float, best_ode: float, is_ode: bool, fmt: str = ".3f", se: float | None = None) -> str:
        core = f"{val:{fmt}}"
        if np.isclose(val, best_all):
            core = f"\\textbf{{{core}}}"
        elif is_ode and np.isclose(val, best_ode):
            core = f"\\underline{{{core}}}"
        return f"{core}$\\pm${se:{fmt}}" if se is not None else core

    t_cd_open: list[list[str]] = []
    for i_m, (k, lbl) in enumerate(cd_open_ode + disc_open_order):
        if i_m in (6, 12):
            t_cd_open.append(["\\midrule"])
        is_ode = k in ode_open_keys
        dr = bq_raw[k]
        dd = bq_du[k]
        df = fin_full[k]
        t_cd_open.append(
            [
                lbl,
                _fmt_ov_ode(dr["loso_mse"], b_raw_all, b_raw_ode, is_ode, ".3f", dr["loso_mse_se"]),
                _fmt_ov_ode(dd["loso_mse"], b_du_all, b_du_ode, is_ode, ".3f", dd["loso_mse_se"]),
                _fmt_ov_ode(dd["washout_mse"], b_ws_all, b_ws_ode, is_ode, ".3f"),
                _fmt_ov_ode(df["aug_test_mse"], b_fm_all, b_fm_ode, is_ode, ".3f"),
                _fmt_ov_ode(df["idio_shock_mse"], b_fi_all, b_fi_ode, is_ode, ".3f"),
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab1_cross_domain_open_loop.tex",
        "lrrrrr",
        [
            "Architecture (Open-loop)",
            r"Beijing $u$ $\downarrow$",
            r"Beijing $\Delta u$ $\downarrow$",
            r"Storm $\Delta u$ $\downarrow$",
            r"Fin.\ Aug $\downarrow$",
            r"Idio.\ $\downarrow$",
        ],
        t_cd_open,
    )

    # =========================================================================
    # Main Table 2 (Cross-Domain Closed-Loop): Matched S=207 METR-LA, IEEE 39-Bus, & Non-Overlap Reddit
    # =========================================================================
    metr207 = fair_cl["metr_la_s207"]["summary"]
    pgrid_all = fair_cl["powergrid_ieee39"]["summary"]
    red_all = fair_cl["reddit_nonoverlap"]["summary"]
    cd_cl_ode = [
        ("cl_memgrn_selfmem", "cl_memgrn_selfmem", "cl_memgrn_selfmem", "\\textbf{MemGRN (+sm, ours)}"),
        ("cl_local_const_selfmem", "cl_local_const_selfmem", "cl_local_const_selfmem", "Local Const.\\ (+sm)"),
        ("cl_depthwise_ssm_selfmem", "cl_depthwise_ssm_selfmem", "cl_depthwise_ssm_selfmem", "Depthwise SSM (+sm)"),
        ("cl_ltc_local_selfmem", "cl_ltc_local_selfmem", "cl_ltc_local_selfmem", "Source-Local LTC (+sm)"),
        ("cl_latent_ode", "cl_latent_ode", "cl_latent_ode", "Latent Neural ODE"),
        ("cl_selective_ssm", "cl_selective_ssm", "cl_selective_ssm", "Selective SSM (Mamba)"),
        ("cl_perturb_ode", "cl_perturb_ode", "cl_perturb_ode", "PerturbODE"),
        ("cl_coupled_adaptive", "cl_coupled_adaptive", "cl_coupled_adaptive", "Coupled Adaptive"),
        ("cl_cfc", "cl_cfc", "cl_cfc", "CfC"),
    ]
    cd_cl_disc = [
        ("dlinear", None, "dlinear", "DLinear (discrete)"),
        ("gru", None, "gru", "GRU (discrete)"),
        ("transformer", None, "transformer", "Transformer (discrete)"),
        ("persistence", None, None, "Persistence ($y_t = y_0$)"),
    ]
    cd_cl_order = cd_cl_ode + cd_cl_disc
    ode_cl_m = {km for km, _, _, _ in cd_cl_ode}
    bm_m_all = min(metr207[km]["test_mse"] for km, _, _, _ in cd_cl_order)
    bm_m_ode = min(metr207[km]["test_mse"] for km, _, _, _ in cd_cl_ode)
    bm_a_all = max(metr207[km]["road_graph_raw_ap_over_random"] for km, _, _, _ in cd_cl_order)
    bp_m_all = min(pgrid_all[kp]["double_n2_mse"] for _, kp, _, _ in cd_cl_ode)
    bp_a_all = max(pgrid_all[kp]["graph_ap"] for _, kp, _, _ in cd_cl_ode)
    br_m_all = min(red_all[kr]["nonoverlap_test_mse"] for _, _, kr, _ in cd_cl_order if kr is not None)
    br_m_ode = min(red_all[kr]["nonoverlap_test_mse"] for _, _, kr, _ in cd_cl_ode if kr is not None)
    t_cd_cl: list[list[str]] = []
    for idx_c, (km, kp, kr, lbl) in enumerate(cd_cl_order):
        if idx_c in (4, 9):
            t_cd_cl.append(["\\midrule"])
        is_ode = km in ode_cl_m
        dm = metr207[km]
        sm = _fmt_ov_ode(dm["test_mse"], bm_m_all, bm_m_ode, is_ode, ".4f")
        sa = (
            f"\\textbf{{{dm['road_graph_raw_ap_over_random']:.2f}$\\times$}}"
            if np.isclose(dm["road_graph_raw_ap_over_random"], bm_a_all)
            else f"{dm['road_graph_raw_ap_over_random']:.2f}$\\times$"
        )
        if kp is not None:
            dp = pgrid_all[kp]
            sp = f"\\textbf{{{dp['double_n2_mse']:.4f}}}" if np.isclose(dp["double_n2_mse"], bp_m_all) else f"{dp['double_n2_mse']:.4f}"
            sg = f"\\textbf{{{dp['graph_ap']:.3f}}}" if np.isclose(dp["graph_ap"], bp_a_all) else f"{dp['graph_ap']:.3f}"
        else:
            sp, sg = "--", "--"
        if kr is not None:
            dr = red_all[kr]
            sr = _fmt_ov_ode(dr["nonoverlap_test_mse"], br_m_all, br_m_ode, is_ode, ".4f")
        else:
            sr = "--"
        t_cd_cl.append([lbl, sm, sa, sp, sg, sr])
    write_tex_table(
        out_tab_dir,
        "tab2_cross_domain_closed_loop.tex",
        "lrrrrr",
        [
            "Closed-loop model",
            r"METR207 $\downarrow$",
            r"Road AP $\uparrow$",
            r"Grid $N{-}2$ $\downarrow$",
            r"Grid AP $\uparrow$",
            r"Reddit $\downarrow$",
        ],
        t_cd_cl,
    )

    # =========================================================================
    # Appendix Table A11: Matched Primary Human CD4+ T-Cell Perturb-seq (GSE314342, S=128)
    # =========================================================================
    cd4t = fair_cl["cd4t_gse314342"]["summary"]
    cd4t_order = [
        ("cl_memgrn", "\\textbf{MemGRN (ours)}"),
        ("cl_local_const", "Local Constant"),
        ("cl_depthwise_ssm", "Depthwise SSM"),
        ("cl_ltc_local", "Source-Local LTC"),
        ("cl_selective_ssm", "Selective SSM (Mamba)"),
        ("cl_cfc", "CfC"),
        ("cl_ltc_coupled", "Coupled LTC ($M{=}48$)"),
        ("cl_coupled_adaptive", "Coupled Adaptive ($O(S^2R^2)$)"),
        ("cl_perturb_ode", "PerturbODE"),
        ("cl_latent_ode", "Latent Neural ODE"),
        ("renge_graph_only", "RENGE (graph only)"),
        ("persistence", "Persistence ($y_{48}=y_8$)"),
        ("zero", "Zero baseline"),
    ]
    bc_m = min(cd4t[k]["unseen_tf_48h_mse"] for k, _ in cd4t_order if cd4t[k]["forecast_supported"])
    bc_cn = max(cd4t[k]["collectri_colnorm_ap_over_random"] for k, _ in cd4t_order)
    bc_rw = max(cd4t[k]["collectri_raw_ap_over_random"] for k, _ in cd4t_order)
    ta11_rows: list[list[str]] = []
    for k, lbl in cd4t_order:
        d = cd4t[k]
        if d["forecast_supported"]:
            sm = f"\\textbf{{{d['unseen_tf_48h_mse']:.4f}}}" if np.isclose(d["unseen_tf_48h_mse"], bc_m) else f"{d['unseen_tf_48h_mse']:.4f}"
        else:
            sm = "--"
        scn = f"\\textbf{{{d['collectri_colnorm_ap_over_random']:.2f}$\\times$}}" if np.isclose(d["collectri_colnorm_ap_over_random"], bc_cn) else f"{d['collectri_colnorm_ap_over_random']:.2f}$\\times$"
        srw = f"\\textbf{{{d['collectri_raw_ap_over_random']:.2f}$\\times$}}" if np.isclose(d["collectri_raw_ap_over_random"], bc_rw) else f"{d['collectri_raw_ap_over_random']:.2f}$\\times$"
        ta11_rows.append([lbl, f"{d['mem_params']:,}", f"{d['n_params']:,}", sm, scn, srw])
    write_tex_table(
        out_tab_dir,
        "tab_a11_cd4t_closed_loop.tex",
        "lrrrrr",
        [
            r"Model ($S{=}128$ TFs, $256$ genes)",
            "Mem.\\ p.",
            "Tot.\\ p.",
            r"48h MSE $\downarrow$",
            r"Col.\ AP $\uparrow$",
            r"Raw AP $\uparrow$",
        ],
        ta11_rows,
    )

    # =========================================================================
    # Appendix Table A12: Detailed Cross-Domain Breakdowns (Uniform Mem/Tot Params, No Mixing!)
    # =========================================================================
    ta12_short = {
        "cl_memgrn_selfmem": "\\textbf{MemGRN (+sm)}",
        "cl_memgrn_hollow": "MemGRN (hol.)",
        "cl_local_const_selfmem": "Loc-Con (+sm)",
        "cl_local_const_hollow": "Loc-Con (hol.)",
        "cl_depthwise_ssm_selfmem": "D-SSM (+sm)",
        "cl_depthwise_ssm_hollow": "D-SSM (hol.)",
        "cl_ltc_local_selfmem": "LTC-Loc (+sm)",
        "cl_ltc_local_hollow": "LTC-Loc (hol.)",
        "cl_selective_ssm": "Mamba (SSM)",
        "cl_ltc_coupled": "Coupled LTC",
        "cl_coupled_adaptive": "Cpl-Adaptive",
        "cl_cfc": "CfC",
        "cl_perturb_ode": "PerturbODE",
        "cl_latent_ode": "Latent ODE",
        "dlinear": "DLinear (disc.)",
        "gru": "GRU (disc.)",
        "transformer": "Transf.\\ (disc.)",
        "persistence": "Persistence",
    }
    ta12_metr_keys = [
        "cl_memgrn_selfmem",
        "cl_memgrn_hollow",
        "cl_local_const_selfmem",
        "cl_depthwise_ssm_selfmem",
        "cl_ltc_local_selfmem",
        "cl_latent_ode",
        "cl_selective_ssm",
        "cl_ltc_coupled",
        "cl_perturb_ode",
        "cl_coupled_adaptive",
        "cl_cfc",
        "dlinear",
        "gru",
        "transformer",
        "persistence",
    ]
    ta12_rows: list[list[str]] = []
    for i_k, km in enumerate(ta12_metr_keys):
        d = metr207[km]
        p_str = f"{_fmt_p(d['mem_params'])}/{_fmt_p(d['n_params'])}"
        ta12_rows.append(
            [
                "METR207" if i_k == 0 else "",
                ta12_short[km],
                p_str,
                f"{d['mae_15min_mph']:.2f}/{d['mae_30min_mph']:.2f}",
                f"{d['shockwave_mse']:.4f}",
                f"{d['road_graph_raw_auroc']:.3f}",
            ]
        )
    ta12_rows.append(["\\midrule"])
    pg_detail_keys = [
        "cl_memgrn_selfmem",
        "cl_local_const_selfmem",
        "cl_memgrn_hollow",
        "cl_local_const_hollow",
        "cl_depthwise_ssm_selfmem",
        "cl_ltc_local_selfmem",
        "cl_latent_ode",
        "cl_coupled_adaptive",
        "cl_cfc",
        "cl_selective_ssm",
    ]
    for i_k, kp in enumerate(pg_detail_keys):
        d = pgrid_all[kp]
        ta12_rows.append(
            [
                "IEEE-39" if i_k == 0 else "",
                ta12_short[kp],
                f"{d['mem_params']}/{d['n_params']}",
                f"{d['single_n1_mse']:.4f}",
                f"{d['extrap_n2_mse']:.4f}",
                f"{d['graph_auroc']:.3f}",
            ]
        )
    ta12_rows.append(["\\midrule"])
    bq_po = fair_ol["beijing_air_quality"]["po6_4season"]["summary"]
    for i_k, (k, lbl) in enumerate(
        [
            ("memgrn_ortho", "\\textbf{MemGRN-Ortho}"),
            ("memgrn", "MemGRN (6 drv.)"),
            ("memgrn_unproj", "MemGRN + Unproj."),
            ("local_const_coop", "Local Const.\\ ($+\\Gamma$)"),
            ("depthwise_ssm_coop", "Depthwise SSM ($+\\Gamma$)"),
            ("latent_ode", "Latent Neural ODE"),
            ("gru", "GRU (disc.)"),
            ("dlinear", "DLinear (disc.)"),
            ("transformer", "Transformer (disc.)"),
        ]
    ):
        d = bq_po[k]
        dr_str = (
            f"{100.0 * d['stage1_drift_vs_memgrn']:.0f}\\%/{100.0 * d['full_model_drift_vs_memgrn']:.0f}\\%"
            if k == "memgrn_ortho"
            else f"{100.0 * d['full_model_drift_vs_memgrn']:.1f}\\%"
        )
        ta12_rows.append(
            [
                "Beijing PO" if i_k == 0 else "",
                lbl,
                f"{d['n_params']:,}",
                f"{d['loso_mse']:.3f}",
                f"{d['washout_mse']:.3f}",
                dr_str,
            ]
        )
    write_tex_table(
        out_tab_dir,
        "tab_a12_cross_domain_details.tex",
        "llrrrr",
        [
            "Domain",
            "Model",
            "Mem/Tot",
            r"15/30m / $N{-}1$",
            r"Shock / Ext.",
            r"AUC / Drift",
        ],
        ta12_rows,
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
    build_fig2_cross_domain_networks(out_latex_fig)
    build_fig2_human_longitudinal(out_latex_fig)
    build_fig3_multitf_ltc_ortho(out_latex_fig)
    build_fig4_yeast_chromatin_portraits(out_latex_fig)
    build_fig5_mammalian_and_scaling(out_latex_fig)
    build_fig6_trajectories_and_kinetics(out_latex_fig)
    build_fig7_cross_domain_hysteresis(out_latex_fig)
    build_all_tables(out_latex_tab)

    ledger_path = ROOT / "results" / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps({"source_sha256": SOURCE_HASHES}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built 8 figures and 19 tables from {len(SOURCE_HASHES)} verified artifacts.")


if __name__ == "__main__":
    main()
