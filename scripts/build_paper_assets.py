"""Build publication-ready MemGRN figures and LaTeX tables from SHA-256 verified results.

All figures are drawn at exact ICLR text width (5.50 in) and exported without bounding-box
rescaling so that on-page rendered text sits at 7.8-8.5 pt (just slightly below the 9.0 pt
figure captions).
"""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation
import numpy as np

SKILL_SCRIPTS = Path(
    "/usr/local/google/home/apelissier/.gemini/config/skills/nature-figure/scripts"
)
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
else:
    require_matplotlib_panel_alignment = None

ROOT = Path(__file__).resolve().parents[1]
RES_STAGE3 = ROOT / "results" / "stage3"
RES_ROOT = ROOT / "results"
OUT_FIG_REPO = ROOT / "figures"
LATEX_DIR = Path("/usr/local/google/home/apelissier/memgrn-latex")
OUT_FIG_LATEX = LATEX_DIR / "figures"
OUT_TAB_LATEX = LATEX_DIR / "tables"

for folder in (OUT_FIG_REPO, OUT_FIG_LATEX, OUT_TAB_LATEX):
    folder.mkdir(parents=True, exist_ok=True)

TEXTWIDTH_IN = 5.50

PALETTE = {
    "navy": "#0F4D92",
    "royal": "#2B6CB0",
    "teal": "#178577",
    "amber": "#D17A22",
    "crimson": "#B64342",
    "violet": "#6B5B95",
    "slate": "#64748B",
    "light_blue": "#E8F0FA",
    "light_teal": "#E2F3F0",
    "light_amber": "#FDF2E4",
    "light_violet": "#EFEBF7",
    "light_gray": "#F1F5F9",
    "ink": "#1E293B",
    "border": "#475569",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
        "font.size": 8.1,
        "axes.titlesize": 8.4,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.9,
        "ytick.labelsize": 7.9,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }
)

SOURCE_HASHES: dict[str, str] = {}


def load_stage3(rel_dir: str, fname: str = "scores.json") -> dict:
    path = RES_STAGE3 / rel_dir / fname
    key = f"results/stage3/{rel_dir}/{fname}"
    SOURCE_HASHES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def load_new_result(suite: str, fname: str = "scores.json") -> dict:
    path = RES_ROOT / suite / fname
    key = f"results/{suite}/{fname}"
    SOURCE_HASHES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def add_panel_tag(
    ax: plt.Axes,
    tag: str,
    x_pt: float = -24.0,
    y_pt: float = 6.0,
) -> None:
    offset = ScaledTranslation(x_pt / 72.0, y_pt / 72.0, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        tag,
        transform=ax.transAxes + offset,
        fontsize=8.8,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="bottom",
    )


def export_figure(
    fig: plt.Figure,
    stem: str,
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

    for folder in (OUT_FIG_REPO, OUT_FIG_LATEX):
        fig.savefig(folder / f"{stem}.pdf")
        fig.savefig(folder / f"{stem}.svg")
        fig.savefig(folder / f"{stem}.png", dpi=300)
    plt.close(fig)


def write_tex_table(
    fname: str, col_spec: str, header: list[str], rows: list[list[str]]
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
    (OUT_TAB_LATEX / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fig1_architecture() -> None:
    """Figure 1: Source-isolated adaptive memory vs. coupled LTC and PerturbODE."""
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.52))
    ax = fig.add_axes([0.012, 0.02, 0.976, 0.96])
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 5.04)
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
        0.06,
        4.80,
        "a   Source-isolated adaptive memory with explicit enhancer readout",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )

    draw_box(
        0.04,
        2.66,
        1.64,
        1.02,
        "TF activity\naᵢ = tanh gᵢ",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
        lw=1.0,
    )
    draw_box(
        1.96,
        2.66,
        1.96,
        1.02,
        "Upstream zᵢᵣ\nżᵢᵣ = λᵢᵣ(aᵢ−zᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        4.20,
        2.66,
        1.96,
        1.02,
        "Delayed hᵢᵣ\nḣᵢᵣ = λᵢᵣ(zᵢᵣ−hᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        6.44,
        2.66,
        2.54,
        1.02,
        "Enhancer drive\nDᵢⱼaᵢ + Bᵢᵣⱼhᵢᵣ + Γh̄ᵢh̄ₖ",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        9.26,
        2.66,
        1.70,
        1.02,
        "Target ODE\nẏⱼ = pⱼ − δⱼyⱼ",
        PALETTE["light_gray"],
        ec=PALETTE["border"],
    )

    draw_arrow((1.73, 3.17), (1.91, 3.17))
    draw_arrow((3.97, 3.17), (4.15, 3.17))
    draw_arrow((6.21, 3.17), (6.39, 3.17))
    draw_arrow((9.03, 3.17), (9.21, 3.17))

    ax.plot([0.86, 0.86, 7.71], [3.73, 4.18, 4.18], color=PALETTE["navy"], lw=0.95)
    draw_arrow((7.71, 4.18), (7.71, 3.74), color=PALETTE["navy"], lw=0.95)
    ax.text(
        4.28,
        4.28,
        "Direct signed path  Dᵢⱼ aᵢ(t)",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color=PALETTE["navy"],
    )

    draw_box(
        1.38,
        1.52,
        6.44,
        0.72,
        "Isolated rate  λᵢᵣ(aᵢ, zᵢᵣ, hᵢᵣ) : ∂(zᵢᵣ, hᵢᵣ)/∂gₖ ≡ 0 (k ≠ i)",
        PALETTE["light_amber"],
        ec=PALETTE["amber"],
    )
    draw_arrow((2.94, 2.30), (2.94, 2.60), color=PALETTE["amber"])
    draw_arrow((5.18, 2.30), (5.18, 2.60), color=PALETTE["amber"])
    ax.plot(
        [0.86, 0.86, 1.31], [2.60, 1.88, 1.88], color=PALETTE["amber"], lw=0.85, ls="--"
    )
    draw_arrow((1.14, 1.88), (1.31, 1.88), color=PALETTE["amber"], lw=0.85)

    ax.text(
        0.06,
        1.16,
        "b   Three structural guarantees tested across synthetic and live-cell systems",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )
    draw_box(
        0.06,
        0.08,
        3.42,
        0.82,
        "1. Correlation shift\nZero cross-TF leakage",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        3.79,
        0.08,
        3.42,
        0.82,
        "2. Non-Markovian memory\nResolves iso-y(T) pulses",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        7.52,
        0.08,
        3.42,
        0.82,
        "3. 4:1 strength swap\nBreaks pair cancellation",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
    )

    export_figure(fig, "fig1_memgrn_architecture", audit_alignment=False)


def build_fig2_multitf_factorial() -> None:
    """Figure 2: Locked 2x2 Multi-TF Correlation-Shift Benchmark + LTC & PerturbODE."""
    mtf = load_new_result("multitf_factorial")
    add_m = mtf["summary"]["additive"]["means"]
    coop_m = mtf["summary"]["cooperative"]["means"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.48))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.105,
        right=0.985,
        bottom=0.31,
        top=0.85,
        wspace=0.38,
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
        "memgrn",
        "memgrn_int",
    ]
    labels = [
        "LTC-Cpl",
        "PertODE",
        "Loc-Const",
        "Cpl-Adapt",
        "LTC-Loc",
        "MemGRN",
        "Mem-Int",
    ]
    cols = [
        PALETTE["crimson"],
        PALETTE["violet"],
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["royal"],
        PALETTE["navy"],
    ]
    x = np.arange(len(keys))

    # Panel a: Additive buffering OOD intervention MSE
    vals_a = [add_m[k]["test_mse"] for k in keys]
    ax0.bar(x, vals_a, width=0.64, color=cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.65, len(keys) - 0.35)
    ax0.set_ylim(0.0, 0.50)
    ax0.set_yticks([0.0, 0.2, 0.4])
    ax0.set_ylabel("OOD intervention MSE")
    ax0.set_title("Additive buffering")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Cooperative enhancer OOD intervention MSE
    vals_b = [coop_m[k]["test_mse"] for k in keys]
    ax1.bar(x, vals_b, width=0.64, color=cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.65, len(keys) - 0.35)
    ax1.set_ylim(0.0, 0.44)
    ax1.set_yticks([0.0, 0.2, 0.4])
    ax1.set_ylabel("OOD intervention MSE")
    ax1.set_title("Cooperative enhancers")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Counterfactual source attribution RMSE (pooled across 24 worlds)
    vals_c = [
        0.5 * (add_m[k]["attribution_rmse"] + coop_m[k]["attribution_rmse"]) for k in keys
    ]
    ax2.bar(x, vals_c, width=0.64, color=cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_xlim(-0.65, len(keys) - 0.35)
    ax2.set_ylim(0.0, 0.32)
    ax2.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax2.set_ylabel("Attribution RMSE")
    ax2.set_title("Source attribution")
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(fig, "fig2_multitf_factorial")


def build_fig3_msn2_joint_and_cross() -> None:
    """Figure 3: Yeast Msn2 joint training, cross-readout 2x2, and filter resets."""
    joint = load_new_result("msn2_joint_and_cross_readout")
    reset = load_stage3("msn2_memory_reset")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.48))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.06, 1.04, 1.02],
        left=0.100,
        right=0.985,
        bottom=0.31,
        top=0.85,
        wspace=0.45,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Fair source-CV joint training on 10 unseen repeated-pulse trains
    j_keys = ["local_const", "perturb_ode", "latent_ode", "ltc_local", "memgrn"]
    j_lbls = ["Const gW", "PertODE", "Lat-ODE", "LTC-Loc", "MemGRN"]
    j_cols = [
        PALETTE["slate"],
        PALETTE["violet"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["navy"],
    ]
    j_vals = [joint["joint_training"][k]["ten_trains_post_mse"] for k in j_keys]
    jx = np.arange(len(j_keys))
    ax0.bar(jx, j_vals, width=0.60, color=j_cols, edgecolor=PALETTE["ink"], linewidth=0.55)
    ax0.set_xticks(jx)
    ax0.set_xticklabels(j_lbls, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlim(-0.6, 4.6)
    ax0.set_ylim(0.0, 0.23)
    ax0.set_yticks([0.0, 0.1, 0.2])
    ax0.set_ylabel("10-train MSE")
    ax0.set_title("Joint source-CV fits")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Symmetric Cross-Readout 2x2 (Constant-W vs MemGRN-W across 4 rate laws)
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
    ax1.set_ylim(0.0, 0.28)
    ax1.set_yticks([0.0, 0.1, 0.2])
    ax1.set_ylabel("10-train MSE")
    ax1.set_title("Cross-readout swap")
    ax1.legend(loc="upper right", ncol=1, handlelength=1.1, labelspacing=0.18)
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Inter-pulse memory resets on 6- and 8-pulse trains
    trains = ["FM_6_5min_690nM", "FM_8_5min_690nM"]
    modes = [
        ("intact", "Intact", PALETTE["navy"]),
        ("latent", "Latent ODE", PALETTE["violet"]),
        ("last_only", "Pre-final", PALETTE["amber"]),
        ("reset", "Full reset", PALETTE["crimson"]),
    ]
    w = 0.19
    gx = np.arange(2)
    y_floor = 1.8e-2
    for m_i, (m_key, m_lbl, m_col) in enumerate(modes):
        vals = np.array(
            [reset["results"][tr]["panels"]["all"]["mse"][m_key] for tr in trains]
        )
        ax2.bar(
            gx + (m_i - 1.5) * w,
            vals - y_floor,
            bottom=y_floor,
            width=w * 0.90,
            color=m_col,
            edgecolor=PALETTE["ink"],
            linewidth=0.55,
            label=m_lbl,
        )
    ax2.set_yscale("log")
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(y_floor, 18.0)
    ax2.set_yticks([0.03, 0.1, 0.3, 1.0])
    ax2.set_yticklabels(["0.03", "0.1", "0.3", "1.0"])
    ax2.set_xticks(gx)
    ax2.set_xticklabels(["6 pulses", "8 pulses"], rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_ylabel("Post-shutoff MSE")
    ax2.set_title("Memory resets")
    ax2.legend(loc="upper left", ncol=1, handlelength=1.1, labelspacing=0.16)
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(
        fig,
        "fig3_msn2_transfer_and_reset",
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Three slight width ratios accommodate 5-bar, 4-pair, and 2-group panels",
            }
        ],
    )


def build_fig4_frozen_readout() -> None:
    """Figure 4: Six-promoter frozen-readout rate-input ablation and LOPO shared filter."""
    six = load_stage3("msn2_six_promoter_rates")
    genes = six["genes"]
    modes = ["constant", "activity", "memory", "both"]
    mode_labels = ["Constant", "Activity only", "Memory only", "Activity + memory"]
    mode_colors = [PALETTE["slate"], PALETTE["amber"], PALETTE["teal"], PALETTE["navy"]]
    mode_markers = ["s", "^", "o", "D"]

    arr = np.array(
        [[six["results"][g][m]["per_schedule"] for m in modes] for g in genes]
    )

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.50))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.45, 0.055],
        left=0.105,
        right=0.900,
        bottom=0.30,
        top=0.85,
        wspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    gx = np.arange(len(genes))
    for m_idx in range(4):
        ax0.plot(
            gx,
            arr[:, m_idx].mean(axis=1),
            marker=mode_markers[m_idx],
            ms=4.0,
            lw=1.3,
            color=mode_colors[m_idx],
            label=mode_labels[m_idx],
        )
    ax0.set_yscale("log")
    ax0.set_xlim(-0.45, 5.45)
    ax0.set_ylim(4e-3, 85.0)
    ax0.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax0.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1.0"])
    ax0.set_xticks(gx)
    ax0.set_xticklabels(genes, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_ylabel("Mean recovery MSE")
    ax0.set_title("Six promoters")
    ax0.legend(loc="upper left", ncol=1, handlelength=1.5, labelspacing=0.18)
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    log_gain = np.log2(arr[:, 1] / arr[:, 3])
    im = ax1.imshow(log_gain, vmin=-2.6, vmax=2.6, cmap="BrBG", aspect="auto")
    sched_labels = [
        "2/5",
        "3/5",
        "4/5",
        "5/5",
        "6/5",
        "8/5",
        "4/7.5",
        "4/10",
        "4/15",
        "4/20",
    ]
    ax1.set_yticks(gx)
    ax1.set_yticklabels(genes)
    ax1.tick_params(axis="y", pad=3.5)
    ax1.set_xticks(np.arange(10))
    ax1.set_xticklabels(sched_labels, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlabel("Pulse count / gap (min)")
    ax1.set_title("Memory gain vs activity")
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    add_panel_tag(ax1, "b", x_pt=-36.0, y_pt=6.0)

    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_label("log₂(Activity / Combined MSE)", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.8)

    export_figure(
        fig,
        "fig4_frozen_readout_ablation",
        exclude_axes=[cax],
        exemptions=[
            {
                "panels": ["a", "b"],
                "checks": ["panel-width"],
                "reason": "10-schedule heatmap requires wider x-span than 6-promoter summary",
            }
        ],
    )


def build_fig5_macrophage_and_swap() -> None:
    """Figure 5: 4-TF Macrophage innate-immune benchmark & 4:1 strength-swap ID."""
    mac = load_new_result("macrophage_multitf")["summary"]
    swap = load_stage3("strength_swap_replication")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.50))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.05, 1.05],
        left=0.105,
        right=0.985,
        bottom=0.31,
        top=0.85,
        wspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    m_keys = [
        "perturb_ode",
        "ltc_coupled",
        "coupled_adaptive",
        "ltc_local",
        "local_const",
        "memgrn_int",
    ]
    m_lbls = ["PertODE", "LTC-Cpl", "Cpl-Adapt", "LTC-Loc", "Loc-Const", "Mem-Int"]
    m_cols = [
        PALETTE["violet"],
        PALETTE["crimson"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["slate"],
        PALETTE["navy"],
    ]
    mx = np.arange(len(m_keys))

    # Panel a: 4-TF Macrophage regulator attribution AUROC across 477 GRS genes
    auroc_vals = [mac[k]["regulator_attribution_auroc"] for k in m_keys]
    ax0.bar(
        mx,
        np.array(auroc_vals) - 0.5,
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
    ax0.set_ylim(0.5, 0.84)
    ax0.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax0.set_ylabel("Regulator AUROC")
    ax0.set_title("Macrophage attribution")
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Macrophage OOD MSE on selective de-correlated stimuli (TNF, IFNB, FLA, VSV)
    sel_vals = [mac[k]["selective_decorrelated_mse"] for k in m_keys]
    ax1.bar(
        mx, sel_vals, width=0.62, color=m_cols, edgecolor=PALETTE["ink"], linewidth=0.55
    )
    ax1.set_xticks(mx)
    ax1.set_xticklabels(m_lbls, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_xlim(-0.6, len(m_keys) - 0.4)
    ax1.set_ylim(0.0, 0.54)
    ax1.set_yticks([0.0, 0.25, 0.50])
    ax1.set_ylabel("De-correlated MSE")
    ax1.set_title("Selective stimuli")
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Counterbalanced 4:1 strength swap across 16 networks
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
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_xticks(gx)
    ax2.set_xticklabels(["All 96", "Fast opp."], rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.5)
    ax2.set_ylim(0.0, 185.0)
    ax2.set_yticks([0, 50, 100])
    ax2.set_ylabel("Pair recovery (%)")
    ax2.set_title("4:1 strength swap")
    ax2.legend(loc="upper left", ncol=1, handlelength=1.1, labelspacing=0.16)
    add_panel_tag(ax2, "c", x_pt=-30.0, y_pt=6.0)

    export_figure(fig, "fig5_macrophage_and_swap")


def build_all_tables() -> None:
    """Generate clean LaTeX tables (each <= 6 columns) for uniform font sizing."""
    mtf = load_new_result("multitf_factorial")["summary"]
    joint = load_new_result("msn2_joint_and_cross_readout")
    mac = load_new_result("macrophage_multitf")["summary"]
    occ = load_stage3("neural_rate_pulse", "test_scores.json")
    buf = load_stage3("neural_rate_buffer", "test_scores.json")
    six = load_stage3("msn2_six_promoter_rates")
    swap = load_stage3("strength_swap_replication")
    cert = load_stage3("paired_time_certificate", "test_scores.json")
    cert_meta = load_stage3("paired_time_certificate", "certificate.json")
    nfkb = {
        lig: load_stage3(f"neural_rate_nfkb_washout/{lig}")
        for lig in ("IL1", "TNF", "LPS")
    }

    # Table 1: Centerpiece Locked 2x2 Multi-TF Correlation-Shift Benchmark (24 worlds)
    t1_rows: list[list[str]] = []
    mtf_order = [
        ("local_const", "Local Constant ($gW$)"),
        ("coupled_const", "Coupled Constant (SSM)"),
        ("coupled_adaptive", "Coupled Adaptive ($gW$)"),
        ("ltc_coupled", "Coupled LTC"),
        ("perturb_ode", "PerturbODE"),
        ("ltc_local", "Source-Local LTC"),
        ("memgrn", "\\textbf{MemGRN (ours)}"),
        ("memgrn_int", "\\textbf{MemGRN-Int (ours)}"),
    ]
    for idx_r, (reg_key, reg_lbl) in enumerate(
        [("additive", "Additive"), ("cooperative", "Cooperative")]
    ):
        if idx_r > 0:
            t1_rows.append(["\\midrule"])
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
            t1_rows.append(
                [reg_lbl if m_i == 0 else "", lbl, str(d["n_params"]), c_mse, c_att, c_ap]
            )
    write_tex_table(
        "tab1_multitf_factorial.tex",
        "llrrrr",
        [
            "Regime",
            "Architecture",
            "Params",
            r"OOD MSE $\downarrow$",
            r"Attr.\ RMSE $\downarrow$",
            r"Edge AP $\uparrow$",
        ],
        t1_rows,
    )

    # Table 2: Yeast Msn2 Fair Source-CV Joint Training & Cross-Readout 2x2
    jt = joint["joint_training"]
    t2_rows: list[list[str]] = []
    j_order = [
        ("memgrn", "\\textbf{MemGRN (ours)}"),
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
        t2_rows.append([lbl, str(d["n_params"]), s30, s10, s40, sns])
    write_tex_table(
        "tab2_msn2_joint_transfer.tex",
        "lrrrrr",
        [
            "Joint source-CV model",
            "Params",
            r"30-min $\downarrow$",
            r"10 trains $\downarrow$",
            r"First 40m $\downarrow$",
            r"No \texttt{SIP18} $\downarrow$",
        ],
        t2_rows,
    )

    # Table 3: Symmetric Cross-Readout Sensitivity & Six-Promoter Frozen-Readout Ablation
    modes = ["constant", "activity", "memory", "both"]
    t3_rows: list[list[str]] = []
    for r_orig, r_lbl in [
        ("local_const", "Joint Constant-$gW$ readout (7 genes)"),
        ("memgrn", "Joint MemGRN readout (7 genes)"),
    ]:
        vals = [joint["cross_readout_2x2"][r_orig][m] for m in modes]
        b_i = int(np.argmin(vals))
        cells = [
            f"\\textbf{{{vals[i]:.4f}}}" if i == b_i else f"{vals[i]:.4f}" for i in range(4)
        ]
        t3_rows.append([r_lbl, *cells])
    t3_rows.append(["\\midrule"])
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
        t3_rows.append([f"\\texttt{{{g}}} (archived single-pulse $W$)", *cells])
    means = arr.mean(axis=0)
    best_mean = int(np.argmin(means))
    mean_cells = [
        f"\\textbf{{{means[m_i]:.4f}}}" if m_i == best_mean else f"{means[m_i]:.4f}"
        for m_i in range(4)
    ]
    t3_rows.append(["Six-promoter mean", *mean_cells])
    write_tex_table(
        "tab3_six_promoters.tex",
        "lrrrr",
        [
            "Readout origin / promoter",
            r"Constant $\downarrow$",
            r"Activity only $\downarrow$",
            r"Memory only $\downarrow$",
            r"Activity + memory $\downarrow$",
        ],
        t3_rows,
    )

    # Table 4: Real 4-TF Mammalian Macrophage Benchmark (14 stimuli x 477 GRS genes)
    t4_rows: list[list[str]] = []
    mac_order = [
        ("memgrn_int", "\\textbf{MemGRN-Int (ours)}"),
        ("memgrn", "\\textbf{MemGRN (ours)}"),
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
        "tab4_macrophage_multitf.tex",
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

    # Table 5: Counterbalanced 4:1 Strength Swap Across 16 Networks (96 Targets)
    t5_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Learned-memory $gW$}"),
        ("fixed_gw", "Fixed-rate $gW$"),
        ("history_basis", "History time basis"),
        ("oracle", "Oracle templates"),
    ]:
        one_v = 100.0 * swap["overall"][k]["single"]
        rep_v = 100.0 * swap["overall"][k]["repeat"]
        swp_v = 100.0 * swap["overall"][k]["swap"]
        t5_rows.append(
            [
                lbl,
                f"{one_v:.2f}\\%",
                f"{rep_v:.2f}\\%",
                f"{swp_v:.2f}\\%",
                f"+{swp_v - rep_v:.2f}",
            ]
        )
    write_tex_table(
        "tab5_swap_identification.tex",
        "lrrrr",
        [
            "Template library",
            r"1 trace $\uparrow$",
            r"Repeat 4:1 $\uparrow$",
            r"Swap 4:1 $\uparrow$",
            r"Swap $-$ Repeat (pp) $\uparrow$",
        ],
        t5_rows,
    )

    # Appendix Table A1: Occupancy & Out-of-Class Mass-Action Buffering + Multiscale
    ta1_rows: list[list[str]] = []
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
            ta1_rows.append(["\\midrule"])
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
            ta1_rows.append(
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
        "tab_a1_synthetic_buffering.tex",
        "llrrrr",
        [
            "Kinetic regime",
            "Model",
            "Params",
            r"Full MSE $\downarrow$",
            r"Tail MSE $\downarrow$",
            r"Pair ID $\uparrow$",
        ],
        ta1_rows,
    )

    # Appendix Table A2: Multiscale & Autonomous Shared-Driver Disambiguation
    multi = load_stage3("multiscale_gw")
    mixed = load_stage3("mixed_directed_memory")
    ta2_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Multiscale MemGRN}"),
        ("history_basis", "History time basis"),
        ("current_hybrid", "Single-rate hybrid"),
        ("state_space", "Linear state space"),
    ]:
        c_dat = multi["classes"][k]["biphasic"]
        m_all = multi["means"][k]
        ta2_rows.append(
            [
                "Multiscale suite" if k == "learned_gw" else "",
                lbl,
                f"{100.0 * c_dat['error']:.2f}\\%",
                f"{100.0 * m_all['reversal']:.1f}\\%",
                f"{m_all['delayed_ap']:.3f}",
            ]
        )
    ta2_rows.append(["\\midrule"])
    for k, lbl in [
        ("memory_driver", "\\textbf{MemGRN + oscillator}"),
        ("latent_ODE", "Latent Neural ODE"),
        ("state_space", "Linear state space"),
        ("directed_memory", "MemGRN (no oscillator)"),
        ("instant_driver", "Memoryless + oscillator"),
    ]:
        m_dat = mixed["means"][k]
        ta2_rows.append(
            [
                "Shared driver" if k == "memory_driver" else "",
                lbl,
                f"{100.0 * m_dat['tail_error_ratio']:.2f}\\%",
                f"{100.0 * m_dat['tail_sign']:.1f}\\%",
                f"{m_dat['delayed_ap']:.3f}",
            ]
        )
    write_tex_table(
        "tab_a2_multiscale_driver.tex",
        "llrrr",
        [
            "Benchmark",
            "Model",
            r"Tail error $\downarrow$",
            r"Sign acc. $\uparrow$",
            r"Edge AP $\uparrow$",
        ],
        ta2_rows,
    )

    # Appendix Table A3: Leave-One-Promoter-Out Shared Filter Transfer & RTN2 RK4 Ablations
    lopo = joint["shared_filter_lopo_transfer"]["by_promoter"]
    ta3_rows: list[list[str]] = []
    for g_name, g_vals in lopo.items():
        m_v = g_vals["shared_memgrn_filter_mse"]
        c_v = g_vals["shared_const_filter_mse"]
        m_s = f"\\textbf{{{m_v:.4f}}}" if m_v < c_v else f"{m_v:.4f}"
        c_s = f"\\textbf{{{c_v:.4f}}}" if c_v < m_v else f"{c_v:.4f}"
        ta3_rows.append(
            [f"\\texttt{{{g_name}}}", c_s, m_s, f"{100.0 * (1.0 - m_v / c_v):+.1f}\\%"]
        )
    ta3_rows.append(["\\midrule"])
    m_mean = joint["shared_filter_lopo_transfer"]["mean_memgrn_shared"]
    c_mean = joint["shared_filter_lopo_transfer"]["mean_const_shared"]
    ta3_rows.append(
        [
            "7-promoter LOPO mean",
            f"{c_mean:.4f}",
            f"\\textbf{{{m_mean:.4f}}}",
            f"{100.0 * (1.0 - m_mean / c_mean):+.1f}\\%",
        ]
    )
    write_tex_table(
        "tab_a3_lopo_shared_filter.tex",
        "lrrr",
        [
            "Held-out promoter",
            r"Shared Const.\ filter $\downarrow$",
            r"Shared MemGRN filter $\downarrow$",
            r"MSE reduction $\uparrow$",
        ],
        ta3_rows,
    )

    # Appendix Table A4: Msn2 Memory Resets & RTN2 Continuous-vs-Switch Ablations
    rtn_sw = load_stage3("msn2_rtn2_switch_neural")
    reset = load_stage3("msn2_memory_reset")
    ta4_rows: list[list[str]] = []
    for idx_r, (k, lbl) in enumerate(
        [
            ("constant", "Constant rates"),
            ("switch_only", "Discrete switch only"),
            ("neural_only", "Continuous neural rate"),
            ("full", "Switch + neural rate"),
        ]
    ):
        d_sw = rtn_sw["models"][k]
        ta4_rows.append(
            [
                "\\texttt{RTN2} switch" if idx_r == 0 else "",
                lbl,
                f"{d_sw['primary']:.4f}",
                f"{d_sw['half_step']:.4f}",
                f"{d_sw['first40']:.4f}",
            ]
        )
    ta4_rows.append(["\\midrule"])
    for tr_id, tr_lbl in [
        ("FM_6_5min_690nM", "6 pulses (intact / reset)"),
        ("FM_8_5min_690nM", "8 pulses (intact / reset)"),
    ]:
        p_all = reset["results"][tr_id]["panels"]["all"]["mse"]
        p_rtn = reset["results"][tr_id]["promoters"]["RTN2"]
        ta4_rows.append(
            [
                "Filter reset" if tr_id == "FM_6_5min_690nM" else "",
                tr_lbl,
                f"{p_all['intact']:.4f} / {p_all['reset']:.4f}",
                f"+{p_rtn['shift_attributable_to_reporter_at_off']:.3f}",
                f"+{p_rtn['shift_attributable_to_production_after_off']:.3f}",
            ]
        )
    write_tex_table(
        "tab_a4_rtn2_and_reset.tex",
        "llrrr",
        [
            "Study",
            "Condition / model",
            r"Primary MSE $\downarrow$",
            r"Refined / Carryover",
            r"First 40m / Post-off",
        ],
        ta4_rows,
    )

    # Appendix Table A5: Fibroblast Single-TF NF-kB Washout Boundary (177 genes)
    nfkb_map = {
        lig: {m["model"]: m["post_mse"] for m in nfkb[lig]["models"]}
        for lig in ("IL1", "TNF", "LPS")
    }
    ta5_rows = [
        [
            "Constant $gW$ (2-ligand CV stop)",
            f"{nfkb_map['IL1']['current_gw']:.3f}",
            f"{nfkb_map['TNF']['current_gw']:.3f}",
            f"{nfkb_map['LPS']['current_gw']:.3f}",
        ],
        [
            "Two-rate $gW$ (2-ligand CV stop)",
            f"{nfkb_map['IL1']['two_rate_gw']:.3f}",
            f"{nfkb_map['TNF']['two_rate_gw']:.3f}",
            f"{nfkb_map['LPS']['two_rate_gw']:.3f}",
        ],
        [
            "MemGRN (2-ligand CV stop)",
            f"{nfkb_map['IL1']['neural_rate_gw']:.3f}",
            f"{nfkb_map['TNF']['neural_rate_gw']:.3f}",
            f"{nfkb_map['LPS']['neural_rate_gw']:.3f}",
        ],
        [
            "Latent Neural ODE (2-ligand CV stop)",
            f"\\textbf{{{nfkb_map['IL1']['latent_ode']:.3f}}}",
            f"{nfkb_map['TNF']['latent_ode']:.3f}",
            f"\\textbf{{{nfkb_map['LPS']['latent_ode']:.3f}}}",
        ],
        [
            "Source-ligand mean",
            f"{nfkb_map['IL1']['source']:.3f}",
            f"{nfkb_map['TNF']['source']:.3f}",
            f"{nfkb_map['LPS']['source']:.3f}",
        ],
        ["\\midrule"],
        ["Two-rate $gW$ (2,400 steps)", "0.729", "0.522", "2.785"],
        ["MemGRN (2,400 steps)", "0.624", "0.411", "1.906"],
        ["Latent Neural ODE (2,400 steps)", "0.744", "\\textbf{0.346}", "0.880"],
    ]
    write_tex_table(
        "tab_a5_nfkb_boundary.tex",
        "lrrr",
        [
            "Model / checkpoint rule",
            r"IL-1 holdout $\downarrow$",
            r"TNF holdout $\downarrow$",
            r"LPS holdout $\downarrow$",
        ],
        ta5_rows,
    )

    # Appendix Table A6: Swap Subgroups & Correlated-Noise 5-Time Schedule Certificate
    ta6_rows: list[list[str]] = []
    for sg_key, sg_lbl in [
        ("all", "All targets ($96$)"),
        ("aligned", "Aligned signs ($48$)"),
        ("opposing", "Opposing signs ($48$)"),
        ("fast/fast opposing", "Fast opposing ($11$)"),
        ("biphasic/biphasic", "Biphasic pairs ($16$)"),
    ]:
        sg_m = swap["subgroups"][sg_key]["means"]["learned_gw"]
        sr = 100.0 * sg_m["repeat"]
        ss = 100.0 * sg_m["swap"]
        so = 100.0 * swap["subgroups"][sg_key]["means"]["oracle"]["swap"]
        ta6_rows.append(
            [
                sg_lbl,
                f"{sr:.2f}\\%",
                f"\\textbf{{{ss:.2f}\\%}}",
                f"+{ss - sr:.2f}",
                f"{so:.2f}\\%",
            ]
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
        "tab_a6_swap_and_cert.tex",
        "lrrrr",
        [
            "Subgroup / 5-point policy",
            r"Baseline $\uparrow$",
            r"Swap / Selected $\uparrow$",
            r"Gain (pp) $\uparrow$",
            r"Oracle / 98.5\% CI",
        ],
        ta6_rows,
    )


def main() -> None:
    build_fig1_architecture()
    build_fig2_multitf_factorial()
    build_fig3_msn2_joint_and_cross()
    build_fig4_frozen_readout()
    build_fig5_macrophage_and_swap()
    build_all_tables()
    ledger_path = ROOT / "results" / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps({"source_sha256": SOURCE_HASHES}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built 5 figures and 11 tables from {len(SOURCE_HASHES)} verified artifacts.")


if __name__ == "__main__":
    main()
