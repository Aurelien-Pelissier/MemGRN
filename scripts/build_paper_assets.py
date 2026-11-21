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
RES = ROOT / "results" / "stage3"
OUT_FIG_REPO = ROOT / "figures"
LATEX_DIR = Path("/usr/local/google/home/apelissier/memgrn-latex")
OUT_FIG_LATEX = LATEX_DIR / "figures"
OUT_TAB_LATEX = LATEX_DIR / "tables"

for folder in (OUT_FIG_REPO, OUT_FIG_LATEX, OUT_TAB_LATEX):
    folder.mkdir(parents=True, exist_ok=True)

# Exact ICLR \textwidth = 5.50 true in (396.0 PDF pt).
# Captions (\small) are 9.0 pt; figure labels are 7.8-8.5 pt at 1:1 scale.
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
        "legend.fontsize": 7.9,
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


def load_score(rel_dir: str, fname: str = "scores.json") -> dict:
    path = RES / rel_dir / fname
    key = f"results/stage3/{rel_dir}/{fname}"
    SOURCE_HASHES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def add_panel_tag(
    ax: plt.Axes,
    tag: str,
    x_pt: float = -24.0,
    y_pt: float = 6.0,
) -> None:
    """Place bold lowercase panel label with fixed physical point offset."""
    offset = ScaledTranslation(
        x_pt / 72.0, y_pt / 72.0, ax.figure.dpi_scale_trans
    )
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
    """Export figure at exact 5.50-in canvas width (zero bbox rescaling) to both repos."""
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


def write_tex_table(fname: str, col_spec: str, header: list[str], rows: list[list[str]]) -> None:
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
    """Figure 1: Modular architecture and three separable diagnostic protocols."""
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
        weight: str = "normal",
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
            fontweight=weight,
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

    # Top row: Source-local MemGRN architecture
    ax.text(
        0.06,
        4.80,
        "a   Source-local adaptive memory with explicit regulatory readout",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )

    draw_box(
        0.06,
        2.66,
        1.74,
        1.02,
        "TF activity\naᵢ = tanh gᵢ",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
        lw=1.0,
    )
    draw_box(
        2.12,
        2.66,
        2.02,
        1.02,
        "Upstream zᵢᵣ\nżᵢᵣ = λᵢᵣ(aᵢ−zᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        4.46,
        2.66,
        2.02,
        1.02,
        "Delayed hᵢᵣ\nḣᵢᵣ = λᵢᵣ(zᵢᵣ−hᵢᵣ)",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        6.84,
        2.66,
        2.08,
        1.02,
        "Explicit drive\nDᵢⱼaᵢ + ∑ᵣBᵢᵣⱼhᵢᵣ",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        9.22,
        2.66,
        1.72,
        1.02,
        "Target ODE\nẏⱼ = pⱼ − δⱼyⱼ",
        PALETTE["light_gray"],
        ec=PALETTE["border"],
    )

    draw_arrow((1.85, 3.17), (2.07, 3.17))
    draw_arrow((4.19, 3.17), (4.41, 3.17))
    draw_arrow((6.53, 3.17), (6.79, 3.17))
    draw_arrow((8.97, 3.17), (9.17, 3.17))

    # Direct bypass path a_i(t) * D_{ij}
    ax.plot(
        [0.93, 0.93, 7.88],
        [3.73, 4.18, 4.18],
        color=PALETTE["navy"],
        lw=0.95,
    )
    draw_arrow((7.88, 4.18), (7.88, 3.74), color=PALETTE["navy"], lw=0.95)
    ax.text(
        4.40,
        4.28,
        "Direct signed path  Dᵢⱼ aᵢ(t)",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color=PALETTE["navy"],
    )

    # Bounded neural rate law below filters
    draw_box(
        1.68,
        1.52,
        5.50,
        0.72,
        "Local rate  λᵢᵣ(aᵢ, zᵢᵣ, hᵢᵣ) ∈ [λₘᵢₙ, λₘₐₓ]",
        PALETTE["light_amber"],
        ec=PALETTE["amber"],
    )
    draw_arrow((3.13, 2.30), (3.13, 2.60), color=PALETTE["amber"])
    draw_arrow((5.47, 2.30), (5.47, 2.60), color=PALETTE["amber"])
    ax.plot([0.93, 0.93, 1.61], [2.60, 1.88, 1.88], color=PALETTE["amber"], lw=0.85, ls="--")
    draw_arrow((1.42, 1.88), (1.61, 1.88), color=PALETTE["amber"], lw=0.85)

    # Bottom row: Three separable evaluations
    ax.text(
        0.06,
        1.16,
        "b   Separable mechanistic and identifiability tests",
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
        "1. Frozen readout\nFix (D, B, δ); vary ξᵢᵣ",
        PALETTE["light_teal"],
        ec=PALETTE["teal"],
    )
    draw_box(
        3.79,
        0.08,
        3.42,
        0.82,
        "2. Splice accounting\nCarryover vs post-off",
        PALETTE["light_violet"],
        ec=PALETTE["violet"],
    )
    draw_box(
        7.52,
        0.08,
        3.42,
        0.82,
        "3. 4:1 strength swap\nBreak pair symmetry",
        PALETTE["light_blue"],
        ec=PALETTE["navy"],
    )

    export_figure(fig, "fig1_memgrn_architecture", audit_alignment=False)


def build_fig2_synthetic() -> None:
    """Figure 2: Synthetic misspecified kinetics, buffer efficiency, and multiscale reversals."""
    occ = load_score("neural_rate_pulse", "test_scores.json")
    buf = load_score("neural_rate_buffer", "test_scores.json")
    multi = load_score("multiscale_gw", "scores.json")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.38))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.165,
        right=0.982,
        bottom=0.29,
        top=0.85,
        wspace=0.44,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Paired tail-MSE reduction (%) vs Two-rate gW with simultaneous 95% CIs
    regimes = ["Occupancy\n(n = 8)", "Buffer\n(n = 12)", "Control\n(n = 12)"]
    comps = [occ["comparisons"][1][1], buf["comparisons"][1][1], buf["comparisons"][0][1]]
    colors_a = [PALETTE["royal"], PALETTE["navy"], PALETTE["slate"]]
    y_pos = np.arange(3)
    for idx, comp in enumerate(comps):
        eff = 100.0 * (comp["effect"] - 1.0)
        lo = 100.0 * (comp["ci"][0] - 1.0)
        hi = 100.0 * (comp["ci"][1] - 1.0)
        ax0.errorbar(
            eff,
            idx,
            xerr=[[eff - lo], [hi - eff]],
            fmt="o",
            ms=4.8,
            lw=1.3,
            capsize=2.8,
            color=colors_a[idx],
        )
    ax0.axvline(0.0, color=PALETTE["border"], ls=":", lw=0.9)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(regimes)
    ax0.set_ylim(2.45, -0.45)
    ax0.set_xlim(-102.0, 24.0)
    ax0.set_xticks([-80, -40, 0])
    ax0.set_xlabel("Tail MSE change (%)")
    ax0.set_title("Paired kinetic gain")
    add_panel_tag(ax0, "a", x_pt=-48.0, y_pt=6.0)

    # Panel b: Reversible buffer recovery MSE across model classes
    buf_means = buf["means"][1]
    m_labels = ["Const.", "2-rate", "Latent", "MemGRN"]
    vals_b = [
        1e6 * buf_means[0][1],
        1e6 * buf_means[1][1],
        1e6 * buf_means[3][1],
        1e6 * buf_means[2][1],
    ]
    cols_b = [PALETTE["slate"], PALETTE["amber"], PALETTE["violet"], PALETTE["navy"]]
    ax1.bar(
        np.arange(4),
        vals_b,
        width=0.58,
        color=cols_b,
        edgecolor=PALETTE["ink"],
        linewidth=0.65,
    )
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(m_labels, rotation=90, ha="center", va="top")
    ax1.tick_params(axis="x", pad=4.0)
    ax1.set_xlim(-0.6, 3.6)
    ax1.set_ylim(0.0, 195.0)
    ax1.set_yticks([0, 50, 100, 150])
    ax1.set_ylabel("Recovery MSE (×10⁻⁶)")
    ax1.set_title("Out-of-class buffer")
    ax1.text(
        3.0,
        vals_b[3] + 6.0,
        f"{vals_b[3]:.1f}",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    add_panel_tag(ax1, "b", x_pt=-30.0, y_pt=6.0)

    # Panel c: Multiscale biphasic waveform relative error (%)
    ms_keys = ["state_space", "current_hybrid", "history_basis", "learned_gw"]
    ms_labels = ["SSM", "1-rate", "Basis", "MemGRN"]
    vals_c = [100.0 * multi["classes"][k]["biphasic"]["error"] for k in ms_keys]
    cols_c = [PALETTE["slate"], PALETTE["amber"], PALETTE["violet"], PALETTE["teal"]]
    ax2.bar(
        np.arange(4),
        vals_c,
        width=0.58,
        color=cols_c,
        edgecolor=PALETTE["ink"],
        linewidth=0.65,
    )
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(ms_labels, rotation=90, ha="center", va="top")
    ax2.tick_params(axis="x", pad=4.0)
    ax2.set_xlim(-0.6, 3.6)
    ax2.set_ylim(0.0, 54.0)
    ax2.set_yticks([0, 15, 30, 45])
    ax2.set_ylabel("Biphasic error (%)")
    ax2.set_title("Sign-reversing edges")
    ax2.text(
        3.0,
        vals_c[3] + 2.0,
        "2.4%",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["teal"],
    )
    add_panel_tag(ax2, "c", x_pt=-28.0, y_pt=6.0)

    export_figure(fig, "fig2_synthetic_dynamics")


def build_fig3_msn2_transfer() -> None:
    """Figure 3: Live-cell Msn2 repeated-pulse transfer and inter-pulse memory resets."""
    rep = load_score("msn2_repeated_pulses")
    reset = load_score("msn2_memory_reset")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.46))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.38, 1.0],
        left=0.105,
        right=0.980,
        bottom=0.30,
        top=0.85,
        wspace=0.34,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

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
    x_idx = np.arange(10)
    series = [
        ("two_rate_gw", "Two-rate gW", PALETTE["amber"], "--", "s"),
        ("latent_ode", "Latent ODE", PALETTE["violet"], "-.", "^"),
        ("neural_rate_gw", "MemGRN (ours)", PALETTE["navy"], "-", "o"),
    ]
    for key, label, col, ls, marker in series:
        vals = rep["models"][key]["all"]["by_schedule"]
        ax0.plot(
            x_idx,
            vals,
            linestyle=ls,
            marker=marker,
            ms=3.8,
            lw=1.3,
            color=col,
            label=label,
        )
    ax0.set_yscale("log")
    ax0.set_xlim(-0.55, 9.45)
    ax0.set_ylim(1.2e-3, 38.0)
    ax0.set_yticks([1e-2, 1e-1, 1.0])
    ax0.set_yticklabels(["0.01", "0.1", "1.0"])
    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(sched_labels, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_xlabel("Pulse count / gap (min)")
    ax0.set_ylabel("Post-shutoff MSE")
    ax0.set_title("Zero-shot transfer (10 trains)")
    ax0.legend(loc="upper left", ncol=1, handlelength=1.7, labelspacing=0.20)
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Inter-pulse memory reset interventions on 6- and 8-pulse trains
    trains = ["FM_6_5min_690nM", "FM_8_5min_690nM"]
    modes = [
        ("intact", "Intact MemGRN", PALETTE["navy"]),
        ("latent", "Latent ODE", PALETTE["violet"]),
        ("last_only", "Reset pre-final", PALETTE["amber"]),
        ("reset", "Reset each pulse", PALETTE["crimson"]),
    ]
    w = 0.18
    gx = np.arange(2)
    y_floor = 1.8e-2
    for m_i, (m_key, m_lbl, m_col) in enumerate(modes):
        vals = np.array(
            [reset["results"][tr]["panels"]["all"]["mse"][m_key] for tr in trains]
        )
        ax1.bar(
            gx + (m_i - 1.5) * w,
            vals - y_floor,
            bottom=y_floor,
            width=w * 0.90,
            color=m_col,
            edgecolor=PALETTE["ink"],
            linewidth=0.6,
            label=m_lbl,
        )
    ax1.set_yscale("log")
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(y_floor, 16.0)
    ax1.set_yticks([0.03, 0.1, 0.3, 1.0])
    ax1.set_yticklabels(["0.03", "0.1", "0.3", "1.0"])
    ax1.set_xticks(gx)
    ax1.set_xticklabels(["6 pulses\n(5-min gap)", "8 pulses\n(5-min gap)"])
    ax1.tick_params(axis="x", pad=4.5)
    ax1.set_ylabel("Post-shutoff MSE")
    ax1.set_title("Model-state memory resets")
    ax1.legend(loc="upper left", ncol=1, handlelength=1.2, labelspacing=0.18)
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    export_figure(
        fig,
        "fig3_msn2_transfer_and_reset",
        exemptions=[
            {
                "panels": ["a", "b"],
                "checks": ["panel-width"],
                "reason": "10-schedule curve panel requires wider x-axis than 2-group bar panel",
            }
        ],
    )


def build_fig4_frozen_readout() -> None:
    """Figure 4: Frozen-readout four-way rate-input ablation across six yeast promoters."""
    six = load_score("msn2_six_promoter_rates")
    genes = six["genes"]
    modes = ["constant", "activity", "memory", "both"]
    mode_labels = ["Constant", "Activity only", "Memory only", "Activity + memory"]
    mode_colors = [PALETTE["slate"], PALETTE["amber"], PALETTE["teal"], PALETTE["navy"]]
    mode_markers = ["s", "^", "o", "D"]

    arr = np.array(
        [[six["results"][g][m]["per_schedule"] for m in modes] for g in genes]
    )

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.52))
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
    ax0.set_ylim(4e-3, 18.0)
    ax0.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax0.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1.0"])
    ax0.set_xticks(gx)
    ax0.set_xticklabels(genes, rotation=90, ha="center", va="top")
    ax0.tick_params(axis="x", pad=4.5)
    ax0.set_ylabel("Mean recovery MSE")
    ax0.set_title("Six promoters")
    ax0.legend(loc="upper left", ncol=1, handlelength=1.5, labelspacing=0.18)
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: log2(Activity MSE / Combined MSE) across 6 promoters x 10 schedules
    log_gain = np.log2(arr[:, 1] / arr[:, 3])
    im = ax1.imshow(
        log_gain,
        vmin=-2.6,
        vmax=2.6,
        cmap="BrBG",
        aspect="auto",
    )
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


def build_fig5_swap_and_nfkb() -> None:
    """Figure 5: Counterbalanced 4:1 strength swap and mammalian NF-kB boundary."""
    swap = load_score("strength_swap_replication")
    nfkb_scores = {
        lig: load_score(f"neural_rate_nfkb_washout/{lig}")
        for lig in ("IL1", "TNF", "LPS")
    }

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.34))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.15],
        left=0.105,
        right=0.980,
        bottom=0.22,
        top=0.84,
        wspace=0.34,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Panel a: Paired repeat vs 4:1 strength swap vs oracle swap across 16 networks
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

    cats = ["All 96 targets\n(16 networks)", "Fast opposing\nparents"]
    rep_vals = [overall_rep, opp_fast_rep]
    swp_vals = [overall_swp, opp_fast_swp]
    orc_vals = [overall_orc, opp_fast_orc]
    gx = np.arange(2)
    w = 0.23
    ax0.bar(
        gx - w,
        rep_vals,
        width=w * 0.90,
        color=PALETTE["amber"],
        edgecolor=PALETTE["ink"],
        linewidth=0.6,
        label="Repeated 4:1",
    )
    ax0.bar(
        gx,
        swp_vals,
        width=w * 0.90,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.6,
        label="Swapped 4:1",
    )
    ax0.bar(
        gx + w,
        orc_vals,
        width=w * 0.90,
        color=PALETTE["teal"],
        edgecolor=PALETTE["ink"],
        linewidth=0.6,
        label="Oracle swap",
    )
    ax0.set_xlim(-0.55, 1.55)
    ax0.set_xticks(gx)
    ax0.set_xticklabels(cats)
    ax0.tick_params(axis="x", pad=4.0)
    ax0.set_ylim(0.0, 188.0)
    ax0.set_yticks([0, 25, 50, 75, 100])
    ax0.set_ylabel("Exact pair recovery (%)")
    ax0.set_title("Counterbalanced parent ID")
    ax0.text(
        0.0,
        max(overall_swp, overall_orc) + 4.0,
        f"+{overall_swp - overall_rep:.1f} pp",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    ax0.text(
        1.0,
        max(opp_fast_swp, opp_fast_orc) + 4.0,
        f"+{opp_fast_swp - opp_fast_rep:.1f} pp",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    ax0.legend(loc="upper left", ncol=1, handlelength=1.2, labelspacing=0.16)
    add_panel_tag(ax0, "a", x_pt=-32.0, y_pt=6.0)

    # Panel b: Mammalian NF-kB cross-ligand washout (2,400-step fixed vs Latent ODE)
    followup = load_score("neural_rate_nfkb_washout", "followup_design.json")
    _ = followup
    nfkb_map = {
        lig: {m["model"]: m["post_mse"] for m in nfkb_scores[lig]["models"]}
        for lig in ("IL1", "TNF", "LPS")
    }
    lig_labels = ["IL-1 holdout", "TNF holdout", "LPS holdout"]
    two_rate_2400 = [0.729, 0.522, 2.785]
    memgrn_cv = [
        nfkb_map["IL1"]["neural_rate_gw"],
        nfkb_map["TNF"]["neural_rate_gw"],
        nfkb_map["LPS"]["neural_rate_gw"],
    ]
    memgrn_2400 = [0.624, 0.411, 1.906]
    latent_cv = [
        nfkb_map["IL1"]["latent_ode"],
        nfkb_map["TNF"]["latent_ode"],
        nfkb_map["LPS"]["latent_ode"],
    ]

    lx = np.arange(3)
    bw = 0.19
    series_b = [
        (two_rate_2400, "Two-rate (2.4k)", PALETTE["amber"]),
        (memgrn_cv, "MemGRN (CV stop)", PALETTE["slate"]),
        (memgrn_2400, "MemGRN (2.4k)", PALETTE["navy"]),
        (latent_cv, "Latent ODE (CV)", PALETTE["violet"]),
    ]
    for s_i, (vals_s, lbl_s, col_s) in enumerate(series_b):
        ax1.bar(
            lx + (s_i - 1.5) * bw,
            vals_s,
            width=bw * 0.92,
            color=col_s,
            edgecolor=PALETTE["ink"],
            linewidth=0.6,
            label=lbl_s,
        )
    ax1.set_xticks(lx)
    ax1.set_xticklabels(lig_labels)
    ax1.set_ylim(0.0, 3.25)
    ax1.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax1.set_ylabel("Post-washout RNA MSE")
    ax1.set_title("Mammalian NF-κB boundary")
    ax1.legend(loc="upper left", ncol=1, handlelength=1.3, labelspacing=0.20)
    add_panel_tag(ax1, "b", x_pt=-28.0, y_pt=6.0)

    export_figure(
        fig,
        "fig5_swap_identification_and_nfkb",
        exemptions=[
            {
                "panels": ["a", "b"],
                "checks": ["panel-width"],
                "reason": "3-ligand 4-bar mammalian comparison uses slightly wider axis than 2-group swap summary",
            }
        ],
    )


def build_all_tables() -> None:
    """Generate clean LaTeX tables (each <= 6 columns) for uniform font sizing."""
    occ = load_score("neural_rate_pulse", "test_scores.json")
    buf = load_score("neural_rate_buffer", "test_scores.json")
    single = load_score("neural_rate_msn2_transfer")
    rep = load_score("msn2_repeated_pulses")
    six = load_score("msn2_six_promoter_rates")
    swap = load_score("strength_swap_replication")
    cert = load_score("paired_time_certificate", "test_scores.json")
    nfkb = {
        lig: load_score(f"neural_rate_nfkb_washout/{lig}")
        for lig in ("IL1", "TNF", "LPS")
    }

    # Table 1: Synthetic kinetics & out-of-class buffering
    t1_rows: list[list[str]] = []
    params_map = [168, 186, 274, 1158]
    model_names = ["Constant $gW$", "Two-rate $gW$", "\\textbf{MemGRN (ours)}", "Latent Neural ODE"]
    for idx, tag, data, r_idx in [
        (0, "Occupancy-gated ($n=8$)", occ, 1),
        (1, "Mass-action buffer ($n=12$)", buf, 1),
        (2, "Constant control ($n=12$)", buf, 0),
    ]:
        if idx > 0:
            t1_rows.append(["\\midrule"])
        means = data["means"][r_idx]
        best_full = int(np.argmin([m[0] for m in means[:4]]))
        best_tail = int(np.argmin([m[1] for m in means[:4]]))
        best_rec = int(np.argmax([m[3] for m in means[:4]]))
        for m_i in range(4):
            vals = means[m_i]
            f_str = f"{1e6 * vals[0]:.2f}"
            t_str = f"{1e6 * vals[1]:.2f}"
            r_str = f"{100.0 * vals[3]:.2f}\\%"
            if m_i == best_full:
                f_str = f"\\textbf{{{f_str}}}"
            if m_i == best_tail:
                t_str = f"\\textbf{{{t_str}}}"
            if m_i == best_rec:
                r_str = f"\\textbf{{{r_str}}}"
            reg_cell = tag if m_i == 0 else ""
            t1_rows.append([reg_cell, model_names[m_i], str(params_map[m_i]), f_str, t_str, r_str])

    write_tex_table(
        "tab1_synthetic.tex",
        "llrrrr",
        [
            "Kinetic regime",
            "Model",
            "Params",
            r"Full MSE $\downarrow$",
            r"Tail MSE $\downarrow$",
            r"Pair ID $\uparrow$",
        ],
        t1_rows,
    )

    # Table 2: Msn2 single-pulse and 10-schedule pulse-train zero-shot transfer
    t2_rows: list[list[str]] = []
    msn2_models = [
        ("neural_rate_gw", "\\textbf{MemGRN (ours)}", "69"),
        ("two_rate_gw", "Two-rate $gW$", "41"),
        ("latent_ode", "Latent Neural ODE", "998"),
        ("source_mean", "Pooled source mean", "--"),
        ("dose_matched_mean", "Dose-matched mean", "--"),
    ]
    s_vals = [single["results"][k]["post"] for k, _, _ in msn2_models]
    r_all = [rep["models"][k]["all"]["primary"] for k, _, _ in msn2_models]
    r_nosip = [rep["models"][k]["without_SIP18"]["primary"] for k, _, _ in msn2_models]
    r_f40 = [rep["models"][k]["all"]["first40"] for k, _, _ in msn2_models]
    for i, (k, label, p_cnt) in enumerate(msn2_models):
        c1 = f"{s_vals[i]:.4f}"
        c2 = f"{r_all[i]:.4f}"
        c3 = f"{r_f40[i]:.4f}"
        c4 = f"{r_nosip[i]:.4f}"
        if i == int(np.argmin(s_vals)):
            c1 = f"\\textbf{{{c1}}}"
        if i == int(np.argmin(r_all)):
            c2 = f"\\textbf{{{c2}}}"
        if i == int(np.argmin(r_f40)):
            c3 = f"\\textbf{{{c3}}}"
        if i == int(np.argmin(r_nosip)):
            c4 = f"\\textbf{{{c4}}}"
        t2_rows.append([label, p_cnt, c1, c2, c3, c4])

    write_tex_table(
        "tab2_msn2_transfer.tex",
        "lrrrrr",
        [
            "Model",
            "Params",
            r"30-min pulse $\downarrow$",
            r"10 trains (all 7) $\downarrow$",
            r"First 40 min $\downarrow$",
            r"No \texttt{SIP18} $\downarrow$",
        ],
        t2_rows,
    )

    # Table 3: Frozen-readout rate-input ablation across six native yeast promoters
    modes = ["constant", "activity", "memory", "both"]
    genes = six["genes"]
    arr = np.array([[six["results"][g][m]["mse"] for m in modes] for g in genes])
    t3_rows: list[list[str]] = []
    for g_i, g in enumerate(genes):
        best_m = int(np.argmin(arr[g_i]))
        cells = [
            f"\\textbf{{{arr[g_i, m_i]:.4f}}}"
            if m_i == best_m
            else f"{arr[g_i, m_i]:.4f}"
            for m_i in range(4)
        ]
        t3_rows.append([f"\\texttt{{{g}}}", *cells])
    t3_rows.append(["\\midrule"])
    means = arr.mean(axis=0)
    best_mean = int(np.argmin(means))
    mean_cells = [
        f"\\textbf{{{means[m_i]:.4f}}}" if m_i == best_mean else f"{means[m_i]:.4f}"
        for m_i in range(4)
    ]
    t3_rows.append(["Pooled mean", *mean_cells])
    t3_rows.append(["Rate parameters", "3", "23", "27", "31"])
    write_tex_table(
        "tab3_six_promoters.tex",
        "lrrrr",
        [
            "Promoter",
            r"Constant $\downarrow$",
            r"Activity only $\downarrow$",
            r"Memory only $\downarrow$",
            r"Activity + memory $\downarrow$",
        ],
        t3_rows,
    )

    # Table 4: Counterbalanced 4:1 strength swap & correlated-noise schedule selection
    t4_rows: list[list[str]] = []
    sw_methods = [
        ("learned_gw", "\\textbf{Learned-memory $gW$}"),
        ("fixed_gw", "Fixed-rate $gW$"),
        ("history_basis", "History time basis"),
        ("oracle", "Oracle templates"),
    ]
    for k, lbl in sw_methods:
        one_v = 100.0 * swap["overall"][k]["single"]
        rep_v = 100.0 * swap["overall"][k]["repeat"]
        swp_v = 100.0 * swap["overall"][k]["swap"]
        gain_v = swp_v - rep_v
        t4_rows.append(
            [
                lbl,
                f"{one_v:.2f}\\%",
                f"{rep_v:.2f}\\%",
                f"{swp_v:.2f}\\%",
                f"+{gain_v:.2f}",
            ]
        )
    write_tex_table(
        "tab4_swap_identification.tex",
        "lrrrr",
        [
            "Template library",
            r"1 trace $\uparrow$",
            r"Repeat 4:1 $\uparrow$",
            r"Swap 4:1 $\uparrow$",
            r"Swap $-$ Repeat (pp) $\uparrow$",
        ],
        t4_rows,
    )

    # Table 5: Mammalian NF-kB cross-ligand washout boundary
    nfkb_map = {
        lig: {m["model"]: m["post_mse"] for m in nfkb[lig]["models"]}
        for lig in ("IL1", "TNF", "LPS")
    }
    t5_rows = [
        [
            "Constant $gW$ (CV stop)",
            f"{nfkb_map['IL1']['current_gw']:.3f}",
            f"{nfkb_map['TNF']['current_gw']:.3f}",
            f"{nfkb_map['LPS']['current_gw']:.3f}",
        ],
        [
            "Two-rate $gW$ (CV stop)",
            f"{nfkb_map['IL1']['two_rate_gw']:.3f}",
            f"{nfkb_map['TNF']['two_rate_gw']:.3f}",
            f"{nfkb_map['LPS']['two_rate_gw']:.3f}",
        ],
        [
            "MemGRN (CV stop)",
            f"{nfkb_map['IL1']['neural_rate_gw']:.3f}",
            f"{nfkb_map['TNF']['neural_rate_gw']:.3f}",
            f"{nfkb_map['LPS']['neural_rate_gw']:.3f}",
        ],
        [
            "Latent Neural ODE (CV stop)",
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
        "tab5_nfkb_boundary.tex",
        "lrrr",
        [
            "Model / checkpoint rule",
            r"IL-1 holdout $\downarrow$",
            r"TNF holdout $\downarrow$",
            r"LPS holdout $\downarrow$",
        ],
        t5_rows,
    )

    # Appendix Table A1: Multiscale waveform & autonomous shared-driver disambiguation
    multi = load_score("multiscale_gw")
    mixed = load_score("mixed_directed_memory")
    ta1_rows: list[list[str]] = []
    for k, lbl in [
        ("learned_gw", "\\textbf{Multiscale MemGRN}"),
        ("history_basis", "History time basis"),
        ("current_hybrid", "Single-rate hybrid"),
        ("state_space", "Linear state space"),
    ]:
        c_dat = multi["classes"][k]["biphasic"]
        m_all = multi["means"][k]
        ta1_rows.append(
            [
                "Multiscale suite" if k == "learned_gw" else "",
                lbl,
                f"{100.0 * c_dat['error']:.2f}\\%",
                f"{100.0 * m_all['reversal']:.1f}\\%",
                f"{m_all['delayed_ap']:.3f}",
            ]
        )
    ta1_rows.append(["\\midrule"])
    for k, lbl in [
        ("memory_driver", "\\textbf{MemGRN + oscillator}"),
        ("latent_ODE", "Latent Neural ODE"),
        ("state_space", "Linear state space"),
        ("directed_memory", "MemGRN (no oscillator)"),
        ("instant_driver", "Memoryless + oscillator"),
    ]:
        m_dat = mixed["means"][k]
        ta1_rows.append(
            [
                "Shared driver" if k == "memory_driver" else "",
                lbl,
                f"{100.0 * m_dat['tail_error_ratio']:.2f}\\%",
                f"{100.0 * m_dat['tail_sign']:.1f}\\%",
                f"{m_dat['delayed_ap']:.3f}",
            ]
        )
    write_tex_table(
        "tab_a1_multiscale_driver.tex",
        "llrrr",
        [
            "Benchmark",
            "Model",
            r"Tail error $\downarrow$",
            r"Sign acc. $\uparrow$",
            r"Edge AP $\uparrow$",
        ],
        ta1_rows,
    )

    # Appendix Table A2: RTN2 continuous-vs-switch and rate-input RK4 ablations
    rtn_sw = load_score("msn2_rtn2_switch_neural")
    rtn_in = load_score("msn2_rtn2_rate_inputs")
    ta2_rows: list[list[str]] = []
    for idx_r, (k, lbl) in enumerate(
        [
            ("constant", "Constant rates"),
            ("switch_only", "Discrete switch only"),
            ("neural_only", "Continuous neural rate"),
            ("full", "Switch + neural rate"),
        ]
    ):
        d_sw = rtn_sw["models"][k]
        ta2_rows.append(
            [
                "Switch vs. neural" if idx_r == 0 else "",
                lbl,
                f"{d_sw['primary']:.4f}",
                f"{d_sw['half_step']:.4f}",
                f"{d_sw['first40']:.4f}",
            ]
        )
    ta2_rows.append(["\\midrule"])
    for idx_r, (k, lbl) in enumerate(
        [
            ("constant", "Constant ($\\xi = \\varnothing$)"),
            ("activity", "Activity only ($a_i$)"),
            ("memory", "Memory only ($z_{ir}, h_{ir}$)"),
            ("both", "\\textbf{Activity + memory}"),
        ]
    ):
        d_in = rtn_in["models"][k]
        ta2_rows.append(
            [
                "RK4 rate inputs" if idx_r == 0 else "",
                lbl,
                f"{d_in['primary']:.4f}",
                f"{d_in['half_step']:.4f}",
                f"{d_in['first40']:.4f}",
            ]
        )
    write_tex_table(
        "tab_a2_rtn2_ablations.tex",
        "llrrr",
        [
            "Ablation",
            "Rate parameterization",
            r"Primary MSE $\downarrow$",
            r"Refined RK4 $\downarrow$",
            r"First 40 min $\downarrow$",
        ],
        ta2_rows,
    )

    # Appendix Table A3: Msn2 inter-pulse memory resets and Proposition 1 splice accounting
    reset = load_score("msn2_memory_reset")
    ta3_rows: list[list[str]] = []
    for tr_id, tr_lbl in [
        ("FM_6_5min_690nM", "6 pulses (5-min gap)"),
        ("FM_8_5min_690nM", "8 pulses (5-min gap)"),
    ]:
        p_all = reset["results"][tr_id]["panels"]["all"]["mse"]
        p_rtn = reset["results"][tr_id]["promoters"]["RTN2"]
        ta3_rows.append(
            [
                tr_lbl,
                f"\\textbf{{{p_all['intact']:.4f}}}",
                f"{p_all['last_only']:.4f}",
                f"{p_all['reset']:.4f}",
                f"+{p_rtn['shift_attributable_to_reporter_at_off']:.3f}",
                f"+{p_rtn['shift_attributable_to_production_after_off']:.3f}",
            ]
        )
    write_tex_table(
        "tab_a3_msn2_reset.tex",
        "lrrrrr",
        [
            "Pulse train",
            r"Intact $\downarrow$",
            r"Pre-final reset $\downarrow$",
            r"Full reset $\downarrow$",
            r"\texttt{RTN2} carryover",
            r"\texttt{RTN2} post-off",
        ],
        ta3_rows,
    )

    # Appendix Table A4: Strength-swap recovery across structural parent-pair subgroups
    ta4_rows: list[list[str]] = []
    for sg_key, sg_lbl in [
        ("all", "All targets ($96$)"),
        ("aligned", "Aligned signs ($48$)"),
        ("opposing", "Opposing signs ($48$)"),
        ("fast/fast opposing", "Fast opposing ($11$)"),
        ("biphasic/biphasic", "Biphasic pairs ($16$)"),
    ]:
        sg_m = swap["subgroups"][sg_key]["means"]["learned_gw"]
        s1 = 100.0 * sg_m["single"]
        sr = 100.0 * sg_m["repeat"]
        ss = 100.0 * sg_m["swap"]
        so = 100.0 * swap["subgroups"][sg_key]["means"]["oracle"]["swap"]
        ta4_rows.append(
            [
                sg_lbl,
                f"{s1:.2f}\\%",
                f"{sr:.2f}\\%",
                f"\\textbf{{{ss:.2f}\\%}}",
                f"+{ss - sr:.2f}",
                f"{so:.2f}\\%",
            ]
        )
    write_tex_table(
        "tab_a4_swap_subgroups.tex",
        "lrrrrr",
        [
            "Target subgroup",
            r"1 trace $\uparrow$",
            r"Repeat 4:1 $\uparrow$",
            r"Swap 4:1 $\uparrow$",
            r"Gain (pp) $\uparrow$",
            r"Oracle swap $\uparrow$",
        ],
        ta4_rows,
    )

    # Appendix Table A5: AR(1) 5-time schedule selection and paired certificate
    cert_meta = load_score("paired_time_certificate", "certificate.json")
    ta5_rows: list[list[str]] = []
    name_map = {
        "learned_gw": "\\textbf{Learned-memory $gW$}",
        "fixed_gw": "Fixed-rate $gW$",
        "history_basis": "History time basis",
    }
    for idx_c, r_cert in enumerate(cert["rows"]):
        p_meta = cert_meta["panels"][idx_c]
        ci_lo = 100.0 * p_meta["ci"][0]
        ci_hi = 100.0 * p_meta["ci"][1]
        ta5_rows.append(
            [
                name_map[r_cert["model"]],
                f"{100.0 * r_cert['progression']:.2f}\\%",
                f"{100.0 * r_cert['full_window']:.2f}\\%",
                f"+{100.0 * r_cert['secondary_gap']:.2f}",
                f"[{ci_lo:+.2f}, {ci_hi:+.2f}]",
            ]
        )
    write_tex_table(
        "tab_a5_schedule_cert.tex",
        "lrrrr",
        [
            "Template library",
            r"Selected 5-pt $\uparrow$",
            r"Full-window 5-pt $\uparrow$",
            r"Net gain (pp) $\uparrow$",
            r"Certified 98.5\% CI",
        ],
        ta5_rows,
    )


def main() -> None:
    build_fig1_architecture()
    build_fig2_synthetic()
    build_fig3_msn2_transfer()
    build_fig4_frozen_readout()
    build_fig5_swap_and_nfkb()
    build_all_tables()
    ledger_path = ROOT / "results" / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps({"source_sha256": SOURCE_HASHES}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built 5 figures and 10 tables from {len(SOURCE_HASHES)} verified artifacts.")


if __name__ == "__main__":
    main()
