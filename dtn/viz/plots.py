"""
Visualisation helpers for DTN experiment results.

All public functions accept pre-computed result dicts and save
individual PDF + PNG figures to ``save_dir``.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from dtn.models import DTN

# ── style constants ───────────────────────────────────────────────────────────

COLORS = {
    "DTN"  : "#1565C0",
    "Dense": "#C62828",
    "EWC"  : "#E65100",
    "PNN"  : "#2E7D32",
}
MARKERS = {"DTN": "o", "Dense": "s", "EWC": "^", "PNN": "D"}


def _style() -> None:
    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "font.size"        : 11,
        "axes.titlesize"   : 13,
        "axes.labelsize"   : 11,
        "figure.facecolor" : "white",
        "axes.facecolor"   : "#F5F5F5",
        "axes.grid"        : True,
        "grid.alpha"       : 0.35,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, stem: str, save_dir: str) -> None:
    fig.tight_layout()
    pdf = os.path.join(save_dir, f"{stem}.pdf")
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(pdf.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")


# ── Experiment 1 overview (8 panels saved as individual files) ────────────────


def fig_mnist_overview(
    results: dict,
    summary: dict,
    pcount: dict,
    jaccard: dict,
    dtn_model: DTN,
    save_dir: str,
) -> None:
    _style()
    N  = 5
    fs = (6, 4.5)

    # 1a – Task-1 retention
    fig, ax = plt.subplots(figsize=fs)
    for m, col in COLORS.items():
        vals = results[m].get(0, [])
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, MARKERS[m] + "-",
                    color=col, label=m, lw=2.2, ms=8)
    ax.set(title="Task-1 Retention Accuracy", xlabel="Tasks Seen",
           ylabel="Accuracy (%)", ylim=(0, 105), xticks=range(1, N + 1))
    ax.legend(fontsize=9)
    _save(fig, "fig1_a_mnist_task1_retention", save_dir)

    # 1b – Global average accuracy
    fig, ax = plt.subplots(figsize=fs)
    for m, col in COLORS.items():
        avgs = []
        for step in range(N):
            buf = []
            for ev in range(step + 1):
                v   = results[m].get(ev, [])
                idx = step - ev
                buf.append(v[idx] if idx < len(v) else (v[-1] if v else 0.0))
            avgs.append(np.mean(buf))
        ax.plot(range(1, N + 1), avgs, MARKERS[m] + "-",
                color=col, label=m, lw=2.2, ms=8)
    ax.set(title="Global Average Accuracy", xlabel="Tasks Seen",
           ylabel="Avg Accuracy (%)", ylim=(0, 105), xticks=range(1, N + 1))
    ax.legend(fontsize=9)
    _save(fig, "fig1_b_mnist_avg_accuracy", save_dir)

    # 1c – DTN topological breathing
    fig, ax = plt.subplots(figsize=fs)
    if pcount["DTN"]:
        ph   = pcount["DTN"]
        step = max(len(ph) // N, 1)
        ax.fill_between(range(len(ph)), ph, alpha=0.2, color=COLORS["DTN"])
        ax.plot(range(len(ph)), ph, "-", color=COLORS["DTN"], lw=2.5, label="DTN")
        for i in range(1, N):
            ax.axvline(i * step, color="#757575", ls="--", alpha=0.7, lw=1.2)
            ax.text(i * step + 0.3, max(ph) * 0.96, f"T{i+1}", fontsize=8, color="#555")
        ax.axhline(ph[0], color="red", ls=":", lw=1.3, alpha=0.6, label=f"Init: {ph[0]}")
        ax.set(title='DTN Topological "Breathing"',
               xlabel="Training Epoch", ylabel="Active Parameters")
        ax.legend(fontsize=9)
    _save(fig, "fig1_c_mnist_dtn_breathing", save_dir)

    # 1d – Jaccard overlap
    fig, ax = plt.subplots(figsize=fs)
    if jaccard:
        lbs, vals = list(jaccard.keys()), list(jaccard.values())
        bars = ax.bar(lbs, vals, color="#7B1FA2", edgecolor="black", lw=0.7, width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set(title="Sub-network Isolation\n(Jaccard Overlap vs Task 1)",
               ylabel="Jaccard Similarity (%)", xlabel="Task Pair",
               ylim=(0, max(vals) * 1.35 + 5))
    _save(fig, "fig1_d_mnist_jaccard", save_dir)

    # 1e – Final avg acc bar
    fig, ax = plt.subplots(figsize=fs)
    ms   = list(summary.keys())
    avgs = [summary[m]["avg_accuracy"] for m in ms]
    bars = ax.bar(ms, avgs, color=[COLORS[m] for m in ms],
                  edgecolor="black", lw=0.7, width=0.55)
    for b, v in zip(bars, avgs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(title="Final Avg Accuracy (All 5 Tasks)", ylabel="Accuracy (%)", ylim=(0, 110))
    ax.axhline(50, color="gray", ls=":", alpha=0.4)
    _save(fig, "fig1_e_mnist_final_avg_acc", save_dir)

    # 1f – Forgetting rate
    fig, ax = plt.subplots(figsize=fs)
    frs  = [summary[m]["forgetting_rate"] for m in ms]
    bars = ax.bar(ms, frs, color=[COLORS[m] for m in ms],
                  edgecolor="black", lw=0.7, width=0.55)
    for b, v in zip(bars, frs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(title="Task-1 Forgetting Rate", ylabel="Forgetting (%)", ylim=(0, 110))
    _save(fig, "fig1_f_mnist_forgetting_rate", save_dir)

    # 1g – Layer-wise sparsity
    fig, ax = plt.subplots(figsize=fs)
    sps    = [s * 100 for s in dtn_model.sparsity_per_layer()]
    lnames = [f"Backbone L{i+1}" for i in range(len(sps))]
    pal    = ["#1565C0", "#1976D2", "#42A5F5"]
    bars   = ax.bar(lnames, sps, color=pal[:len(sps)], edgecolor="black", lw=0.7)
    for b, v in zip(bars, sps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(title="Layer-wise Sparsity (DTN After All Tasks)",
           ylabel="Sparsity Ratio (%)", ylim=(0, 105))
    _save(fig, "fig1_g_mnist_layer_sparsity", save_dir)

    # 1h – Mitosis events
    fig, ax = plt.subplots(figsize=fs)
    if dtn_model.mitosis_log:
        ep, cnt = zip(*dtn_model.mitosis_log)
        ax.stem(ep, cnt, linefmt=COLORS["DTN"], markerfmt="o", basefmt="gray")
        ax.set(title="Mitosis Events Over Training",
               xlabel="Epoch", ylabel="Neurons Spawned")
    else:
        ax.text(0.5, 0.5, "No mitosis events\n(H_crit not exceeded)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Mitosis Events")
    _save(fig, "fig1_h_mnist_mitosis_events", save_dir)


# ── Accuracy heatmaps ─────────────────────────────────────────────────────────


def fig_accuracy_heatmaps(
    results: dict,
    N: int,
    models_to_show: List[str],
    save_dir: str,
) -> None:
    _style()
    cmap = LinearSegmentedColormap.from_list("rg", ["#C62828", "#FFD54F", "#2E7D32"], N=256)

    for mname in models_to_show:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        mat = np.full((N, N), np.nan)
        for t_seen in range(N):
            for t_eval in range(t_seen + 1):
                vals = results[mname].get(t_eval, [])
                idx  = t_seen - t_eval
                mat[t_seen, t_eval] = vals[idx] if idx < len(vals) else (vals[-1] if vals else 0.0)

        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set(
            xticks=range(N), yticks=range(N),
            xticklabels=[f"T{i+1}" for i in range(N)],
            yticklabels=[f"After T{i+1}" for i in range(N)],
            title=f"{mname} Per-Task Accuracy Matrix",
            xlabel="Evaluated on Task", ylabel="Trained through Task",
        )
        plt.colorbar(im, ax=ax, label="Acc (%)", fraction=0.046)
        for i in range(N):
            for j in range(N):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if mat[i, j] < 45 else "black")
        _save(fig, f"fig2_heatmap_{mname}", save_dir)


# ── Ablation plots ────────────────────────────────────────────────────────────


def fig_ablations(
    ab_lambda: dict,
    ab_hcrit: dict,
    ab_sparsity: dict,
    save_dir: str,
) -> None:
    _style()
    fs = (6, 5)

    # λ ablation
    fig, ax = plt.subplots(figsize=fs)
    lams = sorted(ab_lambda.keys())
    avgs = [ab_lambda[l]["avg_acc"] for l in lams]
    t1s  = [ab_lambda[l]["task1_ret"] for l in lams]
    x    = range(len(lams))
    ax.bar([xi - 0.2 for xi in x], avgs, 0.38, label="Avg Acc", color="#1565C0", edgecolor="k", lw=0.6)
    ax.bar([xi + 0.2 for xi in x], t1s,  0.38, label="T1 Ret.", color="#2E7D32", edgecolor="k", lw=0.6)
    ax.set(xticks=list(x), xticklabels=[f"{l:.0e}" for l in lams],
           xlabel="Basal Metabolic Rate (λ)", ylabel="Accuracy (%)",
           title="λ Ablation (Metabolic Rate)", ylim=(0, 110))
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    _save(fig, "fig3_a_ablation_lambda", save_dir)

    # H_crit ablation
    fig, ax = plt.subplots(figsize=fs)
    hcs  = sorted(ab_hcrit.keys())
    avgs = [ab_hcrit[h]["avg_acc"] for h in hcs]
    t1s  = [ab_hcrit[h]["task1_ret"] for h in hcs]
    mts  = [ab_hcrit[h]["n_mitosis"] for h in hcs]
    ax2  = ax.twinx()
    ax.bar([xi - 0.2 for xi in range(len(hcs))], avgs, 0.38, label="Avg Acc", color="#E65100", edgecolor="k", lw=0.6)
    ax.bar([xi + 0.2 for xi in range(len(hcs))], t1s,  0.38, label="T1 Ret.", color="#7B1FA2", edgecolor="k", lw=0.6)
    ax2.plot(range(len(hcs)), mts, "k--o", lw=1.8, ms=6, label="Mitosis Events")
    ax.set(xticks=range(len(hcs)), xticklabels=[str(h) for h in hcs],
           xlabel="H_crit", ylabel="Accuracy (%)",
           title="H_crit Ablation (Heat Threshold)", ylim=(0, 110))
    ax2.set_ylabel("Mitosis Count")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9)
    _save(fig, "fig3_b_ablation_hcrit", save_dir)

    # Sparsity ablation
    fig, ax = plt.subplots(figsize=fs)
    sps  = sorted(ab_sparsity.keys())
    avgs = [ab_sparsity[s]["avg_acc"] for s in sps]
    t1s  = [ab_sparsity[s]["task1_ret"] for s in sps]
    fps  = [ab_sparsity[s]["final_params"] for s in sps]
    ax2  = ax.twinx()
    ax.plot(sps, avgs, "o-",  color="#1565C0", lw=2.2, ms=8, label="Avg Acc")
    ax.plot(sps, t1s,  "s--", color="#C62828", lw=2.2, ms=8, label="T1 Ret.")
    ax2.plot(sps, fps, "^:",  color="#555",    lw=1.8, ms=7, label="Final Params")
    ax.set(xlabel="Initial Sparsity", ylabel="Accuracy (%)",
           title="Initial Sparsity Ablation", ylim=(0, 110))
    ax2.set_ylabel("Final Active Params")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9)
    _save(fig, "fig3_c_ablation_sparsity", save_dir)


# ── CIFAR-100 results ─────────────────────────────────────────────────────────


def fig_cifar_results(
    results: dict,
    summary: dict,
    pcount: dict,
    dtn_c: DTN,
    n_tasks: int,
    save_dir: str,
) -> None:
    _style()
    fs = (6, 4.5)
    CIFAR_COLORS = {k: COLORS[k] for k in ("DTN", "Dense", "EWC")}

    # 4a – Task-1 retention
    fig, ax = plt.subplots(figsize=fs)
    for m, col in CIFAR_COLORS.items():
        v = results[m].get(0, [])
        if v:
            ax.plot(range(1, len(v) + 1), v, MARKERS[m] + "-",
                    color=col, label=m, lw=2.2, ms=7)
    ax.set(title="Task-1 Retention (CIFAR-100)", xlabel="Tasks Seen",
           ylabel="Task-1 Accuracy (%)", ylim=(0, 105),
           xticks=range(1, n_tasks + 1))
    ax.legend()
    _save(fig, "fig4_a_cifar_task1_retention", save_dir)

    # 4b – Global average accuracy
    fig, ax = plt.subplots(figsize=fs)
    for m, col in CIFAR_COLORS.items():
        avgs = []
        for step in range(n_tasks):
            buf = []
            for ev in range(step + 1):
                v   = results[m].get(ev, [])
                idx = step - ev
                buf.append(v[idx] if idx < len(v) else (v[-1] if v else 0.0))
            avgs.append(np.mean(buf))
        ax.plot(range(1, n_tasks + 1), avgs, MARKERS[m] + "-",
                color=col, label=m, lw=2.2, ms=7)
    ax.set(title="Global Avg Accuracy (CIFAR-100)", xlabel="Tasks Seen",
           ylabel="Avg Accuracy (%)", ylim=(0, 105))
    ax.legend()
    _save(fig, "fig4_b_cifar_avg_accuracy", save_dir)

    # 4c – DTN breathing
    fig, ax = plt.subplots(figsize=fs)
    if pcount["DTN"]:
        ph   = pcount["DTN"]
        step = max(len(ph) // n_tasks, 1)
        ax.fill_between(range(len(ph)), ph, alpha=0.2, color=COLORS["DTN"])
        ax.plot(range(len(ph)), ph, "-", color=COLORS["DTN"], lw=2.5)
        for i in range(1, n_tasks):
            ax.axvline(i * step, color="#757575", ls="--", alpha=0.6, lw=1)
        ax.set(title="DTN Parameter Evolution (CIFAR-100)",
               xlabel="Epoch", ylabel="Active Params (DTN Head)")
    _save(fig, "fig4_c_cifar_dtn_breathing", save_dir)

    # 4d – Final avg acc bar
    fig, ax = plt.subplots(figsize=fs)
    ms   = list(summary.keys())
    avgs = [summary[m]["avg_accuracy"] for m in ms]
    bars = ax.bar(ms, avgs, color=[CIFAR_COLORS[m] for m in ms],
                  edgecolor="k", lw=0.7, width=0.55)
    for b, v in zip(bars, avgs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(title="Final Avg Accuracy (CIFAR-100)", ylabel="Accuracy (%)", ylim=(0, 110))
    _save(fig, "fig4_d_cifar_final_avg_acc", save_dir)

    # 4e – Task-1 retention bar
    fig, ax = plt.subplots(figsize=fs)
    t1s  = [summary[m]["task1_retention"] for m in ms]
    bars = ax.bar(ms, t1s, color=[CIFAR_COLORS[m] for m in ms],
                  edgecolor="k", lw=0.7, width=0.55)
    for b, v in zip(bars, t1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(title="Task-1 Retention (CIFAR-100)", ylabel="Accuracy (%)", ylim=(0, 110))
    _save(fig, "fig4_e_cifar_task1_retention_bar", save_dir)

    # 4f – Layer-wise sparsity
    fig, ax = plt.subplots(figsize=fs)
    sps    = [s * 100 for s in dtn_c.sparsity_per_layer()]
    lnames = [f"L{i+1}" for i in range(len(sps))]
    bars   = ax.bar(lnames, sps,
                    color=["#1565C0", "#1976D2", "#42A5F5"][:len(sps)],
                    edgecolor="k", lw=0.7)
    for b, v in zip(bars, sps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(title="Layer-wise Sparsity (CIFAR-100 DTN Head)",
           ylabel="Sparsity (%)", ylim=(0, 105))
    _save(fig, "fig4_f_cifar_layer_sparsity", save_dir)
