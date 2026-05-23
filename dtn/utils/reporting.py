"""
Reporting utilities — console tables and JSON persistence.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from dtn.models import DTN


SEP = "─" * 72


def print_and_save_tables(
    mnist_sum: dict,
    cifar_sum: dict,
    jaccard: dict,
    dtn_mnist: DTN,
    save_dir: str,
) -> None:
    """Print summary tables to stdout and write them to ``tables.txt``."""

    # ── TABLE 1 ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TABLE 1 — Split-MNIST End-of-Sequence Performance")
    print(SEP)
    print(f"  {'Model':<10}{'Avg Acc (AT)':>14}{'T1 Ret. (R1)':>14}{'Forgetting':>13}")
    print(SEP)
    for m, s in mnist_sum.items():
        print(
            f"  {m:<10}{s['avg_accuracy']:>13.1f}%"
            f"{s['task1_retention']:>13.1f}%"
            f"{s['forgetting_rate']:>12.1f}%"
        )
    dtn_p = dtn_mnist.count_active_params()
    print(f"\n  DTN final active params: {dtn_p}  |  Initial sparsity: 95%")
    print(SEP)

    # ── TABLE 2 ──────────────────────────────────────────────────
    if cifar_sum:
        print(f"\n  TABLE 2 — Sequential CIFAR-100 End-of-Sequence Performance")
        print(SEP)
        print(f"  {'Model':<10}{'Avg Acc (AT)':>14}{'T1 Ret. (R1)':>14}")
        print(SEP)
        for m, s in cifar_sum.items():
            print(f"  {m:<10}{s['avg_accuracy']:>13.1f}%{s['task1_retention']:>13.1f}%")
        print(SEP)

    # ── TABLE 3 ──────────────────────────────────────────────────
    if jaccard:
        print(f"\n  TABLE 3 — Jaccard Sub-network Isolation (vs Task 1)")
        print(SEP)
        print(f"  {'Pair':<10}{'Jaccard Overlap':>17}")
        print(SEP)
        for pair, val in jaccard.items():
            print(f"  {pair:<10}{val:>16.2f}%")
        print(SEP)

    # ── Write to file ─────────────────────────────────────────────
    txt_path = os.path.join(save_dir, "tables.txt")
    with open(txt_path, "w") as f:
        f.write("TABLE 1 — Split-MNIST\n")
        for m, s in mnist_sum.items():
            f.write(
                f"{m}: avg={s['avg_accuracy']:.2f}%, "
                f"T1={s['task1_retention']:.2f}%, "
                f"forget={s['forgetting_rate']:.2f}%\n"
            )
        f.write("\nTABLE 2 — CIFAR-100\n")
        for m, s in cifar_sum.items():
            f.write(f"{m}: avg={s['avg_accuracy']:.2f}%, T1={s['task1_retention']:.2f}%\n")
        f.write("\nTABLE 3 — Jaccard\n")
        for k, v in jaccard.items():
            f.write(f"{k}: {v:.2f}%\n")
    print(f"\n  Tables saved → {txt_path}")


def save_results_json(
    mnist_sum: dict,
    cifar_sum: dict,
    jaccard: dict,
    ab_lam: dict,
    ab_hc: dict,
    ab_sp: dict,
    param_hist: list,
    save_dir: str,
) -> None:
    """Serialise all results to ``all_results.json``."""
    blob = {
        "mnist_summary"    : mnist_sum,
        "cifar_summary"    : cifar_sum,
        "jaccard"          : jaccard,
        "ablation_lambda"  : {str(k): v for k, v in ab_lam.items()},
        "ablation_hcrit"   : {str(k): v for k, v in ab_hc.items()},
        "ablation_sparsity": {str(k): v for k, v in ab_sp.items()},
        "dtn_param_history": param_hist,
    }
    path = os.path.join(save_dir, "all_results.json")
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"  Results JSON saved → {path}")
