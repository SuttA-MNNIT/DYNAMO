"""
Ablation Studies
================
Three ablations over Split-MNIST:
  ablation_lambda    – sweep basal metabolic rate λ
  ablation_hcrit     – sweep mitosis heat threshold H_crit
  ablation_sparsity  – sweep initial edge sparsity
"""

from __future__ import annotations

import numpy as np
import torch

from dtn.data import get_split_mnist
from dtn.models import DTN
from dtn.training import accuracy, train_one_task_dtn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_dtn_on_split_mnist(epochs_per_task: int = 10, **dtn_kwargs) -> DTN:
    """Utility: build a DTN, train on all 5 Split-MNIST tasks, return it."""
    tr, _ = get_split_mnist(batch_size=128)
    dtn = DTN(784, [64, 32], 2, **dtn_kwargs).to(DEVICE)
    for t in range(5):
        dtn.add_head()
        dtn.heads[-1].to(DEVICE)
        train_one_task_dtn(dtn, tr[t], lr=0.01, n_epochs=epochs_per_task)
    return dtn


def ablation_lambda(epochs_per_task: int = 10) -> dict:
    """Sweep λ ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}."""
    print("\n  [ABLATION] Basal Metabolic Rate λ")
    _, te = get_split_mnist(batch_size=128)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    results = {}

    for lam in lambdas:
        dtn   = _run_dtn_on_split_mnist(
            epochs_per_task=epochs_per_task,
            sparsity=0.95, lam=lam, alpha=0.9, beta=1.0,
            h_crit=0.85, max_w=256, cap_mitosis=6,
        )
        final = [accuracy(dtn, te[t], t) for t in range(5)]
        results[lam] = {
            "avg_acc"     : float(np.mean(final)),
            "task1_ret"   : final[0],
            "final_params": dtn.count_active_params(),
            "param_hist"  : list(dtn.param_hist),
        }
        print(
            f"    λ={lam:.0e} → avg={results[lam]['avg_acc']:.1f}%  "
            f"T1={results[lam]['task1_ret']:.1f}%  "
            f"params={results[lam]['final_params']}"
        )

    return results


def ablation_hcrit(epochs_per_task: int = 10) -> dict:
    """Sweep H_crit ∈ {0.3, 0.5, 0.7, 0.85, 0.95, 1.5}."""
    print("\n  [ABLATION] Mitosis Heat Threshold H_crit")
    _, te = get_split_mnist(batch_size=128)
    h_crits = [0.3, 0.5, 0.7, 0.85, 0.95, 1.5]
    results = {}

    for hc in h_crits:
        dtn   = _run_dtn_on_split_mnist(
            epochs_per_task=epochs_per_task,
            sparsity=0.95, lam=1e-4, alpha=0.9, beta=1.0,
            h_crit=hc, max_w=256, cap_mitosis=6,
        )
        final = [accuracy(dtn, te[t], t) for t in range(5)]
        results[hc] = {
            "avg_acc"     : float(np.mean(final)),
            "task1_ret"   : final[0],
            "n_mitosis"   : len(dtn.mitosis_log),
            "final_params": dtn.count_active_params(),
        }
        print(
            f"    H_crit={hc:.2f} → avg={results[hc]['avg_acc']:.1f}%  "
            f"T1={results[hc]['task1_ret']:.1f}%  "
            f"mitosis events={results[hc]['n_mitosis']}"
        )

    return results


def ablation_sparsity(epochs_per_task: int = 10) -> dict:
    """Sweep initial sparsity ∈ {0.50, 0.70, 0.80, 0.90, 0.95, 0.98}."""
    print("\n  [ABLATION] Initial Sparsity")
    _, te = get_split_mnist(batch_size=128)
    sparsities = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    results = {}

    for sp in sparsities:
        dtn   = _run_dtn_on_split_mnist(
            epochs_per_task=epochs_per_task,
            sparsity=sp, lam=1e-4, alpha=0.9, beta=1.0,
            h_crit=0.85, max_w=256, cap_mitosis=6,
        )
        final = [accuracy(dtn, te[t], t) for t in range(5)]
        results[sp] = {
            "avg_acc"     : float(np.mean(final)),
            "task1_ret"   : final[0],
            "final_params": dtn.count_active_params(),
        }
        print(
            f"    sparsity={sp:.2f} → avg={results[sp]['avg_acc']:.1f}%  "
            f"T1={results[sp]['task1_ret']:.1f}%"
        )

    return results
