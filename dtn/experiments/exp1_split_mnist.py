"""
Experiment 1 — Split-MNIST
==========================
Trains DTN, Dense-SGD, EWC, and PNN on 5 binary MNIST tasks and returns
a rich dict of per-task accuracy traces, summary statistics, and Jaccard
sub-network overlap scores.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from dtn.data import get_split_mnist
from dtn.models import DTN, DenseMLPMultiHead, EWCWrapper, PNN
from dtn.training import (
    accuracy,
    train_one_task_dtn,
    train_one_task_dense,
    train_one_task_ewc,
    train_one_task_pnn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TASKS = 5


def run(epochs_per_task: int = 12) -> Tuple[dict, dict, dict, dict, DTN]:
    """
    Run Experiment 1.

    Returns
    -------
    results  : per-model, per-task accuracy traces
    summary  : end-of-sequence scalar metrics per model
    pcount   : active-parameter history per model
    jaccard  : Jaccard sub-network overlap between task pairs (DTN only)
    dtn      : trained DTN instance (for post-hoc analysis / plots)
    """
    print("\n" + "═" * 62)
    print("  EXPERIMENT 1 — Split-MNIST  (5 binary tasks, 10 classes)")
    print("═" * 62)

    tr_loaders, te_loaders = get_split_mnist(batch_size=128)

    results = {m: defaultdict(list) for m in ["DTN", "Dense", "EWC", "PNN"]}
    pcount  = {"DTN": [], "Dense": [], "EWC": [], "PNN": []}

    # ── DTN ──────────────────────────────────────────────────────
    print("\n  [1/4] DTN")
    dtn = DTN(
        input_size=784,
        hidden_sizes=[64, 32],
        n_classes_per_task=2,
        sparsity=0.95,
        lam=1e-4, alpha=0.9, beta=1.0,
        h_crit=0.85, max_w=256, cap_mitosis=6,
    ).to(DEVICE)

    dtn_masks: Dict[int, torch.Tensor] = {}

    for t in range(N_TASKS):
        print(f"    Task {t+1}/{N_TASKS}  (active params: {dtn.count_active_params()})", end=" ")
        dtn.add_head(); dtn.heads[-1].to(DEVICE)
        train_one_task_dtn(dtn, tr_loaders[t], lr=0.01, n_epochs=epochs_per_task)

        dtn_masks[t] = dtn.adjacency_mask_flat(0).clone()

        for ev in range(t + 1):
            results["DTN"][ev].append(accuracy(dtn, te_loaders[ev], ev))
        print(f"→ avg acc: {np.mean([results['DTN'][ev][-1] for ev in range(t+1)]):.1f}%")

    pcount["DTN"] = list(dtn.param_hist)

    # ── Dense SGD ────────────────────────────────────────────────
    print("\n  [2/4] Dense SGD")
    dense = DenseMLPMultiHead(784, [128, 64], 2).to(DEVICE)

    for t in range(N_TASKS):
        print(f"    Task {t+1}/{N_TASKS}", end=" ")
        dense.add_head(); dense.heads[-1].to(DEVICE)
        train_one_task_dense(dense, tr_loaders[t], task_id=t, n_epochs=epochs_per_task)
        for ev in range(t + 1):
            results["Dense"][ev].append(accuracy(dense, te_loaders[ev], ev))
        print(f"→ avg acc: {np.mean([results['Dense'][ev][-1] for ev in range(t+1)]):.1f}%")

    pcount["Dense"] = [sum(p.numel() for p in dense.parameters())] * (N_TASKS * epochs_per_task)

    # ── EWC ──────────────────────────────────────────────────────
    print("\n  [3/4] EWC")
    ewc_base = DenseMLPMultiHead(784, [128, 64], 2).to(DEVICE)
    ewc_w    = EWCWrapper(ewc_base, lam=5000)

    for t in range(N_TASKS):
        print(f"    Task {t+1}/{N_TASKS}", end=" ")
        ewc_base.add_head(); ewc_base.heads[-1].to(DEVICE)
        train_one_task_ewc(ewc_w, tr_loaders[t], task_id=t, n_epochs=epochs_per_task)
        ewc_w.compute_fisher(tr_loaders[t], task_id=t, n=400)
        for ev in range(t + 1):
            results["EWC"][ev].append(accuracy(ewc_base, te_loaders[ev], ev))
        print(f"→ avg acc: {np.mean([results['EWC'][ev][-1] for ev in range(t+1)]):.1f}%")

    # ── PNN ──────────────────────────────────────────────────────
    print("\n  [4/4] PNN")
    pnn = PNN(784, 64, 2).to(DEVICE)

    for t in range(N_TASKS):
        print(f"    Task {t+1}/{N_TASKS}", end=" ")
        pnn.add_column(); pnn.to(DEVICE)
        train_one_task_pnn(pnn, tr_loaders[t], task_id=t, n_epochs=epochs_per_task)
        for ev in range(t + 1):
            results["PNN"][ev].append(accuracy(pnn, te_loaders[ev], ev))
        print(f"→ avg acc: {np.mean([results['PNN'][ev][-1] for ev in range(t+1)]):.1f}%")

    pcount["PNN"] = [pnn.total_params()] * (N_TASKS * epochs_per_task)

    # ── Summary ──────────────────────────────────────────────────
    summary = {}
    for m in results:
        t1_ret = results[m][0][-1] if results[m][0] else 0.0
        final  = [results[m][t][-1] for t in range(N_TASKS) if results[m][t]]
        summary[m] = {
            "task1_retention": t1_ret,
            "avg_accuracy"   : float(np.mean(final)) if final else 0.0,
            "forgetting_rate": 100.0 - t1_ret,
        }

    # ── Jaccard sub-network overlap ───────────────────────────────
    jaccard: Dict[str, float] = {}
    if dtn_masks:
        ref = dtn_masks[0].float()
        for t in range(1, N_TASKS):
            oth     = dtn_masks[t].float()
            max_len = max(len(ref), len(oth))
            ref_pad = torch.zeros(max_len)
            oth_pad = torch.zeros(max_len)
            ref_pad[: len(ref)] = ref
            oth_pad[: len(oth)] = oth
            inter   = (ref_pad * oth_pad).sum().item()
            union   = ((ref_pad + oth_pad) > 0).float().sum().item()
            jaccard[f"T1-T{t+1}"] = 100.0 * inter / max(union, 1)

    return results, summary, pcount, jaccard, dtn
