"""
Experiment 2 — Sequential CIFAR-100
=====================================
Uses a frozen pretrained ViT-B/16 as a feature extractor, then trains
DTN, Dense-SGD, and EWC classifiers on 10 disjoint CIFAR-100 task groups.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dtn.data import get_seq_cifar100
from dtn.models import DTN, DenseMLPMultiHead, EWCWrapper
from dtn.training import accuracy

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIM = 768  # ViT-B/16 output dimension


# ── ViT feature extractor ────────────────────────────────────────────────────


def _get_frozen_vit() -> nn.Module:
    """Load pretrained ViT-B/16 with the classification head removed."""
    import torchvision.models as tvm
    vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
    vit.heads = nn.Identity()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


@torch.no_grad()
def _extract_features(
    vit: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vit.eval()
    xs, ys = [], []
    for x, y in loader:
        xs.append(vit(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def _feat_loader(
    vit: nn.Module,
    raw_loader: DataLoader,
    bs: int = 128,
) -> DataLoader:
    fx, fy = _extract_features(vit, raw_loader, DEVICE)
    return DataLoader(TensorDataset(fx, fy), batch_size=bs, shuffle=True, num_workers=0)


# ── main experiment ───────────────────────────────────────────────────────────


def run(
    n_tasks: int = 10,
    epochs_per_task: int = 10,
) -> Tuple[dict, dict, dict, DTN, int]:
    """
    Run Experiment 2.

    Returns
    -------
    results  : per-model, per-task accuracy traces
    summary  : end-of-sequence scalar metrics per model
    pcount   : active-parameter history (DTN only; others are static)
    dtn_c    : trained DTN instance
    n_tasks  : echoed back for downstream plot helpers
    """
    print("\n" + "═" * 62)
    print(f"  EXPERIMENT 2 — Sequential CIFAR-100  ({n_tasks} tasks)")
    print("═" * 62)

    tr_loaders, te_loaders, cpt = get_seq_cifar100(n_tasks=n_tasks, batch_size=64)
    results = {m: defaultdict(list) for m in ["DTN", "Dense", "EWC"]}
    pcount  = {"DTN": [], "Dense": [], "EWC": []}

    # ── DTN + ViT-B/16 ───────────────────────────────────────────
    print("\n  [1/3] DTN + ViT-B/16")
    vit_dtn = _get_frozen_vit().to(DEVICE)

    dtn_c = DTN(
        input_size=FEAT_DIM,
        hidden_sizes=[512, 256],
        n_classes_per_task=cpt,
        sparsity=0.50,
        lam=1e-4, alpha=0.9, beta=2.0,
        h_crit=0.30, max_w=1024, cap_mitosis=16,
    ).to(DEVICE)

    for t in range(n_tasks):
        print(f"    Task {t+1}/{n_tasks}", end=" ")
        fl = _feat_loader(vit_dtn, tr_loaders[t])
        dtn_c.add_head(); dtn_c.heads[-1].to(DEVICE)

        opt = torch.optim.SGD(dtn_c.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5)
        for _ in range(epochs_per_task):
            dtn_c.train()
            for x, y in fl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                F.cross_entropy(dtn_c(x, t), y).backward()
                opt.step()
                dtn_c.thermo_update()
            dtn_c.mitosis_step()
            dtn_c.record()

        for ev in range(t + 1):
            fl_ev = _feat_loader(vit_dtn, te_loaders[ev], bs=256)
            cor, tot = 0, 0
            dtn_c.eval()
            with torch.no_grad():
                for x, y in fl_ev:
                    x, y  = x.to(DEVICE), y.to(DEVICE)
                    cor  += (dtn_c(x, ev).argmax(1) == y).sum().item()
                    tot  += y.size(0)
            results["DTN"][ev].append(100.0 * cor / max(tot, 1))
        print(f"→ avg acc: {np.mean([results['DTN'][ev][-1] for ev in range(t+1)]):.1f}%")

    pcount["DTN"] = list(dtn_c.param_hist)

    # ── Dense + ViT-B/16 ─────────────────────────────────────────
    print("\n  [2/3] Dense SGD + ViT-B/16")
    vit_d   = _get_frozen_vit().to(DEVICE)
    dense_c = DenseMLPMultiHead(FEAT_DIM, [512, 256], cpt).to(DEVICE)

    for t in range(n_tasks):
        print(f"    Task {t+1}/{n_tasks}", end=" ")
        fl = _feat_loader(vit_d, tr_loaders[t])
        dense_c.add_head(); dense_c.heads[-1].to(DEVICE)

        opt = torch.optim.SGD(dense_c.parameters(), lr=0.01, momentum=0.9)
        for _ in range(epochs_per_task):
            dense_c.train()
            for x, y in fl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                F.cross_entropy(dense_c(x, t), y).backward()
                opt.step()

        for ev in range(t + 1):
            fl_ev = _feat_loader(vit_d, te_loaders[ev], bs=256)
            cor, tot = 0, 0
            dense_c.eval()
            with torch.no_grad():
                for x, y in fl_ev:
                    x, y  = x.to(DEVICE), y.to(DEVICE)
                    cor  += (dense_c(x, ev).argmax(1) == y).sum().item()
                    tot  += y.size(0)
            results["Dense"][ev].append(100.0 * cor / max(tot, 1))
        print(f"→ avg acc: {np.mean([results['Dense'][ev][-1] for ev in range(t+1)]):.1f}%")

    # ── EWC + ViT-B/16 ───────────────────────────────────────────
    print("\n  [3/3] EWC + ViT-B/16")
    vit_e  = _get_frozen_vit().to(DEVICE)
    ewc_c  = DenseMLPMultiHead(FEAT_DIM, [512, 256], cpt).to(DEVICE)
    ewc_wc = EWCWrapper(ewc_c, lam=1000)

    for t in range(n_tasks):
        print(f"    Task {t+1}/{n_tasks}", end=" ")
        fl = _feat_loader(vit_e, tr_loaders[t])
        ewc_c.add_head(); ewc_c.heads[-1].to(DEVICE)

        opt = torch.optim.SGD(ewc_c.parameters(), lr=0.01, momentum=0.9)
        for _ in range(epochs_per_task):
            ewc_c.train()
            for x, y in fl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                (F.cross_entropy(ewc_c(x, t), y) + ewc_wc.penalty()).backward()
                opt.step()
        ewc_wc.compute_fisher(fl, task_id=t, n=200)

        for ev in range(t + 1):
            fl_ev = _feat_loader(vit_e, te_loaders[ev], bs=256)
            cor, tot = 0, 0
            ewc_c.eval()
            with torch.no_grad():
                for x, y in fl_ev:
                    x, y  = x.to(DEVICE), y.to(DEVICE)
                    cor  += (ewc_c(x, ev).argmax(1) == y).sum().item()
                    tot  += y.size(0)
            results["EWC"][ev].append(100.0 * cor / max(tot, 1))
        print(f"→ avg acc: {np.mean([results['EWC'][ev][-1] for ev in range(t+1)]):.1f}%")

    # ── Summary ──────────────────────────────────────────────────
    summary = {}
    for m in results:
        t1    = results[m][0][-1] if results[m][0] else 0.0
        final = [results[m][t][-1] for t in range(n_tasks) if results[m][t]]
        summary[m] = {
            "task1_retention": t1,
            "avg_accuracy"   : float(np.mean(final)) if final else 0.0,
        }

    return results, summary, pcount, dtn_c, n_tasks
