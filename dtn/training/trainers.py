"""
Training loops for DTN and all baseline models.

Each ``train_one_task_*`` function trains a model for a single task
and returns nothing (or a loss history for DTN).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dtn.models.dtn import DTN
from dtn.models.baselines import DenseMLPMultiHead, EWCWrapper, PNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── evaluation helper ─────────────────────────────────────────────────────────


def accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    task_id: int,
) -> float:
    """Return top-1 accuracy (%) of ``model`` on ``loader`` for ``task_id``."""
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x, task_id)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
    return 100.0 * cor / max(tot, 1)


# ── DTN ──────────────────────────────────────────────────────────────────────


def train_one_task_dtn(
    dtn: DTN,
    loader: DataLoader,
    lr: float = 0.01,
    n_epochs: int = 12,
) -> List[float]:
    """
    Train the most recently added DTN head for one task.

    After each batch:  ``thermo_update()`` (metabolic update + pruning).
    After each epoch:  ``mitosis_step()``  (neuron duplication if hot).

    Returns list of per-epoch average cross-entropy losses.
    """
    opt     = torch.optim.SGD(dtn.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    task_id = len(dtn.heads) - 1
    losses  = []

    for _ in range(n_epochs):
        dtn.train()
        ep_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(dtn(x, task_id), y)
            loss.backward()
            opt.step()
            dtn.thermo_update()
            ep_loss += loss.item()
        dtn.mitosis_step()
        dtn.record()
        losses.append(ep_loss / len(loader))

    return losses


# ── Dense SGD ────────────────────────────────────────────────────────────────


def train_one_task_dense(
    model: DenseMLPMultiHead,
    loader: DataLoader,
    task_id: int,
    lr: float = 0.01,
    n_epochs: int = 12,
) -> None:
    """Standard SGD fine-tuning on a single task."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    for _ in range(n_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x, task_id), y).backward()
            opt.step()


# ── EWC ──────────────────────────────────────────────────────────────────────


def train_one_task_ewc(
    ewc_w: EWCWrapper,
    loader: DataLoader,
    task_id: int,
    lr: float = 0.01,
    n_epochs: int = 12,
) -> None:
    """SGD with EWC penalty on a single task."""
    opt = torch.optim.SGD(ewc_w.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    for _ in range(n_epochs):
        ewc_w.model.train()
        for x, y in loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            tl = F.cross_entropy(ewc_w.model(x, task_id), y)
            pl = ewc_w.penalty()
            (tl + pl).backward()
            opt.step()


# ── PNN ──────────────────────────────────────────────────────────────────────


def train_one_task_pnn(
    pnn: PNN,
    loader: DataLoader,
    task_id: int,
    lr: float = 0.01,
    n_epochs: int = 12,
) -> None:
    """Train the latest (unfrozen) PNN column on a single task."""
    opt = torch.optim.SGD(pnn.trainable_params(), lr=lr, momentum=0.9)
    for _ in range(n_epochs):
        pnn.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(pnn(x, task_id), y).backward()
            opt.step()
