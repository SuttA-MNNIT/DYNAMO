"""
Baseline continual-learning models.

  DenseMLPMultiHead – plain SGD fine-tuning (catastrophic-forgetting oracle)
  EWCWrapper        – Elastic Weight Consolidation (Kirkpatrick et al., 2017)
  PNN               – Progressive Neural Networks (Rusu et al., 2016)
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dense MLP ────────────────────────────────────────────────────────────────


class DenseMLPMultiHead(nn.Module):
    """Standard MLP with one output head per task (no CL mechanism)."""

    def __init__(self, in_f: int, hiddens: List[int], n_cpt: int):
        super().__init__()
        layers, widths = [], [in_f] + hiddens
        for i in range(len(widths) - 1):
            layers += [nn.Linear(widths[i], widths[i + 1]), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.heads    = nn.ModuleList()
        self.n_cpt    = n_cpt
        self._last_w  = hiddens[-1]

    def add_head(self) -> None:
        h = nn.Linear(self._last_w, self.n_cpt)
        nn.init.kaiming_normal_(h.weight, nonlinearity="linear")
        self.heads.append(h)

    def forward(self, x: torch.Tensor, task_id: int = -1) -> torch.Tensor:
        return self.heads[task_id](self.backbone(x.view(x.size(0), -1)))


# ── EWC ──────────────────────────────────────────────────────────────────────


class EWCWrapper:
    """
    Elastic Weight Consolidation wrapped around a DenseMLPMultiHead.

    After finishing each task call ``compute_fisher`` to anchor weights.
    Add ``penalty()`` to the training loss during subsequent tasks.

    Parameters
    ----------
    model : DenseMLPMultiHead
    lam   : EWC regularisation coefficient λ.
    """

    def __init__(self, model: DenseMLPMultiHead, lam: float = 5000.0):
        self.model = model
        self.lam   = lam
        self._fish: Dict[int, Dict[str, torch.Tensor]] = {}
        self._star: Dict[int, Dict[str, torch.Tensor]] = {}

    def compute_fisher(
        self,
        loader: torch.utils.data.DataLoader,
        task_id: int,
        n: int = 300,
    ) -> None:
        """Approximate diagonal Fisher for ``task_id`` using ``n`` samples."""
        fish = {
            nm: torch.zeros_like(p)
            for nm, p in self.model.named_parameters()
        }
        self.model.eval()
        cnt = 0
        for x, y in loader:
            if cnt >= n:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.model.zero_grad()
            F.cross_entropy(self.model(x, task_id), y).backward()
            for nm, p in self.model.named_parameters():
                if p.grad is not None:
                    fish[nm] += p.grad.data ** 2
            cnt += x.size(0)
        for nm in fish:
            fish[nm] /= max(cnt, 1)
        self._fish[task_id] = {nm: v.clone() for nm, v in fish.items()}
        self._star[task_id] = {
            nm: p.data.clone() for nm, p in self.model.named_parameters()
        }

    def penalty(self) -> torch.Tensor:
        """Return the EWC quadratic-anchor regularisation term."""
        loss       = torch.tensor(0.0, device=DEVICE)
        curr_params = dict(self.model.named_parameters())
        for tid in self._fish:
            for nm in self._fish[tid]:
                if nm in curr_params:
                    p     = curr_params[nm]
                    loss += (
                        self._fish[tid][nm] * (p - self._star[tid][nm]) ** 2
                    ).sum()
        return self.lam * loss / 2.0


# ── Progressive Neural Networks ──────────────────────────────────────────────


class PNN(nn.Module):
    """
    Progressive Neural Networks — one isolated column per task.
    All previous columns are frozen; only the latest column trains.

    Parameters
    ----------
    in_f   : input dimensionality
    hidden : width of each hidden layer (same for all columns)
    n_cpt  : classes per task
    """

    def __init__(self, in_f: int, hidden: int, n_cpt: int):
        super().__init__()
        self.in_f  = in_f
        self.h     = hidden
        self.n_cpt = n_cpt
        self.columns: nn.ModuleList = nn.ModuleList()

    def add_column(self) -> None:
        col = nn.Sequential(
            nn.Linear(self.in_f, self.h), nn.ReLU(),
            nn.Linear(self.h, self.h),   nn.ReLU(),
            nn.Linear(self.h, self.n_cpt),
        )
        self.columns.append(col)
        # Freeze all previous columns
        for i in range(len(self.columns) - 1):
            for p in self.columns[i].parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, task_id: int = -1) -> torch.Tensor:
        return self.columns[task_id](x.view(x.size(0), -1))

    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
