"""
ThermoLayer: masked linear layer with per-edge metabolic state and per-node heat.

State buffers (non-parameter, not in optimizer):
  mask    – binary adjacency (float, 0/1)
  fisher  – EMA of squared weight gradient  → metabolic currency
  energy  – running budget per edge          → gates survival
  heat    – EMA of squared bias gradient     → triggers mitosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermoLayer(nn.Module):
    """
    Masked linear layer with per-edge metabolic state and per-node heat.

    Deviations from paper (see README for full list):
      - Uses masked-dense ops instead of true COO sparse tensors.
      - Energy initialised to 1.0 for all active edges (paper is silent on this).
      - Bias-gradient heat uses EMA over squared bias gradients (≈ Eq. 4).
    """

    def __init__(self, in_f: int, out_f: int, sparsity: float = 0.95):
        super().__init__()
        self.in_f  = in_f
        self.out_f = out_f

        # Kaiming init scaled for sparse inputs
        std = (2.0 / in_f) ** 0.5
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * std)
        self.bias   = nn.Parameter(torch.zeros(out_f))

        # Erdős-Rényi sparse mask; every row guaranteed ≥ 1 connection
        m = (torch.rand(out_f, in_f) > sparsity).float()
        for i in range(out_f):
            if m[i].sum() == 0:
                m[i, torch.randint(0, in_f, (1,))] = 1.0

        self.register_buffer("mask",   m)
        self.register_buffer("fisher", torch.zeros(out_f, in_f))
        self.register_buffer("energy", torch.ones(out_f, in_f))
        self.register_buffer("heat",   torch.zeros(out_f))

        # CPU gradient accumulators (populated by backward hooks)
        self._w_acc = torch.zeros(out_f, in_f)
        self._b_acc = torch.zeros(out_f)
        self._n     = 0

    # ── forward ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    # ── convenience props ─────────────────────────────────────────

    @property
    def active_edges(self) -> int:
        return int(self.mask.sum().item())

    def sparsity_ratio(self) -> float:
        return 1.0 - self.active_edges / self.mask.numel()
