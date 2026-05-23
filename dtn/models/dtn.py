"""
Dynamic Thermodynamic Network (DTN)
====================================
Open-system continual-learning network governed by Algorithmic Thermodynamics.

Architecture: shared sparse backbone → per-task output heads.
Topology evolves via:
  (a) Synaptic starvation  – prune edge when energy < 0
  (b) Algorithmic mitosis  – duplicate node when heat > H_crit
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtn.models.thermo_layer import ThermoLayer


class DTN(nn.Module):
    """
    Dynamic Thermodynamic Network.

    Parameters
    ----------
    input_size : int
        Dimensionality of flattened input features.
    hidden_sizes : list[int]
        Initial width of each backbone layer.
    n_classes_per_task : int
        Output classes per task head.
    sparsity : float
        Initial edge-dropout probability for Erdős-Rényi mask.
    lam : float
        Basal metabolic cost λ (Eq. 3).
    alpha : float
        Fisher / heat EMA retention α (Eqs. 2, 4).
    beta : float
        Nutrient absorption coefficient β (Eq. 3).
    h_crit : float
        Heat threshold triggering mitosis.
    max_w : int
        Maximum neurons per backbone layer.
    cap_mitosis : int
        Maximum new neurons spawned per mitosis step.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        n_classes_per_task: int = 2,
        sparsity: float = 0.95,
        lam: float = 1e-4,
        alpha: float = 0.9,
        beta: float = 1.0,
        h_crit: float = 0.85,
        max_w: int = 256,
        cap_mitosis: int = 6,
    ):
        super().__init__()
        self.input_size   = input_size
        self.n_cpt        = n_classes_per_task
        self.lam          = lam
        self.alpha        = alpha
        self.beta         = beta
        self.h_crit       = h_crit
        self.max_w        = max_w
        self.cap          = cap_mitosis
        self.hidden_sizes = list(hidden_sizes)
        self.sparsity     = sparsity

        # Build backbone
        self.backbone = nn.ModuleList()
        widths = [input_size] + self.hidden_sizes
        for i in range(len(widths) - 1):
            self.backbone.append(ThermoLayer(widths[i], widths[i + 1], sparsity))

        # Per-task output heads (added dynamically via add_head)
        self.heads: nn.ModuleList = nn.ModuleList()

        self._hooks: list = []
        self._register_hooks()

        # Telemetry
        self.param_hist : List[int]          = []
        self.mitosis_log: List[tuple]        = []
        self.prune_log  : List[tuple]        = []
        self._epoch     : int                = 0

    # ── public API ───────────────────────────────────────────────

    def add_head(self) -> None:
        """Append a new output head for the next task."""
        last_w = self.hidden_sizes[-1]
        head   = nn.Linear(last_w, self.n_cpt)
        nn.init.kaiming_normal_(head.weight, nonlinearity="linear")
        self.heads.append(head)

    def forward(self, x: torch.Tensor, task_id: int = -1) -> torch.Tensor:
        h = x.view(x.size(0), -1)
        for layer in self.backbone:
            h = F.relu(layer(h))
        return self.heads[task_id](h)

    # ── thermodynamic update (call after optimizer.step) ─────────

    def thermo_update(self) -> int:
        """Update Fisher, energy, heat; prune dead edges. Returns #pruned."""
        n_pruned = 0
        with torch.no_grad():
            for L in self.backbone:
                if L._n == 0:
                    continue
                dev  = L.weight.device
                w_sq = (L._w_acc / L._n).to(dev)
                b_sq = (L._b_acc / L._n).to(dev)

                # Eq. (2) Fisher EMA
                L.fisher = self.alpha * L.fisher + (1 - self.alpha) * w_sq

                # Eq. (3) Energy update (active edges only)
                L.energy += (self.beta * L.fisher - self.lam) * L.mask

                # Synaptic starvation → prune
                dead = (L.energy < 0) & (L.mask > 0)
                n_pruned    += int(dead.sum().item())
                L.mask[dead] = 0.0
                L.energy[dead] = 0.0

                # Eq. (4) Heat EMA
                L.heat = self.alpha * L.heat + (1 - self.alpha) * b_sq

                # Reset accumulators
                L._w_acc.zero_()
                L._b_acc.zero_()
                L._n = 0

        self.prune_log.append((self._epoch, n_pruned))
        return n_pruned

    def mitosis_step(self) -> int:
        """
        Duplicate hot neurons (heat > h_crit) in every backbone layer.
        Returns total new neurons spawned this step.
        """
        total = 0
        for li in range(len(self.backbone)):
            total += self._layer_mitosis(li)
        if total:
            self._sync_heads()
            self._register_hooks()
            self.mitosis_log.append((self._epoch, total))
        return total

    # ── bookkeeping ──────────────────────────────────────────────

    def record(self) -> None:
        """Increment epoch counter and snapshot active-parameter count."""
        self._epoch += 1
        self.param_hist.append(self.count_active_params())

    def count_active_params(self) -> int:
        n = 0
        for L in self.backbone:
            n += L.active_edges + L.out_f
        for h in self.heads:
            n += h.weight.numel() + h.bias.numel()
        return n

    def sparsity_per_layer(self) -> List[float]:
        return [L.sparsity_ratio() for L in self.backbone]

    def adjacency_mask_flat(self, layer_idx: int = 0) -> torch.Tensor:
        return self.backbone[layer_idx].mask.cpu().bool().flatten()

    # ── private helpers ──────────────────────────────────────────

    def _register_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

        for layer in self.backbone:
            def _bias_hook(g, L=layer):
                # Scale by 50 so heat can realistically reach h_crit (0.85)
                L._b_acc += torch.abs(g.detach().cpu()) * 50.0
                L._n     += 1
                return g

            def _weight_hook(g, L=layer):
                L._w_acc += (g.detach().cpu() ** 2) * L.mask.cpu()
                return g

            self._hooks.append(layer.bias.register_hook(_bias_hook))
            self._hooks.append(layer.weight.register_hook(_weight_hook))

    def _layer_mitosis(self, li: int) -> int:
        L   = self.backbone[li]
        dev = L.weight.device
        hot = (L.heat > self.h_crit).nonzero(as_tuple=True)[0]
        if not len(hot):
            return 0

        avail = self.max_w - L.out_f
        if avail <= 0:
            return 0

        n_new   = min(len(hot), avail, self.cap)
        hot     = hot[:n_new]
        new_out = L.out_f + n_new

        NL = ThermoLayer(L.in_f, new_out, sparsity=0.0).to(dev)
        with torch.no_grad():
            # Copy existing state
            NL.weight.data[:L.out_f] = L.weight.data
            NL.bias.data[:L.out_f]   = L.bias.data
            NL.mask[:L.out_f]        = L.mask
            NL.fisher[:L.out_f]      = L.fisher
            NL.energy[:L.out_f]      = L.energy
            NL.heat[:L.out_f]        = L.heat

            # Reset parent heat
            for idx in hot:
                NL.heat[idx] = 0.0

            # Eq. (6) symmetry-breaking spawn
            eps_std = 0.01
            for k, pidx in enumerate(hot):
                nidx = L.out_f + k
                eps  = torch.randn_like(L.weight.data[pidx]) * eps_std
                NL.weight.data[nidx] = L.weight.data[pidx] + eps
                NL.bias.data[nidx]   = L.bias.data[pidx] - eps.mean()
                # Sparse random-walk init (10 % density)
                row = (torch.rand(L.in_f, device=dev) < 0.10).float()
                if row.sum() == 0:
                    row[torch.randint(0, L.in_f, (1,))] = 1.0
                NL.mask[nidx]   = row
                NL.energy[nidx] = 1.0
                NL.heat[nidx]   = 0.0

        self.backbone[li]     = NL
        self.hidden_sizes[li] = new_out

        if li + 1 < len(self.backbone):
            self._expand_layer_inputs(li + 1, L.out_f, n_new, dev)

        return n_new

    def _expand_layer_inputs(
        self, li: int, old_in: int, n_new: int, dev: torch.device
    ) -> None:
        L      = self.backbone[li]
        new_in = old_in + n_new
        NL = ThermoLayer(new_in, L.out_f, sparsity=0.0).to(dev)
        with torch.no_grad():
            NL.weight.data[:, :old_in]  = L.weight.data
            NL.bias.data                = L.bias.data
            NL.mask[:, :old_in]         = L.mask
            NL.mask[:, old_in:]         = (
                torch.rand(L.out_f, n_new, device=dev) < 0.1
            ).float()
            NL.fisher[:, :old_in]       = L.fisher
            NL.energy[:, :old_in]       = L.energy
            NL.energy[:, old_in:]       = 1.0
            NL.heat                     = L.heat
        self.backbone[li] = NL

    def _sync_heads(self) -> None:
        """Resize output heads when the last backbone layer has grown."""
        last_w = self.hidden_sizes[-1]
        for i, head in enumerate(self.heads):
            if head.in_features == last_w:
                continue
            old_in    = head.in_features
            dev       = head.weight.device
            new_head  = nn.Linear(last_w, self.n_cpt).to(dev)
            with torch.no_grad():
                new_head.weight.data[:, :old_in] = head.weight.data
                new_head.bias.data               = head.bias.data
                nn.init.normal_(new_head.weight.data[:, old_in:], 0, 0.01)
            self.heads[i] = new_head
