"""
Unit tests for DTN model components.

Run with:  pytest tests/ -v
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dtn.models.thermo_layer import ThermoLayer
from dtn.models.dtn import DTN
from dtn.models.baselines import DenseMLPMultiHead, EWCWrapper, PNN
from dtn.training.trainers import accuracy


# ── ThermoLayer ───────────────────────────────────────────────────────────────


class TestThermoLayer:
    def test_forward_shape(self):
        layer = ThermoLayer(16, 8, sparsity=0.5)
        x = torch.randn(4, 16)
        assert layer(x).shape == (4, 8)

    def test_mask_respected(self):
        layer = ThermoLayer(16, 8, sparsity=0.0)
        layer.mask.zero_()
        layer.mask[0, 0] = 1.0
        x = torch.ones(1, 16)
        out = layer(x)
        # Only one edge per row is active for row 0; others are all-zero weights
        assert out.shape == (1, 8)

    def test_at_least_one_edge_per_row(self):
        layer = ThermoLayer(32, 16, sparsity=0.99)
        assert (layer.mask.sum(dim=1) >= 1).all()

    def test_sparsity_ratio(self):
        layer = ThermoLayer(100, 50, sparsity=0.9)
        ratio = layer.sparsity_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_active_edges_positive(self):
        layer = ThermoLayer(16, 8, sparsity=0.5)
        assert layer.active_edges > 0


# ── DTN ──────────────────────────────────────────────────────────────────────


class TestDTN:
    def _make_dtn(self):
        return DTN(
            input_size=16,
            hidden_sizes=[8, 4],
            n_classes_per_task=2,
            sparsity=0.5,
            lam=1e-4, alpha=0.9, beta=1.0,
            h_crit=0.5, max_w=32, cap_mitosis=2,
        )

    def test_forward_shape(self):
        dtn = self._make_dtn()
        dtn.add_head()
        x = torch.randn(5, 16)
        assert dtn(x, 0).shape == (5, 2)

    def test_add_multiple_heads(self):
        dtn = self._make_dtn()
        for _ in range(3):
            dtn.add_head()
        assert len(dtn.heads) == 3
        x = torch.randn(4, 16)
        for t in range(3):
            assert dtn(x, t).shape == (4, 2)

    def test_thermo_update_runs(self):
        dtn = self._make_dtn()
        dtn.add_head()
        x, y = torch.randn(8, 16), torch.randint(0, 2, (8,))
        opt = torch.optim.SGD(dtn.parameters(), lr=0.01)
        opt.zero_grad()
        F.cross_entropy(dtn(x, 0), y).backward()
        opt.step()
        n_pruned = dtn.thermo_update()
        assert isinstance(n_pruned, int)

    def test_mitosis_runs(self):
        dtn = self._make_dtn()
        dtn.add_head()
        # Force heat above h_crit
        for L in dtn.backbone:
            L.heat.fill_(1.0)
        before = sum(dtn.hidden_sizes)
        dtn.mitosis_step()
        after = sum(dtn.hidden_sizes)
        assert after >= before   # may stay same if cap hits max_w

    def test_count_active_params_positive(self):
        dtn = self._make_dtn()
        dtn.add_head()
        assert dtn.count_active_params() > 0

    def test_record_increments_epoch(self):
        dtn = self._make_dtn()
        dtn.add_head()
        assert dtn._epoch == 0
        dtn.record()
        assert dtn._epoch == 1
        assert len(dtn.param_hist) == 1

    def test_sparsity_per_layer_length(self):
        dtn = self._make_dtn()
        assert len(dtn.sparsity_per_layer()) == len(dtn.backbone)

    def test_adjacency_mask_flat_bool(self):
        dtn = self._make_dtn()
        mask = dtn.adjacency_mask_flat(0)
        assert mask.dtype == torch.bool


# ── Baselines ────────────────────────────────────────────────────────────────


class TestDenseMLPMultiHead:
    def test_forward(self):
        m = DenseMLPMultiHead(16, [8, 4], 3)
        m.add_head()
        x = torch.randn(5, 16)
        assert m(x, 0).shape == (5, 3)

    def test_multiple_heads_independent(self):
        m = DenseMLPMultiHead(16, [8, 4], 2)
        m.add_head(); m.add_head()
        x = torch.randn(3, 16)
        out0 = m(x, 0)
        out1 = m(x, 1)
        assert out0.shape == out1.shape == (3, 2)


class TestEWCWrapper:
    def test_penalty_zero_before_training(self):
        model = DenseMLPMultiHead(16, [8], 2)
        model.add_head()
        ewc = EWCWrapper(model, lam=100)
        # No Fisher computed yet → penalty should be zero
        assert ewc.penalty().item() == pytest.approx(0.0)

    def test_compute_fisher_and_penalty(self):
        model = DenseMLPMultiHead(16, [8], 2)
        model.add_head()
        ewc = EWCWrapper(model, lam=100)
        xs = torch.randn(20, 16)
        ys = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(xs, ys), batch_size=10)
        ewc.compute_fisher(loader, task_id=0, n=20)
        assert ewc.penalty().item() >= 0.0


class TestPNN:
    def test_add_column_and_forward(self):
        pnn = PNN(16, 8, 2)
        pnn.add_column()
        x = torch.randn(4, 16)
        assert pnn(x, 0).shape == (4, 2)

    def test_previous_columns_frozen(self):
        pnn = PNN(16, 8, 2)
        pnn.add_column()
        pnn.add_column()
        for p in pnn.columns[0].parameters():
            assert not p.requires_grad

    def test_trainable_params_only_latest(self):
        pnn = PNN(16, 8, 2)
        pnn.add_column()
        pnn.add_column()
        tp = pnn.trainable_params()
        assert len(tp) > 0
        # All trainable params belong to the latest column
        latest_ids = {id(p) for p in pnn.columns[-1].parameters()}
        for p in tp:
            assert id(p) in latest_ids


# ── Accuracy helper ───────────────────────────────────────────────────────────


class TestAccuracy:
    def test_perfect_model(self):
        """A model that always predicts class 0 on all-class-0 data → 100 %."""
        class AlwaysZero(torch.nn.Module):
            def forward(self, x, task_id=0):
                return torch.tensor([[10.0, -10.0]]).expand(x.size(0), -1)

        xs = torch.randn(20, 4)
        ys = torch.zeros(20, dtype=torch.long)
        loader = DataLoader(TensorDataset(xs, ys), batch_size=10)
        acc = accuracy(AlwaysZero(), loader, task_id=0)
        assert acc == pytest.approx(100.0)
