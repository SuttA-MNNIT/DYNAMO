#!/usr/bin/env python3
"""
run_all.py — Full DTN experiment pipeline
==========================================

Usage
-----
  python scripts/run_all.py                         # all experiments, default config
  python scripts/run_all.py --config my.yaml        # custom config
  python scripts/run_all.py --no-cifar              # skip CIFAR-100 (much faster)
  python scripts/run_all.py --no-ablations          # skip ablation grid
  python scripts/run_all.py --output-dir /tmp/out   # override output directory

The script reads hyperparameters from ``configs/default.yaml`` (or a file
supplied via ``--config``) and writes all figures, tables, and a JSON
summary to the output directory.
"""

import argparse
import os
import sys
import time
import warnings

# ── make package importable when run as a script ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run DTN experiment suite")
    parser.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--no-cifar",     action="store_true", help="Skip Experiment 2 (CIFAR-100)")
    parser.add_argument("--no-ablations", action="store_true", help="Skip ablation studies")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    SEED = cfg.get("seed", 42)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT = args.output_dir or cfg.get("output_dir", "./results")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    print(f"[DTN] Device : {DEVICE}")
    print(f"[DTN] PyTorch: {torch.__version__}")
    print(f"[DTN] Output : {OUT}\n")

    t0 = time.time()

    # ── lazy imports (keep startup fast) ─────────────────────────────────────
    from dtn.experiments.exp1_split_mnist import run as run_mnist
    from dtn.experiments.ablations import (
        ablation_lambda,
        ablation_hcrit,
        ablation_sparsity,
    )
    from dtn.utils.reporting import print_and_save_tables, save_results_json
    from dtn.viz.plots import (
        fig_mnist_overview,
        fig_accuracy_heatmaps,
        fig_ablations,
        fig_cifar_results,
    )

    # ── Experiment 1: Split-MNIST ─────────────────────────────────────────────
    ept_mnist = cfg["experiments"]["split_mnist"]["epochs_per_task"]
    m_res, m_sum, m_pc, m_jac, dtn_mnist = run_mnist(epochs_per_task=ept_mnist)

    # ── Experiment 2: Sequential CIFAR-100 ───────────────────────────────────
    c_res, c_sum, c_pc, dtn_cifar, n_tasks = {}, {}, {}, None, 0
    run_cifar = (
        not args.no_cifar
        and cfg["experiments"]["seq_cifar100"].get("enabled", True)
    )
    if run_cifar:
        from dtn.experiments.exp2_seq_cifar100 import run as run_cifar100
        ept_cifar = cfg["experiments"]["seq_cifar100"]["epochs_per_task"]
        n_t       = cfg["experiments"]["seq_cifar100"]["n_tasks"]
        c_res, c_sum, c_pc, dtn_cifar, n_tasks = run_cifar100(
            n_tasks=n_t, epochs_per_task=ept_cifar
        )

    # ── Ablations ─────────────────────────────────────────────────────────────
    ab_lam = ab_hc = ab_sp = {}
    run_abl = (
        not args.no_ablations
        and cfg["experiments"]["ablations"].get("enabled", True)
    )
    if run_abl:
        ept_abl = cfg["experiments"]["ablations"]["epochs_per_task"]
        ab_lam  = ablation_lambda(epochs_per_task=ept_abl)
        ab_hc   = ablation_hcrit(epochs_per_task=ept_abl)
        ab_sp   = ablation_sparsity(epochs_per_task=ept_abl)

    # ── Tables ────────────────────────────────────────────────────────────────
    print_and_save_tables(m_sum, c_sum, m_jac, dtn_mnist, OUT)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[Plotting]")
    fig_mnist_overview(m_res, m_sum, m_pc, m_jac, dtn_mnist, OUT)
    fig_accuracy_heatmaps(m_res, 5, ["DTN", "Dense", "EWC", "PNN"], OUT)

    if run_abl and ab_lam and ab_hc and ab_sp:
        fig_ablations(ab_lam, ab_hc, ab_sp, OUT)

    if run_cifar and dtn_cifar is not None:
        fig_cifar_results(c_res, c_sum, c_pc, dtn_cifar, n_tasks, OUT)

    # ── JSON ──────────────────────────────────────────────────────────────────
    save_results_json(m_sum, c_sum, m_jac, ab_lam, ab_hc, ab_sp, m_pc["DTN"], OUT)

    elapsed = (time.time() - t0) / 60
    print(f"\n✓ Total runtime : {elapsed:.1f} min")
    print(f"✓ All outputs   → {OUT}/")


if __name__ == "__main__":
    main()
