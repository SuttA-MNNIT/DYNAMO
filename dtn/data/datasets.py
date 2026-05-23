"""
Dataset builders for continual-learning benchmarks.

  get_split_mnist    – 5 binary tasks from MNIST digits (0/1 … 8/9)
  get_seq_cifar100   – n_tasks groups of CIFAR-100 classes
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

SEED = 42


# ── Helper dataset ────────────────────────────────────────────────────────────


class RelabelDataset(Dataset):
    """Wrap a dataset and remap class labels via ``label_map``."""

    def __init__(self, base_ds: Dataset, label_map: dict):
        self.base = base_ds
        self.lmap = label_map

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        y_int = y.item() if isinstance(y, torch.Tensor) else int(y)
        return x, self.lmap[y_int]


# ── Split-MNIST ───────────────────────────────────────────────────────────────


def get_split_mnist(
    root: str = "./data",
    batch_size: int = 128,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Return (train_loaders, test_loaders) for Split-MNIST.

    5 binary tasks: (0 vs 1), (2 vs 3), (4 vs 5), (6 vs 7), (8 vs 9).
    Labels within each task are remapped to {0, 1}.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr_full = datasets.MNIST(root, train=True,  download=True, transform=tf)
    te_full = datasets.MNIST(root, train=False, download=True, transform=tf)

    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tr_loaders, te_loaders = [], []

    for c1, c2 in tasks:
        lmap   = {c1: 0, c2: 1}
        tr_idx = [i for i, (_, y) in enumerate(tr_full) if y in (c1, c2)]
        te_idx = [i for i, (_, y) in enumerate(te_full) if y in (c1, c2)]

        tr_loaders.append(DataLoader(
            RelabelDataset(Subset(tr_full, tr_idx), lmap),
            batch_size=batch_size, shuffle=True, num_workers=0,
        ))
        te_loaders.append(DataLoader(
            RelabelDataset(Subset(te_full, te_idx), lmap),
            batch_size=256, shuffle=False, num_workers=0,
        ))

    return tr_loaders, te_loaders


# ── Sequential CIFAR-100 ──────────────────────────────────────────────────────


def get_seq_cifar100(
    root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
) -> Tuple[List[DataLoader], List[DataLoader], int]:
    """
    Return (train_loaders, test_loaders, classes_per_task) for Seq-CIFAR-100.

    100 classes randomly permuted and split into ``n_tasks`` equal groups.
    Images are resized to 224×224 and normalised with ImageNet statistics so
    they are compatible with pretrained ViT-B/16 feature extraction.
    """
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    tf_tr = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_te = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tr_full = datasets.CIFAR100(root, train=True,  download=True, transform=tf_tr)
    te_full = datasets.CIFAR100(root, train=False, download=True, transform=tf_te)

    rng       = np.random.RandomState(SEED)
    cls_order = rng.permutation(100)
    cpt       = 100 // n_tasks

    tr_loaders, te_loaders = [], []
    for t in range(n_tasks):
        tc   = set(cls_order[t * cpt : (t + 1) * cpt].tolist())
        lmap = {c: i for i, c in enumerate(sorted(tc))}

        tr_idx = [i for i, (_, y) in enumerate(tr_full) if y in tc]
        te_idx = [i for i, (_, y) in enumerate(te_full) if y in tc]

        tr_loaders.append(DataLoader(
            RelabelDataset(Subset(tr_full, tr_idx), lmap),
            batch_size=batch_size, shuffle=True, num_workers=0,
        ))
        te_loaders.append(DataLoader(
            RelabelDataset(Subset(te_full, te_idx), lmap),
            batch_size=256, shuffle=False, num_workers=0,
        ))

    return tr_loaders, te_loaders, cpt
