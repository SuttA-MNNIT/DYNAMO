# DYNAMO --- DYnamic Neural Architecture with Metabolic Optimization
# Imp: DTN = DYNAMO So, don't confuse with DTN in the code. I will commit changes later. 
**DYNAMO: Algorithmic Thermodynamics for Continual Learning via Neural Morphogenesis**

DYNAMO is a biologically-inspired continual-learning architecture that treats a neural network as an **open thermodynamic system**. Neurons grow, prune, and replicate in response to gradient-derived "metabolic signals", enabling a single model to learn a sequence of tasks without catastrophic forgetting.

---

## Key Concepts

| Mechanism | Biological Analogy | Implementation |
|---|---|---|
| **Synaptic starvation** | Synaptic pruning | Edge pruned when energy < 0 |
| **Algorithmic mitosis** | Neurogenesis | Neuron duplicated when heat > H_crit |
| **Fisher metabolism** | ATP synthesis | EMA of squared weight gradients |
| **Topological breathing** | Synaptic plasticity | Mask-gated dense weight matrix |

### Architecture

```
Input → [ThermoLayer L1] → [ThermoLayer L2] → … → [Task Head t]
                ↑
        Sparse binary mask (Hadamard)
        Per-edge energy budget
        Per-node heat accumulator
```

Each `ThermoLayer` maintains four non-parameter buffers updated after every optimizer step:

- **`mask`** — binary adjacency; 0 = pruned, 1 = active  
- **`fisher`** — EMA of squared weight gradients (metabolic currency)  
- **`energy`** — running per-edge budget: `energy += β·fisher − λ`  
- **`heat`** — EMA of squared bias gradients (mitosis trigger)

---

## Repository Layout

```
dtn/
├── dtn/                       # main package
│   ├── models/
│   │   ├── thermo_layer.py    # ThermoLayer (masked linear + metabolic state)
│   │   ├── dtn.py             # DTN (backbone + heads + thermo/mitosis logic)
│   │   └── baselines.py       # DenseMLPMultiHead, EWCWrapper, PNN
│   ├── data/
│   │   └── datasets.py        # get_split_mnist, get_seq_cifar100
│   ├── training/
│   │   └── trainers.py        # per-model training loops + accuracy()
│   ├── experiments/
│   │   ├── exp1_split_mnist.py
│   │   ├── exp2_seq_cifar100.py
│   │   └── ablations.py       # λ / H_crit / sparsity sweeps
│   ├── viz/
│   │   └── plots.py           # all matplotlib figures
│   └── utils/
│       └── reporting.py       # table printing + JSON serialisation
├── configs/
│   └── default.yaml           # all hyperparameters
├── scripts/
│   └── run_all.py             # main entry point
├── tests/
│   └── test_models.py         # pytest unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/SuttA-MNNIT/DYNAMO.git
pip install -e ".[dev]"
```

### 2. Run experiments

```bash
# Full pipeline (Split-MNIST + CIFAR-100 + ablations)
python scripts/run_all.py

# Fast run: MNIST only, no ablations
python scripts/run_all.py --no-cifar --no-ablations

# Custom config
python scripts/run_all.py --config configs/default.yaml --output-dir /tmp/dtn_out
```

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Experiments

### Experiment 1 — Split-MNIST

Five binary tasks from MNIST: (0/1), (2/3), (4/5), (6/7), (8/9).  
Models compared: **DYNAMO**, **Dense SGD**, **EWC**, **PNN**.

Key metrics reported:
- Average accuracy after all tasks (AT)
- Task-1 retention (R1)
- Forgetting rate
- Jaccard sub-network overlap between task 1 and later tasks

### Experiment 2 — Sequential CIFAR-100

100 classes split into 10 task groups.  
A frozen **ViT-B/16** (pretrained on ImageNet) extracts 768-d features.  
Models compared: **DYNAMO**, **Dense SGD**, **EWC**.

### Ablations

| Study | Swept variable | Fixed defaults |
|---|---|---|
| Metabolic rate | λ ∈ {1e-6 … 1e-2} | H_crit=0.85, sparsity=0.95 |
| Heat threshold | H_crit ∈ {0.3 … 1.5} | λ=1e-4, sparsity=0.95 |
| Initial sparsity | sparsity ∈ {0.5 … 0.98} | λ=1e-4, H_crit=0.85 |

---

## Configuration

All hyperparameters live in `configs/default.yaml`. The most important knobs:

```yaml
dtn_mnist:
  lam: 1.0e-4      # basal metabolic cost — higher = more aggressive pruning
  h_crit: 0.85     # heat threshold — lower = more frequent mitosis
  sparsity: 0.95   # initial edge density — higher = sparser start
  beta: 1.0        # nutrient absorption rate — higher = faster energy gain
  alpha: 0.9       # Fisher/heat EMA — higher = longer memory
  max_w: 256       # neuron growth cap per layer
  cap_mitosis: 6   # max new neurons per mitosis step
```

---

## Implementation Notes (Deviations from Paper)

1. **Masked-dense layers** instead of true COO sparse tensors — semantically identical math, much faster on modern hardware with Tensor Cores.
2. **Multi-head output** — task-incremental scenario; the shared backbone is where forgetting occurs.
3. **Uniform Bernoulli nascent nodes** — simplified random-walk init at fixed 10 % density instead of distance-decay (Eq. 7), since graph distance is ambiguous in the masked-dense setting.
4. **Bias-gradient heat via EMA** — Eq. (4) approximated as EMA over squared bias gradients (α=0.9), equivalent to the continuous integral with γ as time-constant.
5. **Energy initialised to 1.0** — paper is silent on initial conditions.

---

## Citation

```bibtex
@article{dynamo2026,
  title   = {DYNAMO: Algorithmic Thermodynamics for Continual Learning via Neural Morphogenesis},
  journal = {IEEE Transactions on Neural Networks and Learning Systems}
  author  = {Ravi Prakash},
  year    = {2026},
}
```

---

## License

MIT
