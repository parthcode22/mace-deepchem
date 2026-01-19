# 🧪 MACE–DeepChem

**Native Multi-Atomic Cluster Expansion (MACE) Model for DeepChem**

A PyTorch-based implementation of **MACE (Multi-Atomic Cluster Expansion)** fully integrated into the **DeepChem** ecosystem for molecular property prediction using graph-based workflows and real 3D molecular geometry.

---

## 📌 Overview

This project implements **MACE as a native DeepChem `TorchModel`**, enabling seamless use with DeepChem datasets, metrics, and workflows—similar to existing graph models such as GCN, GAT, and AttentiveFP.

The implementation focuses on:

* Correct handling of **atomic numbers and embeddings**
* Robust **3D coordinate extraction**
* Compatibility with **PyTorch Geometric**
* Stable training on both **CPU and GPU**

This work is being developed as part of a **GSoC 2026 proposal for DeepChem**.

---

## ✨ Key Features

* ✅ **Native DeepChem integration** (`fit`, `predict`, `evaluate`)
* ✅ **Complete MACE architecture** with message-passing interactions
* ✅ **Real 3D molecular coordinates** extracted from SMILES via RDKit
* ✅ **Correct atom-type handling** aligned with DeepChem graph features
* ✅ **PyTorch Geometric batching** support
* ✅ **Custom training loop** for CPU/GPU stability
* ✅ **Scales to thousands of molecules**
* ✅ **Research-ready codebase**

---

## 🚀 What’s New (Latest Update)

### 🔧 Implementation Highlights

* Full MACE architecture including:

  * Radial basis functions
  * Interaction blocks
  * Energy prediction head
* DeepChem-compatible batching via `TorchModel`
* Automatic fallback between **CPU and GPU**
* Robust handling of:

  * Missing atomic numbers
  * Mismatched atom counts
  * Coordinate padding and truncation
* End-to-end pipeline from **SMILES → 3D → Graph → Energy**

---

## 📊 Current Results

### QM9 Dataset (Subset)

| Setting        | Value              |
| -------------- | ------------------ |
| Molecules      | 5,000              |
| Epochs         | 30                 |
| Device         | CPU                |
| Training Time  | ~26 minutes        |
| Validation MAE | **0.811 kcal/mol** |

✔ Uses **real 3D coordinates** generated from SMILES
✔ Stable training with no NaN / Inf issues
✔ Competitive performance for dataset size and CPU-only training

---

## 🧠 Why MACE + DeepChem?

* MACE is **state-of-the-art** for atomistic energy prediction
* DeepChem provides:

  * Dataset handling
  * Metrics
  * Standardized ML workflows
* This project bridges **modern equivariant models** with **chemical ML infrastructure**

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/mace-deepchem
cd mace-deepchem
pip install -r requirements.txt
```

### Requirements

* Python 3.8+
* PyTorch ≥ 2.0
* PyTorch Geometric
* DeepChem ≥ 2.8
* RDKit
* NumPy

---

## ⚡ Quick Start

```python
import deepchem as dc
from mace_deepchem import MACEModel

# Load QM9 dataset
tasks, datasets, _ = dc.molnet.load_qm9(featurizer='GraphConv')
train, valid, test = datasets

# Create model
model = MACEModel(
    hidden_dim=32,
    num_interactions=2
)

# Train
model.fit(train, nb_epoch=10)

# Evaluate
metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
scores = model.evaluate(valid, [metric])

print(f"Validation MAE: {scores['mean_absolute_error']:.3f} kcal/mol")
```

---

## 🧱 Architecture

### Model Components

* **MACE Neural Network (PyTorch)**

  * Radial basis expansion
  * Multi-interaction message passing
  * Energy aggregation head

* **DeepChem Wrapper**

  * Extends `TorchModel`
  * Handles batching, loss, and optimization
  * Compatible with DeepChem metrics and evaluators

* **Data Pipeline**

  * SMILES → RDKit molecule
  * RDKit → atomic numbers + 3D coordinates
  * PyTorch Geometric graph construction

---

## 🗂 Project Structure

```
mace-deepchem/
├── mace_components.py    # Core MACE architecture (PyTorch)
├── mace_deepchem.py      # DeepChem TorchModel wrapper
├── train_mace.py         # Training script
├── examples/             # Example notebooks
├── tests/                # Unit tests
└── README.md
```

---

## 🔄 Current Status

### ✅ Completed

* Full MACE model implementation
* DeepChem TorchModel integration
* CPU/GPU compatible training
* Stable training on QM9
* Validation benchmarking

### 🚧 In Progress

* Scaling to larger datasets (10k–100k molecules)
* Hyperparameter optimization
* Energy normalization strategies
* Unit test expansion
* Documentation + examples

---

## 🌱 Future Work 

* Add force prediction support
* Support periodic boundary conditions
* Integrate equivariant tensor outputs
* Benchmark against SchNet, DimeNet, PaiNN
* Contribute model upstream to DeepChem

---

## 🤝 Contributing

Contributions, feedback, and discussions are welcome!
This project is actively developed with the goal of **upstreaming into DeepChem**.

---

## 📄 License

MIT License

---



