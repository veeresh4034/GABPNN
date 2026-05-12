# GABPNN: Graph-Augmented Belief-Propagation Neural Network for Quantum Error Correction

## Overview

This repository contains a  framework for simulating and decoding rotated surface codes using both classical and neural-network-based decoding algorithms.

The project implements:

* A validated rotated surface code simulator
* Exact Minimum-Weight Perfect Matching (MWPM) decoding
* Classical baseline decoders
* A new hybrid neural decoder:
  **GABPNN (Graph-Augmented Belief-Propagation Neural Network)**

The implementation is written primarily in pure Python with NumPy/SciPy and is designed to be:

* Lightweight
* Reproducible
* Easy to extend
* Independent of heavy ML frameworks such as PyTorch

---

# Repository Structure

```text
.
├── surface_code.py          # Rotated surface code construction and validation
├── exact_mwpm_nx.py         # Exact MWPM decoder using NetworkX Blossom algorithm
├── classical_decoders.py    # Classical baseline decoders
├── hybrid_decoder.py        # GABPNN hybrid neural decoder
└── results_final.json       # Example simulation / evaluation results
```

---

# Features

## 1. Rotated Surface Code Simulator

Implemented in `surface_code.py`.

Features:

* Distance-d rotated surface code construction
* Bulk weight-4 plaquettes
* Boundary weight-2 stabilizers
* Full stabilizer commutativity validation
* Logical operator generation
* CSS code consistency checks

Validated for:

```text
d ∈ {3, 5, 7, 9}
```

Logical operators:

* `L_x` → top-row logical X string
* `L_z` → left-column logical Z string

---

## 2. MWPM Decoder

Implemented in `exact_mwpm_nx.py`.

Features:

* Minimum-Weight Perfect Matching
* Uses NetworkX Blossom matching algorithm
* Boundary-aware matching
* Topological parity tracking
* Similar decoding logic to PyMatching

Capabilities:

* Z-logical parity detection
* X-logical parity detection
* Syndrome graph construction
* BFS-based shortest-path distances

---

## 3. Classical Baseline Decoders

Implemented in `classical_decoders.py`.

Includes:

### Greedy MWPM Approximation

* Nearest-defect matching
* O(n²) complexity
* Pure NumPy implementation
* Useful baseline for comparison

### Union-Find Decoder

* Inspired by Delfosse & Nickerson
* Near-linear complexity
* Cluster-growth decoding
* Efficient for larger code distances

---

## 4. Hybrid Neural Decoder (GABPNN)

Implemented in `hybrid_decoder.py`.

### Core Idea

The decoder combines:

* Belief-propagation-inspired graph preprocessing
* Syndrome graph feature extraction
* Residual MLP classification
* Channel-conditioned learning

without requiring PyTorch or TensorFlow.

### Main Contributions

#### Graph-Augmented Features

The decoder extracts:

* Syndrome activity
* Node degree
* Adjacency interactions
* Local graph structure

using message-passing operations.

#### Channel Conditioning

A channel-bias indicator is concatenated to node features, enabling:

* Symmetric noise decoding
* Asymmetric noise decoding
* Single-model adaptability

#### Residual Neural Architecture

The final decoder uses:

* Residual fully connected layers
* ReLU activations
* Softmax output layer
* Adam optimization

---

# Installation

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy networkx
```

Recommended:

```bash
Python >= 3.10
```

---

# Quick Start

## Example: Create a Surface Code

```python
from surface_code import SurfaceCode

code = SurfaceCode(d=5)

print(code.n_data)
print(code.n_stab)
```

---

## Example: Exact MWPM Decoding

```python
from surface_code import SurfaceCode
from exact_mwpm_nx import ExactMWPM

code = SurfaceCode(5)
decoder = ExactMWPM(code)
```

---

## Example: Classical Decoders

```python
from classical_decoders import GreedyMWPM

decoder = GreedyMWPM(code)
```

---

## Example: Hybrid Neural Decoder

```python
from hybrid_decoder import BPFeatureExtractor

extractor = BPFeatureExtractor(code)
```

---



# Example Workflow

```text
Physical Error
      ↓
Syndrome Extraction
      ↓
Graph Feature Construction
      ↓
BP-Inspired Message Passing
      ↓
Residual Neural Network
      ↓
Coset Prediction
      ↓
Logical Error Correction
```

---

# Results

Example output data is included in:

```text
results_final.json
```

This may contain:

* Logical error rates
* Decoder comparisons
* Threshold experiments
* Performance metrics
* Training/evaluation summaries

---

