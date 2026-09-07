# EGPC-GNN

Official implementation of:

**EGPC-GNN: Evidence-Guided Propagation Completion for Class-Imbalanced
Node Classification**

EGPC-GNN is a graph neural network framework designed for
class-imbalanced node classification.

## Overview

EGPC-GNN focuses on propagation-induced evidence degradation in
imbalanced graphs. Instead of only considering class frequency, it
models node-specific evidence reliability and adaptively completes
unreliable propagation representations.

The framework contains three stages:

1.  **Evidence Modeling**
    -   Extract self-evidence from node attributes.
    -   Extract propagation evidence through graph neural networks.
    -   Compare both evidence sources to characterize propagation
        degradation.
2.  **Propagation Degradation Estimation**
    -   Estimate node-specific evidence degradation using prediction
        discrepancy and evidence margin reduction.
    -   Generate adaptive correction weights.
3.  **Evidence-Guided Propagation Completion**
    -   Construct completion anchors using self representations and
        class prototypes.
    -   Combine propagation features with completed evidence.

------------------------------------------------------------------------

# Requirements

Recommended environment:

-   Python \>= 3.8
-   PyTorch \>= 2.0
-   PyTorch Geometric \>= 2.4
-   CUDA \>= 11.7

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Repository Structure

``` text
EGPC-GNN/
│
├── main.py
│
├── predictor/
│   ├── egpcgnn_Predictor.py
│   └── model/
│       └── EGPC_GNN.py
│
├── utils/
│   ├── dataloader.py
│   ├── datasplit.py
│   └── tools.py
│
├── conf/
│
├── data/
│
└── README.md
```

------------------------------------------------------------------------

# Datasets

Supported datasets include:

-   Cora
-   CiteSeer
-   PubMed
-   Amazon-Computers
-   Amazon-Photo
-   Coauthor-CS
-   Chameleon
-   Squirrel
-   Actor

Datasets are automatically downloaded through `utils/dataloader.py`.

The dataloader uses PyTorch Geometric and supports:

-   Planetoid datasets;
-   Amazon datasets;
-   Coauthor datasets;
-   WikipediaNetwork datasets;
-   WebKB datasets;
-   Actor dataset.

Example:

``` python
from utils.dataloader import Dataset

dataset = Dataset(
    data='cora',
    feat_norm=True,
    split_type='default',
    path='data',
    device='cuda:0'
)
```

------------------------------------------------------------------------

# Experimental Settings

Default settings:

  Parameter               Value
  ----------------------- -----------------
  Train/Validation/Test   10% / 10% / 80%
  Imbalance Ratio         100
  Hidden Dimension        64
  GNN Layers              2
  Dropout                 0.5
  Optimizer               Adam
  Learning Rate           1e-2
  Weight Decay            5e-4
  Epochs                  200
  Warm-up Epochs          5

Random seeds:

``` text
100, 110, 120
```

------------------------------------------------------------------------

# Usage

Run EGPC-GNN:

``` bash
python main.py --method egpcgnn --dataset cora
```

Example:

``` bash
python main.py --method egpcgnn --dataset squirrel
```

Evaluation metrics:

-   Accuracy
-   Balanced Accuracy
-   Macro-F1

------------------------------------------------------------------------

# Reproduce Results

To reproduce the experiments:

1.  Install dependencies.
2.  Run experiments with the provided configurations.
3.  Repeat experiments using different random seeds.
4.  Compute mean and standard deviation.

------------------------------------------------------------------------

# Citation

If you find EGPC-GNN useful, please cite:

``` bibtex
@article{egpcgnn2026,
  title={Evidence-Guided Propagation Completion for Class-Imbalanced Node Classification},
  year={2026}
}
```

Citation information will be updated upon publication.

------------------------------------------------------------------------

# Acknowledgement

This implementation is built upon PyTorch Geometric and related
open-source graph learning projects.
