# Deep Learning Basics with PyTorch  
**Based on _Deep Learning with PyTorch_ (O’Reilly, forthcoming) by Dr. Yves Hilpisch**

---

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Conda-Ready-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-Educational--Use-lightgrey.svg)](#license)
![Last Updated](https://img.shields.io/badge/Last%20Updated-October%202025-blueviolet)

---

## Overview

This repository accompanies a **self-directed study and applied implementation** of the pre-release manuscript of  
**_Deep Learning with PyTorch_** by **Dr. Yves Hilpisch** (O’Reilly, forthcoming).  

It reconstructs and extends the book’s exercises, linking **classical machine-learning foundations** to **modern deep-learning practice** in a fully reproducible environment.  
Each notebook forms part of a progressive learning path—from **NumPy fundamentals** and **regression models** to **PyTorch-based neural networks**, **optimisation routines**, and **training diagnostics**.

---

## Learning Outcomes

By following the notebooks, you will:

- Understand the mathematical foundations of regression and classification.  
- Implement, train, and evaluate neural networks using **PyTorch**.  
- Compare classical ML vs. deep-learning behaviour under varying hyperparameters.  
- Apply core training techniques: **SGD**, **Adam**, **early stopping**, **gradient clipping**, **LR scheduling**, and **mini-batch optimisation**.  
- Develop clear intuition for **bias–variance trade-offs** and **convergence dynamics** through hands-on experiments.  

---

## Repository Structure

```
├── part1_foundations
│   ├── chapter_1.ipynb              # NumPy foundations & linear algebra
│   ├── chapter_2.ipynb              # Linear & Ridge Regression
│   ├── chapter_3.ipynb              # Classification (LogReg, SVM, Trees)
│   ├── chapter_4.ipynb              # Overfitting & Learning Curves
│   ├── capstone_california_housing  # Feature engineering & MLP
│   └── exercises_challenges         # Review exercises & solutions
│
├── part2_pytorch_basics
│   ├── chapter_5.ipynb              # Introduction to PyTorch
│   ├── chapter_6.ipynb              # Tensors, Autograd, and Training Loops
│   ├── chapter_7_exercises.ipynb    # Training Tiny Networks + Challenges
│   └── exercises_challenges         # Collected solutions for Part II
│
├── data/
│   └── adr_prices_and_vol.csv       # Finance-oriented sample dataset
│
└── README.md
```

> **Note:**  
> In addition to standard sample datasets referenced in the book (e.g., `iris`, `make_moons`, `california_housing`), this repository includes a small supplementary file — `adr_prices_and_vol.csv` — used for exploratory testing of financial data workflows and to illustrate how PyTorch and scikit-learn models can be applied to quantitative finance use cases.

---

## Environment Setup

### Create and activate environment

```bash
conda create -n pytorch_dl python=3.12 -y
conda activate pytorch_dl
```

### Install core dependencies

```bash
conda install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy numba               jupyterlab notebook ipykernel -y
```

> 💡 **PyTorch** and related libraries are introduced in **Part II (from Chapter 5)**.  
> Install via the official instructions for your OS/GPU configuration:  
> https://pytorch.org/get-started/

### Launch JupyterLab

```bash
jupyter lab
```

---

## Chapter Progress

| Part | Chapter | Focus | Status |
|:--|:--|:--|:--:|
| I | 1 | NumPy & Linear Algebra Foundations | ✅ |
| I | 2 | Linear & Ridge Regression | ✅ |
| I | 3 | Classification & Ensemble Models | ✅ |
| I | 4 | Overfitting & Complexity Control | ✅ |
| II | 5 | PyTorch Neural-Network Fundamentals | ✅ |
| II | 6 | Autograd, Optimisers, Training Loops | ✅ |
| II | 7 | **Exercises & Challenges: Training Tiny Networks** | ✅ (Full) |

---

## Highlights from Chapter 7 – Exercises & Challenges

Implemented training refinements:

- **Activation comparison:** ReLU vs tanh  
- **Model capacity study:** hidden sizes {4, 8, 32}  
- **Early stopping:** patience = 5  
- **Gradient clipping:** `clip_grad_norm_` stabilisation  
- **Cosine LR scheduling:** dynamic learning-rate decay  
- **Mini-batch training:** stochastic optimisation with `DataLoader`

Each experiment includes quantitative diagnostics and decision-boundary visualisations.

---

## Development Notes

- All notebooks adhere to **reproducible, educational research standards**.  
- Each section documents hyperparameters, metrics, and plots inline.  
- The environment is **fully Conda-based** for cross-platform reproducibility.

---

## Citation & Attribution

> **Hilpisch, Y. (2025, forthcoming).** *Deep Learning with PyTorch.* O’Reilly Media.  
> Original teaching materials © Dr. Yves Hilpisch / The Python Quants GmbH.  
> Adaptations © 2025 Francisco Salazar — academic, non-commercial use only.

---

## License

This repository is distributed under an **Educational-Use License** for research and learning purposes.  
It is **not affiliated with** or endorsed by O’Reilly Media or The Python Quants GmbH.

---

**Maintained by:** Francisco Salazar  
📅 _Last Updated: October 2025_
