# Deep Learning Basics with PyTorch 

### Based on the *upcoming* book by **Yves Hilpisch, Ph.D. — “Deep Learning with PyTorch” (O’Reilly, forthcoming)**

---

## Overview

This repository accompanies a self-directed study of the *pre-release draft* of  
**“Deep Learning with PyTorch”** by **Dr. Yves Hilpisch** (O’Reilly, forthcoming).  

It reproduces and extends selected chapters from the manuscript, linking classical machine-learning foundations to modern deep-learning concepts within a **quantitative-finance** context.  

Each notebook builds progressively—from core NumPy and scikit-learn workflows to PyTorch-based architectures—providing a transparent and reproducible learning path for applied machine learning in finance.

---

## Repository Structure

```
├── part1_foundations
│   ├── adr_prices_and_vol.csv		# Supporting dataset (finance-oriented)
│   ├── capstone_california_housing	# Capstone: feature engineering & PyTorch MLP
│   ├── chapter_1.ipynb 				# NumPy foundations & linear algebra
│   ├── chapter_2.ipynb				# Linear & Ridge Regression
│   ├── chapter_3.ipynb				# Classification models (LogReg, SVM, Trees)
│   ├── chapter_4.ipynb				# Limits of Classical ML — Overfitting, Complexity 
│   └── exercises_challenges			# Chapter review exercises and challenges 
├── part2_pytorch_basics
│   ├── chapter_5.ipynb				# Introduction to Deep Learning with PyTorch
│   └── exercises_challenges			# Exercises for neural network foundations			
└── README.md
```
---

> **Note:**  
> In addition to standard sample datasets referenced in the book (e.g., `iris`, `make_moons`, `california_housing`),  
> this repository includes a small supplementary file — `adr_prices_and_vol.csv` — used for exploratory testing  
> of financial data workflows and to illustrate how PyTorch and scikit-learn models can be applied to  
> quantitative finance use cases.

---
## Current Progress

The project currently covers **Part I (Foundations)** and introduces **Part II (PyTorch Basics)**.  
Each notebook builds upon the previous one, forming a coherent learning path from classical ML to deep learning.

---

### 📘 Chapter Checklist

* [x] Chapter 1 – NumPy Foundations
* [x] Chapter 2 – Linear & Ridge Regression 
* [x] Chapter 3 – Classification & Ensemble Comparison
* [x] Chapter 4 – Limits of Classical ML (Overfitting & Learning Curves)
* [x] Chapter 5 – PyTorch Neural Networks

---

### 🧩 Notebook Summary

| Notebook          | Chapter                    | Focus                                                    |
| ----------------- | -------------------------- | -------------------------------------------------------- |
| `chapter_1.ipynb` | Introduction               | NumPy, linear algebra, and environment setup             |
| `chapter_2.ipynb` | Classical ML               | Linear, Ridge, and Polynomial Regression                 |
| `chapter_3.ipynb` | ML Models                  | Logistic Regression, SVM, Decision Trees, Random Forests |
| `chapter_4.ipynb` | Limits of Classical ML     | Complexity, Overfitting, Learning Curves                 |
| `chapter_5.ipynb` | Deep Learning with PyTorch | First Steps with PyTorch                                 |

---

## Environment Setup

### Create environment

```bash
conda create -n pytorch_dl python=3.12 -y
conda activate pytorch_dl
```

### Install core dependencies

```bash
conda install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy numba jupyterlab notebook ipykernel -y
```

* 💡 (PyTorch will be added starting from Chapter 5.)*

### Launch Jupyter

```bash
jupyter lab
```

---

## Citation and Attribution

> *This repository reproduces and extends materials from the pre-release manuscript of*
> **Hilpisch, Y. (2025, forthcoming). Deep Learning with PyTorch. O’Reilly Media.**
> *Adapted for educational and research purposes within a quantitative-finance context*

* Original content © Dr. Yves Hilpisch / The Python Quants GmbH
* Adaptations © 2025 Francisco Salazar — Academic, non-commercial use only

---

## License

This repository is provided for **educational and research purposes only**.  
It is **not affiliated with** or endorsed by **O’Reilly Media** or **The Python Quants GmbH**.

All adaptations are shared under a *non-commercial educational use* framework to support open learning in quantitative finance and deep learning.

---
**Maintained by:** Francisco Salazar  
*Last updated: October 2025*


