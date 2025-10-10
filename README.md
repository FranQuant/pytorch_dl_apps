# Deep Learning Basics with PyTorch — Quant Edition

### Based on the *upcoming* book by **Yves Hilpisch, Ph.D. — “Deep Learning with PyTorch” (O’Reilly, forthcoming)**

---

## 📘 Overview

This repository reproduces and extends exercises from the **pre-release draft** of *“Deep Learning with PyTorch”* by **Dr. Yves Hilpisch** (O’Reilly, forthcoming), re-implemented with **real ADR market data** instead of toy datasets.

Each notebook mirrors a **chapter** from the book and demonstrates how classical ML foundations evolve into deep learning within a **quantitative finance** context.

---

## 🎯 Objective

The purpose of this project is to:

* Translate *Hilpisch’s* pedagogical examples into **real-world financial data experiments**.
* Bridge the gap between **classical machine-learning intuition** and **deep-learning implementation** in quant research.
* Provide a transparent, reproducible workflow for ADR-based financial modeling.

---

## 📂 Repository Structure

| Notebook          | Chapter                    | Focus                                                    |
| ----------------- | -------------------------- | -------------------------------------------------------- |
| `chapter_1.ipynb` | Introduction               | NumPy, linear algebra, and environment setup             |
| `chapter_2.ipynb` | Classical ML               | Linear, Ridge, and Polynomial Regression                 |
| `chapter_3.ipynb` | ML Models                  | Logistic Regression, SVM, Decision Trees, Random Forests |
| `chapter_4.ipynb` | Limits of Classical ML     | Complexity, Overfitting, Learning Curves                 |
| *(Upcoming)*      | Deep Learning with PyTorch | Feedforward & CNN architectures for financial series     |

---

## 💼 Dataset

All examples use **ADR (American Depositary Receipt)** data — for instance, ticker `CIB` — with engineered financial features:

* Daily and multi-day returns
* Rolling volatility
* Volume and volatility changes
* Binary targets (`1` = up day, `0` = down day)

These datasets replace synthetic ones such as `make_moons` or `iris`to create **realistic market-modeling challenges**.

---

## 🧰 Environment Setup

### Create environment

```bash
conda create -n pytorch_dl python=3.12 -y
conda activate pytorch_dl
```

### Install dependencies

```bash
conda install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy numba jupyterlab notebook ipykernel -y
```

*(PyTorch will be added starting from Chapter 5.)*

### Launch Jupyter

```bash
conda activate pytorch_dl
jupyter lab
```

---

## ✍️ Citation and Attribution

> *This repository reproduces and extends materials from the pre-release manuscript of*
> **Hilpisch, Y. (2025, forthcoming). Deep Learning with PyTorch. O’Reilly Media.**
> *Adapted for educational and research purposes in quantitative finance.*

* Original materials © Dr. Yves Hilpisch / The Python Quants GmbH
* Adaptations © 2025 Francisco Salazar — Academic, non-commercial use only

---

## 📊 Current Progress

* [x] Chapter 1 – NumPy Foundations
* [x] Chapter 2 – Linear & Ridge Regression (ADR returns)
* [x] Chapter 3 – Classification & Ensemble Comparison
* [x] Chapter 4 – Limits of Classical ML (Overfitting & Learning Curves)
* [ ] Chapter 5 – PyTorch Neural Networks (Next phase)

---

## 🧩 Next Steps

* Integrate **PyTorch models** for regression and classification
* Compare **classical vs deep learning** approaches on financial data
* Add **regularization**, **dropout**, and **learning-rate schedules**
* Extend visual diagnostics for generalization and risk metrics

---

## 📄 License

This repository is for **educational and research purposes only**.
Not affiliated with O’Reilly Media or The Python Quants GmbH.
All code © Francisco Salazar unless otherwise noted.

---

### ✅ Recommended commit command

```bash
git add README.md
git commit -m "docs: refine README with pre-release attribution and quant-context structure"
git push origin main
```

---

Would you like me to save this as a ready-to-commit file (`README_refined.md`) so you can preview the diff before overwriting your current one?
