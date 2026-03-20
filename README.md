# Deep Learning with PyTorch — Study Repository

A structured notebook-based repository documenting my implementation work in machine learning and introductory deep learning with PyTorch, based on educational material associated with Dr. Yves Hilpisch.

## Scope

This repository is organized as a progressive study workflow covering:

- foundational machine-learning concepts
- PyTorch basics and neural-network training
- section-based exercise notebooks
- later practice notebooks
- a compact applied capstone on the California Housing dataset

The repository is educational in purpose and is intended to preserve a clean, reproducible record of chapter work, exercises, and applied extensions.

## How to Use This Repository

The notebooks are intended to be read in sequence:

- work through the chapter notebooks first, from foundations into PyTorch basics and practice
- use the exercise notebooks alongside their matching chapters for reinforcement
- treat `part1_foundations/capstone_california_housing` as a staged applied mini-project that brings the earlier material together in a multi-notebook sequence

## Repository Structure

```text
.
├── part1_foundations
│   ├── chapter_1.ipynb
│   ├── chapter_2.ipynb
│   ├── chapter_3.ipynb
│   ├── chapter_4.ipynb
│   ├── capstone_california_housing/
│   └── exercises_challenges/
│
├── part2_pytorch_basics
│   ├── chapter_5.ipynb
│   ├── chapter_6.ipynb
│   ├── chapter_7.ipynb
│   ├── chapter_8.ipynb
│   └── exercises_challenges/
│
├── part3_practice
│   ├── chapter_9.ipynb
│   ├── chapter_10.ipynb
│   ├── chapter_11.ipynb
│   └── exercises_challenges/
│
├── environment.yml
└── README.md
```

## Section Summary

### Part I — Foundations
Introductory notebooks covering numerical and machine-learning basics, including regression, classification, evaluation, and overfitting diagnostics.

### Part II — PyTorch Basics
Notebooks focused on tensors, autograd, optimization, and simple neural-network training workflows.

### Part III — Practice
Follow-on notebooks used to consolidate the material through additional implementation work and exercises.

### Capstone
The `capstone_california_housing` folder contains a staged California Housing mini-project that progresses from data familiarization and preprocessing through baseline models, diagnostics, feature engineering, MLP experiments, and final stretch-goal extensions.

## Notes

- Saved `.pt` and `.pkl` artifacts are kept in the capstone folder to preserve outputs from the applied workflow.
- This repository is a study and implementation record, not a production package.

## Attribution

This repository is an independent educational implementation based on study of material associated with Dr. Yves Hilpisch.

Pedagogical credit for the original course and book material belongs to Dr. Yves Hilpisch and the associated teaching context. Repository organization, exercise solutions, and extensions in this version are my own.

## License

No formal license file is included in this repository. The contents are shared here for educational reference; if you need reuse permissions beyond normal study/reference use, contact the maintainer.

## Maintainer

Francisco Salazar

