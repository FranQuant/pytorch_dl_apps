# CLAUDE.md — pytorch_dl_apps

This is the primary briefing document for Claude Code working in this repository.
Read this file completely before touching anything else.

---

## Project Identity

| Field | Value |
|-------|-------|
| Repo | `pytorch_dl_apps` (private, local) |
| Owner | Francisco Salazar (`FranQuant`) |
| Purpose | Personal study & implementation record — chapter notebooks, exercise solutions, and applied extensions |
| Status | Active — Part 3 exercises still in progress |

This is **not** a production package. It is a structured educational repository.

---

## Source Material

### Book
**Title:** *Deep Learning Basics with PyTorch — An approachable, code-first introduction to deep learning*
**Author:** Dr. Yves J. Hilpisch (The Python Quants GmbH)
**Date:** March 2026
**Companion repo:** https://github.com/yhilpisch/dlcode

The companion repo (`yhilpisch/dlcode`) contains:
- `notebooks/` — reference chapter notebooks (`chNN_*.ipynb`)
- `code/chNN/` — Python scripts per chapter
- `code/figures/` — figure-generation scripts
- `tools/` — validation scripts (`validate_code.py`, `validate_notebooks.py`, `validate_figures.py`)
- `requirements.txt` — minimal dependency set

When Claude Code needs to cross-reference the upstream implementation, refer to `yhilpisch/dlcode`.
Do NOT copy files from that repo into this one without asking the user first.

### Course
**Program:** The AI Engineer (theaiengineer.dev)
**Format:** 4-week full-time module (or 8-week medium track)
**This repo covers:** Week 1 (Foundations + PyTorch Basics) and Week 2 (Deep Learning with PyTorch)

| Week | Core Track Focus | Engineering Focus | Key Deliverable |
|------|-----------------|-------------------|-----------------|
| Ramp-Up | Math refresh, toolchain | Environment setup, Git hygiene | Baseline repo commit |
| Week 1 | Python & Math for ML/DL | Foundation Primers, CLI tools | Notebook assignments + saved figure |
| Week 2 | Deep Learning with PyTorch | Software Engineering, CI setup | Train/Test loop with metrics dashboard |
| Week 3 | LLM from Scratch | ML Engineering, pipelines | Tiny transformer + evaluation pipeline |
| Week 4 | AI Agents & Automation | AI Engineering | Production-ready components |

Francisco is currently in **Week 1–2** territory, using this repo to work through exercises
and build toward the 4 capstone projects that complete the program.

---

## Book Structure — Full Chapter Map

| Part | Ch | Title | Repo Location |
|------|----|-------|---------------|
| **I — Foundations of ML** | 1 | Introduction to Machine Learning | `part1_foundations/chapter_1.ipynb` |
| | 2 | Data, Features, and Splits | `part1_foundations/chapter_2.ipynb` |
| | 3 | Basic Models | `part1_foundations/chapter_3.ipynb` |
| | 4 | Limits of Classical ML | `part1_foundations/chapter_4.ipynb` |
| **II — Neural Networks & PyTorch Basics** | 5 | First Steps with PyTorch | `part2_pytorch_basics/chapter_5.ipynb` |
| | 6 | Building Blocks of Neural Networks | `part2_pytorch_basics/chapter_6.ipynb` |
| | 7 | Training Neural Networks | `part2_pytorch_basics/chapter_7.ipynb` |
| | 8 | Organizing with nn.Module | `part2_pytorch_basics/chapter_8.ipynb` |
| **III — Supervised Deep Learning in Practice** | 9 | Data Pipelines | `part3_practice/ch09_data_pipelines.ipynb` |
| | 10 | Improving Training | `part3_practice/ch10_improving_training.ipynb` |
| | 11 | Deeper Architectures | `part3_practice/ch11_deeper_architectures.ipynb` |
| | 12 | Training at Scale | `part3_practice/ch12_training_at_scale.ipynb` |
| **IV — Toward Large Language Models** | 13 | Sequences and Language | `part3_practice/ch13_sequences_language.ipynb` |
| | 14 | RNNs, Attention, and Transformers | `part3_practice/ch14_rnn_attention_transformers.ipynb` |
| | 15 | Training Large Models | ⚠️ Not yet in repo |
| **V — Broader Context & Next Steps** | 16 | Ethics, Risks, and Applications | ⚠️ Not yet in repo |
| | 17 | Next Steps | ⚠️ Not yet in repo |

**Appendices in book (reference only, not implemented as notebooks):**
A — Python & NumPy | B — Probability & Statistics | C — Linear Algebra |
D — Calculus | E — Installation & Environment | F — Full Scripts |
G — Notebooks Index | H — Glossary

---

## Repository Structure

```text
pytorch_dl_apps/
├── part1_foundations/                    # Book Part I — Chs 1–4 ✅ COMPLETE
│   ├── chapter_1.ipynb                   # Intro to ML, linear regression
│   ├── chapter_2.ipynb                   # Data, features, splits, Iris
│   ├── chapter_3.ipynb                   # Basic models (LinReg, LogReg, Trees, SVMs)
│   ├── chapter_4.ipynb                   # Limits of classical ML, overfitting
│   ├── exercises_challenges/             # ✅ All exercise notebooks complete
│   │   ├── exercises_ch1_basics.ipynb
│   │   ├── exercises_ch2_features.ipynb
│   │   ├── exercises_ch3_eval.ipynb
│   │   └── exercises_ch4_limits.ipynb
│   └── capstone_california_housing/      # ✅ Applied mini-project COMPLETE
│       ├── 01_data_familiarization.ipynb
│       ├── 02_splits_preprocessing.ipynb
│       ├── 03_baseline_models.ipynb
│       ├── 04_diagnostics_iteration.ipynb
│       ├── 05_mlp_feature_engineering.ipynb
│       ├── 06_stretch_goals.ipynb
│       ├── 06_stretch_goals.py
│       ├── best_mlp.pt                   # Saved MLP — intentional artifact, do not delete
│       ├── hgb_model.pkl                 # Saved HistGradientBoosting — do not delete
│       ├── preprocessor.pkl              # Saved sklearn preprocessor — do not delete
│       ├── ridge_best.pkl                # Saved Ridge model — do not delete
│       ├── capstone.png
│       ├── env_snapshot.txt
│       └── results/
│
├── part2_pytorch_basics/                 # Book Part II — Chs 5–8 ✅ COMPLETE
│   ├── chapter_5.ipynb                   # First steps with PyTorch, tensors
│   ├── chapter_6.ipynb                   # Neurons, activations, building blocks
│   ├── chapter_7.ipynb                   # Training neural networks
│   ├── chapter_8.ipynb                   # Organizing code with nn.Module
│   └── exercises_challenges/             # ✅ All exercise notebooks complete
│       ├── exercises_ch5_pytorch_basics.ipynb
│       ├── exercises_ch6_neurons_activations.ipynb
│       ├── exercises_ch7_training_nn.ipynb
│       └── exercises_ch8_organize_code.ipynb
│
├── part3_practice/                       # Book Parts III & IV — Chs 9–14
│   ├── ch09_data_pipelines.ipynb         # Data pipelines ✅
│   ├── ch10_improving_training.ipynb     # Improving training ✅
│   ├── ch11_deeper_architectures.ipynb   # Deeper architectures ✅
│   ├── ch12_training_at_scale.ipynb      # Training at scale ✅
│   ├── ch13_sequences_language.ipynb     # Sequences and language ✅
│   ├── ch14_rnn_attention_transformers.ipynb  # RNNs, attention, transformers ✅
│   └── exercises_challenges/             # ⚠️ WORK IN PROGRESS
│       ├── exercises_ch_9.ipynb
│       ├── exercises_ch_10.ipynb
│       ├── exercises_ch_11.ipynb
│       ├── exercises_ch_12.ipynb
│       ├── exercises_ch_13.ipynb
│       ├── exercises_ch_14.ipynb
│       ├── data/                         # Local datasets for Part 3
│       ├── cache.pkl                     # Runtime cache — treat as read-only
│       └── run_manifest.json             # Runtime manifest — treat as read-only
│
├── scripts/
│   └── notebook_workflow.sh             # Shell utility for notebook operations
│
├── environment.yml                       # Conda environment spec
└── README.md
```

---

## Environment

| Item | Value |
|------|-------|
| Machine | Apple M4 mini (Apple Silicon — arm64) |
| Package manager | Conda |
| Active environment | `pytorch_dl_apps` |
| Activate with | `conda activate pytorch_dl_apps` |

**Rules:**
- Always use the `pytorch_dl_apps` conda environment
- Do NOT install packages globally or into `base`
- If a new dependency is needed, propose adding it to `environment.yml` and ask first
- The M4 chip uses Apple Silicon — prefer MPS backend over CUDA for GPU acceleration in PyTorch

---

## Work-in-Progress Boundary

| Section | Chapters | Notebooks | Exercises |
|---------|----------|-----------|-----------|
| Part 1 — Foundations | Ch 1–4 | ✅ Complete | ✅ Complete |
| Capstone California Housing | — | ✅ Complete | — |
| Part 2 — PyTorch Basics | Ch 5–8 | ✅ Complete | ✅ Complete |
| Part 3 — Practice | Ch 9–11 | ✅ Complete | ⚠️ In progress |
| Part 3 — Advanced | Ch 12–14 | ✅ Complete | ⚠️ In progress |
| Book Ch 15–17 | — | ❌ Not started | — |

**Do not assume Part 3 exercises are finished.**
Always inspect cell output before drawing conclusions about completion.
Confirm with the user before extending or modifying any Part 3 exercise notebook.

---

## Broader Project Context

This repository is **one part of a larger learning journey** tied to The AI Engineer program.
The program culminates in **4 capstone projects** (hosted in a separate repo, not this one).

This repo covers the foundational work (Weeks 1–2 of the course) that feeds into those capstones.
Claude Code may be asked to review both repos at different points — when that happens, ask the user for the capstone repo path.

The transformer exercises in Ch 13–14 are directly upstream of the Week 3 capstone (tiny decoder-only LLM).
Keep that trajectory in mind when helping with Part 3 exercises — the goal is not just completion but readiness for what comes next.

---

## Coding Conventions

- All notebooks follow the code-first style from the Hilpisch book
- Prefer clean, readable cells — this is pedagogical code, not production code
- Add markdown cells to explain key steps when adding or reviewing content
- Each notebook should have a clear header cell: chapter title, topic, date
- Use fixed random seeds where relevant (matches upstream reproducibility convention from `yhilpisch/dlcode`)
- Saved artifacts (`.pt`, `.pkl`) in the capstone folder are **intentional — do not delete them**
- `run_manifest.json` and `cache.pkl` in Part 3 are runtime artifacts — **treat as read-only**

---

## How to Run Things

```bash
# Always activate the environment first
conda activate pytorch_dl_apps

# Launch a specific notebook
jupyter notebook part2_pytorch_basics/chapter_5.ipynb

# Run a shell utility
bash scripts/notebook_workflow.sh

# Cross-reference upstream book code (if cloned separately)
# cd ../dlcode && jupyter notebook notebooks/ch05_pytorch_basics.ipynb
```

---

## What Francisco Is Trying to Accomplish

Two simultaneous goals:

1. **Review** — working through Deep Learning with PyTorch concepts chapter by chapter, reinforcing understanding through exercises
2. **Build** — creating a clean, well-documented educational repo that serves as a personal reference and demonstrates competence for The AI Engineer program

Claude Code should prioritize:
- **Clarity and explanation** over brevity or cleverness
- **Preserving the existing chapter/exercise structure** — do not reorganize without asking
- **Helping complete remaining Part 3 exercises** when asked, with attention to the transformer capstone trajectory
- **Suggesting improvements** to existing notebooks: markdown quality, code clarity, reproducibility, visual outputs
- **Keeping the upstream book and course context in mind** when reviewing or extending any notebook

**Always ask before making structural changes to the repo layout.**
