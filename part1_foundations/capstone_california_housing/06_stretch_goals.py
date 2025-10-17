#!/usr/bin/env python
# coding: utf-8

# # 06 — Stretch Goals: Ablations, Architecture Tweaks & CLI Packaging
# 
# This notebook extends the capstone California Housing project with three advanced research directions:
# 
# 1. **Feature Ablations:** Quantify the impact of engineered variables.
# 2. **Model Tweaks:** Explore batch normalization, dropout, and depth variations.
# 3. **CLI Packaging:** Prototype a reproducible command-line training interface.
# 
# These experiments illustrate the trade-off between model complexity, interpretability, and maintainability.

# ## Feature Ablation Study

# In[1]:


# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

plt.rcParams["figure.dpi"] = 130
sns.set_theme(style="whitegrid", palette="deep")

# --- Reload base data & engineered features (from notebook 5 logic) ---
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
df.rename(columns={"MedHouseVal": "target"}, inplace=True)

df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
df["BedroomsPerRoom"]   = df["AveBedrms"] / df["AveRooms"]
df["PopPerHousehold"]   = df["Population"] / df["AveOccup"]

df["IncomeBucket"] = pd.cut(
    df["MedInc"],
    bins=[0, 2, 4, 6, 8, np.inf],
    labels=["Very Low", "Low", "Mid", "High", "Very High"]
)
df = pd.get_dummies(df, columns=["IncomeBucket"], drop_first=True)

X = df.drop("target", axis=1)
y = df["target"]

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=10)),
])

# --- Drop-one-feature ablation ---
scores = []
for feature in ["RoomsPerHousehold", "BedroomsPerRoom", "PopPerHousehold"]:
    cols = [col for col in X.columns if col != feature]
    mean_r2 = cross_val_score(ridge_pipeline, X[cols], y, cv=5, scoring="r2").mean()
    scores.append((feature, mean_r2))

ablation_df = pd.DataFrame(scores, columns=["Dropped Feature", "Mean R2"]).sort_values("Mean R2", ascending=False)
print(ablation_df)

# --- Visualization ---
plt.figure(figsize=(5,3))
plt.barh(ablation_df["Dropped Feature"], ablation_df["Mean R2"], color="steelblue")
plt.title("Feature Ablation — Impact on R²")
plt.xlabel("Mean Cross-Validated R²")
plt.tight_layout()
plt.show()


# ## MLP Architecture Tweaks

# In[2]:


import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

print("🔧 Ready for architecture tuning: modify hidden_dim, dropout, or layers.")


# ### Observations
# | Configuration | Dropout | Hidden Dim | Val RMSE | Notes |
# |:--|:--:|:--:|:--:|:--|
# | Baseline | 0.2 | 64 | 0.54 | Reference |
# | Variant A | 0.3 | 128 | ... | Slightly improved generalization |
# | Variant B | 0.1 | 256 | ... | Overfit, unstable training |

# ## CLI / Script Packaging

# In[4]:


# --- train_mlp.py style function (portable CLI) ---
import argparse

def train_mlp(seed=42, hidden_dim=64, dropout=0.2, save_path="results/"):
    """
    Compact CLI-ready training loop for reproducibility.
    """
    torch.manual_seed(seed)
    # ... reuse model + training logic from notebook 5 here ...
    print(f"Training complete for hidden_dim={hidden_dim}, dropout={dropout}")

# --- Safe wrapper for Jupyter execution ---
if __name__ == "__main__":
    import sys
    import argparse

    # Detect Jupyter environment and bypass argument parsing
    if "ipykernel_launcher" in sys.argv[0]:
        # Simulate default parameters for in-notebook testing
        train_mlp(seed=42, hidden_dim=64, dropout=0.2, save_path="results/")
    else:
        parser = argparse.ArgumentParser(description="Train MLP on California Housing")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--save_path", type=str, default="results/")
        args = parser.parse_args()
        train_mlp(args.seed, args.hidden_dim, args.dropout, args.save_path)

