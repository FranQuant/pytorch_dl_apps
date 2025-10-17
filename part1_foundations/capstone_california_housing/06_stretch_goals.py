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
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.serialization as ts

def train_mlp(seed=42, hidden_dim=64, dropout=0.2, save_path="results/"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Data & engineered features ---
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomsPerRoom"]   = df["AveBedrms"] / df["AveRooms"]
    df["PopPerHousehold"]   = df["Population"] / df["AveOccup"]
    df["IncomeBucket"] = pd.cut(
        df["MedInc"], bins=[0, 2, 4, 6, 8, np.inf],
        labels=["Very Low", "Low", "Mid", "High", "Very High"]
    )
    df = pd.get_dummies(df, columns=["IncomeBucket"], drop_first=True)

    X = df.drop("target", axis=1)
    y = df["target"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train_s = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_val_s   = torch.tensor(scaler.transform(X_val),   dtype=torch.float32)
    X_test_s  = torch.tensor(scaler.transform(X_test),  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

    # --- Model ---
    class MLPRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim=hidden_dim, dropout=dropout):
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
        def forward(self, x): return self.net(x)

    model = MLPRegressor(X_train_s.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Training loop with early stopping ---
    best_val = float("inf")
    wait, patience = 0, 10
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_s)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_s), y_val_t).item()

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            ts.add_safe_globals([StandardScaler])
            torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, "best_mlp.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {epoch}, best val loss={best_val:.4f}")
                break

    # --- Evaluate on test set ---
    ts.add_safe_globals([StandardScaler])
    bundle = torch.load("best_mlp.pt", weights_only=False)
    model.load_state_dict(bundle["model_state_dict"])
    scaler = bundle["scaler"]

    with torch.no_grad():
        y_pred = model(X_test_s).squeeze().numpy()
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_bundle = Path(save_path) / f"mlp_bundle_{stamp}.pt"
    torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, out_bundle)

    print(f"[MLP] seed={seed} hidden_dim={hidden_dim} dropout={dropout}  "
          f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
    print(f"Saved bundle → {out_bundle}")


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

