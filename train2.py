import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

df = pd.read_parquet("features.parquet")
features = ["grid", "driver_form", "team_form", "track_hist", "team_dnf_rate"]

train = df[df["year"] <= 2024].copy()
test = df[df["year"] == 2025].copy()

params = dict(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_child_samples=20, verbose=-1, importance_type="gain",
)

# Model A: predict the finishing position directly (what we did before)
model_a = lgb.LGBMRegressor(**params).fit(train[features], train["finish"])
test["pred_a"] = model_a.predict(test[features])

# Model B: predict places gained or lost versus the grid
model_b = lgb.LGBMRegressor(**params).fit(train[features], train["finish"] - train["grid"])
test["pred_b"] = test["grid"] + model_b.predict(test[features])

for col in ["grid", "pred_a", "pred_b"]:
    test[col + "_rank"] = test.groupby(["year", "round"])[col].rank(method="first")

def score(col, finishers_only=False):
    rhos, maes = [], []
    for _, race in test.groupby(["year", "round"]):
        if finishers_only:
            race = race[~race["dnf"]]
        rhos.append(spearmanr(race[col], race["finish"]).correlation)
        maes.append((race[col] - race["finish"]).abs().mean())
    return sum(rhos) / len(rhos), sum(maes) / len(maes)

print("All drivers:")
for name, col in [("Baseline (grid)", "grid_rank"), ("Model A (finish)", "pred_a_rank"), ("Model B (change)", "pred_b_rank")]:
    rho, mae = score(col)
    print(f"  {name:17s} Spearman {rho:.3f}   Position error {mae:.2f}")

print()
print("Finishers only (DNFs removed):")
for name, col in [("Baseline (grid)", "grid_rank"), ("Model A (finish)", "pred_a_rank"), ("Model B (change)", "pred_b_rank")]:
    rho, _ = score(col, finishers_only=True)
    print(f"  {name:17s} Spearman {rho:.3f}")

print()
print("Feature importance (Model B, by gain):")
for f, imp in sorted(zip(features, model_b.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:14s} {imp:.0f}")