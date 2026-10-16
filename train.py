import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

df = pd.read_parquet("features.parquet")
features = ["grid", "driver_form", "team_form", "track_hist", "team_dnf_rate"]

train = df[df["year"] <= 2024]
test = df[df["year"] == 2025].copy()

model = lgb.LGBMRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_child_samples=20, verbose=-1,
)
model.fit(train[features], train["finish"])

# turn predictions into a predicted finishing order within each race
test["pred"] = model.predict(test[features])
test["model_rank"] = test.groupby(["year", "round"])["pred"].rank(method="first")
test["grid_rank"] = test.groupby(["year", "round"])["grid"].rank(method="first")

def score(col):
    rhos, maes = [], []
    for _, race in test.groupby(["year", "round"]):
        rhos.append(spearmanr(race[col], race["finish"]).correlation)
        maes.append((race[col] - race["finish"]).abs().mean())
    return sum(rhos) / len(rhos), sum(maes) / len(maes)

print("Test races:", test.groupby(["year", "round"]).ngroups)
for name, col in [("Baseline (grid)", "grid_rank"), ("Model", "model_rank")]:
    rho, mae = score(col)
    print(f"{name:16s} Spearman {rho:.3f}   Position error {mae:.2f}")

print()
print("Feature importance:")
for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:14s} {imp}")

test.to_parquet("predictions.parquet")