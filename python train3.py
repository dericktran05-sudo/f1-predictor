import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

df = pd.read_parquet("features_all.parquet")
features = ["grid", "driver_form", "team_form", "track_hist", "team_dnf_rate"]
params = dict(n_estimators=200, learning_rate=0.05, max_depth=3,
              min_child_samples=20, verbose=-1)

def fit(train):
    return lgb.LGBMRegressor(**params).fit(train[features], train["finish"])

def score(data, col):
    rhos, maes = [], []
    for _, race in data.groupby(["year", "round"]):
        rhos.append(spearmanr(race[col], race["finish"]).correlation)
        maes.append((race[col] - race["finish"]).abs().mean())
    return sum(rhos) / len(rhos), sum(maes) / len(maes)

test = df[df["year"] == 2026].copy()
test["grid_rank"] = test.groupby(["year", "round"])["grid"].rank(method="first")

# A: frozen model, trained on 2023-2025 only
frozen = fit(df[df["year"] <= 2025])
test["pred_frozen"] = frozen.predict(test[features])

# B: walk-forward, retrained before every 2026 race
parts = []
for rnd in sorted(test["round"].unique()):
    past = df[(df["year"] < 2026) | ((df["year"] == 2026) & (df["round"] < rnd))]
    model = fit(past)
    race = test[test["round"] == rnd]
    parts.append(pd.Series(model.predict(race[features]), index=race.index))
test["pred_walk"] = pd.concat(parts)

for col in ["pred_frozen", "pred_walk"]:
    test[col.replace("pred", "rank")] = test.groupby(["year", "round"])[col].rank(method="first")

cols = [("Grid baseline", "grid_rank"), ("Frozen (2023-25)", "rank_frozen"),
        ("Walk-forward", "rank_walk")]
subsets = [("All 2026 races", test),
           ("Races 1-8", test[test["round"] <= 8]),
           ("Races 9-17", test[test["round"] >= 9])]

for label, subset in subsets:
    print(label, "-", subset.groupby(["year", "round"]).ngroups, "races")
    for name, col in cols:
        rho, mae = score(subset, col)
        print(f"  {name:18s} Spearman {rho:.3f}   Position error {mae:.2f}")
    print()

test.to_parquet("predictions_2026.parquet")