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

test["pred"] = model.predict(test[features])
test["pred_rank"] = test.groupby(["year", "round"])["pred"].rank(method="first")
test["grid_rank"] = test.groupby(["year", "round"])["grid"].rank(method="first")

rows = []
for (year, rnd), race in test.groupby(["year", "round"]):
    rows.append({
        "round": rnd,
        "circuit": race["circuit"].iloc[0],
        "dnfs": int(race["dnf"].sum()),
        "model_rho": spearmanr(race["pred_rank"], race["finish"]).correlation,
        "grid_rho": spearmanr(race["grid_rank"], race["finish"]).correlation,
        "model_err": (race["pred_rank"] - race["finish"]).abs().mean(),
    })
res = pd.DataFrame(rows)
res.to_parquet("race_results.parquet")

print("5 WORST RACES (lowest Spearman):")
print(res.sort_values("model_rho").head(5).round(2).to_string(index=False))

print("\n5 BEST RACES:")
print(res.sort_values("model_rho", ascending=False).head(5).round(2).to_string(index=False))

res["chaos"] = pd.cut(res["dnfs"], [-1, 1, 3, 100], labels=["0-1 DNFs", "2-3 DNFs", "4+ DNFs"])
print("\nBY NUMBER OF DNFs IN THE RACE:")
summary = res.groupby("chaos", observed=True).agg(
    races=("round", "count"),
    model_rho=("model_rho", "mean"),
    grid_rho=("grid_rho", "mean"),
    model_err=("model_err", "mean"),
)
print(summary.round(3).to_string())

# Drivers the model consistently misjudges (positive = finished worse than predicted)
test["miss"] = test["finish"] - test["pred_rank"]
drv = test.groupby("driver").agg(races=("miss", "count"), avg_miss=("miss", "mean"))
drv = drv[drv["races"] >= 8].sort_values("avg_miss")
print("\nDRIVERS THE MODEL UNDERRATES (finished better than predicted):")
print(drv.head(5).round(2).to_string())
print("\nDRIVERS THE MODEL OVERRATES (finished worse than predicted):")
print(drv.tail(5).round(2).to_string())