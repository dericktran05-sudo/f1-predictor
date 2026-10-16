import pandas as pd
from scipy.stats import spearmanr

df = pd.read_parquet("clean.parquet")

results = []
for (year, rnd), race in df.groupby(["year", "round"]):
    pred = race["grid"].rank(method="first")
    rho = spearmanr(pred, race["finish"]).correlation
    mae = (pred - race["finish"]).abs().mean()
    results.append({"year": year, "round": rnd, "spearman": rho, "mae": mae})

res = pd.DataFrame(results)
print("Races scored:", len(res))
print("Average Spearman:", round(res["spearman"].mean(), 3))
print("Average position error:", round(res["mae"].mean(), 2))
print()
print(res.groupby("year")[["spearman", "mae"]].mean().round(3))