import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

res = pd.read_parquet("race_results.parquet")

# Chart 1: every 2025 race, DNFs vs model accuracy
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(res["dnfs"], res["model_rho"], s=60)
for _, r in res.iterrows():
    ax.annotate(r["circuit"], (r["dnfs"], r["model_rho"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
slope, intercept = np.polyfit(res["dnfs"], res["model_rho"], 1)
xs = np.array([res["dnfs"].min(), res["dnfs"].max()])
ax.plot(xs, slope * xs + intercept, linestyle="--")
ax.set_xlabel("Number of DNFs in the race")
ax.set_ylabel("Model accuracy (Spearman)")
ax.set_title("More retirements, worse predictions (2025 season)")
plt.tight_layout()
plt.savefig("chart_dnfs.png", dpi=150)

# Chart 2: model vs grid baseline by how chaotic the race was
res["chaos"] = pd.cut(res["dnfs"], [-1, 1, 3, 100],
                      labels=["0-1 DNFs", "2-3 DNFs", "4+ DNFs"])
summary = res.groupby("chaos", observed=True)[["model_rho", "grid_rho"]].mean()
summary.columns = ["Model", "Grid baseline"]
summary.plot(kind="bar", figsize=(7, 5))
plt.ylabel("Average Spearman")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Both predictors fall apart in chaotic races")
plt.tight_layout()
plt.savefig("chart_chaos.png", dpi=150)

print("Saved chart_dnfs.png and chart_chaos.png")