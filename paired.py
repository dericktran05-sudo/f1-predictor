import numpy as np
import pandas as pd
from scipy.stats import spearmanr

test = pd.read_parquet("predictions_2026.parquet")

rows = []
for (year, rnd), race in test.groupby(["year", "round"]):
    rows.append({
        "round": rnd,
        "grid": spearmanr(race["grid_rank"], race["finish"]).correlation,
        "frozen": spearmanr(race["rank_frozen"], race["finish"]).correlation,
        "walk": spearmanr(race["rank_walk"], race["finish"]).correlation,
    })
r = pd.DataFrame(rows)

def compare(a, b, label):
    d = r[a] - r[b]
    se = d.std() / np.sqrt(len(d))
    print(f"{label}: mean {d.mean():+.3f} (+/- {se:.3f}), "
          f"better in {int((d > 0).sum())} of {len(d)} races")

compare("walk", "grid", "Walk-forward vs grid")
compare("frozen", "grid", "Frozen vs grid     ")
compare("walk", "frozen", "Walk-forward vs frozen")