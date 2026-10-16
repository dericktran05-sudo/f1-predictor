import pandas as pd

def clean(df):
    df = df.rename(columns={
        "Abbreviation": "driver", "TeamName": "team", "GridPosition": "grid",
        "Position": "finish", "Status": "status", "Points": "points",
    })
    n = df.groupby(["year", "round"])["driver"].transform("count")
    df["grid"] = df["grid"].where(df["grid"] > 0, n)   # pit-lane start = last
    df["dnf"] = ~df["status"].isin(["Finished", "Lapped"])
    return df.dropna(subset=["grid", "finish"])

old = pd.read_parquet("clean.parquet")                      # 2023-2025, already cleaned
new = clean(pd.read_parquet("races_2026.parquet"))          # 2026, cleaned the same way

keep = ["driver", "team", "grid", "finish", "status", "year", "round", "circuit", "dnf"]
df = pd.concat([old[keep], new[keep]], ignore_index=True)
df["race_id"] = df["year"] * 100 + df["round"]
df = df.sort_values(["driver", "race_id"]).reset_index(drop=True)

# Driver features carry across seasons
df["driver_form"] = (df.groupby("driver")["finish"]
                       .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
df["track_hist"] = (df.groupby(["driver", "circuit"])["finish"]
                      .transform(lambda x: x.shift(1).expanding().mean()))

# Team features reset every season (new car, new rules)
team_race = (df.groupby(["team", "year", "race_id"])
               .agg(team_finish=("finish", "mean"), team_dnf=("dnf", "mean"))
               .reset_index().sort_values(["team", "year", "race_id"]))
team_race["team_form"] = (team_race.groupby(["team", "year"])["team_finish"]
                          .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean()))
team_race["team_dnf_rate"] = (team_race.groupby(["team", "year"])["team_dnf"]
                              .transform(lambda x: x.shift(1).expanding().mean()))
df = df.merge(team_race[["team", "race_id", "team_form", "team_dnf_rate"]],
              on=["team", "race_id"])

df["driver_form"] = df["driver_form"].fillna(10.5)
df["team_form"] = df["team_form"].fillna(10.5)
df["team_dnf_rate"] = df["team_dnf_rate"].fillna(df["dnf"].mean())
df["track_hist"] = df["track_hist"].fillna(df["driver_form"])

df = df.sort_values(["race_id", "finish"]).reset_index(drop=True)
df.to_parquet("features_all.parquet")

print(df.groupby("year").size())

# Sanity check: each team's recent form going into the last 2026 race (lower = better)
last = df[(df["year"] == 2026) & (df["round"] == df[df["year"] == 2026]["round"].max())]
print()
print(last.groupby("team")["team_form"].first().sort_values().round(1))