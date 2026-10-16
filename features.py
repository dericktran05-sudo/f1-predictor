import pandas as pd

df = pd.read_parquet("clean.parquet")
df["race_id"] = df["year"] * 100 + df["round"]
df = df.sort_values(["driver", "race_id"]).reset_index(drop=True)

# 1. Driver form: average finish over their previous 5 races
df["driver_form"] = (
    df.groupby("driver")["finish"]
      .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# 2. Track history: driver's average finish at this circuit in past visits
df["track_hist"] = (
    df.groupby(["driver", "circuit"])["finish"]
      .transform(lambda x: x.shift(1).expanding().mean())
)

# 3 & 4. Team features, calculated per race so teammates don't leak into each other
team_race = (
    df.groupby(["team", "race_id"])
      .agg(team_finish=("finish", "mean"), team_dnf=("dnf", "mean"))
      .reset_index()
      .sort_values(["team", "race_id"])
)
team_race["team_form"] = (
    team_race.groupby("team")["team_finish"]
             .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
)
team_race["team_dnf_rate"] = (
    team_race.groupby("team")["team_dnf"]
             .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
)
df = df.merge(
    team_race[["team", "race_id", "team_form", "team_dnf_rate"]],
    on=["team", "race_id"],
)

# Fill gaps for a driver/team's first races (no history yet)
MID = 10.5
df["driver_form"] = df["driver_form"].fillna(MID)
df["team_form"] = df["team_form"].fillna(MID)
df["team_dnf_rate"] = df["team_dnf_rate"].fillna(df["dnf"].mean())
df["track_hist"] = df["track_hist"].fillna(df["driver_form"])

df = df.sort_values(["race_id", "finish"]).reset_index(drop=True)
df.to_parquet("features.parquet")

print(df.shape)
print(df[["driver_form", "track_hist", "team_form", "team_dnf_rate"]].describe().round(2))

# Leakage check: VER's form before round 2 should equal his round 1 finish
print()
print(df[df["driver"] == "VER"][["year", "round", "finish", "driver_form"]].head(5))