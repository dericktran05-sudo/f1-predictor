import sys
import fastf1
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

fastf1.Cache.enable_cache("cache")
YEAR = 2026
today = pd.Timestamp.today().normalize()
schedule = fastf1.get_event_schedule(YEAR, include_testing=False)

# pass a round number to test on a past race, or run with no number for the next race
if len(sys.argv) > 1:
    ROUND = int(sys.argv[1])
else:
    ROUND = int(schedule[schedule["EventDate"] >= today].iloc[0]["RoundNumber"])
event = schedule[schedule["RoundNumber"] == ROUND].iloc[0]
circuit = event["Location"]
target_id = YEAR * 100 + ROUND
print("Predicting:", event["EventName"], "- round", ROUND)

df = pd.read_parquet("features_all.parquet")
hist = df[df["race_id"] < target_id]      # only races before this one

# 1. Qualifying result becomes the grid
try:
    q = fastf1.get_session(YEAR, ROUND, "Q")
    q.load(laps=False, telemetry=False, weather=False, messages=False)
    quali = q.results[["Abbreviation", "TeamName", "Position"]].copy()
except Exception as e:
    print("Could not load qualifying (has it happened yet?):", e)
    sys.exit()
if quali.empty:
    print("No qualifying results yet. Run this after qualifying on Saturday, Oct 24.")
    sys.exit()
quali.columns = ["driver", "team", "grid"]
quali["grid"] = quali["grid"].fillna(len(quali)).astype(float)

# 2. Features, built the same way as before, from past races only
cur = hist[hist["year"] == YEAR]
team_race = (cur.groupby(["team", "race_id"])
                .agg(f=("finish", "mean"), d=("dnf", "mean"))
                .reset_index().sort_values("race_id"))

def driver_form(d):
    x = hist[hist["driver"] == d].sort_values("race_id")["finish"]
    return x.tail(5).mean() if len(x) else 10.5

def track_hist(d, fallback):
    x = hist[(hist["driver"] == d) & (hist["circuit"] == circuit)]["finish"]
    return x.mean() if len(x) else fallback

def team_form(t):
    x = team_race[team_race["team"] == t]["f"]
    return x.tail(6).mean() if len(x) else 10.5

def team_dnf(t):
    x = team_race[team_race["team"] == t]["d"]
    return x.mean() if len(x) else hist["dnf"].mean()

quali["driver_form"] = quali["driver"].map(driver_form)
quali["track_hist"] = [track_hist(d, f) for d, f in zip(quali["driver"], quali["driver_form"])]
quali["team_form"] = quali["team"].map(team_form)
quali["team_dnf_rate"] = quali["team"].map(team_dnf)

# 3. Train on every earlier race and predict
features = ["grid", "driver_form", "team_form", "track_hist", "team_dnf_rate"]
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                          min_child_samples=20, verbose=-1)
model.fit(hist[features], hist["finish"])
quali["predicted"] = model.predict(quali[features])
quali["predicted"] = quali["predicted"].rank(method="first").astype(int)
out = quali.sort_values("predicted")[["predicted", "driver", "team", "grid"]]

# 4. Show it, and score it if the real result is already known
actual = df[df["race_id"] == target_id][["driver", "finish"]]
if len(actual):
    m = out.merge(actual, on="driver")
    m["finish"] = m["finish"].astype(int)
    print(m.to_string(index=False))
    print("\nSpearman - model:", round(spearmanr(m["predicted"], m["finish"]).correlation, 3),
          " grid:", round(spearmanr(m["grid"], m["finish"]).correlation, 3))
else:
    print(out.to_string(index=False))

out.to_csv(f"prediction_round_{ROUND}.csv", index=False)