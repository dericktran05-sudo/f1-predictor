import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

today = pd.Timestamp.today().normalize()
schedule = fastf1.get_event_schedule(2026, include_testing=False)

rows = []
for _, event in schedule.iterrows():
    if event["EventDate"] > today:
        continue   # race hasn't happened yet
    rnd = event["RoundNumber"]
    try:
        session = fastf1.get_session(2026, rnd, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)

        res = session.results[
            ["Abbreviation", "TeamName", "GridPosition", "Position", "Status", "Points"]
        ].copy()
        res["year"] = 2026
        res["round"] = rnd
        res["circuit"] = event["Location"]
        res["rain"] = None
        res["track_temp"] = None

        rows.append(res)
        print("done:", rnd, event["Location"])
    except Exception as e:
        print("skipped:", rnd, event["Location"], "-", e)

df = pd.concat(rows, ignore_index=True)
df.to_parquet("races_2026.parquet")

print()
print("Saved", len(df), "rows from", df["round"].nunique(), "races")
print("Teams:", sorted(df["TeamName"].unique()))