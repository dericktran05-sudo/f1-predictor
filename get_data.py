import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache")

YEARS = range(2023, 2026)

rows = []
for year in YEARS:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        rnd = event["RoundNumber"]
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=False, telemetry=False, weather=True, messages=False)

            res = session.results[
                ["Abbreviation", "TeamName", "GridPosition", "Position", "Status", "Points"]
            ].copy()
            res["year"] = year
            res["round"] = rnd
            res["circuit"] = event["Location"]

            # weather is optional: if it didn't load, leave it blank
            try:
                w = session.weather_data
                res["rain"] = bool(w["Rainfall"].any())
                res["track_temp"] = w["TrackTemp"].mean()
            except Exception:
                res["rain"] = None
                res["track_temp"] = None

            rows.append(res)
            print("done:", year, rnd, event["Location"])
        except Exception as e:
            print("skipped:", year, rnd, "-", e)

df = pd.concat(rows, ignore_index=True)
df.to_parquet("races.parquet")
print("Saved", len(df), "rows")