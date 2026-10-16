import fastf1

fastf1.Cache.enable_cache("cache")

session = fastf1.get_session(2024, 1, "R")
session.load(laps=False, telemetry=False, weather=False, messages=False)

print(session.results[["Abbreviation", "GridPosition", "Position"]])