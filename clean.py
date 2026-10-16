import pandas as pd

df = pd.read_parquet("races.parquet")

# friendlier column names
df = df.rename(columns={
    "Abbreviation": "driver",
    "TeamName": "team",
    "GridPosition": "grid",
    "Position": "finish",
    "Status": "status",
    "Points": "points",
})

# grid position 0 means a pit-lane start, so treat it as last on the grid
df["grid"] = df["grid"].replace(0, 20)

# mark drivers who did not finish the race (lapped cars still count as finishing)
df["dnf"] = ~df["status"].isin(["Finished", "Lapped"])

# drop any rows missing the essentials, then sort by date
df = df.dropna(subset=["grid", "finish"])
df = df.sort_values(["year", "round", "finish"]).reset_index(drop=True)

df.to_parquet("clean.parquet")
print(df.shape)
print("DNF rate:", round(df["dnf"].mean(), 3))