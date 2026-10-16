import pandas as pd

df = pd.read_parquet("races.parquet")
print(df.head(10))
print(df.shape)
print(df["Status"].value_counts().head(10))