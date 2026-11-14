import pandas as pd
df = pd.read_csv("data/census.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df["salary"].value_counts(normalize=True))
print((df == "?").sum())
