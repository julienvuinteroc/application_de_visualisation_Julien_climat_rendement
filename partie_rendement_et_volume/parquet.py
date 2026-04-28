import pandas as pd

df = pd.read_csv(
    "dataset_mvttdb_final.csv",
    sep=";",
    decimal=",",
    low_memory=False
)

df.to_parquet("dataset_mvttdb_final.parquet", index=False)
