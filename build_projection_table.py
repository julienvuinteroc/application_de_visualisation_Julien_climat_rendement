import duckdb
import pandas as pd
from pathlib import Path

DATA_DIR = Path("partie_climat/data/Projections")
DB_PATH = "db/mvttdb.duckdb"

conn = duckdb.connect(DB_PATH)

all_files = list(DATA_DIR.glob("*.csv"))

dfs = []

for file in all_files:
    print(f"Chargement : {file.name}")

    try:
        df = pd.read_csv(file, sep=";", encoding="utf-8", engine="python")
    except:
        # fallback si encodage bizarre
        df = pd.read_csv(file, sep=";", encoding="latin-1", engine="python")

    name = file.name.lower()

    # scénario
    if "optimiste" in name:
        scenario = "optimiste"
    elif "pessimiste" in name:
        scenario = "pessimiste"
    elif "neutre" in name:
        scenario = "neutre"
    else:
        scenario = "unknown"

    # période
    if "2021_2040" in name:
        periode = "2021-2040"
    elif "2041_2060" in name:
        periode = "2041-2060"
    else:
        periode = "unknown"

    df["scenario"] = scenario
    df["periode"] = periode

    dfs.append(df)

df_final = pd.concat(dfs, ignore_index=True)

# nettoyage
df_final["cluster"] = pd.to_numeric(df_final.get("zone"), errors="coerce")
df_final["annee"] = pd.to_numeric(df_final.get("annee"), errors="coerce")

conn.execute("DROP TABLE IF EXISTS climat_projection_final")

conn.execute("""
CREATE TABLE climat_projection_final AS
SELECT * FROM df_final
""")

conn.close()

print("Table climat_projection_final créée avec succès")