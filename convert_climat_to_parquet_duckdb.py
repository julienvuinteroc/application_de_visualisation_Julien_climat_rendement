from pathlib import Path
import duckdb

CLIMAT_DATA_DIR = Path("/home/gelvic/mvttdb/application_mvp/Partie climat/data")
PROJECTIONS_DIR = CLIMAT_DATA_DIR / "Projections"

con = duckdb.connect()

def convert(csv_path: Path):
    parquet_path = csv_path.with_suffix(".parquet")
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{csv_path}')
        ) TO '{parquet_path}' (FORMAT PARQUET)
    """)
    print(f"✅ {csv_path.name} -> {parquet_path.name}")

for csv_file in CLIMAT_DATA_DIR.glob("*.csv"):
    convert(csv_file)

for csv_file in PROJECTIONS_DIR.glob("*.csv"):
    convert(csv_file)

print("🎉 Conversion DuckDB terminée.")
