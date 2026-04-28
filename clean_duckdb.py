from pathlib import Path
import duckdb


DB_PATH = Path("db/mvttdb.duckdb")
SOURCE_TABLE = "climat_clean"
TARGET_TABLE = "climat_clean_fixed"


NUMERIC_DOUBLE_COLUMNS = {
    "Yield",
    "Huglin_Index",
    "Hot_D",
    "Very_Hot_D",
    "Sever_Heat",
    "Frost_D",
    "Late_Frost",
    "Sever_Frost",
    "Climatic_Dryness_Index",
    "Soil_Water_Stock",
    "Soil_pH",
    "stress_climatique",
    "jours_secs",
    "jours_pluie",
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "deficit_hydrique",
    "precipitation_total",
    "precipitation_total_avril_septembre",
    "amplitude_thermique",
    "latitude",
    "longitude",
}

INTEGER_COLUMNS = {
    "Year",
    "annee",
    "cluster",
    "zone",
    "code_departement",
}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_clean_expression(col_name: str) -> str:
    quoted = qident(col_name)

    # INTEGER
    if col_name in INTEGER_COLUMNS:
        return (
            f"TRY_CAST(NULLIF(TRIM(CAST({quoted} AS VARCHAR)), '') AS INTEGER) AS {quoted}"
        )

    # DOUBLE
    if col_name in NUMERIC_DOUBLE_COLUMNS:
        return (
            f"TRY_CAST("
            f"NULLIF(REPLACE(TRIM(CAST({quoted} AS VARCHAR)), ',', '.'), '') "
            f"AS DOUBLE"
            f") AS {quoted}"
        )

    # AUTRES (texte)
    return f"{quoted}"


def main():
    print("Connexion DuckDB...")
    conn = duckdb.connect(str(DB_PATH))

    try:
        # Vérification table source
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if SOURCE_TABLE not in tables:
            raise ValueError(f"Table '{SOURCE_TABLE}' introuvable")

        print(f"Lecture du schéma de {SOURCE_TABLE}...")
        schema_rows = conn.execute(f"DESCRIBE {qident(SOURCE_TABLE)}").fetchall()
        column_names = [row[0] for row in schema_rows]

        if not column_names:
            raise ValueError("Aucune colonne trouvée")

        print("Construction de la requête...")
        select_exprs = [build_clean_expression(col) for col in column_names]
        select_sql = ",\n        ".join(select_exprs)

        # 🔥 FIX CRITIQUE : suppression ligne parasite
        create_sql = f"""
        CREATE OR REPLACE TABLE {qident(TARGET_TABLE)} AS
        SELECT
            {select_sql}
        FROM {qident(SOURCE_TABLE)}
        WHERE TRY_CAST(CAST("Year" AS VARCHAR) AS INTEGER) IS NOT NULL
        """

        print(f"Création de {TARGET_TABLE}...")
        conn.execute(create_sql)

        # =====================================================
        # VERIFICATION
        # =====================================================

        print("Vérification...")

        count_source = conn.execute(
            f"SELECT COUNT(*) FROM {qident(SOURCE_TABLE)}"
        ).fetchone()[0]

        count_target = conn.execute(
            f"SELECT COUNT(*) FROM {qident(TARGET_TABLE)}"
        ).fetchone()[0]

        print(f"Lignes source : {count_source}")
        print(f"Lignes cible  : {count_target}")

        print("\nSchéma cible :")
        schema_fixed = conn.execute(
            f"DESCRIBE {qident(TARGET_TABLE)}"
        ).fetchall()

        for row in schema_fixed:
            print(f"{row[0]} -> {row[1]}")

        print("\nAperçu :")
        preview = conn.execute(
            f"""
            SELECT *
            FROM {qident(TARGET_TABLE)}
            LIMIT 5
            """
        ).fetchdf()

        print(preview)

        print(f"\nTable créée : {TARGET_TABLE}")

    finally:
        conn.close()
        print("Connexion fermée.")


if __name__ == "__main__":
    main()