from pathlib import Path
import duckdb

DB_PATH = Path("db/mvttdb.duckdb")

SOURCE_TABLE = "climat_clean"
TARGET_TABLE = "climat_final"


def main():
    print("Connexion DuckDB...")
    conn = duckdb.connect(str(DB_PATH))

    try:
        print("Normalisation finale...")

        conn.execute(f"""
        CREATE OR REPLACE TABLE {TARGET_TABLE} AS
        SELECT

            -- CLÉS STANDARD
            CAST(cluster AS INTEGER) AS cluster,
            CAST(Year AS INTEGER) AS Year,
            CAST(Municipality AS VARCHAR) AS Municipality,

            -- GEO
            nom_commune,
            nom_region,
            TRY_CAST(code_departement AS INTEGER) AS code_departement,

            TRY_CAST(latitude AS DOUBLE) AS latitude,
            TRY_CAST(longitude AS DOUBLE) AS longitude,

            -- CLIMAT
            TRY_CAST(temp_moyenne AS DOUBLE) AS temp_moyenne,
            TRY_CAST(tmax_mean AS DOUBLE) AS tmax_mean,
            TRY_CAST(tmin_mean AS DOUBLE) AS tmin_mean,
            TRY_CAST(precipitation_total AS DOUBLE) AS precipitation_total,
            TRY_CAST(precipitation_total_avril_septembre AS DOUBLE) AS precipitation_total_avril_septembre,
            TRY_CAST(amplitude_thermique AS DOUBLE) AS amplitude_thermique,

            -- INDICATEURS
            TRY_CAST(Huglin_Index AS DOUBLE) AS Huglin_Index,
            TRY_CAST(Hot_D AS DOUBLE) AS Hot_D,
            TRY_CAST(Very_Hot_D AS DOUBLE) AS Very_Hot_D,
            TRY_CAST(Frost_D AS DOUBLE) AS Frost_D,
            TRY_CAST(Late_Frost AS DOUBLE) AS Late_Frost,

            TRY_CAST(stress_climatique AS DOUBLE) AS stress_climatique,
            TRY_CAST(deficit_hydrique AS DOUBLE) AS deficit_hydrique,
            TRY_CAST(Climatic_Dryness_Index AS DOUBLE) AS Climatic_Dryness_Index,

            TRY_CAST(Soil_Water_Stock AS DOUBLE) AS Soil_Water_Stock,
            TRY_CAST(Soil_pH AS DOUBLE) AS Soil_pH,

            TRY_CAST(jours_secs AS DOUBLE) AS jours_secs,
            TRY_CAST(jours_pluie AS DOUBLE) AS jours_pluie

        FROM {SOURCE_TABLE}

        WHERE
            TRY_CAST(Year AS INTEGER) IS NOT NULL
            AND TRY_CAST(cluster AS INTEGER) BETWEEN 1 AND 7
        """)

        print("Table climat_final créée")

        # INDEX
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_climat_keys
            ON {TARGET_TABLE}(Municipality, Year)
        """)

        # VERIF
        count = conn.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}").fetchone()[0]
        print(f"Lignes : {count}")

        print("\nColonnes :")
        schema = conn.execute(f"DESCRIBE {TARGET_TABLE}").fetchall()
        for col in schema:
            print(f"{col[0]} -> {col[1]}")

        print("\nAperçu :")
        preview = conn.execute(f"""
            SELECT *
            FROM {TARGET_TABLE}
            LIMIT 5
        """).fetchdf()
        print(preview)

        print("\nContrôle qualité :")

        duplicates = conn.execute(f"""
            SELECT Municipality, Year, COUNT(*) as n
            FROM {TARGET_TABLE}
            GROUP BY Municipality, Year
            HAVING COUNT(*) > 1
        """).fetchall()

        print("Doublons détectés :" if duplicates else "Aucun doublon")

        null_years = conn.execute(f"""
            SELECT COUNT(*) FROM {TARGET_TABLE}
            WHERE Year IS NULL
        """).fetchone()[0]

        print(f"Years NULL : {null_years}")

        print("\nBase finale propre et prête")

    finally:
        conn.close()
        print("\nOK")


if __name__ == "__main__":
    main()