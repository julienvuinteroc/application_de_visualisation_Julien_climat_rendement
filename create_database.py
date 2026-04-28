# -*- coding: utf-8 -*-

from pathlib import Path
import duckdb


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "db" / "mvttdb.duckdb"


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    rows = conn.execute(f"DESCRIBE {qident(table_name)}").fetchall()
    return [row[0] for row in rows]


def first_existing_column(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    candidates: list[str],
) -> str | None:
    cols = set(get_columns(conn, table_name))
    for col in candidates:
        if col in cols:
            return col
    return None


def text_expr_from_candidates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    candidates: list[str],
    alias: str,
    default_sql: str = "NULL",
) -> str:
    cols = set(get_columns(conn, table_name))
    existing = []

    for col in candidates:
        if col in cols:
            existing.append(f"NULLIF(TRIM(CAST({qident(col)} AS VARCHAR)), '')")

    if not existing:
        return f"{default_sql} AS {qident(alias)}"

    return f"COALESCE({', '.join(existing)}) AS {qident(alias)}"


def numeric_expr_from_candidates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    candidates: list[str],
    alias: str,
    cast_type: str = "DOUBLE",
    default_sql: str = "NULL",
) -> str:
    cols = set(get_columns(conn, table_name))
    existing = []

    for col in candidates:
        if col in cols:
            existing.append(
                f"TRY_CAST(NULLIF(REPLACE(TRIM(CAST({qident(col)} AS VARCHAR)), ',', '.'), '') AS {cast_type})"
            )

    if not existing:
        return f"{default_sql} AS {qident(alias)}"

    return f"COALESCE({', '.join(existing)}) AS {qident(alias)}"


def build_projection_metric_exprs(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> list[str]:
    exprs = []

    exprs.append(
        text_expr_from_candidates(
            conn,
            table_name,
            ["Municipality", "code_commune", "commune_code"],
            "commune_code",
        )
    )
    exprs.append(
        text_expr_from_candidates(
            conn,
            table_name,
            ["nom_commune", "commune", "nom"],
            "nom_commune",
        )
    )
    exprs.append(
        text_expr_from_candidates(
            conn,
            table_name,
            ["nom_region", "region"],
            "nom_region",
        )
    )

    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            ["cluster", "zone"],
            "zone",
            cast_type="INTEGER",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            ["code_departement", "departement", "code_dep"],
            "code_departement",
            cast_type="INTEGER",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            ["latitude", "lat"],
            "latitude",
            cast_type="DOUBLE",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            ["longitude", "lon"],
            "longitude",
            cast_type="DOUBLE",
        )
    )

    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            [
                "temp_moyenne",
                "EC_Earth_tmoy",
                "HadGEM3_tmoy",
                "MPI-ESM1_tmoy",
                "UKESM1_tmoy",
                "IPSL_tmoy",
                "tmoy",
                "tmean",
            ],
            "temp_moyenne",
            cast_type="DOUBLE",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            [
                "tmax_mean",
                "EC_Earth_tmax",
                "HadGEM3_tmax",
                "MPI-ESM1_tmax",
                "UKESM1_tmax_mean",
                "IPSL_tmax_mean",
                "tmax",
            ],
            "tmax_mean",
            cast_type="DOUBLE",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            [
                "tmin_mean",
                "EC_Earth_tmin",
                "HadGEM3_tmin",
                "MPI-ESM1_tmin",
                "UKESM1_tmin",
                "IPSL_tmin",
                "tmin",
            ],
            "tmin_mean",
            cast_type="DOUBLE",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            [
                "precipitation_total",
                "EC_Earth_precipitation_total",
                "HadGEM3_precipitation_total",
                "MPI-ESM1_precipitation_total",
                "UKESM1_precipitation_total",
                "IPSL_precipitation_total",
                "precip",
                "prec",
                "pr",
            ],
            "precipitation_total",
            cast_type="DOUBLE",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            ["precipitation_total_avril_septembre"],
            "precipitation_total_avril_septembre",
            cast_type="DOUBLE",
        )
    )
    exprs.append(
        numeric_expr_from_candidates(
            conn,
            table_name,
            ["amplitude_thermique"],
            "amplitude_thermique",
            cast_type="DOUBLE",
        )
    )

    return exprs


def init_db():
    print("Connexion DuckDB...")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))

    try:
        climat_file = DATA_DIR / "climat" / "dataset_clean_climat_V0_2_final_bis.csv"
        geo_file = DATA_DIR / "geojson" / "zones_pedoclimatiques.geojson"
        projections_dir = DATA_DIR / "projections"
        rendement_file = DATA_DIR / "rendement" / "dataset_mvttdb_final.parquet"

        # =====================================================
        # 1. CLIMAT HISTORIQUE BRUT
        # =====================================================
        print("Import climat_clean...")

        conn.execute(f"""
            CREATE OR REPLACE TABLE climat_clean AS
            SELECT *
            FROM read_csv_auto(
                '{climat_file.as_posix()}',
                SAMPLE_SIZE=-1,
                HEADER=TRUE,
                ALL_VARCHAR=TRUE
            )
        """)

        print("climat_clean OK")

        # =====================================================
        # 2. CLIMAT HISTORIQUE NORMALISÉ
        # =====================================================
        print("Création climat_final...")

        climat_year_col = first_existing_column(conn, "climat_clean", ["Year", "annee", "year"])
        climat_cluster_col = first_existing_column(conn, "climat_clean", ["cluster", "zone"])

        climat_year_expr = numeric_expr_from_candidates(
            conn,
            "climat_clean",
            ["Year", "annee", "year"],
            "Year",
            cast_type="INTEGER",
        )
        climat_cluster_expr = numeric_expr_from_candidates(
            conn,
            "climat_clean",
            ["cluster", "zone"],
            "cluster",
            cast_type="INTEGER",
        )
        climat_muni_expr = text_expr_from_candidates(
            conn,
            "climat_clean",
            ["Municipality", "code_commune", "commune_code", "commune"],
            "Municipality",
        )

        if climat_year_col is None or climat_cluster_col is None:
            raise ValueError(
                "Impossible de normaliser climat_clean : colonnes Year/annee ou cluster/zone introuvables."
            )

        climat_year_filter = (
            f"TRY_CAST(NULLIF(REPLACE(TRIM(CAST({qident(climat_year_col)} AS VARCHAR)), ',', '.'), '') AS INTEGER) IS NOT NULL"
        )
        climat_cluster_filter = (
            f"TRY_CAST(NULLIF(REPLACE(TRIM(CAST({qident(climat_cluster_col)} AS VARCHAR)), ',', '.'), '') AS INTEGER) BETWEEN 1 AND 7"
        )

        conn.execute(f"""
            CREATE OR REPLACE TABLE climat_final AS
            SELECT
                {climat_cluster_expr},
                {climat_year_expr},
                {climat_muni_expr},

                {text_expr_from_candidates(conn, "climat_clean", ["nom_commune"], "nom_commune")},
                {text_expr_from_candidates(conn, "climat_clean", ["nom_region"], "nom_region")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["code_departement"], "code_departement", cast_type="INTEGER")},

                {numeric_expr_from_candidates(conn, "climat_clean", ["latitude"], "latitude", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["longitude"], "longitude", cast_type="DOUBLE")},

                {numeric_expr_from_candidates(conn, "climat_clean", ["temp_moyenne"], "temp_moyenne", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["tmax_mean"], "tmax_mean", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["tmin_mean"], "tmin_mean", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["precipitation_total"], "precipitation_total", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["precipitation_total_avril_septembre"], "precipitation_total_avril_septembre", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["amplitude_thermique"], "amplitude_thermique", cast_type="DOUBLE")},

                {numeric_expr_from_candidates(conn, "climat_clean", ["Huglin_Index"], "Huglin_Index", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["Hot_D"], "Hot_D", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["Very_Hot_D"], "Very_Hot_D", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["Frost_D"], "Frost_D", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["Late_Frost"], "Late_Frost", cast_type="DOUBLE")},

                {numeric_expr_from_candidates(conn, "climat_clean", ["stress_climatique"], "stress_climatique", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["deficit_hydrique"], "deficit_hydrique", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["Climatic_Dryness_Index"], "Climatic_Dryness_Index", cast_type="DOUBLE")},

                {numeric_expr_from_candidates(conn, "climat_clean", ["Soil_Water_Stock"], "Soil_Water_Stock", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["Soil_pH"], "Soil_pH", cast_type="DOUBLE")},

                {numeric_expr_from_candidates(conn, "climat_clean", ["jours_secs"], "jours_secs", cast_type="DOUBLE")},
                {numeric_expr_from_candidates(conn, "climat_clean", ["jours_pluie"], "jours_pluie", cast_type="DOUBLE")}
            FROM climat_clean
            WHERE
                {climat_year_filter}
                AND {climat_cluster_filter}
        """)

        print("climat_final OK")

        # =====================================================
        # 3. GEO ZONE
        # =====================================================
        print("Import geo_zone...")

        conn.execute(f"""
            CREATE OR REPLACE TABLE geo_zone AS
            SELECT
                CAST(
                    COALESCE(
                        json_extract_string(f.value, '$.properties.zone'),
                        json_extract_string(f.value, '$.properties.cluster')
                    ) AS INTEGER
                ) AS zone,
                COALESCE(
                    json_extract_string(f.value, '$.properties.code_commune'),
                    json_extract_string(f.value, '$.properties.code'),
                    json_extract_string(f.value, '$.properties.insee')
                ) AS commune_code,
                json_extract_string(f.value, '$.properties.nom') AS nom_commune,
                json_extract(f.value, '$.geometry') AS geometry
            FROM read_json_auto(
                '{geo_file.as_posix()}',
                maximum_object_size=200000000
            ),
            LATERAL json_each(features) AS f
            WHERE CAST(
                COALESCE(
                    json_extract_string(f.value, '$.properties.zone'),
                    json_extract_string(f.value, '$.properties.cluster')
                ) AS INTEGER
            ) BETWEEN 1 AND 7
        """)

        print("geo_zone OK")

        # =====================================================
        # 4. PROJECTIONS BRUTES
        # =====================================================
        print("Import climat_projection_raw...")

        conn.execute(f"""
            CREATE OR REPLACE TABLE climat_projection_raw AS
            SELECT *
            FROM read_csv_auto(
                '{(projections_dir / "*.csv").as_posix()}',
                SAMPLE_SIZE=-1,
                union_by_name=True,
                filename=True,
                HEADER=TRUE,
                ALL_VARCHAR=TRUE
            )
        """)

        print("climat_projection_raw OK")

        # =====================================================
        # 5. PROJECTIONS NORMALISÉES
        # =====================================================
        print("Création climat_projection_final...")

        projection_exprs = build_projection_metric_exprs(conn, "climat_projection_raw")
        projection_sql = ",\n                    ".join(projection_exprs)

        conn.execute(f"""
            CREATE OR REPLACE TABLE climat_projection_final AS
            SELECT
                zone,
                commune_code,
                CASE
                    WHEN lower(filename) LIKE '%2021_2040%' THEN 2030
                    WHEN lower(filename) LIKE '%2041_2060%' THEN 2050
                    ELSE NULL
                END AS year,

                nom_commune,
                nom_region,
                code_departement,
                latitude,
                longitude,

                temp_moyenne,
                tmax_mean,
                tmin_mean,
                precipitation_total,
                precipitation_total_avril_septembre,
                amplitude_thermique,

                CASE
                    WHEN lower(filename) LIKE '%ssp126%' OR lower(filename) LIKE '%optimiste%' THEN 'optimiste'
                    WHEN lower(filename) LIKE '%ssp245%' OR lower(filename) LIKE '%neutre%' THEN 'neutre'
                    WHEN lower(filename) LIKE '%ssp585%' OR lower(filename) LIKE '%pessimiste%' THEN 'pessimiste'
                    ELSE 'sans_scenario'
                END AS scenario,

                CASE
                    WHEN lower(filename) LIKE '%2021_2040%' THEN '2021-2040'
                    WHEN lower(filename) LIKE '%2041_2060%' THEN '2041-2060'
                    ELSE 'unknown'
                END AS periode,

                filename
            FROM (
                SELECT
                    {projection_sql},
                    filename
                FROM climat_projection_raw
            ) t
            WHERE zone BETWEEN 1 AND 7
        """)

        print("climat_projection_final OK")

        # =====================================================
        # 6. RENDEMENT
        # =====================================================
        print("Import rendement...")

        rendement_ok = False

        try:
            conn.execute(f"""
                CREATE OR REPLACE TABLE rendement_raw AS
                SELECT *
                FROM read_parquet('{rendement_file.as_posix()}')
            """)

            cols = set(get_columns(conn, "rendement_raw"))

            year_col = next((c for c in ["Year", "annee", "year"] if c in cols), None)
            zone_col = next((c for c in ["cluster", "zone"] if c in cols), None)
            commune_col = next((c for c in ["Municipality", "code_commune", "commune_code"] if c in cols), None)

            if year_col and zone_col and commune_col:
                conn.execute(f"""
                    CREATE OR REPLACE TABLE rendement AS
                    SELECT
                        TRY_CAST({qident(year_col)} AS INTEGER) AS Year,
                        TRY_CAST({qident(zone_col)} AS INTEGER) AS cluster,
                        CAST({qident(commune_col)} AS VARCHAR) AS Municipality,
                        *
                    FROM rendement_raw
                """)
                rendement_ok = True
                print("rendement OK")
            else:
                print("rendement ignoré (colonnes non standard)")

        except Exception as e:
            print(f"rendement skipped : {e}")

        # =====================================================
        # 7. INDEXES
        # =====================================================
        print("Création des index...")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_climat_final_keys
            ON climat_final(Municipality, Year)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_climat_final_cluster_year
            ON climat_final(cluster, Year)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_projection_final_zone_year_scenario
            ON climat_projection_final(zone, year, scenario)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_projection_final_commune
            ON climat_projection_final(commune_code)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_geo_zone_commune
            ON geo_zone(commune_code)
        """)

        print("Index OK")

        # =====================================================
        # 8. VÉRIFICATION
        # =====================================================
        print("\nVerification")

        tables = [
            "climat_clean",
            "climat_final",
            "geo_zone",
            "climat_projection_raw",
            "climat_projection_final",
        ]

        if rendement_ok:
            tables.append("rendement")

        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0]
            print(f"{table} -> {count} lignes")

        print("\nSchéma climat_final :")
        for row in conn.execute("DESCRIBE climat_final").fetchall():
            print(f"{row[0]} -> {row[1]}")

        print("\nSchéma climat_projection_final :")
        for row in conn.execute("DESCRIBE climat_projection_final").fetchall():
            print(f"{row[0]} -> {row[1]}")

        print("\nAperçu climat_projection_final :")
        preview = conn.execute("""
            SELECT
                zone,
                commune_code,
                year,
                scenario,
                periode,
                temp_moyenne,
                tmax_mean,
                tmin_mean,
                precipitation_total,
                precipitation_total_avril_septembre
            FROM climat_projection_final
            LIMIT 10
        """).fetchdf()
        print(preview)

        print("\nBase prête.")

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()