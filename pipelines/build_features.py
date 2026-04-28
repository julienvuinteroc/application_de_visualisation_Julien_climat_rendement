def build(con):

    print("🗺️ Construction features...")

    # =========================================================
    # 1️⃣ GEO FEATURES (raw)
    # =========================================================
    con.execute("""
    CREATE OR REPLACE TABLE geo_features AS
    SELECT 
        TRY_CAST(t.feature.properties.cluster AS INTEGER) AS zone,
        t.feature.geometry AS geometry
    FROM geo_raw
    CROSS JOIN UNNEST(geo_raw.features) AS t(feature)
    WHERE TRY_CAST(t.feature.properties.cluster AS INTEGER) IS NOT NULL
    """)

    count_features = con.execute("SELECT COUNT(*) FROM geo_features").fetchone()[0]
    print(f"📈 lignes geo_features: {count_features}")

    # =========================================================
    # 2️⃣ GEO UNIQUE PAR ZONE (CRITIQUE)
    # =========================================================
    con.execute("""
    CREATE OR REPLACE TABLE geo_zone AS
    SELECT 
        zone,
        ANY_VALUE(geometry) AS geometry
    FROM geo_features
    GROUP BY zone
    """)

    count_geo = con.execute("SELECT COUNT(*) FROM geo_zone").fetchone()[0]
    print(f"📈 lignes geo_zone (unique): {count_geo}")

    # =========================================================
    # 3️⃣ JOIN PROPRE
    # =========================================================
    con.execute("""
    CREATE OR REPLACE TABLE climat_rendement_geo AS
    SELECT 
        cr.*,
        g.geometry
    FROM climat_rendement cr
    LEFT JOIN geo_zone g
    ON cr.zone = g.zone
    """)

    count_final = con.execute("SELECT COUNT(*) FROM climat_rendement_geo").fetchone()[0]
    print(f"📈 lignes finales: {count_final}")

    # =========================================================
    # 4️⃣ CHECK
    # =========================================================
    null_geom = con.execute("""
        SELECT COUNT(*) 
        FROM climat_rendement_geo
        WHERE geometry IS NULL
    """).fetchone()[0]

    print(f"⚠️ géométries nulles: {null_geom}")

    print("✅ build terminé (CORRIGÉ)")