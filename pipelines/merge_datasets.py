def merge(con):

    print("🔗 Fusion climat + rendement (niveau commune)...")

    # =========================================================
    # 1️⃣ sécurisation clés
    # =========================================================
    print("⚙️ normalisation clés")

    con.execute("""
    CREATE OR REPLACE TABLE climat_clean AS
    SELECT
        CAST(zone AS INTEGER) AS zone,
        CAST(annee AS INTEGER) AS annee,
        CAST(commune AS VARCHAR) AS commune,
        temp_moyenne,
        precipitation_total,
        tmax_mean,
        tmin_mean,
        amplitude_thermique,
        stress_climatique,
        deficit_hydrique
    FROM climat
    WHERE commune IS NOT NULL
    """)

    con.execute("""
    CREATE OR REPLACE TABLE rendement_clean AS
    SELECT
        CAST(zone AS INTEGER) AS zone,
        CAST(annee AS INTEGER) AS annee,
        CAST(commune AS VARCHAR) AS commune,
        CAST(code_departement AS VARCHAR) AS code_departement,
        rendement
    FROM rendement
    WHERE commune IS NOT NULL
    """)

    # =========================================================
    # 2️⃣ merge principal (clé parfaite)
    # =========================================================
    print("⚙️ jointure commune + année")

    con.execute("""
    CREATE OR REPLACE TABLE climat_rendement AS
    SELECT 
        c.zone,
        c.annee,
        c.commune,

        -- climat
        c.temp_moyenne,
        c.precipitation_total,
        c.tmax_mean,
        c.tmin_mean,
        c.amplitude_thermique,
        c.stress_climatique,
        c.deficit_hydrique,

        -- rendement
        r.rendement,
        r.code_departement

    FROM climat_clean c
    LEFT JOIN rendement_clean r
    ON c.commune = r.commune
    AND c.annee = r.annee
    """)

    # =========================================================
    # 3️⃣ contrôle qualité
    # =========================================================
    total = con.execute("SELECT COUNT(*) FROM climat_rendement").fetchone()[0]

    missing_rendement = con.execute("""
        SELECT COUNT(*) 
        FROM climat_rendement
        WHERE rendement IS NULL
    """).fetchone()[0]

    print(f"📈 lignes totales: {total}")
    print(f"⚠️ rendement manquant: {missing_rendement}")

    # =========================================================
    # 4️⃣ index (perf)
    # =========================================================
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_commune ON climat_rendement(commune)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_annee ON climat_rendement(annee)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_zone ON climat_rendement(zone)")
    except:
        pass

    print("✅ merge terminé (niveau commune)")