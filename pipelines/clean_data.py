def clean(con):

    print("🧹 Nettoyage...")

    # =========================================================
    # 🔍 CLIMAT
    # =========================================================
    cols = [c[0] for c in con.execute("DESCRIBE climat").fetchall()]
    print(f"📊 colonnes climat: {cols}")

    # ----------------------------
    # cluster → zone
    # ----------------------------
    if "zone" not in cols and "cluster" in cols:
        print("⚙️ cluster → zone")
        con.execute("ALTER TABLE climat RENAME COLUMN cluster TO zone")

    # refresh
    cols = [c[0] for c in con.execute("DESCRIBE climat").fetchall()]

    # ----------------------------
    # commune (clé critique)
    # ----------------------------
    if "Municipality" in cols:
        print("⚙️ création colonne commune")

        con.execute("""
        ALTER TABLE climat ADD COLUMN IF NOT EXISTS commune VARCHAR
        """)

        con.execute("""
        UPDATE climat
        SET commune = CAST(Municipality AS VARCHAR)
        """)

    # ----------------------------
    # année
    # ----------------------------
    if "annee" not in cols and "Year" in cols:
        print("⚙️ création annee depuis Year")

        con.execute("""
        ALTER TABLE climat ADD COLUMN annee INTEGER
        """)

        con.execute("""
        UPDATE climat
        SET annee = CAST(Year AS INTEGER)
        """)

    # ----------------------------
    # nettoyage numérique climat
    # ----------------------------
    numeric_cols = [
        "temp_moyenne",
        "precipitation_total",
        "tmax_mean",
        "tmin_mean",
        "amplitude_thermique",
        "stress_climatique",
        "deficit_hydrique"
    ]

    for col in numeric_cols:
        if col in cols:
            print(f"⚙️ nettoyage + cast {col}")

            con.execute(f"""
            UPDATE climat
            SET {col} = REGEXP_REPLACE(CAST({col} AS VARCHAR), '[^0-9,.-]', '')
            WHERE {col} IS NOT NULL
            """)

            con.execute(f"""
            UPDATE climat
            SET {col} = REPLACE({col}, ',', '.')
            WHERE {col} IS NOT NULL
            """)

            con.execute(f"""
            ALTER TABLE climat ALTER COLUMN {col} TYPE DOUBLE
            USING TRY_CAST(NULLIF({col}, '') AS DOUBLE)
            """)

    # ----------------------------
    # zone climat → INTEGER
    # ----------------------------
    if "zone" in cols:
        print("⚙️ cast zone climat → INTEGER")
        con.execute("ALTER TABLE climat ALTER COLUMN zone TYPE INTEGER")

    # =========================================================
    # 📊 RENDEMENT
    # =========================================================
    rend_cols = [c[0] for c in con.execute("DESCRIBE rendement").fetchall()]
    print(f"📊 colonnes rendement: {rend_cols}")

    # ----------------------------
    # Zone → zone
    # ----------------------------
    if "Zone" in rend_cols:
        print("⚙️ renommage Zone → zone")
        con.execute("ALTER TABLE rendement RENAME COLUMN Zone TO zone")

    # refresh
    rend_cols = [c[0] for c in con.execute("DESCRIBE rendement").fetchall()]

    # ----------------------------
    # 🔥 correction décalage zone (0→7 → 1→7)
    # ----------------------------
    if "zone" in rend_cols:
        print("⚙️ correction zone rendement (0→7 → 1→7)")

        con.execute("""
        UPDATE rendement
        SET zone = CAST(zone AS INTEGER) + 1
        """)

        con.execute("""
        ALTER TABLE rendement ALTER COLUMN zone TYPE INTEGER
        """)

    # ----------------------------
    # nettoyage rendement
    # ----------------------------
    if "rendement" in rend_cols:
        print("⚙️ nettoyage + cast rendement")

        con.execute("""
        UPDATE rendement
        SET rendement = REGEXP_REPLACE(CAST(rendement AS VARCHAR), '[^0-9,.-]', '')
        WHERE rendement IS NOT NULL
        """)

        con.execute("""
        UPDATE rendement
        SET rendement = REPLACE(CAST(rendement AS VARCHAR), ',', '.')
        WHERE rendement IS NOT NULL
        """)

        con.execute("""
        ALTER TABLE rendement ALTER COLUMN rendement TYPE DOUBLE
        USING TRY_CAST(NULLIF(CAST(rendement AS VARCHAR), '') AS DOUBLE)
        """)

    # ----------------------------
    # commune rendement (clé join)
    # ----------------------------
    if "commune" in rend_cols:
        print("⚙️ normalisation commune rendement")

        con.execute("""
        UPDATE rendement
        SET commune = CAST(commune AS VARCHAR)
        """)

    print("✅ clean terminé")