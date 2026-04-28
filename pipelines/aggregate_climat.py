def aggregate(con):

    print("📊 Agrégation climat...")

    cols = [c[0] for c in con.execute("DESCRIBE climat").fetchall()]
    print(f"📊 colonnes climat pour agrégation: {cols}")

    # ----------------------------
    # 🌡️ température
    # ----------------------------
    if "temp_moyenne" in cols:
        temp_expr = "AVG(temp_moyenne)"
    elif "temperature" in cols:
        temp_expr = "AVG(temperature)"
    else:
        raise Exception("❌ aucune colonne température")

    # ----------------------------
    # 🌧️ precipitation
    # ----------------------------
    if "precipitation_total" in cols:
        precip_expr = "SUM(precipitation_total)"
    elif "precipitation" in cols:
        precip_expr = "SUM(precipitation)"
    else:
        raise Exception("❌ aucune colonne precipitation")

    # ----------------------------
    # 📊 aggregation
    # ----------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE climat_agrege AS
    SELECT 
        zone,
        annee,
        {temp_expr} AS temperature_moyenne,
        {precip_expr} AS precipitation_totale
    FROM climat
    WHERE zone IS NOT NULL AND annee IS NOT NULL
    GROUP BY zone, annee
    """)

    count = con.execute("SELECT COUNT(*) FROM climat_agrege").fetchone()[0]
    print(f"📈 lignes climat_agrege: {count}")

    print("✅ agrégation terminée")