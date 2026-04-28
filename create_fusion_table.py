import duckdb

con = duckdb.connect("observatoire_viticole_climat.duckdb")

con.execute("""
CREATE OR REPLACE TABLE viticole_climat AS
SELECT
    v.*,

    c.Huglin_Index,
    c.hot_days,
    c.very_hot_days,
    c.frost_days,
    c.dryness_index,
    c.soil_ph

FROM viticole v

LEFT JOIN climat c

ON v.annee = c.annee
AND v.commune = c.commune
AND v.Zone = c.Zone
""")

print("Table viticole_climat créée")
