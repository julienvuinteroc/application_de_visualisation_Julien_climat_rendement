import duckdb

con = duckdb.connect("observatoire_viticole_climat.duckdb")

con.execute("""
CREATE OR REPLACE TABLE climat AS
SELECT
    CAST(Year AS INTEGER) AS annee,
    CAST(Municipality AS VARCHAR) AS commune,
    CAST(cluster AS VARCHAR) AS Zone,

    CAST(REPLACE(CAST(Yield AS VARCHAR), ',', '.') AS DOUBLE) AS yield_climat,
    CAST(REPLACE(CAST(Huglin_Index AS VARCHAR), ',', '.') AS DOUBLE) AS Huglin_Index,
    CAST(REPLACE(CAST(Hot_D AS VARCHAR), ',', '.') AS DOUBLE) AS hot_days,
    CAST(REPLACE(CAST(Very_Hot_D AS VARCHAR), ',', '.') AS DOUBLE) AS very_hot_days,
    CAST(REPLACE(CAST(Frost_D AS VARCHAR), ',', '.') AS DOUBLE) AS frost_days,
    CAST(REPLACE(CAST(Climatic_Dryness_Index AS VARCHAR), ',', '.') AS DOUBLE) AS dryness_index,
    CAST(REPLACE(CAST(Soil_pH AS VARCHAR), ',', '.') AS DOUBLE) AS soil_ph

FROM read_csv_auto(
    'Partie climat/data/Data_Eng_7_zones.csv',
    delim=';'
)
""")

print("Table climat reconstruite correctement")