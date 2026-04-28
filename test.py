import duckdb

conn = duckdb.connect("db/mvttdb.duckdb")

df = conn.execute("""
SELECT json_extract(f.value, '$.properties')
FROM read_json_auto('data/geojson/zones_pedoclimatiques.geojson'),
LATERAL json_each(features) AS f
LIMIT 5
""").df()

print(df)
