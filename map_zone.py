from pathlib import Path
import json

import duckdb
import pandas as pd
import geopandas as gpd
import folium

from shapely.geometry import shape
from branca.colormap import linear


# =====================================================
# CONFIG
# =====================================================

DB_PATH = Path("db/mvttdb.duckdb")
OUTPUT_HTML = Path("map_zone.html")


# =====================================================
# UTILS
# =====================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_geometry(value):
    """
    Convertit une géométrie stockée en texte JSON DuckDB vers objet shapely.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, dict):
        return shape(value)

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return shape(json.loads(value))
        except json.JSONDecodeError:
            return None

    return None


# =====================================================
# LOAD GEO
# =====================================================

def load_geo_zone(conn: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    df = conn.execute("""
        SELECT zone, geometry
        FROM geo_zone
        WHERE zone BETWEEN 1 AND 7
    """).df()

    if df.empty:
        return gpd.GeoDataFrame(columns=["zone", "geometry"], geometry="geometry", crs="EPSG:4326")

    df["zone"] = safe_numeric(df["zone"])
    df["geometry"] = df["geometry"].apply(parse_geometry)

    df = df.dropna(subset=["zone", "geometry"]).copy()
    df["zone"] = df["zone"].astype(int)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    if not gdf.empty:
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf = gdf[gdf.is_valid].copy()

    return gdf


# =====================================================
# LOAD CLIMAT
# =====================================================

def load_climat(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("""
        SELECT
            cluster,
            temp_moyenne,
            precipitation_total
        FROM climat_clean
        WHERE cluster BETWEEN 1 AND 7
    """).df()

    if df.empty:
        return df

    df["cluster"] = safe_numeric(df["cluster"])
    df["temp_moyenne"] = safe_numeric(df["temp_moyenne"])
    df["precipitation_total"] = safe_numeric(df["precipitation_total"])

    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    return df


# =====================================================
# CREATE MAP
# =====================================================

def create_zone_map(conn: duckdb.DuckDBPyConnection, df_climat: pd.DataFrame, indicator: str) -> folium.Map:
    gdf = load_geo_zone(conn)

    if gdf.empty:
        return folium.Map(location=[43.5, 3.5], zoom_start=7, tiles="cartodbpositron")

    if df_climat.empty or indicator not in df_climat.columns:
        return folium.Map(location=[43.5, 3.5], zoom_start=7, tiles="cartodbpositron")

    # 1. Dissolve pour obtenir 7 zones propres
    gdf_zone = gdf.dissolve(by="zone", as_index=False)

    # 2. Agrégation climat par zone
    df_zone = (
        df_climat.groupby("cluster", as_index=False)[indicator]
        .mean()
    )

    # 3. Merge géométrie + climat
    gdf_final = gdf_zone.merge(
        df_zone,
        left_on="zone",
        right_on="cluster",
        how="left",
    )

    if gdf_final.empty:
        return folium.Map(location=[43.5, 3.5], zoom_start=7, tiles="cartodbpositron")

    # 4. Centre carte
    bounds = gdf_final.total_bounds
    center = [
        (bounds[1] + bounds[3]) / 2,
        (bounds[0] + bounds[2]) / 2,
    ]

    m = folium.Map(
        location=center,
        zoom_start=8,
        tiles="cartodbpositron",
        control_scale=True,
    )

    # 5. Échelle de couleur
    values = gdf_final[indicator].dropna()

    if values.empty:
        min_val, max_val = 0.0, 1.0
    else:
        min_val, max_val = float(values.min()), float(values.max())
        if min_val == max_val:
            max_val = min_val + 1.0

    colormap = linear.YlOrRd_09.scale(min_val, max_val)

    def style_function(feature):
        val = feature["properties"].get(indicator)

        if val is None or pd.isna(val):
            fill_color = "#bdbdbd"
        else:
            fill_color = colormap(val)

        return {
            "fillColor": fill_color,
            "color": "black",
            "weight": 1.2,
            "fillOpacity": 0.8,
        }

    folium.GeoJson(
        gdf_final,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", indicator],
            aliases=["Zone", "Valeur"],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)

    colormap.caption = indicator
    colormap.add_to(m)

    return m


# =====================================================
# MAIN
# =====================================================

def main():
    print("Connexion DB...")
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        print("Chargement climat...")
        df_climat = load_climat(conn)

        if df_climat.empty:
            print("Aucune donnée climat trouvée.")
            return

        print("Création carte...")
        m = create_zone_map(conn, df_climat, "temp_moyenne")

        m.save(str(OUTPUT_HTML))
        print(f"Carte générée : {OUTPUT_HTML.resolve()}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()