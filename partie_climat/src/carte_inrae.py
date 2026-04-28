# -*- coding: utf-8 -*-

from pathlib import Path
import duckdb
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import folium
from branca.colormap import linear, LinearColormap


DB_PATH = Path("db/mvttdb.duckdb")
PAYS_DOC_DEPS = {"11", "30", "34", "66"}


def _normalize_code_commune(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _safe_numeric_columns(df: pd.DataFrame, exclude: set | None = None) -> pd.DataFrame:
    out = df.copy()
    exclude = exclude or set()
    for col in out.columns:
        if col in exclude:
            continue
        if out[col].dtype == object:
            converted = pd.to_numeric(
                out[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if converted.notna().sum() > 0:
                out[col] = converted
    return out


def _get_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_clean_data_insee_oc(file_path_insee: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path_insee)
        df.columns = [str(c).strip() for c in df.columns]

        region_col = _get_column(df, "region_name", "nom_region", "region")
        insee_col = _get_column(df, "insee_code", "commune_code", "code_commune")
        city_col = _get_column(df, "city_code", "city", "nom_commune", "name")

        if insee_col is None:
            return pd.DataFrame()

        df[insee_col] = _normalize_code_commune(df[insee_col])

        if region_col is not None:
            df = df[df[region_col].astype(str).str.lower().eq("occitanie")].copy()

        df["code_dep"] = df[insee_col].str[:2]
        df = df[df["code_dep"].isin(PAYS_DOC_DEPS)].copy()

        if city_col is not None and city_col != "city_code":
            df["city_code"] = df[city_col]

        if insee_col != "insee_code":
            df["insee_code"] = df[insee_col]

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Erreur chargement INSEE : {e}")
        return pd.DataFrame()


def get_commune_name_by_insee(df_communes: pd.DataFrame, code_insee: str):
    if df_communes.empty or "insee_code" not in df_communes.columns:
        return None

    code_insee = str(code_insee).strip()
    matched = df_communes[df_communes["insee_code"].astype(str) == code_insee]
    if matched.empty:
        return None

    if "city_code" in matched.columns:
        return matched.iloc[0]["city_code"]
    if "nom_commune" in matched.columns:
        return matched.iloc[0]["nom_commune"]
    return None


def load_data_inrae(file_path_inrae: str | None = None) -> pd.DataFrame:
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

        if "geo_zone" in tables:
            df = conn.execute(
                """
                SELECT DISTINCT
                    CAST(zone AS INTEGER) AS cluster,
                    CAST(code_commune AS VARCHAR) AS Municipality,
                    nom_commune
                FROM geo_zone
                WHERE zone BETWEEN 1 AND 7
                """
            ).df()
        elif "climat_final" in tables:
            df = conn.execute(
                """
                SELECT DISTINCT
                    CAST(zone AS INTEGER) AS cluster,
                    CAST(commune_code AS VARCHAR) AS Municipality,
                    nom_commune
                FROM climat_final
                WHERE zone BETWEEN 1 AND 7
                """
            ).df()
        else:
            conn.close()
            return pd.DataFrame()

        conn.close()
        df["Municipality"] = _normalize_code_commune(df["Municipality"])
        return df.sort_values(["cluster", "Municipality"]).reset_index(drop=True)

    except Exception as e:
        print(f"Erreur chargement INRAE : {e}")
        return pd.DataFrame()


def _cached_communes_gdf() -> gpd.GeoDataFrame:
    url = "https://raw.githubusercontent.com/Juralexx/france-geojson-datas/master/communes.geojson"
    gdf = gpd.read_file(url)

    gdf["Municipality"] = _normalize_code_commune(gdf["code"])
    gdf = gdf[gdf["Municipality"].str[:2].isin(PAYS_DOC_DEPS)].copy()

    if "nom" not in gdf.columns:
        gdf["nom"] = gdf["Municipality"]

    gdf["geometry"] = gdf["geometry"].simplify(0.01, preserve_topology=True)
    return gdf.reset_index(drop=True)


def _cached_departements_gdf() -> gpd.GeoDataFrame:
    url = "https://raw.githubusercontent.com/Juralexx/france-geojson-datas/master/departements.geojson"
    gdf = gpd.read_file(url)
    gdf["code"] = gdf["code"].astype(str)
    gdf = gdf[gdf["code"].isin(PAYS_DOC_DEPS)].copy()
    gdf["geometry"] = gdf["geometry"].simplify(0.01, preserve_topology=True)
    return gdf.reset_index(drop=True)


def build_cluster_geometries(df_clusters: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return df_clusters.dissolve(by="cluster", as_index=False)


def display_data_map(
    df: pd.DataFrame,
    highlight_municipality=None,
    zones_visibles=None,
):
    try:
        if df.empty:
            return None

        communes_gdf = _cached_communes_gdf()
        deps_gdf = _cached_departements_gdf()

        work = df.copy()
        muni_col = _get_column(work, "Municipality", "commune_code", "code_commune")
        cluster_col = _get_column(work, "cluster", "zone")

        if muni_col is None or cluster_col is None:
            raise ValueError("Colonnes Municipality/cluster introuvables")

        work["Municipality"] = _normalize_code_commune(work[muni_col])
        work["cluster"] = pd.to_numeric(work[cluster_col], errors="coerce").astype("Int64")
        work = work.dropna(subset=["Municipality", "cluster"]).copy()
        work["cluster"] = work["cluster"].astype(int)

        merged = communes_gdf.merge(
            work[["Municipality", "cluster"]].drop_duplicates(),
            on="Municipality",
            how="inner",
        )

        color_dict = {
            1: "#000000",
            2: "red",
            3: "green",
            4: "#00008B",
            5: "#ADD8E6",
            6: "purple",
            7: "yellow",
        }

        merged = merged[merged["cluster"].isin(color_dict.keys())].copy()

        if zones_visibles is not None:
            merged = merged[merged["cluster"].isin(zones_visibles)].copy()

        merged["color"] = merged["cluster"].map(color_dict)

        fig, ax = plt.subplots(1, 1, figsize=(19, 10))
        merged.plot(color=merged["color"], ax=ax, edgecolor="black", linewidth=0.4)
        deps_gdf.boundary.plot(ax=ax, color="black", linewidth=2.5)

        legend_labels = {
            1: "Zone 1: zone humide de l'arrière-pays",
            2: "Zone 2: zone de montagne avec des sols acides et peu profonds",
            3: "Zone 3: zone de piémont avec une réserve utile limitante",
            4: "Zone 4: zone froide et sèche autour du Pic Saint-Loup",
            5: "Zone 5: zone de sols de qualité moyenne dans l’arrière-pays",
            6: "Zone 6: zone de sols profonds sur côtes tempérées",
            7: "Zone 7: zone avec le plus grand nombre de jours très chauds mais sols profonds",
        }

        visible_keys = sorted(set(merged["cluster"].unique()) & set(legend_labels.keys()))
        legend_patches = [
            mpatches.Patch(color=color_dict[k], label=legend_labels[k])
            for k in visible_keys
        ]

        legend = ax.legend(
            handles=legend_patches,
            title="Zones pédoclimatiques",
            title_fontsize=14,
            fontsize=12,
            loc="lower right",
            bbox_to_anchor=(1.38, 0),
            borderaxespad=0.5,
            frameon=True,
        )
        legend.get_frame().set_facecolor("#f1c01f")
        legend.get_frame().set_alpha(1)
        legend.get_frame().set_edgecolor("black")

        if highlight_municipality is not None:
            highlight_code = str(highlight_municipality).strip()
            highlight_gdf = merged[merged["Municipality"] == highlight_code].copy()

            if not highlight_gdf.empty:
                highlight_proj = highlight_gdf.to_crs(epsg=2154)
                merged_proj = merged.to_crs(epsg=2154)

                centroid = highlight_proj.geometry.centroid.iloc[0]
                cluster_val = int(highlight_gdf["cluster"].iloc[0])

                x = centroid.x
                y = centroid.y

                ax.scatter(
                    x,
                    y,
                    color="orange",
                    s=400,
                    edgecolor="black",
                    marker="P",
                    zorder=10,
                )

                median_x = merged_proj.geometry.centroid.x.median()
                if x < median_x:
                    offset_x = -38
                    ha = "right"
                else:
                    offset_x = 38
                    ha = "left"

                ax.annotate(
                    f"Zone {cluster_val}",
                    xy=(x, y),
                    xytext=(offset_x, 6),
                    textcoords="offset points",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    ha=ha,
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f1c01f", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black", alpha=0.9),
                    zorder=11,
                )

                commune_leg = ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            marker="P",
                            color="orange",
                            markersize=14,
                            markeredgecolor="black",
                            linestyle="None",
                            label="Commune sélectionnée",
                        )
                    ],
                    loc="lower right",
                    bbox_to_anchor=(1.38, 0.30),
                    fontsize=12,
                )
                commune_leg.get_frame().set_facecolor("#f1c01f")
                commune_leg.get_frame().set_edgecolor("black")
                commune_leg.get_frame().set_alpha(1)
                ax.add_artist(legend)

        ax.set_title("Carte des zones pédoclimatiques", fontsize=16)
        ax.axis("off")
        fig.patch.set_facecolor("#f1c01f")
        ax.set_facecolor("#f1c01f")
        plt.figtext(0.85, 0.05, "Source: INRAE / DuckDB", ha="right", fontsize=10)

        return fig

    except Exception as e:
        print(f"Erreur affichage carte : {e}")
        return None


def create_data_map(df, indicator, year=None, highlight_municipality=None):
    df = df.copy()

    year_col = _get_column(df, "year", "Year")
    code_col = _get_column(df, "commune_code", "Municipality", "code_commune")

    if code_col is None:
        raise ValueError("Colonne commune introuvable")

    df["commune_code"] = _normalize_code_commune(df[code_col])

    if year is not None and year_col is not None:
        df = df[pd.to_numeric(df[year_col], errors="coerce") == int(year)].copy()

    if indicator not in df.columns:
        raise ValueError(f"Indicateur introuvable: {indicator}")

    df[indicator] = _to_numeric_series(df[indicator])

    df_avg = (
        df.groupby("commune_code", as_index=False)[indicator]
        .mean()
    )

    communes_gdf = _cached_communes_gdf()
    gdf = communes_gdf.merge(
        df_avg,
        left_on="Municipality",
        right_on="commune_code",
        how="left",
    )

    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=8, control_scale=True)

    values = gdf[indicator].dropna()
    if values.empty:
        return m

    colormap = linear.YlOrRd_09.scale(values.min(), values.max())

    def style(feature):
        val = feature["properties"].get(indicator)
        return {
            "fillColor": colormap(val) if val is not None and pd.notna(val) else "#cccccc",
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.8,
        }

    tooltip_fields = ["nom", indicator] if "nom" in gdf.columns else ["Municipality", indicator]
    tooltip_aliases = ["Commune", indicator]

    folium.GeoJson(
        gdf,
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
        ),
    ).add_to(m)

    if highlight_municipality is not None:
        highlight_code = str(highlight_municipality).strip()
        highlight = gdf[gdf["Municipality"] == highlight_code]
        for _, row in highlight.iterrows():
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                popup=f"Commune: {row.get('nom', row['Municipality'])}",
            ).add_to(m)

    colormap.caption = f"{indicator} ({year})" if year is not None else indicator
    colormap.add_to(m)

    return m


def create_cluster_indicator_map(
    df,
    indicator,
    year=None,
    highlight_mu=None,
    communes_gdf=None,
):
    work = df.copy()

    year_col = _get_column(work, "year", "Year")
    zone_col = _get_column(work, "zone", "cluster")
    code_col = _get_column(work, "commune_code", "Municipality", "code_commune")

    if zone_col is None or code_col is None:
        raise ValueError("Colonnes zone/commune introuvables")

    work["zone"] = pd.to_numeric(work[zone_col], errors="coerce")
    work["commune_code"] = _normalize_code_commune(work[code_col])

    if year is not None and year_col is not None:
        work = work[pd.to_numeric(work[year_col], errors="coerce") == int(year)].copy()

    if indicator not in work.columns:
        raise ValueError(f"Indicateur introuvable: {indicator}")

    work[indicator] = _to_numeric_series(work[indicator])
    work = work.dropna(subset=["zone"]).copy()
    work["zone"] = work["zone"].astype(int)

    if communes_gdf is None:
        communes_gdf = _cached_communes_gdf()

    df_zone_map = work[["commune_code", "zone"]].drop_duplicates()
    gdf = communes_gdf.merge(
        df_zone_map,
        left_on="Municipality",
        right_on="commune_code",
        how="left",
    )
    gdf = gdf.dropna(subset=["zone"]).copy()
    gdf["zone"] = gdf["zone"].astype(int)

    gdf_clusters = gdf.dissolve(by="zone", as_index=False)

    df_zone_values = (
        work.groupby("zone", as_index=False)[indicator]
        .mean()
    )

    gdf_clusters = gdf_clusters.merge(df_zone_values, on="zone", how="left")
    gdf_clusters["cluster_fmt"] = gdf_clusters["zone"].apply(lambda x: f"Zone {int(x)}")
    gdf_clusters["valeur_indicateur_fmt"] = gdf_clusters[indicator].apply(
        lambda x: "Non renseigné" if pd.isna(x) else f"{x:.1f}"
    )

    bounds = gdf_clusters.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(
        location=center,
        zoom_start=7,
        min_zoom=6,
        max_zoom=12,
        zoomSnap=0.1,
        control_scale=True,
    )

    INDICATOR_BOUNDS = {
        "Climatic_Dryness_Index": (-300, -100),
        "Hot_D": (10, 70),
        "Very_Hot_D": (2, 18),
        "jours_pluie": (70, 140),
        "precipitation_total": (400, 1200),
        "deficit_hydrique": (180, 280),
        "stress_climatique": (20, 150),
        "temp_moyenne": (13, 18),
        "Late_Frost": (0, 2),
        "tmax_mean": (17, 22),
        "tmin_mean": (9, 13),
        "Huglin_Index": (1800, 2800),
    }

    min_val, max_val = INDICATOR_BOUNDS.get(
        indicator,
        (
            float(gdf_clusters[indicator].min()) if gdf_clusters[indicator].notna().any() else 0.0,
            float(gdf_clusters[indicator].max()) if gdf_clusters[indicator].notna().any() else 1.0,
        ),
    )

    if indicator == "Climatic_Dryness_Index":
        colormap = LinearColormap(colors=["#ff0000", "#ffffff"], vmin=-300, vmax=-100).to_step(
            index=[-300, -250, -200, -150, -100]
        )
    elif indicator == "precipitation_total":
        colormap = LinearColormap(
            colors=["red", "#f75a2a", "#e46e0f", "yellow", "#b8ceeb", "#749dd3", "#022f69"],
            vmin=400,
            vmax=1200,
        ).to_step(index=[400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
    elif indicator == "jours_pluie":
        colormap = LinearColormap(
            colors=["red", "#f75a2a", "#e46e0f", "yellow", "#b8ceeb", "#749dd3", "#022f69"],
            vmin=70,
            vmax=140,
        ).to_step(index=[70, 80, 90, 100, 110, 120, 130, 140])
    elif indicator == "deficit_hydrique":
        colormap = LinearColormap(
            colors=["red", "#f75a2a", "#e46e0f", "yellow", "#b8ceeb", "#749dd3", "#022f69"],
            vmin=180,
            vmax=280,
        ).to_step(index=[180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
    elif indicator == "Late_Frost":
        colormap = LinearColormap(colors=["#ccdff8", "#022f69"], vmin=0, vmax=2).to_step(
            index=[0, 1, 2]
        )
    elif indicator == "Very_Hot_D":
        colormap = LinearColormap(colors=["white", "orange", "red"], vmin=2, vmax=18).to_step(
            index=[2, 4, 6, 8, 10, 12, 14, 16, 18]
        )
    else:
        colormap = linear.YlOrRd_09.scale(min_val, max_val)

    def scaled_color(val):
        if val is None or pd.isna(val):
            return "grey"
        val = float(val)
        val = max(min_val, min(max_val, val))
        return colormap(val)

    def style_zone(feature):
        val = feature["properties"].get(indicator)
        zone = feature["properties"].get("zone")
        if zone is None or pd.isna(zone):
            return {
                "fillColor": "grey",
                "color": "grey",
                "weight": 0.3,
                "fillOpacity": 0.7,
            }
        return {
            "fillColor": scaled_color(val),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.8,
        }

    translation_dict = {
        "Huglin_Index": "Indice Huglin (°C-jours)",
        "temp_moyenne": "Température moyenne (°C)",
        "tmax_mean": "Température maximale moyenne (°C)",
        "tmin_mean": "Température minimale moyenne (°C)",
        "Climatic_Dryness_Index": "Indice de sécheresse climatique (mm)",
        "jours_pluie": "Nombre de jours de pluie",
        "Hot_D": "Jours de chaleur",
        "Very_Hot_D": "Jours de forte chaleur",
        "Late_Frost": "Jours de gel tardif",
        "precipitation_total": "Précipitations totales (mm)",
        "deficit_hydrique": "Déficit hydrique (mm)",
        "stress_climatique": "Stress climatique",
    }

    indicator_fr = translation_dict.get(indicator, indicator)

    folium.GeoJson(
        gdf_clusters,
        style_function=style_zone,
        tooltip=folium.GeoJsonTooltip(
            fields=["cluster_fmt", "valeur_indicateur_fmt"],
            aliases=["Zone :", f"{indicator_fr} :"],
            sticky=True,
            localize=False,
        ),
    ).add_to(m)

    departements_borders = _cached_departements_gdf()
    folium.GeoJson(
        departements_borders,
        style_function=lambda feature: {
            "fillColor": "transparent",
            "color": "white",
            "weight": 3,
        },
    ).add_to(m)

    if highlight_mu is not None:
        highlight_code = str(highlight_mu).strip()
        highlight = communes_gdf[communes_gdf["Municipality"] == highlight_code]
        if not highlight.empty:
            centroid = highlight.geometry.centroid.iloc[0]
            folium.Marker(
                location=[centroid.y, centroid.x],
                popup=f"Commune {highlight_code}",
            ).add_to(m)

    if hasattr(colormap, "index"):
        if indicator in {"temp_moyenne", "tmax_mean", "tmin_mean", "Huglin_Index"}:
            colormap.tick_labels = [f"{v:.1f}" for v in colormap.index]
        else:
            colormap.tick_labels = [f"{int(round(v))}" for v in colormap.index]

    colormap.caption = f"{indicator_fr} ({year})" if year is not None else indicator_fr
    colormap.add_to(m)

    return m


def create_cluster_indicator_map_cycle_wine(
    df,
    indicator,
    year=None,
    highlight_mu=None,
    communes_gdf=None,
):
    return create_cluster_indicator_map(
        df=df,
        indicator=indicator,
        year=year,
        highlight_mu=highlight_mu,
        communes_gdf=communes_gdf,
    )