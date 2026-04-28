# -*- coding: utf-8 -*-

from pathlib import Path
import gc

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import geopandas as gpd
from branca.colormap import linear, LinearColormap


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "db" / "mvttdb.duckdb"

st.set_page_config(layout="wide")
st.title("Modélisation climatique - Pays d'Oc")


ZONE_COLOR_MAP = {
    "0": "#BDBDBD",
    "1": "#000000",
    "2": "#FF0000",
    "3": "#1A8F2A",
    "4": "#0033CC",
    "5": "#AFC6D9",
    "6": "#7A1FA2",
    "7": "#FFD800",
}

ZONE_LABELS = {
    "0": "Zone 0 : non classée / hors zonage",
    "1": "Zone 1 : zone humide de l'arrière-pays",
    "2": "Zone 2 : montagne, sols acides et peu profonds",
    "3": "Zone 3 : Piémont, réserve utile limitée",
    "4": "Zone 4 : froide et sèche autour du Pic Saint-Loup",
    "5": "Zone 5 : sols de qualité moyenne dans l'arrière-pays",
    "6": "Zone 6 : sols profonds, côtes tempérées",
    "7": "Zone 7 : très chaud, sols profonds",
}

SCENARIO_COLOR_MAP = {
    "optimiste": "#AFC6D9",
    "neutre": "#C99700",
    "pessimiste": "#C00000",
}

MAP_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
]

SCENARIO_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
]

HISTORICAL_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
    "Huglin_Index",
    "Hot_D",
    "Very_Hot_D",
    "Frost_D",
    "Late_Frost",
    "stress_climatique",
    "deficit_hydrique",
    "Climatic_Dryness_Index",
    "Soil_Water_Stock",
    "Soil_pH",
    "jours_secs",
    "jours_pluie",
]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def indicator_label(indicator: str) -> str:
    labels = {
        "temp_moyenne": "Température moyenne (°C)",
        "tmax_mean": "Température maximale moyenne (°C)",
        "tmin_mean": "Température minimale moyenne (°C)",
        "precipitation_total": "Précipitations totales (mm)",
        "precipitation_total_avril_septembre": "Précipitations avril-septembre (mm)",
        "Huglin_Index": "Indice de Huglin",
        "Hot_D": "Nombre de jours de chaleur",
        "Very_Hot_D": "Nombre de jours de forte chaleur",
        "Frost_D": "Nombre de jours de gel",
        "Late_Frost": "Nombre de jours de gel tardif",
        "stress_climatique": "Stress climatique",
        "deficit_hydrique": "Déficit hydrique",
        "Climatic_Dryness_Index": "Indice de sécheresse climatique",
        "Soil_Water_Stock": "Réserve utile en eau du sol",
        "Soil_pH": "pH du sol",
        "jours_secs": "Jours secs",
        "jours_pluie": "Jours de pluie",
    }
    return labels.get(indicator, indicator)


def format_value(value, indicator: str) -> str:
    if pd.isna(value):
        return "NA"

    no_decimal_indicators = {
        "Hot_D",
        "Huglin_Index",
        "Climatic_Dryness_Index",
        "stress_climatique",
    }

    if indicator in no_decimal_indicators:
        return f"{value:.0f}"

    return f"{value:.1f}"


def ensure_hist_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cluster"] = safe_numeric(df["cluster"]).astype("Int64")
    df["Year"] = safe_numeric(df["Year"]).astype("Int64")
    df["Municipality"] = df["Municipality"].astype(str)

    numeric_cols = [
        "code_departement",
        "latitude",
        "longitude",
        "temp_moyenne",
        "tmax_mean",
        "tmin_mean",
        "precipitation_total",
        "precipitation_total_avril_septembre",
        "amplitude_thermique",
        "Huglin_Index",
        "Hot_D",
        "Very_Hot_D",
        "Frost_D",
        "Late_Frost",
        "stress_climatique",
        "deficit_hydrique",
        "Climatic_Dryness_Index",
        "Soil_Water_Stock",
        "Soil_pH",
        "jours_secs",
        "jours_pluie",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    return df


def ensure_proj_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["zone"] = safe_numeric(df["zone"]).astype("Int64")
    df["year"] = safe_numeric(df["year"]).astype("Int64")
    df["commune_code"] = df["commune_code"].astype(str)

    if "code_departement" in df.columns:
        df["code_departement"] = safe_numeric(df["code_departement"]).astype("Int64")

    for col in [
        "latitude",
        "longitude",
        "temp_moyenne",
        "tmax_mean",
        "tmin_mean",
        "precipitation_total",
        "precipitation_total_avril_septembre",
        "amplitude_thermique",
    ]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    return df


@st.cache_data
def load_climat() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = conn.execute("SELECT * FROM climat_final").df()
    finally:
        conn.close()
    return ensure_hist_schema(df)


@st.cache_data
def load_projection() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = conn.execute("SELECT * FROM climat_projection_final").df()
    finally:
        conn.close()
    return ensure_proj_schema(df)


@st.cache_data
def load_geo_zone() -> pd.DataFrame:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = conn.execute("SELECT * FROM geo_zone").df()
    finally:
        conn.close()

    df["zone"] = safe_numeric(df["zone"]).astype("Int64")
    df["commune_code"] = df["commune_code"].astype(str)
    return df


@st.cache_data
def load_communes_geojson() -> gpd.GeoDataFrame:
    url = "https://raw.githubusercontent.com/Juralexx/france-geojson-datas/master/communes.geojson"
    gdf = gpd.read_file(url)
    gdf["commune_code"] = gdf["code"].astype(str)
    gdf = gdf[gdf["commune_code"].str[:2].isin(["11", "30", "34", "66"])].copy()
    return gdf


df_climat = load_climat()
df_proj = load_projection()
df_geo_zone = load_geo_zone()


def get_zone_legend_patches():
    patches = []
    for zone, label in ZONE_LABELS.items():
        if zone == "0":
            continue
        patches.append(
            mpatches.Patch(
                color=ZONE_COLOR_MAP[zone],
                label=label,
            )
        )
    return patches


def build_indicator_colormap(values: pd.Series, indicator: str):
    values = values.dropna()
    if values.empty:
        return None

    min_val = float(values.min())
    max_val = float(values.max())

    if indicator == "Climatic_Dryness_Index":
        colormap = LinearColormap(colors=["#ff0000", "#ffffff"], vmin=-300, vmax=-100)
        colormap = colormap.to_step(index=[-300, -250, -200, -150, -100])

    elif indicator == "precipitation_total":
        colormap = LinearColormap(
            colors=["red", "#f75a2a", "#e46e0f", "yellow", "#b8ceeb", "#749dd3", "#022f69"],
            vmin=400,
            vmax=1200,
        )
        colormap = colormap.to_step(index=[400, 500, 600, 700, 800, 900, 1000, 1100, 1200])

    elif indicator == "jours_pluie":
        colormap = LinearColormap(
            colors=["red", "#f75a2a", "#e46e0f", "yellow", "#b8ceeb", "#749dd3", "#022f69"],
            vmin=70,
            vmax=140,
        )
        colormap = colormap.to_step(index=[70, 80, 90, 100, 110, 120, 130, 140])

    elif indicator == "deficit_hydrique":
        colormap = LinearColormap(
            colors=["red", "#f75a2a", "#e46e0f", "yellow", "#b8ceeb", "#749dd3", "#022f69"],
            vmin=180,
            vmax=280,
        )
        colormap = colormap.to_step(index=[180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])

    elif indicator == "Late_Frost":
        colormap = LinearColormap(colors=["#ccdff8", "#022f69"], vmin=0, vmax=2)
        colormap = colormap.to_step(index=[0, 1, 2])

    elif indicator == "Very_Hot_D":
        colormap = LinearColormap(colors=["white", "orange", "red"], vmin=2, vmax=18)
        colormap = colormap.to_step(index=[2, 4, 6, 8, 10, 12, 14, 16, 18])

    else:
        if min_val == max_val:
            max_val = min_val + 1.0
        colormap = linear.YlOrRd_09.scale(min_val, max_val)

    colormap.caption = indicator_label(indicator)
    return colormap


def create_zone_map(df_zone_values: pd.DataFrame, indicator: str, title_label: str):
    communes_gdf = load_communes_geojson()
    geo_zone = df_geo_zone[["commune_code", "zone"]].drop_duplicates().copy()

    gdf = communes_gdf.merge(geo_zone, on="commune_code", how="left")
    gdf["zone"] = safe_numeric(gdf["zone"]).astype("Int64")
    gdf = gdf.dropna(subset=["zone"]).copy()

    if gdf.empty:
        return folium.Map(location=[43.7, 3.5], zoom_start=8)

    zone_geom = gdf[["zone", "geometry"]].dissolve(by="zone", as_index=False)

    values_df = df_zone_values[["zone", indicator]].copy()
    values_df["zone"] = safe_numeric(values_df["zone"]).astype("Int64")
    values_df[indicator] = safe_numeric(values_df[indicator])

    zones_gdf = zone_geom.merge(values_df, on="zone", how="left")
    zones_gdf["zone_str"] = zones_gdf["zone"].astype("Int64").astype(str)
    zones_gdf["zone_label"] = zones_gdf["zone_str"].map(ZONE_LABELS)
    zones_gdf["indicator_fmt"] = zones_gdf[indicator].apply(lambda x: format_value(x, indicator))

    bounds = zones_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    colormap = build_indicator_colormap(zones_gdf[indicator], indicator)
    if colormap is None:
        return m

    def style_function(feature):
        val = feature["properties"].get(indicator)
        zone_str = str(feature["properties"].get("zone"))
        border_color = ZONE_COLOR_MAP.get(zone_str, "#BDBDBD")

        return {
            "fillColor": "#D9D9D9" if val is None else colormap(val),
            "color": border_color,
            "weight": 3,
            "fillOpacity": 0.8,
        }

    folium.GeoJson(
        zones_gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone_label", "indicator_fmt"],
            aliases=["Zone", indicator_label(indicator)],
            localize=False,
            sticky=True,
        ),
    ).add_to(m)

    colormap.add_to(m)

    legend_items = ""
    for zone, label in ZONE_LABELS.items():
        if zone == "0":
            continue
        legend_items += f"""
        <div style="display:flex; align-items:center; margin-bottom:4px;">
            <div style="
                width:12px;
                height:12px;
                background:white;
                border:3px solid {ZONE_COLOR_MAP[zone]};
                margin-right:8px;
            "></div>
            <span>{label}</span>
        </div>
        """

    zone_legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index: 9999;
        background: white;
        border: 2px solid #666;
        border-radius: 6px;
        padding: 10px;
        font-size: 12px;
        width: 320px;
        box-shadow: 0 0 6px rgba(0,0,0,0.25);
    ">
        <b>Zonage pédoclimatique</b>
        <hr style="margin:6px 0;">
        {legend_items}
    </div>
    """

    title_html = f"""
    <div style="
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: white;
        border: 2px solid #666;
        border-radius: 6px;
        padding: 8px 14px;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 0 0 6px rgba(0,0,0,0.25);
    ">
        {title_label}
    </div>
    """

    m.get_root().html.add_child(folium.Element(zone_legend_html))
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def plot_historical_curves(df: pd.DataFrame, selected_zones: list[int], indicator: str):
    fig, ax = plt.subplots(figsize=(11, 5))

    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if df_zone.empty or indicator not in df_zone.columns:
            continue

        grouped = (
            df_zone.groupby("Year", dropna=True)[indicator]
            .mean()
            .reset_index()
            .dropna()
            .sort_values("Year")
        )

        if grouped.empty:
            continue

        grouped["Year"] = grouped["Year"].astype(int)
        zone_str = str(int(zone))

        ax.plot(
            grouped["Year"],
            grouped[indicator],
            marker="o",
            linewidth=2,
            color=ZONE_COLOR_MAP.get(zone_str, "#333333"),
            label=ZONE_LABELS.get(zone_str, f"Zone {zone_str}"),
        )

    ax.set_title(f"Historique - {indicator_label(indicator)}")
    ax.set_xlabel("Année")
    ax.set_ylabel(indicator_label(indicator))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def build_no_scenario_projection(df_hist: pd.DataFrame, zones: list[int], indicators: list[str]) -> pd.DataFrame:
    rows = []

    for zone in zones:
        df_zone = df_hist[df_hist["cluster"] == zone].copy()
        if df_zone.empty:
            continue

        for indicator in indicators:
            if indicator not in df_zone.columns:
                continue

            grouped = (
                df_zone.groupby("Year", dropna=True)[indicator]
                .mean()
                .reset_index()
                .dropna()
                .sort_values("Year")
            )

            if grouped.empty:
                continue

            grouped["Year"] = grouped["Year"].astype(int)

            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "cluster": int(zone),
                        "Year": int(row["Year"]),
                        "indicator": indicator,
                        "value": float(row[indicator]),
                        "type": "historique",
                    }
                )

            if len(grouped) >= 2:
                x = grouped["Year"].to_numpy(dtype=float)
                y = grouped[indicator].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, 1)

                for year in range(int(grouped["Year"].max()) + 1, 2041):
                    value = slope * year + intercept
                    rows.append(
                        {
                            "cluster": int(zone),
                            "Year": int(year),
                            "indicator": indicator,
                            "value": float(value),
                            "type": "projection_sans_scenario",
                        }
                    )

    return pd.DataFrame(rows)


def plot_no_scenario_curves(df_no_scenario: pd.DataFrame, selected_zones: list[int], indicator: str):
    fig, ax = plt.subplots(figsize=(11, 5))

    for zone in selected_zones:
        df_zone = df_no_scenario[
            (df_no_scenario["cluster"] == zone) & (df_no_scenario["indicator"] == indicator)
        ].copy()

        if df_zone.empty:
            continue

        df_hist = df_zone[df_zone["type"] == "historique"].sort_values("Year")
        df_proj = df_zone[df_zone["type"] == "projection_sans_scenario"].sort_values("Year")

        zone_str = str(int(zone))
        color = ZONE_COLOR_MAP.get(zone_str, "#333333")
        label = ZONE_LABELS.get(zone_str, f"Zone {zone_str}")

        if not df_hist.empty:
            ax.plot(
                df_hist["Year"].astype(int),
                df_hist["value"],
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )

        if not df_proj.empty:
            ax.plot(
                df_proj["Year"].astype(int),
                df_proj["value"],
                marker="x",
                linestyle="--",
                linewidth=2,
                color=color,
            )

    ax.set_title(f"Projection sans scénario jusqu'en 2040 - {indicator_label(indicator)}")
    ax.set_xlabel("Année")
    ax.set_ylabel(indicator_label(indicator))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def build_scenario_table(df_proj_period: pd.DataFrame, selected_zones: list[int]) -> pd.DataFrame:
    rows = []

    for zone in selected_zones:
        for scenario in ["optimiste", "neutre", "pessimiste"]:
            subset = df_proj_period[
                (df_proj_period["zone"] == zone)
                & (df_proj_period["scenario"] == scenario)
            ].copy()

            rows.append(
                {
                    "zone": int(zone),
                    "scenario": scenario,
                    "temp_moyenne": subset["temp_moyenne"].mean() if "temp_moyenne" in subset.columns else np.nan,
                    "tmax_mean": subset["tmax_mean"].mean() if "tmax_mean" in subset.columns else np.nan,
                    "tmin_mean": subset["tmin_mean"].mean() if "tmin_mean" in subset.columns else np.nan,
                    "precipitation_total": subset["precipitation_total"].mean() if "precipitation_total" in subset.columns else np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_scenario_comparison(df_table: pd.DataFrame, indicator: str, period: str):
    fig, ax = plt.subplots(figsize=(11, 5))

    scenarios = ["optimiste", "neutre", "pessimiste"]
    sorted_zones = sorted(df_table["zone"].unique())
    x = np.arange(len(sorted_zones))
    width = 0.22

    for i, scenario in enumerate(scenarios):
        vals = []
        for zone in sorted_zones:
            subset = df_table[(df_table["zone"] == zone) & (df_table["scenario"] == scenario)]
            vals.append(subset[indicator].iloc[0] if not subset.empty else np.nan)

        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=scenario.capitalize(),
            color=SCENARIO_COLOR_MAP[scenario],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {int(z)}" for z in sorted_zones])
    ax.set_title(f"Comparaison des scénarios - {indicator_label(indicator)} - {period}")
    ax.set_xlabel("Zone")
    ax.set_ylabel(indicator_label(indicator))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    zone_patches = get_zone_legend_patches()
    extra_legend = ax.legend(
        handles=zone_patches,
        title="Zones",
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(1.02, 0.45),
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(extra_legend)

    fig.tight_layout()
    return fig


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Filtres")

available_zones = sorted([int(z) for z in df_climat["cluster"].dropna().unique()])
selected_zones = st.sidebar.multiselect(
    "Zones",
    options=available_zones,
    default=available_zones[:3] if len(available_zones) >= 3 else available_zones,
)

available_years = sorted([int(y) for y in df_climat["Year"].dropna().unique()])
selected_year = st.sidebar.selectbox(
    "Année historique",
    options=available_years,
    index=len(available_years) - 1,
)

use_last_5y_mean = st.sidebar.checkbox("Moyenne des 5 dernières années", value=False)

selected_indicator = st.sidebar.selectbox(
    "Indicateur historique",
    options=HISTORICAL_INDICATORS,
    format_func=indicator_label,
)

if not selected_zones:
    st.warning("Sélectionne au moins une zone.")
    st.stop()


# =====================================================
# HISTORIQUE
# =====================================================

st.header("Analyse historique")

if use_last_5y_mean:
    year_max = int(df_climat["Year"].dropna().max())
    year_min = year_max - 4
    hist_filtered = df_climat[
        (df_climat["cluster"].isin(selected_zones))
        & (df_climat["Year"].between(year_min, year_max))
    ].copy()
    period_label = f"Moyenne {year_min}-{year_max}"
else:
    hist_filtered = df_climat[
        (df_climat["cluster"].isin(selected_zones))
        & (df_climat["Year"] == selected_year)
    ].copy()
    period_label = str(int(selected_year))

col1, col2 = st.columns(2)

with col1:
    metric_val = hist_filtered[selected_indicator].mean() if selected_indicator in hist_filtered.columns else np.nan
    st.metric(
        label=f"{indicator_label(selected_indicator)} - {period_label}",
        value=format_value(metric_val, selected_indicator),
    )

with col2:
    st.write("Zones sélectionnées :", ", ".join(str(int(z)) for z in selected_zones))

st.subheader("Tableau par zone")

if use_last_5y_mean:
    table_zone = (
        df_climat[df_climat["Year"].between(year_min, year_max)]
        .groupby("cluster", as_index=False)[selected_indicator]
        .mean()
        .rename(columns={"cluster": "zone"})
        .sort_values("zone")
    )
else:
    table_zone = (
        df_climat[df_climat["Year"] == selected_year]
        .groupby("cluster", as_index=False)[selected_indicator]
        .mean()
        .rename(columns={"cluster": "zone"})
        .sort_values("zone")
    )

table_zone_display = table_zone.copy()
table_zone_display["zone"] = table_zone_display["zone"].apply(lambda z: f"Zone {int(z)}")
table_zone_display[selected_indicator] = table_zone_display[selected_indicator].apply(
    lambda x: format_value(x, selected_indicator)
)
st.dataframe(table_zone_display, use_container_width=True)

st.subheader("Carte historique par zone")

map_indicator_hist = st.selectbox(
    "Indicateur cartographié (historique)",
    options=MAP_INDICATORS,
    format_func=indicator_label,
    key="map_indicator_hist",
)

try:
    if use_last_5y_mean:
        zone_values_hist = (
            df_climat[df_climat["Year"].between(year_min, year_max)]
            .groupby("cluster", as_index=False)[map_indicator_hist]
            .mean()
            .rename(columns={"cluster": "zone"})
        )
        map_title = f"Carte historique - {indicator_label(map_indicator_hist)} - moyenne {year_min}-{year_max}"
    else:
        zone_values_hist = (
            df_climat[df_climat["Year"] == selected_year]
            .groupby("cluster", as_index=False)[map_indicator_hist]
            .mean()
            .rename(columns={"cluster": "zone"})
        )
        map_title = f"Carte historique - {indicator_label(map_indicator_hist)} - {int(selected_year)}"

    map_hist = create_zone_map(zone_values_hist, map_indicator_hist, map_title)
    components.html(map_hist._repr_html_(), height=650)
except Exception as e:
    st.error(f"Erreur carte : {e}")

st.subheader("Courbes historiques")
try:
    fig_hist = plot_historical_curves(df_climat, selected_zones, selected_indicator)
    st.pyplot(fig_hist)
    plt.close(fig_hist)
except Exception as e:
    st.error(f"Erreur graphique historique : {e}")


# =====================================================
# SANS SCÉNARIO
# =====================================================

st.header("Projection sans scénario jusqu'en 2040")

try:
    df_no_scenario = build_no_scenario_projection(
        df_hist=df_climat,
        zones=selected_zones,
        indicators=HISTORICAL_INDICATORS,
    )

    no_scenario_indicator = st.selectbox(
        "Indicateur sans scénario",
        options=HISTORICAL_INDICATORS,
        format_func=indicator_label,
        key="no_scenario_indicator",
    )

    fig_no_scenario = plot_no_scenario_curves(
        df_no_scenario=df_no_scenario,
        selected_zones=selected_zones,
        indicator=no_scenario_indicator,
    )
    st.pyplot(fig_no_scenario)
    plt.close(fig_no_scenario)

except Exception as e:
    st.error(f"Erreur projection sans scénario : {e}")


# =====================================================
# SCÉNARIOS
# =====================================================

st.header("Comparaison des scénarios climatiques")

scenario_period = st.selectbox(
    "Période de projection",
    options=["2021-2040", "2041-2060"],
)

proj_period = df_proj[df_proj["periode"] == scenario_period].copy()

if proj_period.empty:
    st.warning("Aucune donnée de projection disponible pour cette période.")
else:
    scenario_table = build_scenario_table(proj_period, selected_zones)

    st.subheader("Tableau comparatif par zone")
    scenario_table_display = scenario_table.copy()
    scenario_table_display["zone"] = scenario_table_display["zone"].apply(lambda z: f"Zone {int(z)}")

    for col in SCENARIO_INDICATORS:
        scenario_table_display[col] = scenario_table_display[col].apply(lambda x: format_value(x, col))

    st.dataframe(scenario_table_display, use_container_width=True)

    compare_indicator = st.selectbox(
        "Indicateur comparé",
        options=SCENARIO_INDICATORS,
        format_func=indicator_label,
        key="compare_indicator",
    )

    fig_scenario = plot_scenario_comparison(
        df_table=scenario_table,
        indicator=compare_indicator,
        period=scenario_period,
    )
    st.pyplot(fig_scenario)
    plt.close(fig_scenario)

    st.subheader("Carte projetée par zone")

    map_scenario = st.selectbox(
        "Scénario cartographié",
        options=["optimiste", "neutre", "pessimiste"],
        key="map_scenario",
    )

    map_indicator_proj = st.selectbox(
        "Indicateur cartographié (projection)",
        options=MAP_INDICATORS,
        format_func=indicator_label,
        key="map_indicator_proj",
    )

    try:
        values_proj_zone = (
            proj_period[proj_period["scenario"] == map_scenario]
            .groupby("zone", as_index=False)[map_indicator_proj]
            .mean()
        )

        map_proj = create_zone_map(
            values_proj_zone,
            map_indicator_proj,
            f"Carte projetée - {indicator_label(map_indicator_proj)} - {map_scenario} - {scenario_period}",
        )
        components.html(map_proj._repr_html_(), height=650)
    except Exception as e:
        st.error(f"Erreur carte projetée : {e}")


gc.collect()