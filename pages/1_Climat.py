# -*- coding: utf-8 -*-
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import folium
import base64
import geopandas as gpd
from branca.colormap import linear, LinearColormap
from utils.db import get_conn
from config.constants import ZONE_COLOR_MAP, ZONE_LABELS


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "db" / "mvttdb.duckdb"
st.set_page_config(
    page_title="Observatoire Viticole Climat - Pays d'Oc IGP",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ZONE_LABELS_Tick = {
    "0": "0: non classee / hors zonage",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
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
    "Huglin_Index",
    "Hot_D",
    "Very_Hot_D",
    "stress_climatique",
    "deficit_hydrique",
    "Climatic_Dryness_Index",
    "jours_pluie",
]

SCENARIO_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
]

WINE_CYCLE_SCENARIO_INDICATORS = [
    "precipitation_total_avril_septembre"
]
HISTORICAL_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
    "Huglin_Index",
    "Hot_D",
    "Very_Hot_D",
    "stress_climatique",
    "deficit_hydrique",
    "Climatic_Dryness_Index",
    "Soil_Water_Stock",
    "Soil_pH",
    "jours_pluie",
]
# Style CSS personnalise
st.markdown("""
<style> 
    .main-header {
        background: linear-gradient(135deg, #ff7900 0%, #ff7900 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #ff7900;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.85rem;
    }
    hr {
        margin: 1.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 20px 20px 0 0;
        padding: 10px 20px;
        background-color: #ff7900;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Global */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #ff7900 0%, #ff7900 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        width: 280px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #2c3e50;
    }
    
    .nav-card h3 {
        color: #2c3e50;
        margin: 1rem 0 0.5rem 0;
        font-size: 1.3rem;
    }
    
    .nav-card p {
        color: #6c757d;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    
    .nav-icon {
        font-size: 2.5rem;
    }
    
    /* Boutons */
    .nav-btn {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .nav-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(44,62,80,0.3);
    }
    
    /* Popup overlay */
    .popup-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        z-index: 999;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .popup-content {
        background: white;
        border-radius: 15px;
        max-width: 90%;
        max-height: 90%;
        overflow-y: auto;
        padding: 2rem;
        position: relative;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .popup-close {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #dc3545;
        color: white;
        border: none;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .popup-close:hover {
        background: #c82333;
        transform: scale(1.1);
    }
    
    /* Legend items */
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 10px;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #2c3e50;
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.85rem;
    }
    
    hr {
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.85rem;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state pour les popups
if "show_zone_map" not in st.session_state:
    st.session_state.show_zone_map = False
if "show_dept_map" not in st.session_state:
    st.session_state.show_dept_map = False


def show_zone_popup():
    st.session_state.show_zone_map = True


def show_dept_popup():
    st.session_state.show_dept_map = True


def close_popup(popup_name):
    st.session_state[popup_name] = False

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def indicator_label(indicator: str) -> str:
    labels = {
        "temp_moyenne": "Temperature moyenne (°C)",
        "tmax_mean": "Temperature maximale moyenne (°C)",
        "tmin_mean": "Temperature minimale moyenne (°C)",
        "precipitation_total": "Precipitations totales (mm)",
        "precipitation_total_avril_septembre": "Precipitations avril-septembre (mm)",
        "Huglin_Index": "Indice de Huglin (°C-jours)",
        "Hot_D": "Nombre de jours de chaleur",
        "Very_Hot_D": "Nombre de jours de forte chaleur",
        "Frost_D": "Nombre de jours de gel",
        "Late_Frost": "Nombre de jours de gel tardif",
        "stress_climatique": "Stress climatique",
        "deficit_hydrique": "Deficit hydrique (mm)",
        "Climatic_Dryness_Index": "Indice de secheresse climatique (mm)",
        "Soil_Water_Stock": "Reserve utile en eau du sol (mm)",
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
        "Very_Hot_D",
        "deficit_hydrique",
        "precipitations_total"
    }
    val_round = round(value)
    if indicator in no_decimal_indicators:
        return f"{val_round:.0f}"
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
    conn = get_conn()
    df = conn.execute("SELECT * FROM climat_final").df()
    return df

@st.cache_data
def load_projection() -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute("SELECT * FROM climat_projection_final").df()
    return ensure_proj_schema(df)


@st.cache_data
def load_geo_communes() -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute("SELECT * FROM geo_communes").df()
    df["code_commune"] = df["code_commune"].astype(str)
    return df


@st.cache_data
def load_geo_zone() -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute("SELECT * FROM geo_zone").df()
    
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
df_geo_communes = load_geo_communes()


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
            colors=["#022f69", "#749dd3", "yellow","#e46e0f","red"],
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
        
    elif indicator == "stress_climatique":
        colormap = LinearColormap(colors=["white", "orange", "red"], vmin=0, vmax=100)
        colormap = colormap.to_step(index=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    elif indicator == "temp_moyenne":
        colormap = LinearColormap(colors=["white","yellow", "orange", "red"], vmin=9, vmax=21)
        colormap = colormap.to_step(index=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    elif indicator == "tmax_mean":
        colormap = LinearColormap(colors=["white","yellow", "orange", "red"], vmin=13, vmax=25)
        colormap = colormap.to_step(index=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    elif indicator == "tmin_mean":
        colormap = LinearColormap(colors=["white", "yellow", "orange", "red"], vmin=5, vmax=17)
        colormap = colormap.to_step(index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
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
        return folium.Map(location=[43.7, 3.5], zoom_start=7)

    zone_geom = gdf[["zone", "geometry"]].dissolve(by="zone", as_index=False)

    values_df = df_zone_values[["zone", indicator]].copy()
    values_df["zone"] = safe_numeric(values_df["zone"]).astype("Int64")
    values_df[indicator] = safe_numeric(values_df[indicator])

    zones_gdf = zone_geom.merge(values_df, on="zone", how="left")
    zones_gdf["zone_str"] = zones_gdf["zone"].astype("Int64").astype(str)
    zones_gdf["zone_label"] = zones_gdf["zone_str"].map(ZONE_LABELS_Tick)
    zones_gdf["indicator_fmt"] = zones_gdf[indicator].apply(lambda x: format_value(x, indicator))
    bounds = zones_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, tiles=None, zoom_start=7)
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
            "weight": 0.60,
            "fillOpacity": 0.85,
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
    title_html = f"""
    <div style="
        position: fixed;
        top: 65px;
        left: 40%;
        transform: translateX(-50%);
        z-index: 9999;
        background: white;
        border: 2px solid #666;
        border-radius: 4px;
        padding: 7px 14px;
        font-size: 11px;
        font-weight: bold;
        box-shadow: 0 0 6px rgba(0,0,0,0.25);
    ">
        {title_label}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element("""
        <style>
        .leaflet-container {
            background: white !important;
        }
        </style>
    """))
    return m


def create_communes_map(
    df_communes_values: pd.DataFrame,
    indicator: str,
    title_label: str
):

    communes_gdf = load_communes_geojson()
    communes_gdf = communes_gdf.rename(columns={"commune_code": "code_commune"})
    geo_communes = df_geo_communes[["code_commune"]].drop_duplicates().copy()
    communes_gdf["code_commune"] = safe_numeric(communes_gdf["code_commune"]).astype("Int64")
    geo_communes["code_commune"] = safe_numeric(geo_communes["code_commune"]).astype("Int64")
    gdf = communes_gdf.merge(geo_communes, on="code_commune", how="left")
    df_values = df_communes_values.copy()
    df_values["code_commune"] = safe_numeric(df_values["code_commune"]).astype("Int64")
    df_values[indicator] = safe_numeric(df_values[indicator])
    df_values = df_values.rename(columns={indicator: "value"})
    gdf = gdf.merge(df_values[["code_commune", "value"]], on="code_commune", how="left")
    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=7, tiles=None)
    colormap = build_indicator_colormap(gdf["value"], indicator)
    if colormap is None:
        return m
    def format_indicator (val_etiquette):
        if pd.isnull(val_etiquette):
            return "Non renseigne"
        if any(x in indicator for x in ["temp_moyenne","tmax_mean", "tmin_mean"]):
            return f"{val_etiquette:.1f}"
        elif "precipitation_total" in indicator:
            return f"{val_etiquette:.0f}"
        return f"{val_etiquette:.0f}"
    gdf["value_fmt"] = gdf["value"].apply(format_indicator)
    def style_function(feature):
        val = feature["properties"].get("value")
        return {
            "fillColor": "white" if val is None else colormap(val),
            "color": "white",
            "weight": 0.25,
            "fillOpacity": 0.8,
        }
        
    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["code_commune", "value_fmt"],
            aliases=["Commune", indicator_label(indicator)],
            localize=False,
            sticky=True,
        ),
    ).add_to(m)

    colormap.add_to(m)
    title_html = f"""
    <div style="
        position: fixed;
        top: 65px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: white;
        border: 2px solid #666;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: bold;
        box-shadow: 0 0 6px rgba(0,0,0,0.25);
    ">
        {title_label}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element("""
        <style>
        .leaflet-container {
            background: white !important;
        }
        </style>
    """))
    return m

def plot_historical_curves(
    df: pd.DataFrame,
    selected_zones: list[int],
    indicator: str,
    mode: str
):

    fig, ax = plt.subplots(figsize=(16, 10))

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    ).astype("Int64")

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

        if indicator == "Climatic_Dryness_Index":
            grouped = grouped[grouped["Year"] >= 2011]
        grouped = grouped[grouped["Year"] >= 2008]
        grouped["Year"] = grouped["Year"].astype(int)
        zone_str = str(int(zone))
        if mode == "Historique":
    
            ax.plot(
                grouped["Year"],
                grouped[indicator],
                marker="o",
                linewidth=1,
                color=ZONE_COLOR_MAP.get(zone_str, "#333333"),
                label=ZONE_LABELS.get(
                    zone_str,
                    f"Zone {zone_str}"
                ),
            )
        elif mode == "Tendance":

            x = grouped["Year"].to_numpy(dtype=float)
            y = grouped[indicator].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope * x + intercept
            ax.plot(
                grouped["Year"],
                trend,
                linestyle="--",
                linewidth=2,
                color=ZONE_COLOR_MAP.get(zone_str, "#333333"),
                label=ZONE_LABELS.get(
                    zone_str,
                    f"Zone {zone_str}"
                ),
            )
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True)
    )
    title = (
        "Fluctuations historiques"
        if mode == "Historique"
        else "Courbes de tendance"
    )
    ax.set_title(
        f"{title} - {indicator_label(indicator)}"
    )
    temp_indicators = {"temp_moyenne", "tmax_mean", "tmin_mean"}
    if indicator in temp_indicators:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.1f}".replace(",", " "))
        )
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
        )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    ax.set_xlabel("Annee")
    ax.set_ylabel(indicator_label(indicator))
    ax.grid(axis="y",linestyle="--",alpha=0.5)
    ax.legend(fontsize=13,bbox_to_anchor=(0.5, -0.15),loc="upper center",title="Legende",title_fontsize=15)
    fig.subplots_adjust(bottom=0.28)
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
            if indicator == "Climatic_Dryness_Index":
                grouped = grouped[grouped["Year"] >= 2011]
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
                slope, _ = np.polyfit(x, y, 1)
                last_year = int(grouped["Year"].max())
                last_val = grouped[indicator].iloc[-1]
                for i, year in enumerate(range(2007 + 1, 2041), start=1):
                    value = slope * i + last_val
                    rows.append(
                        {
                            "cluster": int(zone),
                            "Year": int(year),
                            "indicator": indicator,
                            "value": value,
                            "type": "projection_sans_scenario",
                        }
                    )

    return pd.DataFrame(rows)


def plot_no_scenario_curves(df_no_scenario: pd.DataFrame, selected_zones: list[int], indicator: str):
    fig, ax = plt.subplots(figsize=(16, 10))

    for zone in selected_zones:
        df_zone = df_no_scenario[
            (df_no_scenario["cluster"] == zone) & (df_no_scenario["indicator"] == indicator)
        ].copy()

        if df_zone.empty:
            continue
        df_proj = df_zone[df_zone["type"] == "projection_sans_scenario"].sort_values("Year")
        zone_str = str(int(zone))
        color = ZONE_COLOR_MAP.get(zone_str, "#333333")
        label = ZONE_LABELS.get(zone_str, f"Zone {zone_str}")
        if not df_proj.empty:
            ax.plot(
                df_proj["Year"].astype(int),
                df_proj["value"],
                marker="x",
                linestyle="--",
                linewidth=2,
                color=color,
                label=label,
            )
    ax.set_title(f"Projection sans scenario - {indicator_label(indicator)}")
    ax.set_xlabel("Annee")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )
    ax.set_ylabel(indicator_label(indicator))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=15, bbox_to_anchor=(0.5, -0.15), loc="upper center", title="Legende", title_fontsize=17)
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


def plot_scenario_comparison(df_table, indicator, period, scenario):
    fig, ax = plt.subplots(figsize=(16, 6))
    df = df_table[df_table["scenario"] == scenario].copy()
    sorted_zones = sorted(df["zone"].unique())
    x = np.arange(len(sorted_zones))
    hist_col_map = {
        "temp_moyenne": "Temperature 2008-2024 (°C)",
        "tmax_mean": "Temperature 2008-2024 (°C)",
        "tmin_mean": "Temperature 2008-2024 (°C)",
        "precipitation_total": "Precipitations 2008-2024 (mm)"
    }
    trend_col_map = {
        "temp_moyenne": "Tendance temperature (°C)",
        "tmax_mean": "Tendance temperature (°C)",
        "tmin_mean": "Tendance temperature (°C)",
        "precipitation_total": "Tendance precipitations (mm)"
    }
    hist_col = hist_col_map[indicator]
    trend_col = trend_col_map[indicator]
    df_indexed = df.set_index("zone")
    hist_vals = df_indexed.loc[sorted_zones, hist_col].values
    trend_vals = df_indexed.loc[sorted_zones, trend_col].values
    hist_vals = np.array(hist_vals, dtype=float)
    trend_vals = np.array(trend_vals, dtype=float)
    if indicator == "temp_moyenne" or indicator =="tmax_mean" or indicator =="tmin_mean":
        for i in range(len(x)):
            projection = hist_vals[i] + trend_vals[i]
            ax.text(
                x[i],
                projection+0.4,
                f"{projection:.1f}",
                ha="center",
                va ="bottom",
                fontsize=15,
            )
            ax.text(
                x[i],
                hist_vals[i] + trend_vals[i] / 2,
                f"{trend_vals[i]:.1f} °C",
                ha="center",
                fontsize=15,
                fontweight="bold",
                color ="black"
            )
            ax.bar(x, hist_vals, color="yellow", alpha=0.85, width=0.8)
            ax.bar(x, trend_vals, bottom=hist_vals, color="red", alpha=0.85, width=0.8)
    if indicator == "precipitation_total":
        for i in range(len(x)):
            projection = hist_vals[i] + trend_vals[i]
            ax.text(
                x[i],
                projection + 0.5,
                f"{projection:.0f}",
                fontsize=18,
                ha="center",
                va="bottom"
            )
            ax.text(
                x[i],
                (hist_vals[i] + trend_vals[i])* 0.75,
                f"{trend_vals[i]:.0f} %",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color ="black"
            )
            ax.bar(x, hist_vals, color="yellow", alpha=0.85, width=0.8)
            ax.bar(x, trend_vals, bottom=hist_vals, color= "red", alpha=0.85, width=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Zone")
    ax.set_ylabel(indicator_label(indicator))
    ax.set_xticklabels([f"Zone {int(z)}" for z in sorted_zones])
    ax.set_title(f"{indicator_label(indicator)} - {period} ({scenario})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    y_bottom = min(0, np.min(hist_vals + trend_vals))
    y_top = np.max(hist_vals + trend_vals)
    ax.set_ylim(y_bottom, y_top + (y_top - y_bottom) * 0.30)
    return fig

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-header">
    <h1>Observatoire Viticole</h1>
    <h2>Pays d'Oc IGP</h2>
    <p>Analyse du climat</p>
</div>
""", unsafe_allow_html=True)

tab_details, tab_histo_2008_2024, tab_future = st.tabs(["Details",
    "Historique 2008-2024", "Projections"
])
st.markdown("""
    <style>
    div[data-testid="stRadio"] label p {
        font-size: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

@st.dialog("Carte des zones pédoclimatiques")
def show_carte_inrae_clusters():
    st.markdown("""
        <style>
        div[data-testid="stDialog"] > div > div {
            max-width: 67vw !important;
            width: 67vw !important;
        }
        </style>
    """, unsafe_allow_html=True)
    with open("carte_zones_pedoclimatiques_.png", "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    components.html(f"""
        <img id="img" src="data:image/png;base64,{img_data}" 
             style="width:90%; cursor:zoom-in;"
             onclick="this.style.width = this.style.width=='90%' ? '110%' : '90%'">
    """, height=750, scrolling=True)


with st.sidebar:
    if st.button("Carte des zones pédoclimatiques"):
        show_carte_inrae_clusters()
    
with tab_details:
    
    st.header("Informations sur l'historique et les projections climatiques")
    st.info(
        "Les donnees historiques proviennent de la plateforme [Open-Meteo]"
        "(https://open-meteo.com/en/docs/historical-weather-api). "
        "Les projections climatiques issues du [GIEC](https://meteofrance.fr/actualite/presse-0/"
        "6e-rapport-du-giec-les-contributions-de-meteo-france) reposent sur "
        "les donnees de la base mondiale [WorldClim]"
        "(https://www.worldclim.org/data/cmip6/cmip6_clim2.5m.html) "
        "pour 2021-2040 et 2041-2060. Plusieurs modeles climatiques globaux sont analyses. "
        "Des disparites subsistent "
        "entre les simulations des temperatures et des precipitations "
        "en fonction des scenarios SSP126 (optimiste), SSP245 (neutre) et SSP585 (pessimiste). "
        "Ces ecarts decoulent de plusieurs sources d'incertitude "
        "scientifique et technique evoquees ci-dessous :\n\n"
        "1.\tLa [variabilite naturelle du climat](https://culturesciencesphysique.ens-lyon.fr/"
        "pdf/GIEC-climat.pdf) peu previsible comprend les variations des courants oceaniques "
        "et de la temperature de surface de la mer impactant le developpement de la vigne. "
        "À partir de deux etats climatiques tres proches, les evolutions climatiques "
        "peuvent être considerablement differentes.\n\n"
        "2.\tLa modelisation des [retroactions](https://web.lmd.jussieu.fr/~jldufres/Exposes/"
        "Duf_ChEDF_juin_2014.pdf) reste complexe, en particulier pour la "
        "vapeur d'eau sur la stratosphere. Cela peut être causee par certains phenomenes "
        "regionaux: les episodes cevenols ou les vents "
        "regionaux (Autan, Tramontane).\n\n"
        "3.\tLes fortes incertitudes physiques sur la representation des [precipitations](https://www.foret-mediterraneenne.org/"
        "_0/upload/biblio/foret_med_2011_2_205-212.pdf) rendent l'estimation "
        "du bilan hydrique des sols complexe, particulierement en hiver. Cette mesure "
        "influence la croissance de la vigne et la maturation des raisins. \n\n"
        "4.\tLa resolution spatiale et temporelle est propre à chaque modele.\n\n"
        "Sur la base de cinq modeles europeens, les valeurs centrales "
        "des variables climatiques sont determinees à l'echelle pluriannuelle et "
        "sur le cycle vegetatif de la vigne pour l'ensemble des communes des sept "
        "zones pedoclimatiques. "
        "Seuls certains indicateurs sont representes en raison des limites liees à la disponibilite"
        " des donnees.\n\n"
        "Pour connaître les significations des indicateurs climatiques, vous pouvez naviguer [ici]"
        "(https://www.vignevin-occitanie.com/wp-content/uploads/2023/01/guide-vitisad-fr-FINAL.pdf)."
    )
    
with tab_histo_2008_2024:
    st.header(
            "Analyse de l'evolution des indicateurs climatiques des zones"
    )
    view_historical = st.radio("ds", 
                           options=["Graphiques",
                                    "Cartographie"], 
                           index=0, 
                           horizontal=True,
                           label_visibility="collapsed"
                        )
    if view_historical == "Cartographie":
        available_years = sorted([int(y) for y in df_climat["Year"].dropna().unique()])
        year_max = int(df_climat["Year"].dropna().max())
        year_min = year_max - 4
        selected_year = st.selectbox(
            "Annee historique",
            options=available_years,
            index=len(available_years) - 1,
            label_visibility="collapsed"
        )
        use_last_5y_mean = st.checkbox("Moyenne des 5 dernieres annees", value=False)
        map_indicator_hist = st.selectbox(
            "Indicateur climatique",
            options=MAP_INDICATORS,
            format_func=indicator_label,
            key="map_indicator_hist",
            label_visibility="collapsed"
        )

        try:
            if use_last_5y_mean:
                zone_values_hist = (
                    df_climat[df_climat["Year"].between(year_min, year_max)]
                    .groupby("cluster", as_index=False)[map_indicator_hist]
                    .mean()
                    .rename(columns={"cluster": "zone"})
                )
                map_title = f"{indicator_label(map_indicator_hist)} - {year_min}-{year_max}"
            else:
                zone_values_hist = (
                    df_climat[df_climat["Year"] == selected_year]
                    .groupby("cluster", as_index=False)[map_indicator_hist]
                    .mean()
                    .rename(columns={"cluster": "zone"})
                )
                map_title = f"{indicator_label(map_indicator_hist)} - {int(selected_year)}"

            map_hist = create_zone_map(zone_values_hist, map_indicator_hist, map_title)
            components.html(map_hist._repr_html_(), height=500)
            st.markdown("**Legende**")
            st.markdown("""
                <div>
                    <span style="color:#000000">O</span> Zone 1: zone humide de l'arriere-pays<br>
                    <span style="color:#FF0000">O</span> Zone 2: zone de montagne avec des sols acides et peu profonds<br>
                    <span style="color:#1A8F2A">O</span> Zone 3: zone de piemont avec une reserve utile limitante<br>
                    <span style="color:#0033CC">O</span> Zone 4: zone froide et seche autour du Pic Saint-Loup<br>
                    <span style="color:#AFC6D9">O</span> Zone 5: zone de sols de qualite moyenne dans l’arriere-pays<br>
                    <span style="color:#7A1FA2">O</span> Zone 6: zone de sols profonds sur côtes temperees<br>
                    <span style="color:#FFD800">O</span> Zone 7: zone avec le plus grand nombre de jours tres chauds mais sols profonds
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur carte : {e}")
    if view_historical == "Graphiques":
        available_zones = sorted([int(z) for z in df_climat["cluster"].dropna().unique()])
        selected_zones = st.multiselect(
            "Zones",
            options=available_zones,
            default=available_zones[:3] if len(available_zones) >= 3 else available_zones,
            key="histo_zones",
            label_visibility="collapsed"
        )
        
        if not selected_zones:
            st.warning("Selectionnez au moins une zone.")
            st.stop()
        
        selected_indicator = st.selectbox(
            "Indicateur climatique",
            options=HISTORICAL_INDICATORS,
            format_func=indicator_label,
            label_visibility="collapsed"
        )
        graph_mode = st.radio(
            "Type de graphique",
            options=["Fluctuations historiques", "Courbes de tendance"],
            horizontal=True,
            label_visibility="collapsed"
        )

        mode_map = {
            "Fluctuations historiques": "Historique",
            "Courbes de tendance": "Tendance"
        }

        try:
            fig_hist = plot_historical_curves(df_climat, selected_zones, selected_indicator, mode_map[graph_mode])
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        except Exception as e:
            st.error(f"Erreur graphique historique : {e}")


# =====================================================
# SANS SCeNARIO
# =====================================================
with tab_future:
    st.header(
            "Analyse des perspectives climatiques des zones"
    )
    view_future = st.radio(" ",
                           options=["Projections des tendances passees", "Projections des scenarios – graphiques",
                                    "Projections – cartes"],
                           index=0,
                           horizontal=True,
                           label_visibility="collapsed"
                           )
    
    if view_future == "Projections des tendances passees":
        st.markdown("**Quel sera le climat si les tendances actuelles se poursuivent d'ici 2040 ?**")
        available_zones = sorted([int(z) for z in df_climat["cluster"].dropna().unique()])
        selected_zones = st.multiselect(
            "Zones",
            options=available_zones,
            default=available_zones[:3] if len(available_zones) >= 3 else available_zones,
            key="histo_zones_future",
            label_visibility="collapsed"
        )
        available_years = sorted([int(y) for y in df_climat["Year"].dropna().unique()])
        if not selected_zones:
            st.warning("Selectionne au moins une zone.")
            st.stop()
        try:
            df_no_scenario = build_no_scenario_projection(
                df_hist=df_climat,
                zones=selected_zones,
                indicators=HISTORICAL_INDICATORS,
            )

            no_scenario_indicator = st.selectbox(
                "Indicateur climatique",
                options=HISTORICAL_INDICATORS,
                format_func=indicator_label,
                key="no_scenario_indicator",
                label_visibility="collapsed"
            )

            fig_no_scenario = plot_no_scenario_curves(
                df_no_scenario=df_no_scenario,
                selected_zones=selected_zones,
                indicator=no_scenario_indicator,
            )
            st.pyplot(fig_no_scenario)
            plt.close(fig_no_scenario)

        except Exception as e:
            st.error(f"Erreur projection sans scenario : {e}")
    if view_future == "Projections des scenarios – graphiques":
    # =====================================================
    # SCeNARIOS
    # =====================================================
        st.markdown("**Evolution des temperatures et des precipitations entre les scenarios à court et moyen terme**")
        available_zones = sorted([int(z) for z in df_climat["cluster"].dropna().unique()])
        selected_zones = st.multiselect(
            "Zones",
            options=available_zones,
            default=available_zones[:7] if len(available_zones) >= 3 else available_zones,
            key="histo_zones_future",
            label_visibility="collapsed"
        )
        available_years = sorted([int(y) for y in df_climat["Year"].dropna().unique()])
        scenario_period = st.selectbox(
            "Periode future consideree",
            options=["2021-2040", "2041-2060"],
            key="map_scenario_period_1",
            label_visibility="collapsed"
        )
        compare_indicator = st.selectbox(
            "Indicateur climatique",
            options=["Temperature moyenne (°C)", "Temperature maximale moyenne (°C)", "Temperature minimale moyenne (°C)", "Precipitations totales (mm)"],
            key="compare_indicator_scenario_tableau",
            label_visibility="collapsed"
        )
        mask = (
            (df_proj["periode"] == scenario_period)
        )
        indicator_map = {
            "Temperature moyenne (°C)": "temp_moyenne",
            "Temperature maximale moyenne (°C)": "tmax_mean",
            "Temperature minimale moyenne (°C)": "tmin_mean",
            "Precipitations totales (mm)": "precipitation_total",
        }
        hist_map = {
            "temp_moyenne": "Temperature 2008-2024 (°C)",
            "tmax_mean": "Temperature 2008-2024 (°C)",
            "tmin_mean": "Temperature 2008-2024 (°C)",
            "precipitation_total": "Precipitations 2008-2024 (mm)"
        }
        trend_map = {
            "temp_moyenne": "Tendance temperature (°C)",
            "tmax_mean": "Tendance temperature (°C)",
            "tmin_mean": "Tendance temperature (°C)",
            "precipitation_total": "Tendance precipitations (mm)"
        }
        label_map = {
            "temp_moyenne": "Temperature moyenne (°C)",
            "tmax_mean": "Temperature maximale moyenne (°C)",
            "tmin_mean": "Temperature minimale moyenne (°C)",
            "precipitation_total": "Precipitations totales (mm)",
        }
        proj_period = df_proj.loc[mask].copy()
        if proj_period.empty:
            st.warning("Aucune donnee de projection disponible pour cette periode.")
        else:
            scenario_table = build_scenario_table(proj_period, selected_zones)
            df_no_scenario = build_no_scenario_projection(
                df_hist=df_climat,
                zones=selected_zones,
                indicators=HISTORICAL_INDICATORS,
            )
            col = indicator_map[compare_indicator]
            no_scenario_table = (
                df_no_scenario[df_no_scenario["indicator"] == col]
                .groupby("cluster", as_index=False)["value"]
                .mean()
                .rename(columns={
                    "cluster": "zone",
                    "value": col
                })
            )
            no_scenario_table["scenario"] = "sans scenario"
            no_scenario_table[col] = no_scenario_table[col].astype(float)
            df_hist_zone = df_climat[df_climat["cluster"].isin(selected_zones)].copy()
            hist_means = df_hist_zone.groupby("cluster").agg({
                "temp_moyenne": "mean",
                "tmax_mean": "mean",
                "tmin_mean": "mean",
                "precipitation_total": "mean"
            })
            hist_col_name = hist_map[col]
            trend_col_name = trend_map[col]
            scenario_table[hist_col_name] = (
                scenario_table["zone"]
                .map(hist_means[col])
                .astype(float)
                .round(1)
            )
            if col == "precipitation_total":
                scenario_table[trend_col_name] = (
                    (scenario_table["precipitation_total"] - scenario_table[hist_col_name])
                    / scenario_table[hist_col_name]
                    * 100
                ).astype(float).round(0)
                no_scenario_table[hist_col_name] = no_scenario_table["zone"].map(hist_means[col]).astype(float).round(0)
                no_scenario_table[trend_col_name] = (
                    (no_scenario_table[col] - no_scenario_table[hist_col_name])
                    / no_scenario_table[hist_col_name]
                    * 100
                ).astype(float).round(0)
                
            else:
                scenario_table[trend_col_name] = (
                    scenario_table[col] - scenario_table[hist_col_name]
                ).astype(float).round(1)
                no_scenario_table[hist_col_name] = no_scenario_table["zone"].map(hist_means[col]).astype(float).round(1)
                no_scenario_table[trend_col_name] = (
                    no_scenario_table[col] - no_scenario_table[hist_col_name]
                ).astype(float).round(1)
            display_col = label_map[col]
            for c in ["temp_moyenne", "tmax_mean", "tmin_mean"]:
                if c in no_scenario_table.columns:
                    no_scenario_table[c] = no_scenario_table[c].astype(float).round(1)
                if c in scenario_table.columns:
                    scenario_table[c] = scenario_table[c].astype(float).round(1)
            for c in ["precipitation_total"]:
                if c in no_scenario_table.columns:
                    no_scenario_table[c] = no_scenario_table[c].astype(float).round(0)
                if c in scenario_table.columns:
                    scenario_table[c] = scenario_table[c].astype(float).round(0)  
            scenario_table = pd.concat([scenario_table, no_scenario_table], ignore_index=True)
            scenario_table_display = scenario_table.copy()
            scenario_table_display["zone"] = scenario_table_display["zone"].apply(lambda z: f"Zone {int(z)}")
            scenario_table_display = scenario_table_display.rename(
                columns={col: display_col}
            )
            cols_to_keep = [
                "zone",
                hist_col_name,
                "scenario",
                trend_col_name,
                display_col
            ]
            scenario_table_display = scenario_table_display[cols_to_keep]
            st.dataframe(
                scenario_table_display,
                width="stretch",
                height=900,
                row_height=30
            )
            map_scenario = st.selectbox(
                "Scenario etudie",
                options=["optimiste", "neutre", "pessimiste", "sans scenario"],
                key="map_scenario_tab",
                label_visibility="collapsed"
            )
            indicator = indicator_map[compare_indicator]
            if map_scenario == "sans scenario":
                df_graph = scenario_table[
                    scenario_table["scenario"] == "sans scenario"
                ]
            else:
                df_graph = scenario_table[
                    scenario_table["scenario"] == map_scenario
                ]
            fig_scenario = plot_scenario_comparison(
                df_table=df_graph,
                indicator=indicator,
                period=scenario_period,
                scenario=map_scenario
            )

            st.pyplot(fig_scenario)
            st.markdown("**Legende**")
            if col == "precipitation_total":
                st.markdown("""
                <div>
                    <span style="color:#FFFF00">⬤</span> Projections sur les precipitations<br>
                    <span style="color:#FF0000">⬤</span> Ecart entre le passe et les predictions<br>

                </div>""",
                unsafe_allow_html=True
                )
            else:
                st.markdown("""
                <div>
                    <span style="color:#FFFF00">⬤</span> Donnees historiques des temperatures<br>
                    <span style="color:#FF0000">⬤</span> Ecart entre le passe et les predictions<br>

                </div>""",
                unsafe_allow_html=True
                )
            plt.close(fig_scenario)
    if view_future == "Projections – cartes":
        st.markdown("**Evolution des tendances climatiques futures à court et moyen terme par zone**")
        map_scenario = st.selectbox(
            "Scenario etudie",
            options=["optimiste", "neutre", "pessimiste"],
            key="map_scenario",
            label_visibility="collapsed"
        )
        scenario_period = st.selectbox(
            "Periode future consideree",
            options=["2021-2040", "2041-2060"],
            key="map_scenario_period_future",
            label_visibility="collapsed"
        )
        map_indicator_proj = st.selectbox(
            "Indicateur climatique",
            options=SCENARIO_INDICATORS,
            format_func=indicator_label,
            key="map_indicator_proj",
            label_visibility="collapsed"
        )
        proj_period = df_proj[df_proj["periode"] == scenario_period].copy()
        try:
            values_proj_zone = (
                proj_period[proj_period["scenario"] == map_scenario]
                .groupby("zone", as_index=False)[map_indicator_proj]
                .mean()
            )

            map_proj = create_zone_map(
                values_proj_zone,
                map_indicator_proj,
                f"{indicator_label(map_indicator_proj)} - {map_scenario} - {scenario_period}",
            )
            components.html(map_proj._repr_html_(), height=500)
            st.markdown("**Legende**")
            st.markdown("""
                <div>
                    <span style="color:#000000">O</span> Zone 1: zone humide de l'arriere-pays<br>
                    <span style="color:#FF0000">O</span> Zone 2: zone de montagne avec des sols acides et peu profonds<br>
                    <span style="color:#1A8F2A">O</span> Zone 3: zone de piemont avec une reserve utile limitante<br>
                    <span style="color:#0033CC">O</span> Zone 4: zone froide et seche autour du Pic Saint-Loup<br>
                    <span style="color:#AFC6D9">O</span> Zone 5: zone de sols de qualite moyenne dans l’arriere-pays<br>
                    <span style="color:#7A1FA2">O</span> *Zone 6: zone de sols profonds sur côtes temperees<br>
                    <span style="color:#FFD800">O</span> *Zone 7: zone avec le plus grand nombre de jours tres chauds mais sols profonds
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:#ff4b00-m
            st.error(f"Erreur carte  : {e}")
        st.markdown("**Si nous poursuivons les tendances passees, quelles seraient les projections des temperatures et des precipitations ?**")
        available_zones = sorted([int(z) for z in df_climat["cluster"].dropna().unique()])
        map_indicator_proj = SCENARIO_INDICATORS[0]
        
        compare_indicator = st.selectbox(
            "Indicateur climatique",
            options=["Temperature moyenne (°C)", "Temperature maximale moyenne (°C)", "Temperature minimale moyenne (°C)", "Precipitations totales (mm)"],
            key="compare_indicator_scenario_tableau",
            label_visibility="collapsed"
        )
        indicator_map = {
            "Temperature moyenne (°C)": "temp_moyenne",
            "Temperature maximale moyenne (°C)": "tmax_mean",
            "Temperature minimale moyenne (°C)": "tmin_mean",
            "Precipitations totales (mm)": "precipitation_total"
        }
        hist_map = {
            "temp_moyenne": "Temperature 2008-2024 (°C)",
            "tmax_mean": "Temperature 2008-2024 (°C)",
            "tmin_mean": "Temperature 2008-2024 (°C)",
            "precipitation_total": "Precipitations 2008-2024 (mm)"
        }
        trend_map = {
            "temp_moyenne": "Tendance temperature (°C)",
            "tmax_mean": "Tendance temperature (°C)",
            "tmin_mean": "Tendance temperature (°C)",
            "precipitation_total": "Tendance precipitations (mm)"
        }
        label_map = {
            "temp_moyenne": "Temperature moyenne (°C)",
            "tmax_mean": "Temperature maximale moyenne (°C)",
            "tmin_mean": "Temperature minimale moyenne (°C)",
            "precipitation_total": "Precipitations totales (mm)"
        }
        scenario_period = st.selectbox(
            "Periode future consideree",
            options=["2021-2040"],
            key="map_scenario_period_future_bis_bis",
            label_visibility="collapsed"
        )
        proj_period = df_proj[df_proj["periode"] == scenario_period].copy()
        available_years = sorted([int(y) for y in df_climat["Year"].dropna().unique()])
        selected_zones = st.multiselect(
            "Zones",
            options=available_zones,
            default=available_zones[:7] if len(available_zones) >= 3 else available_zones,
            key="histo_zones_future",
            label_visibility="collapsed"
        )
        if not selected_zones: 
            st.warning("Selectionnez au moins une zone.")
            st.stop()
        scenario_table = build_scenario_table(proj_period, selected_zones)
        df_no_scenario = build_no_scenario_projection(
            df_hist=df_climat,
            zones=selected_zones,
            indicators=HISTORICAL_INDICATORS,
        )
        col = indicator_map[compare_indicator]
        indicator = indicator_map[compare_indicator]
        if "indicator" not in df_no_scenario.columns or df_no_scenario.empty:
            st.warning ("Aucune donnee disponible sur les projections")
            st.stop()
        map_indicator_proj = col
        no_scenario_table = (
            df_no_scenario[df_no_scenario["indicator"] == col]
            .groupby("cluster", as_index=False)["value"]
            .mean()
            .rename(columns={
                "cluster": "zone",
                "value": col
            })
        )
        no_scenario_table["scenario"] = "sans scenario"
        no_scenario_table[col] = no_scenario_table[col].astype(float)
        df_hist_zone = df_climat[df_climat["cluster"].isin(selected_zones)].copy()
        hist_means = df_hist_zone.groupby("cluster").agg({
            "temp_moyenne": "mean",
            "tmax_mean": "mean",
            "tmin_mean": "mean",
            "precipitation_total": "mean"
        })
        hist_col_name = hist_map[col]
        trend_col_name = trend_map[col]
        scenario_table[hist_col_name] = (
            scenario_table["zone"]
            .map(hist_means[col])
            .astype(float)
            .round(1)
        )
        if col == "precipitation_total":
            scenario_table[trend_col_name] = (
                (scenario_table["precipitation_total"] - scenario_table[hist_col_name])
                / scenario_table[hist_col_name]
                * 100
            ).astype(float).round(0)
            no_scenario_table[hist_col_name] = no_scenario_table["zone"].map(hist_means[col]).astype(float).round(0)
            no_scenario_table[trend_col_name] = (
                (no_scenario_table[col] - no_scenario_table[hist_col_name])
                / no_scenario_table[hist_col_name]
                * 100
            ).astype(float).round(0)
            
        else:
            scenario_table[trend_col_name] = (
                scenario_table[col] - scenario_table[hist_col_name]
            ).astype(float).round(1)
            no_scenario_table[hist_col_name] = no_scenario_table["zone"].map(hist_means[col]).astype(float).round(1)
            no_scenario_table[trend_col_name] = (
                no_scenario_table[col] - no_scenario_table[hist_col_name]
            ).astype(float).round(1)
        display_col = label_map[col]
        for c in ["temp_moyenne", "tmax_mean", "tmin_mean"]:
            if c in no_scenario_table.columns:
                no_scenario_table[c] = no_scenario_table[c].astype(float).round(1)
            if c in scenario_table.columns:
                scenario_table[c] = scenario_table[c].astype(float).round(1)
        for c in ["precipitation_total"]:
            if c in no_scenario_table.columns:
                no_scenario_table[c] = no_scenario_table[c].astype(float).round(0)
            if c in scenario_table.columns:
                scenario_table[c] = scenario_table[c].astype(float).round(0)  
        scenario_table = pd.concat([scenario_table, no_scenario_table], ignore_index=True)
        scenario_table_display = scenario_table.copy()
        scenario_table_display["zone"] = scenario_table_display["zone"].apply(lambda z: f"Zone {int(z)}")
        scenario_table_display = scenario_table_display.rename(
            columns={col: display_col}
        )
        cols_to_keep = [
            "zone",
            hist_col_name,
            "scenario",
            trend_col_name,
            display_col
        ]
        scenario_table_display = scenario_table_display[cols_to_keep]
        df_graph = scenario_table[
                    scenario_table["scenario"] == "sans scenario"
                ]
        try:
            map_proj = create_zone_map(
                df_graph,
                map_indicator_proj,
                f"{indicator_label(map_indicator_proj)} - {scenario_period}",
            )
            components.html(map_proj._repr_html_(), height=500)
            st.markdown("**Legende**")
            st.markdown("""
                <div>
                    <span style="color:#000000">O</span> Zone 1: zone humide de l'arriere-pays<br>
                    <span style="color:#FF0000">O</span> Zone 2: zone de montagne avec des sols acides et peu profonds<br>
                    <span style="color:#1A8F2A">O</span> Zone 3: zone de piemont avec une reserve utile limitante<br>
                    <span style="color:#0033CC">O</span> Zone 4: zone froide et seche autour du Pic Saint-Loup<br>
                    <span style="color:#AFC6D9">O</span> Zone 5: zone de sols de qualite moyenne dans l’arriere-pays<br>
                    <span style="color:#7A1FA2">O</span> *Zone 6: zone de sols profonds sur côtes temperees<br>
                    <span style="color:#FFD800">O</span> *Zone 7: zone avec le plus grand nombre de jours tres chauds mais sols profonds
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur carte  : {e}")
        st.markdown("**Projection des precipitations sur le cycle vegetatif de la vigne à long terme**")
        map_scenario = st.selectbox(
            "Scenario etudie",
            options=["optimiste", "neutre", "pessimiste"],
            key="map_scenario_cycle_wine",
            label_visibility="collapsed"
        )
        scenario_period = st.selectbox(
            "Periode future consideree",
            options=["2041-2060"],
            key="map_scenario_period_future_cycle",
            label_visibility="collapsed"
        )
        map_indicator_proj = st.selectbox(
            "Indicateur climatique",
            options=WINE_CYCLE_SCENARIO_INDICATORS,
            format_func=indicator_label,
            key="map_indicator_proj_precip_cycle",
            label_visibility="collapsed"
        )
        proj_period = df_proj[df_proj["periode"] == scenario_period].copy()
        try:
            values_proj_zone = (
                proj_period[proj_period["scenario"] == map_scenario]
                .groupby("zone", as_index=False)[map_indicator_proj]
                .mean()
            )
            map_proj = create_zone_map(
                values_proj_zone,
                map_indicator_proj,
                f"{indicator_label(map_indicator_proj)} - {map_scenario} - {scenario_period}",
            )
            components.html(map_proj._repr_html_(), height=500)
            st.markdown("**Legende**")
            st.markdown("""
                <div>
                    <span style="color:#000000">O</span> Zone 1: zone humide de l'arriere-pays<br>
                    <span style="color:#FF0000">O</span> Zone 2: zone de montagne avec des sols acides et peu profonds<br>
                    <span style="color:#1A8F2A">O</span> Zone 3: zone de piemont avec une reserve utile limitante<br>
                    <span style="color:#0033CC">O</span> Zone 4: zone froide et seche autour du Pic Saint-Loup<br>
                    <span style="color:#AFC6D9">O</span> Zone 5: zone de sols de qualite moyenne dans l’arriere-pays<br>
                    <span style="color:#7A1FA2">O</span> *Zone 6: zone de sols profonds sur côtes temperees<br>
                    <span style="color:#FFD800">O</span> *Zone 7: zone avec le plus grand nombre de jours tres chauds mais sols profonds
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur carte  : {e}")
        

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Observatoire Viticole - Pays d'Oc IGP| Donnees mises a jour regulierement</p>
    <p style="font-size: 0.75rem;">(c) 2024 - Analyse des rendements et volumes viticoles</p>
</div>
""", unsafe_allow_html=True)

gc.collect()