# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.db import get_conn
from modules.data_loader import load_geojson
from modules.ai_engine import AIAnalyzer


st.set_page_config(
    page_title="Observatoire Viticole - Pays d'Oc IGP",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Style CSS personnalise
st.markdown("""
<style> 
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
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
        border-left: 4px solid #2c3e50;
    }
    .metric-value {
        font-size: 1.8rem;
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
        background-color: #f1f3f5;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #2c3e50;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.9rem;  
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;  
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="main-header">
    <h1>Observatoire Viticole - Pays d'Oc IGP</h1>
    <p>Mise en relation entre le climat et la production viticole</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# =====================================================
# CONFIG
# =====================================================

CLIMATE_VARS = [
    "temp_moyenne",
    "precipitation_total"
]

DISPLAY_LABELS = {
    "temp_moyenne": "Temperature moyenne (°C)",
    "precipitation_total": "Precipitations totales (mm)",
    "rendement": "Rendement (hl/ha)",
    "volume": "Volume (hl)",
    "zone": "Zone",
    "annee": "Annee",
    "code_couleur": "Couleur",
    "code_cepage": "Cepage",
    "code_departement": "Departement",
}

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

DISPLAY_LABELS_scoring_ = {
    "rendement_moy": "Rendement moyen (hl/ha)",
    "rendement_std": "Mesure de dispersion du rendement (hl/ha)",
    "score_final": "Score global",
    "classe_final": "Classe qualitative",
    "zone": "Zone"
}

WINE_COLOR_MAP = {
    "BL": "#F4D03F",
    "RG": "#A93226",
    "RS": "#F1948A",
}

WINE_CORRESPONDANCE = {
    "BL": "Blanc",
    "RG": "Rouge",
    "RS": "Rosé",
}

# =====================================================
# OUTILS
# =====================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def format_number(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def label_of(col: str) -> str:
    return DISPLAY_LABELS.get(col, col)


def minmax_scale(series: pd.Series) -> pd.Series:
    series = safe_numeric(series)
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(1.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def inverse_minmax_scale(series: pd.Series) -> pd.Series:
    scaled = minmax_scale(series)
    return 1 - scaled


def build_zone_color_dict(zones) -> dict:
    result = {}
    for z in zones:
        z_str = str(int(z)) if pd.notna(z) else "0"
        result[z] = ZONE_COLOR_MAP.get(z_str, "#7F7F7F")
    return result


def class_from_score(score: float) -> str:
    if pd.isna(score):
        return "Non classee"
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def weighted_corr(df_in: pd.DataFrame, x: str, y: str) -> float:
    tmp = df_in[[x, y]].dropna().copy()
    if len(tmp) < 3:
        return np.nan
    return tmp[x].corr(tmp[y])

def rename_columns_scoring(df):
    return df.rename(columns=DISPLAY_LABELS_scoring_)

# =====================================================
# CHARGEMENT DES DONNEES
# =====================================================

@st.cache_data
def load_climate_yield_geo() -> pd.DataFrame:
    """Charge les donnees climat_rendement_geo"""
    conn = get_conn()
    try:
        df = conn.execute("SELECT * FROM climat_rendement_geo").df()
    finally:
        pass

    df.columns = df.columns.astype(str).str.strip()

    numeric_cols = [
        "zone",
        "annee",
        "temp_moyenne",
        "precipitation_total",
        "rendement",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "zone" in df.columns:
        df["zone"] = df["zone"].astype("Int64")

    if "annee" in df.columns:
        df["annee"] = df["annee"].astype("Int64")

    for col in ["commune", "code_departement"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_data
def load_fusion_analysis() -> pd.DataFrame:
    """Charge les donnees fusion"""
    conn = get_conn()
    try:
        df = conn.execute(
            """
            SELECT
                type_mvt,
                code_couleur,
                annee,
                volume,
                surface,
                code_cepage,
                cvi,
                rendement,
                commune,
                zone,
                code_departement,
                departement,
                Huglin_Index,
                Hot_D,
                Very_Hot_D,
                Climatic_Dryness_Index,
                temp_moyenne,
                precipitation_total
            FROM fusion
            """
        ).df()
    finally:
        pass

    df.columns = df.columns.astype(str).str.strip()

    numeric_cols = [
        "annee",
        "volume",
        "surface",
        "rendement",
        "zone",
        "Huglin_Index",
        "Hot_D",
        "Very_Hot_D",
        "Climatic_Dryness_Index",
        "temp_moyenne",
        "precipitation_total",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "annee" in df.columns:
        df["annee"] = df["annee"].astype("Int64")

    if "zone" in df.columns:
        df["zone"] = df["zone"].astype("Int64")

    text_cols = [
        "type_mvt",
        "code_couleur",
        "code_cepage",
        "commune",
        "code_departement",
        "departement",
        "cvi",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df

ai_analyzer = AIAnalyzer()

@st.cache_data
def call_climate_ai(zone_id, temp, precip):
    return ai_analyzer.agent_climate_similarity(zone_id, temp, precip)

# Chargement des données
with st.spinner("Chargement des donnees..."):
    df_geo = load_climate_yield_geo()
    df_fusion = load_fusion_analysis()

if df_geo.empty:
    st.warning("Aucune donnee climat_rendement_geo disponible.")
    st.stop()

if df_fusion.empty:
    st.warning("Aucune donnee fusion disponible.")
    st.stop()


# =====================================================
# SIDEBAR - FILTRES
# =====================================================

with st.sidebar:
    st.header("Filtres")
    st.divider()
    with st.expander("Carte des zones pedoclimatiques"):
        if st.button("Afficher la carte", key="show_zone_map_btn"):
            st.image("carte_zones_pedoclimatiques_.png")
    available_zones = sorted([int(z) for z in df_geo["zone"].dropna().unique()])
    selected_zones = st.multiselect(
        "Zones pedoclimatiques",
        available_zones,
        default=available_zones[:3] if len(available_zones) >= 3 else available_zones,
        help="Selectionnez une ou plusieurs zones a analyser"
    )

    available_deps = sorted(df_geo["code_departement"].dropna().astype(str).unique().tolist())
    selected_deps = st.multiselect(
        "Departements",
        available_deps,
        default=available_deps,
        help="Filtrer par departement"
    )

    available_years = sorted([int(y) for y in df_geo["annee"].dropna().unique()])
    
    col_year1, col_year2 = st.columns(2)
    with col_year1:
        year_min = st.number_input(
            "Annee min",
            min_value=int(min(available_years)),
            max_value=int(max(available_years)),
            value=int(min(available_years))
        )
    with col_year2:
        year_max = st.number_input(
            "Annee max",
            min_value=int(min(available_years)),
            max_value=int(max(available_years)),
            value=int(max(available_years))
        )
    
    year_min = year_max - 4
    selected_years = (year_min, year_max)
    available_colors = sorted([c for c in df_fusion["code_couleur"].dropna().unique() if c and c != "nan"])
    selected_colors = st.multiselect(
        "Couleurs",
        available_colors,
        default=available_colors,
        help="Type de vin : BL (Blanc), RG (Rouge), RS (Rose)"
    )

    available_cepages = sorted([c for c in df_fusion["code_cepage"].dropna().unique() if c and c != "nan"])
    selected_cepages = st.multiselect(
        "Cepages",
        available_cepages,
        default=[],
        help="Filtrer par cepage (optionnel)"
    )

    st.divider()

if not selected_zones:
    st.warning("Selectionnez au moins une zone pour commencer l'analyse.")
    st.stop()

# Application des filtres
df_geo_filtered = df_geo[
    (df_geo["zone"].isin(selected_zones))
    & (df_geo["code_departement"].isin(selected_deps))
    & (df_geo["annee"].between(selected_years[0], selected_years[1]))
].copy()

df_fusion_filtered = df_fusion[
    (df_fusion["zone"].isin(selected_zones))
    & (df_fusion["code_departement"].isin(selected_deps))
    & (df_fusion["annee"].between(selected_years[0], selected_years[1]))
].copy()

if selected_colors:
    df_fusion_filtered = df_fusion_filtered[df_fusion_filtered["code_couleur"].isin(selected_colors)].copy()

if selected_cepages:
    df_fusion_filtered = df_fusion_filtered[df_fusion_filtered["code_cepage"].isin(selected_cepages)].copy()

if df_geo_filtered.empty or df_fusion_filtered.empty:
    st.warning("Aucune donnee disponible apres filtrage. Veuillez elargir vos criteres.")
    st.stop()


# =====================================================
# KPI CARDS
# =====================================================

st.header("Indicateurs cles")

col1, col2, col3, col4 = st.columns(4)
with col1:
    rendement_moy = df_geo_filtered["rendement"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(rendement_moy, 0)} hl</div>
        <div class="metric-label">Rendement moyen</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    temp_moy = df_geo_filtered["temp_moyenne"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(temp_moy, 1)} °C</div>
        <div class="metric-label">Temperature moyenne</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    precip_moy = df_geo_filtered["precipitation_total"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(precip_moy, 0)} mm</div>
        <div class="metric-label">Precipitations moyennes</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    reve = df_fusion_filtered[df_fusion_filtered["type_mvt"] == "REVE"].copy()
    volume_total = reve["volume"].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(volume_total, 0)} hl</div>
        <div class="metric-label">Volume total</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# =====================================================
# SCORING INTELLIGENT 
# =====================================================

with st.expander("Scoring intelligent des zones", expanded=True):
    zone_scoring = (
        df_geo_filtered.groupby("zone", as_index=False)
        .agg(
            rendement_moy=("rendement", "mean"),
            rendement_std=("rendement", "std"),
            temp_moy=("temp_moyenne", "mean"),
            precip_moy=("precipitation_total", "mean"),
        )
        .sort_values("zone")
    )

    # Climat de reference = zones les plus productrices
    top_yield_threshold = zone_scoring["rendement_moy"].quantile(0.75) if len(zone_scoring) >= 4 else zone_scoring["rendement_moy"].median()
    top_zone_ref = zone_scoring[zone_scoring["rendement_moy"] >= top_yield_threshold].copy()

    if top_zone_ref.empty:
        ref_temp = zone_scoring["temp_moy"].median()
        ref_precip = zone_scoring["precip_moy"].median()
       
    else:
        ref_temp = top_zone_ref["temp_moy"].median()
        ref_precip = top_zone_ref["precip_moy"].median()

    zone_scoring["score_rendement"] = minmax_scale(zone_scoring["rendement_moy"]) * 100
    zone_scoring["score_stabilite"] = inverse_minmax_scale(zone_scoring["rendement_std"].fillna(zone_scoring["rendement_std"].max())) * 100
    zone_scoring["score_temp_equilibre"] = (
        1 - (
            (zone_scoring["temp_moy"] - ref_temp).abs() /
            max((zone_scoring["temp_moy"] - ref_temp).abs().max(), 1e-9)
        )
    ) * 100

    zone_scoring["score_precip_equilibre"] = (
        1 - (
            (zone_scoring["precip_moy"] - ref_precip).abs() /
            max((zone_scoring["precip_moy"] - ref_precip).abs().max(), 1e-9)
        )
    ) * 100

    for col in [
        "score_temp_equilibre",
        "score_precip_equilibre"
    ]:
        zone_scoring[col] = zone_scoring[col].clip(lower=0, upper=100)

    zone_scoring["score_final"] = (
        0.35 * zone_scoring["score_rendement"]
        + 0.20 * zone_scoring["score_stabilite"]
        + 0.10 * zone_scoring["score_temp_equilibre"]
        + 0.05 * zone_scoring["score_precip_equilibre"]
    ).round(1)

    zone_scoring["classe_final"] = zone_scoring["score_final"].apply(class_from_score)
    zone_color_dict = build_zone_color_dict(zone_scoring["zone"].tolist())
    fig_score = px.bar(
        zone_scoring.sort_values("score_final", ascending=False),
        x="zone",
        y="score_final",
        color="zone",
        color_discrete_map=zone_color_dict,
        text="classe_final",
        title="Classement qualitatif des zones",
        labels={"zone": "Zone", "score_final": "Score global"},
        height=500
    )
    fig_score.update_traces(textposition="outside", textfont_size=14)
    fig_score.update_layout(
        xaxis_type="category",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    st.plotly_chart(fig_score, key="score_bar_chart", use_container_width=True)
    with st.info(""):
        st.markdown("""
        **N.B :** Le score global traduit la performance globale de chaque zone en tenant compte des deux critères suivants :
        - stabilite des rendements pour les 3 couleurs
        - repartition optimale des temperatures annuelles et des precipitations annuelles
        - Les zones sont classees en fonction de leur score global, allant de 0 à 100 :
        - **Classe A** : score global egal à 80 et plus
        - **Classe B** : score global de 65 à 79
        - **Classe C** : score global de 50 à 64
        - **Classe D** : score global de moins de 50
        """)

with st.expander("Cartographie climat-production", expanded=True):
    try:
        geo_zones = load_geojson("zones")

        if geo_zones is None:
            st.error("Impossible de charger le GeoJSON des zones.")
        else:
            map_agg = (
                df_geo_filtered.groupby("zone", as_index=False)
                .agg(
                    rendement=("rendement", "mean"),
                    temp_moyenne=("temp_moyenne", "mean"),
                    precipitation_total=("precipitation_total", "mean")
                )
            )

            commune_zone = df_geo_filtered[["commune", "zone"]].drop_duplicates().copy()
            map_display = commune_zone.merge(map_agg, on="zone", how="inner")
            map_display["zone"] = map_display["zone"].astype(str)

            fig_map = px.choropleth(
                map_display,
                geojson=geo_zones,
                locations="commune",
                featureidkey="properties.code_commune",
                color="rendement",
                color_continuous_scale="YlOrRd",
                title="Rendement par zone",
                hover_data={
                    "zone": True,
                    "rendement": ":.1f",
                    "temp_moyenne": ":.1f",
                    "precipitation_total": ":.0f",
                },
                labels={
                    "zone": "Zone",
                    "rendement": label_of("rendement"),
                },
                height=500
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=50, b=0),
                coloraxis_colorbar=dict(title="Rendement (hl/ha)", thickness=15)
            )
            st.plotly_chart(fig_map, key="climate_map", use_container_width=True)

    except Exception as e:
        st.warning(f"Carte indisponible : {e}")


# =====================================================
# EVOLUTION PAR ZONE
# =====================================================

st.header("Evolution des indicateurs par zone")

# Selecteurs pour l'evolution
col_evol1, col_evol2 = st.columns([1, 1])

with col_evol1:
    evol_zone = st.multiselect(
        "Zones à comparer",
        options=selected_zones,
        default=selected_zones[:min(3, len(selected_zones))] if len(selected_zones) > 1 else selected_zones,
        help="Selectionnez les zones a afficher sur les graphiques",
        label_visibility="collapsed"
    )

with col_evol2:
    evol_indicator = st.selectbox(
        "Indicateur à visualiser",
        options=["rendement"] + CLIMATE_VARS,
        format_func=label_of,
        index=0,
        key="evol_indicator",
        label_visibility="collapsed"
    )

if not evol_zone:
    st.info("Selectionnez au moins une zone pour visualiser l'evolution.")
else:
    # Preparation des donnees d'evolution
    evol_data = df_geo_filtered[df_geo_filtered["zone"].isin(evol_zone)].copy()
    max_year_evol = evol_data["annee"].max()
    min_year_evol = max_year_evol - 4
    evol_data = evol_data[evol_data["annee"].between(min_year_evol, max_year_evol)].copy()
    
    evol_agg = (
        evol_data.groupby(["zone", "annee"], as_index=False)[evol_indicator]
        .mean()
        .sort_values(["zone", "annee"])
    )
    
    # Graphique d'evolution avec Plotly
    zone_colors = build_zone_color_dict(evol_zone)
    
    fig_evolution = go.Figure()
    
    for zone in evol_zone:
        zone_data = evol_agg[evol_agg["zone"] == zone].copy()
        if not zone_data.empty:
            color = zone_colors.get(zone, "#7F7F7F")
            fig_evolution.add_trace(go.Scatter(
                x=zone_data["annee"].astype(int),
                y=zone_data[evol_indicator],
                mode="lines+markers",
                name=f"Zone {zone}",
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"Zone {zone}<br>Annee: %{{x}}<br>{label_of(evol_indicator)}: %{{y:.1f}}<extra></extra>"
            ))
    fig_evolution.update_xaxes(
        tickmode="linear",
        dtick=1
    )
    fig_evolution.update_layout(
        title=f"Evolution de {label_of(evol_indicator)} par zone",
        xaxis_title="Annee",
        yaxis_title=label_of(evol_indicator),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray")
    )
    
    st.plotly_chart(fig_evolution, key="evolution_chart", use_container_width=True)
    
    # Tableau recapitulatif par zone
    with st.expander("Tableau recapitulatif par zone", expanded=False):
        recap = evol_data.groupby("zone")[evol_indicator].agg(["mean", "std", "min", "max"]).round(2).reset_index()
        recap.columns = ["zone", "Moyenne", "Ecart-type", "Minimum", "Maximum"]

        st.dataframe(recap, width="stretch", hide_index=True)


# =====================================================
# HISTOGRAMMES DE DISTRIBUTION
# =====================================================

with st.expander("Distributions des indicateurs", expanded=False):
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        hist_var = st.selectbox(
            "Variable a analyser",
            ["rendement"] + CLIMATE_VARS,
            format_func=label_of,
            key="hist_var_select",
        )
    
    with col_hist2:
        hist_group = st.radio(
            "Grouper par",
            ["Zone", "Couleur"],
            horizontal=True,
            key="hist_group_radio",
        )
    
    group_col = "zone" if hist_group == "Zone" else "code_couleur"
    
    if hist_group == "Zone":
        hist_source = df_geo_filtered[[hist_var, group_col]].dropna().copy()
    else:
        hist_source = df_fusion_filtered[[hist_var, group_col]].dropna().copy()
    
    if not hist_source.empty:
        fig_hist = go.Figure()
        
        groups = hist_source[group_col].dropna().unique()
        
        for grp in groups:
            data = hist_source[hist_source[group_col] == grp][hist_var].dropna()
            if not data.empty:
                if hist_group == "Zone":
                    color = ZONE_COLOR_MAP.get(str(int(grp)), "#7F7F7F")
                    name = f"Zone {grp}"
                else:
                    color = WINE_COLOR_MAP.get(str(grp), "#7F7F7F")
                    name = f"{grp}"
                
                fig_hist.add_trace(go.Histogram(
                    x=data,
                    name=name,
                    marker_color=color,
                    opacity=0.6,
                    nbinsx=20,
                    hovertemplate=f"{name}<br>Valeur: %{{x:.1f}}<br>Frequence: %{{y}}<extra></extra>"
                ))
        
        fig_hist.update_layout(
            title=f"Distribution du {label_of(hist_var)}",
            xaxis_title=label_of(hist_var),
            yaxis_title="Frequence",
            barmode="overlay",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray")
        )
        
        st.plotly_chart(fig_hist, key="histogram_chart", use_container_width=True)
    else:
        st.info("Donnees insuffisantes pour l'histogramme.")


# =====================================================
# INFLUENCE POSITIVE / NEGATIVE SUR RENDEMENT ET VOLUME
# =====================================================
recent_max = int(df_fusion_filtered["annee"].dropna().max())
recent_min = recent_max - 4
fusion_scope = df_fusion_filtered[df_fusion_filtered["annee"].between(recent_min, recent_max)].copy()
geo_scope = df_geo_filtered[df_geo_filtered["annee"].between(recent_min, recent_max)].copy()
# Correlations rendement
corr_rows = []
for var in CLIMATE_VARS:
    if var in geo_scope.columns:
        corr_rows.append({
            "indicateur": var,
            "corr_rendement": weighted_corr(geo_scope, var, "rendement"),
        })

corr_table = pd.DataFrame(corr_rows)
corr_table["effet_rendement"] = np.where(corr_table["corr_rendement"] >= 0, "positif (+)", "negatif (-)")

# Correlations volume
reve_scope = fusion_scope[fusion_scope["type_mvt"] == "REVE"].copy()
volume_corr_rows = []
for var in ["temp_moyenne", "precipitation_total", "Hot_D", "Very_Hot_D", "Huglin_Index", "Climatic_Dryness_Index"]:
    if var in reve_scope.columns:
        volume_corr_rows.append({
            "indicateur": var,
            "corr_volume": weighted_corr(reve_scope, var, "volume"),
        })

volume_corr_table = pd.DataFrame(volume_corr_rows)
volume_corr_table["effet_volume"] = np.where(volume_corr_table["corr_volume"] >= 0, "positif (+)", "negatif (-)")

col_corr1, col_corr2 = st.columns(2)


# =====================================================
# ANALYSE PAR COULEUR ET PAR CEPAGE
# =====================================================

with st.expander("Analyse par couleur et par cepage", expanded=False):
    subtab1, subtab2 = st.tabs(["Analyse par couleur", "Analyse par cepage"])
    
    with subtab1:
        color_scope = fusion_scope[fusion_scope["type_mvt"] == "DECR"].copy()
        
        if color_scope.empty or color_scope["code_couleur"].dropna().empty:
            st.info("Aucune donnee couleur disponible.")
        else:
            color_zone = (
                color_scope.groupby(["zone", "code_couleur"], as_index=False)
                .agg(rendement=("rendement", "mean"))
            )
            color_zone["rendement"] = color_zone["rendement"].round(0)
            fig_color = px.bar(
                color_zone,
                x="zone",
                y="rendement",
                color="code_couleur",
                barmode="group",
                color_discrete_map=WINE_COLOR_MAP,
                title="Rendement moyen par zone et par couleur",
                labels={"zone": "Zone", "rendement": "Rendement (hl/ha)", "code_couleur": "Couleur"},
                height=500
            )
            st.plotly_chart(fig_color, key="color_bar_chart", use_container_width=True)
            
            color_reve = fusion_scope[fusion_scope["type_mvt"] == "REVE"].copy()
            if not color_reve.empty:
                color_ratio = (
                    color_reve.groupby("code_couleur", as_index=False)["volume"]
                    .sum()
                    .sort_values("volume", ascending=False)
                )
                total_volume = color_ratio["volume"].sum()
                color_ratio["ratio_volume_pct"] = np.where(
                    total_volume > 0,
                    100 * color_ratio["volume"] / total_volume,
                    np.nan,
                )
                color_ratio["ratio_volume_pct"] = color_ratio["ratio_volume_pct"].round(1)
                fig_color_ratio = px.pie(
                    color_ratio,
                    values="ratio_volume_pct",
                    names="code_couleur",
                    color="code_couleur",
                    color_discrete_map=WINE_COLOR_MAP,
                    title="Poids relatif de chaque couleur dans le volume total",
                    hole=0.4,
                    height=500
                )
                st.plotly_chart(fig_color_ratio, key="color_ratio_chart", use_container_width=True)
    
    with subtab2:
        cepage_scope = fusion_scope[fusion_scope["type_mvt"] == "REVE"].copy()
        
        if cepage_scope.empty or cepage_scope["code_cepage"].dropna().empty:
            st.info("Aucune donnee cepage disponible.")
        else:
            top_cepages = (
                cepage_scope.groupby("code_cepage", as_index=False)["volume"]
                .sum()
                .sort_values("volume", ascending=False)
                .head(12)["code_cepage"]
                .tolist()
            )
            
            cepage_scope = cepage_scope[cepage_scope["code_cepage"].isin(top_cepages)].copy()
            
            cepage_zone_year = (
                cepage_scope.groupby(["annee", "zone", "code_cepage"], as_index=False)["volume"]
                .sum()
            )
            cepage_zone_year["volume"] = cepage_zone_year["volume"].round(0)
            fig_cepage = px.bar(
                cepage_zone_year,
                x="annee",
                y="volume",
                color="code_cepage",
                barmode="group",
                facet_row="zone" if len(selected_zones) <= 4 else None,
                title="Evolution des cepages par zone",
                labels={"volume": "Volume (hl)", "annee": "Annee", "code_cepage": "Cepage"},
                height=500
            )
            st.plotly_chart(fig_cepage, key="cepage_evolution_chart", use_container_width=True)


# =====================================================
# TABLEAU DE SYNTHESE
# =====================================================

with st.expander("Tableau de synthese par zone et annee", expanded=False):
    table_zone_year = (
        df_geo_filtered.groupby(["zone", "annee"], as_index=False)
        .agg(
            rendement=("rendement", "mean"),
            temp_moyenne=("temp_moyenne", "mean"),
            precipitation_total=("precipitation_total", "mean")
        ).round(1)
        .sort_values(["zone", "annee"])
    )
    table_zone_year_display = table_zone_year.rename(columns=DISPLAY_LABELS)
    st.dataframe(table_zone_year_display, width="stretch", hide_index=True, row_height=20, height=750)


# =====================================================
# NARRATION AUTOMATIQUE
# =====================================================

with st.expander("Analyse automatique", expanded=False):
    if not zone_scoring.empty:
        top_zone = zone_scoring.sort_values("score_final", ascending=False).iloc[0]
        bottom_zone = zone_scoring.sort_values("score_final", ascending=True).iloc[0]
        most_stable = zone_scoring.sort_values("rendement_std", ascending=True).iloc[0]
        best_yield = zone_scoring.sort_values("rendement_moy", ascending=False).iloc[0]
        
        # Meilleure couleur
        best_color_text = "Information indisponible"
        color_perf = (
            fusion_scope[fusion_scope["type_mvt"] == "DECR"]
            .groupby("code_couleur", as_index=False)["rendement"]
            .mean()
            .sort_values("rendement", ascending=False)
        )
        if not color_perf.empty:
            row = color_perf.iloc[0]
            best_color_text = f"La couleur la plus performante est le {WINE_CORRESPONDANCE.get(row['code_couleur'], row['code_couleur'])} avec un rendement moyen de {row['rendement']:.1f} hl/ha."
        
        # Meilleur cepage
        best_cepage_text = "Information indisponible"
        cepage_perf = (
            fusion_scope[fusion_scope["type_mvt"] == "REVE"]
            .groupby("code_cepage", as_index=False)["volume"]
            .sum()
            .sort_values("volume", ascending=False)
        )
        if not cepage_perf.empty:
            row = cepage_perf.iloc[0]
            best_cepage_text = f"Le cepage le plus present est le {row['code_cepage']} avec {row['volume']:.0f} hl."
        
        narrative = f"""
        ### Synthese de l'analyse
        
        Sur la periode analysee :

        - **Zone la mieux classee** : Zone {int(top_zone['zone'])} avec un score global de {top_zone['score_final']:.1f} (classe {top_zone['classe_final']})
        - **Zone la plus productive** : Zone {int(best_yield['zone'])} avec {best_yield['rendement_moy']:.0f} hl/ha
        - **Zone la plus stable** : Zone {int(most_stable['zone'])} (variation de {most_stable['rendement_std']:.0f})
        {best_color_text}
        {best_cepage_text}
        """
        st.markdown(narrative)

with st.expander("Analogie climatique et viticole", expanded=False):
    zone_focused = st.selectbox("Choisir une zone", selected_zones, label_visibility="collapsed")
    data_zone_answer = df_geo_filtered[df_geo_filtered["zone"] == zone_focused]
    if not data_zone_answer.empty:
        temperature_moyenne = data_zone_answer["temp_moyenne"].mean()
        precipitation_total = data_zone_answer["precipitation_total"].mean()
        with st.spinner("Analyse analogie climatique et viticole en cours..."):
            st.markdown(f"**Analogie climatique et viticole pour la zone {zone_focused} :**")
            resul_analogy_climate_wine = call_climate_ai(
                zone_focused,
                temperature_moyenne,
                precipitation_total
            )
            st.markdown(resul_analogy_climate_wine)

st.markdown("---")
st.caption(f"Analyse mise a jour le {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")


gc.collect()