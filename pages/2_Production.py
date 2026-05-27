# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import base64
import gc
import matplotlib.ticker as mticker
import streamlit.components.v1 as components
import matplotlib.pyplot as plt



# Configuration de la page
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
<div class="main-header">
    <h1>Observatoire Viticole</h1>
    <h2>Pays d'Oc IGP</h2>
    <p>Analyse des rendements et volumes par zone pedoclimatique, couleur et cepage</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# IMPORTS DES MODULES
# =====================================================
from config.constants import (
    COLOR_MAP, ZONE_COLOR_MAP, ZONE_LABELS,
    DEPARTEMENTS, DEPT_COLOR_MAP
)
from modules.data_loader import load_data, load_geojson, apply_filters
from modules.utils import (
    zone_sort_key, zone_order_from_series, complete_years,
    format_number
)
from modules.visualization import (
    create_rendement_chart, create_volume_chart
)
from modules.prediction import run_prediction
from modules.ai_engine import AIAnalyzer
from modules.map_utils import prepare_map_data, display_map_with_legend, compare_two_maps

# =====================================================
# INITIALISATION
# =====================================================

@st.cache_data
def initialize_data():
    """Charge les donnees avec cache"""
    return load_data()

# Chargement des donnees
with st.spinner("Chargement des donnees viticoles..."):
    df_all = initialize_data()

if df_all.empty:
    st.error("Impossible de charger les donnees. Veuillez verifier la base.")
    st.stop()

# Initialisation de l'IA
ai_analyzer = AIAnalyzer()

# =====================================================
# FONCTIONS AIDEES
# =====================================================

def get_last_5_years(df_data):
    """Recupere les 5 dernieres annees de donnees"""
    max_year = df_data["annee"].max()
    min_year = max_year - 4
    return df_data[df_data["annee"].between(min_year, max_year)]

def get_zones_1_7(df):
    """Filtre pour garder uniquement les zones 1 a 7"""
    if "Zone" in df.columns:
        return df[~df["Zone"].isin(["0", "None", "nan", 0, "8", 8])].copy()
    return df

def create_cepage_map(df, df_all, indicator, level, year, couleur, cepage=None):
    """Cree une carte pour les cepages"""
    indicator_key = {"Volume": "volume", "Rendement": "rendement", "Surface": "surface"}[indicator]
    mvt_key = "DECR" if indicator_key == "rendement" else "REVE"
    
    # Filtrer les donnees
    df_filtered = df[(df["type_mvt"] == mvt_key)]
    
    if year:
        df_filtered = df_filtered[df_filtered["annee"] == year]
    
    if couleur:
        df_filtered = df_filtered[df_filtered["code_couleur"] == couleur]
    
    if cepage:
        df_filtered = df_filtered[df_filtered["code_cepage"] == cepage]
    
    if df_filtered.empty:
        return None
    
    # Aggregation par niveau
    if level == "Departement":
        geo = load_geojson("departements")
        if geo is None:
            return None
        map_display = df_filtered.groupby("code_departement")[indicator_key].sum().reset_index()
        locations = "code_departement"
        feature_key = "properties.dep"
        title_level = "departement"
    else:
        geo = load_geojson("zones")
        if geo is None:
            return None
        zone_agg = df_filtered.groupby("Zone")[indicator_key].sum().reset_index()
        map_display = df_all[["commune", "Zone"]].drop_duplicates().merge(zone_agg, on="Zone", how="inner")
        map_display = get_zones_1_7(map_display)
        locations = "commune"
        feature_key = "properties.code_commune"
        title_level = "zone"
    
    if map_display.empty:
        return None
    
    return {
        'geo': geo,
        'map_display': map_display,
        'locations': locations,
        'feature_key': feature_key,
        'var': indicator_key,
        'scale': "Blues" if indicator_key == "volume" else "RdYlGn" if indicator_key == "rendement" else "Greens",
        'indicator': indicator,
        'level': level,
        'unit': "hl" if indicator_key == "volume" else "hl/ha" if indicator_key == "rendement" else "ha",
        'title_level': title_level
    }

# =====================================================
# SIDEBAR - FILTRES
# =====================================================
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
    st.header("Filtres")
    st.divider()
    # POPUP carte des zones
    show_ref_map = False
    if st.button("Carte des zones pédoclimatiques"):
        show_carte_inrae_clusters()
    # Filtres principaux
    departements = st.multiselect(
        "Departements",
        DEPARTEMENTS,
        default=DEPARTEMENTS,
        key="sb_dep",
        help="Selectionnez les departements a analyser"
    )
    
    # Zones 1 a 7 uniquement
    all_zones = [z for z in zone_order_from_series(df_all["Zone"]) if z not in ["0", 0, "8", 8]]
    zones = st.multiselect(
        "Zones pedoclimatiques",
        all_zones,
        default=all_zones,
        key="sb_zone",
        help="Filtrer par zones pedoclimatiques (1 a 7)"
    )
    couleurs = st.multiselect(
        "Couleurs de vin",
        ["BL", "RG", "RS"],
        default=["BL", "RG", "RS"],
        key="sb_coul",
        help=None
    )
    with st.expander("Aide couleurs"):
        st.write("BL : Blanc")
        st.write("RG : Rouge")
        st.write("RS : Rose")
    with st.expander("Aide cepages (par ordre alphabétique)"):
        st.write("1705 : Caladoc")
        st.write("3039 : Alvarinho")
        st.write("ALIC : Alicante H. Bouschet")
        st.write("BOUR : Bourboulenc")
        st.write("CAFR : Cabernet Franc")
        st.write("CARI : Carignan")
        st.write("CASA : Cabernet Sauvignon")
        st.write("CHAR : Chardonnay")
        st.write("CHEN : Chenin")
        st.write("CINS : Cinsault")
        st.write("COLO : Colombard")
        st.write("COT : Malbec")
        st.write("GEBL : Generique Blanc")
        st.write("GERG : Generique Rouge")
        st.write("GERS : Generique Rose")
        st.write("GEWU : Gewurztraminer")
        st.write("GREB : Grenache Blanc")
        st.write("GREG : Grenache Gris")
        st.write("GREN : Grenache Noir")
        st.write("MARS : Marsanne")
        st.write("MASE : Marselan")
        st.write("MAUZ : Mauzac")
        st.write("MERL : Merlot")
        st.write("MOUR : Mourvedre")
        st.write("MUPG : Muscat Petits Grains blanc")
        st.write("MUSC : Muscat d'Alexandrie")
        st.write("PIBL : Pinot Blanc")
        st.write("PIGR : Pinot Gris")
        st.write("PINO : Pinot Noir")
        st.write("RIES : Riesling")
        st.write("ROUS : Roussanne")
        st.write("SAGR : Sauvignon Gris")
        st.write("SAUV : Sauvignon Blanc")
        st.write("SYRA : Syrah")
        st.write("TERB : Terret blanc")
        st.write("VERD : Petit Verdot")
        st.write("VERM : Rolle (Vermentino)")
        st.write("VIOG : Viognier")
    annee_min = int(df_all["annee"].dropna().min())
    annee_max = int(df_all["annee"].dropna().max())
    
    annees = st.slider(
        "Periode d'analyse",
        annee_min, annee_max, (annee_min, annee_max),
        key="sb_years",
        help="Selectionnez la plage d'annees"
    )
    
    st.divider()
    
    # Option pour la moyenne glissante
    use_moving_average = st.checkbox("Moyenne mobile sur 5 ans", value=False)

# Application des filtres
df = apply_filters(df_all, departements, zones, couleurs, annees)

if df.empty:
    st.error("Aucune donnee ne correspond aux filtres selectionnes. Veuillez elargir vos criteres.")
    st.stop()

YEAR_MIN = int(df["annee"].dropna().min())
YEAR_MAX = int(df["annee"].dropna().max())

# =====================================================
# CARTE DE REFERENCE DES ZONES
# =====================================================

if show_ref_map:
    st.header("Carte de reference - Zones pedoclimatiques")
    
    try:
        geo_zones = load_geojson("zones")
        
        if geo_zones is None:
            st.error("Impossible de charger la carte des zones.")
        else:
            ref = df_all[["commune", "Zone"]].drop_duplicates().copy()
            ref["Zone"] = ref["Zone"].astype(str)
            ref["Nom"] = ref["Zone"].map(ZONE_LABELS).fillna(ref["Zone"].apply(lambda z: f"Zone {z}"))
            ref = get_zones_1_7(ref)
            
            fig_ref = px.choropleth(
                ref,
                geojson=geo_zones,
                locations="commune",
                featureidkey="properties.code_commune",
                color="Zone",
                color_discrete_map=ZONE_COLOR_MAP,
                category_orders={"Zone": [str(i) for i in range(1, 8)]},
                hover_name="Nom",
                title="Zones pedoclimatiques en Pays d'Oc (zones 1 a 7)",
                height=500
            )
            
            fig_ref.update_geos(fitbounds="locations", visible=False)
            fig_ref.update_layout(template="plotly_white", legend_title_text="Zone")
            
            st.plotly_chart(fig_ref, key="ref_map", width="stretch")
            
            with st.expander("Description des zones"):
                cols = st.columns(2)
                zone_list = [str(i) for i in range(1, 8)]
                for i, zone in enumerate(zone_list):
                    color = ZONE_COLOR_MAP.get(zone, "#808080")
                    desc = ZONE_LABELS.get(zone, f"Zone {zone}")
                    with cols[i % 2]:
                        st.markdown(
                            f"<span style='color:{color};font-weight:bold'>■</span> **{desc}**",
                            unsafe_allow_html=True
                        )
    except Exception as e:
        st.error(f"Erreur lors du chargement de la carte: {e}")
    
    st.stop()

# =====================================================
# METRIQUES GLOBALES
# =====================================================
WINE_CORRESPONDANCE = {
    "BL": "Blanc",
    "RG": "Rouge",
    "RS": "Rose",
}
st.header("Indicateurs cles")
col1, col2, col3, col4 = st.columns(4)

with col1:
    rendement_moy = df[df["type_mvt"] == "DECR"]["rendement"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(rendement_moy, 0)} hl/ha</div>
        <div class="metric-label">Rendement moyen</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    volume_total = df[df["type_mvt"] == "REVE"]["volume"].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(volume_total, 0)} hl</div>
        <div class="metric-label">Volume total</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    surface_totale = df[df["type_mvt"] == "DECR"]["surface"].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{format_number(surface_totale, 0)} ha</div>
        <div class="metric-label">Surface totale</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    n_communes = df["commune"].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_communes}</div>
        <div class="metric-label">Communes analysees</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# TABS PRINCIPAUX
# =====================================================

tab_rdt, tab_vol, tab_pred, tab_map, tab_quant = st.tabs([
    "Rendement", "Volume", "Prediction",
    "Cartographie", "Analyse quantitative"
])

# =====================================================
# RENDEMENT
# =====================================================

with tab_rdt:
    st.header("Analyse des rendements")
    
    decr = df[df["type_mvt"] == "DECR"].copy()
    
    # Filtrage des valeurs aberrantes
    decr_bl = decr[(decr["code_couleur"] == "BL") & (decr["rendement"].between(5, 90))]
    decr_rg = decr[(decr["code_couleur"] == "RG") & (decr["rendement"].between(5, 90))]
    decr_rs = decr[(decr["code_couleur"] == "RS") & (decr["rendement"].between(5, 100))]
    decr = pd.concat([decr_bl, decr_rg, decr_rs], ignore_index=True)
    
    if decr.empty:
        st.warning("Aucune donnee de rendement conforme aux plafonds avec les filtres actuels")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            mode = st.radio(
                "Comparer par",
                ["Couleur", "Departement", "Zone"],
                horizontal=True,
                key="rdt_mode",
                label_visibility="collapsed"
            )
        
        col_map = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",
            "Zone": "Zone"
        }[mode]
        
        available_options = sorted(
            decr[col_map].dropna().unique(),
            key=(zone_sort_key if mode == "Zone" else None)
        )
        
        with col2:
            selections = st.multiselect(
                "Selectionner les elements a comparer",
                available_options,
                default=available_options[:min(3, len(available_options))],
                key="rdt_sel",
                label_visibility="collapsed"
            )
        
        
        if not selections:
            st.info("Veuillez selectionner au moins un element a analyser")
        else:
            data = decr[decr[col_map].isin(selections)].copy()
            data["code_departement"] = data["code_departement"].astype(str)
            data = data[data["code_departement"].isin(["11", "30", "34", "66"])]
            # Verification des plafonds
            with st.expander("Verification des plafonds reglementaires", expanded=False):
                plafond_check = data.groupby("code_couleur")["rendement"].agg(['max', 'count']).round(2)
                plafond_check['plafond'] = plafond_check.index.map(lambda x: 90 if x in ['BL', 'RG'] else 100)
                plafond_check['conforme'] = plafond_check['max'] <= plafond_check['plafond']
                plafond_check = plafond_check.rename(columns={'max': 'Rendement max'})
                st.dataframe(plafond_check[['Rendement max', 'plafond', 'conforme']], width="stretch")
                gc.collect()
            # Graphique d'evolution
            evol = data.groupby(["annee", col_map])["rendement"].mean().reset_index()
            evol_full = complete_years(evol, col_map, "rendement", "mean", YEAR_MIN, YEAR_MAX)
            evol_full["annee"] = evol_full["annee"].astype(str)
            if use_moving_average:
                evol = data.groupby(["annee", col_map])["rendement"].mean().reset_index()
                evol_full = complete_years(evol, col_map, "rendement", "mean", YEAR_MAX-4, YEAR_MAX)
            evol_full["rendement"] = evol_full["rendement"].round(0)
            evol_full["annee"] = evol_full["annee"].astype(str)
            # Utilisation des couleurs selon le mode
            if mode == "Couleur":
                fig = px.line(
                    evol_full,
                    x="annee",
                    y="rendement",
                    color=col_map,
                    color_discrete_map=COLOR_MAP,
                    markers=True,
                    title="Evolution des rendements"
                )
            elif mode == "Zone":
                fig = px.line(
                    evol_full,
                    x="annee",
                    y="rendement",
                    color=col_map,
                    color_discrete_map=ZONE_COLOR_MAP,
                    markers=True,
                    title="Evolution des rendements par zone"
                )
            elif mode == "Departement":
                fig = px.line(
                    evol_full,
                    x="annee",
                    y="rendement",
                    color=col_map,
                    color_discrete_map=DEPT_COLOR_MAP,
                    markers=True,
                    title="Evolution des rendements par departement"
                )
            else:
                fig = create_rendement_chart(evol_full, col_map, mode)
            
            fig.update_layout(
                yaxis=dict(range=[0, 110], dtick=10),
                template="plotly_white",
                hovermode="x unified",
                height=500
            )
            fig.update_xaxes(type="category")
            fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Plafond BL/RG (90 hl/ha)")
            fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Plafond RS (100 hl/ha)")
            
            st.plotly_chart(fig, key="rendement_chart", width="stretch")
            
            # Histogrammes de comparaison
            st.subheader("Comparaison des rendements")
            
            # Histogramme rendement par departement et couleur
            st.markdown("#### Rendement par departement et couleur")
            if use_moving_average:
                last_year = data["annee"].max()
                data = data[data["annee"] >= last_year - 4]
            dept_color_rdt = data.groupby(["code_departement", "code_couleur"])["rendement"].agg(["mean", "std"]).reset_index()
            dept_color_rdt["code_departement"] = pd.Categorical(
                dept_color_rdt["code_departement"], 
                categories=["11", "30", "34", "66"]
            )
            dept_color_rdt["mean"] = dept_color_rdt["mean"].round(0)
            dept_color_rdt["std"] = dept_color_rdt["std"].round(0)
            fig_dept_color = px.bar(
                dept_color_rdt,
                x="code_departement",
                y="mean",
                color="code_couleur",
                barmode="group",
                error_y="std",
                text_auto=".0f",
                color_discrete_map=COLOR_MAP,
                title="Rendement moyen par departement et couleur",
                labels={"code_departement": "Departement", "mean": "Rendement (hl/ha)", "code_couleur": "Couleur"}
            )
            fig_dept_color.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
                insidetextanchor="start"
            )
            fig_dept_color.update_layout(xaxis_type="category")
            st.plotly_chart(fig_dept_color, key="dept_color_rdt", width="stretch")
            
            # Histogramme rendement par zone et couleur
            st.markdown("#### Rendement par zone et couleur")
            if use_moving_average:
                last_year = data["annee"].max()
                data = data[data["annee"] >= last_year - 4]
            zone_color_rdt = data.groupby(["Zone", "code_couleur"])["rendement"].agg(["mean", "std"]).reset_index()
            
            zone_color_rdt = get_zones_1_7(zone_color_rdt)
            zone_color_rdt["mean"] = zone_color_rdt["mean"].round(0)
            zone_color_rdt["std"] = zone_color_rdt["std"].round(0)
            fig_zone_color = px.bar(
                zone_color_rdt,
                x="Zone",
                y="mean",
                color="code_couleur",
                text_auto=".0f",
                barmode="group",
                error_y="std",
                color_discrete_map=COLOR_MAP,
                title="Rendement moyen par zone et couleur",
                labels={"Zone": "Zone", "mean": "Rendement (hl/ha)", "code_couleur": "Couleur"}
            )
            fig_zone_color.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
                insidetextanchor="start"
            )
            st.plotly_chart(fig_zone_color, key="zone_color_rdt", width="stretch")
            
            # Analyse IA
            with st.expander("Analyse IA du rendement", expanded=False):
                with st.spinner("Analyse en cours..."):
                    if use_moving_average:
                        last_year = data["annee"].max()
                        data = data[data["annee"] >= last_year - 4]
                    ai_analysis = ai_analyzer.analyze_rendement(data, mode, selections)
                    st.markdown(ai_analysis['natural_analysis'])
            
            # Statistiques descriptives
            with st.expander("Statistiques descriptives", expanded=False):
                if use_moving_average:
                    last_year = data["annee"].max()
                    data = data[data["annee"] >= last_year - 4]
                stats_df = data.groupby(col_map)["rendement"].describe(percentiles=[.25, .5, .75]).round(1)
                stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                stats_df['std'] = stats_df['std'].round(0)
                stats_df.columns = ['Occurrence', 'Moyenne', 'Ecart-type', 'Valeur minimale', '1er quartile', 'Mediane', '3ème quartile', 'Valeur maximale']
                st.dataframe(stats_df, width="stretch")

# =====================================================
# VOLUME
# =====================================================

with tab_vol:
    st.header("Analyse des volumes")
    reve = df[(df["type_mvt"] == "REVE") & (df["volume"] > 0)].copy()
    if reve.empty:
        st.warning("Aucune donnee de volume disponible avec les filtres actuels")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            mode = st.radio(
                "Comparer par",
                ["Couleur", "Departement", "Zone"],
                horizontal=True,
                key="vol_mode",
                label_visibility="collapsed"
            )
        
        col_map = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",
            "Zone": "Zone"
        }[mode]
        
        available_options = sorted(
            reve[col_map].dropna().unique(),
            key=(zone_sort_key if mode == "Zone" else None)
        )
        
        with col2:
            selections = st.multiselect(
                "Selectionner les elements a comparer",
                available_options,
                default=available_options[:min(3, len(available_options))],
                key="vol_sel",
                label_visibility="collapsed"
            )
        
        if not selections:
            st.info("Veuillez selectionner au moins un element a analyser")
        else:
            data = reve[reve[col_map].isin(selections)].copy()
            data["code_departement"] = data["code_departement"].astype(str)
            data = data[data["code_departement"].isin(["11", "30", "34", "66"])]
            evol = data.groupby(["annee", col_map])["volume"].sum().reset_index()
            evol_full = complete_years(evol, col_map, "volume", "sum", YEAR_MIN, YEAR_MAX)
            evol_full["annee"] = evol_full["annee"].astype(str)
            if use_moving_average:
                evol = data.groupby(["annee", col_map])["volume"].sum().reset_index()
                evol_full = complete_years(evol, col_map, "volume", "sum", YEAR_MAX-4, YEAR_MAX)
            # Utilisation des couleurs selon le mode
            if mode == "Couleur":
                fig = px.line(
                    evol_full,
                    x="annee",
                    y="volume",
                    color=col_map,
                    color_discrete_map=COLOR_MAP,
                    markers=True,
                    title="Evolution des volumes"
                )
                fig.update_layout(
                    separators=". "
                )

                fig.update_yaxes(
                    tickformat=",.0f"
                )
            elif mode == "Zone":
                fig = px.line(
                    evol_full,
                    x="annee",
                    y="volume",
                    color=col_map,
                    color_discrete_map=ZONE_COLOR_MAP,
                    markers=True,
                    title="Evolution des volumes par zone"
                )
                fig.update_layout(
                    separators=". "
                )

                fig.update_yaxes(
                    tickformat=",.0f"
                )
            elif mode == "Departement":
                fig = px.line(
                    evol_full,
                    x="annee",
                    y="volume",
                    color=col_map,
                    color_discrete_map=DEPT_COLOR_MAP,
                    markers=True,
                    title="Evolution des volumes par departement"
                )
                fig.update_layout(
                    separators=". "
                )

                fig.update_yaxes(
                    tickformat=",.0f"
                )
            else:
                fig = create_volume_chart(evol_full, col_map, mode)
            
            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                yaxis_title="Volume (hl)",
                height=500
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, key="volume_chart", width="stretch")
            
            # Histogrammes de comparaison
            st.subheader("Comparaison des volumes")
            
            # Volume par departement et couleur
            st.markdown("#### Volume par departement et couleur")
            if use_moving_average:
                last_year = data["annee"].max()
                data = data[data["annee"] >= last_year - 4]
            dept_color_vol = (
                data.groupby(
                    ["annee", "code_departement", "code_couleur"]
                )["volume"]
                .sum()
                .reset_index()
            )
            dept_color_vol = (
                dept_color_vol.groupby(
                    ["code_departement", "code_couleur"]
                )["volume"]
                .mean()
                .reset_index()
            )
            color_order = (
                dept_color_vol.groupby(["code_departement", "code_couleur"])["volume"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            dept_color_vol["code_departement"] = pd.Categorical(
                dept_color_vol["code_departement"], 
                categories=["11", "30", "34", "66"]
            )
            fig_dept_color_vol = px.bar(
                dept_color_vol,
                x="code_departement",
                y="volume",
                color="code_couleur",
                barmode="stack",
                color_discrete_map=COLOR_MAP,
                text_auto=".0f",
                category_orders={"code_couleur": color_order},
                title="Volume total par departement et couleur",
                labels={"code_departement": "Departement", "volume": "Volume (hl)", "code_couleur": "Couleur"}
            )
            fig_dept_color_vol.update_layout(xaxis_type="category")
            fig_dept_color_vol.update_layout(
                separators=". "
            )
            fig_dept_color_vol.update_yaxes(
                tickformat=",.0f"
            )
            fig_dept_color_vol.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
            )
            fig_dept_color_vol.update_layout(
                legend_traceorder="normal"
            )
            st.plotly_chart(fig_dept_color_vol, key="dept_color_vol", width="stretch")
            # Volume par zone et couleur
            st.markdown("#### Volume par zone et couleur")
            if use_moving_average:
                last_year = data["annee"].max()
                data = data[data["annee"] >= last_year - 4]
            zone_color_vol = data.groupby(["annee", "Zone", "code_couleur"])["volume"].sum().reset_index()
            zone_color_vol = zone_color_vol.groupby(["Zone", "code_couleur"])["volume"].mean().reset_index()
            zone_color_vol = get_zones_1_7(zone_color_vol)
            color_order = (
                zone_color_vol.groupby(["Zone", "code_couleur"])["volume"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            fig_zone_color_vol = px.bar(
                zone_color_vol,
                x="Zone",
                y="volume",
                color="code_couleur",
                barmode="stack",
                color_discrete_map=COLOR_MAP,
                text_auto=".0f",
                title="Volume total par zone et couleur",
                category_orders={"code_couleur": color_order},
                labels={"Zone": "Zone", "volume": "Volume (hl)", "code_couleur": "Couleur"}
            )
            fig_zone_color_vol.update_layout(
                separators=". "
            )
            fig_zone_color_vol.update_yaxes(
                tickformat=",.0f"
            )
            fig_zone_color_vol.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
            )
            fig_zone_color_vol.update_layout(
                legend_traceorder="normal"
            )
            st.plotly_chart(fig_zone_color_vol, key="zone_color_vol", width="stretch")
            
            # Volume par zone et cepage (top 10 cepages)
            st.markdown("#### Volume par zone et cepage (Top 10 cepages)")
            top10_cepages = data.groupby("code_cepage")["volume"].sum().nlargest(10).index.tolist()
            zone_cepage_vol = data[data["code_cepage"].isin(top10_cepages)].groupby(["annee", "Zone", "code_cepage"])["volume"].sum().reset_index()
            zone_cepage_vol = zone_cepage_vol.groupby(["Zone", "code_cepage"])["volume"].mean().reset_index()
            color_order = (
                zone_cepage_vol.groupby(["Zone", "code_cepage"])["volume"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            zone_cepage_vol = get_zones_1_7(zone_cepage_vol)
            fig_zone_cepage = px.bar(
                zone_cepage_vol,
                x="Zone",
                y="volume",
                color="code_cepage",
                barmode="stack",
                text_auto=".0f",
                title="Volume par zone et cepage (Top 10)",
                category_orders={"code_couleur": color_order},
                labels={"Zone": "Zone", "volume": "Volume (hl)", "code_cepage": "Cepage"}
            )
            fig_zone_cepage.update_layout(
                separators=". "
            )
            fig_zone_cepage.update_yaxes(
                tickformat=",.0f"
            )
            fig_zone_cepage.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
            )
            fig_zone_cepage.update_layout(
                legend_traceorder="normal"
            )
            st.plotly_chart(fig_zone_cepage, key="zone_cepage_vol", width="stretch")
            # Volume par departement et cepage
            st.markdown("#### Volume par departement et cepage (Top 10 cepages)")
            dept_cepage_vol = data[data["code_cepage"].isin(top10_cepages)].groupby(["annee","code_departement", "code_cepage"])["volume"].sum().reset_index()
            dept_cepage_vol = (
                dept_cepage_vol.groupby(
                    ["code_departement", "code_cepage"]
                )["volume"]
                .mean()
                .reset_index()
            )
            color_order = (
                dept_cepage_vol.groupby(["code_departement", "code_cepage"])["volume"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            fig_dept_cepage = px.bar(
                dept_cepage_vol,
                x="code_departement",
                y="volume",
                color="code_cepage",
                barmode="stack",
                text_auto=".0f",
                category_orders={"code_couleur":color_order},
                title="Volume par departement et cepage (Top 10)",
                labels={"code_departement": "Departement", "volume": "Volume (hl)", "code_cepage": "Cepage"}
            )
            fig_dept_cepage.update_layout(xaxis_type="category")
            fig_dept_cepage.update_layout(
                separators=". "
            )
            fig_dept_cepage.update_yaxes(
                tickformat=",.0f"
            )
            fig_dept_cepage.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
            )
            fig_dept_cepage.update_layout(
                legend_traceorder="normal"
            )
            st.plotly_chart(fig_dept_cepage, key="dept_cepage_vol", width="stretch")
    
    # Top 20 cepages des 5 dernieres annees
    if use_moving_average:
        if not reve.empty:
            st.divider()
            st.subheader("Top 20 cepages - 5 dernieres annees")
            last_5_years = get_last_5_years(reve)
            top20_last5 = (
                last_5_years.groupby(["annee", "code_cepage"])["volume"]
                .sum()
                .reset_index()
            )
            top20_last5 = (
                top20_last5.groupby("code_cepage")["volume"]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )
            color_order = (
                top20_last5.groupby(["code_cepage"])["volume"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            fig_top_last5 = px.bar(
                top20_last5,
                x="code_cepage",
                y="volume",
                color="code_cepage",
                title="Top 20 cepages par volume produit (5 dernieres annees)",
                text_auto=".0f",
                category_orders={"code_couleur": color_order},
                color_discrete_sequence=px.colors.qualitative.Set3,
                height=500
            )
            fig_top_last5.update_layout(showlegend=False, xaxis_title="Cepage", yaxis_title="Volume (hl)")
            fig_top_last5.update_layout(
                separators=". "
            )
            fig_top_last5.update_yaxes(
                tickformat=",.0f"
            )
            fig_top_last5.update_traces(
                textfont_color="black",
                texttemplate="%{y:,.0f}",
                textposition="inside",
            )
            fig_top_last5.update_layout(
                legend_traceorder="normal"
            )
            st.plotly_chart(fig_top_last5, key="top20_last5", width="stretch")

# =====================================================
# PREDICTION
# =====================================================

with tab_pred:
    run_prediction(df)

# =====================================================
# CARTOGRAPHIE
# =====================================================

with tab_map:
    st.header("Cartographie viticole")
    # Selection du type de carte
    map_category = st.radio(
        "Type de carte",
        ["Indicateur", "Couleur", "Cepage", "Moyenne sur periode", "Cepage dominant 5 ans"],
        horizontal=True,
        key="map_category",
        label_visibility="collapsed"
    )
    
    if map_category == "Indicateur":
        map_type = st.radio(
            "Sous-type",
            ["Carte simple", "Comparaison de cartes"],
            horizontal=True,
            key="map_type_main",
            label_visibility="collapsed"
        )
        
        if map_type == "Carte simple":
            st.subheader("Carte simple par indicateur")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                indicator = st.selectbox("Indicateur", ["Rendement", "Volume", "Surface"], key="map_indicator")
            with col2:
                level = st.selectbox("Niveau", ["Departement", "Zone"], key="map_level")
            with col3:
                available_years = sorted(df["annee"].dropna().unique(), reverse=True)
                year = st.selectbox("Annee", available_years, key="map_year")
            with col4:
                map_couleurs = st.multiselect("Couleurs", ["BL", "RG", "RS"], default=["BL", "RG", "RS"], key="map_couleurs")
            
            if st.button("Generer la carte", key="btn_map_simple"):
                with st.spinner("Creation de la carte en cours..."):
                    data = prepare_map_data(df, df_all, indicator, level, year, map_couleurs)
                    if data:
                        data['map_display'] = get_zones_1_7(data['map_display'])
                        display_map_with_legend(data, title_suffix=str(year), height=500, key="map_simple")
        
        else:
            st.subheader("Comparaison de deux cartes")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### Carte 1")
                indicator1 = st.selectbox("Indicateur 1", ["Rendement", "Volume", "Surface"], key="comp_ind1")
                level1 = st.selectbox("Niveau 1", ["Departement", "Zone"], key="comp_level1")
                year1 = st.selectbox("Annee 1", sorted(df["annee"].dropna().unique(), reverse=True), key="comp_year1")
                couleurs1 = st.multiselect("Couleurs 1", ["BL", "RG", "RS"], default=["BL", "RG", "RS"], key="comp_coul1")
            
            with col_right:
                st.markdown("### Carte 2")
                indicator2 = st.selectbox("Indicateur 2", ["Rendement", "Volume", "Surface"], key="comp_ind2")
                level2 = st.selectbox("Niveau 2", ["Departement", "Zone"], key="comp_level2")
                year2 = st.selectbox("Annee 2", sorted(df["annee"].dropna().unique(), reverse=True), key="comp_year2")
                couleurs2 = st.multiselect("Couleurs 2", ["BL", "RG", "RS"], default=["BL", "RG", "RS"], key="comp_coul2")
            
            if st.button("Comparer les cartes", key="btn_map_compare"):
                with st.spinner("Creation des cartes en cours..."):
                    data1 = prepare_map_data(df, df_all, indicator1, level1, year1, couleurs1)
                    data2 = prepare_map_data(df, df_all, indicator2, level2, year2, couleurs2)
                    if data1:
                        data1['map_display'] = get_zones_1_7(data1['map_display'])
                    if data2:
                        data2['map_display'] = get_zones_1_7(data2['map_display'])
                    title1 = f"{indicator1} - {level1} - {year1}"
                    title2 = f"{indicator2} - {level2} - {year2}"
                    compare_two_maps(data1, data2, title1, title2)
    
    elif map_category == "Couleur":
        st.subheader("Carte par couleur de vin")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            indicator_color = st.selectbox("Indicateur", ["Rendement", "Volume", "Surface"], key="color_indicator")
        with col2:
            level_color = st.selectbox("Niveau", ["Departement", "Zone"], key="color_level")
        with col3:
            available_years = sorted(df["annee"].dropna().unique(), reverse=True)
            year_color = st.selectbox("Annee", available_years, key="color_year")
        with col4:
            couleur_selected = st.selectbox("Couleur", ["BL", "RG", "RS"], key="color_select")
        
        if st.button("Generer la carte par couleur", key="btn_color_map"):
            with st.spinner("Creation de la carte en cours..."):
                data = prepare_map_data(df, df_all, indicator_color, level_color, year_color, [couleur_selected])
                if data:
                    data['map_display'] = get_zones_1_7(data['map_display'])
                    display_map_with_legend(data, title_suffix=f"{year_color} - {couleur_selected}", height=500)
        
        # Comparaison des couleurs
        st.subheader("Comparaison des 3 couleurs")
        
        if st.button("Comparer les 3 couleurs", key="btn_compare_colors"):
            with st.spinner("Creation des cartes..."):
                cols = st.columns(3)
                color_infos = [("BL", "Blanc", "#F4D03F"), ("RG", "Rouge", "#A93226"), ("RS", "Rose", "#F1948A")]
                
                for idx, (coul, color_name, color_hex) in enumerate(color_infos):
                    data = prepare_map_data(df, df_all, indicator_color, level_color, year_color, [coul])
                    if data:
                        data['map_display'] = get_zones_1_7(data['map_display'])
                        with cols[idx]:
                            st.markdown(f"**{color_name}**")
                            fig = px.choropleth(
                                data['map_display'],
                                geojson=data['geo'],
                                locations=data['locations'],
                                featureidkey=data['feature_key'],
                                color=data['var'],
                                color_continuous_scale=['#FFFFFF', color_hex],
                                title=f"{indicator_color} - {color_name}",
                                height=500
                            )
                            fig.update_geos(fitbounds="locations", visible=False)
                            fig.update_layout(coloraxis_showscale=False)
                            st.plotly_chart(fig, key=f"color_compare_{coul}", width="stretch")
    
    elif map_category == "Cepage":
        st.subheader("Carte par cepage")
        
        reve_data = df[df["type_mvt"] == "REVE"].copy()
        available_cepages = sorted(reve_data["code_cepage"].dropna().unique())
        
        if available_cepages:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                indicator_cepage = st.selectbox("Indicateur", ["Volume", "Rendement", "Surface"], key="cepage_indicator")
            with col2:
                level_cepage = st.selectbox("Niveau", ["Departement", "Zone"], key="cepage_level")
            with col3:
                available_years = sorted(df["annee"].dropna().unique(), reverse=True)
                year_cepage = st.selectbox("Annee", available_years, key="cepage_year")
            with col4:
                selected_cepage = st.selectbox("Cepage", available_cepages, key="cepage_select")
            
            if st.button("Generer la carte par cepage", key="btn_cepage_map"):
                with st.spinner("Creation de la carte en cours..."):
                    data = create_cepage_map(df, df_all, indicator_cepage, level_cepage, year_cepage, None, selected_cepage)
                    if data:
                        display_map_with_legend(data, title_suffix=f"{selected_cepage} - {year_cepage}", height=500)
                    else:
                        st.warning("Aucune donnee disponible pour ce cepage")
            
            # Carte cepage par couleur
            st.subheader("Carte des cepages par couleur")
            
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                indicator_couleur = st.selectbox("Indicateur", ["Volume", "Rendement", "Surface"], key="cepage_color_indicator")
            with col_c2:
                level_couleur = st.selectbox("Niveau", ["Departement", "Zone"], key="cepage_color_level")
            with col_c3:
                year_couleur = st.selectbox("Annee", sorted(df["annee"].dropna().unique(), reverse=True), key="cepage_color_year")
            with col_c4:
                selected_couleur = st.selectbox("Couleur", ["BL", "RG", "RS"], key="cepage_color_select")
            
            if st.button("Generer la carte cepage par couleur", key="btn_cepage_color_map"):
                with st.spinner("Creation de la carte en cours..."):
                    data = create_cepage_map(df, df_all, indicator_couleur, level_couleur, year_couleur, selected_couleur, None)
                    if data:
                        data['map_display'] = get_zones_1_7(data['map_display'])
                        display_map_with_legend(data, title_suffix=f"{selected_couleur} - {year_couleur}", height=500)
                    else:
                        st.warning("Aucune donnee disponible pour cette couleur")
        else:
            st.info("Aucune donnee de cepage disponible")
    
    elif map_category == "Moyenne sur periode":
        st.subheader("Moyenne sur periode")
        
        analysis_type = st.radio(
            "Type d'analyse",
            ["Indicateur", "Couleur", "Cepage"],
            horizontal=True,
            key="mean_analysis_type"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            years_range = st.slider("Periode d'analyse", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), key="mean_years_slider")
        with col2:
            level_mean = st.selectbox("Niveau", ["Departement", "Zone"], key="mean_level")
        with col3:
            if analysis_type == "Indicateur":
                indicator_mean = st.selectbox("Indicateur", ["Rendement", "Volume", "Surface"], key="mean_indicator")
            elif analysis_type == "Couleur":
                couleur_mean = st.selectbox("Couleur", ["BL", "RG", "RS"], key="mean_color")
                indicator_mean = st.selectbox("Indicateur", ["Volume", "Rendement", "Surface"], key="mean_indicator_color")
            else:
                cepage_mean = st.selectbox("Cepage", sorted(df[df["type_mvt"] == "REVE"]["code_cepage"].dropna().unique()), key="mean_cepage")
                indicator_mean = st.selectbox("Indicateur", ["Volume", "Rendement", "Surface"], key="mean_indicator_cepage")
        
        if st.button("Generer la carte", key="btn_mean_map"):
            with st.spinner("Creation de la carte en cours..."):
                indicator_key = {"Rendement": "rendement", "Volume": "volume", "Surface": "surface"}[indicator_mean]
                mvt_key = "DECR" if indicator_key == "rendement" else "REVE"
                
                df_period = df[(df["annee"].between(years_range[0], years_range[1])) & (df["type_mvt"] == mvt_key)].copy()
                
                if analysis_type == "Couleur":
                    df_period = df_period[df_period["code_couleur"] == couleur_mean]
                elif analysis_type == "Cepage":
                    df_period = df_period[df_period["code_cepage"] == cepage_mean]
                
                if level_mean == "Departement":
                    geo = load_geojson("departements")
                    if geo is None:
                        st.error("Impossible de charger la carte des departements")
                        st.stop()
                    map_display = df_period.groupby("code_departement")[indicator_key].mean().reset_index()
                    locations = "code_departement"
                    feature_key = "properties.dep"
                else:
                    geo = load_geojson("zones")
                    if geo is None:
                        st.error("Impossible de charger la carte des zones")
                        st.stop()
                    zone_agg = df_period.groupby("Zone")[indicator_key].mean().reset_index()
                    map_display = df_all[["commune", "Zone"]].drop_duplicates().merge(zone_agg, on="Zone", how="inner")
                    map_display = get_zones_1_7(map_display)
                    locations = "commune"
                    feature_key = "properties.code_commune"
                
                if geo and not map_display.empty:
                    scale_map = {"Rendement": "RdYlGn", "Volume": "Blues", "Surface": "Greens"}[indicator_mean]
                    unit_map = {"Rendement": "hl/ha", "Volume": "hl", "Surface": "ha"}[indicator_mean]
                    
                    title_suffix = f"{years_range[0]}-{years_range[1]}"
                    if analysis_type == "Couleur":
                        title_suffix += f" - {couleur_mean}"
                    elif analysis_type == "Cepage":
                        title_suffix += f" - {cepage_mean}"
                    
                    fig_map = px.choropleth(
                        map_display,
                        geojson=geo,
                        locations=locations,
                        featureidkey=feature_key,
                        color=indicator_key,
                        color_continuous_scale=scale_map,
                        title=f"{indicator_mean} moyen - {title_suffix}",
                        labels={indicator_key: f"{indicator_mean} ({unit_map})"}
                    )
                    fig_map.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig_map, key="mean_map", width="stretch")
                else:
                    st.error("Impossible de generer la carte")
    
    else:  # Cepage dominant 5 ans
        st.subheader("Cepage dominant par zone - 5 dernieres annees")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_couleur = st.selectbox("Filtrer par couleur", ["Toutes", "BL", "RG", "RS"], key="dom_couleur")
        with col_f2:
            top_n = st.slider("Nombre de cepages a afficher", 5, 20, 10, key="dom_top_n")
        
        try:
            geo_zones = load_geojson("zones")
            
            # Filtrer les 5 dernieres annees
            last_5_years_cepage = get_last_5_years(df)
            last_5_years_cepage = last_5_years_cepage[last_5_years_cepage["type_mvt"] == "REVE"].copy()
            
            if filter_couleur != "Toutes":
                last_5_years_cepage = last_5_years_cepage[last_5_years_cepage["code_couleur"] == filter_couleur]
            
            # Calcul des cepages dominants
            dom = (
                last_5_years_cepage
                .groupby(["Zone", "code_cepage"])["volume"]
                .sum()
                .reset_index()
            )
            
            if not dom.empty:
                # Prendre le top N par zone
                dom_sorted = dom.sort_values(["Zone", "volume"], ascending=[True, False])
                dom_top = dom_sorted.groupby("Zone").head(top_n).reset_index(drop=True)
                
                # Carte avec tous les cepages (couleur differente par cepage)
                map_df = df_all[["commune", "Zone"]].drop_duplicates().merge(dom_top, on="Zone", how="inner")
                map_df = get_zones_1_7(map_df)
                
                if not map_df.empty:
                    # Compter le nombre de cepages par zone pour l'affichage
                    cepage_counts = dom_top.groupby("Zone").size().reset_index(name="nb_cepages")
                    map_df = map_df.merge(cepage_counts, on="Zone", how="left")
                    
                    fig_dom = px.choropleth(
                        map_df,
                        geojson=geo_zones,
                        locations="commune",
                        featureidkey="properties.code_commune",
                        color="code_cepage",
                        title=f"Cepages dominants par zone (Top {top_n} - 5 dernieres annees{f' - {filter_couleur}' if filter_couleur != 'Toutes' else ''})",
                        hover_data=["Zone", "code_cepage", "volume", "nb_cepages"],
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        height=500
                    )
                    
                    fig_dom.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig_dom, key="dominant_cepage_map_5y", width="stretch")
                    
                    # Tableau des cepages dominants
                    st.subheader("Detail des cepages dominants")
                    display_df = dom_top[["Zone", "code_cepage", "volume"]].sort_values(["Zone", "volume"], ascending=[True, False])
                    display_df["volume"] = display_df["volume"].apply(lambda x: f"{x:,.0f}").replace(",", " ")
                    st.dataframe(display_df, width="stretch", hide_index=True)
                else:
                    st.info("Aucune donnee disponible pour les criteres selectionnes")
            else:
                st.info("Aucune donnee de volume par cepage disponible pour les 5 dernieres annees")
        except Exception as e:
            st.warning(f"Erreur carte cepage dominant: {e}")

# =====================================================
# ANALYSE QUANTITATIVE - VERSION SIMPLIFIEE
# =====================================================

with tab_quant:
    st.header("Analyse quantitative approfondie")
    
    base = df.copy()
    base["prod_hl_ha"] = np.where(base["surface"] > 0, base["volume"] / base["surface"], np.nan)
    
    st.subheader("Indicateurs cles de performance")

    col_k1, col_k2, col_k3 = st.columns(3)

    prod_global = (base["volume"].sum() / base["surface"].sum()) if base["surface"].sum() > 0 else np.nan
    corr_vr = base[["volume", "rendement"]].dropna().corr().iloc[0, 1] if len(base[["volume", "rendement"]].dropna()) > 2 else np.nan
    vol_rdt = base["rendement"].std()
    col_k1.metric("Productivite moyenne", f"{prod_global:.0f} hl/ha" if pd.notna(prod_global) else "N/A")
    col_k2.metric("Correlation Volume/Rendement", f"{corr_vr:.1f}" if pd.notna(corr_vr) else "N/A")
    col_k3.metric("Volatilite rendement", f"{vol_rdt:.1f} hl/ha" if pd.notna(vol_rdt) else "N/A")
    
    tab_q2, tab_q3, tab_q4, tab_q5, tab_q6 = st.tabs([
       "Rendement vs Volume", "Productivite et rendement par zone", "Productivite par couleur", "Volatilite", "Tableau de bord"
    ])
    with tab_q2:
        st.subheader("Relation Rendement - Volume")
        st.markdown("*Analyse de la correlation entre le rendement et le volume produit*")
        
        tmp = base.dropna(subset=["rendement", "volume", "code_couleur"]).copy()
        if not tmp.empty:
            # Graphique 1: Nuage de points avec regression
            fig4, ax4 = plt.subplots(figsize=(16, 6))
            
            for couleur in ["BL", "RG", "RS"]:
                subset = tmp[tmp["code_couleur"] == couleur]
                if not subset.empty:
                    color = COLOR_MAP.get(couleur, "#808080")
                    ax4.scatter(subset["rendement"], subset["volume"], 
                               alpha=0.5, label=f"{couleur}", c=color, s=30)
                    
                    # Regression lineaire
                    z = np.polyfit(subset["rendement"], subset["volume"], 1)
                    p = np.poly1d(z)
                    ax4.plot(sorted(subset["rendement"]), p(sorted(subset["rendement"])), 
                            "--", color=color, linewidth=1.5)
            ax4.set_xlim(0, 100)
            ax4.set_xlabel("Rendement (hl/ha)", fontsize=12)
            ax4.set_ylabel("Volume (hl)", fontsize=12)
            ax4.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
            )
            ax4.set_title("Relation Rendement - Volume par couleur", fontsize=14, fontweight="bold")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            plt.close(fig4)
            gc.collect()
        # Interpretation
        corr_value = tmp["rendement"].corr(tmp["volume"])
        st.info(f"""
        **Interpretation :**
        - La correlation entre rendement et volume est de {corr_value:.2f}
        - Un rendement eleve n'implique pas automatiquement un volume eleve
        """)
            
    
    with tab_q3:
        st.subheader("Productivite par zone")
        
        st.markdown("*Analyse comparative de la productivite (volume/surface) entre les zones*")
        tmp = base.dropna(subset=["surface", "volume", "code_couleur"]).copy()
        if not tmp.empty:
            # Productivite par zone
            prod_zone = tmp.groupby("Zone").agg({
                "surface": "sum",
                "volume": "sum"
            }).reset_index()
            prod_zone["productivite"] = prod_zone["volume"] / prod_zone["surface"]
            prod_zone = get_zones_1_7(prod_zone)
            
            fig2, ax2 = plt.subplots(figsize=(16, 6))
            bars = ax2.bar(prod_zone["Zone"].astype(str), prod_zone["productivite"], 
                            color=[ZONE_COLOR_MAP.get(str(int(z)), "#808080") for z in prod_zone["Zone"]])
            ax2.set_xlabel("Zone", fontsize=13)
            ax2.set_ylabel("Productivite (hl/ha)", fontsize=15)
            ax2.set_title("Productivite moyenne par zone", fontsize=17, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3)
            
            # Ajout des valeurs sur les barres
            for bar, val in zip(bars, prod_zone["productivite"]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig2)
            plt.close(fig2)
            gc.collect()
            # Interpretation automatique
            st.info(f"""
            **Interpretation :**
            - La productivite moyenne est de {prod_global:.0f} hl/ha
            """)
            col_rank1, col_rank2 = st.columns(2)
            
            with col_rank1:
                # Top 1 zone
                top1 = prod_zone.loc[prod_zone["productivite"].idxmax()]
                st.markdown("**La zone la plus productive**")
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 5px 0;">
                    <b>Zone {int(top1['Zone'])}</b> : {top1['productivite']:.0f} hl/ha
                </div>
                """, unsafe_allow_html=True)
            
            with col_rank2:
                # Bottom 1 zone
                bottom1 = prod_zone.loc[prod_zone["productivite"].idxmin()]
                st.markdown("**La zone la moins productive**")
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 5px 0;">
                    <b>Zone {int(bottom1['Zone'])}</b> : {bottom1['productivite']:.0f} hl/ha
                </div>
                """, unsafe_allow_html=True)
            # Graphique 2: Distribution du rendement par zone (boxplot simplifie)
            fig5, ax5 = plt.subplots(figsize=(16, 6))
            tmp_zone = tmp[~tmp["Zone"].isin(["0", "None", "nan"])].copy()
            zones_sorted = sorted(tmp_zone["Zone"].unique(), key=lambda x: int(x))
            
            box_data = []
            positions = []
            colors_box = []
            for i, zone in enumerate(zones_sorted):
                zone_data = tmp_zone[tmp_zone["Zone"] == zone]["rendement"].dropna()
                if not zone_data.empty:
                    box_data.append(zone_data)
                    positions.append(i + 1)
                    colors_box.append(ZONE_COLOR_MAP.get(str(int(zone)), "#808080"))
            
            bp = ax5.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax5.set_xticks(positions)
            ax5.set_xticklabels([f"Zone {int(z)}" for z in zones_sorted])
            ax5.set_xlabel("Zone", fontsize=12)
            ax5.set_ylim(0,100)
            ax5.set_ylabel("Rendement (hl/ha)", fontsize=12)
            ax5.set_title("Distribution du rendement par zone", fontsize=14, fontweight="bold")
            ax5.grid(axis="y", alpha=0.3)
            st.pyplot(fig5)
            st.info ("N.B: cv: Coefficient de variation (en %)")
            plt.close(fig5)
            gc.collect()

    with tab_q4:
        st.subheader("Productivite par couleur")
        st.markdown("*Analyse comparative de la productivite (volume/surface) entre les couleurs*")
        tmp = base.dropna(subset=["surface", "volume", "code_couleur"]).copy()
        if not tmp.empty:
            # Productivite par couleur
            prod_couleur = tmp.groupby("code_couleur").agg({
                "surface": "sum",
                "volume": "sum"
            }).reset_index()
            prod_couleur["productivite"] = prod_couleur["volume"] / prod_couleur["surface"]
            
            fig20, ax2 = plt.subplots(figsize=(16, 6))
            bars = ax2.bar(prod_couleur["code_couleur"].astype(str), prod_couleur["productivite"],
                            color=[COLOR_MAP.get(c, "#808080") for c in prod_couleur["code_couleur"]])
            ax2.set_xlabel("Couleur", fontsize=13)
            ax2.set_ylabel("Productivite (hl/ha)", fontsize=15)
            ax2.set_title("Productivite moyenne par couleur", fontsize=17, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3)
            
            # Ajout des valeurs sur les barres
            for bar, val in zip(bars, prod_couleur["productivite"]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig20)
            plt.close(fig20)
            gc.collect()
            col_rank1, col_rank2 = st.columns(2)
            with col_rank1:
                # Top 1 zone
                top1 = prod_couleur.loc[prod_couleur["productivite"].idxmax()]
                st.markdown("**La couleur la plus productive**")
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 5px 0;">
                    <b>Couleur {top1['code_couleur']}</b> : {top1['productivite']:.0f} hl/ha
                </div>
                """, unsafe_allow_html=True)
            
            with col_rank2:
                # Bottom 1 zone
                bottom1 = prod_couleur.loc[prod_couleur["productivite"].idxmin()]
                st.markdown("**La couleur la moins productive**")
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 5px 0;">
                    <b>Couleur {bottom1['code_couleur']}</b> : {bottom1['productivite']:.0f} hl/ha
                </div>
                """, unsafe_allow_html=True)
    with tab_q5:
        st.subheader("Volatilite du rendement par zone")
        st.markdown("*Analyse de la stabilite interannuelle du rendement*")
        tmp = base.dropna(subset=["rendement", "Zone", "annee"]).copy()
        tmp = get_zones_1_7(tmp)
        
        if not tmp.empty:
            # CORRECTION: Calcul des statistiques de volatilite
            vol_stats = tmp.groupby("Zone")["rendement"].agg(['mean', 'std']).reset_index()
            vol_stats['cv'] = (vol_stats['std'] / vol_stats['mean'] * 100).round(1)
            vol_stats = get_zones_1_7(vol_stats)
            vol_stats = vol_stats.sort_values("cv")
            # Graphique 1: Coefficient de variation
            fig7, ax7 = plt.subplots(figsize=(16, 6))
            color_map_zones = {
                str(k): v for k, v in ZONE_COLOR_MAP.items()
            }
            colors_zones = [
                color_map_zones.get(str(int(z)), "#808080")
                for z in vol_stats["Zone"]
            ]
            colors_cv = ['#2e7d32' if cv < 15 else '#f9a825' if cv < 25 else '#c62828' for cv in vol_stats['cv']]
            bars = ax7.bar(vol_stats["Zone"].astype(str), vol_stats["cv"], color=colors_zones, edgecolor="black")
            ax7.set_xlabel("Zone", fontsize=12)
            ax7.set_ylabel("Coefficient de variation (%)", fontsize=12)
            ax7.set_title("Stabilite du rendement par zone", fontsize=14, fontweight="bold")
            ax7.grid(axis="y", alpha=0.3)
            # Ajout des seuils
            ax7.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Seuil de stabilite (15%)')
            ax7.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Seuil de variabilite (25%)')
            ax7.legend()
            # Ajout des valeurs
            for bar, val in zip(bars, vol_stats["cv"]):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
            
            st.pyplot(fig7)
            plt.close(fig7)
            gc.collect()
            # Interpretation
            st.markdown("""
            **Guide de lecture :**
            - **Indice de stabilite du rendement inférieur à 15%** : Rendement tres stable (vert)
            - **Indice de stabilite du rendement entre 15% et 25%** : Variabilite moderee (orange)
            - **Indice de stabilite du rendement supérieur à 25%** : Rendement tres variable (rouge)
            """)
            
            # Graphique 2: Evolution temporelle pour les zones les plus/moins stables
            st.subheader("Evolution temporelle comparative")
            # Zone la plus stable
            most_stable = vol_stats.loc[vol_stats["cv"].idxmin(), "Zone"]
            st.markdown(f"**Zone la plus stable : Zone {int(most_stable)} (Stabilite du rendement={vol_stats.loc[vol_stats['cv'].idxmin(), 'cv']:.0f}%)**")
            
            df_stable = tmp[tmp["Zone"] == most_stable].groupby("annee")["rendement"].mean().reset_index()
            zone_str_most_stable = str(int(most_stable))
            zone_color_stable = ZONE_COLOR_MAP.get(zone_str_most_stable, "#808080")
            fig8, ax8 = plt.subplots(figsize=(16, 6))
            ax8.plot(df_stable["annee"], df_stable["rendement"], 'o-', color=zone_color_stable, linewidth=2, markersize=6)
            ax8.axhline(y=df_stable["rendement"].mean(), color='green', linestyle='--', alpha=0.7, label='Moyenne')
            ax8.set_xlabel("Annee", fontsize=12)
            ax8.set_ylabel("Rendement (hl/ha)", fontsize=12)
            ax8.set_title(f"Evolution Zone {int(most_stable)}", fontsize=14)
            ax8.grid(True, alpha=0.3)
            ax8.legend()
            st.pyplot(fig8)
            plt.close(fig8)
            gc.collect()
            # Zone la plus variable
            most_variable = vol_stats.loc[vol_stats["cv"].idxmax(), "Zone"]
            st.markdown(f"**Zone la plus variable : Zone {int(most_variable)} (Stabilite du rendement={vol_stats.loc[vol_stats['cv'].idxmax(), 'cv']:.0f}%)**")
            df_variable = tmp[tmp["Zone"] == most_variable].groupby("annee")["rendement"].mean().reset_index()
            zone_str_most_variable = str(int(most_variable))
            zone_color_variable = ZONE_COLOR_MAP.get(zone_str_most_variable, "#808080")
            fig9, ax9 = plt.subplots(figsize=(16, 6))
            ax9.plot(df_variable["annee"], df_variable["rendement"], 'o-', color=zone_color_variable, linewidth=2, markersize=6)
            ax9.axhline(y=df_variable["rendement"].mean(), color='red', linestyle='--', alpha=0.7, label='Moyenne')
            ax9.set_xlabel("Annee", fontsize=12)
            ax9.set_ylabel("Rendement (hl/ha)", fontsize=12)
            ax9.set_title(f"Evolution Zone {int(most_variable)}", fontsize=14)
            ax9.grid(True, alpha=0.3)
            ax9.legend()
            st.pyplot(fig9)
            plt.close(fig9)
            gc.collect()
            
    
    with tab_q6:
        st.subheader("Tableau de bord analytique")
        tmp = base.dropna(subset=["surface", "volume", "code_couleur"]).copy()
        if not tmp.empty:
            # Productivite par zone
            prod_zone = tmp.groupby("Zone").agg({
                "surface": "sum",
                "volume": "sum",
                "rendement": "mean",
            }).reset_index()
            prod_zone["productivite"] = prod_zone["volume"] / prod_zone["surface"]
            prod_zone = get_zones_1_7(prod_zone)
            prod_zone['% volume'] = (prod_zone['volume'] / prod_zone['volume'].sum() * 100).round(1)
            prod_zone = get_zones_1_7(prod_zone)
            prod_zone = prod_zone[prod_zone["Zone"] != "1"]
            prod_zone = prod_zone.sort_values("Zone", key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else 0))
            prod_zone["surface"] = prod_zone["surface"].apply(lambda x: f"{x:,.0f}".replace(",", " "))
            prod_zone["rendement"] = prod_zone["rendement"].apply(lambda x: f"{x:,.0f}".replace(",", " "))
            prod_zone["volume"] = prod_zone["volume"].apply(lambda x: f"{x:,.0f}".replace(",", " "))
            prod_zone = prod_zone.sort_values(
                "Zone",
                key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else 0)
            )
            # Version simplifiee avec mise en forme conditionnelle
            st.dataframe(
                prod_zone,
                column_config={
                    "Zone": st.column_config.TextColumn("Zone", width="small"),
                    "surface": st.column_config.TextColumn("Surface (ha)", width="medium"),
                    "volume": st.column_config.TextColumn("Volume (hl)", width="medium"),
                    "rendement": st.column_config.NumberColumn("Rendement (hl/ha)", format="%.0f", width="medium"),
                    "productivite": st.column_config.NumberColumn("productivite", format="%.0f", width="medium"),
                    "% volume": st.column_config.NumberColumn("% Volume", format="%.1f", width="small")
                },
                width="stretch",
                hide_index=True
            )
        
# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Observatoire Viticole - Pays d'Oc IGP | Donnees mises a jour regulierement</p>
    <p style="font-size: 0.75rem;">(c) 2024 - Analyse des rendements et volumes viticoles</p>
</div>
""", unsafe_allow_html=True)
gc.collect()