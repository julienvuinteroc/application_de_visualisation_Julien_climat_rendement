import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(layout="wide", page_title="Analyse Viticole")

# Style CSS personnalisé pour un design professionnel
st.markdown("""
<style>
    /* Style global */
    .stApp {
        background-color: #f8f9fa;
    }

    /* En-têtes */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 500;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }

    /* Conteneurs */
    .stContainer {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Métriques */
    .stMetric {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Boutons */
    .stButton button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 500;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }

    /* Tabs */
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

    /* Captions */
    .stCaption {
        color: #6c757d;
        font-style: italic;
    }

    /* Alertes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }

    /* Progress bar */
    .stProgress .st-bo {
        background-color: #2c3e50;
    }

    /* Cartes */
    .map-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# IMPORTS DES MODULES
# =====================================================
from config.constants import (
    COLOR_MAP, ZONE_COLOR_MAP, ZONE_LABELS,
    DEPARTEMENTS, GEOJSON_FILES,
    BASE_DIR, DB_FILE, PARQUET_FILE
)
from modules.data_loader import load_data, load_geojson, apply_filters
from modules.utils import (
    zone_sort_key, zone_order_from_series, complete_years,
    bootstrap_ci, format_number
)
from modules.visualization import (
    create_rendement_chart, create_volume_chart,
    create_comparison_bar_chart, create_pie_chart,
    create_scatter_with_trend, create_choropleth_map,
    create_anomaly_chart
)
from modules.prediction import run_prediction


# =====================================================
# INITIALISATION
# =====================================================

@st.cache_data
def initialize_data():
    """Charge les données avec cache"""
    return load_data()

# Chargement des données
df_all = initialize_data()


# =====================================================
# SIDEBAR - FILTRES
# =====================================================

with st.sidebar:
    st.title("Filtres globaux")
    st.divider()

    # Carte de référence toggle
    show_ref_map = st.toggle("Voir la carte des zones", value=False, key="show_map_toggle")

    # Filtres principaux
    departements = st.multiselect(
        "Départements",
        DEPARTEMENTS,
        default=DEPARTEMENTS,
        key="sb_dep",
        help="Selectionnez les departements a analyser"
    )

    all_zones = zone_order_from_series(df_all["Zone"])
    zones = st.multiselect(
        "Zones pedoclimatiques",
        all_zones,
        default=all_zones,
        key="sb_zone",
        help="Filtrer par zones pedoclimatiques"
    )

    couleurs = st.multiselect(
        "Couleurs",
        ["BL", "RG", "RS"],
        default=["BL", "RG", "RS"],
        key="sb_coul",
        help="Type de vin : BL (Blanc), RG (Rouge), RS (Rose)"
    )

    annee_min = int(df_all["annee"].dropna().min())
    annee_max = int(df_all["annee"].dropna().max())

    annees = st.slider(
        "Periode d'analyse",
        annee_min, annee_max, (annee_min, annee_max),
        key="sb_years",
        help="Selectionnez la plage d'annees"
    )

# Application des filtres
df = apply_filters(df_all, departements, zones, couleurs, annees)

# Verification que les donnees ne sont pas vides
if df.empty:
    st.error("Aucune donnee ne correspond aux filtres selectionnes. Veuillez elargir vos criteres.")
    st.stop()

YEAR_MIN = int(df["annee"].dropna().min())
YEAR_MAX = int(df["annee"].dropna().max())


# =====================================================
# CARTE DE REFERENCE
# =====================================================

if show_ref_map:
    st.header("Carte de reference - Zones pedoclimatiques")

    try:
        geo_zones = load_geojson("zones")

        if geo_zones is None:
            st.error("Impossible de charger la carte des zones. Verifiez que le fichier GeoJSON est present.")
        else:
            ref = df_all[["commune", "Zone"]].drop_duplicates().copy()
            ref["Zone"] = ref["Zone"].astype(str)
            ref["Nom"] = ref["Zone"].map(ZONE_LABELS).fillna(ref["Zone"].apply(lambda z: f"Zone {z}"))

            fig_ref = px.choropleth(
                ref,
                geojson=geo_zones,
                locations="commune",
                featureidkey="properties.code_commune",
                color="Zone",
                color_discrete_map=ZONE_COLOR_MAP,
                category_orders={"Zone": zone_order_from_series(ref["Zone"])},
                hover_name="Nom",
                title="Zones pedoclimatiques de la region"
            )

            fig_ref.update_geos(fitbounds="locations", visible=False)
            fig_ref.update_layout(
                template="plotly_white",
                legend_title_text="Zone",
                height=600
            )

            st.plotly_chart(fig_ref, width="stretch")

            with st.expander("Description des zones"):
                for zone, desc in ZONE_LABELS.items():
                    if zone in ref["Zone"].unique():
                        color = ZONE_COLOR_MAP.get(zone, "#808080")
                        st.markdown(
                            f"<span style='color:{color};font-weight:bold'>■</span> **{desc}**",
                            unsafe_allow_html=True
                        )

    except Exception as e:
        st.error(f"Erreur lors du chargement de la carte : {str(e)}")
        st.info("Verifiez que les fichiers GeoJSON sont presents dans le dossier assets/geojson/")

    st.stop()


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
    st.caption("Analyse des rendements en hl/ha - Donnees DECR (Declarations de recolte)")

    decr = df[df["type_mvt"] == "DECR"].copy()

    decr_bl = decr[(decr["code_couleur"] == "BL") & (decr["rendement"].between(5, 90))]
    decr_rg = decr[(decr["code_couleur"] == "RG") & (decr["rendement"].between(5, 90))]
    decr_rs = decr[(decr["code_couleur"] == "RS") & (decr["rendement"].between(5, 100))]

    decr = pd.concat([decr_bl, decr_rg, decr_rs], ignore_index=True)

    if decr.empty:
        st.warning("Aucune donnee de rendement conforme aux plafonds avec les filtres actuels")
    else:
        total_initial = len(df[df["type_mvt"] == "DECR"])
        total_filtre = len(decr)
        if total_filtre < total_initial:
            st.info(f"{total_initial - total_filtre} lignes avec rendements > plafonds ont ete exclues de l'analyse")

        col1, col2 = st.columns([1, 1])

        with col1:
            mode = st.radio(
                "Comparer par",
                ["Couleur", "Departement", "Zone"],
                horizontal=True,
                key="rdt_mode"
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
                key="rdt_sel"
            )

        if not selections:
            st.info("Veuillez selectionner au moins un element a analyser")
        else:
            data = decr[decr[col_map].isin(selections)].copy()

            with st.expander("Verification des plafonds", expanded=False):
                plafond_check = data.groupby("code_couleur")["rendement"].agg(['max', 'count']).round(2)
                plafond_check['plafond'] = plafond_check.index.map(lambda x: 90 if x in ['BL', 'RG'] else 100)
                plafond_check['conforme'] = plafond_check['max'] <= plafond_check['plafond']
                plafond_check = plafond_check.rename(columns={'max': 'Rendement max', 'count': 'Nb observations'})
                st.dataframe(plafond_check[['Rendement max', 'plafond', 'conforme', 'Nb observations']], width="stretch")

            evol = data.groupby(["annee", col_map])["rendement"].mean().reset_index()
            evol_full = complete_years(evol, col_map, "rendement", "mean", YEAR_MIN, YEAR_MAX)

            fig = create_rendement_chart(evol_full, col_map, mode)
            st.plotly_chart(fig, width="stretch")

            with st.expander("Statistiques descriptives", expanded=False):
                stats_df = data.groupby(col_map)["rendement"].describe(percentiles=[.25, .5, .75]).round(2)
                stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                stats_df.columns = ['Nb annees', 'Moyenne', 'Ecart-type', 'Min', 'Q1', 'Mediane', 'Q3', 'Max']
                st.dataframe(stats_df, width="stretch")


# =====================================================
# VOLUME
# =====================================================

with tab_vol:
    st.header("Analyse des volumes")
    st.caption("Analyse des volumes en hectolitres - Donnees REVE (Revendications)")

    reve = df[
        (df["type_mvt"] == "REVE") &
        (df["volume"] > 0)
    ].copy()

    if reve.empty:
        st.warning("Aucune donnee de volume disponible avec les filtres actuels")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            mode = st.radio(
                "Comparer par",
                ["Couleur", "Departement", "Zone", "Cepage"],
                horizontal=True,
                key="vol_mode"
            )

        col_map = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",
            "Zone": "Zone",
            "Cepage": "code_cepage"
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
                key="vol_sel"
            )

        if not selections:
            st.info("Veuillez selectionner au moins un element a analyser")
        else:
            data = reve[reve[col_map].isin(selections)].copy()

            evol = data.groupby(["annee", col_map])["volume"].sum().reset_index()
            evol_full = complete_years(evol, col_map, "volume", "sum", YEAR_MIN, YEAR_MAX)

            fig = create_volume_chart(evol_full, col_map, mode)
            st.plotly_chart(fig, width="stretch")


# =====================================================
# PREDICTION - Random Forest uniquement
# =====================================================

with tab_pred:
    st.header("Prediction - Random Forest")
    st.caption("Modele de prediction base sur Random Forest")

    run_prediction(df)


# =====================================================
# CARTOGRAPHIE COMPLETE AVEC LEGENDES (sans communes)
# =====================================================

with tab_map:
    st.header("Cartographie viticole")
    st.caption("Visualisation spatiale des donnees par departement et zone")

    map_type = st.radio(
        "Type de carte",
        ["Carte simple", "Comparaison de cartes", "Carte des zones"],
        horizontal=True,
        key="map_type_main"
    )

    def prepare_map_data(indicator, level, year, couleurs):
        """Prepare les donnees pour une carte"""

        indicator_map = {
            "Rendement": {"var": "rendement", "mvt": "DECR", "agg": "mean", "unit": "hl/ha", "scale": "RdYlGn"},
            "Volume": {"var": "volume", "mvt": "REVE", "agg": "sum", "unit": "hl", "scale": "Blues"},
            "Surface": {"var": "surface", "mvt": "DECR", "agg": "sum", "unit": "ha", "scale": "Greens"}
        }

        var_info = indicator_map[indicator]

        map_data = df[
            (df["type_mvt"] == var_info["mvt"]) &
            (df["annee"] == year) &
            (df["code_couleur"].isin(couleurs))
        ].copy()

        if map_data.empty:
            return None

        try:
            if level == "Departement":
                geo = load_geojson("departements")
                if geo is None:
                    st.error("Fichier departements.geojson non trouve")
                    return None

                map_display = map_data.groupby("code_departement")[var_info["var"]].agg(var_info["agg"]).reset_index()
                locations = "code_departement"
                feature_key = "properties.dep"
                title_level = "departement"

            elif level == "Zone":
                geo = load_geojson("zones")
                if geo is None:
                    st.error("Fichier zones.geojson non trouve")
                    return None

                zone_agg = map_data.groupby("Zone")[var_info["var"]].agg(var_info["agg"]).reset_index()
                map_display = df_all[["commune", "Zone"]].drop_duplicates().merge(zone_agg, on="Zone", how="inner")
                locations = "commune"
                feature_key = "properties.code_commune"
                title_level = "zone"

            return {
                'geo': geo,
                'map_display': map_display,
                'locations': locations,
                'feature_key': feature_key,
                'var': var_info["var"],
                'scale': var_info["scale"],
                'indicator': indicator,
                'level': level,
                'unit': var_info["unit"],
                'title_level': title_level
            }

        except Exception as e:
            st.error(f"Erreur de preparation: {str(e)}")
            return None

    def display_map_with_legend(data, title_suffix="", height=500):
        """Cree une carte avec sa legende statistique"""
        if data is None or data['map_display'].empty:
            st.warning("Aucune donnee a afficher")
            return None

        fig = px.choropleth(
            data['map_display'],
            geojson=data['geo'],
            locations=data['locations'],
            featureidkey=data['feature_key'],
            color=data['var'],
            color_continuous_scale=data['scale'],
            range_color=[
                data['map_display'][data['var']].quantile(0.05),
                data['map_display'][data['var']].quantile(0.95)
            ] if len(data['map_display']) > 5 else None,
            title=f"{data['indicator']} par {data['title_level']} - {title_suffix}",
            labels={data['var']: f"{data['indicator']} ({data['unit']})"},
            hover_data={data['locations']: True, data['var']: ':.1f'}
        )

        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            template="plotly_white",
            height=height,
            margin=dict(l=0, r=0, t=50, b=80),
            coloraxis_colorbar=dict(
                title=f"{data['indicator']}<br>({data['unit']})",
                len=0.4,
                thickness=15,
                y=0.5
            )
        )

        vals = data['map_display'][data['var']].dropna()
        if not vals.empty:
            stats = {
                'mean': vals.mean(),
                'median': vals.median(),
                'min': vals.min(),
                'max': vals.max()
            }

            fig.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.2,
                        xref="paper",
                        yref="paper",
                        text=f"<b>Statistiques</b><br>"
                             f"Moyenne: {stats['mean']:.1f} {data['unit']} | "
                             f"Mediane: {stats['median']:.1f} {data['unit']}<br>"
                             f"Minimum: {stats['min']:.1f} {data['unit']} | "
                             f"Maximum: {stats['max']:.1f} {data['unit']}",
                        showarrow=False,
                        font=dict(size=11),
                        align="center",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="lightgray",
                        borderwidth=1,
                        borderpad=6
                    )
                ]
            )

        st.plotly_chart(fig, width="stretch")

        with st.expander("Voir les donnees detaillees"):
            st.dataframe(
                data['map_display'].sort_values(data['var'], ascending=False),
                width="stretch",
                hide_index=True
            )

        return fig

    def compare_two_maps(data1, data2, title1, title2):
        """Compare deux cartes cote a cote"""
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"**{title1}**")
            if data1:
                display_map_with_legend(data1, title_suffix=title1, height=400)
            else:
                st.warning("Carte 1 non disponible")

        with col_right:
            st.markdown(f"**{title2}**")
            if data2:
                display_map_with_legend(data2, title_suffix=title2, height=400)
            else:
                st.warning("Carte 2 non disponible")

        if data1 and data2 and not data1['map_display'].empty and not data2['map_display'].empty:
            st.subheader("Comparaison statistique")

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)

            vals1 = data1['map_display'][data1['var']].dropna()
            vals2 = data2['map_display'][data2['var']].dropna()

            if not vals1.empty and not vals2.empty:
                with col_s1:
                    st.metric(
                        f"{data1['indicator']} moyenne",
                        f"{vals1.mean():.1f} {data1['unit']}"
                    )
                with col_s2:
                    st.metric(
                        f"{data2['indicator']} moyenne",
                        f"{vals2.mean():.1f} {data2['unit']}"
                    )
                with col_s3:
                    diff = vals1.mean() - vals2.mean()
                    st.metric(
                        "Difference",
                        f"{diff:.1f}",
                        delta=f"{diff:.1f}"
                    )
                with col_s4:
                    ratio = (vals1.mean() / vals2.mean() * 100) if vals2.mean() != 0 else 0
                    st.metric(
                        "Ratio",
                        f"{ratio:.1f}%"
                    )

    if map_type == "Carte simple":
        st.subheader("Carte simple par indicateur")

        with st.container():
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                indicator = st.selectbox(
                    "Indicateur",
                    ["Rendement", "Volume", "Surface"],
                    key="map_indicator_simple"
                )

            with col2:
                level = st.selectbox(
                    "Niveau",
                    ["Departement", "Zone"],
                    key="map_level_simple"
                )

            with col3:
                available_years = sorted(df["annee"].dropna().unique(), reverse=True)
                year = st.selectbox(
                    "Annee",
                    available_years,
                    key="map_year_simple"
                )

            with col4:
                map_couleurs = st.multiselect(
                    "Couleurs",
                    ["BL", "RG", "RS"],
                    default=["BL", "RG", "RS"],
                    key="map_couleurs_simple"
                )

        with st.expander("Options avancees"):
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                show_stats = st.checkbox("Afficher les statistiques", value=True, key="show_stats_simple")
            with col_o2:
                color_scale = st.selectbox(
                    "Echelle de couleurs",
                    ["RdYlGn", "Blues", "Greens", "Reds", "Viridis", "Plasma"],
                    index=0,
                    key="color_scale_simple"
                )

        if st.button("Generer la carte", key="btn_map_simple", width="stretch"):
            with st.spinner("Creation de la carte en cours..."):
                data = prepare_map_data(indicator, level, year, map_couleurs)
                if data:
                    if color_scale != data['scale'] and color_scale in ["RdYlGn", "Blues", "Greens", "Reds", "Viridis", "Plasma"]:
                        data['scale'] = color_scale

                    with st.container():
                        st.markdown('<div class="map-container">', unsafe_allow_html=True)
                        display_map_with_legend(data, title_suffix=str(year), height=600)
                        st.markdown('</div>', unsafe_allow_html=True)

    elif map_type == "Comparaison de cartes":
        st.subheader("Comparaison de deux cartes")

        with st.container():
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

        if st.button("Comparer les cartes", key="btn_map_compare", width="stretch"):
            with st.spinner("Creation des cartes en cours..."):
                data1 = prepare_map_data(indicator1, level1, year1, couleurs1)
                data2 = prepare_map_data(indicator2, level2, year2, couleurs2)

                title1 = f"{indicator1} - {level1} - {year1}"
                title2 = f"{indicator2} - {level2} - {year2}"

                with st.container():
                    st.markdown('<div class="map-container">', unsafe_allow_html=True)
                    compare_two_maps(data1, data2, title1, title2)
                    st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.subheader("Carte des zones pedoclimatiques")

        try:
            geo_zones = load_geojson("zones")

            if geo_zones:
                zones_data = df_all[["commune", "Zone"]].drop_duplicates().copy()
                zones_data["Zone"] = zones_data["Zone"].astype(str)
                zones_data["Nom"] = zones_data["Zone"].map(ZONE_LABELS).fillna("Zone inconnue")

                zone_counts = zones_data.groupby("Zone").size().to_dict()

                fig_zones = px.choropleth(
                    zones_data,
                    geojson=geo_zones,
                    locations="commune",
                    featureidkey="properties.code_commune",
                    color="Zone",
                    color_discrete_map=ZONE_COLOR_MAP,
                    category_orders={"Zone": zone_order_from_series(zones_data["Zone"])},
                    hover_name="Nom",
                    hover_data={"Zone": True},
                    title="Zones pedoclimatiques de la region"
                )

                fig_zones.update_geos(fitbounds="locations", visible=False)
                fig_zones.update_layout(
                    template="plotly_white",
                    height=600,
                    legend_title_text="Zone",
                    margin=dict(l=0, r=0, t=50, b=100)
                )

                zone_text = "<br>".join([
                    f"<b>Zone {zone}</b>: {zone_counts.get(zone, 0)} communes"
                    for zone in zone_order_from_series(zones_data["Zone"])
                ])

                fig_zones.update_layout(
                    annotations=[
                        dict(
                            x=0.5,
                            y=-0.15,
                            xref="paper",
                            yref="paper",
                            text=f"<b>Repartition des communes par zone</b><br>{zone_text}",
                            showarrow=False,
                            font=dict(size=11),
                            align="center",
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="lightgray",
                            borderwidth=1,
                            borderpad=6
                        )
                    ]
                )

                st.plotly_chart(fig_zones, width="stretch")

                with st.expander("Details des zones"):
                    zone_stats = []
                    for zone in zone_order_from_series(zones_data["Zone"]):
                        if zone in zone_counts:
                            zone_stats.append({
                                "Zone": zone,
                                "Nombre de communes": zone_counts[zone],
                                "Description": ZONE_LABELS.get(zone, "Description non disponible")
                            })

                    if zone_stats:
                        st.dataframe(
                            pd.DataFrame(zone_stats),
                            column_config={
                                "Zone": "Zone",
                                "Nombre de communes": st.column_config.NumberColumn("Nb communes", format="%d"),
                                "Description": "Description"
                            },
                            width="stretch",
                            hide_index=True
                        )

        except Exception as e:
            st.error(f"Erreur lors de la creation de la carte des zones: {str(e)}")


# =====================================================
# ANALYSE QUANTITATIVE
# =====================================================

with tab_quant:
    st.header("Analyse quantitative approfondie")

    base = df.copy()
    base["prod_hl_ha"] = np.where(base["surface"] > 0, base["volume"] / base["surface"], np.nan)

    st.subheader("Indicateurs cles de performance")

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    prod_global = (base["volume"].sum() / base["surface"].sum()) if base["surface"].sum() > 0 else np.nan
    corr_vr = base[["volume", "rendement"]].dropna().corr().iloc[0, 1] if len(base[["volume", "rendement"]].dropna()) > 2 else np.nan
    vol_rdt = base["rendement"].std()
    share_hors = (base["statut_plafond"].eq("Hors plafond").mean() * 100) if base["rendement"].notna().any() else np.nan

    col_k1.metric("Productivite moyenne", f"{prod_global:.1f} hl/ha" if pd.notna(prod_global) else "N/A")
    col_k2.metric("Correlation V/R", f"{corr_vr:.2f}" if pd.notna(corr_vr) else "N/A")
    col_k3.metric("Volatilite rendement", f"{vol_rdt:.1f}" if pd.notna(vol_rdt) else "N/A")
    col_k4.metric("% Hors plafond", f"{share_hors:.1f}%" if pd.notna(share_hors) else "N/A")

    tab_q1, tab_q2, tab_q3, tab_q4, tab_q5 = st.tabs([
        "Surface vs Volume",
        "Rendement vs Volume",
        "Productivite par zone",
        "Volatilite",
        "Tableau de bord"
    ])

    with tab_q1:
        st.subheader("Relation Surface - Volume")
        tmp = base.dropna(subset=["surface", "volume"]).copy()
        if not tmp.empty:
            fig = create_scatter_with_trend(
                tmp, "surface", "volume", "code_couleur",
                "Surface vs Volume (avec tendance lineaire)",
                COLOR_MAP
            )
            st.plotly_chart(fig, width="stretch")

    with tab_q2:
        st.subheader("Relation Rendement - Volume")
        tmp = base.dropna(subset=["rendement", "volume"]).copy()
        if not tmp.empty:
            fig = create_scatter_with_trend(
                tmp, "rendement", "volume", "code_couleur",
                "Rendement vs Volume (avec tendance lineaire)",
                COLOR_MAP
            )
            st.plotly_chart(fig, width="stretch")

    with tab_q3:
        st.subheader("Productivite par zone")
        tmp = base.dropna(subset=["prod_hl_ha", "Zone"]).copy()
        if not tmp.empty:
            prod_zone = tmp.groupby("Zone")["prod_hl_ha"].agg(['mean', 'std', 'count']).reset_index()
            prod_zone = prod_zone.sort_values("Zone", key=lambda s: s.map(zone_sort_key))

            fig = create_comparison_bar_chart(
                prod_zone, "Zone", "mean", "Zone",
                "Productivite moyenne par zone (hl/ha)",
                ZONE_COLOR_MAP
            )
            fig.update_traces(error_y=dict(type='data', array=prod_zone['std']))
            st.plotly_chart(fig, width="stretch")

    with tab_q4:
        st.subheader("Volatilite du rendement par zone")
        tmp = base.dropna(subset=["rendement", "Zone"]).copy()
        if not tmp.empty:
            vol_zone = tmp.groupby("Zone")["rendement"].std().reset_index()
            vol_zone = vol_zone.sort_values("Zone", key=lambda s: s.map(zone_sort_key))

            fig = create_comparison_bar_chart(
                vol_zone, "Zone", "rendement", "Zone",
                "Ecart-type du rendement par zone",
                ZONE_COLOR_MAP
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    with tab_q5:
        st.subheader("Tableau de bord analytique")

        summary = base.groupby("Zone").agg({
            'surface': 'sum',
            'volume': 'sum',
            'rendement': 'mean',
            'prod_hl_ha': 'mean'
        }).round(2).reset_index()

        summary['% volume'] = (summary['volume'] / summary['volume'].sum() * 100).round(1)
        summary = summary.sort_values("Zone", key=lambda s: s.map(zone_sort_key))

        st.dataframe(
            summary,
            column_config={
                "Zone": "Zone",
                "surface": st.column_config.NumberColumn("Surface (ha)", format="%.0f"),
                "volume": st.column_config.NumberColumn("Volume (hl)", format="%.0f"),
                "rendement": st.column_config.NumberColumn("Rendement moy", format="%.1f"),
                "prod_hl_ha": st.column_config.NumberColumn("Productivite", format="%.1f"),
                "% volume": st.column_config.NumberColumn("% Volume", format="%.1f%%")
            },
            width="stretch",
            hide_index=True
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()
st.caption(f"Analyse viticole - Donnees mises a jour • {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")

