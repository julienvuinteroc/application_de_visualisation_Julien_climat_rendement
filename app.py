# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.data_loader import load_geojson
from config.constants import ZONE_COLOR_MAP, ZONE_LABELS, DEPARTEMENTS

st.set_page_config(
    page_title="Observatoire Viticole - Pays d'Oc",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Style CSS personnalise
st.markdown("""
<style>
    /* Global */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
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


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-header">
    <h1>Observatoire Viticole - Pays d'Oc IGP</h1>
    <p>Analyse du climat, du rendement et de leurs interactions</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# STATS RAPIDES
# =====================================================

st.markdown("### Apercu du territoire")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">7</div>
        <div class="stat-label">Zones pedoclimatiques</div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">4</div>
        <div class="stat-label">Departements</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">58</div>
        <div class="stat-label">Cepages suivis</div>
    </div>
    """, unsafe_allow_html=True)

with col_s4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">2007-2024</div>
        <div class="stat-label">Periode d'analyse</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# =====================================================
# BOUTONS CARTES
# =====================================================

st.markdown("### Cartes de reference")

col_map1, col_map2 = st.columns(2)

with col_map1:
    if st.button("Afficher la carte des zones pedoclimatiques", key="btn_zone_map", use_container_width=True):
        show_zone_popup()

with col_map2:
    if st.button("Afficher la carte des departements", key="btn_dept_map", use_container_width=True):
        show_dept_popup()

st.markdown("---")


# =====================================================
# POPUP CARTE DES ZONES - VERSION CORRIGEE
# =====================================================

if st.session_state.show_zone_map:
    with st.container():
        st.markdown('<div class="popup-overlay">', unsafe_allow_html=True)
        st.markdown('<div class="popup-content">', unsafe_allow_html=True)
        
        col_close, col_title = st.columns([1, 10])
        with col_close:
            if st.button("X", key="close_zone_btn"):
                close_popup("show_zone_map")
        with col_title:
            st.markdown("### Carte des zones pedoclimatiques")
        
        try:
            geo_zones = load_geojson("zones")
            
            if geo_zones is not None and 'features' in geo_zones:
                # Preparation des donnees par zone
                zone_data = {}
                for feature in geo_zones['features']:
                    zone_id = str(feature['properties'].get('zone', '0'))
                    code_commune = feature['properties'].get('code_commune')
                    if zone_id not in zone_data:
                        zone_data[zone_id] = []
                    zone_data[zone_id].append(code_commune)
                
                # Creation de la carte avec Choroplethmap (nouvelle version)
                fig_zone = go.Figure()
                
                for zone_id, communes in zone_data.items():
                    if zone_id != "0":
                        color = ZONE_COLOR_MAP.get(zone_id, "#808080")
                        # Utilisation de Choroplethmap au lieu de Choroplethmapbox
                        fig_zone.add_trace(go.Choroplethmap(
                            geojson=geo_zones,
                            locations=communes,
                            z=[1] * len(communes),
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            name=f"Zone {zone_id}",
                            marker_opacity=0.7,
                            marker_line_width=0.5,
                            hovertemplate=f"Zone {zone_id}<extra></extra>"
                        ))
                
                fig_zone.update_layout(
                    map_style="carto-positron",
                    map_zoom=7,
                    map_center={"lat": 43.6, "lon": 3.5},
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=0.02,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                )
                
                st.plotly_chart(fig_zone, key="zone_popup_map", use_container_width=True)
                
                # Legende
                st.markdown("#### Legende des zones")
                
                cols = st.columns(2)
                zone_list = [z for z in ZONE_LABELS.keys() if z != "0"]
                for i, zone_id in enumerate(zone_list):
                    color = ZONE_COLOR_MAP.get(str(zone_id), "#808080")
                    label = ZONE_LABELS.get(str(zone_id), f"Zone {zone_id}")
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div class="legend-item">
                            <div class="legend-color" style="background: {color};"></div>
                            <span><b>Zone {zone_id}</b> : {label}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Impossible de charger la carte des zones")
                
        except Exception as e:
            st.error(f"Erreur lors du chargement de la carte: {e}")
        
        if st.button("Fermer", key="close_zone", use_container_width=True):
            close_popup("show_zone_map")
        
        st.markdown('</div></div>', unsafe_allow_html=True)


# =====================================================
# POPUP CARTE DES DEPARTEMENTS - VERSION CORRIGEE
# =====================================================

if st.session_state.show_dept_map:
    with st.container():
        st.markdown('<div class="popup-overlay">', unsafe_allow_html=True)
        st.markdown('<div class="popup-content">', unsafe_allow_html=True)
        
        col_close, col_title = st.columns([1, 10])
        with col_close:
            if st.button("X", key="close_dept_btn"):
                close_popup("show_dept_map")
        with col_title:
            st.markdown("### Carte des departements")
        
        try:
            geo_depts = load_geojson("departements")
            
            if geo_depts is not None and 'features' in geo_depts:
                # Preparation des donnees par departement
                dept_colors = {
                    "11": "#3498db",
                    "30": "#2ecc71",
                    "34": "#e74c3c",
                    "66": "#f39c12",
                }
                
                dept_names = {
                    "11": "Aude (11)",
                    "30": "Gard (30)",
                    "34": "Herault (34)",
                    "66": "Pyrenees-Orientales (66)"
                }
                
                dept_data = {}
                for feature in geo_depts['features']:
                    dept_id = feature['properties'].get('dep')
                    code_commune = feature['properties'].get('code_commune')
                    if dept_id and dept_id in DEPARTEMENTS:
                        if dept_id not in dept_data:
                            dept_data[dept_id] = []
                        dept_data[dept_id].append(code_commune)
                
                # Creation de la carte avec Choroplethmap
                fig_dept = go.Figure()
                
                for dept_id, communes in dept_data.items():
                    color = dept_colors.get(dept_id, "#95a5a6")
                    name = dept_names.get(dept_id, dept_id)
                    
                    fig_dept.add_trace(go.Choroplethmap(
                        geojson=geo_depts,
                        locations=communes,
                        z=[1] * len(communes),
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        name=name,
                        marker_opacity=0.7,
                        marker_line_width=0.5,
                        hovertemplate=f"{name}<extra></extra>"
                    ))
                
                fig_dept.update_layout(
                    map_style="carto-positron",
                    map_zoom=7,
                    map_center={"lat": 43.6, "lon": 3.5},
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                )
                
                st.plotly_chart(fig_dept, key="dept_popup_map", use_container_width=True)
                
                # Legende
                st.markdown("#### Legende des departements")
                
                cols = st.columns(2)
                for i, dept in enumerate(DEPARTEMENTS):
                    color = dept_colors.get(dept, "#95a5a6")
                    name = dept_names.get(dept, dept)
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div class="legend-item">
                            <div class="legend-color" style="background: {color};"></div>
                            <span><b>{name}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Impossible de charger la carte des departements")
                
        except Exception as e:
            st.error(f"Erreur lors du chargement de la carte: {e}")
        
        if st.button("Fermer", key="close_dept", use_container_width=True):
            close_popup("show_dept_map")
        
        st.markdown('</div></div>', unsafe_allow_html=True)


# =====================================================
# NAVIGATION PRINCIPALE - CENTREE
# =====================================================

st.markdown("### Analyses disponibles")

col_left, col_center1, col_center2, col_center3, col_right = st.columns([1, 1, 1, 1, 1])

with col_center1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🌡️</div>
        <h3>Climat</h3>
        <p>Analyse des indicateurs climatiques</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Acceder a Climat", key="nav_climat", use_container_width=True):
        st.switch_page("pages/1_Climat.py")

with col_center2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🍇</div>
        <h3>Production</h3>
        <p>Analyse des rendements et volumes</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Acceder a Rendement", key="nav_rendement", use_container_width=True):
        st.switch_page("pages/2_Rendement.py")

with col_center3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📈</div>
        <h3>Climat-Production</h3>
        <p>Analyse croisee et correlations</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Acceder a Climat-Production", key="nav_climat_rendement", use_container_width=True):
        st.switch_page("pages/3_Climat_Rendement.py")


# =====================================================
# INFORMATIONS COMPLEMENTAIRES
# =====================================================

st.markdown("---")
st.markdown("### A propos")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **Sources de donnees**
    - Donnees climatiques : Meteo France
    - Donnees viticoles : Declarations de recolte (DECR/REVE)
    - Periode : 2000 - 2024
    """)

with col_info2:
    st.markdown("""
    **Methodologie**
    - Zones pedoclimatiques : classification basee sur le terroir
    - Scoring intelligent : ponderation multi-criteres
    - Clustering : segmentation automatique des zones
    """)

st.markdown("---")

# Footer
st.markdown("""
<div class="footer">
    <p>Observatoire Viticole - Pays d'Oc IGP | Donnees mises a jour regulierement</p>
    <p style="font-size: 0.75rem;">(c) 2024 - Analyse du climat et du rendement viticole</p>
</div>
""", unsafe_allow_html=True)