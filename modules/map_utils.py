# modules/map_utils.py
"""
Utilitaires spécifiques pour les cartes
"""
import streamlit as st
import plotly.express as px
from modules.data_loader import load_geojson


def prepare_map_data(df, df_all, indicator, level, year, couleurs):
    """
    Prépare les données pour une carte avec vérification des codes communes
    """
    
    # Mapping des indicateurs
    indicator_map = {
        "Rendement": {"var": "rendement", "mvt": "DECR", "agg": "mean", "unit": "hl/ha", "scale": "RdYlGn"},
        "Volume": {"var": "volume", "mvt": "REVE", "agg": "sum", "unit": "hl", "scale": "Blues"},
        "Surface": {"var": "surface", "mvt": "DECR", "agg": "sum", "unit": "ha", "scale": "Greens"}
    }
    
    var_info = indicator_map[indicator]
    
    # Filtrage des données
    map_data = df[
        (df["type_mvt"] == var_info["mvt"]) &
        (df["annee"] == year) &
        (df["code_couleur"].isin(couleurs))
    ].copy()
    
    if map_data.empty:
        return None
    
    try:
        # Configuration selon le niveau
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
            
        else:  # Commune
            geo = load_geojson("communes")
            if geo is None:
                st.error("Fichier communes.geojson non trouve")
                return None
            
            # Agrégation par commune
            map_display = map_data.groupby("commune")[var_info["var"]].agg(var_info["agg"]).reset_index()
            
            # Vérifier que les codes communes sont au bon format (5 chiffres)
            map_display["commune"] = map_display["commune"].astype(str).str.zfill(5)
            
            # Extraire les codes communes du GeoJSON pour vérification
            geo_codes = []
            if 'features' in geo:
                for feature in geo['features']:
                    if 'properties' in feature and 'code_commune' in feature['properties']:
                        code = str(feature['properties']['code_commune']).zfill(5)
                        geo_codes.append(code)
            
            # Filtrer pour ne garder que les communes présentes dans le GeoJSON
            initial_count = len(map_display)
            map_display = map_display[map_display["commune"].isin(geo_codes)]
            
            locations = "commune"
            feature_key = "properties.code_commune"
            title_level = "commune"
            
            if initial_count > 0:
                st.info(f"Donnees disponibles pour {len(map_display)} communes sur {initial_count}")
        
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


def display_map_with_legend(data, title_suffix="", height=500, key=None):
    """
    Affiche une carte avec sa légende statistique
    """
    if data is None or data['map_display'].empty:
        st.warning("Aucune donnee a afficher")
        return None
    
    # Créer la carte
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
        hover_data={data['locations']: True, data['var']: ':.0f'}
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
            y=0.5,
            tickformat=",.0f",
            separatethousands=True
        )
    )
    
    # Ajouter la légende statistique
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
                         f"Moyenne: {stats['mean']:.0f} {data['unit']} | "
                         f"Mediane: {stats['median']:.0f} {data['unit']}<br>"
                         f"Minimum: {stats['min']:.0f} {data['unit']} | "
                         f"Maximum: {stats['max']:.0f} {data['unit']}",
                    showarrow=False,
                    font=dict(size=15),
                    align="center",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="lightgray",
                    borderwidth=1,
                    borderpad=6
                )
            ]
        )
    
    st.plotly_chart(fig, use_container_width=True, key=key)
    
    # Afficher un tableau des 10 premières valeurs
    with st.expander("Voir les donnees detailles"):
        df_format = data['map_display'].copy()
        df_format[data['var']] = df_format[data['var']].round(0)
        st.dataframe(
            df_format.sort_values(data['var'], ascending=False).head(10),
            use_container_width=True,
            hide_index=True
        )
    
    return fig


def compare_two_maps(data1, data2, title1, title2):
    """
    Affiche deux cartes côte à côte avec leurs légendes
    """
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"**{title1}**")
        if data1:
            display_map_with_legend(data1, title_suffix=title1, height=400, key=f"map_left_{title1}")
        else:
            st.warning("Carte 1 non disponible")
    
    with col_right:
        st.markdown(f"**{title2}**")
        if data2:
            display_map_with_legend(data2, title_suffix=title2, height=400, key=f"map_right_{title2}")
        else:
            st.warning("Carte 2 non disponible")
    
    # Statistiques comparatives
    if data1 and data2 and not data1['map_display'].empty and not data2['map_display'].empty:
        st.subheader("Comparaison statistique")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        vals1 = data1['map_display'][data1['var']].dropna()
        vals2 = data2['map_display'][data2['var']].dropna()
        
        if not vals1.empty and not vals2.empty:
            with col_s1:
                if data1['indicator'] == "Rendement":
                    st.metric(
                        f"{data1['indicator']} moyen carte 1",
                        f"{vals1.mean():.1f} {data1['unit']}"
                    )
                if data1['indicator'] == "Volume":
                    st.metric(
                        f"{data1['indicator']} moyen carte 1",
                        f"{vals1.mean():.0f} {data1['unit']}"
                    )
                if data1['indicator'] == "Surface":
                    st.metric(
                        f"{data1['indicator']} moyenne carte 1",
                        f"{vals1.mean():.0f} {data1['unit']}"
                    )
            with col_s2:
                if data2['indicator'] == "Rendement":
                    st.metric(
                        f"{data2['indicator']} moyen carte 2",
                        f"{vals2.mean():.1f} {data2['unit']}"
                    )
                if data2['indicator'] == "Volume":
                    st.metric(
                        f"{data2['indicator']} moyen carte 2",
                        f"{vals2.mean():.0f} {data2['unit']}"
                    )
                if data2['indicator'] == "Surface":
                    st.metric(
                        f"{data2['indicator']} moyenne carte 2",
                        f"{vals2.mean():.0f} {data2['unit']}"
                    )
            with col_s3:
                diff = vals1.mean() - vals2.mean()
                if data1['indicator'] == "Rendement":
                    st.metric(
                        "Difference entre les 2 cartes",
                        f"{diff:.1f} hl/ha"
                    )
                if data1['indicator'] == "Volume":
                    st.metric(
                        "Difference entre les 2 cartes",
                        f"{diff:.0f} hl"
                    )
                if data1['indicator'] == "Surface":
                    st.metric(
                        "Difference entre les 2 cartes",
                        f"{diff:.0f} ha"
                    )