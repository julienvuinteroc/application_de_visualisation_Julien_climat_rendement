# modules/visualization.py
"""
Fonctions de visualisation pour les graphiques
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from config.constants import COLOR_MAP, ZONE_COLOR_MAP


def create_rendement_chart(
    data: pd.DataFrame, 
    color_col: str, 
    mode: str,
    show_plafonds: bool = True
) -> go.Figure:
    """
    Crée un graphique d'évolution des rendements
    """
    fig = px.line(
        data,
        x="annee", 
        y="rendement",
        color=color_col,
        markers=True,
        title="Évolution des rendements"
    )
    
    # Ajout des plafonds réglementaires
    if show_plafonds:
        fig.add_hline(
            y=90, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Plafond BL/RG (90 hl/ha)",
            annotation_position="bottom right"
        )
        fig.add_hline(
            y=100, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Plafond RS (100 hl/ha)",
            annotation_position="top right"
        )
    
    fig.update_layout(
        yaxis=dict(range=[0, 110], dtick=10),
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    
    return fig


def create_volume_chart(
    data: pd.DataFrame,
    color_col: str,
    mode: str
) -> go.Figure:
    """
    Crée un graphique d'évolution des volumes
    """
    fig = px.line(
        data,
        x="annee", 
        y="volume",
        color=color_col,
        markers=True,
        title="Évolution des volumes"
    )
    
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Volume (hl)",
        height=500
    )
    
    return fig


def create_comparison_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    color_map: Optional[Dict] = None
) -> go.Figure:
    """
    Crée un graphique à barres comparatif
    """
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode="group",
        color_discrete_map=color_map,
        title=title
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title=y_col,
        height=400
    )
    
    return fig


def create_pie_chart(
    data: pd.DataFrame,
    values_col: str,
    names_col: str,
    title: str,
    color_map: Optional[Dict] = None
) -> go.Figure:
    """
    Crée un graphique en camembert
    """
    fig = go.Figure(data=[go.Pie(
        labels=data[names_col],
        values=data[values_col],
        hole=0.3,
        marker=dict(colors=[color_map.get(str(x), "#CCCCCC") for x in data[names_col]] if color_map else None)
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400
    )
    
    return fig


def create_scatter_with_trend(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    color_map: Optional[Dict] = None,
    trendline: str = "ols"
) -> go.Figure:
    """
    Crée un nuage de points avec ligne de tendance
    """
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        trendline=trendline,
        title=title,
        hover_data=["annee", "code_departement", "Zone"]
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig


def create_choropleth_map(
    data: pd.DataFrame,
    geojson: Dict,
    locations_col: str,
    color_col: str,
    title: str,
    color_scale: str = "RdYlGn",
    labels: Optional[Dict] = None,
    animation_frame: Optional[str] = None
) -> go.Figure:
    """
    Crée une carte choroplèthe avec légende améliorée
    """
    # Déterminer la clé feature selon le niveau
    if locations_col == "code_departement":
        featureidkey = "properties.dep"
    else:
        featureidkey = "properties.code_commune"
    
    fig = px.choropleth(
        data,
        geojson=geojson,
        locations=locations_col,
        featureidkey=featureidkey,
        color=color_col,
        animation_frame=animation_frame,
        color_continuous_scale=color_scale,
        range_color=[data[color_col].quantile(0.05), data[color_col].quantile(0.95)] if len(data) > 5 else None,
        labels=labels or {color_col: color_col},
        title=title
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        template="plotly_white",
        height=500,
        coloraxis_colorbar=dict(
            title=labels.get(color_col, color_col) if labels else color_col,
            len=0.5,
            thickness=15
        )
    )
    
    return fig


def add_map_legend(fig, title, unit, stats):
    """
    Ajoute une légende sous la carte avec les statistiques
    """
    fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=f"<b>{title}</b><br>"
                     f"Moyenne: {stats['mean']:.1f} {unit} | "
                     f"Médiane: {stats['median']:.1f} {unit} | "
                     f"Min: {stats['min']:.1f} {unit} | "
                     f"Max: {stats['max']:.1f} {unit}",
                showarrow=False,
                font=dict(size=12),
                align="center",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgray",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    return fig


def create_anomaly_chart(
    data: pd.DataFrame,
    x_col: str = "annee",
    y_col: str = "rendement",
    color_col: str = "classe",
    size_col: Optional[str] = None
) -> go.Figure:
    """
    Crée un graphique des anomalies détectées
    """
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_data=["Zone", "zscore"],
        title="Anomalies détectées par année"
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig