# modules/data_loader.py

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.db import get_conn
from config.constants import GEOJSON_FILES


# =====================================================
# CONNEXION
# =====================================================

@st.cache_resource
def get_connection():
    """
    Connexion DuckDB partagée pour toute l'app.
    ⚠️ Ne jamais fermer cette connexion.
    """
    return get_conn()


# =====================================================
# LOAD DATA PRINCIPAL
# =====================================================

@st.cache_data
def load_data():
    """
    Charge la table fusion (climat + rendement).
    """

    conn = get_connection()

    try:
        df = conn.execute("SELECT * FROM fusion").df()
    except Exception as e:
        st.error(f"Erreur chargement données : {e}")
        st.stop()

    # =================================================
    # NETTOYAGE STRUCTURE
    # =================================================

    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # =================================================
    # NORMALISATION COLONNES
    # =================================================

    rename_map = {}

    if "zone" in df.columns and "Zone" not in df.columns:
        rename_map["zone"] = "Zone"

    if "departement" in df.columns and "code_departement" not in df.columns:
        rename_map["departement"] = "code_departement"

    if rename_map:
        df = df.rename(columns=rename_map)

    # =================================================
    # CONVERSION NUMÉRIQUE
    # =================================================

    numeric_cols = [
        "rendement",
        "surface",
        "volume",
        "annee",
        "temp_moyenne",
        "precipitation_total",
        "Hot_D",
        "Very_Hot_D",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Colonnes avec virgule
    special_numeric_cols = [
        "Huglin_Index",
        "Climatic_Dryness_Index",
    ]

    for col in special_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    # =================================================
    # TEXTE
    # =================================================

    string_cols = [
        "code_departement",
        "Zone",
        "commune",
        "code_couleur",
        "code_cepage",
        "type_mvt",
        "cvi",
    ]

    for col in string_cols:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = df[col].astype("string").str.strip()

    # =================================================
    # ANNÉE
    # =================================================

    if "annee" in df.columns:
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")

    # =================================================
    # LOGIQUE MÉTIER
    # =================================================

    if "code_couleur" in df.columns and "rendement" in df.columns:
        df["plafond"] = df["code_couleur"].map({
            "BL": 90,
            "RG": 90,
            "RS": 100
        })

        df["statut_plafond"] = "Conforme"

        mask_valid = df["rendement"].notna() & df["plafond"].notna()

        df.loc[
            mask_valid & (df["rendement"] > df["plafond"]),
            "statut_plafond"
        ] = "Hors plafond"

    return df


# =====================================================
# GEOJSON
# =====================================================

@st.cache_data
def load_geojson(file_key: str):
    """
    Charge un GeoJSON depuis les constantes du projet.
    """

    try:
        file_path = GEOJSON_FILES.get(file_key)

        if file_path is None:
            st.error(f"Clé GeoJSON inconnue : {file_key}")
            return None

        file_path = Path(file_path)

        if not file_path.exists():
            st.error(f"Fichier introuvable : {file_key}")
            st.error(f"Chemin recherché : {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        return geojson

    except Exception as e:
        st.error(f"Erreur chargement GeoJSON ({file_key}) : {e}")
        return None


# =====================================================
# FILTRES
# =====================================================

def apply_filters(df, departements, zones, couleurs, annees):
    """
    Applique les filtres sélectionnés.
    """

    required_cols = ["code_departement", "Zone", "code_couleur", "annee"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    annee_series = pd.to_numeric(df["annee"], errors="coerce")

    mask = (
        df["code_departement"].isin(departements)
        & df["Zone"].isin(zones)
        & df["code_couleur"].isin(couleurs)
        & annee_series.between(annees[0], annees[1])
    )

    return df.loc[mask].copy()