# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# =====================================================
# CLEAN NUMERIC
# =====================================================

def to_numeric_safe(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."),
                errors="coerce"
            )

    return df


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def build_features(df, rolling_window=5):
    df = df.copy()

    df = df.sort_values(["Municipality", "Year"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # rolling mean
    for col in numeric_cols:
        if col not in ["Year"]:
            df[f"{col}_rollmean"] = (
                df.groupby("Municipality")[col]
                .rolling(rolling_window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df


# =====================================================
# PREPARE DATA FOR ONE COMMUNE
# =====================================================

def prepare_commune_history(df, municipality, ref_start, ref_end):
    df = df.copy()

    df = df[df["Municipality"] == str(municipality)]

    df = df[
        (df["Year"] >= ref_start) &
        (df["Year"] <= ref_end)
    ]

    if df.empty:
        raise ValueError(f"Aucune donnée historique pour {municipality}")

    df = to_numeric_safe(df)

    return df


# =====================================================
# MAIN ML FUNCTION
# =====================================================

def predict_agronomic_indicators(
    model_A_dict,
    metrics_A,
    year,
    municipality,
    df_history,
    rolling_window=5,
    ref_start=2007,
    ref_end=2024,
):

    # =====================================================
    # 1. FILTER COMMUNE
    # =====================================================

    df_commune = prepare_commune_history(
        df_history,
        municipality,
        ref_start,
        ref_end
    )

    # =====================================================
    # 2. FEATURE ENGINEERING
    # =====================================================

    df_features = build_features(df_commune, rolling_window)

    # dernière ligne = état actuel
    last_row = df_features.sort_values("Year").iloc[-1:].copy()

    # année cible
    last_row["Year"] = int(year)

    # =====================================================
    # 3. PREDICTIONS
    # =====================================================

    predictions = {}

    for target, model in model_A_dict.items():

        # récupérer les features attendues
        if hasattr(model, "feature_names_in_"):
            features = model.feature_names_in_
        else:
            # fallback
            features = last_row.columns

        X = last_row.reindex(columns=features, fill_value=0)

        try:
            y_pred = model.predict(X)[0]
        except Exception:
            y_pred = np.nan

        predictions[target] = y_pred

    # =====================================================
    # 4. FORMAT RESULT
    # =====================================================

    df_result = pd.DataFrame([predictions])

    df_result["Municipality"] = municipality
    df_result["Year"] = year

    # arrondir
    for col in df_result.columns:
        if df_result[col].dtype in [float, int]:
            df_result[col] = df_result[col].round(2)

    # =====================================================
    # 5. AJOUT METRICS (optionnel)
    # =====================================================

    if metrics_A is not None:
        for key, val in metrics_A.items():
            df_result[f"{key}_score"] = val

    return df_result