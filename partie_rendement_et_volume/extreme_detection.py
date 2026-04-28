# extreme_detection.py
from __future__ import annotations

import pandas as pd
import numpy as np


def detect_extreme_events(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Détection d'événements extrêmes via z-score annuel (rendement moyen par zone).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not {"annee", "Zone", "rendement"}.issubset(df.columns):
        return pd.DataFrame()

    tmp = df.copy()
    tmp["annee"] = pd.to_numeric(tmp["annee"], errors="coerce")
    tmp["rendement"] = pd.to_numeric(tmp["rendement"], errors="coerce")
    tmp = tmp.dropna(subset=["annee", "Zone", "rendement"])

    grouped = tmp.groupby(["annee", "Zone"])["rendement"].mean().reset_index()

    stats = grouped.groupby("annee")["rendement"].agg(["mean", "std"]).reset_index()
    grouped = grouped.merge(stats, on="annee", how="left")

    grouped["std"] = grouped["std"].replace(0, np.nan)
    grouped["zscore"] = (grouped["rendement"] - grouped["mean"]) / grouped["std"]

    extremes = grouped[grouped["zscore"].abs() >= z_threshold].copy()
    extremes = extremes.sort_values("zscore", key=lambda s: s.abs(), ascending=False)

    return extremes[["annee", "Zone", "rendement", "zscore"]]