# modules/utils.py
"""
Fonctions utilitaires
"""

import numpy as np
import pandas as pd
import json
import hashlib
from typing import List, Dict, Any


def to_float_series(s: pd.Series) -> pd.Series:
    """
    Convertit une série en float de manière robuste
    Gère les virgules, les valeurs manquantes, etc.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s2 = (
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .replace({"nan": None, "None": None, "": None})
    )
    return pd.to_numeric(s2, errors="coerce")


def zone_sort_key(z: str) -> int:
    """Clé de tri pour les zones (ordre numérique)"""
    try:
        return int(z)
    except Exception:
        return 999


def zone_order_from_series(s: pd.Series) -> List[str]:
    """Retourne les zones dans l'ordre numérique"""
    zs = [str(x) for x in s.dropna().unique()]
    return sorted(zs, key=zone_sort_key)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1500, seed: int = 42) -> tuple:
    """
    Calcule l'intervalle de confiance à 95% par bootstrap
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 5:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def generate_context_hash(data: pd.DataFrame, context: str) -> str:
    """Génère un hash unique pour un contexte d'analyse"""
    data_summary = f"{len(data)}_{data['annee'].min()}_{data['annee'].max()}_{context}"
    return hashlib.md5(data_summary.encode()).hexdigest()


def calculate_gini(values: np.ndarray) -> float:
    """Calcule le coefficient de Gini"""
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))


def format_number(n: float, decimals: int = 0) -> str:
    """Formate un nombre avec séparateur de milliers"""
    if pd.isna(n):
        return "N/A"
    return f"{n:,.{decimals}f}".replace(",", " ")


def complete_years(
    df_in: pd.DataFrame, 
    group_col: str, 
    value_col: str, 
    aggfunc: str,
    years_min: int, 
    years_max: int
) -> pd.DataFrame:
    """
    Force une continuité temporelle (toutes les années) par groupe
    """
    if df_in.empty:
        return df_in

    yrs = np.arange(years_min, years_max + 1)
    groups = sorted(df_in[group_col].dropna().unique())

    out_rows = []
    for g in groups:
        sub = df_in[df_in[group_col] == g].groupby("annee")[value_col].agg(aggfunc)
        sub = sub.reindex(yrs)  # années manquantes -> NaN
        tmp = pd.DataFrame({"annee": yrs, group_col: g, value_col: sub.values})
        out_rows.append(tmp)

    return pd.concat(out_rows, ignore_index=True)