# advanced_analysis_engine.py

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.signal import find_peaks


def detect_crossings(df, x_col, y_col, group_col):
    """
    Détecte les croisements entre séries.
    """
    crossings = []
    groups = df[group_col].unique()

    for g1, g2 in combinations(groups, 2):
        s1 = df[df[group_col] == g1].sort_values(x_col)
        s2 = df[df[group_col] == g2].sort_values(x_col)

        merged = pd.merge(s1[[x_col, y_col]], s2[[x_col, y_col]],
                          on=x_col, suffixes=("_1", "_2"))

        diff = merged[f"{y_col}_1"] - merged[f"{y_col}_2"]

        sign_change = np.sign(diff).diff()

        for i in sign_change[sign_change != 0].index:
            year = merged.loc[i, x_col]
            crossings.append(
                f"Croisement détecté entre {g1} et {g2} en {year}"
            )

    return crossings


def detect_structural_break(df, x_col, y_col, group_col):
    """
    Détecte une rupture via variation forte de pente.
    """
    breaks = []
    for g in df[group_col].unique():
        sub = df[df[group_col] == g].sort_values(x_col)
        y = sub[y_col].values

        if len(y) < 6:
            continue

        slopes = np.diff(y)
        slope_var = np.abs(np.diff(slopes))

        if len(slope_var) > 0:
            idx = np.argmax(slope_var)
            if slope_var[idx] > np.std(slopes) * 1.5:
                year = sub.iloc[idx + 1][x_col]
                breaks.append(
                    f"Rupture structurelle probable pour {g} en {year}"
                )

    return breaks


def detect_abnormal_volatility(df, x_col, y_col, group_col):
    """
    Détecte une volatilité anormalement élevée.
    """
    alerts = []
    global_std = df[y_col].std()

    for g in df[group_col].unique():
        sub = df[df[group_col] == g]
        local_std = sub[y_col].std()

        if local_std > global_std * 1.5:
            alerts.append(
                f"Volatilité anormalement élevée pour {g}"
            )

    return alerts


def detect_turning_points(df, x_col, y_col, group_col):
    """
    Identifie pics et creux majeurs.
    """
    turning_points = []

    for g in df[group_col].unique():
        sub = df[df[group_col] == g].sort_values(x_col)
        y = sub[y_col].values

        if len(y) < 5:
            continue

        peaks, _ = find_peaks(y)
        troughs, _ = find_peaks(-y)

        for p in peaks:
            year = sub.iloc[p][x_col]
            turning_points.append(f"Pic majeur pour {g} en {year}")

        for t in troughs:
            year = sub.iloc[t][x_col]
            turning_points.append(f"Creux majeur pour {g} en {year}")

    return turning_points


def detect_cycles(df, x_col, y_col, group_col):
    """
    Analyse simple des cycles via autocorrélation.
    """
    cycles = []

    for g in df[group_col].unique():
        sub = df[df[group_col] == g].sort_values(x_col)
        y = sub[y_col].values

        if len(y) < 8:
            continue

        autocorr = pd.Series(y).autocorr(lag=2)

        if autocorr > 0.5:
            cycles.append(
                f"Cycle probable (~2 ans) détecté pour {g}"
            )

    return cycles


def full_structural_analysis(df, x_col, y_col, group_col):
    report = []

    report.append("## Analyse structurelle avancée\n")

    report += detect_crossings(df, x_col, y_col, group_col)
    report += detect_structural_break(df, x_col, y_col, group_col)
    report += detect_abnormal_volatility(df, x_col, y_col, group_col)
    report += detect_turning_points(df, x_col, y_col, group_col)
    report += detect_cycles(df, x_col, y_col, group_col)

    if len(report) == 1:
        report.append("Aucune anomalie structurelle majeure détectée.")

    return "\n".join(report)
