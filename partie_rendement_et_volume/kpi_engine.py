# kpi_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_global_kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"prod_global": np.nan, "corr_vr": np.nan, "vol_rdt": np.nan, "share_hors": np.nan}

    vol = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce")
    surf = pd.to_numeric(df.get("surface", pd.Series(dtype=float)), errors="coerce")
    rdt = pd.to_numeric(df.get("rendement", pd.Series(dtype=float)), errors="coerce")

    prod_global = (vol.sum() / surf.sum()) if surf.sum() > 0 else np.nan

    tmp = df[["volume", "rendement"]].copy() if {"volume", "rendement"}.issubset(df.columns) else pd.DataFrame()
    if not tmp.empty:
        tmp["volume"] = pd.to_numeric(tmp["volume"], errors="coerce")
        tmp["rendement"] = pd.to_numeric(tmp["rendement"], errors="coerce")
        tmp = tmp.dropna()
        corr_vr = tmp.corr().iloc[0, 1] if len(tmp) > 2 else np.nan
    else:
        corr_vr = np.nan

    vol_rdt = float(rdt.std()) if rdt.notna().any() else np.nan

    if "statut_plafond" in df.columns and df["statut_plafond"].notna().any():
        share_hors = float(df["statut_plafond"].eq("Hors plafond").mean() * 100)
    else:
        share_hors = np.nan

    return {
        "prod_global": float(prod_global) if pd.notna(prod_global) else np.nan,
        "corr_vr": float(corr_vr) if pd.notna(corr_vr) else np.nan,
        "vol_rdt": float(vol_rdt) if pd.notna(vol_rdt) else np.nan,
        "share_hors": float(share_hors) if pd.notna(share_hors) else np.nan,
    }


def compute_zone_efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    needed = {"Zone", "surface", "volume", "rendement", "plafond", "statut_plafond"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    tmp = df.copy()
    tmp["surface"] = pd.to_numeric(tmp["surface"], errors="coerce")
    tmp["volume"] = pd.to_numeric(tmp["volume"], errors="coerce")
    tmp["rendement"] = pd.to_numeric(tmp["rendement"], errors="coerce")
    tmp["plafond"] = pd.to_numeric(tmp["plafond"], errors="coerce")

    tmp = tmp.dropna(subset=["Zone"])

    tmp["prod"] = np.where(tmp["surface"] > 0, tmp["volume"] / tmp["surface"], np.nan)

    out = (
        tmp.groupby("Zone")
        .agg(
            surface_ha=("surface", "sum"),
            volume_hl=("volume", "sum"),
            rendement_moy=("rendement", "mean"),
            plafond_moy=("plafond", "mean"),
            prod_hl_ha=("prod", "mean"),
            pct_hors=("statut_plafond", lambda s: (s == "Hors plafond").mean() * 100),
        )
        .reset_index()
    )
    return out