import pandas as pd
import numpy as np


# =========================
# CLEANING UTILITAIRE
# =========================
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # supprimer colonnes inutiles
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # uniformiser noms
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    rename_map = {
        "Temperature_moyenne": "temp_moyenne",
        "Temperature_maximale": "tmax",
        "Temperature_minimale": "tmin",
        "Pluie_quotidienne": "precipitation",
        "Index_Huglin": "Huglin_Index",
    }

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    return df


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "."),
            errors="coerce"
        )
    return df


# =========================
# INDICATEURS CNIV
# =========================
def indicators_2012_2025(cniv_file: str):
    df = pd.read_csv(cniv_file, sep=";", encoding="utf-8", comment="#")

    df = _clean_columns(df)
    df = _to_numeric(df)

    # date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values(["Municipality", "date"]).copy()

    # =========================
    # CONDITIONS
    # =========================
    wine_cycle = df["date"].dt.month.isin([4, 5, 6, 7, 8, 9])
    dryness_cycle = df["date"].dt.month.isin([4, 5, 6, 7, 8])
    huglin_condition = df["temp_moyenne"] > 10

    K = 1.03

    # =========================
    # INITIALISATION
    # =========================
    df["Huglin_Index"] = 0.0
    df["Hot_D"] = 0
    df["Very_Hot_D"] = 0
    df["Late_Frost"] = 0
    df["Severe_Heat"] = 0.0
    df["Severe_Frost"] = 0.0
    df["Climatic_Dryness_Index"] = np.nan

    # =========================
    # CALCULS
    # =========================
    mask = wine_cycle & huglin_condition

    df.loc[mask, "Huglin_Index"] = K * (
        (df.loc[mask, "tmax"] + df.loc[mask, "temp_moyenne"]) / 2 - 10
    ).clip(lower=0)

    df.loc[wine_cycle, "Hot_D"] = (df.loc[wine_cycle, "tmax"] > 30).astype(int)

    df.loc[wine_cycle, "Very_Hot_D"] = (df.loc[wine_cycle, "tmax"] >= 35).astype(int)

    df.loc[wine_cycle, "Severe_Heat"] = (
        df.loc[wine_cycle, "temp_moyenne"] - 28
    ).clip(lower=0)

    df.loc[wine_cycle, "Late_Frost"] = (df.loc[wine_cycle, "tmin"] < 0).astype(int)

    df.loc[wine_cycle, "Severe_Frost"] = (
        2 - df.loc[wine_cycle, "tmin"]
    ).clip(lower=0)

    # éviter division par 0
    huglin_safe = df["Huglin_Index"].replace(0, np.nan)

    df.loc[dryness_cycle, "Climatic_Dryness_Index"] = (
        df.loc[dryness_cycle, "precipitation"] / huglin_safe
    )

    # =========================
    # AGRÉGATION ANNUELLE
    # =========================
    df = df.set_index("date")

    annual = (
        df.groupby("Municipality")
        .resample("Y")[
            [
                "Huglin_Index",
                "Hot_D",
                "Very_Hot_D",
                "Severe_Heat",
                "Late_Frost",
                "Severe_Frost",
                "Climatic_Dryness_Index",
            ]
        ]
        .sum()
        .reset_index()
    )

    annual["Year"] = annual["date"].dt.year

    annual = annual[
        [
            "Year",
            "Municipality",
            "Huglin_Index",
            "Hot_D",
            "Very_Hot_D",
            "Severe_Heat",
            "Late_Frost",
            "Severe_Frost",
            "Climatic_Dryness_Index",
        ]
    ]

    return annual


# =========================
# LOAD METEO
# =========================
def load_data_meteo(file_path: str):
    try:
        df = pd.read_csv(file_path, sep=";", encoding="utf-8", comment="#")

        df = _clean_columns(df)
        df = _to_numeric(df)

        return df

    except Exception as e:
        print(f"Erreur chargement données météo: {e}")
        return pd.DataFrame()