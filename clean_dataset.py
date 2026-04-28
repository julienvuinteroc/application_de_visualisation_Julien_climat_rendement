from pathlib import Path
import pandas as pd
import numpy as np


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = Path("data/dataset_clean_climat_V0_2_final_bis.csv")
OUTPUT_FILE = Path("data/dataset_clean_climat_cleaned.csv")


# =====================================================
# CLEAN FUNCTIONS
# =====================================================

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Nettoie une colonne numérique corrompue :
    - remplace les virgules par des points
    - extrait le premier nombre valide
    - convertit en float
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Colonnes à garder en texte
    categorical_cols = ["Municipality", "cluster", "code_departement"]

    for col in df.columns:

        if col in categorical_cols:
            df[col] = df[col].astype(str)
            continue

        # conversion numérique robuste
        df[col] = clean_numeric_column(df[col])

    # nettoyage Year
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    # suppression lignes vides
    df = df.dropna(how="all")

    return df


# =====================================================
# VALIDATION
# =====================================================

def validate_dataframe(df: pd.DataFrame):

    print("\nValidation dataset")

    # types
    print("\nTypes :")
    print(df.dtypes)

    # NaN
    print("\nNaN par colonne :")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # valeurs aberrantes
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print("\nRésumé numérique :")
    print(df[numeric_cols].describe().T[["min", "max"]])


# =====================================================
# MAIN
# =====================================================

def main():

    print("Chargement dataset...")
    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        encoding="utf-8",
        comment="#",
        index_col=0
    )

    print("Nettoyage...")
    df_clean = clean_dataframe(df)

    validate_dataframe(df_clean)

    print("\nSauvegarde...")
    df_clean.to_csv(OUTPUT_FILE, sep=";", index=False)

    print(f"Fichier nettoyé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
