from pathlib import Path
import pandas as pd


CLIMAT_DATA_DIR = Path("/home/gelvic/mvttdb/application_mvp/Partie climat/data")
PROJECTIONS_DIR = CLIMAT_DATA_DIR / "Projections"


def robust_read_csv(path: Path) -> pd.DataFrame:
    """
    Lecture robuste CSV :
    - essaie ; puis ,
    - garde low_memory=False
    """
    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            # si une seule colonne, c'est souvent le mauvais séparateur
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

    # dernier essai auto
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def convert_one_csv(csv_path: Path, parquet_path: Path) -> None:
    df = robust_read_csv(csv_path)
    df = normalize_columns(df)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"✅ {csv_path.name} -> {parquet_path}")


def main():
    out_root = CLIMAT_DATA_DIR

    # fichiers CSV racine climat/data
    for csv_file in CLIMAT_DATA_DIR.glob("*.csv"):
        parquet_file = csv_file.with_suffix(".parquet")
        convert_one_csv(csv_file, parquet_file)

    # projections
    for csv_file in PROJECTIONS_DIR.glob("*.csv"):
        parquet_file = csv_file.with_suffix(".parquet")
        convert_one_csv(csv_file, parquet_file)

    print("\n🎉 Conversion terminée.")


if __name__ == "__main__":
    main()
