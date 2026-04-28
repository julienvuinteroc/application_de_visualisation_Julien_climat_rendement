# -*- coding: utf-8 -*-

from pathlib import Path

# =====================================================
# RACINE PROJET
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# FICHIERS PRINCIPAUX
# =====================================================

DB_PATH = BASE_DIR / "db" / "mvttdb.duckdb"
PARQUET_PATH = BASE_DIR / "partie_rendement_et_volume" / "dataset_mvttdb_final.parquet"

DB_FILE = str(DB_PATH)
PARQUET_FILE = str(PARQUET_PATH)

# =====================================================
# DOSSIER GEOJSON
# =====================================================

GEOJSON_DIR = BASE_DIR / "partie_rendement_et_volume" / "assets" / "geojson"

GEOJSON_FILES = {
    "departements": GEOJSON_DIR / "departements_languedoc.geojson",
    "zones": GEOJSON_DIR / "zones_pedoclimatiques.geojson",
    "communes": GEOJSON_DIR / "communes_languedoc.geojson",
}

# =====================================================
# CONSTANTES MÉTIER
# =====================================================
DEPT_COLOR_MAP = {
    "11": "#3498db",  # Aude - bleu
    "30": "#2ecc71",  # Gard - vert
    "34": "#e74c3c",  # Herault - rouge
    "66": "#f39c12",  # Pyrenees-Orientales - orange
}
DEPT_NAMES = {
    "11": "Aude",
    "30": "Gard",
    "34": "Herault",
    "66": "Pyrenees-Orientales"
}
COLOR_MAP = {
    "BL": "#F4D03F",
    "RG": "#A93226",
    "RS": "#F1948A",
}

ZONE_COLOR_MAP = {
    "0": "#BDBDBD",
    "1": "#000000",
    "2": "#FF0000",
    "3": "#1A8F2A",
    "4": "#0033CC",
    "5": "#AFC6D9",
    "6": "#7A1FA2",
    "7": "#FFD800",
}

DEPARTEMENTS = ["11", "30", "34", "66"]
#fsd
ZONE_LABELS = {
    "0": "Zone 0: non classée / hors zonage",
    "1": "Zone 1: zone humide de l'arrière-pays",
    "2": "Zone 2: zone de montagne avec des sols acides et peu profonds",
    "3": "Zone 3: zone de piémont avec une réserve utile limitante",
    "4": "Zone 4: zone froide et sèche autour du Pic Saint-Loup",
    "5": "Zone 5: zone de sols de qualité moyenne dans l’arrière-pays",
    "6": "*Zone 6: zone de sols profonds sur côtes tempérées",
    "7": "*Zone 7: zone avec le plus grand nombre de jours très chauds mais sols profonds",
}
# =====================================================
# CHECK
# =====================================================
def check_paths():
    missing = []

    if not DB_PATH.exists():
        missing.append(f"DB manquante: {DB_PATH}")

    if not PARQUET_PATH.exists():
        missing.append(f"Parquet manquant: {PARQUET_PATH}")

    for name, path in GEOJSON_FILES.items():
        if not Path(path).exists():
            missing.append(f"GEOJSON manquant ({name}): {path}")

    return missing