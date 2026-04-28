from utils.db import get_connection
from pathlib import Path
import logging

# ----------------------------
# 🔧 CONFIG LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def load_all(debug: bool = False):
    con = get_connection()

    # chemins
    climat_path = Path("data/climat")
    rendement_path = Path("data/rendement")
    geo_path = Path("data/geojson/zones_pedoclimatiques.geojson")

    # ----------------------------
    # 📥 CLIMAT
    # ----------------------------
    logging.info("📥 Chargement climat...")

    climat_files = list(climat_path.glob("dataset_clean_climat_V0_*.parquet"))

    if not climat_files:
        raise FileNotFoundError("❌ Aucun fichier climat trouvé")

    logging.info(f"📁 {len(climat_files)} fichiers climat détectés")

    file_list = ", ".join([f"'{str(f)}'" for f in climat_files])

    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE climat AS
        SELECT * 
        FROM read_parquet(
            [{file_list}],
            union_by_name=True
        )
        """)

        logging.info("✅ Table climat créée")

        if debug:
            logging.info("📊 STRUCTURE CLIMAT")
            print(con.execute("DESCRIBE climat").df())

            logging.info("🔍 SAMPLE CLIMAT")
            print(con.execute("SELECT * FROM climat LIMIT 5").df())

    except Exception as e:
        logging.error(f"❌ Erreur chargement climat : {e}")
        raise

    # ----------------------------
    # 📥 RENDEMENT
    # ----------------------------
    logging.info("📥 Chargement rendement...")

    rendement_files = list(rendement_path.glob("*.parquet"))

    if not rendement_files:
        raise FileNotFoundError("❌ Aucun fichier rendement trouvé")

    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE rendement AS
        SELECT * 
        FROM read_parquet(
            '{rendement_path}/*.parquet',
            union_by_name=True
        )
        """)

        logging.info("✅ Table rendement créée")

        if debug:
            print(con.execute("DESCRIBE rendement").df())

    except Exception as e:
        logging.error(f"❌ Erreur rendement : {e}")
        raise

    # ----------------------------
    # 🌍 GEOJSON
    # ----------------------------
    logging.info("📥 Chargement geojson...")

    if not geo_path.exists():
        raise FileNotFoundError(f"❌ GeoJSON introuvable : {geo_path}")

    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE geo_raw AS
        SELECT * 
        FROM read_json_auto('{geo_path}')
        """)

        logging.info("✅ Table geo_raw créée")

    except Exception as e:
        logging.error(f"❌ Erreur geojson : {e}")
        raise

    # ----------------------------
    # 📊 STATS RAPIDES
    # ----------------------------
    try:
        logging.info("📊 Stats rapides")

        climat_count = con.execute("SELECT COUNT(*) FROM climat").fetchone()[0]
        rendement_count = con.execute("SELECT COUNT(*) FROM rendement").fetchone()[0]

        logging.info(f"📈 climat rows: {climat_count}")
        logging.info(f"📈 rendement rows: {rendement_count}")

    except Exception:
        logging.warning("⚠️ impossible de calculer les stats")

    logging.info("🎉 LOAD COMPLET OK")