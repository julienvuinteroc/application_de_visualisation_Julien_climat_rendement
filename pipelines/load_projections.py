from utils.db import get_connection
from pathlib import Path
import logging

def load_projections():

    con = get_connection()

    projections_path = Path("data/projections")

    logging.info("📥 Chargement projections...")

    files = list(projections_path.glob("*.parquet"))

    if not files:
        raise FileNotFoundError("❌ Aucun fichier projection trouvé")

    logging.info(f"📁 {len(files)} fichiers projections détectés")

    file_list = ", ".join([f"'{str(f)}'" for f in files])

    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE climat_projection AS
        SELECT * 
        FROM read_parquet(
            [{file_list}],
            union_by_name=True
        )
        """)

        logging.info("✅ projections chargées")

    except Exception as e:
        logging.error(f"❌ erreur projections : {e}")
        raise