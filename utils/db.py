from pathlib import Path
import duckdb
import streamlit as st

# =====================================================
# PATH DB
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "mvttdb.duckdb"


# =====================================================
# CONNEXION PRINCIPALE (STREAMLIT)
# =====================================================

@st.cache_resource
def get_conn():
    """
    Connexion partagée (cache Streamlit)
    ⚠️ Ne jamais fermer cette connexion !
    """
    return duckdb.connect(str(DB_PATH), read_only=False)


# =====================================================
# CONNEXION TEMPORAIRE (scripts)
# =====================================================

def get_connection(read_only: bool = False):
    """
    Connexion classique (hors Streamlit)
    À utiliser dans scripts type create_database.py
    """
    return duckdb.connect(str(DB_PATH), read_only=read_only)


# =====================================================
# RUN QUERY
# =====================================================

def run_query(query: str):
    """
    Exécute une requête simple via connexion partagée
    """
    conn = get_conn()
    return conn.execute(query).df()