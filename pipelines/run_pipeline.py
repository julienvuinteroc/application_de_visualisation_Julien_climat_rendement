from utils.db import get_connection
from load_data import load_all
from load_projections import load_projections
from clean_data import clean
from aggregate_climat import aggregate
from merge_datasets import merge
from build_features import build

def run():

    con = get_connection()

    print("🚀 Lancement pipeline...")

    load_all()
    load_projections()
    clean(con)
    aggregate(con)
    merge(con)
    build(con)

    print("🎉 PIPELINE COMPLET OK")

if __name__ == "__main__":
    run()