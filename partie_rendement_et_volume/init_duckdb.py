# init_duckdb.py
import duckdb
import pandas as pd
import os

# Connexion à DuckDB
con = duckdb.connect('mvttdb.duckdb')

# Vérifier si le fichier CSV existe
csv_file = 'dataset_mvttdb_final.csv'
parquet_file = 'dataset_mvttdb_final.parquet'

if os.path.exists(csv_file):
    print(f"📁 Chargement du fichier CSV: {csv_file}")
    
    # Lire le CSV avec le bon séparateur (point-virgule)
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    
    print(f"📊 Colonnes trouvées: {list(df.columns)}")
    print(f"📈 Nombre de lignes: {len(df)}")
    print("👀 Aperçu des 3 premières lignes:")
    print(df.head(3))
    
    # Créer la table dans DuckDB
    con.execute("DROP TABLE IF EXISTS dataset")
    con.execute("CREATE TABLE dataset AS SELECT * FROM df")
    print("✅ Table 'dataset' créée dans DuckDB")
    
    # Exporter en Parquet
    con.execute(f"COPY dataset TO '{parquet_file}' (FORMAT PARQUET)")
    print(f"✅ Fichier Parquet créé: {parquet_file}")
    
    # Vérification
    result = con.execute("SELECT COUNT(*) FROM dataset").fetchone()
    print(f"✅ Vérification: {result[0]} lignes dans la table")
    
else:
    print(f"❌ Fichier CSV non trouvé: {csv_file}")
    print("Fichiers disponibles dans le répertoire:")
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            print(f"  - {f}")

con.close()
print("🎉 Initialisation terminée!")