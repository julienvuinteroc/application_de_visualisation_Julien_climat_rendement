# config/__init__.py
"""
Configuration de l'application
"""

# Ne pas importer constants ici pour éviter les imports circulaires
# Les constantes seront importées directement depuis config.constants

__all__ = ['COLOR_MAP', 'ZONE_COLOR_MAP', 'ZONE_LABELS', 'DEPARTEMENTS', 
           'GEOJSON_FILES', 'AI_MEMORY_FILE', 'BASE_DIR', 'DB_FILE', 'PARQUET_FILE']