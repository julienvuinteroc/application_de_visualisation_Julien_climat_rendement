import json

def load_geojson(level):
    path = f"data/geojson/{level}.geojson"
    with open(path) as f:
        return json.load(f)
