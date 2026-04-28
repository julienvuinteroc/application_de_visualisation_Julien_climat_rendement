import pytest
import pandas as pd
import src.carte_inrae as ci

insee_csv = """insee_code,region_name,some_column
30001,occitanie,value1
34002,occitanie,value2
11003,occitanie,value3
66004,occitanie,value4
75001,ile-de-france,value5
"""

inrae_csv = """Municipality;cluster;other_column
30001;1;abc
34002;2;def
11003;7;ghi
66004;4;jkl
"""

def test_load_clean_data_insee_oc_invalid_file():
    df = ci.load_clean_data_insee_oc("fichier_inexistant.csv")
    assert df.empty
    

def test_load_data_inrae_invalid_file():
    df = ci.load_data_inrae("fichier_inexistant.csv")
    assert df.empty


def test_get_commune_name_by_insee_empty_df():
    df = pd.DataFrame(columns=["insee_code", "city_code"])
    assert ci.get_commune_name_by_insee(df, "12345") is None
    
    
def test_display_data_map_colors():
    data = {
        "Municipality": ["30001", "34002", "11003", "66004"],
        "cluster": [1, 2, 3, 4],
    }
    df = pd.DataFrame(data)
    fig = ci.display_data_map(df)
    assert fig is not None
    ax = fig.axes[0]
    # Récupère toutes les couleurs des polygones
    facecolors = [tuple(fc) for coll in ax.collections for fc in coll.get_facecolor()]
    # Vérifie qu'on a bien des couleurs RGBA
    assert all(len(c) == 4 for c in facecolors)
    
    
def test_load_clean_data_insee_oc(tmp_path):
    # Création d’un fichier temporaire CSV pour charger les données de l'INSEE
    csv_file = tmp_path / "insee_test.csv"
    csv_file.write_text(insee_csv)

    df = ci.load_clean_data_insee_oc(str(csv_file))
    assert not df.empty
    assert all(df["region_name"] == "occitanie")
    assert set(df["code_dep"]).issubset({"30", "34", "11", "66"})


def test_get_commune_name_by_insee():

    data = {
        "insee_code": ["30001", "34002", "11003", "66004"],
        "city_code": ["CommuneA", "CommuneB", "CommuneC", "CommuneD"]
    }
    df = pd.DataFrame(data)
    assert ci.get_commune_name_by_insee(df, "30001") == "CommuneA"
    assert ci.get_commune_name_by_insee(df, " 34002 ") == "CommuneB"
    assert ci.get_commune_name_by_insee(df, 11003) == "CommuneC"
    assert ci.get_commune_name_by_insee(df, "99999") is None


def test_load_data_inrae(tmp_path):
    # Création d’un fichier temporaire CSV pour charger les données de l'INRAE
    csv_file = tmp_path / "inrae_test.csv"
    csv_file.write_text(inrae_csv)

    df = ci.load_data_inrae(str(csv_file))
    assert not df.empty
    assert "Municipality" in df.columns
    assert "cluster" in df.columns

def test_display_data_map_runs_without_error():
    data = {
        "Municipality": ["30001", "34002", "11003", "66004"],
        "cluster": [1, 2, 7, 4],  # test remplacement des clusters 7->5, 2->3 
    }
    df = pd.DataFrame(data)

    try:
        ci.display_data_map(df)
    except Exception as e:
        pytest.fail(f"display_data_map raised an exception: {e}")


def test_load_clean_data_insee_oc_other_region(tmp_path):
    csv_file = tmp_path / "insee_other.csv"
    csv_file.write_text("insee_code,region_name,some_column\n75001,ile-de-france,value")
    df = ci.load_clean_data_insee_oc(str(csv_file))
    assert df.empty


def test_display_data_map_missing_columns():
    df = pd.DataFrame({"Municipality": ["30001"]})
    fig = ci.display_data_map(df)
    assert fig is None
    

def test_display_data_map_invalid_clusters():
    data = {"Municipality": ["30001", "34002"], "cluster": [9, 99]}
    df = pd.DataFrame(data)
    fig = ci.display_data_map(df)
    assert fig is None
    