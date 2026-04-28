import pytest


@pytest.fixture
def cniv_csv(tmp_path):
    csv_content = (
        "Unnamed: 0;date;Municipality;Temperature maximale;Temperature moyenne;"
        "Temperature minimale;"
        "ETP;Pluie quotidienne;Cumul quotidien\n"
        "0;01/04/2012;30029;25;18;5;;2;\n"
        "1;02/04/2012;30029;30;20;2;;0;\n"
        "2;03/04/2012;30030;35;25;1;;0;\n"
        "3;04/04/2012;30030;28;22;0;;1;\n"
        "4;05/04/2013;30029;32;21;3;;0;\n"
    )
    file = tmp_path / "cniv_test.csv"
    file.write_text(csv_content)
    return str(file)

@pytest.fixture
def meteo_csv(tmp_path):
    csv_content = (
        "Year;Municipality;Yield;Huglin_Index;rendement_plafonne;stress_climatique"
        ";deficit hydrique;"
        "temp_moyenne;tmax_mean;tmin_mean;amplitude_thermique;precipitation_total;pluie_avril_septembre;"
        "Hot_D;Very_Hot_D;latitude;longitude;Sever_Heat;Frost_D;Climatic_Dryness_Index;Soil_Water_Stock;Soil_pH;productivite\n"
        "2012;30029;74;2085;80;0;0;18;25;5;7;10;3;2;1;44;4;0;0;0;88;8;70\n"
        "2012;30030;70;2050;78;0;0;20;28;2;8;12;4;1;2;45;5;1;0;1;87;7;65\n"
    )
    file = tmp_path / "meteo_test.csv"
    file.write_text(csv_content)
    return str(file)


@pytest.fixture
def autre_csv(tmp_path):
    csv_content = (
        "Col1;Col2;Col3\n"
        "A;1;0.5\n"
        "B;2;1.5\n"
        "C;3;2.5\n"
    )
    file = tmp_path / "autre_test.csv"
    file.write_text(csv_content)
    return str(file)


def test_autre_fichier(autre_csv):
    import pandas as pd
    df = pd.read_csv(autre_csv, sep=";")
    assert not df.empty
    assert list(df.columns) == ["Col1", "Col2", "Col3"]
    assert df["Col2"].dtype == int
