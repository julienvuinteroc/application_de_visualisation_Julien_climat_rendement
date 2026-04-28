import pytest
import src.carte_inrae_previsions as ci  
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Données de test simulées
test_data_csv = (
    ";Year;Municipality;Yield;Huglin_Index;Hot_D;Very_Hot_D;Sever_Heat;Frost_D;code_departement"
    "Late_Frost;Sever_Frost;Climatic_Dryness_Index;Soil_Water_Stock;Soil_pH;code_departement;stress_climatique;cluster\n"
    "1;2010;30029;74.25;2085.65;36;0;0;61;0;0;-125.52;88.13;8.05;;;2\n"
    "2;2010;30030;70.12;2050.22;30;1;0;65;0;0;-130.00;87.50;7.90;;;1\n"
    "3;2011;30029;83.50;2292.83;25;3;0;28;0;0;-178.64;88.13;8.05;;;2\n"
    "4;2011;30030;85.00;2250.00;22;2;0;30;0;0;-170.00;87.60;7.95;;;1\n"
    "5;2012;30029;81.53;2161.09;37;7;0;33;0;0;-131.43;88.13;8.05;;;2\n"
    "6;2012;30030;78.00;2150.00;35;6;0;35;0;0;-135.00;87.70;8.00;;;1\n"
    "7;2013;30029;74.50;1948.60;29;0;0;47;0;0;-67.56;88.13;8.05;;;2\n"
    "8;2013;30030;72.00;1900.00;27;1;0;50;0;0;-70.00;87.80;7.85;;;1\n"
)

@pytest.fixture
def test_data_file(tmp_path):
    file = tmp_path / "test_data.csv"
    file.write_text(test_data_csv)
    return str(file)


def test_get_reference_means(test_data_file):
    df = pd.read_csv(test_data_file, sep=";", index_col=0)
    ref_means = ci.get_reference_means(df, 30029, 2010, 2011)
    assert isinstance(ref_means, pd.DataFrame)
    assert "Huglin_Index" in ref_means.columns