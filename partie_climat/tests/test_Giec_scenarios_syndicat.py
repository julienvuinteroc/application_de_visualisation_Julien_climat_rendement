import pytest
import pandas as pd
import matplotlib.pyplot as plt
import src.Giec_scenarios_syndicat as ci


@pytest.fixture
def df_indicators():
    data = {
        "Municipality": [30029, 30029, 30030, 30030,11003,11004],
        "Year": [2019, 2020, 2021, 2022,2023,2024],
        "Huglin_Index": [2000, 2100, 2050, 2150,2200,2300],
        "Hot_D": [5, 3, 4, 6,2,2],
        "Very_Hot_D": [2, 1, 3, 2,2,2],
        "Frost_D": [0, 1, 0, 0,2,2],
        "jours_secs": [10, 12, 9, 11,2,2],
        "stress_climatique": [10, 12, 9, 11,20,20],
        "Climatic_Dryness_Index": [-151, -162, -1755, -303,-200,-230],
    }
    return pd.DataFrame(data)

@pytest.fixture
def df_cluster():
    data = {
        "cluster": [1, 1, 1, 2],
        "Year": [2019, 2020, 2021, 2022],
        "Hot_D": [5, 6, 4, 5],
        "Very_Hot_D": [1, 2, 2, 3],
        "Frost_D": [0, 1, 0, 0],
        "Huglin_Index": [2000, 2100, 2050, 2150],
        "jours_secs": [10, 12, 11, 13],
    }
    return pd.DataFrame(data)


def test_plot_indicators_trend_multiple_columns(df_indicators):
    cols = ["Huglin_Index", "Hot_D", "Very_Hot_D", "Frost_D"]
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30029",
        start_year=2019,
        end_year=2020,
        columns=cols,
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    for col in cols:
        assert col in figs
        assert isinstance(figs[col], plt.Figure)

        
def test_plot_indicators_trendfdsg(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30029",
        start_year=2019,
        end_year=2020,
        columns=["Huglin_Index", "Hot_D"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "Huglin_Index" in figs
    assert "Hot_D" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        

def test_plot_indicators_trend_aze(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30029",
        start_year=2019,
        end_year=2020,
        columns=["Huglin_Index", "Hot_D"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "Huglin_Index" in figs
    assert "Hot_D" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)


def test_plot_indicators_trend_dfs(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30030",
        start_year=2021,
        end_year=2022,
        columns=["Huglin_Index", "Hot_D"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "Huglin_Index" in figs
    assert "Hot_D" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        

def test_plot_indicators_trend_sdf(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="11003",
        start_year=2023,
        end_year=2024,
        columns=["Huglin_Index", "Hot_D"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "Huglin_Index" in figs
    assert "Hot_D" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        
        
def test_plot_indicators_trend_no_data(df_indicators):
    import pytest
    with pytest.raises(ValueError):
        ci.plot_indicators_trend(
            df_indicators,
            municipality="99999",  
            start_year=2019,
            end_year=2020,
            columns=["Huglin_Index"],
            title="Test"
        )
        
def test_plot_indicators_trend96(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30029",
        start_year=2019,
        end_year=2020,
        columns=["jours_secs", "stress_climatique"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "jours_secs" in figs
    assert "stress_climatique" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        
def test_plot_indicators_trend56(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30029",
        start_year=2019,
        end_year=2020,
        columns=["Frost_D", "climatic_dryness_index"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)


def test_plot_indicators_trend2(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30030",
        start_year=2021,
        end_year=2022,
        columns=["Frost_D", "climatic_dryness_index"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)

def test_plot_indicators_trend3(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30030",
        start_year=2021,
        end_year=2022,
        columns=["Hot_D", "Very_Hot_D"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "Hot_D" in figs
    assert "Very_Hot_D" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)


def test_plot_indicators_trend4(df_indicators):
    figs = ci.plot_indicators_trend(
        df_indicators,
        municipality="30030",
        start_year=2021,
        end_year=2022,
        columns=["stress_climatique", "Huglin_Index"],
        title="Test Indicateurs"
    )
    assert isinstance(figs, dict)
    assert "stress_climatique" in figs
    assert "Huglin_Index" in figs
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
