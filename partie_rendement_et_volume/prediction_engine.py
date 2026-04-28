# prediction_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    ARIMA_OK = True
except Exception:
    ARIMA_OK = False


@dataclass
class ForecastResult:
    model: str
    history: pd.DataFrame         # columns: annee, y
    forecast: pd.DataFrame        # columns: annee, yhat


def build_series(df: pd.DataFrame, value_col: str, group_col: str, group_value: str, agg: str = "mean") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["annee", "y"])

    if not {"annee", value_col, group_col}.issubset(df.columns):
        return pd.DataFrame(columns=["annee", "y"])

    sub = df[df[group_col].astype(str) == str(group_value)].copy()
    sub["annee"] = pd.to_numeric(sub["annee"], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=["annee", value_col])

    if sub.empty:
        return pd.DataFrame(columns=["annee", "y"])

    if agg == "sum":
        s = sub.groupby("annee")[value_col].sum().reset_index()
    else:
        s = sub.groupby("annee")[value_col].mean().reset_index()

    s = s.sort_values("annee").rename(columns={value_col: "y"})
    s["annee"] = s["annee"].astype(int)
    return s


def forecast_linear(series: pd.DataFrame, horizon: int = 3) -> Optional[ForecastResult]:
    if series is None or series.empty or len(series) < 5:
        return None

    s = series.copy().sort_values("annee")
    x = s["annee"].to_numpy(dtype=float)
    y = s["y"].to_numpy(dtype=float)

    if not np.isfinite(y).all():
        return None

    # y = a + b*annee
    X = np.vstack([np.ones(len(x)), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta[0], beta[1]

    last_year = int(s["annee"].max())
    fut_years = np.arange(last_year + 1, last_year + horizon + 1)
    yhat = a + b * fut_years

    fdf = pd.DataFrame({"annee": fut_years.astype(int), "yhat": yhat.astype(float)})
    return ForecastResult(model="Linear", history=s, forecast=fdf)


def forecast_arima(series: pd.DataFrame, horizon: int = 3) -> Optional[ForecastResult]:
    if not ARIMA_OK:
        return None
    if series is None or series.empty or len(series) < 7:
        return None

    s = series.copy().sort_values("annee")
    y = s["y"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        return None

    try:
        # petit modèle robuste
        model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
        fc = model.forecast(steps=horizon)
        last_year = int(s["annee"].max())
        fut_years = np.arange(last_year + 1, last_year + horizon + 1)
        fdf = pd.DataFrame({"annee": fut_years.astype(int), "yhat": np.asarray(fc, dtype=float)})
        return ForecastResult(model="ARIMA", history=s, forecast=fdf)
    except Exception:
        return None