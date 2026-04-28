# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Optional

from modules.utils import zone_sort_key
from config.constants import COLOR_MAP, ZONE_COLOR_MAP, DEPT_COLOR_MAP


def run_prediction(df: pd.DataFrame) -> Optional[pd.DataFrame]:

    st.subheader("Parametres de prediction")

    # Parametres
    col1, col2, col3 = st.columns(3)

    with col1:
        var_lbl = st.radio("Variable a predire", ["Rendement", "Volume"], horizontal=True, key="pred_var")

    with col2:
        group = st.radio(
            "Regrouper par",
            ["Couleur", "Departement", "Zone"],
            horizontal=True,
            key="pred_group"
        )

    with col3:
        horizon = st.slider("Horizon de prediction (annees)", 1, 2, 1, key="pred_h")

    # Mapping des colonnes
    col_map = {
        "Couleur": "code_couleur",
        "Departement": "code_departement",
        "Zone": "Zone"
    }[group]

    # Variable a predire
    var = "rendement" if var_lbl == "Rendement" else "volume"
    mvt = "DECR" if var == "rendement" else "REVE"

    # Filtrage des donnees
    base = df[df["type_mvt"] == mvt].dropna(subset=[var, col_map]).copy()

    if base.empty:
        st.warning("Pas assez de donnees pour la prediction")
        return None

    # Selection des elements
    unique_values = sorted(
        base[col_map].dropna().unique(),
        key=(zone_sort_key if col_map == "Zone" else None)
    )

    sel = st.multiselect(
        "Elements a analyser",
        unique_values,
        default=unique_values[:min(2, len(unique_values))],
        key="pred_sel"
    )

    if not sel:
        return None

    MIN_YEARS = 4

    st.info("""
    **Note sur la prediction :**
    Avec des series temporelles courtes (moins de 10 annees) et une forte variabilite,
    les predictions sont fournies a titre indicatif. La confiance est plus elevee pour
    les horizons courts (1 an) et pour les tendances claires.
    """)

    # Graphique
    fig = go.Figure()
    fig.update_layout(template="plotly_white")

    results = []

    for g in sel:
        # Agrégation par année
        dfg = base[base[col_map] == g].groupby("annee")[var].mean().reset_index().sort_values("annee")

        if len(dfg) < MIN_YEARS:
            st.warning(f"Donnees insuffisantes pour {g} ({len(dfg)} annees, minimum {MIN_YEARS} requises)")
            continue

        # Determination de la couleur
        if group == "Couleur":
            color = COLOR_MAP.get(g, "#808080")
            group_name = g
        elif group == "Zone":
            zone_str = str(int(g)) if isinstance(g, (int, float)) else str(g)
            color = ZONE_COLOR_MAP.get(zone_str, "#808080")
            group_name = f"Zone {g}"
        elif group == "Departement":
            color = DEPT_COLOR_MAP.get(g, "#808080")
            group_name = g
        else:
            color = "#808080"
            group_name = g

        # Tracé historique
        fig.add_trace(go.Scatter(
            x=dfg["annee"],
            y=dfg[var],
            mode="lines+markers",
            name=f"{group_name} - Historique",
            line=dict(width=2, color=color),
            marker=dict(size=8, color=color, symbol="circle")
        ))

        # Prediction simple par moyenne mobile ponderee
        n_years = len(dfg)
        years = dfg["annee"].values
        values = dfg[var].values

        # Calcul de la tendance lineaire simple
        if n_years >= 3:
            X = years.reshape(-1, 1)
            y = values.reshape(-1, 1)

            lr = LinearRegression()
            lr.fit(X, y)

            trend_slope = lr.coef_[0][0]
            trend_intercept = lr.intercept_[0]

            # R² de la tendance
            y_pred_trend = lr.predict(X).ravel()
            trend_r2 = r2_score(y, y_pred_trend)

            # Calcul du dernier taux de variation
            if n_years >= 2:
                last_change = (values[-1] - values[-2]) / values[-2] if values[-2] != 0 else 0
            else:
                last_change = 0

            # Prediction future
            last_year = years[-1]
            future_years = np.arange(last_year + 1, last_year + horizon + 1)

            # Ponderation: 70% tendance lineaire, 30% derniere variation
            if trend_r2 > 0.3:
                future_values = []
                for fy in future_years:
                    pred = trend_slope * fy + trend_intercept
                    if last_change > 0:
                        pred = pred * (1 + last_change * 0.3)
                    elif last_change < 0:
                        pred = pred * (1 + last_change * 0.3)
                    future_values.append(max(0, pred))
            else:
                weights = np.arange(1, min(4, n_years) + 1)
                weights = weights / weights.sum()
                recent_values = values[-min(4, n_years):]
                baseline = np.sum(recent_values * weights[-len(recent_values):])
                future_values = [baseline] * len(future_years)

            # Calcul des metriques de validation
            mae_scores = []
            rmse_scores = []

            for i in range(n_years):
                train_idx = [j for j in range(n_years) if j != i]
                test_idx = [i]

                X_train = years[train_idx].reshape(-1, 1)
                y_train = values[train_idx]
                X_test = years[test_idx].reshape(-1, 1)
                y_test = values[test_idx]

                if len(train_idx) >= 2:
                    lr_cv = LinearRegression()
                    lr_cv.fit(X_train, y_train)
                    y_pred_cv = lr_cv.predict(X_test)

                    mae_scores.append(mean_absolute_error([y_test], y_pred_cv))
                    rmse_scores.append(np.sqrt(mean_squared_error([y_test], y_pred_cv)))

            if mae_scores:
                avg_mae = np.mean(mae_scores)
                avg_rmse = np.mean(rmse_scores)
            else:
                avg_mae = np.nan
                avg_rmse = np.nan

            # Tracé de la tendance lineaire
            trend_line = lr.predict(X).ravel()
            fig.add_trace(go.Scatter(
                x=years,
                y=trend_line,
                mode="lines",
                name=f"{group_name} - Tendance",
                line=dict(dash="dot", width=1.5, color=color),
                opacity=0.6
            ))

            # Tracé des predictions futures
            fig.add_trace(go.Scatter(
                x=future_years,
                y=future_values,
                mode="lines+markers",
                name=f"{group_name} - Prediction",
                line=dict(dash="dash", width=2, color=color),
                marker=dict(size=8, color=color, symbol="diamond")
            ))

            # Zone de confiance
            if len(future_years) > 0:
                std_dev = np.std(values[-min(4, n_years):]) if n_years >= 3 else np.std(values) * 0.5
                upper_bound = [v + std_dev for v in future_values]
                lower_bound = [max(0, v - std_dev) for v in future_values]

                r, g_b, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig.add_trace(go.Scatter(
                    x=list(future_years) + list(future_years)[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor=f'rgba({r},{g_b},{b},0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{group_name} - Zone de confiance",
                    showlegend=False
                ))

            results.append({
                "Groupe": group_name,
                "Tendance (hl/ha/an)": round(trend_slope, 2),
                "R² tendance": round(trend_r2, 2),
                "MAE": round(avg_mae, 2) if not np.isnan(avg_mae) else 0,
                "RMSE": round(avg_rmse, 2) if not np.isnan(avg_rmse) else 0,
                "Prediction " + str(int(future_years[0])): round(future_values[0], 1)
            })

        else:
            results.append({
                "Groupe": group_name,
                "Tendance (hl/ha/an)": 0,
                "R² tendance": 0,
                "MAE": 0,
                "RMSE": 0,
                "Prediction": np.nan
            })

    # Mise en page
    fig.update_layout(
        title=f"Evolution et prediction {var_lbl} par {group.lower()}",
        xaxis_title="Annee",
        yaxis_title=f"{var_lbl} ({'hl/ha' if var == 'rendement' else 'hl'})",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, key="prediction_chart", width="stretch")

    # Resultats
    if results:
        res = pd.DataFrame(results)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Performance du modele")
            display_cols = ["Groupe", "Tendance (hl/ha/an)", "R² tendance", "MAE", "RMSE"]
            if "Prediction 2025" in res.columns:
                display_cols.append("Prediction 2025")
            st.dataframe(res[display_cols], width="stretch")

        with col2:
            st.subheader("Interpretation des resultats")

            positive_trends = res[res["Tendance (hl/ha/an)"] > 0.5]
            negative_trends = res[res["Tendance (hl/ha/an)"] < -0.5]

            if not positive_trends.empty:
                st.success(f"**Tendance positive** : {', '.join(positive_trends['Groupe'].tolist())}")
            
            if not negative_trends.empty:
                st.warning(f"**Tendance negative** : {', '.join(negative_trends['Groupe'].tolist())}")

            good_trends = res[res["R² tendance"] > 0.3]
            if not good_trends.empty:
                st.info(f"**Tendances significatives** (R² > 0.3) : {', '.join(good_trends['Groupe'].tolist())}")

            if "RMSE" in res.columns and not res.empty:
                valid_rmse = res[res["RMSE"] > 0]
                if not valid_rmse.empty:
                    best_idx = valid_rmse["RMSE"].idxmin()
                    best_rmse = valid_rmse.loc[best_idx, "RMSE"]
                    st.metric("Meilleure precision", f"{valid_rmse.loc[best_idx, 'Groupe']} (RMSE={best_rmse:.2f})")

            st.markdown("---")
            st.markdown("""
            **Recommandations :**
            - Les predictions a 1 an sont les plus fiables
            - Plus le R² de la tendance est eleve, plus la prediction est fiable
            - En cas de forte variabilite, privilegier des horizons de prediction courts
            """)

        return res

    return None