import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import duckdb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(layout="wide", page_title="Analyse & Prédiction Viticole")

COLOR_MAP = {"BL": "#FFD700", "RG": "#8B0000", "RS": "#FF69B4"}

# 🎨 Palette officielle zones (selon ta carte) + ✅ Zone 0 en gris
ZONE_COLOR_MAP = {
    "0": "#BDBDBD",   # ✅ gris (zone 0)
    "1": "#000000",   # noir
    "2": "#FF0000",   # rouge
    "3": "#1A8F2A",   # vert
    "4": "#0033CC",   # bleu foncé
    "5": "#AFC6D9",   # gris/bleu clair
    "6": "#7A1FA2",   # violet
    "7": "#FFD800",   # jaune
}

ZONE_LABELS = {
    "0": "Zone 0 : non classée / hors zonage",
    "1": "Zone 1 : zone humide de l'arrière-pays",
    "2": "Zone 2 : montagne, sols acides et peu profonds",
    "3": "Zone 3 : Piémont, réserve utile limitée",
    "4": "Zone 4 : froide et sèche autour du Pic Saint-Loup",
    "5": "Zone 5 : sols de qualité moyenne dans l'arrière-pays",
    "6": "Zone 6 : sols profonds, côtes tempérées",
    "7": "Zone 7 : très chaud (beaucoup de jours très chauds), sols profonds",
}


# =====================================================
# DUCKDB + PARQUET
# =====================================================
DB_FILE = "mvttdb.duckdb"
PARQUET_FILE = "dataset_mvttdb_final.parquet"


@st.cache_resource
def get_connection():
    con = duckdb.connect(DB_FILE)

    tables = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name='dataset'
    """).fetchall()

    if not tables:
        con.execute(f"""
            CREATE TABLE dataset AS
            SELECT
                type_mvt,
                code_couleur,
                CAST(annee AS INTEGER) AS annee,

                -- Nettoyage robuste volume
                CAST(
                    REPLACE(NULLIF(volume, ''), ',', '.')
                    AS DOUBLE
                ) AS volume,

                -- Nettoyage robuste surface
                CAST(
                    REPLACE(NULLIF(surface, ''), ',', '.')
                    AS DOUBLE
                ) AS surface,

                code_cepage,
                cvi,

                CAST(rendement AS DOUBLE) AS rendement,

                CAST(commune AS VARCHAR) AS commune,
                CAST(Zone AS VARCHAR) AS Zone,
                CAST(code_departement AS VARCHAR) AS code_departement,

                departement,
                rendement_plafonne_RS,
                rendement_plafonne_RG_BL

            FROM read_parquet('{PARQUET_FILE}')
        """)
        print("Table dataset créée avec types corrigés ✅")

    return con

def _to_float_series(s: pd.Series) -> pd.Series:
    """
    Robustesse max:
    - gère float déjà OK
    - gère strings avec virgules "12,34"
    - gère "nan", "", None
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s2 = (
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .replace({"nan": None, "None": None, "": None})
    )
    return pd.to_numeric(s2, errors="coerce")


@st.cache_data
def load_data():
    con = get_connection()
    df = con.execute("SELECT * FROM dataset").df()

    # Colonnes numériques
    for col in ["rendement", "surface", "volume"]:
        if col in df.columns:
            df[col] = _to_float_series(df[col])

    # Types de colonnes clés
    if "annee" in df.columns:
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce").fillna(0).astype(int)
    for col in ["code_departement", "Zone", "commune", "cvi"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # plafonds réglementaires
    df["plafond"] = df["code_couleur"].map({"BL": 90, "RG": 90, "RS": 100})
    df["statut_plafond"] = np.where(
        df["rendement"] <= df["plafond"], "🟢 Conforme", "🔴 Hors plafond"
    )

    return df


df_all = load_data()


# =====================================================
# GEOJSON
# =====================================================
@st.cache_data
def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# HELPERS – continuité des tendances (séries complètes)
# =====================================================
def complete_years(df_in: pd.DataFrame, group_col: str, value_col: str, aggfunc: str,
                   years_min: int, years_max: int):
    """Force une continuité temporelle (toutes les années) par groupe."""
    if df_in.empty:
        return df_in

    yrs = np.arange(years_min, years_max + 1)
    groups = sorted(df_in[group_col].dropna().unique())

    out_rows = []
    for g in groups:
        sub = df_in[df_in[group_col] == g].groupby("annee")[value_col].agg(aggfunc)
        sub = sub.reindex(yrs)  # années manquantes -> NaN
        tmp = pd.DataFrame({"annee": yrs, group_col: g, value_col: sub.values})
        out_rows.append(tmp)

    return pd.concat(out_rows, ignore_index=True)


# =====================================================
# BOOTSTRAP CI
# =====================================================
def bootstrap_ci(values: np.ndarray, n_boot: int = 1500, seed: int = 42):
    """
    IC 95% bootstrap (percentile) pour la moyenne.
    Robuste pour des distributions asymétriques (ex: volumes).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 5:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    return np.percentile(means, 2.5), np.percentile(means, 97.5)


# =====================================================
# ORDRE ZONES (0 → 7) – utile pour légendes / affichages
# =====================================================
def zone_sort_key(z: str):
    try:
        return int(z)
    except Exception:
        return 999


def zone_order_from_series(s: pd.Series):
    zs = [str(x) for x in s.dropna().unique()]
    return sorted(zs, key=zone_sort_key)


# =====================================================
# SIDEBAR – FILTRES
# =====================================================
st.sidebar.title("Filtres globaux")
st.sidebar.divider()

# ✅ bouton carte référence
show_ref_map = st.sidebar.toggle("🗺️ Voir la carte des zones pédoclimatiques", value=False)

departements = st.sidebar.multiselect(
    "Départements", ["11", "30", "34", "66"],
    default=["11", "30", "34", "66"],
    key="sb_dep"
)

# ✅ inclure zone 0 si elle existe dans le dataset
all_zones = zone_order_from_series(df_all["Zone"])
zones = st.sidebar.multiselect(
    "Zones",
    all_zones,
    default=all_zones,
    key="sb_zone"
)

couleurs = st.sidebar.multiselect(
    "Couleurs", ["BL", "RG", "RS"],
    default=["BL", "RG", "RS"],
    key="sb_coul"
)

annees = st.sidebar.slider(
    "Période", 2011, 2024, (2011, 2024),
    key="sb_years"
)

# filtre global
df = df_all[
    (df_all["code_departement"].isin(departements)) &
    (df_all["Zone"].isin(zones)) &
    (df_all["code_couleur"].isin(couleurs)) &
    (df_all["annee"].between(*annees))
].copy()

YEAR_MIN = int(df["annee"].min()) if not df.empty else int(annees[0])
YEAR_MAX = int(df["annee"].max()) if not df.empty else int(annees[1])


# =====================================================
# CARTE DE RÉFÉRENCE (plein écran)
# =====================================================
if show_ref_map:
    st.header("🗺️ Carte officielle des zones pédoclimatiques (référence)")

    try:
        geo = load_geojson("zones_pedoclimatiques.geojson")

        ref = df_all[["commune", "Zone"]].drop_duplicates().copy()
        ref["Zone"] = ref["Zone"].astype(str)
        ref["Nom"] = ref["Zone"].map(ZONE_LABELS).fillna(ref["Zone"].apply(lambda z: f"Zone {z}"))

        # ✅ Légende en ordre croissant (0 → 7)
        zone_order = zone_order_from_series(ref["Zone"])

        fig_ref = px.choropleth(
            ref,
            geojson=geo,
            locations="commune",
            featureidkey="properties.code_commune",
            color="Zone",
            color_discrete_map=ZONE_COLOR_MAP,
            category_orders={"Zone": zone_order},
            hover_name="Nom",
            title="Zones pédoclimatiques (référence)"
        )
        fig_ref.update_geos(fitbounds="locations", visible=False)
        fig_ref.update_layout(template="plotly_white", legend_title_text="Zone")

        st.plotly_chart(fig_ref, width="stretch")

    except Exception as e:
        st.error(f"Impossible d'afficher la carte de référence : {e}")

    st.stop()


# =====================================================
# TABS
# =====================================================
tab_rdt, tab_vol, tab_pred, tab_map, tab_quant = st.tabs(
    ["📈 Rendement", "🍷 Volume", "🔮 Prédiction", "🗺️ Cartographie", "📊 Analyse quantitative"]
)


# =====================================================
# 📈 RENDEMENT
# =====================================================
with tab_rdt:
    st.header("📈 Rendement – comparaisons")

    decr = df[(df["type_mvt"] == "DECR") & df["rendement"].between(10, 100)].copy()

    mode = st.radio(
        "Comparer par",
        ["Couleur", "Département", "Zone"],
        horizontal=True,
        key="rdt_mode"
    )

    col = {"Couleur": "code_couleur", "Département": "code_departement", "Zone": "Zone"}[mode]

    sel = st.multiselect(
        "Sélection",
        sorted(decr[col].dropna().unique(), key=(zone_sort_key if col == "Zone" else None)),
        default=sorted(decr[col].dropna().unique(), key=(zone_sort_key if col == "Zone" else None))[:3],
        key="rdt_sel"
    )

    data = decr[decr[col].isin(sel)].copy()
    evol = data.groupby(["annee", col])["rendement"].mean().reset_index()
    evol_full = complete_years(evol, col, "rendement", "mean", YEAR_MIN, YEAR_MAX)

    fig = px.line(
        evol_full,
        x="annee", y="rendement",
        color=col,
        category_orders={"Zone": zone_order_from_series(evol_full["Zone"])} if col == "Zone" else None,
        color_discrete_map=(
            COLOR_MAP if col == "code_couleur"
            else (ZONE_COLOR_MAP if col == "Zone" else None)
        ),
        markers=True,
        title="Évolution des rendements (série continue)"
    )

    fig.add_hline(y=90, line_dash="dash", annotation_text="Plafond BL / RG")
    fig.add_hline(y=100, line_dash="dash", annotation_text="Plafond RS")
    fig.update_layout(yaxis=dict(range=[10, 100], dtick=10), template="plotly_white")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistiques")
    st.dataframe(data.groupby(col)["rendement"].describe().round(2), width="stretch")


# =====================================================
# 🍷 VOLUME
# =====================================================
with tab_vol:
    st.header("🍷 Volume – comparaisons")

    reve = df[df["type_mvt"] == "REVE"].dropna(subset=["volume"]).copy()

    mode = st.radio(
        "Comparer par",
        ["Couleur", "Département", "Zone", "Cépage"],
        horizontal=True,
        key="vol_mode"
    )

    col = {
        "Couleur": "code_couleur",
        "Département": "code_departement",
        "Zone": "Zone",
        "Cépage": "code_cepage"
    }[mode]

    sel = st.multiselect(
        "Sélection",
        sorted(reve[col].dropna().unique(), key=(zone_sort_key if col == "Zone" else None)),
        default=sorted(reve[col].dropna().unique(), key=(zone_sort_key if col == "Zone" else None))[:3],
        key="vol_sel"
    )

    data = reve[reve[col].isin(sel)].copy()
    evol = data.groupby(["annee", col])["volume"].sum().reset_index()
    evol_full = complete_years(evol, col, "volume", "sum", YEAR_MIN, YEAR_MAX)

    fig = px.line(
        evol_full,
        x="annee", y="volume",
        color=col,
        category_orders={"Zone": zone_order_from_series(evol_full["Zone"])} if col == "Zone" else None,
        color_discrete_map=(
            COLOR_MAP if col == "code_couleur"
            else (ZONE_COLOR_MAP if col == "Zone" else None)
        ),
        markers=True,
        title="Évolution des volumes (série continue)"
    )
    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistiques")
    st.dataframe(data.groupby(col)["volume"].describe().round(2), width="stretch")

    # -------------------------------------------------
    # 🍇 Cépages par zone (UNIQUEMENT par zone)
    # -------------------------------------------------
    st.divider()
    st.subheader("🍇 Cépages par zone (comparaison)")

    df_cepage = df[df["type_mvt"] == "REVE"].dropna(subset=["volume", "code_cepage", "Zone"]).copy()

    top_n = st.slider("Nombre de cépages affichés (Top N)", 5, 30, 12, key="cepage_topn")

    zones_sel = st.multiselect(
        "Zones à comparer",
        zone_order_from_series(df_cepage["Zone"]),
        default=zone_order_from_series(df_cepage["Zone"])[:3],
        key="cepage_zone_sel"
    )

    analyse_annuelle = st.checkbox(
        "Analyser par année (volume de cépage / année)",
        value=True,
        key="cepage_annual_mode"
    )

    df_cepage = df_cepage[df_cepage["Zone"].isin(zones_sel)].copy()

    if df_cepage.empty:
        st.info("Pas de données REVE pour ces zones.")
    else:
        if analyse_annuelle:
            ce = (
                df_cepage
                .groupby(["annee", "code_cepage", "Zone"])["volume"]
                .sum()
                .reset_index()
            )

            top_cepages = (
                ce.groupby("code_cepage")["volume"].sum()
                .sort_values(ascending=False).head(top_n).index
            )
            ce = ce[ce["code_cepage"].isin(top_cepages)].copy()

            ce["Zone"] = pd.Categorical(ce["Zone"], categories=zone_order_from_series(ce["Zone"]), ordered=True)

            figc = px.line(
                ce,
                x="annee",
                y="volume",
                color="code_cepage",
                facet_row="Zone",
                markers=True,
                title="Évolution annuelle du volume par cépage (facettes par zone)"
            )
            figc.update_layout(template="plotly_white")
            st.plotly_chart(figc, width="stretch")
        else:
            zc = df_cepage.groupby(["Zone", "code_cepage"])["volume"].sum().reset_index()

            top_cepages = (
                zc.groupby("code_cepage")["volume"].sum()
                .sort_values(ascending=False).head(top_n).index
            )
            zc = zc[zc["code_cepage"].isin(top_cepages)].copy()

            figc = px.bar(
                zc,
                x="code_cepage",
                y="volume",
                color="Zone",
                category_orders={"Zone": zone_order_from_series(zc["Zone"])},
                barmode="group",
                color_discrete_map=ZONE_COLOR_MAP,
                title="Comparaison des volumes par cépage et par zone (Top N)"
            )
            figc.update_layout(template="plotly_white")
            st.plotly_chart(figc, width="stretch")


# =====================================================
# 🔮 PRÉDICTION – COURBES + COMPARAISON
# =====================================================
with tab_pred:
    st.header("🔮 Prédiction – comparaison des modèles")

    var_lbl = st.radio("Variable", ["Rendement", "Volume"], horizontal=True, key="pred_var")
    group = st.radio("Comparer par", ["Couleur", "Département", "Zone", "Cépage"], horizontal=True, key="pred_group")
    _ = st.slider("Horizon (années)", 1, 5, 3, key="pred_h")  # UI only (pas utilisé ici)

    col = {
        "Couleur": "code_couleur",
        "Département": "code_departement",
        "Zone": "Zone",
        "Cépage": "code_cepage"
    }[group]

    var = "rendement" if var_lbl == "Rendement" else "volume"
    mvt = "DECR" if var == "rendement" else "REVE"

    base = df[df["type_mvt"] == mvt].dropna(subset=[var, col]).copy()

    sel = st.multiselect(
        "Sélection",
        sorted(base[col].dropna().unique(), key=(zone_sort_key if col == "Zone" else None)),
        default=sorted(base[col].dropna().unique(), key=(zone_sort_key if col == "Zone" else None))[:2],
        key="pred_sel"
    )

    fig = px.line(template="plotly_white")
    results = []
    test_size = 3

    for g in sel:
        dfg = base[base[col] == g].groupby("annee")[var].mean().reset_index().sort_values("annee")
        if len(dfg) < 6:
            continue

        yrs = np.arange(dfg["annee"].min(), dfg["annee"].max() + 1)
        dfg_full = dfg.set_index("annee").reindex(yrs).reset_index().rename(columns={"index": "annee"})
        dfg_full[col] = g

        fig.add_scatter(x=dfg_full["annee"], y=dfg_full[var], mode="lines+markers", name=f"{g} – Historique")

        train = dfg.iloc[:-test_size]
        test = dfg.iloc[-test_size:]
        y_true = test[var].values

        # Linéaire
        lin = LinearRegression().fit(train[["annee"]], train[var])
        y_lin = lin.predict(test[["annee"]])
        fig.add_scatter(x=test["annee"], y=y_lin, line=dict(dash="dot"), name=f"{g} – Linéaire")
        results.append((g, "Linéaire", mean_absolute_error(y_true, y_lin), np.sqrt(mean_squared_error(y_true, y_lin))))

        # ARIMA (tolérant)
        try:
            ar = ARIMA(train[var], order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
            y_ar = ar.forecast(test_size)
            fig.add_scatter(x=test["annee"], y=y_ar, line=dict(dash="dash"), name=f"{g} – ARIMA")
            results.append((g, "ARIMA", mean_absolute_error(y_true, y_ar), np.sqrt(mean_squared_error(y_true, y_ar))))
        except Exception:
            pass

        # Prophet
        try:
            dfp = train.rename(columns={"annee": "ds", var: "y"}).copy()
            dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str), format="%Y")
            pr = Prophet()
            pr.fit(dfp)
            future = pd.DataFrame({"ds": pd.to_datetime(test["annee"].astype(str), format="%Y")})
            y_pr = pr.predict(future)["yhat"].values
            fig.add_scatter(x=test["annee"], y=y_pr, line=dict(dash="longdash"), name=f"{g} – Prophet")
            results.append((g, "Prophet", mean_absolute_error(y_true, y_pr), np.sqrt(mean_squared_error(y_true, y_pr))))
        except Exception:
            pass

    st.plotly_chart(fig, width="stretch")

    if results:
        res = pd.DataFrame(results, columns=[group, "Modèle", "MAE", "RMSE"]).round(2)
        st.subheader("📊 Tableau comparatif des modèles")
        st.dataframe(res.sort_values("RMSE"), width="stretch")

        st.subheader("🏆 Classement global (RMSE moyen)")
        st.dataframe(res.groupby("Modèle")["RMSE"].mean().sort_values().reset_index(), width="stretch")


# =====================================================
# 🗺️ CARTOGRAPHIE – + filtre couleur + carte anomalies
# =====================================================
with tab_map:
    st.header("🗺️ Cartographie viticole")

    indic = st.radio("Indicateur", ["Rendement", "Volume", "Surface"], horizontal=True, key="map_ind")
    level = st.radio("Niveau", ["Département", "Zone"], horizontal=True, key="map_level")

    map_colors = st.multiselect(
        "Couleurs (cartographie)",
        ["BL", "RG", "RS"],
        default=["BL", "RG", "RS"],
        key="map_colors"
    )

    show_mean = st.checkbox("Afficher la moyenne globale sur la carte", value=True, key="map_show_mean")

    years_map = st.multiselect(
        "Années",
        sorted(df["annee"].unique()),
        default=sorted(df["annee"].unique())[-3:] if not df.empty else [],
        key="map_years"
    )

    if indic == "Rendement":
        var, mvt, agg, scale, unit = "rendement", "DECR", "mean", "RdYlGn_r", "hl/ha"
    elif indic == "Volume":
        var, mvt, agg, scale, unit = "volume", "REVE", "sum", "Blues", "hl"
    else:
        var, mvt, agg, scale, unit = "surface", "DECR", "sum", "Greens", "ha"

    base = df[
        (df["type_mvt"] == mvt) &
        (df["annee"].isin(years_map)) &
        (df["code_couleur"].isin(map_colors))
    ].dropna(subset=[var]).copy()

    if base.empty:
        st.info("Aucune donnée pour cette sélection.")
    else:
        if level == "Département":
            geo = load_geojson("departements_languedoc.geojson")
            agg_df = base.groupby(["code_departement", "annee"])[var].agg(agg).reset_index()
            join, fid = "code_departement", "properties.dep"
        else:
            geo = load_geojson("zones_pedoclimatiques.geojson")
            zone_agg = base.groupby(["Zone", "annee"])[var].agg(agg).reset_index()
            agg_df = df_all[["commune", "Zone"]].drop_duplicates().merge(zone_agg, on="Zone")
            join, fid = "commune", "properties.code_commune"

        fig = px.choropleth(
            agg_df,
            geojson=geo,
            locations=join,
            featureidkey=fid,
            color=var,
            animation_frame="annee",
            color_continuous_scale=scale,
            labels={var: f"{indic} ({unit})"},
            title=f"{indic} – {level}"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            template="plotly_white",
            coloraxis_colorbar=dict(title=f"{indic} ({unit})", thickness=18, len=0.7)
        )

        if show_mean:
            mean_val = base[var].mean()
            fig.update_layout(title=f"{indic} – {level} | moyenne = {mean_val:.2f} {unit}")
            fig.add_annotation(
                x=0.01, y=0.98, xref="paper", yref="paper",
                text=f"Moyenne globale : {mean_val:.2f} {unit}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )

        st.plotly_chart(fig, width="stretch")

    # =====================================================
    # 🌡️ Carte animée des anomalies climatiques (z-score)
    # =====================================================
    st.divider()
    st.subheader("🌡️ Carte animée des anomalies climatiques (z-score)")

    anom_base = df[
        (df["type_mvt"] == "DECR") &
        (df["code_couleur"].isin(map_colors))
    ].dropna(subset=["rendement", "annee"]).copy()

    if anom_base.empty:
        st.info("Pas assez de données pour calculer les anomalies.")
    else:
        if level == "Département":
            rz = (
                anom_base.dropna(subset=["code_departement"])
                .groupby(["annee", "code_departement"])["rendement"]
                .mean()
                .reset_index()
                .rename(columns={"code_departement": "unit"})
            )
            geo_a = load_geojson("departements_languedoc.geojson")
            join_a, fid_a = "unit", "properties.dep"
        else:
            rz = (
                anom_base.dropna(subset=["Zone"])
                .groupby(["annee", "Zone"])["rendement"]
                .mean()
                .reset_index()
                .rename(columns={"Zone": "unit"})
            )
            geo_a = load_geojson("zones_pedoclimatiques.geojson")
            join_a, fid_a = "commune", "properties.code_commune"

        stats_y = rz.groupby("annee")["rendement"].agg(["mean", "std"]).reset_index()
        rz = rz.merge(stats_y, on="annee", how="left")
        rz["std"] = rz["std"].replace(0, np.nan)
        rz["zscore"] = (rz["rendement"] - rz["mean"]) / rz["std"]

        if level == "Département":
            amap = rz.copy()
            amap["unit"] = amap["unit"].astype(str)
            fig_anom = px.choropleth(
                amap,
                geojson=geo_a,
                locations="unit",
                featureidkey=fid_a,
                color="zscore",
                animation_frame="annee",
                color_continuous_scale="RdBu_r",
                range_color=(-3, 3),
                title="Anomalies climatiques (z-score annuel) – par département"
            )
        else:
            amap = (
                df_all[["commune", "Zone"]].drop_duplicates()
                .rename(columns={"Zone": "unit"})
                .merge(rz, on="unit", how="inner")
            )
            fig_anom = px.choropleth(
                amap,
                geojson=geo_a,
                locations=join_a,
                featureidkey=fid_a,
                color="zscore",
                animation_frame="annee",
                color_continuous_scale="RdBu_r",
                range_color=(-3, 3),
                title="Anomalies climatiques (z-score annuel) – par zone"
            )

        fig_anom.update_geos(fitbounds="locations", visible=False)
        fig_anom.update_layout(template="plotly_white")
        st.plotly_chart(fig_anom, width="stretch")


# =====================================================
# 📊 ANALYSE QUANTITATIVE
# =====================================================
with tab_quant:
    st.header("📊 Analyse quantitative (volume & rendement)")

    base = df.copy()
    base["prod_hl_ha"] = np.where(base["surface"] > 0, base["volume"] / base["surface"], np.nan)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    prod_global = (base["volume"].sum() / base["surface"].sum()) if base["surface"].sum() > 0 else np.nan
    corr_vr = base[["volume", "rendement"]].dropna().corr().iloc[0, 1] if base[["volume", "rendement"]].dropna().shape[0] > 2 else np.nan
    vol_rdt = base["rendement"].std()
    share_hors = (base["statut_plafond"].eq("🔴 Hors plafond").mean() * 100) if base["rendement"].notna().any() else np.nan

    c1.metric("Productivité moyenne (hl/ha)", f"{prod_global:.2f}" if pd.notna(prod_global) else "NA")
    c2.metric("Corrélation Volume / Rendement", f"{corr_vr:.2f}" if pd.notna(corr_vr) else "NA")
    c3.metric("Volatilité rendement (σ)", f"{vol_rdt:.2f}" if pd.notna(vol_rdt) else "NA")
    c4.metric("% Hors plafond", f"{share_hors:.1f}%" if pd.notna(share_hors) else "NA")

    st.divider()

    # 1) Surface vs Volume
    st.subheader("1) Relation Surface → Volume")
    tmp = base.dropna(subset=["surface", "volume"]).copy()
    fig1 = px.scatter(
        tmp, x="surface", y="volume",
        color="code_couleur",
        color_discrete_map=COLOR_MAP,
        trendline="ols",
        hover_data=["annee", "code_departement", "Zone"],
        title="Surface vs Volume (tendance OLS)"
    )
    fig1.update_layout(template="plotly_white")
    st.plotly_chart(fig1, width="stretch")

    st.divider()

    # 2) Rendement vs Volume
    st.subheader("2) Relation Rendement → Volume")
    tmp = base.dropna(subset=["rendement", "volume"]).copy()
    fig2 = px.scatter(
        tmp, x="rendement", y="volume",
        color="code_couleur",
        color_discrete_map=COLOR_MAP,
        trendline="ols",
        hover_data=["annee", "code_departement", "Zone"],
        title="Rendement vs Volume (tendance OLS)"
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, width="stretch")

    st.divider()

    # 3) Productivité par zone
    st.subheader("3) Productivité (hl/ha) par zone")
    tmp = base.dropna(subset=["prod_hl_ha"]).copy()
    prod_zone = tmp.groupby("Zone")["prod_hl_ha"].mean().reset_index()
    prod_zone = prod_zone.sort_values("Zone", key=lambda s: s.map(zone_sort_key))
    fig4 = px.bar(
        prod_zone, x="Zone", y="prod_hl_ha",
        color="Zone",
        category_orders={"Zone": zone_order_from_series(prod_zone["Zone"])},
        color_discrete_map=ZONE_COLOR_MAP,
        title="Productivité moyenne par zone (hl/ha)"
    )
    fig4.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig4, width="stretch")

    st.divider()

    # 4) Volatilité rendement par zone
    st.subheader("4) Stabilité / Volatilité du rendement")
    tmp = base.dropna(subset=["rendement"]).copy()
    vol_zone = tmp.groupby("Zone")["rendement"].std().reset_index()
    vol_zone = vol_zone.sort_values("Zone", key=lambda s: s.map(zone_sort_key))
    fig5 = px.bar(
        vol_zone, x="Zone", y="rendement",
        color="Zone",
        category_orders={"Zone": zone_order_from_series(vol_zone["Zone"])},
        color_discrete_map=ZONE_COLOR_MAP,
        title="Volatilité (écart-type) du rendement par zone"
    )
    fig5.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig5, width="stretch")

    st.divider()

    # =====================================================
    # IC95% Rendement (classique)
    # =====================================================
    st.subheader("📌 Rendements : moyenne + IC95% (classique) par zone (par couleur)")

    rend_ic = df[
        (df["type_mvt"] == "DECR") &
        (df["rendement"].between(10, 100))
    ].dropna(subset=["rendement", "Zone", "code_couleur"]).copy()

    zones_ic_r = st.multiselect(
        "Zones (IC rendement)",
        zone_order_from_series(rend_ic["Zone"]),
        default=zone_order_from_series(rend_ic["Zone"])[:3],
        key="ic_rend_zones"
    )
    cols_ic_r = st.multiselect(
        "Couleurs (IC rendement)",
        ["BL", "RG", "RS"],
        default=["BL", "RG", "RS"],
        key="ic_rend_cols"
    )

    rend_ic = rend_ic[rend_ic["Zone"].isin(zones_ic_r) & rend_ic["code_couleur"].isin(cols_ic_r)].copy()

    if rend_ic.empty:
        st.info("Pas assez de données pour calculer l’IC95% des rendements.")
    else:
        stats_r = (
            rend_ic.groupby(["Zone", "code_couleur"])["rendement"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        stats_r["se"] = stats_r["std"] / np.sqrt(stats_r["n"])
        stats_r["ic_inf"] = stats_r["mean"] - 1.96 * stats_r["se"]
        stats_r["ic_sup"] = stats_r["mean"] + 1.96 * stats_r["se"]

        fig_ic_r = px.bar(
            stats_r,
            x="Zone",
            y="mean",
            color="code_couleur",
            barmode="group",
            category_orders={"Zone": zone_order_from_series(stats_r["Zone"])},
            color_discrete_map=COLOR_MAP,
            title="Rendement moyen avec IC95% (classique) – par zone et par couleur"
        )
        fig_ic_r.update_traces(
            error_y=dict(
                type="data",
                symmetric=False,
                array=(stats_r["ic_sup"] - stats_r["mean"]).to_numpy(),
                arrayminus=(stats_r["mean"] - stats_r["ic_inf"]).to_numpy(),
            )
        )
        fig_ic_r.update_layout(template="plotly_white", yaxis_title="Rendement (hl/ha)")
        st.plotly_chart(fig_ic_r, width="stretch")

    st.divider()

    # =====================================================
    # IC95% Volume (bootstrap)
    # =====================================================
    st.subheader("📦 Volumes : moyenne + IC95% (bootstrap) par zone (par couleur)")

    vol_ic = df[df["type_mvt"] == "REVE"].dropna(subset=["volume", "Zone", "code_couleur"]).copy()

    zones_ic_v = st.multiselect(
        "Zones (IC volume)",
        zone_order_from_series(vol_ic["Zone"]),
        default=zone_order_from_series(vol_ic["Zone"])[:3],
        key="ic_vol_zones"
    )
    cols_ic_v = st.multiselect(
        "Couleurs (IC volume)",
        ["BL", "RG", "RS"],
        default=["BL", "RG", "RS"],
        key="ic_vol_cols"
    )

    vol_ic = vol_ic[vol_ic["Zone"].isin(zones_ic_v) & vol_ic["code_couleur"].isin(cols_ic_v)].copy()

    if vol_ic.empty:
        st.info("Pas assez de données pour calculer l’IC95% bootstrap des volumes.")
    else:
        rows = []
        for (z, c), sub in vol_ic.groupby(["Zone", "code_couleur"]):
            vals = sub["volume"].to_numpy(dtype=float)
            mean_val = np.nanmean(vals)
            ci_low, ci_high = bootstrap_ci(vals, n_boot=1500, seed=42)
            rows.append({"Zone": z, "code_couleur": c, "mean": mean_val, "ic_inf": ci_low, "ic_sup": ci_high})

        stats_v = pd.DataFrame(rows)

        fig_ic_v = px.bar(
            stats_v,
            x="Zone",
            y="mean",
            color="code_couleur",
            barmode="group",
            category_orders={"Zone": zone_order_from_series(stats_v["Zone"])},
            color_discrete_map=COLOR_MAP,
            title="Volume moyen avec IC95% (bootstrap) – par zone et par couleur"
        )
        fig_ic_v.update_traces(
            error_y=dict(
                type="data",
                symmetric=False,
                array=(stats_v["ic_sup"] - stats_v["mean"]).to_numpy(),
                arrayminus=(stats_v["mean"] - stats_v["ic_inf"]).to_numpy(),
            )
        )
        fig_ic_v.update_layout(template="plotly_white", yaxis_title="Volume (hl)")
        st.plotly_chart(fig_ic_v, width="stretch")

    st.divider()

    # =====================================================
    # Volume min/max par zone
    # =====================================================
    st.subheader("📦 Production en volume : min / max par zone")
    volz = df[df["type_mvt"] == "REVE"].dropna(subset=["volume", "Zone"]).copy()
    if volz.empty:
        st.info("Pas de données REVE pour calculer min/max volume par zone.")
    else:
        vol_minmax = (
            volz.groupby("Zone")["volume"]
            .agg(["min", "max"])
            .reset_index()
        )
        vol_minmax = vol_minmax.sort_values("Zone", key=lambda s: s.map(zone_sort_key))
        figmm = px.bar(
            vol_minmax.melt(id_vars="Zone", value_vars=["min", "max"], var_name="stat", value_name="volume"),
            x="Zone",
            y="volume",
            color="stat",
            category_orders={"Zone": zone_order_from_series(vol_minmax["Zone"])},
            barmode="group",
            title="Volume min et max par zone (sur la sélection)"
        )
        figmm.update_layout(template="plotly_white")
        st.plotly_chart(figmm, width="stretch")

    st.divider()

    # =====================================================
    # Détection anomalies + comparaison + diagnostic
    # =====================================================
    st.subheader("🌡️ Zones atypiques (anomalies climatiques) – année par année")

    an_cols = st.multiselect(
        "Couleurs (anomalies)",
        ["BL", "RG", "RS"],
        default=["BL", "RG", "RS"],
        key="anom_cols_quant"
    )

    an_base = df[
        (df["type_mvt"] == "DECR") &
        (df["rendement"].between(10, 100)) &
        (df["code_couleur"].isin(an_cols))
    ].dropna(subset=["rendement", "Zone", "annee"]).copy()

    if an_base.empty:
        st.info("Pas assez de données pour détecter des anomalies avec ces filtres.")
    else:
        rz = an_base.groupby(["annee", "Zone"])["rendement"].mean().reset_index()
        stats_year = rz.groupby("annee")["rendement"].agg(["mean", "std"]).reset_index()
        rz = rz.merge(stats_year, on="annee", how="left")
        rz["std"] = rz["std"].replace(0, np.nan)
        rz["zscore"] = (rz["rendement"] - rz["mean"]) / rz["std"]

        def classify(z):
            if pd.isna(z):
                return "NA"
            az = abs(z)
            if az < 1.5:
                return "Normal"
            elif az < 2.5:
                return "Atypique"
            else:
                return "Anomalie"

        rz["classe"] = rz["zscore"].apply(classify)

        years_sel = st.multiselect(
            "Années à analyser",
            sorted(rz["annee"].unique()),
            default=[max(rz["annee"].unique())],
            key="anom_years_quant"
        )
        rz_f = rz[rz["annee"].isin(years_sel)].copy()

        st.caption("Trié par |z-score| (les plus atypiques en haut)")
        st.dataframe(
            rz_f.sort_values("zscore", key=lambda s: s.abs(), ascending=False).round(2),
            width="stretch"
        )

        fig_an = px.scatter(
            rz_f,
            x="annee",
            y="rendement",
            color="classe",
            size=rz_f["zscore"].abs().fillna(0),
            hover_data=["Zone", "zscore"],
            title="Anomalies détectées – par année"
        )
        fig_an.update_layout(template="plotly_white")
        st.plotly_chart(fig_an, width="stretch")

        st.divider()
        st.subheader("🔎 Comparer une zone atypique (profil vs moyenne de l'année)")

        year_cmp = st.selectbox(
            "Année (comparaison)",
            sorted(rz["annee"].unique()),
            index=len(sorted(rz["annee"].unique())) - 1,
            key="cmp_year"
        )

        zones_year = zone_order_from_series(rz[rz["annee"] == year_cmp]["Zone"])
        zone_cmp_default = (
            rz[rz["annee"] == year_cmp]
            .assign(absz=lambda d: d["zscore"].abs())
            .sort_values("absz", ascending=False)["Zone"]
            .iloc[0]
            if len(zones_year) > 0 else None
        )

        zone_cmp = st.selectbox(
            "Zone (comparaison)",
            zones_year,
            index=zones_year.index(zone_cmp_default) if zone_cmp_default in zones_year else 0,
            key="cmp_zone"
        )

        base_year = df[(df["annee"] == year_cmp) & (df["type_mvt"] == "DECR")].copy()
        stats = (
            base_year.groupby("Zone")
            .agg(
                rendement=("rendement", "mean"),
                surface=("surface", "sum"),
                volume=("volume", "sum")
            )
            .reset_index()
        )
        stats["prod_hl_ha"] = np.where(stats["surface"] > 0, stats["volume"] / stats["surface"], np.nan)

        mean_vals = stats[["rendement", "surface", "volume", "prod_hl_ha"]].mean(numeric_only=True)
        zone_row = stats[stats["Zone"] == zone_cmp]

        if zone_row.empty:
            st.info("Zone introuvable pour cette année.")
        else:
            zone_vals = zone_row.iloc[0]
            comp = pd.DataFrame({
                "Indicateur": ["Rendement", "Surface", "Volume", "Productivité (hl/ha)"],
                "Zone": [zone_vals["rendement"], zone_vals["surface"], zone_vals["volume"], zone_vals["prod_hl_ha"]],
                "Moyenne (année)": [mean_vals["rendement"], mean_vals["surface"], mean_vals["volume"], mean_vals["prod_hl_ha"]],
            })
            st.dataframe(comp.round(2), width="stretch")

            fig_cmp = px.line_polar(
                comp.melt(id_vars="Indicateur", var_name="Groupe", value_name="Valeur"),
                r="Valeur",
                theta="Indicateur",
                color="Groupe",
                line_close=True,
                title=f"Profil – Zone {zone_cmp} vs moyenne (année {year_cmp})"
            )
            fig_cmp.update_layout(template="plotly_white")
            st.plotly_chart(fig_cmp, width="stretch")

            st.subheader("🧠 Interprétation automatique (cause probable)")

            z_row = rz[(rz["annee"] == year_cmp) & (rz["Zone"] == zone_cmp)]
            zscore = z_row["zscore"].iloc[0] if not z_row.empty else np.nan

            zone_detail = df[(df["annee"] == year_cmp) & (df["Zone"] == zone_cmp)].copy()
            other_detail = df[(df["annee"] == year_cmp) & (df["Zone"] != zone_cmp)].copy()

            def mean_rdt(data, color):
                sub = data[data["code_couleur"] == color]["rendement"].dropna()
                return sub.mean() if len(sub) > 5 else np.nan

            zone_colors = {c: mean_rdt(zone_detail, c) for c in ["BL", "RG", "RS"]}
            other_colors = {c: mean_rdt(other_detail, c) for c in ["BL", "RG", "RS"]}

            diag = []
            if pd.isna(zscore) or abs(zscore) < 1.5:
                diag.append("Situation globalement normale (pas d’anomalie forte sur le rendement).")
            else:
                diff = {c: (zone_colors[c] - other_colors[c]) for c in ["BL", "RG", "RS"]}

                if pd.notna(diff["BL"]) and pd.notna(diff["RG"]):
                    if diff["BL"] < -5 and diff["RG"] > -2:
                        diag.append("Signature compatible avec un gel printanier (blancs plus touchés).")
                    if diff["RG"] < -5 and diff["BL"] > -2:
                        diag.append("Signature compatible avec un stress hydrique estival (rouges plus touchés).")

                if all(pd.notna(diff[c]) for c in ["BL", "RG", "RS"]):
                    if diff["BL"] < -5 and diff["RG"] < -5 and diff["RS"] < -5:
                        diag.append("Baisse sur toutes les couleurs : aléa généralisé (pluie, grêle, maladie…).")

                if pd.notna(zscore) and zscore > 2:
                    diag.append("Rendement anormalement élevé : conditions très favorables / terroir résilient / pratiques culturales.")

                if not diag:
                    diag.append("Cause indéterminée (sol, conduite, cépages, exposition…).")

            for d in diag:
                st.info(d)

    st.divider()

    # =====================================================
    # Tableau "zones efficaces"
    # =====================================================
    st.subheader("🏆 Zones les plus efficaces (productivité & conformité)")
    tmp = base.dropna(subset=["surface", "volume", "rendement", "plafond"]).copy()
    tmp["prod"] = np.where(tmp["surface"] > 0, tmp["volume"] / tmp["surface"], np.nan)
    summary = (
        tmp.groupby("Zone")
        .agg(
            surface_ha=("surface", "sum"),
            volume_hl=("volume", "sum"),
            rendement_moy=("rendement", "mean"),
            plafond_moy=("plafond", "mean"),
            prod_hl_ha=("prod", "mean"),
            pct_hors=("statut_plafond", lambda s: (s == "🔴 Hors plafond").mean() * 100)
        )
        .reset_index()
    )
    summary = summary.sort_values("Zone", key=lambda s: s.map(zone_sort_key))
    st.dataframe(summary.round(2), width="stretch")