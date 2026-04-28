import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


# =====================================================
# NETCDF LOADING
# =====================================================

def open_concat_small(files):
    if not files:
        raise ValueError("Liste de fichiers vide")

    return xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={"time": 1},
        data_vars="all"
    )


# =====================================================
# UTILS
# =====================================================

def fix_lat_lon(da):
    if "lat" in da.coords:
        lat = da["lat"].values
        lon = da["lon"].values
        lat_name, lon_name = "lat", "lon"
    else:
        lat = da["latitude"].values
        lon = da["longitude"].values
        lat_name, lon_name = "latitude", "longitude"

    if lat.ndim == 2:
        lat = lat[:, 0]
    if lon.ndim == 2:
        lon = lon[0, :]

    if np.any(np.diff(lat) < 0):
        lat = lat[::-1]
        da = da.reindex({lat_name: lat})

    if np.any(np.diff(lon) < 0):
        lon = lon[::-1]
        da = da.reindex({lon_name: lon})

    return da, lat, lon


def interpolate_grid(da, lat, lon, points):
    values = np.empty((da.sizes["time"], len(points)))

    for i in range(da.sizes["time"]):
        interp = RegularGridInterpolator(
            (lat, lon),
            da.isel(time=i).values,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        values[i, :] = interp(points)

    return values


def load_scenario(dataset_list, scenario):
    return (
        open_concat_small([f for f in dataset_list if f"tas_{scenario}" in f.lower()]),
        open_concat_small([f for f in dataset_list if f"tasmax_{scenario}" in f.lower()]),
        open_concat_small([f for f in dataset_list if f"tasmin_{scenario}" in f.lower()]),
        open_concat_small([f for f in dataset_list if f"pr_{scenario}" in f.lower()]),
    )


# =====================================================
# CORE COMPUTATION
# =====================================================

def compute_indicators(dataset_list, insee_file, copernicus_file, scenario):

    df_cities = pd.read_csv(insee_file)
    df_cities = df_cities.dropna(subset=["latitude", "longitude", "insee_code"])

    df_cop = pd.read_csv(copernicus_file, sep=";")
    df_cop["Municipality"] = df_cop["Municipality"].astype(str)

    df_cities["insee_code"] = df_cities["insee_code"].astype(str)
    df_cities = df_cities[df_cities["insee_code"].isin(df_cop["Municipality"])]

    city_lat = df_cities["latitude"].values
    city_lon = df_cities["longitude"].values
    points = np.column_stack((city_lat, city_lon))

    tas, tasmax, tasmin, pr = load_scenario(dataset_list, scenario)

    years = np.arange(2026, 2056)

    da_tas = tas["tas"].sel(time=tas["time"].dt.year.isin(years))
    da_max = tasmax["tasmax"].sel(time=tasmax["time"].dt.year.isin(years))
    da_min = tasmin["tasmin"].sel(time=tasmin["time"].dt.year.isin(years))
    da_pr = pr["pr"].sel(time=pr["time"].dt.year.isin(years))

    da_max, lat, lon = fix_lat_lon(da_max)
    da_min, _, _ = fix_lat_lon(da_min)
    da_tas, _, _ = fix_lat_lon(da_tas)

    ts_max = interpolate_grid(da_max, lat, lon, points) - 273.15
    ts_min = interpolate_grid(da_min, lat, lon, points) - 273.15
    tas_val = interpolate_grid(da_tas, lat, lon, points) - 273.15

    results = []

    for i, code in enumerate(df_cities["insee_code"]):

        df_tmp = pd.DataFrame({
            "Year": da_max["time"].dt.year.values,
            "tmax": ts_max[:, i],
            "tmin": ts_min[:, i],
            "tmean": tas_val[:, i]
        })

        df_tmp["Tmean"] = (df_tmp["tmax"] + df_tmp["tmin"]) / 2

        grouped = df_tmp.groupby("Year")

        for year, g in grouped:

            tx = g["tmax"]
            tn = g["tmin"]
            tm = g["Tmean"]

            hot_d = (tx > 30).sum()
            very_hot_d = (tx > 35).sum()
            late_frost = (tn < 0).sum()
            severe_heat = (tm - 28).clip(lower=0).sum()
            severe_frost = (2 - tn).clip(lower=0).sum()

            # Huglin
            huglin = (((tx - 10).clip(lower=0) + (tm - 10).clip(lower=0)) / 2 * 1.03).sum()

            results.append({
                "Year": int(year),
                "Municipality": code,
                "Huglin_Index": float(huglin),
                "Hot_D": int(hot_d),
                "Very_Hot_D": int(very_hot_d),
                "Late_Frost": int(late_frost),
                "Sever_Heat": float(severe_heat),
                "Sever_Frost": float(severe_frost),
                "cluster": df_cop.loc[df_cop["Municipality"] == code, "cluster"].values[0]
                if "cluster" in df_cop.columns else np.nan
            })

    df_result = pd.DataFrame(results)

    return df_result


# =====================================================
# SCENARIOS
# =====================================================

def scenario_optimiste(dataset_list, insee_file, copernicus_file):
    return compute_indicators(dataset_list, insee_file, copernicus_file, "rcp26")


def scenario_neutre(dataset_list, insee_file, copernicus_file):
    return compute_indicators(dataset_list, insee_file, copernicus_file, "rcp45")


def scenario_pessimiste(dataset_list, insee_file, copernicus_file):
    return compute_indicators(dataset_list, insee_file, copernicus_file, "rcp85")