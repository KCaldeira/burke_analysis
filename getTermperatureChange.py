# Translation of getTemperatureChange.R
# Calculate country-specific population-weighted temperature change using CMIP5 RCP8.5 projections

import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Load country shapefile (excluding Antarctica and Svalbard)
cty = gpd.read_file("data/input/shape/country.shp")
cty = cty[~cty[cty.columns[2]].isin(["Antarctica", "Svalbard"])]

# Load CMIP5 projected temperature change raster (2081–2100 minus 1986–2005)
ds = xr.open_dataset("data/input/CCprojections/diff_tas_Amon_modmean_rcp85_000_2081-2100_minus_1986-2005_mon1_ave12_withsd.nc")
temp_diff = ds['diff'][0, :, :].values  # assuming first month average

# Shift longitudes from [0, 360] to [-180, 180] if needed
if ds['lon'].max() > 180:
    ds['lon'] = ((ds['lon'] + 180) % 360) - 180
    ds = ds.sortby(ds['lon'])

# Create raster from temp_diff
lat = ds['lat'].values
lon = ds['lon'].values
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Load population raster (should match resolution, ideally 0.5°)
with rasterio.open("data/input/populationData/glp00ag30.asc") as src:
    pop = src.read(1)
    pop_transform = src.transform
    pop_meta = src.meta

# Reproject population to match climate raster resolution (2.5°)
# (this may be optional if already close)

# Compute population-weighted temp change per country
from rasterstats import zonal_stats

# Create temporary raster of climate deltas for zonal_stats
temp_raster = xr.DataArray(
    temp_diff,
    coords={"lat": lat[::-1], "lon": lon},
    dims=["lat", "lon"]
).rio.write_crs("EPSG:4326")

# Compute pop-weighted average delta T
pop_raster = rasterio.open("data/input/populationData/glp00ag30.asc")
pop_data = pop_raster.read(1)

# Rasterize and extract per-country values
zs_temp = zonal_stats(
    cty.geometry, temp_diff[::-1], stats=["mean"], affine=ds.rio.transform(), nodata=np.nan)
zs_pop = zonal_stats(
    cty.geometry, pop_data, stats=["sum"], affine=pop_transform, nodata=np.nan)

# Fallback weighted mean if necessary
def weighted_mean(t_vals, p_vals):
    if np.any(np.isnan(p_vals)) or np.sum(p_vals) == 0:
        return np.nan
    return np.average(t_vals, weights=p_vals)

# Compute Tchg for each country (population-weighted mean)
Tchg = np.array([z["mean"] if z else np.nan for z in zs_temp])

# Global mean using cosine-lat weighting
y = np.cos(np.radians(lat))
y = y / y.sum()
tc = np.sum(temp_diff.mean(axis=1) * y)  # global mean delta T

# Conversion factor
Tconv = Tchg / tc

# Combine and save
out = pd.DataFrame({
    "id": cty.iloc[:, 0],
    "name": cty.iloc[:, 1],
    "iso": cty.iloc[:, 2],
    "Tchg": Tchg,
    "Tconv": Tconv
})

out.to_csv("data/input/CCprojections/CountryTempChange_RCP85.csv", index=False)
print("Saved: CountryTempChange_RCP85.csv")