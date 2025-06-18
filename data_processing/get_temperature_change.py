import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import xarray as xr
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, MultiPolygon
import warnings

from utils.config import (
    COUNTRY_SHAPE_FILE,
    TEMPERATURE_PROJECTION_FILE,
    POPULATION_DATA_FILE,
    COUNTRY_TEMP_CHANGE_OUTPUT,
    POPULATION_AGGREGATION_FACTOR
)
from utils.helpers import (
    weighted_mean,
    read_netcdf_temperature,
    read_population_data,
    aggregate_population,
    calculate_latitude_weights,
    extract_country_data
)

def main():
    print("Calculating country-specific temperature changes...")
    
    # Read country shapefile
    countries = gpd.read_file(COUNTRY_SHAPE_FILE)
    countries = countries[~countries['NAME'].isin(['Antarctica', 'Svalbard'])]
    
    # Read temperature projections
    temp_data, temp_metadata = read_netcdf_temperature(TEMPERATURE_PROJECTION_FILE)
    
    # Read and process population data
    pop_data = read_population_data(POPULATION_DATA_FILE)
    pop_aggregated = aggregate_population(pop_data, POPULATION_AGGREGATION_FACTOR)
    
    # Calculate country-specific temperature changes
    country_data = extract_country_data(temp_data, pop_aggregated, countries)
    
    # Calculate global mean temperature change
    lat_weights = calculate_latitude_weights(temp_metadata['lat'])
    global_mean_temp = np.average(temp_data, weights=lat_weights)
    
    # Calculate conversion factors
    country_data['conversion_factor'] = country_data['temperature_change'] / global_mean_temp
    
    # Save results
    country_data.to_csv(COUNTRY_TEMP_CHANGE_OUTPUT, index=False)
    print(f"Results saved to {COUNTRY_TEMP_CHANGE_OUTPUT}")

if __name__ == "__main__":
    main() 