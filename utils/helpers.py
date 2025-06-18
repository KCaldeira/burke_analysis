import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin
import rasterio
from rasterio.warp import transform_geom
import warnings

def weighted_mean(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate weighted mean, handling edge cases.
    
    Args:
        x: Values array
        y: Weights array
        
    Returns:
        Weighted mean if valid, otherwise mean of x
    """
    if len(x) > 1 and np.sum(y) != 0:
        return np.average(x, weights=y)
    return np.mean(x)

def read_netcdf_temperature(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Read temperature data from NetCDF file.
    
    Args:
        file_path: Path to NetCDF file
        
    Returns:
        Tuple of (temperature data array, metadata)
    """
    with xr.open_dataset(file_path) as ds:
        temp_data = ds['diff'].values
        metadata = {
            'lat': ds.lat.values,
            'lon': ds.lon.values
        }
    return temp_data, metadata

def read_population_data(file_path: str) -> np.ndarray:
    """
    Read population data from ASCII grid file.
    
    Args:
        file_path: Path to population data file
        
    Returns:
        Population data array
    """
    with rasterio.open(file_path) as src:
        return src.read(1)

def aggregate_population(pop_data: np.ndarray, factor: int) -> np.ndarray:
    """
    Aggregate population data to match GCM resolution.
    
    Args:
        pop_data: Population data array
        factor: Aggregation factor
        
    Returns:
        Aggregated population data
    """
    return np.array([
        np.sum(pop_data[i:i+factor, j:j+factor])
        for i in range(0, pop_data.shape[0], factor)
        for j in range(0, pop_data.shape[1], factor)
    ]).reshape(pop_data.shape[0]//factor, pop_data.shape[1]//factor)

def calculate_latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    """
    Calculate cosine latitude weights for global mean temperature.
    
    Args:
        latitudes: Array of latitudes in degrees
        
    Returns:
        Array of weights
    """
    weights = np.cos(np.radians(latitudes))
    return weights / np.sum(weights)

def extract_country_data(
    temp_data: np.ndarray,
    pop_data: np.ndarray,
    countries: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Extract temperature and population data for each country.
    
    Args:
        temp_data: Temperature data array
        pop_data: Population data array
        countries: GeoDataFrame of country boundaries
        
    Returns:
        DataFrame with country-level data
    """
    results = []
    for idx, country in countries.iterrows():
        if country['NAME'] in ['Antarctica', 'Svalbard']:
            continue
            
        # Extract data for country
        country_temp = extract_country_values(temp_data, country.geometry)
        country_pop = extract_country_values(pop_data, country.geometry)
        
        # Calculate weighted mean
        temp_change = weighted_mean(country_temp, country_pop)
        
        results.append({
            'country_code': country['ISO_A3'],
            'country_name': country['NAME'],
            'temperature_change': temp_change
        })
    
    return pd.DataFrame(results)

def extract_country_values(
    data: np.ndarray,
    geometry: Union[Polygon, MultiPolygon]
) -> np.ndarray:
    """
    Extract values from data array that fall within country geometry.
    
    Args:
        data: Data array
        geometry: Country geometry
        
    Returns:
        Array of values within country
    """
    # Implementation will depend on the exact format of your data
    # This is a placeholder for the actual implementation
    pass 