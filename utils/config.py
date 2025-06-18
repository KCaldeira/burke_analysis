import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
COUNTRY_SHAPE_FILE = INPUT_DIR / "shape" / "country.shp"
TEMPERATURE_PROJECTION_FILE = INPUT_DIR / "CCprojections" / "diff_tas_Amon_modmean_rcp85_000_2081-2100_minus_1986-2005_mon1_ave12_withsd.nc"
POPULATION_DATA_FILE = INPUT_DIR / "populationData" / "glp00ag30.asc"
COUNTRY_TEMP_CHANGE_OUTPUT = INPUT_DIR / "CCprojections" / "CountryTempChange_RCP85.csv"

# Constants
LATITUDE_RANGE = (-90, 90)
LONGITUDE_RANGE = (-180, 180)
POPULATION_AGGREGATION_FACTOR = 5  # For aggregating population data to match GCM resolution 