import shutil
from pathlib import Path
import os

def copy_data_files():
    """
    Copy necessary data files from the original replication package to our new project structure.
    """
    # Define paths
    original_root = Path("../BurkeHsiangMiguel2015_Replication")
    new_root = Path(__file__).parent.parent
    
    # Create necessary directories
    (new_root / "data" / "input" / "CCprojections").mkdir(parents=True, exist_ok=True)
    (new_root / "data" / "input" / "shape").mkdir(parents=True, exist_ok=True)
    (new_root / "data" / "input" / "populationData").mkdir(parents=True, exist_ok=True)
    (new_root / "data" / "output" / "bootstrap").mkdir(parents=True, exist_ok=True)
    
    # Copy temperature projection data
    print("Copying temperature projection data...")
    shutil.copy2(
        original_root / "data" / "input" / "CCprojections" / "diff_tas_Amon_modmean_rcp85_000_2081-2100_minus_1986-2005_mon1_ave12_withsd.nc",
        new_root / "data" / "input" / "CCprojections"
    )
    
    # Copy shapefile data
    print("Copying country shapefile data...")
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.sbx', '.sbn']:
        shutil.copy2(
            original_root / "data" / "input" / "shape" / f"country{ext}",
            new_root / "data" / "input" / "shape"
        )
    
    # Copy population data
    print("Copying population data...")
    shutil.copy2(
        original_root / "data" / "input" / "populationData" / "glp00ag30.asc",
        new_root / "data" / "input" / "populationData"
    )
    
    # Copy growth climate dataset
    print("Copying growth climate dataset...")
    shutil.copy2(
        original_root / "data" / "input" / "GrowthClimateDataset.csv",
        new_root / "data" / "input"
    )
    
    print("Data files copied successfully!")

if __name__ == "__main__":
    copy_data_files() 