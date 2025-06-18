import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import warnings
from typing import List, Tuple, Dict
import os

from utils.config import (
    DATA_DIR,
    INPUT_DIR,
    OUTPUT_DIR
)

def check_inf_nan(df, context=""):
    """Check for inf or NaN values in the DataFrame and print out which columns contain them."""
    inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    nan_cols = df.columns[df.isna().any()].tolist()
    if inf_cols:
        print(f"[{context}] Columns with inf values: {inf_cols}")
    if nan_cols:
        print(f"[{context}] Columns with NaN values: {nan_cols}")

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    # Convert 'iso' to string and drop rows with missing values
    df['iso'] = df['iso'].astype(str)
    df = df.dropna(subset=['iso'])
    # Convert numeric columns to float
    numeric_cols = ['growthWDI', 'UDel_temp_popweight', 'UDel_precip_popweight', 'GDPpctile_WDIppp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Create iso_id for dummy variables
    df['iso_id'] = df['iso']
    return df

def create_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create year and country dummy variables."""
    # Convert year and iso_id to strings to ensure proper dummy creation
    df['year'] = df['year'].astype(str)
    df['iso_id'] = df['iso_id'].astype(str)
    
    # Create dummy variables
    year_dummies = pd.get_dummies(df['year'], prefix='year')
    iso_dummies = pd.get_dummies(df['iso_id'], prefix='iso')
    
    # Check for NaN values in dummy variables
    print("\n[create_dummies] Checking dummy variables for NaN values...")
    year_nan = year_dummies.isna().any(axis=1)
    iso_nan = iso_dummies.isna().any(axis=1)
    if year_nan.any() or iso_nan.any():
        print(f"Found {year_nan.sum()} rows with NaN in year dummies")
        print(f"Found {iso_nan.any().sum()} rows with NaN in iso dummies")
        # Drop rows with NaN in dummies
        df = df[~(year_nan | iso_nan)]
        year_dummies = year_dummies[~(year_nan | iso_nan)]
        iso_dummies = iso_dummies[~(year_nan | iso_nan)]
    
    # Combine with original dataframe
    return pd.concat([df, year_dummies, iso_dummies], axis=1)

def drop_missing(X, y, context=""):
    """Drop rows with missing values in X or y and print how many were dropped."""
    initial_rows = X.shape[0]
    print(f"[{context}] Initial shape: X={X.shape}, y={y.shape}")
    # Replace inf values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # Check for inf values
    inf_cols = X.columns[X.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        print(f"[{context}] Columns with inf values: {inf_cols}")
        for col in inf_cols:
            print(f"[{context}] Values in {col}: {X[col].unique()}")
    # Check for NaN values
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"[{context}] Columns with NaN values: {nan_cols}")
        for col in nan_cols:
            print(f"[{context}] Values in {col}: {X[col].unique()}")
    mask = X.notnull().all(axis=1) & y.notnull()
    X_clean = X[mask].astype(float)
    y_clean = y[mask].astype(float)
    dropped = initial_rows - X_clean.shape[0]
    if dropped > 0:
        print(f"[{context}] Dropped {dropped} rows due to missing data ({dropped/initial_rows:.2%} of rows).")
    print(f"[{context}] Final shape: X={X_clean.shape}, y={y_clean.shape}")
    return X_clean, y_clean

def bootstrap_model(
    data: pd.DataFrame,
    model_type: str = 'no_lag',
    n_bootstrap: int = 1000,
    seed: int = 8675309
) -> pd.DataFrame:
    """
    Run bootstrap analysis for different model specifications.
    
    Args:
        data: Input DataFrame
        model_type: Type of model ('no_lag', 'rich_poor', '5lag', 'rich_poor_5lag')
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with bootstrap results
    """
    np.random.seed(seed)
    df = data.copy()
    # Drop rows with NaN values in original columns
    df = df.dropna(subset=['UDel_temp_popweight', 'UDel_precip_popweight'])
    # Calculate squared terms
    df['UDel_temp_popweight_2'] = df['UDel_temp_popweight'] ** 2
    df['UDel_precip_popweight_2'] = df['UDel_precip_popweight'] ** 2
    check_inf_nan(df, context="bootstrap_model after squared terms")
    if model_type in ['rich_poor', 'rich_poor_5lag']:
        df['poor'] = (df['GDPpctile_WDIppp'] < 50).astype(float)
        df.loc[df['GDPpctile_WDIppp'].isna(), 'poor'] = np.nan
    df = create_dummies(df)
    if model_type == 'no_lag':
        return bootstrap_no_lag(df, n_bootstrap)
    elif model_type == 'rich_poor':
        return bootstrap_rich_poor(df, n_bootstrap)
    elif model_type == '5lag':
        return bootstrap_5lag(df, n_bootstrap)
    elif model_type == 'rich_poor_5lag':
        return bootstrap_rich_poor_5lag(df, n_bootstrap)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def bootstrap_no_lag(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    results = []
    year_cols = [col for col in df.columns if col.startswith('year_')]
    iso_cols = [col for col in df.columns if col.startswith('iso_')]
    X = df[['UDel_temp_popweight', 'UDel_temp_popweight_2',
            'UDel_precip_popweight', 'UDel_precip_popweight_2']]
    X['const'] = 1.0
    X = pd.concat([X, df[year_cols], df[iso_cols]], axis=1)
    y = df['growthWDI']
    # Drop rows with NaN values in all columns used for regression
    X = X.dropna()
    y = y[X.index]
    X, y = drop_missing(X, y, context="no_lag baseline")
    # --- Begin diagnostics ---
    print("[DIAGNOSTIC] Checking for NaN/inf in X before model fit...")
    nan_mask = X.isna().any(axis=1)
    inf_mask = X.applymap(np.isinf).any(axis=1)
    if nan_mask.any() or inf_mask.any():
        print(f"[DIAGNOSTIC] Rows with NaN: {nan_mask.sum()}, Rows with inf: {inf_mask.sum()}")
        print("[DIAGNOSTIC] Columns with NaN values:", X.columns[X.isna().any()].tolist())
        print("[DIAGNOSTIC] Columns with inf values:", X.columns[(X.applymap(np.isinf)).any()].tolist())
        print("[DIAGNOSTIC] Indices with NaN:", X.index[nan_mask].tolist()[:10])
        print("[DIAGNOSTIC] Indices with inf:", X.index[inf_mask].tolist()[:10])
        print("[DIAGNOSTIC] First few problematic rows:")
        print(X[nan_mask | inf_mask].head())
    else:
        print("[DIAGNOSTIC] No NaN or inf in X before model fit.")
    # --- End diagnostics ---
    model = sm.OLS(y, X).fit()
    results.append({
        'run': 0,
        'temp': model.params['UDel_temp_popweight'],
        'temp2': model.params['UDel_temp_popweight_2'],
        'prec': model.params['UDel_precip_popweight'],
        'prec2': model.params['UDel_precip_popweight_2']
    })
    for i in range(1, n_bootstrap + 1):
        countries = df['iso_id'].unique()
        sampled_countries = np.random.choice(countries, size=len(countries), replace=True)
        boot_df = pd.concat([df[df['iso_id'] == c] for c in sampled_countries])
        X = boot_df[['UDel_temp_popweight', 'UDel_temp_popweight_2',
                    'UDel_precip_popweight', 'UDel_precip_popweight_2']]
        X['const'] = 1.0
        X = pd.concat([X, boot_df[year_cols], boot_df[iso_cols]], axis=1)
        y = boot_df['growthWDI']
        # Drop rows with NaN values in all columns used for regression
        X = X.dropna()
        y = y[X.index]
        X, y = drop_missing(X, y, context=f"no_lag bootstrap {i}")
        model = sm.OLS(y, X).fit()
        results.append({
            'run': i,
            'temp': model.params['UDel_temp_popweight'],
            'temp2': model.params['UDel_temp_popweight_2'],
            'prec': model.params['UDel_precip_popweight'],
            'prec2': model.params['UDel_precip_popweight_2']
        })
    return pd.DataFrame(results)

def bootstrap_rich_poor(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    results = []
    year_cols = [col for col in df.columns if col.startswith('year_')]
    iso_cols = [col for col in df.columns if col.startswith('iso_')]
    X = df[['UDel_temp_popweight', 'UDel_temp_popweight_2',
            'UDel_precip_popweight', 'UDel_precip_popweight_2']]
    X['const'] = 1.0
    X = pd.concat([X, df[year_cols], df[iso_cols]], axis=1)
    X['poor_temp'] = df['poor'] * df['UDel_temp_popweight']
    X['poor_temp2'] = df['poor'] * df['UDel_temp_popweight_2']
    X['poor_prec'] = df['poor'] * df['UDel_precip_popweight']
    X['poor_prec2'] = df['poor'] * df['UDel_precip_popweight_2']
    y = df['growthWDI']
    # Drop rows with NaN values in all columns used for regression
    X = X.dropna()
    y = y[X.index]
    X, y = drop_missing(X, y, context="rich_poor baseline")
    model = sm.OLS(y, X).fit()
    results.append({
        'run': 0,
        'temp': model.params['UDel_temp_popweight'],
        'temppoor': model.params['poor_temp'],
        'temp2': model.params['UDel_temp_popweight_2'],
        'temp2poor': model.params['poor_temp2'],
        'prec': model.params['UDel_precip_popweight'],
        'precpoor': model.params['poor_prec'],
        'prec2': model.params['UDel_precip_popweight_2'],
        'prec2poor': model.params['poor_prec2']
    })
    for i in range(1, n_bootstrap + 1):
        countries = df['iso_id'].unique()
        sampled_countries = np.random.choice(countries, size=len(countries), replace=True)
        boot_df = pd.concat([df[df['iso_id'] == c] for c in sampled_countries])
        X = boot_df[['UDel_temp_popweight', 'UDel_temp_popweight_2',
                    'UDel_precip_popweight', 'UDel_precip_popweight_2']]
        X['const'] = 1.0
        X = pd.concat([X, boot_df[year_cols], boot_df[iso_cols]], axis=1)
        X['poor_temp'] = boot_df['poor'] * boot_df['UDel_temp_popweight']
        X['poor_temp2'] = boot_df['poor'] * boot_df['UDel_temp_popweight_2']
        X['poor_prec'] = boot_df['poor'] * boot_df['UDel_precip_popweight']
        X['poor_prec2'] = boot_df['poor'] * boot_df['UDel_precip_popweight_2']
        y = boot_df['growthWDI']
        # Drop rows with NaN values in all columns used for regression
        X = X.dropna()
        y = y[X.index]
        X, y = drop_missing(X, y, context=f"rich_poor bootstrap {i}")
        model = sm.OLS(y, X).fit()
        results.append({
            'run': i,
            'temp': model.params['UDel_temp_popweight'],
            'temppoor': model.params['poor_temp'],
            'temp2': model.params['UDel_temp_popweight_2'],
            'temp2poor': model.params['poor_temp2'],
            'prec': model.params['UDel_precip_popweight'],
            'precpoor': model.params['poor_prec'],
            'prec2': model.params['UDel_precip_popweight_2'],
            'prec2poor': model.params['poor_prec2']
        })
    return pd.DataFrame(results)

def create_lagged_variables(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """Create lagged variables for temperature and precipitation."""
    df = df.sort_values(['iso_id', 'year'])
    for i in range(1, n_lags + 1):
        df[f'L{i}_UDel_temp_popweight'] = df.groupby('iso_id')['UDel_temp_popweight'].shift(i)
        df[f'L{i}_UDel_temp_popweight_2'] = df.groupby('iso_id')['UDel_temp_popweight_2'].shift(i)
        df[f'L{i}_UDel_precip_popweight'] = df.groupby('iso_id')['UDel_precip_popweight'].shift(i)
        df[f'L{i}_UDel_precip_popweight_2'] = df.groupby('iso_id')['UDel_precip_popweight_2'].shift(i)
    check_inf_nan(df, context="create_lagged_variables")
    return df

def bootstrap_5lag(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    results = []
    df = create_lagged_variables(df)
    year_cols = [col for col in df.columns if col.startswith('year_')]
    iso_cols = [col for col in df.columns if col.startswith('iso_')]
    lag_vars = []
    for i in range(6):
        prefix = f'L{i}_' if i > 0 else ''
        lag_vars.extend([
            f'{prefix}UDel_temp_popweight',
            f'{prefix}UDel_temp_popweight_2',
            f'{prefix}UDel_precip_popweight',
            f'{prefix}UDel_precip_popweight_2'
        ])
    X = df[lag_vars]
    X['const'] = 1.0
    X = pd.concat([X, df[year_cols], df[iso_cols]], axis=1)
    y = df['growthWDI']
    # Drop rows with NaN values in all columns used for regression
    X = X.dropna()
    y = y[X.index]
    X, y = drop_missing(X, y, context="5lag baseline")
    model = sm.OLS(y, X).fit()
    result = {'run': 0}
    for var in lag_vars:
        result[var] = model.params[var]
    results.append(result)
    for i in range(1, n_bootstrap + 1):
        countries = df['iso_id'].unique()
        sampled_countries = np.random.choice(countries, size=len(countries), replace=True)
        boot_df = pd.concat([df[df['iso_id'] == c] for c in sampled_countries])
        X = boot_df[lag_vars]
        X['const'] = 1.0
        X = pd.concat([X, boot_df[year_cols], boot_df[iso_cols]], axis=1)
        y = boot_df['growthWDI']
        # Drop rows with NaN values in all columns used for regression
        X = X.dropna()
        y = y[X.index]
        X, y = drop_missing(X, y, context=f"5lag bootstrap {i}")
        model = sm.OLS(y, X).fit()
        result = {'run': i}
        for var in lag_vars:
            result[var] = model.params[var]
        results.append(result)
    results_df = pd.DataFrame(results)
    results_df['tlin'] = results_df[[f'L{i}_UDel_temp_popweight' for i in range(6)]].sum(axis=1)
    results_df['tsq'] = results_df[[f'L{i}_UDel_temp_popweight_2' for i in range(6)]].sum(axis=1)
    return results_df

def bootstrap_rich_poor_5lag(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    results = []
    df = create_lagged_variables(df)
    year_cols = [col for col in df.columns if col.startswith('year_')]
    iso_cols = [col for col in df.columns if col.startswith('iso_')]
    lag_vars = []
    for i in range(6):
        prefix = f'L{i}_' if i > 0 else ''
        lag_vars.extend([
            f'{prefix}UDel_temp_popweight',
            f'{prefix}UDel_temp_popweight_2',
            f'{prefix}UDel_precip_popweight',
            f'{prefix}UDel_precip_popweight_2'
        ])
    X = df[lag_vars]
    X['const'] = 1.0
    X = pd.concat([X, df[year_cols], df[iso_cols]], axis=1)
    for var in lag_vars:
        X[f'poor_{var}'] = df['poor'] * df[var]
    y = df['growthWDI']
    # Drop rows with NaN values in all columns used for regression
    X = X.dropna()
    y = y[X.index]
    X, y = drop_missing(X, y, context="rich_poor_5lag baseline")
    model = sm.OLS(y, X).fit()
    result = {'run': 0}
    for var in lag_vars:
        result[var] = model.params[var]
        result[f'poor_{var}'] = model.params[f'poor_{var}']
    results.append(result)
    for i in range(1, n_bootstrap + 1):
        countries = df['iso_id'].unique()
        sampled_countries = np.random.choice(countries, size=len(countries), replace=True)
        boot_df = pd.concat([df[df['iso_id'] == c] for c in sampled_countries])
        X = boot_df[lag_vars]
        X['const'] = 1.0
        X = pd.concat([X, boot_df[year_cols], boot_df[iso_cols]], axis=1)
        for var in lag_vars:
            X[f'poor_{var}'] = boot_df['poor'] * boot_df[var]
        y = boot_df['growthWDI']
        # Drop rows with NaN values in all columns used for regression
        X = X.dropna()
        y = y[X.index]
        X, y = drop_missing(X, y, context=f"rich_poor_5lag bootstrap {i}")
        model = sm.OLS(y, X).fit()
        result = {'run': i}
        for var in lag_vars:
            result[var] = model.params[var]
            result[f'poor_{var}'] = model.params[f'poor_{var}']
        results.append(result)
    results_df = pd.DataFrame(results)
    results_df['tlin'] = results_df[[f'L{i}_UDel_temp_popweight' for i in range(6)]].sum(axis=1)
    results_df['tlinpoor'] = results_df[[f'poor_L{i}_UDel_temp_popweight' for i in range(6)]].sum(axis=1)
    results_df['tsq'] = results_df[[f'L{i}_UDel_temp_popweight_2' for i in range(6)]].sum(axis=1)
    results_df['tsqpoor'] = results_df[[f'poor_L{i}_UDel_temp_popweight_2' for i in range(6)]].sum(axis=1)
    return results_df

def main():
    """Main function to run all bootstrap analyses."""
    print("Generating bootstrap data...")
    
    # Create output directory
    bootstrap_dir = OUTPUT_DIR / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(INPUT_DIR / "GrowthClimateDataset.csv")
    
    # Run different bootstrap analyses
    models = {
        'no_lag': 'bootstrap_noLag.csv',
        'rich_poor': 'bootstrap_richpoor.csv',
        '5lag': 'bootstrap_5Lag.csv',
        'rich_poor_5lag': 'bootstrap_richpoor_5lag.csv'
    }
    
    for model_type, output_file in models.items():
        print(f"Running {model_type} bootstrap...")
        results = bootstrap_model(data, model_type)
        results.to_csv(bootstrap_dir / output_file, index=False)
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main() 