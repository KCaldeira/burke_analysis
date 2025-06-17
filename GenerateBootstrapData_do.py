import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm

# Load dataset
df = pd.read_csv("data/output/mainDataset.csv")

# Preprocess: temperature bins, fixed effects
df['temp_bin'] = df['UDel_temp_popweight'].round()
df['dlog_gdp'] = df.groupby('iso')['growthWDI'].transform(lambda x: x)  # assuming growthWDI is ∆logGDP

# Drop missing values
df = df.dropna(subset=['dlog_gdp', 'temp_bin'])

# Create bin dummies
temp_bins = pd.get_dummies(df['temp_bin'], prefix='bin')
ref_bin = 'bin_13.0'
if ref_bin in temp_bins.columns:
    temp_bins = temp_bins.drop(columns=[ref_bin])

df = df.join(temp_bins)
bin_vars = temp_bins.columns.tolist()

# Model formula
rhs = ' + '.join(bin_vars)
formula = f'dlog_gdp ~ {rhs} + C(iso) + C(year)'

# Bootstrap parameters
B = 1000
bootstrap_coefs = []

# Unique countries for resampling
countries = df['iso'].unique()

for _ in tqdm(range(B)):
    # Resample countries with replacement
    sampled_countries = np.random.choice(countries, size=len(countries), replace=True)
    sampled_df = pd.concat([df[df['iso'] == c] for c in sampled_countries])
    
    # Fit model
    try:
        model = smf.ols(formula=formula, data=sampled_df).fit()
        coefs = model.params[bin_vars]
        bootstrap_coefs.append(coefs.values)
    except Exception:
        continue  # Skip failed fits

# Convert results to DataFrame
bootstrap_df = pd.DataFrame(bootstrap_coefs, columns=bin_vars)
bootstrap_df.to_csv("data/output/bootstrapTemperatureEffects.csv", index=False)
