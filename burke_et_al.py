import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrix
import matplotlib.pyplot as plt

# === 1. Load Data ===
# Replace filenames with actual paths to the data files from Burke et al. (2015) replication archive
gdp = pd.read_csv('gdp_panel.csv')          # Contains: country, year, gdp_per_capita
temp = pd.read_csv('temperature_panel.csv') # Contains: country, year, avg_temp
cov = pd.read_csv('covariates.csv')         # Contains: country, year, population, etc.

# Merge datasets on country and year
df = gdp.merge(temp, on=['country', 'year']).merge(cov, on=['country', 'year'])
df.dropna(inplace=True)

# === 2. Spline Transformation ===
# Create linear spline with a knot at 13°C
df['t_low'] = np.minimum(df['avg_temp'], 13)
df['t_high13'] = np.maximum(df['avg_temp'] - 13, 0)

# === 3. Compute GDP Growth ===
df['dlog_gdp'] = df.groupby('country')['gdp_per_capita'].transform(lambda x: np.log(x).diff())
df = df.dropna(subset=['dlog_gdp'])

# === 4. Panel Regression ===
# Modify 'some_other_covariates' to match actual column names from the dataset
formula = 'dlog_gdp ~ t_low + t_high13 + population + C(country) + C(year)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['country']})
print(model.summary())

# === 5. Plot Estimated Temperature–Growth Profile ===
temps = np.linspace(df['avg_temp'].min(), df['avg_temp'].max(), 100)
t_low = np.minimum(temps, 13)
t_high = np.maximum(temps - 13, 0)

effect = model.params['t_low'] * t_low + model.params['t_high13'] * t_high

plt.figure(figsize=(8, 6))
plt.axvline(13, color='grey', linestyle='--', label='Knot at 13°C')
plt.plot(temps, effect, label='Estimated Effect')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Estimated Effect on Δ log GDP')
plt.title('Estimated Non-linear Effect of Temperature on GDP Growth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()