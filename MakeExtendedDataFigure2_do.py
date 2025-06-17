# Translation of MakeExtendedDataFigure2.do
# Histogram of population-weighted temperature exposure

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("data/output/mainDataset.csv")

# Use estimation sample (exclude NA)
mask = (~df['UDel_temp_popweight'].isna()) & (~df['Pop'].isna())
temps = df.loc[mask, 'UDel_temp_popweight']
pops = df.loc[mask, 'Pop']

# Bin and weight histogram
bins = np.arange(-10, 35.5, 0.5)
hist_values, bin_edges = np.histogram(temps, bins=bins, weights=pops)

# Normalize to total population
hist_values = hist_values / hist_values.sum()

# Plot
plt.figure(figsize=(8, 6))
plt.bar(bin_edges[:-1], hist_values, width=0.5, color="steelblue", edgecolor="black")
plt.xlabel("Population-weighted Temp (°C)")
plt.ylabel("Fraction of Global Population")
plt.title("Extended Data Figure 2: Global Population by Temperature")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/ExtendedDataFigure2.png")
plt.show()