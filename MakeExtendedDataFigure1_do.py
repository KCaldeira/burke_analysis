# Translation of MakeExtendedDataFigure1.do
# Note: Adjust paths and variable names based on actual dataset structure

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("data/output/mainDataset.csv")

# Subset for countries with enough observations
country_counts = df['iso'].value_counts()
countries_to_plot = country_counts[country_counts >= 10].index  # adjust cutoff if needed

# Calculate average temperature and growth per country
avg_temp = df.groupby('iso')['UDel_temp_popweight'].mean()
avg_growth = df.groupby('iso')['growthWDI'].mean()

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(avg_temp.loc[countries_to_plot], avg_growth.loc[countries_to_plot], alpha=0.7)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel("Average Annual Temperature (°C)")
ax.set_ylabel("Average GDP Growth")
ax.set_title("Extended Data Figure 1: Temp vs. Growth by Country")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/ExtendedDataFigure1.png")
plt.show()