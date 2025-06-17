# Translation of MakeExtendedDataFigure3.do
# Plotting growth-temperature relationship for individual countries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/output/mainDataset.csv")

# Select countries to highlight (e.g., US, IND, NGA)
highlight = ['USA', 'IND', 'NGA']

# Plot setup
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

for iso in highlight:
    country_df = df[df['iso'] == iso].copy()
    plt.scatter(country_df['UDel_temp_popweight'], country_df['growthWDI'], label=iso, alpha=0.6)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Temperature (°C)")
plt.ylabel("GDP Growth")
plt.title("Extended Data Figure 3: Country-Level Growth vs. Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("figures/ExtendedDataFigure3.png")
plt.show()
