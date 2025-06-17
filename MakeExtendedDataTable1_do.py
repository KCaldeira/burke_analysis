# Translation of MakeExtendedDataTable1.do
# Summary statistics by GDP percentile bins

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/output/mainDataset.csv")

# Create GDP percentile bins
df['gdp_bin'] = pd.qcut(df['GDP_WDIppp'], q=10, labels=False)

# Summary stats by bin
summary = df.groupby('gdp_bin').agg({
    'UDel_temp_popweight': ['mean', 'std'],
    'growthWDI': ['mean', 'std'],
    'GDP_WDIppp': ['mean', 'count']
})

summary.columns = ['_'.join(col) for col in summary.columns]
summary.reset_index(inplace=True)

# Save to CSV
summary.to_csv("tables/ExtendedDataTable1.csv", index=False)
print(summary)
