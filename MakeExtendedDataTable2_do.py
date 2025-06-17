# Translation of MakeExtendedDataTable2.do
# Regression results by GDP percentile bins

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Load data
df = pd.read_csv("data/output/mainDataset.csv")

# Create GDP percentile bins
df['gdp_bin'] = pd.qcut(df['GDP_WDIppp'], q=10, labels=False)

# Run regression within each bin
results = []

for b in sorted(df['gdp_bin'].dropna().unique()):
    sub = df[df['gdp_bin'] == b]
    if len(sub) < 10:
        continue  # skip small samples

    model = smf.ols("growthWDI ~ UDel_temp_popweight + C(iso) + C(year)", data=sub).fit()
    coef = model.params["UDel_temp_popweight"]
    se = model.bse["UDel_temp_popweight"]
    results.append({"gdp_bin": b, "coef": coef, "se": se})

res_df = pd.DataFrame(results)
res_df.to_csv("tables/ExtendedDataTable2.csv", index=False)
print(res_df)