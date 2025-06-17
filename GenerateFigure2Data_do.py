import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Panel A ===
resp = pd.read_csv("data/output/estimatedGlobalResponse.csv")
dta = pd.read_csv("data/output/mainDataset.csv")
coef = pd.read_csv("data/output/estimatedCoefficients.csv")

smpl = (~dta['growthWDI'].isna()) & (~dta['UDel_temp_popweight'].isna())

x = resp['x']
est = resp['estimate'] - resp['estimate'].max()
min90 = resp['min90'] - resp['estimate'].max()
max90 = resp['max90'] - resp['estimate'].max()

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

axs[0].plot(x, est, lw=2)
axs[0].fill_between(x, min90, max90, color='lightblue')

# Add vertical lines for country temps
countries = ['USA','CHN','DEU','JPN','IND','NGA','IDN','BRA','FRA','GBR']
for cty in countries:
    tt = dta.loc[dta['iso'] == cty, 'UDel_temp_popweight'].mean(skipna=True)
    axs[0].axvline(tt, ymin=0.3, ymax=1.0, lw=0.5)

# Histograms
bins = np.arange(-7, 30.5, 0.5)
histtemp = dta.loc[smpl, 'UDel_temp_popweight']
histpop = dta.loc[smpl, 'Pop']
histgdp = dta.loc[smpl, 'TotGDP']

pop = np.histogram(histtemp, bins=bins, weights=histpop)[0]
gdp = np.histogram(histtemp, bins=bins, weights=histgdp)[0]
pop = pop / pop.sum()
gdp = gdp / gdp.sum()

base = -0.3
dis = 0.055
bin_centers = bins[:-1]

axs[0].bar(bin_centers, np.histogram(histtemp, bins=bins)[0] / max(np.histogram(histtemp, bins=bins)[0]) * 0.05, width=0.5, bottom=base, color='red')
axs[0].bar(bin_centers, pop / max(pop) * 0.05, width=0.5, bottom=base - dis * 1, color='gray')
axs[0].bar(bin_centers, gdp / max(gdp) * 0.05, width=0.5, bottom=base - dis * 2, color='black')
axs[0].set_xlim(-2, 30)
axs[0].set_ylim(-0.4, 0.1)
axs[0].set_title("Panel A")

# === Panel B: Heterogeneity poor vs rich ===
resp = pd.read_csv("data/output/EffectHeterogeneity.csv")
resp = resp[resp['x'] >= 5]

m = "growthWDI"
smp_poor = (resp['model'] == m) & (resp['interact'] == 1)
smp_rich = (resp['model'] == m) & (resp['interact'] == 0)

xx_poor = resp.loc[smp_poor, 'x']
est_poor = resp.loc[smp_poor, 'estimate'] - resp.loc[smp_poor, 'estimate'].max()
min90_poor = resp.loc[smp_poor, 'min90'] - resp.loc[smp_poor, 'estimate'].max()
max90_poor = resp.loc[smp_poor, 'max90'] - resp.loc[smp_poor, 'estimate'].max()

xx_rich = resp.loc[smp_rich, 'x']
est_rich = resp.loc[smp_rich, 'estimate'] - resp.loc[smp_rich, 'estimate'].max()

axs[1].plot(xx_poor, est_poor, lw=2, color='steelblue')
axs[1].fill_between(xx_poor, min90_poor, max90_poor, color='lightblue')
axs[1].plot(xx_rich, est_rich, lw=2, color='red')
axs[1].set_title("Panel B")
axs[1].set_xlim(5, 30)
axs[1].set_ylim(-0.35, 0.1)

# Panel C: Early vs Late
resp = pd.read_csv("data/output/EffectHeterogeneityOverTime.csv")
smp_early = (resp['interact'] == 1)
smp_late = (resp['interact'] == 0)

xx_early = resp.loc[smp_early, 'x']
est_early = resp.loc[smp_early, 'estimate'] - resp.loc[smp_early, 'estimate'].max()
min90_early = resp.loc[smp_early, 'min90'] - resp.loc[smp_early, 'estimate'].max()
max90_early = resp.loc[smp_early, 'max90'] - resp.loc[smp_early, 'estimate'].max()

xx_late = resp.loc[smp_late, 'x']
est_late = resp.loc[smp_late, 'estimate'] - resp.loc[smp_late, 'estimate'].max()

axs[2].plot(xx_early, est_early, lw=2, color='steelblue')
axs[2].fill_between(xx_early, min90_early, max90_early, color='lightblue')
axs[2].plot(xx_late, est_late, lw=2, color='red')
axs[2].set_title("Panel C")
axs[2].set_xlim(5, 30)
axs[2].set_ylim(-0.35, 0.1)

# Panel D & E: Agriculture vs Non-agriculture
resp = pd.read_csv("data/output/EffectHeterogeneity.csv")
toplot = ["AgrGDPgrowthCap", "NonAgrGDPgrowthCap"]

for idx, m in enumerate(toplot):
    smp_poor = (resp['model'] == m) & (resp['interact'] == 1)
    smp_rich = (resp['model'] == m) & (resp['interact'] == 0)
    
    xx_poor = resp.loc[smp_poor, 'x']
    est_poor = resp.loc[smp_poor, 'estimate'] - resp.loc[smp_poor, 'estimate'].max()
    min90_poor = resp.loc[smp_poor, 'min90'] - resp.loc[smp_poor, 'estimate'].max()
    max90_poor = resp.loc[smp_poor, 'max90'] - resp.loc[smp_poor, 'estimate'].max()

    xx_rich = resp.loc[smp_rich, 'x']
    est_rich = resp.loc[smp_rich, 'estimate'] - resp.loc[smp_rich, 'estimate'].max()

    axs[3+idx].plot(xx_poor, est_poor, lw=2, color='steelblue')
    axs[3+idx].fill_between(xx_poor, min90_poor, max90_poor, color='lightblue')
    axs[3+idx].plot(xx_rich, est_rich, lw=2, color='red')
    axs[3+idx].set_title(f"Panel {'D' if idx == 0 else 'E'}")
    axs[3+idx].set_xlim(5, 30)
    axs[3+idx].set_ylim(-0.35, 0.1)

plt.tight_layout()
plt.savefig("Figure2.png")
plt.show()
