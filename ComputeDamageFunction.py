
# Python equivalent of ComputeDamageFunction.R

import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

# Constants
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "output" / "projectionOutput"
BOOTSTRAP_DIR = DATA_DIR / "output" / "bootstrap"
CCPROJ_FILE = DATA_DIR / "input" / "CCprojections" / "CountryTempChange_RCP85.csv"
KOPP_FILE = DATA_DIR / "input" / "IAMdata" / "ProcessedKoppData.csv"

# Load temperature change and conversion factor data
Tchg = pd.read_csv(CCPROJ_FILE)
Tconv = Tchg["Tconv"].values

# Scenario setup
scens = ["base"] + [f"SSP{i}" for i in range(1, 6)]
sn = [1, 4, 6]
yrs = np.arange(2010, 2100)
inc = sorted(pd.read_csv(KOPP_FILE).query("1 < T <= 6")["T"].unique())
incs = [0.8] + inc

# Load population and growth projections
with open(OUTPUT_DIR / "popProjections.Rdata", "rb") as f:
    popProjections = pickle.load(f)
with open(OUTPUT_DIR / "growthProjections.Rdata", "rb") as f:
    growthProjections = pickle.load(f)

def weighted_mean(values, weights):
    return np.average(values, weights=weights)

def clip_temp(temp):
    return np.minimum(temp, 30)

def run_projection(prj_file, pooled=True, richpoor=False, lag=False, all_bootstraps=False):
    prj = pd.read_csv(prj_file)
    np_boot = len(prj)

    for dt in range(len(incs)):
        dtm = (incs[dt] - 0.8) * Tconv
        ccd = dtm / len(yrs)

        scen_range = [6] if all_bootstraps else sn

        for scen in scen_range:
            growthproj = growthProjections[scen]
            popproj = popProjections[scen]

            basegdp = popproj["gdpCap"].values
            temp = popproj["meantemp"].values
            wt_final = popproj[str(yrs[-1])].values

            if all_bootstraps:
                GDPcapCC = np.zeros((len(basegdp), len(yrs), np_boot))
                GDPcapNoCC = np.zeros((len(basegdp), len(yrs), np_boot))
                GDPcapCC[:, 0, :] = GDPcapNoCC[:, 0, :] = basegdp[:, None]
                tots = np.zeros((np_boot, len(yrs), 2))

                for tt in range(np_boot):
                    bg = prj.loc[tt, "temp"] * temp + prj.loc[tt, "temp2"] * temp ** 2
                    for i in range(1, len(yrs)):
                        y = str(yrs[i])
                        basegrowth = growthproj[y].values
                        newtemp = temp + i * ccd
                        dg = prj.loc[tt, "temp"] * newtemp + prj.loc[tt, "temp2"] * newtemp ** 2
                        dg = np.where(newtemp > 30, prj.loc[tt, "temp"] * 30 + prj.loc[tt, "temp2"] * 900, dg)
                        diff = dg - bg
                        GDPcapNoCC[:, i, tt] = GDPcapNoCC[:, i - 1, tt] * (1 + basegrowth)
                        GDPcapCC[:, i, tt] = GDPcapCC[:, i - 1, tt] * (1 + basegrowth + diff)
                        tots[tt, i, 0] = round(weighted_mean(GDPcapCC[:, i, tt], popproj[y].values), 0)
                        tots[tt, i, 1] = round(weighted_mean(GDPcapNoCC[:, i, tt], popproj[y].values), 0)

                filename = OUTPUT_DIR / f"DamageFunction_pooled_SSP5_{incs[dt]}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(tots, f)

            else:
                GDPcapCC = np.zeros((len(basegdp), len(yrs)))
                GDPcapNoCC = np.zeros((len(basegdp), len(yrs)))
                GDPcapCC[:, 0] = GDPcapNoCC[:, 0] = basegdp

                tt = 0
                if pooled:
                    bg = prj.loc[tt, "temp"] * temp + prj.loc[tt, "temp2"] * temp ** 2 if not lag else                          prj.loc[tt, "tlin"] * temp + prj.loc[tt, "tsq"] * temp ** 2
                elif richpoor:
                    medgdp = np.median(basegdp)
                    poor = basegdp <= medgdp
                    bg = np.where(poor,
                                 prj.loc[tt, "temppoor"] * temp + prj.loc[tt, "temp2poor"] * temp ** 2,
                                 prj.loc[tt, "temp"] * temp + prj.loc[tt, "temp2"] * temp ** 2) if not lag else                          np.where(poor,
                                 prj.loc[tt, "tlinpoor"] * temp + prj.loc[tt, "tsqpoor"] * temp ** 2,
                                 prj.loc[tt, "tlin"] * temp + prj.loc[tt, "tsq"] * temp ** 2)

                for i in range(1, len(yrs)):
                    y = str(yrs[i])
                    basegrowth = growthproj[y].values
                    newtemp = temp + i * ccd
                    if pooled:
                        dg = prj.loc[tt, "temp"] * newtemp + prj.loc[tt, "temp2"] * newtemp ** 2 if not lag else                              prj.loc[tt, "tlin"] * newtemp + prj.loc[tt, "tsq"] * newtemp ** 2
                        dg = np.where(newtemp > 30,
                                      prj.loc[tt, "temp"] * 30 + prj.loc[tt, "temp2"] * 900 if not lag else                                       prj.loc[tt, "tlin"] * 30 + prj.loc[tt, "tsq"] * 900,
                                      dg)
                    elif richpoor:
                        poor = GDPcapCC[:, i - 1] <= medgdp
                        dg = np.where(poor,
                                     prj.loc[tt, "temppoor"] * newtemp + prj.loc[tt, "temp2poor"] * newtemp ** 2,
                                     prj.loc[tt, "temp"] * newtemp + prj.loc[tt, "temp2"] * newtemp ** 2) if not lag else                              np.where(poor,
                                     prj.loc[tt, "tlinpoor"] * newtemp + prj.loc[tt, "tsqpoor"] * newtemp ** 2,
                                     prj.loc[tt, "tlin"] * newtemp + prj.loc[tt, "tsq"] * newtemp ** 2)
                        dg = np.where(newtemp > 30,
                                     prj.loc[tt, "temppoor"] * 30 + prj.loc[tt, "temp2poor"] * 900 if not lag else                                      prj.loc[tt, "tlinpoor"] * 30 + prj.loc[tt, "tsqpoor"] * 900,
                                     dg)

                    diff = dg - bg
                    GDPcapNoCC[:, i] = GDPcapNoCC[:, i - 1] * (1 + basegrowth)
                    GDPcapCC[:, i] = GDPcapCC[:, i - 1] * (1 + basegrowth + diff)

                tots = np.zeros((len(incs), len(sn), 2))
                tots[dt, sn.index(scen), 0] = round(weighted_mean(GDPcapCC[:, -1], wt_final), 0)
                tots[dt, sn.index(scen), 1] = round(weighted_mean(GDPcapNoCC[:, -1], wt_final), 0)

                tag = "pooled" if pooled else "richpoor"
                tag += "5lag" if lag else ""
                with open(OUTPUT_DIR / f"DamageFunction_{tag}.pkl", "wb") as f:
                    pickle.dump(tots, f)


# Run all models
run_projection(BOOTSTRAP_DIR / "bootstrap_noLag.csv", pooled=True, lag=False)
run_projection(BOOTSTRAP_DIR / "bootstrap_richpoor.csv", pooled=False, richpoor=True)
run_projection(BOOTSTRAP_DIR / "bootstrap_5lag.csv", pooled=True, lag=True)
run_projection(BOOTSTRAP_DIR / "bootstrap_richpoor_5lag.csv", pooled=False, richpoor=True, lag=True)
run_projection(BOOTSTRAP_DIR / "bootstrap_noLag.csv", pooled=True, all_bootstraps=True)
