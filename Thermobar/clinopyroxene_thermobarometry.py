import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from pickle import load
import pickle
from pathlib import Path
Thermobar_dir=Path(__file__).parent

from Thermobar.core import *


## Equations for Cpx-Liquid Barometry written as functions

def P_Put1996_eqP1(T, *, lnK_Jd_liq, Na2O_Liq_cat_frac, Al2O3_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Putirka (1996) EqP1

    | SEE=+-4.6 kbar (anhydrous)
    | SEE=+-5.4 kbar (hydrous)

    '''
    return - 54.3 + 299 * T / 10 ** 4 + 36.4 * T * lnK_Jd_liq / \
        10 ** 4 + 367 * Na2O_Liq_cat_frac * Al2O3_Liq_cat_frac


def P_Mas2013_eqPalk1(T, *, lnK_Jd_liq,
                      Na2O_Liq_cat_frac, Al2O3_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid barometer of Putirka (1996) EqP1 by Masotta et al. (2013) for alkaline melts
    | SEE=+-1.71 kbar

    '''
    return - 8.83763538032322 + 79.0497715060127 * T / 10 ** 4 + 11.6474409456619 * \
        T * lnK_Jd_liq / 10 ** 4 + 8.63312603541135 * \
        Na2O_Liq_cat_frac * Al2O3_Liq_cat_frac


def P_Put1996_eqP2(T, *, lnK_Jd_liq, Na2O_Liq_cat_frac, Al2O3_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Putirka (1996) EqP2


    '''
    return (-50.7 + 394 * T / 10 ** 4 + 36.4 * T * lnK_Jd_liq / 10 ** 4 -
            20 * (T / 10**4) * np.log(1 / (Na2O_Liq_cat_frac.astype(float) * Al2O3_Liq_cat_frac.astype(float))))


def P_Mas2013_eqPalk2(T, *, lnK_Jd_liq,
                      Na2O_Liq_cat_frac, Al2O3_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid barometer of Putirka (1996)
    EqP2 by Masotta et al. (2013) for alkaline melts
    | SEE=+-1.70 kbar

    '''
    return (-6.28332277837751 + 38.1796219610587 * T / 10 ** 4
    + 9.42105316105869 * T * lnK_Jd_liq /
    10 ** 4 + 6.15641875019196 * (T / 10**4) * np.log(1 / (Na2O_Liq_cat_frac.astype(float)
    * Al2O3_Liq_cat_frac.astype(float))))


def P_Put2003(T, *, lnK_Jd_liq, CaO_Liq_cat_frac,
              SiO2_Liq_cat_frac, Mg_Number_Liq_NoFe3):
    '''
    Clinopyroxene-liquid barometer of Putirka (2003) Eq1

    | SEE=+-4.8 kbar (anhydrous)
    | SEE=+-5.0 kbar (hydrous)

    '''
    return (- 88.3 + 0.00282 * T * lnK_Jd_liq + 0.0219 * T
    - 25.1 * np.log(CaO_Liq_cat_frac.astype(float) *
    SiO2_Liq_cat_frac.astype(float))
    + 12.4 * np.log(CaO_Liq_cat_frac.astype(float)) + 7.03 * Mg_Number_Liq_NoFe3)


def P_Put2008_eq30(T, *, lnK_Jd_liq, FeOt_Liq_cat_frac, MgO_Liq_cat_frac,
                   DiHd_2003, Mg_Number_Liq_NoFe3, Na2O_Liq_cat_frac, K2O_Liq_cat_frac, H2O_Liq):
    '''
    Clinopyroxene-liquid barometer of Putirka (2008) Eq30

    | SEE=+-3.6 kbar (all data)
    | SEE=+-1.6 kbar (calibration data)

    '''
    return (-48.7 + 271.3 * (T / 10**4) + 31.96 * (T / 10**4) * lnK_Jd_liq - 8.2 * np.log(FeOt_Liq_cat_frac.astype(float))
            + 4.6 * np.log(MgO_Liq_cat_frac.astype(float)) - 0.96 * np.log(K2O_Liq_cat_frac.astype(float))
            - 2.2 * np.log(DiHd_2003.astype(float)) - 31 * Mg_Number_Liq_NoFe3 + 56.2 * (Na2O_Liq_cat_frac + K2O_Liq_cat_frac) + 0.76 * H2O_Liq)


def P_Put2008_eq31(T, *, lnK_Jd_liq, CaO_Liq_cat_frac, Na2O_Liq_cat_frac, K2O_Liq_cat_frac, SiO2_Liq_cat_frac, MgO_Liq_cat_frac,
                   FeOt_Liq_cat_frac, DiHd_2003, EnFs, Al2O3_Cpx_cat_6ox, H2O_Liq):
    '''
    Clinopyroxene-liquid barometer of Putirka (2008) Eq31

    | SEE=+-2.9 kbar (all data)
    '''
    return (-40.73 + 358 * (T / 10**4) + 21.7 * (T / 10**4) * lnK_Jd_liq - 106 * CaO_Liq_cat_frac - 166 * (Na2O_Liq_cat_frac + K2O_Liq_cat_frac)**2
            - 50.2 * SiO2_Liq_cat_frac *
            (MgO_Liq_cat_frac + FeOt_Liq_cat_frac) -
            3.2 * np.log(DiHd_2003.astype(float)) - 2.2 * np.log(EnFs.astype(float))
            + 0.86 * np.log(Al2O3_Cpx_cat_6ox.astype(float)) + 0.4 * H2O_Liq)


def P_Put2008_eq32c(T, *, FeOt_Liq_cat_frac, CaTs, H2O_Liq, CaO_Liq_cat_frac,
                    SiO2_Liq_cat_frac, Al2O3_Cpx_cat_6ox, Al2O3_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Putirka (2008) Eq32c based on partitioning of Al between cpx and liquid

    | SEE=+-5 kbar (all data)
    | SEE=+-1.5 kbar (calibration data)
    '''
    return (-57.9 + 0.0475 * (T) - 40.6 * FeOt_Liq_cat_frac - 47.7 * CaTs + 0.67 * H2O_Liq -
            153 * CaO_Liq_cat_frac * SiO2_Liq_cat_frac + 6.89 * (Al2O3_Cpx_cat_6ox / Al2O3_Liq_cat_frac))


def P_Mas2013_eqalk32c(T, *, FeOt_Liq_cat_frac, CaTs, H2O_Liq, CaO_Liq_cat_frac,
                       SiO2_Liq_cat_frac, Al2O3_Cpx_cat_6ox, Al2O3_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid barometer of Putirka (2008) Eq32c by Masotta et al. (2013) for alkaline melts
    | SEE=+-1.67 kbar
    '''
    return (-16.3446543551989 + 0.0141435837038975 * (T)
    - 12.3909508275802 * FeOt_Liq_cat_frac - 9.19220692402416 * CaTs
    + 0.214041799294945 * H2O_Liq + 38.734045560859 * CaO_Liq_cat_frac * SiO2_Liq_cat_frac
    + 1.5944198112849 * (Al2O3_Cpx_cat_6ox / Al2O3_Liq_cat_frac))


def P_Mas2013_Palk2012(T=None, *, lnK_Jd_liq, H2O_Liq,
                       Na2O_Liq_cat_frac, K2O_Liq_cat_frac, Kd_Fe_Mg_Fet):
    '''
    Clinopyroxene-liquid barometer of Masotta et al. (2013) for alkaline melts
    | SEE=+-1.15 kbar
    '''
    return (-3.88903951262765 + 0.277651046511846 * np.exp(lnK_Jd_liq)
    + 0.0740292491471828 * H2O_Liq + 5.00912129248619 * (Na2O_Liq_cat_frac)
    / (Na2O_Liq_cat_frac + K2O_Liq_cat_frac) + 6.39451438456963 * Kd_Fe_Mg_Fet)


def P_Wieser2021_H2O_indep(T=None, *, MgO_Liq, CaO_Liq_cat_frac, lnK_Jd_liq, Jd,
                           CaTs, Na2O_Liq_cat_frac, FeOt_Liq_cat_frac, Al2O3_Cpx_cat_6ox, Mg_Number_Liq_NoFe3):
    return (3.204423282096874 + 1.21811674 * MgO_Liq - 168.80037558 * CaO_Liq_cat_frac
    + 1.49243994 * lnK_Jd_liq + 58.22419473 * Jd + 76.11682662 * CaTs
    - 29.27503912 * Na2O_Liq_cat_frac + 33.34059394 * FeOt_Liq_cat_frac
    - 8.50428995 * Al2O3_Cpx_cat_6ox + 4.98260164 * Mg_Number_Liq_NoFe3)


def P_Neave2017(T, *, lnK_Jd_liq, DiHd_2003, Al2O3_Liq_cat_frac,
                Na2O_Liq_cat_frac, K2O_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Neave and Putirka (2017) Putirka (2008)  based on Jd in pyroxene

    | SEE=+-1.4 kbar
    '''
    return (-26.2712 + 39.16138 * T * lnK_Jd_liq / 10**4 - 4.21676 * np.log(DiHd_2003.astype(float))
            + 78.43463 * Al2O3_Liq_cat_frac + 393.8126 * (Na2O_Liq_cat_frac + K2O_Liq_cat_frac)**2)

def P_Petrelli2021_Cpx_Liq(T=None, *, cpx_comps, liq_comps):
    '''
    Clinopyroxene-liquid  barometer of Petrelli et al. (2021) based on
    Machine Learning.
    |  SEE==+-2.9 kbar
    '''
    cpx_test=cpx_comps.copy()
    liq_test=liq_comps.copy()
    cpx_test_noID_noT=cpx_test.drop(['Sample_ID_Cpx'], axis=1)
    liq_test_noID_noT=liq_test.drop(['Sample_ID_Liq', 'Fe3Fet_Liq', 'NiO_Liq', 'CoO_Liq', 'CO2_Liq'], axis=1)
    cpx_liq_combo_test=pd.concat([cpx_test_noID_noT, liq_test_noID_noT], axis=1)
    x_test=cpx_liq_combo_test.values


    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Liq.pkl', 'rb') as f:
        scaler_P2020_Cpx_Liq=load(f)

    with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Liq.pkl', 'rb') as f:
        ETR_Press_P2020_Cpx_Liq=load(f)


    x_test_scaled=scaler_P2020_Cpx_Liq.transform(x_test)
    Pred_P_kbar=ETR_Press_P2020_Cpx_Liq.predict(x_test_scaled)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_Liq)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)


    return df_stats

## Equations for Cpx-Liquid Thermometry written as functions


def T_Put1996_eqT1(P=None, *, lnK_Jd_DiHd_liq_1996,
                   Mg_Number_Liq_NoFe3, CaO_Liq_cat_frac):
    '''
    Clinopyroxene-liquid thermometer of Putirka (1996) EqT1 (pressure-independent)

    '''
    return (10 ** 4 / (6.73 - 0.26 * lnK_Jd_DiHd_liq_1996 - 0.86 * np.log(Mg_Number_Liq_NoFe3.astype(float))
                       + 0.52 * np.log(CaO_Liq_cat_frac.astype(float))))


def T_Mas2013_eqTalk1(P=None, *, lnK_Jd_DiHd_liq_1996,
                      Mg_Number_Liq_NoFe3, CaO_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid thermometer of Putirka (1996) EqT1 by Masotta et al. (2013) for alkaline melts
    |  SEE=+-31.6°C
    '''
    return (10 ** 4 / (6.7423126317975 - 0.023236627691972 * lnK_Jd_DiHd_liq_1996 -
            0.68839419999351 * np.log(Mg_Number_Liq_NoFe3.astype(float)) - 0.153193056441978 * np.log(CaO_Liq_cat_frac.astype(float))))


def T_Put1996_eqT2(P, *, lnK_Jd_DiHd_liq_1996,
                   Mg_Number_Liq_NoFe3, CaO_Liq_cat_frac):
    '''
    Clinopyroxene-liquid thermometer of Putirka (1996) EqT2 (pressure-dependent)

    '''
    return (10 ** 4 / (6.59 - 0.16 * lnK_Jd_DiHd_liq_1996 - 0.65 * np.log(Mg_Number_Liq_NoFe3.astype(float))
                       + 0.23 * np.log(CaO_Liq_cat_frac.astype(float)) - 0.02 * P))


def T_Mas2013_eqTalk2(P, *, lnK_Jd_DiHd_liq_1996,
                      Mg_Number_Liq_NoFe3, CaO_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid thermometer of Putirka (1996) EqT2 by Masotta et al. (2013) for alkaline melts
    |  SEE=+-31.2°C
    '''
    return (10 ** 4 / (6.52396326315485 - 0.0396542787609402 * lnK_Jd_DiHd_liq_1996 - 0.680638985726502 *
            np.log(Mg_Number_Liq_NoFe3.astype(float)) - 0.145757123805013 * np.log(CaO_Liq_cat_frac.astype(float)) + 0.0790582631912926 * P))


def T_Put1999(P, *, MgO_Liq_cat_frac, FeOt_Liq_cat_frac,
              CaO_Liq_cat_frac, SiO2_Liq_cat_frac, Al2O3_Liq_cat_frac):
    '''
    Equation in Keith's Cpx-Liquid spreadsheet labelled "Putirka 1999".

    '''

    return (10 ** 4 / (3.12 - 0.0259 * P - 0.37 * np.log(MgO_Liq_cat_frac.astype(float) / (MgO_Liq_cat_frac.astype(float) + FeOt_Liq_cat_frac.astype(float)))
                       + 0.47 * np.log(CaO_Liq_cat_frac.astype(float) * (MgO_Liq_cat_frac.astype(float) +
                        FeOt_Liq_cat_frac.astype(float)) * (SiO2_Liq_cat_frac.astype(float))**2)
                       - 0.78 * np.log((MgO_Liq_cat_frac.astype(float) + FeOt_Liq_cat_frac.astype(float))
                                       ** 2 * (SiO2_Liq_cat_frac.astype(float))**2)
                       - 0.34 * np.log(CaO_Liq_cat_frac.astype(float) * (Al2O3_Liq_cat_frac.astype(float))**2 * SiO2_Liq_cat_frac.astype(float))))


def T_Put2003(P, *, lnK_Jd_DiHd_liq_2003, Mg_Number_Liq_NoFe3,
              Na2O_Liq_cat_frac, SiO2_Liq_cat_frac, Jd):
    '''
    Clinopyroxene-liquid thermometer of Putirka (2003)

    '''
    return (10 ** 4 / (4.6 - 0.437 * lnK_Jd_DiHd_liq_2003 - 0.654 * np.log(Mg_Number_Liq_NoFe3.astype(float))
    - 0.326 * np.log(Na2O_Liq_cat_frac.astype(float)) -0.92 * np.log(SiO2_Liq_cat_frac.astype(float))
    + 0.274 * np.log(Jd.astype(float)) - 0.00632 * P))


def T_Put2008_eq33(P, *, H2O_Liq, Mg_Number_Liq_NoFe3, CaO_Liq_cat_frac, SiO2_Liq_cat_frac,
                   TiO2_Liq_cat_frac, Na2O_Liq_cat_frac, K2O_Liq_cat_frac, EnFs, lnK_Jd_DiHd_liq_2003):
    '''
    Clinopyroxene-liquid  thermometer of Putirka (2008) Eq 33.

    |  SEE=+-°C
    '''
    return (10 ** 4 / (7.53 + 0.07 * H2O_Liq - 1.1 * Mg_Number_Liq_NoFe3
    - 14.9 * (CaO_Liq_cat_frac * SiO2_Liq_cat_frac) -
    0.08 * np.log(TiO2_Liq_cat_frac.astype(float))
    - 3.62 * (Na2O_Liq_cat_frac + K2O_Liq_cat_frac) - 0.18 * np.log(EnFs.astype(float))
    - 0.14 * lnK_Jd_DiHd_liq_2003 - 0.027 * P))


def T_Mas2013_eqalk33(P, *, H2O_Liq, Mg_Number_Liq_NoFe3, CaO_Liq_cat_frac, SiO2_Liq_cat_frac,
                      TiO2_Liq_cat_frac, Na2O_Liq_cat_frac, K2O_Liq_cat_frac, EnFs, lnK_Jd_DiHd_liq_2003):
    '''
    Recalibration of the clinopyroxene-liquid thermometer of Putirka (2008)
    Eq 33 by Masotta et al. (2013) for alkaline melts
    |  SEE=+-24°C
    '''
    return (10 ** 4 / (6.80728851520843 + 0.0500993963259582 * H2O_Liq
    - 1.91449550102791 * Mg_Number_Liq_NoFe3
    - 25.0429785936576 * (CaO_Liq_cat_frac * SiO2_Liq_cat_frac) -
    0.304200646919069 * np.log(TiO2_Liq_cat_frac.astype(float))
    + 2.25444204541222 * (Na2O_Liq_cat_frac + K2O_Liq_cat_frac)
    - 0.021072700182831 * np.log(EnFs.astype(float))
    + 0.00268252978603778 * lnK_Jd_DiHd_liq_2003
    + 0.0614725514133312 * P))


# ones without P in the function
def T_Mas2013_Talk2012(P=None, *, H2O_Liq, Kd_Fe_Mg_Fet, lnK_Jd_DiHd_liq_2003,
Mg_Number_Liq_NoFe3, DiHd_2003, Na2O_Liq_cat_frac, K2O_Liq_cat_frac,
TiO2_Liq_cat_frac, lnK_Jd_liq, CaO_Liq_cat_frac, SiO2_Liq_cat_frac):
    '''
    Clinopyroxene-liquid thermometer of Masotta et al. (2013) for alkaline melts
    | SEE=+-18.2C
    '''
    return (10**4 / (2.90815635794002 - 0.400827676578132 * lnK_Jd_DiHd_liq_2003
        + 0.0375720784518263 * H2O_Liq - 1.6383282971929 *
        (Mg_Number_Liq_NoFe3 / DiHd_2003) + 1.01129776262724 *
        ((Na2O_Liq_cat_frac) / (Na2O_Liq_cat_frac + K2O_Liq_cat_frac))
        - 0.21766733252629 * np.log(TiO2_Liq_cat_frac.astype(float)) + 0.466149612620683
        * lnK_Jd_liq + 1.61626798988239 * Kd_Fe_Mg_Fet + 23.3855047471225 * (CaO_Liq_cat_frac * SiO2_Liq_cat_frac)))


def T_Brug2019(P=None, *, CaTs, DiHd_2003, SiO2_Liq_cat_frac, TiO2_Liq_cat_frac,
FeOt_Liq_cat_frac, MgO_Liq_cat_frac, CaO_Liq_cat_frac, K2O_Liq_cat_frac):
    '''
    Clinopyroxene-liquid  thermometer of Brugmann and Till (2019) for evolved systems (Cpx Mg#>64, Al2O3 Cpx<7 wt%, SiO2_Liq>70 wt%)

    |  SEE==+-20°C
    '''
    return (273.15 + 300 * (-1.8946098 - 0.6010197 * CaTs - 0.1856423 * DiHd_2003
+ 4.71248858 * SiO2_Liq_cat_frac + 77.5861878 * TiO2_Liq_cat_frac +
10.8503727 * FeOt_Liq_cat_frac + 33.6303471 * MgO_Liq_cat_frac
+ 15.4532888 * CaO_Liq_cat_frac + 15.6390115 * K2O_Liq_cat_frac))



def T_Petrelli2021_Cpx_Liq(P=None, *, cpx_comps, liq_comps):
    '''
    Clinopyroxene-liquid  thermometer of Petrelli et al. (2021) based on
    Machine Learning.
    |  SEE==+-51°C
    '''
    cpx_test=cpx_comps.copy()
    liq_test=liq_comps.copy()
    cpx_test_noID_noT=cpx_test.drop(['Sample_ID_Cpx'], axis=1)
    liq_test_noID_noT=liq_test.drop(['Sample_ID_Liq', 'Fe3Fet_Liq', 'NiO_Liq', 'CoO_Liq', 'CO2_Liq'], axis=1)
    cpx_liq_combo_test=pd.concat([cpx_test_noID_noT, liq_test_noID_noT], axis=1)
    x_test=cpx_liq_combo_test.values


    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Liq.pkl', 'rb') as f:
        scaler_P2020_Cpx_Liq=load(f)

    with open(Thermobar_dir/'ETR_Temp_Petrelli2020_Cpx_Liq.pkl', 'rb') as f:
        ETR_Temp_P2020_Cpx_Liq=load(f)

    x_test_scaled=scaler_P2020_Cpx_Liq.transform(x_test)
    Pred_T_K=ETR_Temp_P2020_Cpx_Liq.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_P2020_Cpx_Liq)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats

## Function for calculatin clinopyroxene-liquid pressure
Cpx_Liq_P_funcs = {P_Put1996_eqP1, P_Mas2013_eqPalk1, P_Put1996_eqP2, P_Mas2013_eqPalk2,
P_Put2003, P_Put2008_eq30, P_Put2008_eq31, P_Put2008_eq32c, P_Mas2013_eqalk32c,
P_Mas2013_Palk2012, P_Wieser2021_H2O_indep, P_Neave2017, P_Petrelli2021_Cpx_Liq} # put on outside

Cpx_Liq_P_funcs_by_name = {p.__name__: p for p in Cpx_Liq_P_funcs}


def calculate_cpx_liq_press(*, equationP, cpx_comps=None, liq_comps=None, meltmatch=None,
                            T=None, eq_tests=False, Fe3Fet_Liq=None, H2O_Liq=None,
                           sigma=1, KdErr=0.03):
    '''
    Clinopyroxene-Liquid barometer, calculates pressure in kbar
    (and equilibrium tests as an option)


   Parameters
    -------

    cpx_comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    Or:

    meltmatch: DataFrame
        Combined dataframe of cpx-Liquid compositions
        Used for calculate_cpx_liq_press_temp_matching function.

    EquationP: str
        Cpx-Liquid
        |  P_Put1996_eqP1 (T-dep, H2O-indep)
        |  P_Mas2013_eqPalk1 (T-dep, H2O-indep, alk adaption of P1)
        |  P_Put1996_eqP2 (T-dep, H2O-indep)
        |  P_Mas2013_eqPalk2 (T-dep, H2O-indep, alk adaption of P2)
        |  P_Put2003 ((T-dep, H2O-indep)
        |  P_Neave2017 (T-dep, H2O-indep)
        |  P_Put2008_eq30 (T-dep, H2O-dep)
        |  P_Put2008_eq31 (T-dep, H2O-dep)
        |  P_Put2008_eq32c (T-dep, H2O-dep)
        |  P_Mas2013_eqalk32c (T-dep, H2O-dep, alk adaption of 32c)


    T: float, int, series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool
        If False, just returns pressure (default) as a panda series
        If True, returns pressure, Values of Eq tests,
        as well as user-entered cpx and liq comps and components.


    Returns
    -------
    If eq_tests=False
        pandas.series: Pressure in kbar
    If eq_tests=True
        panda.dataframe: Temperature in Kelvin +
        Eq Tests + cpx+liq comps + components

    '''
    # This checks if your equation is one of the accepted equations
    try:
        func = Cpx_Liq_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None
    sig=inspect.signature(func)


    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T, or specify T="Solve"')
    else:
        if T is not None:
            print('Youve selected a T-independent function')

    if isinstance(T, pd.Series):
        if liq_comps is not None:
            if len(T) != len(liq_comps):
                raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of liquid compositions')



# This replaces H2O and Fe3Fet_Liq in the input
    if liq_comps is not None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None and not isinstance(H2O_Liq, str):
            liq_comps_c['H2O_Liq'] = H2O_Liq
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq


    if equationP == "P_Mas2013_eqalk32c" or equationP == "P_Mas2013_eqPalk2" or equationP == "P_Mas2013_eqPalk1":
        if liq_comps is not None and np.max(liq_comps_c['Fe3Fet_Liq']) > 0:
            w.warn('Some Fe3Fet_Liq are greater than 0. Masotta et al. (2013)'
            ' calibrate their equations assuming all Fe is Fe2+. '
            'You should set Fe3Fet_Liq=0 in the function for consistency. ')
        if meltmatch is not None and np.max(meltmatch['Fe3Fet_Liq']) > 0:
            w.warn('Some Fe3Fet_Liq are greater than 0. Masotta et al. (2013)'
            ' calibrate their equations assuming all Fe is Fe2+. '
            'You should set Fe3Fet_Liq=0 in the function for consistency. ')




    if meltmatch is not None:
        Combo_liq_cpxs = meltmatch
    if liq_comps is not None:
        Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(
            liq_comps=liq_comps_c, cpx_comps=cpx_comps)

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T')
    else:
        if T is not None:
            print('Youve selected a T-independent function')
    # Easiest to treat Machine Learning ones differently



    if equationP == "P_Petrelli2021_Cpx_Liq":
        df_stats=P_Petrelli2021_Cpx_Liq(cpx_comps=cpx_comps, liq_comps=liq_comps_c)
        P_kbar=df_stats['P_kbar_calc']

    else:


        kwargs = {name: Combo_liq_cpxs[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        if isinstance(T, str) or T is None:
            if T == "Solve":
                P_kbar = partial(func, **kwargs)
            if T is None:
                P_kbar=func(**kwargs)

        else:
            P_kbar=func(T, **kwargs)


    if eq_tests is False:
        if isinstance(P_kbar, partial):
            return P_kbar
        else:
            P_kbar_is_bad = (P_kbar == 0) | (P_kbar == 273.15) | (P_kbar ==  -np.inf) | (P_kbar ==  np.inf)
            P_kbar[P_kbar_is_bad] = np.nan

            if equationP == "P_Petrelli2021_Cpx_Liq" and T != "Solve":
                return df_stats
            else:
                return P_kbar



    if eq_tests is True:
        if isinstance(P_kbar, partial):
            raise TypeError('cant calculate equilibrium tests if P_kbar isnt numerical'
            'e.g., if you havent specified a T for a T-dependent thermometer')
        if T is None:
            raise TypeError('You need to specify a T for equilibrium tests')

        if meltmatch is None:
            eq_tests = calculate_cpx_liq_eq_tests(cpx_comps=cpx_comps,
            liq_comps=liq_comps_c, Fe3Fet_Liq=Fe3Fet_Liq, P=P_kbar, T=T, sigma=sigma, KdErr=KdErr)
        if meltmatch is not None:
            eq_tests = calculate_cpx_liq_eq_tests(meltmatch=meltmatch,
            Fe3Fet_Liq=Fe3Fet_Liq, P=P, T=T_K, sigma=sigma, KdErr=KdErr)
        eq_tests.replace([np.inf, -np.inf], np.nan, inplace=True)
        return eq_tests


## Function for calculation Cpx-Liquid temperatures
Cpx_Liq_T_funcs = {T_Put1996_eqT1, T_Mas2013_eqTalk1, T_Put1996_eqT2, T_Mas2013_eqTalk2,
T_Put1999, T_Put2003, T_Put2008_eq33, T_Mas2013_eqalk33,
T_Mas2013_Talk2012, T_Brug2019, T_Petrelli2021_Cpx_Liq} # put on outside

Cpx_Liq_T_funcs_by_name = {p.__name__: p for p in Cpx_Liq_T_funcs}

def calculate_cpx_liq_temp(*, equationT, cpx_comps=None, liq_comps=None, meltmatch=None,
                           P=None, eq_tests=False, H2O_Liq=None, Fe3Fet_Liq=None,
                           sigma=1, KdErr=0.03):
    '''
    Clinopyroxene-Liquid thermometry, calculates temperature in Kelvin
    (and equilibrium tests as an option)

   Parameters
    -------
    cpx_comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.
    Or:

    meltmatch: DataFrame
        Combined dataframe of cpx-Liquid compositions
        Used for calculate_cpx_liq_press_temp_matching function.

    EquationT: str
        Choice of equation:
        Cpx-Liquid
        |  T_Put1996_eqT1  (P-indep, H2O-indep)
        |  T_Mas2013_eqTalk1  (P-indep, H2O-indep, alk adaption of T1)
        |  T_Brug2019  (P-indep, H2O-indep)
        |  T_Put1996_eqT2 (P-dep, H2O-indep)
        |  T_Mas2013_eqTalk2  (P-dep, H2O-indep, alk adaption of T2)
        |  T_Put1999  (P-dep, H2O-indep)
        |  T_Put2003  (P-dep, H2O-indep)
        |  T_Put1999  (P-dep, H2O-indep)
        |  T_Put2008_eq33  (P-dep, H2O-dep)
        |  T_Mas2013_eqalk33  (P-dep, H2O-dep, alk adaption of eq33)
        |  T_Mas2013_Palk2012 (P-indep, H2O_dep)


    P: float, int, series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series


    eq_tests: bool
        If False, just returns pressure (default) as a panda series
        If True, returns pressure, Values of Eq tests,
        Values of Eq tests (Kd, EnFs, DiHd, CaTs, CrCaTs),
        as well as user-entered cpx and liq comps and components.


    Returns
    -------
    If eq_tests=False
        pandas.series: Temperature in Kelvin
    If eq_tests=True
        panda.dataframe: Temperature in Kelvin +
        Eq Tests + cpx+liq comps + components

    '''
    # Various warnings etc. to check inputs make sense.
    try:
        func = Cpx_Liq_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')
    else:
        if P is not None:
            print('Youve selected a P-independent function')

    if isinstance(P, pd.Series):
        if liq_comps is not None:
            if len(P) != len(liq_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of liquid compositions')




    if meltmatch is not None:
        Combo_liq_cpxs = meltmatch

    if liq_comps is not None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq

        if equationT == "T_Mas2013_Palk2012" or equationT == "T_Mas2013_eqalk33" or equationT == "T_Mas2013_eqTalk2" or equationT == "T_Mas2013_eqTalk1":
            if np.max(liq_comps_c['Fe3Fet_Liq']) > 0:
                w.warn('Some Fe3Fet_Liq are greater than 0. Masotta et al. (2013) calibrate their equations assuming all Fe is Fe2+. You should set Fe3Fet_Liq=0 in the function for consistency. ')

        Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(liq_comps=liq_comps_c,
        cpx_comps=cpx_comps)


    if equationT == "T_Brug2019":
        if np.max(Combo_liq_cpxs['Mgno_CPX']) > 0.65:
            w.warn("Some inputted CPX compositions have Cpx Mg#>0.65;.",
                stacklevel=2)
        if np.max(Combo_liq_cpxs['Al2O3_Cpx']) > 7:
            w.warn("Some inputted CPX compositions have Al2O3>7 wt%;.",
                stacklevel=2)
        if np.max(Combo_liq_cpxs['SiO2_Liq']) < 70:
            w.warn("Some inputted Liq compositions have  SiO2<70 wt%;",
                stacklevel=2)
        if np.max(Combo_liq_cpxs['Mgno_CPX']) > 0.65 or Combo_liq_cpxs['Al2O3_Cpx'] or p.max(
                Combo_liq_cpxs['SiO2_Liq']) < 70:
            w.warn("which is outside the recomended calibration range of Brugman and Till (2019)")

    # Easiest to treat Machine Learning ones differently
    if equationT == "T_Petrelli2021_Cpx_Liq":
        df_stats=T_Petrelli2021_Cpx_Liq(cpx_comps=cpx_comps, liq_comps=liq_comps_c)
        T_K=df_stats['T_K_calc']
    else:
        kwargs = {name: Combo_liq_cpxs[name] for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY}


        if isinstance(P, str) or P is None:
            if P == "Solve":
                T_K = partial(func, **kwargs)
            if P is None:
                T_K=func(**kwargs)

        else:
            T_K=func(P, **kwargs)

    if eq_tests is False:
        if isinstance(T_K, partial):
            return T_K


        else:
            T_K_is_bad = (T_K == 0) | (T_K == 273.15) | (T_K ==  -np.inf) | (T_K ==  np.inf)
            T_K[T_K_is_bad] = np.nan

            if equationT == "T_Petrelli2021_Cpx_Liq" and P != "Solve":
                return df_stats
            else:
                return T_K


    if eq_tests is True:
        if isinstance(T_K, partial):
            raise TypeError('cant calculate equilibrium tests if T_K isnt numerical'
            'e.g., if you havent specified a P for a P-dependent thermometer')
        if P is None:
            raise TypeError('You need to specify a P for equilibrium tests')
        if meltmatch is None:
            eq_tests = calculate_cpx_liq_eq_tests(cpx_comps=cpx_comps,
            liq_comps=liq_comps_c, Fe3Fet_Liq=Fe3Fet_Liq, P=P, T=T_K, sigma=sigma, KdErr=KdErr)
        if meltmatch is not None:
            eq_tests = calculate_cpx_liq_eq_tests(meltmatch=meltmatch,
            Fe3Fet_Liq=Fe3Fet_Liq, P=P, T=T_K, sigma=sigma, KdErr=KdErr)
        eq_tests.replace([np.inf, -np.inf], np.nan, inplace=True)
        return eq_tests


## Function for iterating P and T

def calculate_cpx_liq_press_temp(*, liq_comps=None, cpx_comps=None, meltmatch=None, equationP=None, equationT=None,
                              T=None, P=None, iterations=None, Fe3Fet_Liq=None, H2O_Liq=None, T_K_guess=1300, eq_tests=False):
    '''
    Solves simultaneous equations for temperature and pressure
    using clinopyroxene-liquid thermometers and barometers.


   Parameters
    -------

     cpx_comps: DataFrame (opt, either specify cpx_comps AND liq_comps or meltmatch)
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    Or

    meltmatch: DataFrame
        Combined dataframe of cpx-Liquid compositions
        Used for calculate_cpx_liq_press_temp_matching function.

    EquationP: str
        Barometer
        Cpx-Liquid
        |  P_Put1996_eqP1 (T-dep, H2O-indep)
        |  P_Mas2013_eqPalk1 (T-dep, H2O-indep, alk adaption of P1)
        |  P_Put1996_eqP2 (T-dep, H2O-indep)
        |  P_Mas2013_eqPalk2 (T-dep, H2O-indep, alk adaption of P2)
        |  P_Put2003 ((T-dep, H2O-indep)
        |  P_Neave2017 (T-dep, H2O-indep)
        |  P_Put2008_eq30 (T-dep, H2O-dep)
        |  P_Put2008_eq31 (T-dep, H2O-dep)
        |  P_Put2008_eq32c (T-dep, H2O-dep)
        |  P_Mas2013_eqalk32c (T-dep, H2O-dep, alk adaption of 32c)


    EquationT: str
        Thermometer
        Cpx-Liquid
        |  T_Put1996_eqT1  (P-indep, H2O-indep)
        |  T_Mas2013_eqTalk1  (P-indep, H2O-indep, alk adaption of T1)
        |  T_Brug2019  (P-indep, H2O-indep)
        |  T_Put1996_eqT2 (P-dep, H2O-indep)
        |  T_Mas2013_eqTalk2  (P-dep, H2O-indep, alk adaption of T2)
        |  T_Put1999  (P-dep, H2O-indep)
        |  T_Put2003  (P-dep, H2O-indep)
        |  T_Put1999  (P-dep, H2O-indep)
        |  T_Put2008_eq33  (P-dep, H2O-dep)
        |  T_Mas2013_eqalk33  (P-dep, H2O-dep, alk adaption of eq33)
        |  T_Mas2013_Palk2012 (P-indep, H2O_dep)

    Optional:

    iterations: int (optional). Default is 20.
         Number of iterations used to converge to solution

    T_K_guess: int or float. Default is 1300K
         Initial guess of temperature.

    Fe3Fet_Liq: float, int, series,
        Fe3Fet ratio used to assess Kd Fe-Mg equilibrium between cpx and melt.
        If users don't specify, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    H2O_Liq: float, int, series, optional
        If users don't specify, uses H2O_Liq from liq_comps,
        if specified overwrites this.

    eq_tests: bool
        If False, just returns pressure in kbar, tempeature in Kelvin as a dataframe
        If True, returns pressure and temperature, Eq tests (Kd, EnFs, DiHd, CaTs, CrCaTs),
        as well as user-entered cpx and liq comps and components.


    Returns
    -------
    If eq_tests=False
        pandas.DataFrame: Temperature in Kelvin, pressure in Kbar
    If eq_tests=True
        panda.dataframe: Temperature in Kelvin, pressure in Kbar
        Eq Tests + cpx+liq comps + components


    Returns:
    -------
    panda.dataframe: Pressure in Kbar, Temperature in K + Kd-Fe-Mg + cpx+liq comps (if eq_tests=True)

    '''

    # Gives users flexibility to reduce or increase iterations
    if iterations is not None:
        iterations = iterations
    else:
        iterations = 30

    if meltmatch is None:
        liq_comps_c = liq_comps.copy()
        # This overwrites the Fe3Fet in their inputted dataframe
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
        if "Fe3Fet_Liq" not in liq_comps:
            liq_comps_c['Fe3Fet_Liq'] = 0
        if "Sample_ID_Liq" not in liq_comps:
            liq_comps_c['Sample_ID_Liq'] = liq_comps_c.index
        # This overwites H2O_Liq in the input
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

        T_func = calculate_cpx_liq_temp(
            cpx_comps=cpx_comps, liq_comps=liq_comps_c, equationT=equationT, P="Solve")
        P_func = calculate_cpx_liq_press(
            cpx_comps=cpx_comps, liq_comps=liq_comps_c, equationP=equationP, T="Solve")

    if meltmatch is not None:
        T_func = calculate_cpx_liq_temp(
            meltmatch=meltmatch, equationT=equationT, eq_tests=False, P="Solve")
        P_func = calculate_cpx_liq_press(
            meltmatch=meltmatch, equationP=equationP, eq_tests=False, T="Solve")

    if isinstance(P_func, pd.Series) and isinstance(T_func, partial):
        P_guess = P_func
        T_K_guess = T_func(P_guess)
    if isinstance(T_func, pd.Series) and isinstance(P_func, partial):
        T_K_guess = T_func
        P_guess = P_func(T_K_guess)
    if isinstance(T_func, pd.Series) and isinstance(P_func, pd.Series):
        T_K_guess = T_func
        P_guess = P_func

    if isinstance(P_func, partial) and isinstance(T_func, partial):

        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)

    if equationT != "T_Petrelli2021_Cpx_Liq":
        T_K_guess_is_bad = (T_K_guess == 0) | (T_K_guess == 273.15) | (T_K_guess ==  -np.inf) | (T_K_guess ==  np.inf)
        T_K_guess[T_K_guess_is_bad] = np.nan
    if equationP != "P_Petrelli2021_Cpx_Liq":
        P_guess[T_K_guess_is_bad] = np.nan


    # calculates equilibrium tests of Neave and Putirka if eq_tests="True"
    if eq_tests is False:
        PT_out = pd.DataFrame(
            data={'P_kbar_calc': P_guess, 'T_K_calc': T_K_guess})
        return PT_out
    if eq_tests is True:
        if meltmatch is not None:
            eq_tests = calculate_cpx_liq_eq_tests(
            meltmatch=meltmatch, P=P_guess, T=T_K_guess)
        if meltmatch is None:
            eq_tests = calculate_cpx_liq_eq_tests(cpx_comps=cpx_comps,
            liq_comps=liq_comps, P=P_guess, T=T_K_guess)
        return eq_tests

## Clinopyroxene melt-matching algorithm

def calculate_cpx_liq_press_temp_matching(*, liq_comps, cpx_comps, equationT=None,
equationP=None, P=None, T=None, eq_crit="All", PMax=30, sigma=1,
Fe3Fet_Liq=None, KdErr=0.03, KdMatch=None, Cpx_Quality=False,
H2O_Liq=None, Return_All_Matches=False):

    '''
    Evaluates all possible Opx-Liq pairs from  N Liquids, M Cpx compositions
    returns P (kbar) and T (K) for those in equilibrium.


   Parameters
    -------

    liq_comps: DataFrame
        Panda DataFrame of liquid compositions with column headings SiO2_Liq etc.

    cpx_comps: DataFrame
        Panda DataFrame of cpx compositions with column headings SiO2_Cpx etc.

    equationT: str
        Specify equation for cpx thermometry (options):

        |  T_Put1996_eqT1  (P-indep, H2O-indep)
        |  T_Mas2013_eqTalk1  (P-indep, H2O-indep, alk adaption of T1)
        |  T_Brug2019  (P-indep, H2O-indep)
        |  T_Put1996_eqT2 (P-dep, H2O-indep)
        |  T_Mas2013_eqTalk2  (P-dep, H2O-indep, alk adaption of T2)
        |  T_Put1999  (P-dep, H2O-indep)
        |  T_Put2003  (P-dep, H2O-indep)
        |  T_Put1999  (P-dep, H2O-indep)
        |  T_Put2008_eq33  (P-dep, H2O-dep)
        |  T_Mas2013_eqalk33  (P-dep, H2O-dep, alk adaption of eq33)
        |  T_Mas2013_Palk2012 (P-indep, H2O_dep)


    equationP: str
        specify equation for cpx barometry (options):

        |  P_Put1996_eqP1 (T-dep, H2O-indep)
        |  P_Mas2013_eqPalk1 (T-dep, H2O-indep, alk adaption of P1)
        |  P_Put1996_eqP2 (T-dep, H2O-indep)
        |  P_Mas2013_eqPalk2 (T-dep, H2O-indep, alk adaption of P2)
        |  P_Put2003 ((T-dep, H2O-indep)
        |  P_Neave2017 (T-dep, H2O-indep)
        |  P_Put2008_eq30 (T-dep, H2O-dep)
        |  P_Put2008_eq31 (T-dep, H2O-dep)
        |  P_Put2008_eq32c (T-dep, H2O-dep)
        |  P_Mas2013_eqalk32c (T-dep, H2O-dep, alk adaption of 32c)

    Optional:

    eq_crit: str, optional

        If None (default): Doesn't apply any filters
        If "All": applies the 4 equilibrium tests of Neave et al: KdFe-Mg, DiHd,
        EnFs, CaTs. Kd Fe-Mg calculated based on what you specify in KdMatch
        If "FeMg_DiHd": Filters just using KdFe-Mg and DiHd
        If "FeMg_EnFs": Filters just using KdFe-Mg and EnFs


    PMax: int or float,  optional

       Default value of 30 kbar. Uses to apply a preliminary KdFe-Mg filter
       based on the T equation specified by the user.
       Users can set a lower pressure to save computation time (E.g., if
       reasonably sure crystals are forming above 10 kbar)

    sigma: int or float, optional

        Determins how close to the ideal equilbrium test values for DiHd,
        EnFs and CaTs is accepted as a match.
        Default (1 sigma) values from Neave et al. (2017) are: 0.03 for Kd Fe-Mg,
        0.06 for DiHd, 0.05 for EnFs, 0.03 for CaTs.
        If users specify sigma=2, will accept cpx-melt pairs within 0.12 for DiHd,
        0.1 for EnFs, 0.06 for CaTs
        Doesn't appply to Kd

    KdErr: int or float, optional
        Allows users to specify the error on Kd Fe-Mg (default =0.03)


    KdMatch: int, str, optional
        allows users to override the default of calculating Kd Fe-Mg based
        on temperature using eq 35 of putirka
        Set at fixed value (e.g., KdMatch=0.27)
        OR
        specify KdMatch=Masotta to use the Kd model fo Masotta et al. (2013),
        which is also a function of Na and K, for trachytic and phonolitic magmas.

    Fe3Fet_Liq: float, int, series, optional
        Fe3Fet ratio used to assess Kd Fe-Mg equilibrium between cpx and melt.
        If users don't specify, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    H2O_Liq: float, int, series, optional
        If users don't specify, uses H2O_Liq from liq_comps, if specified overwrites this.

    Cpx Quality: bool, optional
        Default False. If True, filters out clinopyroxenes with cation sums outside of
        4.02-3.99 (after Neave et al. 2017)


    Returns: dict

        Av_PTs: Average P and T for each cpx.
        E.g., if cpx1 matches Liq1, Liq4, Liq6, Liq10, averages outputs for all 4 of those liquids.
        Returns mean and 1 sigma of these averaged parameters for each Cpx.

        All_PTs: Returns output parameters for all matches (e.g, cpx1-Liq1, cpx1-Liq4) without any averaging.

    '''
    if KdMatch == "Masotta":
        print('Caution, you have selected to use the Kd-Fe-Mg model of Masotta et al. (2013)'
        'which is only valid for trachyte and phonolitic magmas. '
        ' use PutKd=True to use the Kd model of Putirka (2008)')

    if isinstance(KdMatch, int) or isinstance(KdMatch, float) and KdErr is None:
        raise ValueError('You have entered a numerical value for KdMatch, '
        'You need to specify a KdErr to accept matches within KdMatch+-KdErr')

    if equationP is not None and P is not None:
        raise ValueError('You have entered an equation for P and specified a pressure. '
        'Either enter a P equation, or choose a pressure, not both ')
    if equationT is not None and T is not None:
        raise ValueError('You have entered an equation for T and specified a temperature. '
        'Either enter a T equation, or choose a temperature, not both  ')

    # This allows users to overwrite H2O and Fe3Fet
    liq_comps_c = liq_comps.copy()
    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
    if "Fe3Fet_Liq" not in liq_comps:
        liq_comps_c['Fe3Fet_Liq'] = 0
    if "Sample_ID_Liq" not in liq_comps:
        liq_comps_c['Sample_ID_Liq'] = liq_comps.index
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq'] = H2O_Liq

    if sigma is not None:
        sigma = sigma

    # calculating Cpx and liq components.
    myCPXs1_concat = calculate_clinopyroxene_components(cpx_comps=cpx_comps)
    myLiquids1_concat = calculate_anhydrous_cat_fractions_liquid(
        liq_comps=liq_comps_c)

    # Adding an ID label to help with melt-cpx rematching later
    myCPXs1_concat['ID_CPX'] = myCPXs1_concat.index
    if "Sample_ID_Cpx" not in cpx_comps:
        myCPXs1_concat['Sample_ID_Cpx'] = myCPXs1_concat.index
    else:
        myCPXs1_concat['Sample_ID_Cpx']=cpx_comps['Sample_ID_Cpx']

    myCPXs1_concat['ID_CPX']=myCPXs1_concat.index
    myLiquids1_concat['ID_Liq'] = myLiquids1_concat.index


    # This duplicates CPXs, repeats cpx1-cpx1*N, cpx2-cpx2*N etc.
    DupCPXs = pd.DataFrame(np.repeat(myCPXs1_concat.values,
    np.shape(myLiquids1_concat)[0], axis=0))
    DupCPXs.columns = myCPXs1_concat.columns

    # This duplicates liquids like liq1-liq2-liq3 for cpx1, liq1-liq2-liq3 for
    # cpx2 etc.
    DupLiqs = pd.concat([myLiquids1_concat] *
                        np.shape(myCPXs1_concat)[0]).reset_index(drop=True)

    # Combines these merged liquids and cpx dataframes
    Combo_liq_cpxs = pd.concat([DupLiqs, DupCPXs], axis=1)
    LenCombo = str(np.shape(Combo_liq_cpxs)[0])

    # Status update for user
    print("Considering " + LenCombo +
          " Liq-Cpx pairs, be patient if this is >>1 million!")

    # calculate clinopyroxene-liquid components for this merged dataframe
    Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(meltmatch=Combo_liq_cpxs)

    # This returns the stiched dataframe of cpx-liq, no P or T yet.
    if Return_All_Matches is True:
        return Combo_liq_cpxs


    else:

        if Cpx_Quality is True:
            Combo_liq_cpxs_2 = Combo_liq_cpxs.loc[(Combo_liq_cpxs['Cation_Sum_Cpx'] < 4.02) & (
                Combo_liq_cpxs['Cation_Sum_Cpx'] > 3.99) & (Combo_liq_cpxs['Jd'] > 0.01)]

        if Cpx_Quality is False:
            Combo_liq_cpxs_2 = Combo_liq_cpxs.copy()
        # This section of code is for when users specify a presssure or
        # temperature, its much faster to not have to iterate, so we don't need
        # the preliminary Kd filter

        if P is not None:
            Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs2.copy()
            T_K_calc = calculate_cpx_liq_temp(meltmatch=Combo_liq_cpxs_FeMgMatch,
            equationT=equationT, P=P)
            P_guess = P
            T_K_guess = T_K_calc
        if T is not None:
            Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs.copy()
            P_kbar_calc = calculate_cpx_opx_temp(meltmatch=Combo_liq_cpxs_FeMgMatch,
            equationP=equationP, T=T)
            P_guess = P_kbar_calc
            T_K_guess = T




        if equationP is not None and equationT is not None:

            # Initial Mg# filter, done by calculating temperature for extreme pressures,
            # e.g, 0 and 3 Gpa. Reduces number of P-T solving
            PMin = -10
            PMax = PMax
            KdErr = KdErr

        # Filter out bad analysis first off



            if eq_crit is None:
                Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs_2.copy()
            else:
                Combo_liq_cpxs_2['T_Liq_MinP'] = calculate_cpx_liq_temp(
                    meltmatch=Combo_liq_cpxs_2, equationT=equationT, P=PMin)
                Combo_liq_cpxs_2['T_Liq_MaxP'] = calculate_cpx_liq_temp(
                    meltmatch=Combo_liq_cpxs_2, equationT=equationT, P=PMax)
                # calculating Delta Kd-Fe-Mg using equation 35 of Putirka 2008
                if KdMatch is None or KdMatch == "Putirka":
                    Combo_liq_cpxs_2['Kd_MinP'] = np.exp(
                        -0.107 - 1719 / Combo_liq_cpxs_2['T_Liq_MinP'])
                    Combo_liq_cpxs_2['Kd_MaxP'] = np.exp(
                        -0.107 - 1719 / Combo_liq_cpxs_2['T_Liq_MaxP'])

                    Delta_Kd_T_MinP = abs(
                        Combo_liq_cpxs_2['Kd_MinP'] - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                    Delta_Kd_T_MaxP = abs(
                        Combo_liq_cpxs_2['Kd_MaxP'] - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])

                if KdMatch is not None and KdMatch != "Masotta" and KdMatch != "Putirka":
                    str3 = str(KdMatch)
                    print('the code is evaluating Kd matches using Kd=' + str3)
                    Delta_Kd_T_MinP = abs(
                        KdMatch - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                    Delta_Kd_T_MaxP = abs(
                        KdMatch - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                    Combo_liq_cpxs_2.insert(
                        0, "DeltaKd_userselected=" + str3, Delta_Kd_T_MinP)

                if KdMatch == "Masotta":

                    ratioMasotta = Combo_liq_cpxs_2['Na2O_Liq_cat_frac'] / (
                        Combo_liq_cpxs_2['Na2O_Liq_cat_frac'] + Combo_liq_cpxs_2['K2O_Liq_cat_frac'])
                    Delta_Kd_T_MinP = abs(
                        np.exp(1.735 - 3056 / Combo_liq_cpxs_2['T_Liq_MinP'] - 1.668 * ratioMasotta) - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                    Delta_Kd_T_MaxP = abs(
                        np.exp(1.735 - 3056 / Combo_liq_cpxs_2['T_Liq_MaxP'] - 1.668 * ratioMasotta) - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])

                # The logic here is that if Delta KD with both the max and min temperature are outside the specified KDerror,
                # no temperature inbetween will be inequilibrium.
                Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs_2.loc[~((Delta_Kd_T_MaxP > KdErr) &
                                                                  (Delta_Kd_T_MinP > KdErr))].reset_index(drop = True)

            str2 = str(np.shape(Combo_liq_cpxs_FeMgMatch)[0])
            print(str2 + ' Matches remaining after initial Kd filter. '
            'Now moving onto iterative calculations')

            # Now we have reduced down the number of calculations, we solve for
            # P and T iteratively

            PT_out = calculate_cpx_liq_press_temp(meltmatch=Combo_liq_cpxs_FeMgMatch,
            equationP=equationP, equationT=equationT)
            P_guess = PT_out['P_kbar_calc']
            T_K_guess = PT_out['T_K_calc']

        # Now, we use calculated pressures and temperatures, regardless of
        # whether we iterated or not, to calculate the other CPX components
        Combo_liq_cpxs_eq_comp = calculate_cpx_liq_eq_tests(
            meltmatch=Combo_liq_cpxs_FeMgMatch, P=P_guess, T=T_K_guess)

        combo_liq_cpx_fur_filt = Combo_liq_cpxs_eq_comp.copy()

        # First, make filter based on various Kd optoins
        if KdMatch is not None and KdMatch != "Masotta" and KdMatch != "Putirka":
            Combo_liq_cpxs_eq_comp.loc[:, 'DeltaKd_KdMatch_userSp']= abs(
                KdMatch - Combo_liq_cpxs_eq_comp['Kd_Fe_Mg_Fe2'])
            filtKd = (Combo_liq_cpxs_eq_comp['DeltaKd_KdMatch_userSp'] < KdErr)
        else:
            if KdMatch is None or KdMatch == "Putirka":
                filtKd = (Combo_liq_cpxs_eq_comp['Delta_Kd_Put2008'] < KdErr)
            if KdMatch == "Masotta":
                filtKd = (Combo_liq_cpxs_eq_comp['Delta_Kd_Mas2013'] < KdErr)

        if eq_crit is None:
            combo_liq_cpx_fur_filt = Combo_liq_cpxs_eq_comp.copy()
        if eq_crit == "All":
            combo_liq_cpx_fur_filt = (Combo_liq_cpxs_eq_comp.loc[filtKd & (Combo_liq_cpxs_eq_comp['Delta_DiHd'] < 0.06 * sigma) & (
                Combo_liq_cpxs_eq_comp['Delta_EnFs'] < 0.05 * sigma) & (Combo_liq_cpxs_eq_comp['Delta_CaTs'] < 0.03 * sigma)])
        if eq_crit == "Kd":
            combo_liq_cpx_fur_filt = (Combo_liq_cpxs_eq_comp.loc[filtKd])
        if eq_crit == "Kd_DiHd":
            combo_liq_cpx_fur_filt = Combo_liq_cpxs_eq_comp.loc[filtKd & (
                Combo_liq_cpxs_eq_comp['Delta_DiHd'] < 0.06 * sigma)]
        if eq_crit == "Kd_EnFs":
            combo_liq_cpx_fur_filt = Combo_liq_cpxs_eq_comp.loc[filtKd & (
                Combo_liq_cpxs_eq_comp['Delta_EnFs'] < 0.05 * sigma)]
        if eq_crit == "Kd_CaTs":
            combo_liq_cpx_fur_filt = Combo_liq_cpxs_eq_comp.loc[filtKd & (
                Combo_liq_cpxs_eq_comp['Delta_CaTs'] < 0.03 * sigma)]

        # This just tidies up some columns to put stuff nearer the start
        cols_to_move = ['Sample_ID_Liq', 'Sample_ID_Cpx']
        combo_liq_cpx_fur_filt = combo_liq_cpx_fur_filt[cols_to_move + [
            col for col in combo_liq_cpx_fur_filt.columns if col not in cols_to_move]]

        combo_liq_cpx_fur_filt.drop(["Sample_ID_Liq"], axis=1, inplace=True)
        if T is not None:
            combo_liq_cpx_fur_filt.rename(columns={'T_K_calc': 'T_K_input'}, inplace=True)
        if P is not None:
            combo_liq_cpx_fur_filt.rename(columns={'P_kbar_calc': 'P_kbar_input'}, inplace=True)

        print('Finished calculating Ps and Ts, now just averaging the results. Almost there!')

        # Final step, calcuate a 3rd output which is the average and standard
        # deviation for each CPx (e.g., CPx1-Melt1, CPx1-melt3 etc. )
        CpxNumbers = combo_liq_cpx_fur_filt['ID_CPX'].unique()
        if len(CpxNumbers) > 0:
            df1_M = pd.DataFrame()
            df1_S = pd.DataFrame()
            for cpx in CpxNumbers:
                dff_M = pd.DataFrame(
                    combo_liq_cpx_fur_filt.loc[combo_liq_cpx_fur_filt['ID_CPX'] == cpx].mean(axis=0)).T
                dff_M['Sample_ID_Cpx'] = combo_liq_cpx_fur_filt.loc[combo_liq_cpx_fur_filt['ID_CPX']
                                                                    == cpx, "Sample_ID_Cpx"].iloc[0]
                if cpx == CpxNumbers[0]:
                    df1_M = dff_M
                    df1_M['Sample_ID_Cpx'] = combo_liq_cpx_fur_filt.loc[combo_liq_cpx_fur_filt['ID_CPX']
                                                                        == cpx, "Sample_ID_Cpx"].iloc[0]
                else:
                    df1_M = pd.concat([df1_M, dff_M], sort=False)

            df1_M = df1_M.add_prefix('Mean_')
            cols_to_move = ['Mean_Sample_ID_Cpx',
                            'Mean_T_K_calc', 'Mean_P_kbar_calc']
            df1_M = df1_M[cols_to_move +
                          [col for col in df1_M.columns if col not in cols_to_move]]
            df1_M = df1_M.rename(
                columns={'Mean_Sample_ID_Cpx': 'Sample_ID_Cpx'})
            for cpx in CpxNumbers:
                dff_S = pd.DataFrame(
                    combo_liq_cpx_fur_filt.loc[combo_liq_cpx_fur_filt['ID_CPX'] == cpx].std(axis=0)).T
                # This tells us if there is only 1, in which case std will
                # return Nan
                if np.shape(
                        combo_liq_cpx_fur_filt.loc[combo_liq_cpx_fur_filt['ID_CPX'] == cpx])[0] == 1:
                    dff_S = dff_S.fillna(0)
                    dff_S['N'] = 1
                else:
                    dff_S = dff_S
                    dff_S['N'] = np.shape(
                        combo_liq_cpx_fur_filt.loc[combo_liq_cpx_fur_filt['ID_CPX'] == cpx])[0]
                if cpx == CpxNumbers[0]:
                    df1_S = dff_S
                else:
                    df1_S = pd.concat([df1_S, dff_S])

            df1_S = df1_S.add_prefix('st_dev_')
            df1_M.insert(0, "No. of liquids averaged", df1_S['st_dev_N'])
            if equationP is not None and equationT is not None:
                df1_M.insert(3, "st_dev_T_K_calc", df1_S['st_dev_T_K_calc'])
                df1_M.insert(5, "st_dev_P_kbar_calc",
                             df1_S['st_dev_P_kbar_calc'])
            if P is not None:
                df1_M.insert(3, "st_dev_T_K_calc", df1_S['st_dev_T_K_calc'])
                #df1_M=df1_M.drop(['Mean_Sample_ID_Liq', 'Mean_index'], axis=1)
                df1_M.rename(columns={'Mean_P_kbar_calc': 'P_kbar_input'})
            if T is not None:
                df1_M.insert(5, "st_dev_P_kbar_calc",
                             df1_S['st_dev_P_kbar_calc'])
                #df1_M=df1_M.drop(['Mean_Sample_ID_Liq', 'Mean_index'], axis=1)
                df1_M.rename(columns={'Mean_T_K_calc': 'T_K_input'})

        else:
            raise Exception(
                'No Matches - to set less strict filters, change perhaps change sigma to a value greater'
                ' than one, or specify eq_crit for only a subset of the values in Neave and Putirka')

        if P is not None:
            combo_liq_cpx_fur_filt = combo_liq_cpx_fur_filt.rename(
                columns={'P_kbar_calc': 'P_kbar_input'})
        if T is not None:
            combo_liq_cpx_fur_filt = combo_liq_cpx_fur_filt.rename(
                columns={'T_K_calc': 'T_K_input'})

        print('Done!')
        return {'Av_PTs': df1_M, 'All_PTs': combo_liq_cpx_fur_filt}

## Clinopyroxene-only pressure equations

def P_Wang2021_eq1(T=None, *, Al_VI_cat_6ox, SiO2_Cpx_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox,
FeOt_Cpx_cat_6ox, MnO_Cpx_cat_6ox, MgO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, K2O_Cpx_cat_6ox, CaO_Cpx_cat_6ox):
    '''
    Clinopyroxene-only barometer of Wang et al. (2021) equation 1
    Uses NCT
    - currently on Zenodo - 10.5281/zenodo.4727870
    '''
    NCT=(2.2087*Al_VI_cat_6ox/(2.2087*Al_VI_cat_6ox+9.3594*TiO2_Cpx_cat_6ox
    +1.5117*Cr2O3_Cpx_cat_6ox+1.4768*FeOt_Cpx_cat_6ox-5.7686*MnO_Cpx_cat_6ox-0.0864*MgO_Cpx_cat_6ox))

    return (-7.6551*NCT*np.log(Al_VI_cat_6ox.astype(float))-10.2203*SiO2_Cpx_cat_6ox+4.8343*FeOt_Cpx_cat_6ox
+0.7397*MgO_Cpx_cat_6ox-13.1746*CaO_Cpx_cat_6ox+122.1294*Na2O_Cpx_cat_6ox+23.35)


def P_Wang2021_eq3(T=None, *, Al_VI_cat_6ox, SiO2_Cpx_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox,
FeOt_Cpx_cat_6ox, MnO_Cpx_cat_6ox, MgO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, K2O_Cpx_cat_6ox,
FeII_Wang21, FeIII_Wang21, CaO_Cpx_cat_6ox, Al2O3_Cpx_cat_6ox):
    '''
    Clinopyroxene-only barometer of Wang et al. (2021) equation 3
    - currently on Zenodo - 10.5281/zenodo.4727870.
    Doesnt use NCT
    '''


    return (-1105.84-18.6052*TiO2_Cpx_cat_6ox+252.1033*Al2O3_Cpx_cat_6ox+311.0123*Cr2O3_Cpx_cat_6ox+550.2534*FeOt_Cpx_cat_6ox+
451.6495*MnO_Cpx_cat_6ox+554.0535*MgO_Cpx_cat_6ox+540.2934*CaO_Cpx_cat_6ox
+902.6805*Na2O_Cpx_cat_6ox-535.305*FeIII_Wang21-70.1424*Al_VI_cat_6ox*np.log(Al_VI_cat_6ox.astype(float))
-1.74473*np.log(Al_VI_cat_6ox.astype(float)))



def P_Put2008_eq32a(T, *, MgO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox,
                    Al_VI_cat_6ox, DiHd_2003, EnFs):
    '''
    Clinopyroxene-only barometer of Putirka (2008) Eq32a

    | SEE=+-3.1 kbar (anhydrous)

    '''
    return (3205 - 5.62 * MgO_Cpx_cat_6ox + 83.2 * Na2O_Cpx_cat_6ox + 68.2 * DiHd_2003
    + 2.52 * np.log(Al_VI_cat_6ox.astype(float)) - 51.1 * DiHd_2003**2 + 34.8 * EnFs**2
    + 0.384 * T - 518 * np.log(T))


def P_Put2008_eq32b(T, *, H2O_Liq, CaO_Cpx_cat_6ox, MnO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox,
Al2O3_Cpx_cat_6ox, K2O_Cpx_cat_6ox, Al_IV_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox,
                    Al_VI_cat_6ox, FeOt_Cpx_cat_6ox, DiHd_1996, MgO_Cpx_cat_6ox, Jd):
    '''
    Clinopyroxene-only barometer of Putirka (2008) Eq32b. Unlike 32a, requires H2O_Liq to be specified.

    | SEE=+-2.6 kbar (anhydrous)

    '''

    CNM = CaO_Cpx_cat_6ox + Na2O_Cpx_cat_6ox + MnO_Cpx_cat_6ox  # This is GQ
    Lindley_Fe3_Cpx = Na2O_Cpx_cat_6ox + Al_IV_cat_6ox - Al_VI_cat_6ox - \
        2 * TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox  # This is cell FR
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    R3_plus = Al_VI_cat_6ox + TiO2_Cpx_cat_6ox + \
        Cr2O3_Cpx_cat_6ox + Lindley_Fe3_Cpx  # This is cell GR
    M1_M2_KD_Fe_Mg = np.exp(0.238 * R3_plus + 0.289 * CNM - 2.3315)
    M1_M1_Fe2_tot = FeOt_Cpx_cat_6ox - Lindley_Fe3_Cpx  # GV15
    b = (M1_M2_KD_Fe_Mg * MgO_Cpx_cat_6ox) - \
        (M1_M2_KD_Fe_Mg * (1 - CNM)) + M1_M1_Fe2_tot + (1 - CNM)  # GX
    a_Kd_1 = 1 - M1_M2_KD_Fe_Mg  # GW
    c = -M1_M1_Fe2_tot * (1 - CNM)  # GY
    Fe_M2 = (-b + np.sqrt(b**2 - 4 * a_Kd_1 * c)) / 2 * a_Kd_1
    Mg_M2 = 1 - CNM - Fe_M2
    return (1458 + 0.197 * (T) - 241 * np.log(T) +
    0.453 * H2O_Liq + 55.5 * Al_VI_cat_6ox +
    8.05 * FeOt_Cpx_cat_6ox - 277 * K2O_Cpx_cat_6ox + 18 * Jd
    + 44.1 * DiHd_1996 + 2.2 * np.log(Jd.astype(float))
    - 27.7 * Al2O3_Cpx_cat_6ox**2 + 97.3 * Fe_M2**2
    + 30.7 * Mg_M2**2 - 27.6 * DiHd_1996**2)

def P_Petrelli2021_Cpx_only(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer of Petrelli et al. (2021) based on
    Machine Learning.
    |  SEE==+-3.1 kbar
    '''
    cpx_test=cpx_comps.copy()
    Cpx_test_noID_noT=cpx_test.drop(['Sample_ID_Cpx'], axis=1)
    x_test=Cpx_test_noID_noT.values


    with open(Thermobar_dir/'ML_scaler_Petrelli2020_Cpx_Only.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Only.pkl', 'rb') as f:
        ETR_Press_P2020_Cpx_only=load(f)


    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_P_kbar=ETR_Press_P2020_Cpx_only.predict(x_test_scaled)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_only)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)

    return df_stats


def P_Petrelli2021_Cpx_only_withH2O(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer following the Machine learning approach of
    Petrelli et al. (2021),
    but including the H2O content of the liquid while training the model.
    '''
    cpx_test=cpx_comps.copy()
    Cpx_test_noID_noT=cpx_test.drop(['Sample_ID_Cpx'], axis=1)
    x_test=Cpx_test_noID_noT.values


    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_H2O.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Only_H2O.pkl', 'rb') as f:
        ETR_Press_P2020_Cpx_only=load(f)



    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_P_kbar=ETR_Press_P2020_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_only)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)

    return df_stats



## Clinopyroxene-only temperature equations
def T_Wang2021_eq2(P=None, *, Al_VI_cat_6ox, SiO2_Cpx_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox,
FeOt_Cpx_cat_6ox, MnO_Cpx_cat_6ox, MgO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, K2O_Cpx_cat_6ox,
FeII_Wang21, H2O_Liq, Al2O3_Cpx_cat_6ox, CaO_Cpx_cat_6ox):
    '''
    Clinopyroxene-only thermometer of Wang et al. (2021) equation 2 - currently on Zenodo - 10.5281/zenodo.4727870
    '''
    NCT=(2.2087*Al_VI_cat_6ox/(2.2087*Al_VI_cat_6ox+9.3594*TiO2_Cpx_cat_6ox
    +1.5117*Cr2O3_Cpx_cat_6ox+1.4768*FeOt_Cpx_cat_6ox-5.7686*MnO_Cpx_cat_6ox-0.0864*MgO_Cpx_cat_6ox))


    return (273.15+226.3499*NCT-444.507*TiO2_Cpx_cat_6ox-550.66*Al2O3_Cpx_cat_6ox
    -4290.88*MnO_Cpx_cat_6ox-580.33*MgO_Cpx_cat_6ox-760.789*CaO_Cpx_cat_6ox
    -3612.82*K2O_Cpx_cat_6ox-732.13*FeII_Wang21-23.6413*H2O_Liq+2513.694)


def T_Wang2021_eq4(P=None, *, Al_VI_cat_6ox, SiO2_Cpx_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox,
FeOt_Cpx_cat_6ox, MnO_Cpx_cat_6ox, MgO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, K2O_Cpx_cat_6ox,
FeII_Wang21, H2O_Liq, Al2O3_Cpx_cat_6ox):
    '''
    Clinopyroxene-only thermometer of Wang et al. (2021) equation 2 - currently on Zenodo - 10.5281/zenodo.4727870
    '''
    NCT=(2.2087*Al_VI_cat_6ox/(2.2087*Al_VI_cat_6ox+9.3594*TiO2_Cpx_cat_6ox
    +1.5117*Cr2O3_Cpx_cat_6ox+1.4768*FeOt_Cpx_cat_6ox-5.7686*MnO_Cpx_cat_6ox-0.0864*MgO_Cpx_cat_6ox))


    return (273.15+1270.004-1362.6*Al2O3_Cpx_cat_6ox+2087.355*Cr2O3_Cpx_cat_6ox
    +850.6013*FeOt_Cpx_cat_6ox-2881.1*MnO_Cpx_cat_6ox-5511.84*K2O_Cpx_cat_6ox
    +2821.792*Al_VI_cat_6ox-972.506*FeII_Wang21-26.7148*H2O_Liq)



def T_Put2008_eq32d(P, *, TiO2_Cpx_cat_6ox, FeOt_Cpx_cat_6ox, Al2O3_Cpx_cat_6ox,
                    Cr2O3_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, K2O_Cpx_cat_6ox, a_cpx_En):
    '''
    Clinopyroxene-only thermometer of Putirka (2008) Eq32d. Does overestimate temperature for hydrous data

    | SEE=±58°C (anhydrous)
    | SEE=±87°C (anhydrous)
    '''

    return (((93100 + 544 * P) / (61.1 + 36.6 * TiO2_Cpx_cat_6ox + 10.9 * FeOt_Cpx_cat_6ox
    - 0.95 * (Al2O3_Cpx_cat_6ox + Cr2O3_Cpx_cat_6ox - Na2O_Cpx_cat_6ox - K2O_Cpx_cat_6ox)
     + 0.395 * (np.log(a_cpx_En.astype(float)))**2)))


# This is the version in the Putirka 2-pyroxene spreadsheet. The 544 coefficient is changed to 755, and the coefficient for the enstatie activity is
# changed from 0.395 to 3.5. This is a better fit to subsolidus T
# estimates, but provide poorer fits to pyroxenes from volcanic systems.


def T_Put2008_eq32d_subsol(P, *, TiO2_Cpx_cat_6ox, FeOt_Cpx_cat_6ox, Al2O3_Cpx_cat_6ox,
                           Cr2O3_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, K2O_Cpx_cat_6ox, a_cpx_En):
    '''
    Adapted version of clinopyroxene-only thermoter of Putirka (2008) Eq32d, provides better fit to subsolidus T estimates
    '''
    return (((93100 + 755 * P) / (61.1 + 36.6 * TiO2_Cpx_cat_6ox + 10.9 * FeOt_Cpx_cat_6ox
    - 0.95 * (Al2O3_Cpx_cat_6ox + Cr2O3_Cpx_cat_6ox - Na2O_Cpx_cat_6ox - K2O_Cpx_cat_6ox)
    + 3.5 * (np.log(a_cpx_En.astype(float)))**2)))


def T_Petrelli2021_Cpx_only(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Petrelli et al. (2021) based on
    Machine Learning.
    |  SEE==+-51°C
    '''
    cpx_test=cpx_comps.copy()
    Cpx_test_noID_noT=cpx_test.drop(['Sample_ID_Cpx'], axis=1)
    x_test=Cpx_test_noID_noT.values

    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Temp_Petrelli2020_Cpx_Only.pkl', 'rb') as f:
        ETR_Temp_P2020_Cpx_only=load(f)

    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_T_K=ETR_Temp_P2020_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_P2020_Cpx_only)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats


def T_Petrelli2021_Cpx_only_withH2O(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Petrelli et al. (2021) based on
    Machine Learning.
    |  SEE==+-51°C
    '''
    cpx_test=cpx_comps.copy()
    Cpx_test_noID_noT=cpx_test.drop(['Sample_ID_Cpx'], axis=1)
    x_test=Cpx_test_noID_noT.values



    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_H2O.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Temp_Petrelli2020_Cpx_Only_H2O.pkl', 'rb') as f:
        ETR_Temp_P2020_Cpx_only=load(f)



    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_T_K=ETR_Temp_P2020_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_P2020_Cpx_only)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats

## Function for calculationg Cpx-only pressure
Cpx_only_P_funcs = {P_Put2008_eq32a, P_Put2008_eq32b, P_Wang2021_eq1,
P_Wang2021_eq3, P_Petrelli2021_Cpx_only, P_Petrelli2021_Cpx_only_withH2O}
Cpx_only_P_funcs_by_name = {p.__name__: p for p in Cpx_only_P_funcs}


def calculate_cpx_only_press(*, cpx_comps, equationP, T=None, H2O_Liq=None):
    '''
    Clinopyroxene only barometry. Enter a panda dataframe with Cpx compositions,
    returns a pressure in kbar.

   Parameters
    -------

    cpx_comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    equationP: str
        | P_Put2008_eq32a (T-dependent)
        | P_Put2008_eq32b (T-dependent, H2O dependent)
        | P_Petrelli2021_Cpx_only (T_independent, H2O-independent)
        | P_Petrelli2021_Cpx_only_withH2O (T_independent, H2O-dependent)
        | P_Wang2021_eq1 (T_independent, H2O-independent)
        | P_Wang2021_eq3 (T_independent, H2O-independent)


    T: float, int, series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas series
       Pressure in kbar

    '''

    cpx_comps_c=cpx_comps.copy()
    cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)

    if H2O_Liq is None:
        if equationP == "P_Put2008_eq32b" or  equationP == "P_Petrelli2021_Cpx_only_withH2O":
            w.warn('This Cpx-only barometer is sensitive to H2O content of the liquid. '
        ' By default, this function uses H2O=0 wt%, else you can enter a value of H2O_Liq in the function')
            cpx_components['H2O_Liq']=0
            cpx_comps_c['H2O_Liq']=0
    if H2O_Liq is not None:
        if equationP == "P_Put2008_eq32b" or  equationP == "P_Petrelli2021_Cpx_only_withH2O":
            cpx_components['H2O_Liq']=H2O_Liq
            cpx_comps_c['H2O_Liq']=H2O_Liq


    try:
        func = Cpx_only_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T, or specify T="Solve"')
    else:
        if T is not None:
            print('Youve selected a T-independent function')

    if isinstance(T, pd.Series):
        if len(T) != len(cpx_comps):
            raise ValueError('The panda series entered for temperature isnt the same '
            'length as the dataframe of Cpx compositions')




    if equationP == "P_Petrelli2021_Cpx_only":
        df_stats=P_Petrelli2021_Cpx_only(cpx_comps=cpx_comps_c)
        P_kbar=df_stats['P_kbar_calc']

    elif equationP == "P_Petrelli2021_Cpx_only_withH2O":
        df_stats=P_Petrelli2021_Cpx_only_withH2O(cpx_comps=cpx_comps_c)
        P_kbar=df_stats['P_kbar_calc']

    else:

        kwargs = {name: cpx_components[name] for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY}

        if isinstance(T, str) or T is None:
            if T == "Solve":
                P_kbar = partial(func, **kwargs)
            if T is None:
                P_kbar=func(**kwargs)

        else:
            P_kbar=func(T, **kwargs)
        if not isinstance(P_kbar, partial):
            P_kbar.replace([np.inf, -np.inf], np.nan, inplace=True)



    if equationP == "P_Petrelli2021_Cpx_only" and T != "Solve":
        return df_stats
    elif equationP == "P_Petrelli2021_Cpx_only_withH2O" and T != "Solve":
        return df_stats
    else:
        return P_kbar

## Function for calculating Cpx-only temperature
Cpx_only_T_funcs = {T_Put2008_eq32d, T_Put2008_eq32d_subsol, T_Wang2021_eq4,
T_Wang2021_eq2, T_Petrelli2021_Cpx_only, T_Petrelli2021_Cpx_only_withH2O}
Cpx_only_T_funcs_by_name = {p.__name__: p for p in Cpx_only_T_funcs}


def calculate_cpx_only_temp(*, cpx_comps=None, equationT=None, P=None, H2O_Liq=None):
    '''
    Clinopyroxene only thermometer. Enter a panda dataframe with Cpx compositions,
    returns a temperature in Kelvin.

   Parameters
    -------

    cpx_comps: DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    equationT: str
        | T_Put2008_eq32d (P-dependent)
        | T_Put2008_eq32d_subsol (P-dependent)
        | T_Petrelli2021_Cpx_only (P-independent, H2O-independent)
        | T_Petrelli2021_Cpx_only_withH2O (P-independent, H2O-dependent)
        | T_Wang2021_eq2 (P-independent, H2O-dependent)
        | T_Wang2021_eq4 (P-independent, H2O-dependent)



    P: float, int, series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas series
       Temperature in Kelvin

    '''
    cpx_comps_c=cpx_comps.copy()
    cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)

    if H2O_Liq is None:
        if equationT == "T_Petrelli2021_Cpx_only_withH2O" or  equationT == "T_Wang2021_eq2" or equationT == "T_Wang2021_eq4":
            w.warn('This Cpx-only thermometer is sensitive to H2O content of the liquid. '
        ' By default, this function uses H2O=0 wt%, else you can enter a value of H2O_Liq in the function')
            cpx_components['H2O_Liq']=0
            cpx_comps_c['H2O_Liq']=0
    if H2O_Liq is not None:
        if equationT == "T_Petrelli2021_Cpx_only_withH2O" or  equationT == "T_Wang2021_eq2" or equationT == "T_Wang2021_eq4":
            cpx_components['H2O_Liq']=H2O_Liq
            cpx_comps_c['H2O_Liq']=H2O_Liq


    try:
        func = Cpx_only_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')
    else:
        if P is not None:
            print('Youve selected a P-independent function')

    if isinstance(P, pd.Series):
        if len(P) != len(cpx_comps):
            raise ValueError('The panda series entered for Pressure isnt the same '
            'length as the dataframe of Cpx compositions')


    if equationT == "T_Petrelli2021_Cpx_only":

        df_stats=T_Petrelli2021_Cpx_only(cpx_comps=cpx_comps_c)
        T_K=df_stats['T_K_calc']


    elif equationT == "T_Petrelli2021_Cpx_only_withH2O":
        df_stats=T_Petrelli2021_Cpx_only_withH2O(cpx_comps=cpx_comps_c)
        T_K=df_stats['T_K_calc']


    else:

        kwargs = {name: cpx_components[name] for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY}

        if isinstance(P, str) or P is None:
            if P == "Solve":
                T_K = partial(func, **kwargs)
            if P is None:
                T_K=func(**kwargs)

        else:
            T_K=func(P, **kwargs)

        if not isinstance(T_K, partial):
            T_K.replace([np.inf, -np.inf], np.nan, inplace=True)



    if P != "Solve" and equationT == "T_Petrelli2021_Cpx_only":
        return df_stats
    elif P != "Solve" and equationT == "T_Petrelli2021_Cpx_only_withH2O":
        return df_stats
    else:
        return T_K

## Iterating PT- Cpx only
def calculate_cpx_only_press_temp(*, cpx_comps=None, equationP=None,
                               equationT=None, iterations=30, T_K_guess=1300, H2O_Liq=None):


    '''
    Solves simultaneous equations for temperature and pressure using
    clinopyroxene-lonly thermometers and barometers.


   Parameters
    -------

    cpx_comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    equationP: str
        | P_Put2008_eq32a (T-dependent)
        | P_Put2008_eq32b (T-dependent, H2O dependent)

    equationT: str
        | T_Put2008_eq32d (P-dependent)
        | T_Put2008_eq32d_subsol (P-dependent)

    H2O_Liq: float, int, series, optional
        Needed if you select P_Put2008_eq32b, which is H2O-dependent.

    Optional:


     iterations: int, default=30
         Number of iterations used to converge to solution.

     T_K_guess: int or float. Default is 1300 K
         Initial guess of temperature.


    Returns:
    -------
    panda.dataframe: Pressure in kbar, Temperature in K
    '''


    cpx_comps_c=cpx_comps.copy()

    if H2O_Liq is None:
        T_func = calculate_cpx_only_temp(cpx_comps=cpx_comps_c, equationT=equationT, P="Solve")
    else:
        T_func = calculate_cpx_only_temp(cpx_comps=cpx_comps_c, equationT=equationT, P="Solve", H2O_Liq=H2O_Liq)

    if H2O_Liq is None:
        P_func = calculate_cpx_only_press(
            cpx_comps=cpx_comps_c, equationP=equationP, T="Solve")
    else:
        P_func = calculate_cpx_only_press(
            cpx_comps=cpx_comps_c, equationP=equationP, H2O_Liq=H2O_Liq, T="Solve")


    if isinstance(P_func, pd.Series) and isinstance(T_func, partial):
        P_guess = P_func
        T_K_guess = T_func(P_guess)
    if isinstance(T_func, pd.Series) and isinstance(P_func, partial):
        T_K_guess = T_func
        P_guess = P_func(T_K_guess)
    if isinstance(T_func, pd.Series) and isinstance(P_func, pd.Series):
        T_K_guess = T_func
        P_guess = P_func

    if isinstance(P_func, partial) and isinstance(T_func, partial):

        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)




    PT_out = pd.DataFrame(
            data={'P_kbar_calc': P_guess, 'T_K_calc': T_K_guess})
    PT_out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return PT_out


