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

#Thermobar_dir=Path(__file__).parent
import joblib
# Things for machine learning onnx
from sklearn.preprocessing import StandardScaler
#from skl2onnx import convert_sklearn
#from skl2onnx.common.data_types import FloatTensorType

np.seterr(invalid="ignore")


from Thermobar.core import *
from Thermobar.Nimis_1999 import *

# Machine learning training scripts are in the src/Thermobar folder, both traditional and onnx.
# In this file, you can see there are 4 for the onnx ones
# There are then 3 for Petrelli on its own. Cpx-Liq, Cpx-only, and Cpx-only with water we were playing with.
# Then need to release a new machine learning version.


## Equations for Cpx-Liquid Barometry written as functions

def P_Put1996_eqP1(T, *, lnK_Jd_liq, Na_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Putirka (1996) EqP1
    :cite:`putirka1996thermobarometry`

    SEE=+-1.36 kbar (1996 paper, calibration dataset)

    Revised SEE by Putirka (2008)

    SEE=+-4.6 kbar (anhydrous)

    SEE=+-5.4 kbar (hydrous)

    '''
    return - 54.3 + 299 * T / 10 ** 4 + 36.4 * T * lnK_Jd_liq / \
        10 ** 4 + 367 * Na_Liq_cat_frac * Al_Liq_cat_frac


def P_Mas2013_eqPalk1(T, *, lnK_Jd_liq,
                      Na_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid barometer of Putirka (1996) EqP1 by Masotta et al. (2013) for alkaline melts
    :cite:`masotta2013clinopyroxene`

    SEE=+-1.71 kbar

    '''
    return - 8.83763538032322 + 79.0497715060127 * T / 10 ** 4 + 11.6474409456619 * \
        T * lnK_Jd_liq / 10 ** 4 + 8.63312603541135 * \
        Na_Liq_cat_frac * Al_Liq_cat_frac


def P_Put1996_eqP2(T, *, lnK_Jd_liq, Na_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Putirka (1996) EqP2
    :cite:`putirka1996thermobarometry`

    SEE=+-1.51 kbar (1996 paper, calibration dataset)



    '''
    return (-50.7 + 394 * T / 10 ** 4 + 36.4 * T * lnK_Jd_liq / 10 ** 4 -
            20 * (T / 10**4) * np.log(1 / (Na_Liq_cat_frac.astype(float) * Al_Liq_cat_frac.astype(float))))


def P_Mas2013_eqPalk2(T, *, lnK_Jd_liq,
                      Na_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid barometer of Putirka (1996)
    EqP2 by Masotta et al. (2013) for alkaline melts.
    Cite :cite:`masotta2013clinopyroxene`

    SEE=+-1.70 kbar

    '''
    return (-6.28332277837751 + 38.1796219610587 * T / 10 ** 4
    + 9.42105316105869 * T * lnK_Jd_liq /
    10 ** 4 + 6.15641875019196 * (T / 10**4) * np.log(1 / (Na_Liq_cat_frac.astype(float)
    * Al_Liq_cat_frac.astype(float))))


def P_Put2003(T, *, lnK_Jd_liq, Ca_Liq_cat_frac,
              Si_Liq_cat_frac, Mg_Number_Liq_NoFe3):
    '''
    Clinopyroxene-liquid barometer of Putirka (2003) Eq1.
    Cite :cite:`putirka2003new`

    SEE (2003) paper = +-1.7 kbar

    Stated errors Putirka (2008)
    SEE=+-4.8 kbar (anhydrous)

    SEE=+-5.0 kbar (hydrous)

    '''
    return (- 88.3 + 0.00282 * T * lnK_Jd_liq + 0.0219 * T
    - 25.1 * np.log(Ca_Liq_cat_frac.astype(float) *
    Si_Liq_cat_frac.astype(float))
    + 12.4 * np.log(Ca_Liq_cat_frac.astype(float)) + 7.03 * Mg_Number_Liq_NoFe3)


def P_Put2008_eq30(T, *, lnK_Jd_liq, Fet_Liq_cat_frac, Mg_Liq_cat_frac,
                   DiHd_2003, Mg_Number_Liq_NoFe3, Na_Liq_cat_frac, K_Liq_cat_frac, H2O_Liq):
    '''
    Clinopyroxene-liquid barometer of Putirka (2008) Eq30.
    Cite :cite:`putirka2008thermometers`

    SEE=+-3.6 kbar (all data)
    SEE=+-1.6 kbar (calibration data)

    '''
    return (-48.7 + 271.3 * (T / 10**4) + 31.96 * (T / 10**4) * lnK_Jd_liq - 8.2 * np.log(Fet_Liq_cat_frac.astype(float))
            + 4.6 * np.log(Mg_Liq_cat_frac.astype(float)) - 0.96 * np.log(K_Liq_cat_frac.astype(float))
            - 2.2 * np.log(DiHd_2003.astype(float)) - 31 * Mg_Number_Liq_NoFe3 + 56.2 * (Na_Liq_cat_frac + K_Liq_cat_frac) + 0.76 * H2O_Liq)


def P_Put2008_eq31(T, *, lnK_Jd_liq, Ca_Liq_cat_frac, Na_Liq_cat_frac, K_Liq_cat_frac, Si_Liq_cat_frac, Mg_Liq_cat_frac,
                   Fet_Liq_cat_frac, DiHd_2003, EnFs, Al_Cpx_cat_6ox, H2O_Liq):
    '''
    Clinopyroxene-liquid barometer of Putirka (2008) Eq31
    :cite:`putirka2008thermometers`

    SEE=+-2.9 kbar (all data)
    '''
    return (-40.73 + 358 * (T / 10**4) + 21.7 * (T / 10**4) * lnK_Jd_liq - 106 * Ca_Liq_cat_frac - 166 * (Na_Liq_cat_frac + K_Liq_cat_frac)**2
            - 50.2 * Si_Liq_cat_frac *
            (Mg_Liq_cat_frac + Fet_Liq_cat_frac) -
            3.2 * np.log(DiHd_2003.astype(float)) - 2.2 * np.log(EnFs.astype(float))
            + 0.86 * np.log(Al_Cpx_cat_6ox.astype(float)) + 0.4 * H2O_Liq)


def P_Put2008_eq32c(T, *, Fet_Liq_cat_frac, CaTs, H2O_Liq, Ca_Liq_cat_frac,
                    Si_Liq_cat_frac, Al_Cpx_cat_6ox, Al_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Putirka (2008) Eq32c based on partitioning of Al between cpx and liquid
    :cite:`putirka2008thermometers`

    SEE=+-5 kbar (all data)
    SEE=+-1.5 kbar (calibration data)
    '''
    return (-57.9 + 0.0475 * (T) - 40.6 * Fet_Liq_cat_frac - 47.7 * CaTs + 0.67 * H2O_Liq -
            153 * Ca_Liq_cat_frac * Si_Liq_cat_frac + 6.89 * (Al_Cpx_cat_6ox / Al_Liq_cat_frac))


def P_Mas2013_eqalk32c(T, *, Fet_Liq_cat_frac, CaTs, H2O_Liq, Ca_Liq_cat_frac,
                       Si_Liq_cat_frac, Al_Cpx_cat_6ox, Al_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid barometer of Putirka (2008) Eq32c by Masotta et al. (2013) for alkaline melts
    :cite:`masotta2013clinopyroxene`

    SEE=+-1.67 kbar
    '''
    return (-16.3446543551989 + 0.0141435837038975 * (T)
    - 12.3909508275802 * Fet_Liq_cat_frac - 9.19220692402416 * CaTs
    + 0.214041799294945 * H2O_Liq + 38.734045560859 * Ca_Liq_cat_frac * Si_Liq_cat_frac
    + 1.5944198112849 * (Al_Cpx_cat_6ox / Al_Liq_cat_frac))


def P_Mas2013_Palk2012(T=None, *, lnK_Jd_liq, H2O_Liq,
                       Na_Liq_cat_frac, K_Liq_cat_frac, Kd_Fe_Mg_Fet):
    '''
    Clinopyroxene-liquid barometer of Masotta et al. (2013) for alkaline melts
    :cite:`masotta2013clinopyroxene`

    SEE=+-1.15 kbar
    '''
    return (-3.88903951262765 + 0.277651046511846 * np.exp(lnK_Jd_liq)
    + 0.0740292491471828 * H2O_Liq + 5.00912129248619 * (Na_Liq_cat_frac)
    / (Na_Liq_cat_frac + K_Liq_cat_frac) + 6.39451438456963 * Kd_Fe_Mg_Fet)


def P_Wieser2021_H2O_indep(T=None, *, MgO_Liq, Ca_Liq_cat_frac, lnK_Jd_liq, Jd,
                           CaTs, Na_Liq_cat_frac, Fet_Liq_cat_frac, Al_Cpx_cat_6ox, Mg_Number_Liq_NoFe3):
    return (3.204423282096874 + 1.21811674 * MgO_Liq - 168.80037558 * Ca_Liq_cat_frac
    + 1.49243994 * lnK_Jd_liq + 58.22419473 * Jd + 76.11682662 * CaTs
    - 29.27503912 * Na_Liq_cat_frac + 33.34059394 * Fet_Liq_cat_frac
    - 8.50428995 * Al_Cpx_cat_6ox + 4.98260164 * Mg_Number_Liq_NoFe3)


def P_Neave2017(T, *, lnK_Jd_liq, DiHd_2003, Al_Liq_cat_frac,
                Na_Liq_cat_frac, K_Liq_cat_frac):
    '''
    Clinopyroxene-liquid barometer of Neave and Putirka (2017)
    :cite:`neave2017new`

    SEE=+-1.4 kbar
    '''
    return (-26.2712 + 39.16138 * T * lnK_Jd_liq / 10**4 - 4.21676 * np.log(DiHd_2003.astype(float))
            + 78.43463 * Al_Liq_cat_frac + 393.8126 * (Na_Liq_cat_frac + K_Liq_cat_frac)**2)

def P_Petrelli2020_Cpx_Liq(T=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  barometer of Petrelli et al. (2021) based on
    Machine Learning.
    :cite:`petrelli2020machine`

    SEE==+-2.6 kbar
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'H2O_Liq': cpx_liq_combo['H2O_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],
    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')



    Thermobar_dir=Path(Thermobar_onnx.__file__).parent


    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Liq_sklearn_1_3.pkl', 'rb') as f:
        scaler_P2020_Cpx_Liq=load(f)

    with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Liq_sklearn_1_3.pkl', 'rb') as f:
        ETR_Press_P2020_Cpx_Liq=joblib.load(f)


    x_test_scaled=scaler_P2020_Cpx_Liq.transform(x_test)
    Pred_P_kbar=ETR_Press_P2020_Cpx_Liq.predict(x_test_scaled)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_Liq)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)


    return df_stats

def P_Jorgenson2022_Cpx_Liq_Norm(T=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  barometer of Jorgenson et al. (2022) based on
    Machine Learning. Normalizes, unlike published model.
    :cite:`jorgenson2021machine`

    SEE==+-2.7 kbar
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],
    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'scaler_Jorg21_Cpx_Liq_April24.pkl', 'rb') as f:
        scaler_J22_Cpx_Liq=load(f)

    with open(Thermobar_dir/'ETR_Press_Jorg21_Cpx_Liq_April24.pkl', 'rb') as f:
        ETR_Press_J22_Cpx_Liq=joblib.load(f)


    x_test_scaled=scaler_J22_Cpx_Liq.transform(x_test)
    Pred_P_kbar=ETR_Press_J22_Cpx_Liq.predict(x_test_scaled)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_J22_Cpx_Liq)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)


    return df_stats

def P_Jorgenson2022_Cpx_Liq(T=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  barometer of Jorgenson et al. (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-2.7 kbar
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],
    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'ETR_Press_Jorg21_Cpx_Liq_NotNorm_sklearn_1_3.pkl', 'rb') as f:
        ETR_Press_J22_Cpx_Liq=joblib.load(f)


    Pred_P_kbar=ETR_Press_J22_Cpx_Liq.predict(x_test)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Press_J22_Cpx_Liq)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)


    return df_stats

def P_Jorgenson2022_Cpx_Liq_onnx(T=None, *, cpx_comps=None,
liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  barometer of Jorgenson et al. (2022) based on
    Machine Learning. Uses onnx, so doesnt return voting
    :cite:`jorgenson2021machine`

    SEE==+-2.7 kbar
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],
    })


    x_test=Cpx_Liq_ML_in.values



    #sess = rt.InferenceSession(path+'/'+'Jorg21_Cpx_Liq_Press.onnx')
    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    import onnxruntime as rt

    # path=Path(Thermobar_onnx.__file__).parent
    # sess =  rt.InferenceSession(str(path/"Jorg21_Cpx_Liq_Press.onnx"))
    #
    path = Path(Thermobar_onnx.__file__).parent
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Jorg21_Cpx_Liq_Press.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_P_kbar = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    P_kbar=pd.Series(Pred_P_kbar[:, 0])
    return P_kbar

    return df_stats





def P_Petrelli2020_Cpx_Liq_onnx(T=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  barometer of Petrelli et al. (2021) based on
    Machine Learning. Uses onnx, so wont return voting.
    :cite:`petrelli2020machine`

    SEE==+-2.6 kbar
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'H2O_Liq': cpx_liq_combo['H2O_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],
    })


    x_test=Cpx_Liq_ML_in.values




    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    path=Path(Thermobar_onnx.__file__).parent
    import onnxruntime as rt
    #sess =  rt.InferenceSession(str(path/"Petrelli2020_Cpx_Liq_Press.onnx"))

    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Petrelli2020_Cpx_Liq_Press.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)




    #sess = rt.InferenceSession(Petrelli2020_Cpx_Liq_Temp.onnx)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_P_kbar = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    P_kbar=pd.Series(Pred_P_kbar[:, 0])
    return P_kbar



## Equations for Cpx-Liquid Thermometry written as functions


def T_Put1996_eqT1(P=None, *, lnK_Jd_DiHd_liq_1996,
                   Mg_Number_Liq_NoFe3, Ca_Liq_cat_frac):
    '''
    Clinopyroxene-liquid thermometer of Putirka (1996) EqT1 (pressure-independent)
    :cite:`putirka1996thermobarometry`

    SEE=+-26.8 K (1996 paper, calibration dataset)

    '''
    return (10 ** 4 / (6.73 - 0.26 * lnK_Jd_DiHd_liq_1996 - 0.86 * np.log(Mg_Number_Liq_NoFe3.astype(float))
                       + 0.52 * np.log(Ca_Liq_cat_frac.astype(float))))


def T_Mas2013_eqTalk1(P=None, *, lnK_Jd_DiHd_liq_1996,
                      Mg_Number_Liq_NoFe3, Ca_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid thermometer of Putirka (1996) EqT1
    by Masotta et al. (2013) for alkaline melts.
    :cite:`masotta2013clinopyroxene`

    SEE=+-31.6°C
    '''
    return (10 ** 4 / (6.7423126317975 - 0.023236627691972 * lnK_Jd_DiHd_liq_1996 -
            0.68839419999351 * np.log(Mg_Number_Liq_NoFe3.astype(float)) - 0.153193056441978 * np.log(Ca_Liq_cat_frac.astype(float))))


def T_Put1996_eqT2(P, *, lnK_Jd_DiHd_liq_1996,
                   Mg_Number_Liq_NoFe3, Ca_Liq_cat_frac):
    '''
    Clinopyroxene-liquid thermometer of Putirka (1996) EqT2 (pressure-dependent)
    :cite:`putirka1996thermobarometry`

    SEE=+-23.8 K (1996 paper, calibration dataset)


    '''
    return (10 ** 4 / (6.59 - 0.16 * lnK_Jd_DiHd_liq_1996 - 0.65 * np.log(Mg_Number_Liq_NoFe3.astype(float))
                       + 0.23 * np.log(Ca_Liq_cat_frac.astype(float)) - 0.02 * P))


def T_Mas2013_eqTalk2(P, *, lnK_Jd_DiHd_liq_1996,
                      Mg_Number_Liq_NoFe3, Ca_Liq_cat_frac):
    '''
    Recalibration of the clinopyroxene-liquid thermometer of Putirka (1996),
    EqT2 by Masotta et al. (2013) for alkaline melts
    :cite:`masotta2013clinopyroxene`

    SEE=+-31.2°C
    '''
    return (10 ** 4 / (6.52396326315485 - 0.0396542787609402 * lnK_Jd_DiHd_liq_1996 - 0.680638985726502 *
            np.log(Mg_Number_Liq_NoFe3.astype(float)) - 0.145757123805013 * np.log(Ca_Liq_cat_frac.astype(float)) + 0.0790582631912926 * P))


def T_Put1999(P, *, Mg_Liq_cat_frac, Fet_Liq_cat_frac,
              Ca_Liq_cat_frac, Si_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Equation in Keith's Cpx-Liquid spreadsheet labelled "Putirka 1999".
    :cite:`putirka1999clinopyroxene`


    '''

    return (10 ** 4 / (3.12 - 0.0259 * P - 0.37 * np.log(Mg_Liq_cat_frac.astype(float) / (Mg_Liq_cat_frac.astype(float) + Fet_Liq_cat_frac.astype(float)))
                       + 0.47 * np.log(Ca_Liq_cat_frac.astype(float) * (Mg_Liq_cat_frac.astype(float) +
                        Fet_Liq_cat_frac.astype(float)) * (Si_Liq_cat_frac.astype(float))**2)
                       - 0.78 * np.log((Mg_Liq_cat_frac.astype(float) + Fet_Liq_cat_frac.astype(float))
                                       ** 2 * (Si_Liq_cat_frac.astype(float))**2)
                       - 0.34 * np.log(Ca_Liq_cat_frac.astype(float) * (Al_Liq_cat_frac.astype(float))**2 * Si_Liq_cat_frac.astype(float))))


def T_Put2003(P, *, lnK_Jd_DiHd_liq_2003, Mg_Number_Liq_NoFe3,
              Na_Liq_cat_frac, Si_Liq_cat_frac, Jd):
    '''
    Clinopyroxene-liquid thermometer of Putirka (2003)
    :cite:`putirka2003new`

    SEE=+-33 K (1996 paper, calibration dataset)


    '''
    return (10 ** 4 / (4.6 - 0.437 * lnK_Jd_DiHd_liq_2003 - 0.654 * np.log(Mg_Number_Liq_NoFe3.astype(float))
    - 0.326 * np.log(Na_Liq_cat_frac.astype(float)) -0.92 * np.log(Si_Liq_cat_frac.astype(float))
    + 0.274 * np.log(Jd.astype(float)) - 0.00632 * P))


def T_Put2008_eq33(P, *, H2O_Liq, Mg_Number_Liq_NoFe3, Ca_Liq_cat_frac, Si_Liq_cat_frac,
                   Ti_Liq_cat_frac, Na_Liq_cat_frac, K_Liq_cat_frac, EnFs, lnK_Jd_DiHd_liq_2003):
    '''
    Clinopyroxene-liquid  thermometer of Putirka (2008) Eq 33.
    :cite:`putirka2008thermometers`

    SEE=+-45°C (all data)
    '''
    return (10 ** 4 / (7.53 + 0.07 * H2O_Liq - 1.1 * Mg_Number_Liq_NoFe3
    - 14.9 * (Ca_Liq_cat_frac * Si_Liq_cat_frac) -
    0.08 * np.log(Ti_Liq_cat_frac.astype(float))
    - 3.62 * (Na_Liq_cat_frac + K_Liq_cat_frac) - 0.18 * np.log(EnFs.astype(float))
    - 0.14 * lnK_Jd_DiHd_liq_2003 - 0.027 * P))


def T_Mas2013_eqalk33(P, *, H2O_Liq, Mg_Number_Liq_NoFe3, Ca_Liq_cat_frac, Si_Liq_cat_frac,
                      Ti_Liq_cat_frac, Na_Liq_cat_frac, K_Liq_cat_frac, EnFs, lnK_Jd_DiHd_liq_2003):
    '''
    Recalibration of the clinopyroxene-liquid thermometer of Putirka (2008)
    Eq 33 by Masotta et al. (2013) for alkaline melts
    :cite:`masotta2013clinopyroxene`

    SEE=+-24°C
    '''
    return (10 ** 4 / (6.80728851520843 + 0.0500993963259582 * H2O_Liq
    - 1.91449550102791 * Mg_Number_Liq_NoFe3
    - 25.0429785936576 * (Ca_Liq_cat_frac * Si_Liq_cat_frac) -
    0.304200646919069 * np.log(Ti_Liq_cat_frac.astype(float))
    + 2.25444204541222 * (Na_Liq_cat_frac + K_Liq_cat_frac)
    - 0.021072700182831 * np.log(EnFs.astype(float))
    + 0.00268252978603778 * lnK_Jd_DiHd_liq_2003
    + 0.0614725514133312 * P))


# ones without P in the function
def T_Mas2013_Talk2012(P=None, *, H2O_Liq, Kd_Fe_Mg_Fet, lnK_Jd_DiHd_liq_2003,
Mg_Number_Liq_NoFe3, DiHd_2003, Na_Liq_cat_frac, K_Liq_cat_frac,
Ti_Liq_cat_frac, lnK_Jd_liq, Ca_Liq_cat_frac, Si_Liq_cat_frac):
    '''
    Clinopyroxene-liquid thermometer of Masotta et al. (2013) for alkaline melts
    :cite:`masotta2013clinopyroxene`

    SEE=+-18.2C
    '''
    return (10**4 / (2.90815635794002 - 0.400827676578132 * lnK_Jd_DiHd_liq_2003
        + 0.0375720784518263 * H2O_Liq - 1.6383282971929 *
        (Mg_Number_Liq_NoFe3 / DiHd_2003) + 1.01129776262724 *
        ((Na_Liq_cat_frac) / (Na_Liq_cat_frac + K_Liq_cat_frac))
        - 0.21766733252629 * np.log(Ti_Liq_cat_frac.astype(float)) + 0.466149612620683
        * lnK_Jd_liq + 1.61626798988239 * Kd_Fe_Mg_Fet + 23.3855047471225 * (Ca_Liq_cat_frac * Si_Liq_cat_frac)))


def T_Brug2019(P=None, *, CaTs, DiHd_2003, Si_Liq_cat_frac, Ti_Liq_cat_frac,
Fet_Liq_cat_frac, Mg_Liq_cat_frac, Ca_Liq_cat_frac, K_Liq_cat_frac):
    '''
    Clinopyroxene-liquid  thermometer of Brugmann and Till (2019) for evolved systems,
    (Cpx Mg#>64, Al2O3 Cpx<7 wt%, SiO2_Liq>70 wt%)
    :cite:`brugman2019low`

    SEE==+-20°C
    '''
    return (273.15 + 300 * (-1.8946098 - 0.6010197 * CaTs - 0.1856423 * DiHd_2003
+ 4.71248858 * Si_Liq_cat_frac + 77.5861878 * Ti_Liq_cat_frac +
10.8503727 * Fet_Liq_cat_frac + 33.6303471 * Mg_Liq_cat_frac
+ 15.4532888 * Ca_Liq_cat_frac + 15.6390115 * K_Liq_cat_frac))

# Technically liquid only
def T_Put2008_eq34_cpx_sat(P, *, H2O_Liq, Ca_Liq_cat_frac, Si_Liq_cat_frac, Mg_Liq_cat_frac):
    '''
    Liquid-only thermometer- temperature at which a liquid is saturated in clinopyroxene (for a given P). Equation 34 of Putirka et al. (2008)
    '''
    return (10 ** 4 / (6.39 + 0.076 * H2O_Liq - 5.55 * (Ca_Liq_cat_frac * Si_Liq_cat_frac)
            - 0.386 * np.log(Mg_Liq_cat_frac) - 0.046 * P + 2.2 * (10 ** (-4)) * P**2))



def T_Petrelli2020_Cpx_Liq(P=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  thermometer of Petrelli et al. (2021) based on
    Machine Learning.
    :cite:`petrelli2020machine`

    SEE==+-40°C
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'H2O_Liq': cpx_liq_combo['H2O_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],

    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    # Old version using pickles
    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Liq_sklearn_1_3.pkl', 'rb') as f:
        scaler_P2020_Cpx_Liq=load(f)

    with open(Thermobar_dir/'ETR_Temp_Petrelli2020_Cpx_Liq_sklearn_1_3.pkl', 'rb') as f:
        ETR_Temp_P2020_Cpx_Liq=joblib.load(f)

    x_test_scaled=scaler_P2020_Cpx_Liq.transform(x_test)
    Pred_T_K=ETR_Temp_P2020_Cpx_Liq.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_P2020_Cpx_Liq)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)


    return df_stats


def T_Jorgenson2022_Cpx_Liq_Norm(P=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  thermometer of Jorgenson et al. (2022) based on
    Machine Learning. Normalizes, unlike Jorgenson model.
    :cite:`jorgenson2021machine`

    SEE==+-44.9°C
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()


        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],

    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    # Old version using pickles
    with open(Thermobar_dir/'scaler_Jorg21_Cpx_Liq_April24.pkl', 'rb') as f:
        scaler_J22_Cpx_Liq=load(f)

    with open(Thermobar_dir/'ETR_Temp_Jorg21_Cpx_Liq_April24.pkl', 'rb') as f:
        ETR_Temp_J22_Cpx_Liq=joblib.load(f)

    x_test_scaled=scaler_J22_Cpx_Liq.transform(x_test)
    Pred_T_K=ETR_Temp_J22_Cpx_Liq.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_J22_Cpx_Liq)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)


    return df_stats

def T_Jorgenson2022_Cpx_Liq(P=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  thermometer of Jorgenson et al. (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-44.9°C
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],

    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'ETR_Temp_Jorg21_Cpx_Liq_NotNorm_sklearn_1_3.pkl', 'rb') as f:
        ETR_Temp_J22_Cpx_Liq=joblib.load(f)


    Pred_T_K=ETR_Temp_J22_Cpx_Liq.predict(x_test)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Temp_J22_Cpx_Liq)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)


    return df_stats

def T_Jorgenson2022_Cpx_Liq_onnx(P=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  thermometer of Jorgenson et al. (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-44.9°C
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],

    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    import onnxruntime as rt
    path=Path(Thermobar_onnx.__file__).parent
    #sess =  rt.InferenceSession(str(path/"Jorg21_Cpx_Liq_Temp.onnx"))
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Jorg21_Cpx_Liq_Temp.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)


    #sess = rt.InferenceSession(str(Thermobar_dir/"Jorg21_Cpx_Liq_Temp.onnx"))
    #sess = rt.InferenceSession(Petrelli2020_Cpx_Liq_Temp.onnx)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_T_K = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    #df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Temp_P2020_Cpx_Liq)


    T_K=pd.Series(Pred_T_K[:, 0])
    return T_K

def T_Petrelli2020_Cpx_Liq_onnx(P=None, *, cpx_comps=None, liq_comps=None, meltmatch=None):
    '''
    Clinopyroxene-liquid  thermometer of Petrelli et al. (2021) based on
    Machine Learning. Use onnx, so will always return the same answer, but can't do voting,
    unlike T_Petrelli2020_Cpx_Liq.
    :cite:`petrelli2020machine`

    SEE==+-40°C
    '''
    if meltmatch is None:
        cpx_test=cpx_comps.copy()
        liq_test=liq_comps.copy()
        cpx_liq_combo=pd.concat([cpx_test, liq_test], axis=1)

    if meltmatch is not None:
        cpx_liq_combo=meltmatch



    Cpx_Liq_ML_in=pd.DataFrame(data={
                                'SiO2_Liq': cpx_liq_combo['SiO2_Liq'],
                                'TiO2_Liq': cpx_liq_combo['TiO2_Liq'],
                                'Al2O3_Liq': cpx_liq_combo['Al2O3_Liq'],
                                'FeOt_Liq': cpx_liq_combo['FeOt_Liq'],
                                'MnO_Liq': cpx_liq_combo['MnO_Liq'],
                                'MgO_Liq': cpx_liq_combo['MgO_Liq'],
                                'CaO_Liq': cpx_liq_combo['CaO_Liq'],
                                'Na2O_Liq': cpx_liq_combo['Na2O_Liq'],
                                'K2O_Liq': cpx_liq_combo['K2O_Liq'],
                                'Cr2O3_Liq': cpx_liq_combo['Cr2O3_Liq'],
                                'P2O5_Liq': cpx_liq_combo['P2O5_Liq'],
                                'H2O_Liq': cpx_liq_combo['H2O_Liq'],
                                'SiO2_Cpx': cpx_liq_combo['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_liq_combo['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_liq_combo['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_liq_combo['FeOt_Cpx'],
                                'MnO_Cpx': cpx_liq_combo['MnO_Cpx'],
                                'MgO_Cpx': cpx_liq_combo['MgO_Cpx'],
                                'CaO_Cpx': cpx_liq_combo['CaO_Cpx'],
                                'Na2O_Cpx': cpx_liq_combo['Na2O_Cpx'],
                                'K2O_Cpx': cpx_liq_combo['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_liq_combo['Cr2O3_Cpx'],

    })


    x_test=Cpx_Liq_ML_in.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    import onnxruntime as rt
    path=Path(Thermobar_onnx.__file__).parent
    #sess =  rt.InferenceSession(str(path/"Petrelli2020_Cpx_Liq_Temp.onnx"))
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Petrelli2020_Cpx_Liq_Temp.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)


    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_T_K = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]


    T_K=pd.Series(Pred_T_K[:, 0])
    return T_K



## Clinopyroxene-only pressure equations

def P_Wang2021_eq1(T=None, *, Al_VI_cat_6ox, Si_Cpx_cat_6ox, Ti_Cpx_cat_6ox, Cr_Cpx_cat_6ox,
Fet_Cpx_cat_6ox, Mn_Cpx_cat_6ox, Mg_Cpx_cat_6ox, Na_Cpx_cat_6ox, K_Cpx_cat_6ox, Ca_Cpx_cat_6ox):
    '''
    Clinopyroxene-only barometer of Wang et al. (2021) equation 1
    :cite:`wang2021new`

    SEE=1.66 kbar

    '''
    NCT=(1.4844*Al_VI_cat_6ox/(1.4844*Al_VI_cat_6ox+7.7408*Ti_Cpx_cat_6ox
    +1.1675*Cr_Cpx_cat_6ox+1.0604*Fet_Cpx_cat_6ox+0.0387*Mn_Cpx_cat_6ox-0.0628*Mg_Cpx_cat_6ox))



    return (-7.9509*NCT*np.log(Al_VI_cat_6ox.astype(float))+0.6492*Si_Cpx_cat_6ox-5.9522*Fet_Cpx_cat_6ox-11.1942*Mg_Cpx_cat_6ox-24.2802*Ca_Cpx_cat_6ox+108.663*Na_Cpx_cat_6ox+25.0019)







def P_Put2008_eq32a(T, *, Mg_Cpx_cat_6ox, Na_Cpx_cat_6ox,
                    Al_VI_cat_6ox, DiHd_2003, EnFs):
    '''
    Clinopyroxene-only barometer of Putirka (2008) Eq32a
    :cite:`putirka2008thermometers`

    SEE=+-3.1 kbar (anhydrous)

    '''
    return (3205 - 5.62 * Mg_Cpx_cat_6ox + 83.2 * Na_Cpx_cat_6ox + 68.2 * DiHd_2003
    + 2.52 * np.log(Al_VI_cat_6ox.astype(float)) - 51.1 * DiHd_2003**2 + 34.8 * EnFs**2
    + 0.384 * T - 518 * np.log(T))


def P_Put2008_eq32b(T, *, H2O_Liq, Ca_Cpx_cat_6ox, Mn_Cpx_cat_6ox, Na_Cpx_cat_6ox,
Al_Cpx_cat_6ox, K_Cpx_cat_6ox, Al_IV_cat_6ox, Ti_Cpx_cat_6ox, Cr_Cpx_cat_6ox,
                    Al_VI_cat_6ox, Fet_Cpx_cat_6ox, DiHd_1996, Mg_Cpx_cat_6ox, Jd):
    '''
    Clinopyroxene-only barometer of Putirka (2008) Eq32b. Unlike 32a, requires H2O_Liq to be specified.
    :cite:`putirka2008thermometers`

    SEE=+-2.6 kbar (anhydrous)

    '''

    CNM = Ca_Cpx_cat_6ox + Na_Cpx_cat_6ox + Mn_Cpx_cat_6ox  # This is GQ
    Lindley_Fe3_Cpx = Na_Cpx_cat_6ox + Al_IV_cat_6ox - Al_VI_cat_6ox - \
        2 * Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox  # This is cell FR
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    R3_plus = Al_VI_cat_6ox + Ti_Cpx_cat_6ox + \
        Cr_Cpx_cat_6ox + Lindley_Fe3_Cpx  # This is cell GR
    M1_M2_KD_Fe_Mg = np.exp(0.238 * R3_plus + 0.289 * CNM - 2.3315)
    M1_M1_Fe2_tot = Fet_Cpx_cat_6ox - Lindley_Fe3_Cpx  # GV15
    b = (M1_M2_KD_Fe_Mg * Mg_Cpx_cat_6ox) - \
        (M1_M2_KD_Fe_Mg * (1 - CNM)) + M1_M1_Fe2_tot + (1 - CNM)  # GX
    a_Kd_1 = 1 - M1_M2_KD_Fe_Mg  # GW
    c = -M1_M1_Fe2_tot * (1 - CNM)  # GY
    Fe_M2 = (-b + np.sqrt(b**2 - 4 * a_Kd_1 * c)) / 2 * a_Kd_1
    Mg_M2 = 1 - CNM - Fe_M2
    return (1458 + 0.197 * (T) - 241 * np.log(T) +
    0.453 * H2O_Liq + 55.5 * Al_VI_cat_6ox +
    8.05 * Fet_Cpx_cat_6ox - 277 * K_Cpx_cat_6ox + 18 * Jd
    + 44.1 * DiHd_1996 + 2.2 * np.log(Jd.astype(float))
    - 27.7 * Al_Cpx_cat_6ox**2 + 97.3 * Fe_M2**2
    + 30.7 * Mg_M2**2 - 27.6 * DiHd_1996**2)

def P_Petrelli2020_Cpx_only(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer of Petrelli et al. (2021) based on
    Machine Learning.
    :cite:`petrelli2020machine`

    SEE==+-3.1 kbar
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent


    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_sklearn_1_3.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Only_sklearn_1_3.pkl', 'rb') as f:
        ETR_Press_P2020_Cpx_only=joblib.load(f)


    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_P_kbar=ETR_Press_P2020_Cpx_only.predict(x_test_scaled)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_only)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)

    return df_stats


def P_Petrelli2020_Cpx_only_onnx(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer of Petrelli et al. (2021) based on
    Machine Learning. Uses onnx for consistency, so dont get voting
    :cite:`petrelli2020machine`

    SEE==+-3.1 kbar
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    import onnxruntime as rt
    path=Path(Thermobar_onnx.__file__).parent
    #sess =  rt.InferenceSession(str(path/"Petrelli2020_Cpx_only_Press.onnx"))
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Petrelli2020_Cpx_only_Press.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)


    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_P_kbar = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]


    P_kbar=pd.Series(Pred_P_kbar[:, 0])
    return P_kbar




def P_Jorgenson2022_Cpx_only_Norm(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer of Jorgenson et al. (2022) based on
    Machine Learning. Normalizes inputs unlike published model
    :cite:`jorgenson2021machine`

    SEE==+-3.2 kbar
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values


    with open(Thermobar_dir/'scaler_Jorg21_Cpx_only_April24.pkl', 'rb') as f:
        scaler_J21_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Press_Jorg21_Cpx_only_April24.pkl', 'rb') as f:
        ETR_Press_J21_Cpx_only=joblib.load(f)


    x_test_scaled=scaler_J21_Cpx_only.transform(x_test)
    Pred_P_kbar=ETR_Press_J21_Cpx_only.predict(x_test_scaled)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_J21_Cpx_only)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)

    return df_stats

def P_Jorgenson2022_Cpx_only(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer of Jorgenson et al. (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-3.2 kbar
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'ETR_Press_Jorg21_Cpx_only_NotNorm_sklearn_1_3.pkl', 'rb') as f:
        ETR_Press_J21_Cpx_only=joblib.load(f)



    Pred_P_kbar=ETR_Press_J21_Cpx_only.predict(x_test)

    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Press_J21_Cpx_only)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)

    return df_stats

def P_Jorgenson2022_Cpx_only_onnx(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer of Jorgenson et al. (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-3.2 kbar
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values


    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    path=Path(Thermobar_onnx.__file__).parent
    import onnxruntime as rt
    #sess =  rt.InferenceSession(str(path/"Jorg21_Cpx_only_Press.onnx"))

    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Jorg21_Cpx_only_Press.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_P_kbar = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    P_kbar=pd.Series(Pred_P_kbar[:, 0])

    return P_kbar

def P_Petrelli2020_Cpx_only_withH2O(T=None, *, cpx_comps):
    '''
    Clinopyroxene-only  barometer following the Machine learning approach of
    Petrelli et al. (2021), but including the H2O content of the liquid while training the model.

    :cite:`petrelli2020machine` and cite Thermobar.
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],
                                'H2O_Liq':cpx_comps['H2O_Liq'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_H2O_Jan22.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Only_H2O_Jan22.pkl', 'rb') as f:
        ETR_Press_P2020_Cpx_only=joblib.load(f)



    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_P_kbar=ETR_Press_P2020_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_only)
    df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)

    return df_stats


# def P_Petrelli2020_Cpx_only_noCr(T=None, *, cpx_comps):
#     '''
#     Clinopyroxene-only  barometer following the Machine learning approach of
#     Petrelli et al. (2021),
#     but including the H2O content of the liquid while training the model.
#     '''
#     Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
#                                 'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
#                                 'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
#                                 'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
#                                 'MnO_Cpx': cpx_comps['MnO_Cpx'],
#                                 'MgO_Cpx': cpx_comps['MgO_Cpx'],
#                                 'CaO_Cpx': cpx_comps['CaO_Cpx'],
#                                 'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
#
#
#     })
#     x_test=Cpx_test_noID_noT.values
#
#
#     with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_noCr.pkl', 'rb') as f:
#         scaler_P2020_Cpx_only=load(f)
#
#     with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Only_noCr.pkl', 'rb') as f:
#         ETR_Press_P2020_Cpx_only=load(f)
#
#
#
#     x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
#     Pred_P_kbar=ETR_Press_P2020_Cpx_only.predict(x_test_scaled)
#     df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_only)
#     df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)
#
#     return df_stats
#
# def P_Petrelli2020_Cpx_only_components(T=None, *, cpx_comps):
#     '''
#     Clinopyroxene-only  barometer following the Machine learning approach of
#     Petrelli et al. (2021),
#     but including the H2O content of the liquid while training the model.
#     '''
#     cpx_test=cpx_comps.copy()
#     Cpx_compnts_test=calculate_clinopyroxene_components(cpx_comps)
#     Cpx_test_noID_noT=pd.DataFrame(data={'Mg_Cpx_cat_6ox': Cpx_compnts_test['Mg_Cpx_cat_6ox'],
#                                       'Na_Cpx_cat_6ox': Cpx_compnts_test['Na_Cpx_cat_6ox'],
#                                       'Al_VI_cat_6ox': Cpx_compnts_test['Al_VI_cat_6ox'],
#                                        'DiHd_2003': Cpx_compnts_test['DiHd_2003'],
#                                       'EnFs': Cpx_compnts_test['EnFs'],
#                                                                            'Jd': Cpx_compnts_test['Jd'],
#                                       'Ca_Cpx_cat_6ox': Cpx_compnts_test['Ca_Cpx_cat_6ox'],
#                                       'Fet_Cpx_cat_6ox': Cpx_compnts_test['Fet_Cpx_cat_6ox'],
#                                         'Cr_Cpx_cat_6ox': Cpx_compnts_test['Cr_Cpx_cat_6ox'],
#                                        'Ti_Cpx_cat_6ox': Cpx_compnts_test['Ti_Cpx_cat_6ox'],
#                                        'Mn_Cpx_cat_6ox': Cpx_compnts_test['Mn_Cpx_cat_6ox'],
#                                      })
#     x_test=Cpx_test_noID_noT.values
#
#
#     with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_Comp.pkl', 'rb') as f:
#         scaler_P2020_Cpx_only=load(f)
#
#     with open(Thermobar_dir/'ETR_Press_Petrelli2020_Cpx_Only_Comp.pkl', 'rb') as f:
#         ETR_Press_P2020_Cpx_only=load(f)
#
#
#
#     x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
#     Pred_P_kbar=ETR_Press_P2020_Cpx_only.predict(x_test_scaled)
#     df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Press_P2020_Cpx_only)
#     df_stats.insert(0, 'P_kbar_calc', Pred_P_kbar)
#
#     return df_stats
def P_Nimis1999_BA(T=None):
    '''
        This is a placeholder, function is being called from other .py file')
    '''

## Clinopyroxene-only temperature equations
def T_Wang2021_eq2(P=None, *, Al_VI_cat_6ox, Si_Cpx_cat_6ox, Ti_Cpx_cat_6ox, Cr_Cpx_cat_6ox,
Fet_Cpx_cat_6ox, Mn_Cpx_cat_6ox, Mg_Cpx_cat_6ox, Na_Cpx_cat_6ox, K_Cpx_cat_6ox,
FeII_Wang21, H2O_Liq, Al_Cpx_cat_6ox, Ca_Cpx_cat_6ox):
    '''
    Clinopyroxene-only thermometer of Wang et al. (2021) Eq 2
    :cite:`wang2021new`

    SEE=36.6 C
    '''
    NCT=(1.4844*Al_VI_cat_6ox/(1.4844*Al_VI_cat_6ox+7.7408*Ti_Cpx_cat_6ox
    +1.1675*Cr_Cpx_cat_6ox+1.0604*Fet_Cpx_cat_6ox+0.0387*Mn_Cpx_cat_6ox-0.0628*Mg_Cpx_cat_6ox))


    return (273.15+312.4395*NCT+230.5194*Ti_Cpx_cat_6ox-568.698*Al_Cpx_cat_6ox
    -3212*Mn_Cpx_cat_6ox-476.386*Mg_Cpx_cat_6ox-710.883*Ca_Cpx_cat_6ox
    -651.019*FeII_Wang21-23.384*H2O_Liq+2321.929)





def T_Put2008_eq32d(P, *, Ti_Cpx_cat_6ox, Fet_Cpx_cat_6ox, Al_Cpx_cat_6ox,
                    Cr_Cpx_cat_6ox, Na_Cpx_cat_6ox, K_Cpx_cat_6ox, a_cpx_En):
    '''
    Clinopyroxene-only thermometer of Putirka (2008) Eq32d.
    Overestimates temperature for hydrous data
    :cite:`putirka2008thermometers`

    SEE=±58°C (anhydrous)
    SEE=±87°C (hydrous)
    '''

    return (((93100 + 544 * P) / (61.1 + 36.6 * Ti_Cpx_cat_6ox + 10.9 * Fet_Cpx_cat_6ox
    - 0.95 * (Al_Cpx_cat_6ox + Cr_Cpx_cat_6ox - Na_Cpx_cat_6ox - K_Cpx_cat_6ox)
     + 0.395 * (np.log(a_cpx_En.astype(float)))**2)))


def T_Put2008_eq32dH_Wang2021adap(P, *, Ti_Cpx_cat_6ox, Fet_Cpx_cat_6ox, Al_Cpx_cat_6ox,
                    Cr_Cpx_cat_6ox, Na_Cpx_cat_6ox, K_Cpx_cat_6ox, a_cpx_En, H2O_Liq):
    '''
    Adaptation of the clinopyroxene-only thermometer of Putirka (2008) Eq32d
    by Wang et al. (2021) to account for the effect of H2O
    :cite:`wang2021new` and :cite:`putirka2008thermometers`


    '''

    return (273.15+
    (93100 + 544 * P) / (72.70276 + 74.24179 * Ti_Cpx_cat_6ox + 21.95727 * Fet_Cpx_cat_6ox
    - 0.31648 * (Al_Cpx_cat_6ox + Cr_Cpx_cat_6ox - Na_Cpx_cat_6ox - K_Cpx_cat_6ox)
     + 0.420385 * (np.log(a_cpx_En.astype(float)) )**2 +1.864215*H2O_Liq)
     )

# This is the version in the Putirka 2-pyroxene spreadsheet. The 544 coefficient is changed to 755, and the coefficient for the enstatie activity is
# changed from 0.395 to 3.5. This is a better fit to subsolidus T
# estimates, but provide poorer fits to pyroxenes from volcanic systems.


def T_Put2008_eq32d_subsol(P, *, Ti_Cpx_cat_6ox, Fet_Cpx_cat_6ox, Al_Cpx_cat_6ox,
                           Cr_Cpx_cat_6ox, Na_Cpx_cat_6ox, K_Cpx_cat_6ox, a_cpx_En):
    '''
    Adapted version of clinopyroxene-only thermoter of Putirka (2008) Eq32d,
    provides better fit to subsolidus T estimates
    :cite:`putirka2008thermometers`

    '''
    return (((93100 + 755 * P) / (61.1 + 36.6 * Ti_Cpx_cat_6ox + 10.9 * Fet_Cpx_cat_6ox
    - 0.95 * (Al_Cpx_cat_6ox + Cr_Cpx_cat_6ox - Na_Cpx_cat_6ox - K_Cpx_cat_6ox)
    + 3.5 * (np.log(a_cpx_En.astype(float)))**2)))


def T_Petrelli2020_Cpx_only(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer using the method and training dataset of Petrelli et al. (2020) (although they didnt
    provide a Cpx-only thermometer).
    :cite:`petrelli2020machine`

    SEE= N/A
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values


    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_sklearn_1_3.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Temp_Petrelli2020_Cpx_Only_sklearn_1_3.pkl', 'rb') as f:
        ETR_Temp_P2020_Cpx_only=joblib.load(f)

    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_T_K=ETR_Temp_P2020_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_P2020_Cpx_only)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats

def T_Jorgenson2022_Cpx_only_Norm(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Jorgenson et al (2022) based on
    Machine Learning. Normalized, unlike published model
    :cite:`jorgenson2021machine`

    SEE==+-51°C
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")
    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'scaler_Jorg21_Cpx_only_April24.pkl', 'rb') as f:
        scaler_J22_Cpx_only=load(f)

    with open(Thermobar_dir/'ETR_Temp_Jorg21_Cpx_only_April24.pkl', 'rb') as f:
        ETR_Temp_J22_Cpx_only=joblib.load(f)

    x_test_scaled=scaler_J22_Cpx_only.transform(x_test)
    Pred_T_K=ETR_Temp_J22_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_J22_Cpx_only)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats


def T_Jorgenson2022_Cpx_only(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Jorgenson et al (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-51°C
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent

    with open(Thermobar_dir/'ETR_Temp_Jorg21_Cpx_only_NotNorm_sklearn_1_3.pkl', 'rb') as f:
        ETR_Temp_J22_Cpx_only=joblib.load(f)


    Pred_T_K=ETR_Temp_J22_Cpx_only.predict(x_test)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Temp_J22_Cpx_only)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats

def T_Jorgenson2022_Cpx_only_onnx(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Jorgenson et al (2022) based on
    Machine Learning.
    :cite:`jorgenson2021machine`

    SEE==+-51°C
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values


    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')


    import onnxruntime as rt
    # path=Path(Thermobar_onnx.__file__).parent
    # sess =  rt.InferenceSession(str(path/"Petrelli2020_Cpx_only_Temp.onnx"))
    path = Path(Thermobar_onnx.__file__).parent
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Jorg21_Cpx_only_Temp.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)


    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_T_K = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    #df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Temp_P2020_Cpx_Liq)


    T_K=pd.Series(Pred_T_K[:, 0])
    return T_K

def T_Petrelli2020_Cpx_only_onnx(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Petrelli et al. (2021) based on
    Machine Learning. Using onnx, so doesnt do voting
    :cite:`petrelli2020machine`

    SEE==+-51°C
    '''
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],

    })

    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    import onnxruntime as rt
    # path=Path(Thermobar_onnx.__file__).parent
    # sess =  rt.InferenceSession(str(path/"Petrelli2020_Cpx_only_Temp.onnx"))
    path = Path(Thermobar_onnx.__file__).parent
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    model_path = path / "Petrelli2020_Cpx_only_Temp.onnx"
    sess = rt.InferenceSession(str(model_path), providers=providers)


    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Pred_T_K = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]

    #df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test, ETR_Temp_P2020_Cpx_Liq)


    T_K=pd.Series(Pred_T_K[:, 0])
    return T_K


def T_Petrelli2020_Cpx_only_withH2O(P=None, *, cpx_comps):
    '''
    Clinopyroxene-only  thermometer of Petrelli et al. (2021) based on
    Machine Learning, but adding water.
    :cite:`petrelli2020machine` and this study.

    SEE==+-51°C
    '''
    cpx_test=cpx_comps.copy()
    Cpx_test_noID_noT=pd.DataFrame(data={'SiO2_Cpx': cpx_comps['SiO2_Cpx'],
                                'TiO2_Cpx': cpx_comps['TiO2_Cpx'],
                                'Al2O3_Cpx': cpx_comps['Al2O3_Cpx'],
                                'FeOt_Cpx': cpx_comps['FeOt_Cpx'],
                                'MnO_Cpx': cpx_comps['MnO_Cpx'],
                                'MgO_Cpx': cpx_comps['MgO_Cpx'],
                                'CaO_Cpx': cpx_comps['CaO_Cpx'],
                                'Na2O_Cpx': cpx_comps['Na2O_Cpx'],
                                'K2O_Cpx': cpx_comps['K2O_Cpx'],
                                'Cr2O3_Cpx': cpx_comps['Cr2O3_Cpx'],
                                'H2O_Liq':cpx_comps['H2O_Liq'],

    })
    x_test=Cpx_test_noID_noT.values

    try:
        import Thermobar_onnx
        version=Thermobar_onnx.__version__
        if version != '0.0.4':
            raise RuntimeError(f"Thermobar_onnx version is {version}, but you require version 0.04. Please grab the new tag from github. Scikitlearn had changed too much, v2 doesnt work well anymore")

    except ImportError:
        raise RuntimeError('Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions on how to install this')
    Thermobar_dir=Path(Thermobar_onnx.__file__).parent


    with open(Thermobar_dir/'scaler_Petrelli2020_Cpx_Only_H2O_Jan22.pkl', 'rb') as f:
        scaler_P2020_Cpx_only=joblib.load(f)

    with open(Thermobar_dir/'ETR_Temp_Petrelli2020_Cpx_Only_H2O_Jan22.pkl', 'rb') as f:
        ETR_Temp_P2020_Cpx_only=joblib.load(f)



    x_test_scaled=scaler_P2020_Cpx_only.transform(x_test)
    Pred_T_K=ETR_Temp_P2020_Cpx_only.predict(x_test_scaled)
    df_stats, df_voting=get_voting_stats_ExtraTreesRegressor(x_test_scaled, ETR_Temp_P2020_Cpx_only)
    df_stats.insert(0, 'T_K_calc', Pred_T_K)

    return df_stats

## Function for calculatin clinopyroxene-liquid pressure
# This also includes all the Cpx-only, so you can mix and match cpx and cpx-liq
Cpx_Liq_P_funcs = {P_Put1996_eqP1, P_Mas2013_eqPalk1, P_Put1996_eqP2, P_Mas2013_eqPalk2,
P_Put2003, P_Put2008_eq30, P_Put2008_eq31, P_Put2008_eq32c, P_Mas2013_eqalk32c,
P_Mas2013_Palk2012, P_Wieser2021_H2O_indep, P_Neave2017, P_Petrelli2020_Cpx_Liq,
P_Jorgenson2022_Cpx_Liq, P_Jorgenson2022_Cpx_Liq_onnx, P_Jorgenson2022_Cpx_Liq_Norm,
 P_Jorgenson2022_Cpx_Liq_onnx,
 P_Petrelli2020_Cpx_Liq_onnx,
P_Put2008_eq32a, P_Put2008_eq32b, P_Wang2021_eq1,
P_Petrelli2020_Cpx_only, P_Petrelli2020_Cpx_only_withH2O, P_Nimis1999_BA} # put on outside

Cpx_Liq_P_funcs_by_name = {p.__name__: p for p in Cpx_Liq_P_funcs}


def calculate_cpx_liq_press(*, equationP, cpx_comps=None, liq_comps=None, meltmatch=None,
                            T=None, eq_tests=False, Fe3Fet_Liq=None, H2O_Liq=None,
                           sigma=1, Kd_Err=0.03):
    '''
    Clinopyroxene-Liquid barometer, calculates pressure in kbar
    (and equilibrium tests as an option)

    Parameters
    -------

    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    EquationP: str
        choose from:

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
        |  P_Petrelli2020_Cpx_Liq (Returns voting)
        |  P_Jorgenson2022_Cpx_Liq (Returns voting)
        |  P_Petrelli2020_Cpx_Liq_onnx (Uses onnx, so consistent results, no voting)
        |  P_Wang2021_eq1

    T: float, int, pandas.Series, str
        Temperature in Kelvin to perform calculations at.
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool
        If False, just returns pressure (default) as a panda series
        If True, returns pressure, Values of Eq tests,
        as well as user-entered cpx and liq comps and components

    Returns
    -----------
    Pressure in kbar: pandas.Series
        If eq_tests is False

    Pressure in kbar + eq Tests + input compositions: pandas.DataFrame
        If eq_tests is True

    '''
    # Check liq and cpx same length
    if meltmatch is None and len(cpx_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as cpx  comps. If you want to match up all possible pairs, use the _matching functions instead: calculate_cpx_liq_press_temp_matching')


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

        if 'Jorgenson' in equationP:
            liq_comps_c=normalize_liquid_jorgenson(liq_comps_c)


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

    if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" in equationP:
        P_kbar=func(meltmatch=Combo_liq_cpxs)

    elif ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
        df_stats=func(meltmatch=Combo_liq_cpxs)
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

            if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
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
            liq_comps=liq_comps_c, Fe3Fet_Liq=Fe3Fet_Liq, P=P_kbar, T=T, sigma=sigma, Kd_Err=Kd_Err)
        if meltmatch is not None:
            eq_tests = calculate_cpx_liq_eq_tests(meltmatch=meltmatch,
            Fe3Fet_Liq=Fe3Fet_Liq, P=P, T=T_K, sigma=sigma, Kd_Err=Kd_Err)
        eq_tests.replace([np.inf, -np.inf], np.nan, inplace=True)
        return eq_tests


## Function for calculation Cpx-Liquid temperatures
Cpx_Liq_T_funcs = {T_Put1996_eqT1, T_Mas2013_eqTalk1, T_Put1996_eqT2, T_Mas2013_eqTalk2,
T_Put1999, T_Put2003, T_Put2008_eq33, T_Mas2013_eqalk33,
T_Mas2013_Talk2012, T_Brug2019, T_Petrelli2020_Cpx_Liq,
T_Jorgenson2022_Cpx_Liq,T_Jorgenson2022_Cpx_Liq_onnx,T_Jorgenson2022_Cpx_Liq_Norm,
T_Petrelli2020_Cpx_Liq_onnx,
T_Put2008_eq32d, T_Put2008_eq32d_subsol,
T_Wang2021_eq2, T_Petrelli2020_Cpx_only, T_Petrelli2020_Cpx_only_withH2O,
T_Put2008_eq32dH_Wang2021adap, T_Put2008_eq34_cpx_sat} # put on outside

Cpx_Liq_T_funcs_by_name = {p.__name__: p for p in Cpx_Liq_T_funcs}

def calculate_cpx_liq_temp(*, equationT, cpx_comps=None, liq_comps=None, meltmatch=None,
                           P=None, eq_tests=False, H2O_Liq=None, Fe3Fet_Liq=None,
                           sigma=1, Kd_Err=0.03):
    '''
    Clinopyroxene-Liquid thermometry, calculates temperature in Kelvin
    (and equilibrium tests as an option)

    Parameters
    -------
    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    Or:

    meltmatch: pandas.DataFrame
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
        |  T_Petrelli2020_Cpx_Liq (gives voting)
        |  T_Jorgenson2022_Cpx_Liq (gives voting)
        |  T_Petrelli2020_Cpx_Liq_onnx (gives consistent result every time)


    P: float, int, pandas.Series, str  ("Solve")
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
        pandas.Series: Temperature in Kelvin
    If eq_tests=True
        pandas.DataFrame: Temperature in Kelvin +
        Eq Tests + cpx+liq comps + components

    '''
    # Various warnings etc. to check inputs make sense.

    if meltmatch is None and len(cpx_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as cpx  comps. If you want to match up all possible pairs, use the _matching function: calculate_cpx_liq_press_temp_matching')


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

        if 'Jorgenson' in equationT:
            liq_comps_c=normalize_liquid_jorgenson(liq_comps_c)

        if equationT == "T_Mas2013_Palk2012" or equationT == "T_Mas2013_eqalk33" or equationT == "T_Mas2013_eqTalk2" or equationT == "T_Mas2013_eqTalk1":
            if np.max(liq_comps_c['Fe3Fet_Liq']) > 0:
                w.warn('Some Fe3Fet_Liq are greater than 0. Masotta et al. (2013) calibrate their equations assuming all Fe is Fe2+. You should set Fe3Fet_Liq=0 in the function for consistency. ')

        Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(liq_comps=liq_comps_c,
        cpx_comps=cpx_comps)


    if equationT == "T_Brug2019":
        if np.max(Combo_liq_cpxs['Mgno_Cpx']) > 0.65:
            w.warn("Some inputted CPX compositions have Cpx Mg#>0.65;.",
                stacklevel=2)
        if np.max(Combo_liq_cpxs['Al2O3_Cpx']) > 7:
            w.warn("Some inputted CPX compositions have Al2O3>7 wt%;.",
                stacklevel=2)
        if np.max(Combo_liq_cpxs['SiO2_Liq']) < 70:
            w.warn("Some inputted Liq compositions have  SiO2<70 wt%;",
                stacklevel=2)
        if np.max(Combo_liq_cpxs['Mgno_Cpx']) > 0.65 or Combo_liq_cpxs['Al2O3_Cpx'] or p.max(
                Combo_liq_cpxs['SiO2_Liq']) < 70:
            w.warn("which is outside the recomended calibration range of Brugman and Till (2019)")

    # Easiest to treat Machine Learning ones differently
    if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
        df_stats=func(meltmatch=Combo_liq_cpxs)
        T_K=df_stats['T_K_calc']

    elif ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" in equationT:
        T_K=func(meltmatch=Combo_liq_cpxs)
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

            if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
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
            liq_comps=liq_comps_c, Fe3Fet_Liq=Fe3Fet_Liq, P=P, T=T_K, sigma=sigma, Kd_Err=Kd_Err)
        if meltmatch is not None:
            eq_tests = calculate_cpx_liq_eq_tests(meltmatch=meltmatch,
            Fe3Fet_Liq=Fe3Fet_Liq, P=P, T=T_K, sigma=sigma, Kd_Err=Kd_Err)
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

     cpx_comps: pandas.DataFrame (opt, either specify cpx_comps AND liq_comps or meltmatch)
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    Or

    meltmatch: pandas.DataFrame
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

    Fe3Fet_Liq: float, int, pandas.Series,
        Fe3Fet ratio used to assess Kd Fe-Mg equilibrium between cpx and melt.
        If users don't specify, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    H2O_Liq: float, int, pandas.Series, optional
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
        pandas.DataFrame: Temperature in Kelvin, pressure in Kbar
        Eq Tests + cpx+liq comps + components


    '''

    # Gives users flexibility to reduce or increase iterations
    if meltmatch is None and len(cpx_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as cpx  comps. If you want to match up all possible pairs, use the _matching function: calculate_cpx_liq_press_temp_matching')

    if iterations is not None:
        iterations = iterations
    else:
        iterations = 30

    if meltmatch is not None:
        Combo_liq_cpxs = meltmatch

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

        if equationT is not None:
            if 'Jorgenson' in equationT:
                liq_comps_c=normalize_liquid_jorgenson(liq_comps_c)
        if equationP is not None:
            if 'Jorgenson' in equationP:
                liq_comps_c=normalize_liquid_jorgenson(liq_comps_c)

        Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(liq_comps=liq_comps_c,
        cpx_comps=cpx_comps)

    if equationT is not None:
        if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
            T_func_all=calculate_cpx_liq_temp(meltmatch=Combo_liq_cpxs,
            equationT=equationT, P="Solve")
            T_func = T_func_all.T_K_calc
            Median_T=T_func_all.Median_Trees
            Std_T=T_func_all.Std_Trees
            IQR_T=T_func_all.IQR_Trees
        else:
            T_func = calculate_cpx_liq_temp(meltmatch=Combo_liq_cpxs, equationT=equationT, P="Solve")

    if equationP is not None:
        if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
            P_func_all=calculate_cpx_liq_press(meltmatch=Combo_liq_cpxs,
                equationP=equationP, T="Solve")
            P_func = P_func_all.P_kbar_calc
            Median_P=P_func_all.Median_Trees
            Std_P=P_func_all.Std_Trees
            IQR_P=P_func_all.IQR_Trees
        else:
            P_func = calculate_cpx_liq_press(meltmatch=Combo_liq_cpxs,
            equationP=equationP, T="Solve")



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
        count=0
        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)
            if count==iterations-2:
                # On the second last step, save the pressure
                P_out_loop=P_guess.values
                T_out_loop=T_K_guess.values
            count=count+1

        DeltaP=P_guess-P_out_loop
        DeltaT=T_K_guess-T_out_loop
    else:
        DeltaP=0
        DeltaT=0


    #if equationT != "T_Petrelli2020_Cpx_Liq":
    T_K_guess_is_bad = (T_K_guess == 0) | (T_K_guess == 273.15) | (T_K_guess ==  -np.inf) | (T_K_guess ==  np.inf)
    T_K_guess[T_K_guess_is_bad] = np.nan
    #if equationP != "P_Petrelli2020_Cpx_Liq":
    P_guess[T_K_guess_is_bad] = np.nan


    # calculates equilibrium tests of Neave and Putirka if eq_tests="True"
    if eq_tests is False:
        PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess,
                                    'Delta_P_kbar_Iter': DeltaP,
                                    'Delta_T_K_Iter': DeltaT})
        if equationP is not None:
            if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:

                PT_out.insert(4, "Median_Trees_P", Median_P)
                PT_out.insert(5, "Std_Trees_P", Std_P)
                PT_out.insert(6, "IQR_Trees_P", Std_P)

        if equationT is not None:
            if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
                PT_out.insert(len(PT_out.columns), "Median_Trees_T", Median_T)
                PT_out.insert(len(PT_out.columns), "Std_Trees_T", Std_T)
                PT_out.insert(len(PT_out.columns), "IQR_Trees_T", Std_T)
            return PT_out
    if eq_tests is True:
        if meltmatch is not None:
            eq_tests = calculate_cpx_liq_eq_tests(
            meltmatch=meltmatch, P=P_guess, T=T_K_guess)
        if meltmatch is None:
            eq_tests = calculate_cpx_liq_eq_tests(cpx_comps=cpx_comps,
            liq_comps=liq_comps, P=P_guess, T=T_K_guess)
        if equationP is not None:
            if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
                eq_tests.insert(5, "Median_Trees_P", Median_P)
                eq_tests.insert(6, "Std_Trees_P", Std_P)
                eq_tests.insert(7, "IQR_Trees_P", Std_P)
        if equationT is not None:
            if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
                if "Petrelli" in equationP or "Jorgenson" in equationP:
                    a=7
                else:
                    a=4
                eq_tests.insert(a+1, "Median_Trees_T", Median_T)
                eq_tests.insert(a+2, "Std_Trees_T", Std_T)
                eq_tests.insert(a+3, "IQR_Trees_T", Std_T)

        eq_tests.insert(3,'Delta_P_kbar_Iter',DeltaP )
        eq_tests.insert(4,'Delta_T_K_Iter',DeltaT)
        return eq_tests
## All popular option for Cpx-Liq thermobarometry
import sys
from io import StringIO

def block_prints():
    # Save the current stdout
    original_stdout = sys.stdout

    # Redirect stdout to a StringIO object
    sys.stdout = StringIO()

    # Call the functions that produce prints
    function1()
    function2()
    # ...

    # Restore the original stdout
    sys.stdout = original_stdout


def calculate_cpx_liq_press_all_eqs(cpx_comps, liq_comps, H2O_Liq=None):
    """ This function calculates Cpx-Liq and Cpx-only P and T using a wide range of popular equations in the literature. Happy to add more to this on request!

    Parameters
    -------

     cpx_comps: pandas.DataFrame (opt, either specify cpx_comps AND liq_comps or meltmatch)
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    H2O_Liq: float, int, pandas.Series, optional
        If users don't specify, uses H2O_Liq from liq_comps,
        if specified overwrites this.

    Returns
    -------
    Pandas dataframe with temperature in kelvin, pressure in kbar.

    """



    import warnings
    with w.catch_warnings():
        w.simplefilter('ignore')
        if len(cpx_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as cpx  comps. If you want to match up all possible pairs, use the _matching function instead for 1 equation at a time: calculate_cpx_liq_press_temp_matching')


        cpx_comps_c=cpx_comps.reset_index(drop=True).copy()
        liq_comps_c=liq_comps.reset_index(drop=True).copy()
        print('We have reset the index on Cpx and Liq comps for the Petrelli expressions')
        if isinstance(H2O_Liq, pd.Series):
            H2O_Liq=H2O_Liq.reset_index(drop=True)
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq']=H2O_Liq
        # Different options, only give P and T

        N17_Teq33=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq33", equationP="P_Neave2017")

        Eq30_Teq33=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq33", equationP="P_Put2008_eq30")

        Eq31_Teq33=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq33", equationP="P_Put2008_eq31")

        Eq32c_Teq33=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq33", equationP="P_Put2008_eq32c")

        Eq32c_T2003=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2003", equationP="P_Put2008_eq32c")

        Eq32c_P32d=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq32d", equationP="P_Put2008_eq32c")

        # Default equation 32c when you download spreadsheet For 1st cell (2nd cell different,
        # Emailed Keith about this on Dec 16th).

        # First, calculate pressure using 1996 P1
        T1996_P1_eq34=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq34_cpx_sat", equationP="P_Put1996_eqP1").P_kbar_calc

        # Then uses equation T2 to get temp, referencing the calculated pressure above.
        T1996_T2=calculate_cpx_liq_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put1996_eqT2", P=T1996_P1_eq34)

        # The finally, feeds this into eq32c
        P_eq32c_default_1stcell=calculate_cpx_liq_press(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationP="P_Put2008_eq32c", T=T1996_T2)

        # Default for second cellon - 1st iterate T2 and P1, then feed in
        T1996_P1_T2=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put1996_eqT2", equationP="P_Put1996_eqP1")

        P_eq32c_default_restcell=calculate_cpx_liq_press(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationP="P_Put2008_eq32c", T=T1996_P1_T2.T_K_calc)

        Put2003=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2003", equationP="P_Put2003")


        Teq34_NP17=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq34_cpx_sat", equationP="P_Neave2017")

        Teq34_eq32c=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
        liq_comps=liq_comps_c, equationT="T_Put2008_eq34_cpx_sat", equationP="P_Put2008_eq32c")


        df_out=pd.DataFrame  (data={'P_kbar: (P_Neave17, T_Put2008_eq33)': N17_Teq33.P_kbar_calc,
                                    'T_K: (P_Neave17, T_Put2008_eq33)': N17_Teq33.T_K_calc,

                                    'P_kbar: (P_Neave17, T_Put2008_eq34_cpx_sat)': Teq34_NP17.P_kbar_calc,
                                    'T_K: (P_Neave17, T_Put2008_eq34_cpx_sat)': Teq34_NP17.T_K_calc,

                                    'P_kbar: (P_Put2008_eq30, T_Put2008_eq33)': Eq30_Teq33.P_kbar_calc,
                                    'T_K: (P_Put2008_eq30, T_Put2008_eq33)': Eq30_Teq33.T_K_calc,

                                    'P_kbar: (P_Put2008_eq31, T_Put2008_eq33)': Eq31_Teq33.P_kbar_calc,
                                    'T_K: (P_Put2008_eq31, T_Put2008_eq33)': Eq31_Teq33.T_K_calc,

                                    'P_kbar: (P_Put2008_eq32c, T_Put2008_eq33)': Eq32c_Teq33.P_kbar_calc,
                                    'T_K: (P_Put2008_eq32c, T_Put2008_eq33)': Eq32c_Teq33.T_K_calc,

                                    'P_kbar: (P_Put2008_eq32c, T_Put2003)': Eq32c_T2003.P_kbar_calc,
                                    'T_K: (P_Put2008_eq32c, T_Put2003)': Eq32c_T2003.T_K_calc,


                                    'P_kbar: (P_Put2008_eq32c, T_Put2008_eq32d)': Eq32c_P32d.P_kbar_calc,
                                    'T_K: (P_Put2008_eq32c, T_Put2008_eq32d))': Eq32c_P32d.T_K_calc,

                                    'P_kbar: (P_Put2008_eq32c, default spreadsheet 1st cell)': P_eq32c_default_1stcell,
                                    'P_kbar: (P_Put2008_eq32c, default spreadsheet 2 on)':P_eq32c_default_restcell,

                                    'P_kbar: (P_Put2008_eq32c, T_Put2008_eq34)': Teq34_eq32c.P_kbar_calc,
                                    'T_K: (P_Put2008_eq32c, T_Put2008_eq34))': Teq34_eq32c.T_K_calc,


                                    'P_kbar: (P2003 P&T)': Put2003.P_kbar_calc,
                                    'T_K: (P2003 P&T)': Put2003.T_K_calc,


                                    'T_K: (P_Put1996_eqP1, T_Put1996_eqT2)':T1996_P1_T2.T_K_calc,
                                    'P_kbar: (P_Put1996_eqP1, T_Put1996_eqT2)':T1996_P1_T2.P_kbar_calc,

            })

        try:
            import Thermobar_onnx  # Try importing the package

            Pet2020=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
            liq_comps=liq_comps_c, equationT="T_Petrelli2020_Cpx_Liq", equationP="P_Petrelli2020_Cpx_Liq")

            Jorg2020=calculate_cpx_liq_press_temp(cpx_comps=cpx_comps_c,
            liq_comps=liq_comps_c, equationT="T_Jorgenson2022_Cpx_Liq", equationP="P_Jorgenson2022_Cpx_Liq")

            df_out['P_kbar: (Petrelli, 2020)']=Pet2020.P_kbar_calc
            df_out['T_K: (Petrelli, 2020)']=Pet2020.T_K_calc

            df_out['P_kbar: (Jorgenson, 2022)']=Jorg2020.P_kbar_calc
            df_out['T_K: (Jorgenson, 2022)']=Jorg2020.T_K_calc


        except ImportError:  # If the package is not installed
            # Print a warning message
            print('Warning: Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions')

        # Machine learning - only if available






        return df_out



## Clinopyroxene melt-matching algorithm
def arrange_all_cpx_liq_pairs(liq_comps, cpx_comps, H2O_Liq=None, Fe3Fet_Liq=None):
    # This allows users to overwrite H2O and Fe3Fet

    liq_comps_c = liq_comps.copy().reset_index(drop=True)
    cpx_comps_c = cpx_comps.copy().reset_index(drop=True)


    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
    if "Fe3Fet_Liq" not in liq_comps:
        liq_comps_c['Fe3Fet_Liq'] = 0
    if "Sample_ID_Liq" not in liq_comps:
        liq_comps_c['Sample_ID_Liq'] = liq_comps_c.index
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq'] = H2O_Liq


    # calculating Cpx and liq components.
    myCPXs1_concat = calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)
    myLiquids1_concat = calculate_anhydrous_cat_fractions_liquid(
        liq_comps=liq_comps_c)

    # Adding an ID label to help with melt-cpx rematching later
    myCPXs1_concat['ID_CPX'] = myCPXs1_concat.index
    if "Sample_ID_Cpx" not in cpx_comps:
        myCPXs1_concat['Sample_ID_Cpx'] = myCPXs1_concat.index
    else:
        myCPXs1_concat['Sample_ID_Cpx']=cpx_comps_c['Sample_ID_Cpx']

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
    Combo_liq_cpxs2 = calculate_clinopyroxene_liquid_components(meltmatch=Combo_liq_cpxs)

    return Combo_liq_cpxs2


def calculate_cpx_liq_press_temp_matching(*, liq_comps, cpx_comps, equationT=None,
equationP=None, P=None, T=None, PMax=30, PMin=-10,
Fe3Fet_Liq=None, Kd_Match="Putirka", Kd_Err=0.03, DiHd_Err=0.06, EnFs_Err=0.05, CaTs_Err=0.03, Cpx_Quality=False,
H2O_Liq=None, return_all_pairs=False, iterations=30):

    '''
    Evaluates all possible Opx-Liq pairs from  N Liquids, M Cpx compositions
    returns P (kbar) and T (K) for those in equilibrium.


    Parameters
    -----------

    liq_comps: pandas.DataFrame
        Panda DataFrame of liquid compositions with column headings SiO2_Liq etc.

    cpx_comps: pandas.DataFrame
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

    Or

    T: int, float
        Can also run calculations at a fixed temperature
    P: int, float
        Can also run calculations at a fixed pressure
    Optional:



    Kd_Match: int, str, optional
        allows users to override the default of calculating Kd Fe-Mg based
        on temperature using eq 35 of putirka
        Set at fixed value (e.g., Kd_Match=0.27)
        OR
        specify Kd_Match=Masotta to use the Kd model fo Masotta et al. (2013),
        which is also a function of Na and K, for trachytic and phonolitic magmas.


    Kd_Err: int or float, Default=0.03
        Allows users to specify the permitted error on Kd Fe-Mg (default=0.03 from Neave et al. 2019)

    DiHd_Err: int or float, optional. Default=0.06
        Allows users to specify the permitted error on DiHd (default=0.06 from Neave et al. 2019)
        Compares measured and calculated values from Mollo et al. (2013)

    EnFs_Err: int or float, optional. Default=0.05
        Allows users to specify the permitted error on EnFs (default=0.05 from Neave et al. 2019)
        Compares measured and calculated values from Mollo et al. (2013)

    CaTs_Err: int or float, optional. Default=0.03
        Allows users to specify the permitted error on CaTs (default=0.03 from Neave et al. 2019)
        Compares measured and calculated values from Putirka (1999).

    Fe3Fet_Liq: float, int, pandas.Series, optional
        Fe3Fet ratio used to assess Kd Fe-Mg equilibrium between cpx and melt.
        If users don't specify, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    H2O_Liq: float, int, pandas.Series, optional
        If users don't specify, uses H2O_Liq from liq_comps, if specified overwrites this.

    Cpx Quality: bool, optional
        Default False. If True, filters out clinopyroxenes with cation sums outside of
        4.02-3.99 (after Neave et al. 2017)

    PMax: int or float,  optional

       Default value of 30 kbar. Uses to apply a preliminary KdFe-Mg filter
       based on the T equation specified by the user.


    PMin: int or float, optional
       Default value of -10 kbar. Uses to apply a preliminary KdFe-Mg filter
       based on the T equation specified by the user.


    Returns: dict

        Av_PTs: Average P and T for each cpx.
        E.g., if cpx1 matches Liq1, Liq4, Liq6, Liq10, averages outputs for all 4 of those liquids.
        Returns mean and 1 sigma of these averaged parameters for each Cpx.

        All_PTs: Returns output parameters for all matches (e.g, cpx1-Liq1, cpx1-Liq4) without any averaging.

    '''

    if Kd_Match == "Masotta":
        print('Caution, you have selected to use the Kd-Fe-Mg model of Masotta et al. (2013)'
        'which is only valid for trachyte and phonolitic magmas. '
        ' use PutKd=True to use the Kd model of Putirka (2008)')

    if isinstance(Kd_Match, int) or isinstance(Kd_Match, float) and Kd_Err is None:
        raise ValueError('You have entered a numerical value for Kd_Match, '
        'You need to specify a Kd_Err to accept matches within Kd_Match+-Kd_Err')

    if equationP is not None and P is not None:
        raise ValueError('You have entered an equation for P and specified a pressure. '
        'Either enter a P equation, or choose a pressure, not both ')
    if equationT is not None and T is not None:
        raise ValueError('You have entered an equation for T and specified a temperature. '
        'Either enter a T equation, or choose a temperature, not both  ')

    # This allows users to overwrite H2O and Fe3Fet

    liq_comps_c = liq_comps.copy().reset_index(drop=True)
    cpx_comps_c = cpx_comps.copy().reset_index(drop=True)


    # Status update for user
    LenCpx=len(cpx_comps)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenCpx) + " Cpx & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(LenCpx*LenLiqs) +
          " Liq-Cpx pairs, be patient if this is >>1 million!")


    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
    if "Fe3Fet_Liq" not in liq_comps:
        liq_comps_c['Fe3Fet_Liq'] = 0
    if "Sample_ID_Liq" not in liq_comps:
        liq_comps_c['Sample_ID_Liq'] = liq_comps_c.index
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq'] = H2O_Liq


    # calculating Cpx and liq components.
    myCPXs1_concat = calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)
    myLiquids1_concat = calculate_anhydrous_cat_fractions_liquid(
        liq_comps=liq_comps_c)

    # Adding an ID label to help with melt-cpx rematching later
    myCPXs1_concat['ID_CPX'] = myCPXs1_concat.index
    if "Sample_ID_Cpx" not in cpx_comps:
        myCPXs1_concat['Sample_ID_Cpx'] = myCPXs1_concat.index
    else:
        myCPXs1_concat['Sample_ID_Cpx']=cpx_comps_c['Sample_ID_Cpx']

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





    # calculate clinopyroxene-liquid components for this merged dataframe
    Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(meltmatch=Combo_liq_cpxs)



    if Cpx_Quality is True:
        Combo_liq_cpxs_2 = Combo_liq_cpxs.loc[(Combo_liq_cpxs['Cation_Sum_Cpx'] < 4.02) & (
            Combo_liq_cpxs['Cation_Sum_Cpx'] > 3.99) & (Combo_liq_cpxs['Jd'] > 0.01)]

    if Cpx_Quality is False:
        Combo_liq_cpxs_2 = Combo_liq_cpxs.copy()
    # This section of code is for when users specify a presssure or
    # temperature, its much faster to not have to iterate, so we don't need
    # the preliminary Kd filter



    if P is not None:
        Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs.copy()
        T_K_calc = calculate_cpx_liq_temp(meltmatch=Combo_liq_cpxs_FeMgMatch,
        equationT=equationT, P=P)
        P_guess = P
        T_K_guess = T_K_calc
    if T is not None:
        Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs.copy()
        P_kbar_calc = calculate_cpx_liq_press(meltmatch=Combo_liq_cpxs_FeMgMatch,
        equationP=equationP, T=T)
        P_guess = P_kbar_calc
        T_K_guess = T

    # This is the default, e.g., the function filters for equilibrium test.
    if return_all_pairs is False:

        if equationP is not None and equationT is not None:

            # Initial Mg# filter, done by calculating temperature for extreme pressures,
            # e.g, 0 and 3 Gpa. Reduces number of P-T solving
            PMin = PMin
            PMax = PMax
            Kd_Err = Kd_Err

        # Filter out bad analysis first off


            if "Petrelli" in equationT or 'Jorgenson' in equationT:
                Combo_liq_cpxs_2['T_Liq_MinP'] = calculate_cpx_liq_temp(
                    meltmatch=Combo_liq_cpxs_2, equationT=equationT, P=PMin).T_K_calc
                Combo_liq_cpxs_2['T_Liq_MaxP'] = calculate_cpx_liq_temp(
                meltmatch=Combo_liq_cpxs_2, equationT=equationT, P=PMax).T_K_calc
            else:
                Combo_liq_cpxs_2['T_Liq_MinP'] = calculate_cpx_liq_temp(
                    meltmatch=Combo_liq_cpxs_2, equationT=equationT, P=PMin)
                Combo_liq_cpxs_2['T_Liq_MaxP'] = calculate_cpx_liq_temp(
                meltmatch=Combo_liq_cpxs_2, equationT=equationT, P=PMax)


            # calculating Delta Kd-Fe-Mg using equation 35 of Putirka 2008

            if Kd_Match == "Putirka":
                Combo_liq_cpxs_2['Kd_MinP'] = np.exp(
                    -0.107 - 1719 / Combo_liq_cpxs_2['T_Liq_MinP'])
                Combo_liq_cpxs_2['Kd_MaxP'] = np.exp(
                    -0.107 - 1719 / Combo_liq_cpxs_2['T_Liq_MaxP'])

                Delta_Kd_T_MinP = abs(
                    Combo_liq_cpxs_2['Kd_MinP'] - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                Delta_Kd_T_MaxP = abs(
                    Combo_liq_cpxs_2['Kd_MaxP'] - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])

            if Kd_Match != "Masotta" and Kd_Match != "Putirka":
                str3 = str(Kd_Match)
                print('the code is evaluating Kd matches using Kd=' + str3)
                Delta_Kd_T_MinP = abs(
                    Kd_Match - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                Delta_Kd_T_MaxP = abs(
                    Kd_Match - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                Combo_liq_cpxs_2.insert(
                    0, "DeltaKd_userselected=" + str3, Delta_Kd_T_MinP)

            if Kd_Match == "Masotta":
                print('using sigma='+ str(Kd_Err) +'If you want a different test, specify Kd_err in the function')

                ratioMasotta = Combo_liq_cpxs_2['Na_Liq_cat_frac'] / (
                    Combo_liq_cpxs_2['Na_Liq_cat_frac'] + Combo_liq_cpxs_2['K_Liq_cat_frac'])
                Delta_Kd_T_MinP = abs(
                    np.exp(1.735 - 3056 / Combo_liq_cpxs_2['T_Liq_MinP'] - 1.668 * ratioMasotta) - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])
                Delta_Kd_T_MaxP = abs(
                    np.exp(1.735 - 3056 / Combo_liq_cpxs_2['T_Liq_MaxP'] - 1.668 * ratioMasotta) - Combo_liq_cpxs_2['Kd_Fe_Mg_Fe2'])

            # The logic here is that if Delta KD with both the max and min temperature are outside the specified KDerror,
            # no temperature inbetween will be inequilibrium.
            Combo_liq_cpxs_FeMgMatch = Combo_liq_cpxs_2.loc[~((Delta_Kd_T_MaxP > Kd_Err) &
                                                                (Delta_Kd_T_MinP > Kd_Err))].reset_index(drop = True)



            str2 = str(np.shape(Combo_liq_cpxs_FeMgMatch)[0])
            print(str2 + ' Matches remaining after initial Kd filter. '
            'Now moving onto iterative calculations')

            # Now we have reduced down the number of calculations, we solve for
            # P and T iteratively

            PT_out = calculate_cpx_liq_press_temp(meltmatch=Combo_liq_cpxs_FeMgMatch,
            equationP=equationP, equationT=equationT, iterations=iterations)
            P_guess = PT_out['P_kbar_calc']
            T_K_guess = PT_out['T_K_calc']
            Delta_T_K_Iter=PT_out['Delta_T_K_Iter']
            Delta_P_kbar_Iter=PT_out['Delta_P_kbar_Iter']


        # This performs calculations if user specifies equation for P, but a real temp:
        if equationP is not None and equationT is None:
            P_guess = calculate_cpx_liq_press(meltmatch=Combo_liq_cpxs_FeMgMatch,
            equationP=equationP, T=T)
            T_K_guess = T
            Delta_T_K_Iter=0
            Delta_P_kbar_Iter=0
        # Same if user doesnt specify an equation for P, but a real P
        if equationT is not None and equationP is None:
            T_guess = calculate_cpx_liq_temp(meltmatch=Combo_liq_cpxs_FeMgMatch,
            equationT=equationT, P=P)
            P_guess = P
            Delta_T_K_Iter=0
            Delta_P_kbar_Iter=0

        # Now, we use calculated pressures and temperatures, regardless of
        # whether we iterated or not, to calculate the other CPX components
        Combo_liq_cpxs_eq_comp = calculate_cpx_liq_eq_tests(
            meltmatch=Combo_liq_cpxs_FeMgMatch, P=P_guess, T=T_K_guess)









        combo_liq_cpx_fur_filt = Combo_liq_cpxs_eq_comp.copy()




        # First, make filter based on various Kd optoins
        if Kd_Match != "Masotta" and Kd_Match != "Putirka":


            Combo_liq_cpxs_eq_comp.loc[:, 'DeltaKd_Kd_Match_userSp']= abs(
                Kd_Match - Combo_liq_cpxs_eq_comp['Kd_Fe_Mg_Fe2'])
            filtKd = (Combo_liq_cpxs_eq_comp['DeltaKd_Kd_Match_userSp'] < Kd_Err)
        else:
            if Kd_Match == "Putirka":
                filtKd = (Combo_liq_cpxs_eq_comp['Delta_Kd_Put2008'] < Kd_Err)
            if Kd_Match == "Masotta":
                filtKd = (Combo_liq_cpxs_eq_comp['Delta_Kd_Mas2013'] < Kd_Err)

        # Then filter other components based on user-selected errors
        # (default values from NEave et al. 2009)

        # Lets make filter
        eqfilter=((filtKd) & (Combo_liq_cpxs_eq_comp['Delta_DiHd_Mollo13'] < DiHd_Err)
        & (Combo_liq_cpxs_eq_comp['Delta_EnFs_Mollo13'] < EnFs_Err)
        & (Combo_liq_cpxs_eq_comp['Delta_CaTs_Put1999'] < CaTs_Err))


        combo_liq_cpx_fur_filt = (
        Combo_liq_cpxs_eq_comp.loc[eqfilter])

        CpxNumbers = sum(eqfilter)




    if return_all_pairs is True:
        print('No equilibrium filters applied')
        CpxNumbers=len(cpx_comps)

        if equationP is not None and equationT is not None:


            PT_out = calculate_cpx_liq_press_temp(meltmatch=Combo_liq_cpxs_2,
            equationP=equationP, equationT=equationT, iterations=iterations)
            P_guess = PT_out['P_kbar_calc']
            T_K_guess = PT_out['T_K_calc']
            Delta_T_K_Iter=PT_out['Delta_T_K_Iter'].astype(float)
            Delta_P_kbar_Iter=PT_out['Delta_P_kbar_Iter'].astype(float)


        # This performs calculations if user specifies equation for P, but a real temp:
        if equationP is not None and equationT is None:
            P_guess = calculate_cpx_liq_press(meltmatch=Combo_liq_cpxs_2,
            equationP=equationP, T=T)
            T_K_guess = T
            Delta_T_K_Iter=0
            Delta_P_kbar_Iter=0
        # Same if user doesnt specify an equation for P, but a real P
        if equationT is not None and equationP is None:
            T_guess = calculate_cpx_liq_temp(meltmatch=Combo_liq_cpxs_2,
            equationT=equationT, P=P)
            P_guess = P
            Delta_T_K_Iter=0
            Delta_P_kbar_Iter=0

        combo_liq_cpx_fur_filt = calculate_cpx_liq_eq_tests(
            meltmatch=Combo_liq_cpxs_2, P=P_guess, T=T_K_guess)





    combo_liq_cpx_fur_filt.insert(2, 'Delta_T_K_Iter', Delta_T_K_Iter)
    combo_liq_cpx_fur_filt.insert(3, 'Delta_P_kbar_Iter', Delta_P_kbar_Iter)



    # This just tidies up some columns to put stuff nearer the start


    cols_to_move = ['Sample_ID_Liq', 'Sample_ID_Cpx']
    combo_liq_cpx_fur_filt = combo_liq_cpx_fur_filt[cols_to_move + [
        col for col in combo_liq_cpx_fur_filt.columns if col not in cols_to_move]]



    Liquid_sample_ID=combo_liq_cpx_fur_filt["Sample_ID_Liq"]
    combo_liq_cpx_fur_filt.drop(["Sample_ID_Liq"], axis=1, inplace=True)
    if T is not None:
        combo_liq_cpx_fur_filt.rename(columns={'T_K_calc': 'T_K_input'}, inplace=True)
    if P is not None:
        combo_liq_cpx_fur_filt.rename(columns={'P_kbar_calc': 'P_kbar_input'}, inplace=True)

    print('Finished calculating Ps and Ts, now just averaging the results. Almost there!')

    # Final step, calcuate a 3rd output which is the average and standard
    # deviation for each CPx (e.g., CPx1-Melt1, CPx1-melt3 etc. )


    if CpxNumbers > 0:
        with w.catch_warnings():
            w.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


            df1_Mean_nopref=combo_liq_cpx_fur_filt.groupby(['ID_CPX', 'Sample_ID_Cpx'], as_index=False).mean()

            df1_Std_nopref=combo_liq_cpx_fur_filt.groupby(['ID_CPX', 'Sample_ID_Cpx'], as_index=False).std()
            count=combo_liq_cpx_fur_filt.groupby('ID_CPX',as_index=False).count().iloc[:, 1]
            df1_Mean_nopref['# of Liqs Averaged']=count
            Sample_ID_Cpx_Mean=df1_Mean_nopref['Sample_ID_Cpx']
            Sample_ID_Cpx_Std=df1_Std_nopref['Sample_ID_Cpx']
            df1_Mean=df1_Mean_nopref.add_prefix('Mean_')

            df1_Std=df1_Std_nopref.add_prefix('Std_')
            # Drop columns if present
            if 'Mean_Eq Tests Neave2017?' in df1_Mean.columns:
                df1_Mean = df1_Mean.drop(['Mean_Eq Tests Neave2017?'], axis=1)

            if 'Mean_Sample_ID_Cpx' in df1_Mean.columns:
                df1_Mean = df1_Mean.drop(['Mean_Sample_ID_Cpx'], axis=1)

            if 'Std_Eq Tests Neave2017?' in df1_Std.columns:
                df1_Std = df1_Std.drop(['Std_Eq Tests Neave2017?'], axis=1)

            if 'Std_Sample_ID_Cpx' in df1_Std.columns:
                df1_Std = df1_Std.drop(['Std_Sample_ID_Cpx'], axis=1)

            df1_Mean.rename(columns={"Mean_ID_CPX": "ID_CPX"}, inplace=True)
            df1_Mean.rename(columns={"Mean_# of Liqs Averaged": "# of Liqs Averaged"}, inplace=True)
            df1_Std.rename(columns={"Std_ID_CPX": "ID_CPX"}, inplace=True)



            df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_CPX'])
            df1_M['Sample_ID_Cpx']=Sample_ID_Cpx_Mean

            if equationT is not None and equationP is not None:
                cols_to_move = ['Sample_ID_Cpx', '# of Liqs Averaged',
                            'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_calc',
                            'Std_P_kbar_calc']

            if equationT is not None and equationP is None:
                cols_to_move = ['Sample_ID_Cpx',
                            'Mean_P_kbar_input',
                            'Std_P_kbar_input', 'Mean_T_K_calc', 'Std_T_K_calc']

            if equationT is None and equationP is not None:
                cols_to_move = ['Sample_ID_Cpx',
                            'Mean_T_K_input', 'Std_T_K_input', 'Mean_P_kbar_calc',
                            'Std_P_kbar_calc']

            df1_M = df1_M[cols_to_move +
                        [col for col in df1_M.columns if col not in cols_to_move]]


    else:
        raise Exception(
            'No Matches - what you should do now - type the arguement return_all_pairs=True in this function to get all matches and look at which filters are fail')




    print('Done!!! I found a total of N='+str(len(combo_liq_cpx_fur_filt)) + ' Cpx-Liq matches using the specified filter. N=' + str(len(df1_M)) + ' Cpx out of the N='+str(LenCpx)+' Cpx that you input matched to 1 or more liquids')

    combo_liq_cpx_fur_filt['Sample_ID_Liq']=Liquid_sample_ID

    return {'Av_PTs': df1_M, 'All_PTs': combo_liq_cpx_fur_filt}



## Function for calculationg Cpx-only pressure
Cpx_only_P_funcs = {P_Put2008_eq32a, P_Put2008_eq32b, P_Wang2021_eq1,
 P_Petrelli2020_Cpx_only, P_Jorgenson2022_Cpx_only, P_Jorgenson2022_Cpx_only_onnx,   P_Jorgenson2022_Cpx_only_Norm,
 P_Petrelli2020_Cpx_only_onnx, P_Petrelli2020_Cpx_only_withH2O, P_Nimis1999_BA}
Cpx_only_P_funcs_by_name = {p.__name__: p for p in Cpx_only_P_funcs}


def calculate_cpx_only_press(*, cpx_comps, equationP, T=None, H2O_Liq=None,
eq_tests=False, return_input=None):
    '''
    Clinopyroxene only barometry. Enter a panda dataframe with Cpx compositions,
    returns a pressure in kbar.

    Parameters
    -------

    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    equationP: str
        | P_Nimis1999_BA (T-independent)
        | P_Put2008_eq32a (T-dependent)
        | P_Put2008_eq32b (T-dependent, H2O dependent)
        | P_Petrelli2020_Cpx_only (T_independent, H2O-independent)
        | P_Petrelli2020_Cpx_only_withH2O (T_independent, H2O-dependent)
        | P_Wang2021_eq1 (T_independent, H2O-independent)
        | P_Jorgenson2022_Cpx_only (T_independent, H2O-independent)
        | P_Jorgenson2022_Cpx_only_Norm (T_independent, H2O-independent)* v similar, uses standard scalar



    T: float, int, pandas.Series, float, None
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    H2O_Liq: int, float, pd.Series, None
        H2O content in the liquid. used for some Cpx-only equations.


    eq_tests: bool
        If True, returns cpx_components as well.

    Returns
    -------
    pandas series
       Pressure in kbar

    '''
    if return_input is not None:
        raise TypeError('For consistency with other functions, please now use eq_tests=, not return_input=....')

    cpx_comps_c=cpx_comps.copy()




    if 'Petrelli' in equationP or 'Jorgenson' in equationP:

        if  check_consecative(cpx_comps_c)==False:
            cpx_comps_c=cpx_comps_c.reset_index(drop=True)
            print('Weve reset the index of your Cpx compositions for ML calculations, as non-consecative indexes or missing indices cause problems for iteration' )


    cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)

    if H2O_Liq is None:
        if equationP == "P_Put2008_eq32b" or  equationP == "P_Petrelli2020_Cpx_only_withH2O":
            w.warn('This Cpx-only barometer is sensitive to H2O content of the liquid. '
        ' By default, this function uses H2O=0 wt%, else you can enter a value of H2O_Liq in the function')
            cpx_components['H2O_Liq']=0
            cpx_comps_c['H2O_Liq']=0
    if H2O_Liq is not None:
        if equationP == "P_Put2008_eq32b" or  equationP == "P_Petrelli2020_Cpx_only_withH2O":
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






    # if
    #     df_stats=P_Petrelli2020_Cpx_only_withH2O(cpx_comps=cpx_comps_c)
    #     P_kbar=df_stats['P_kbar_calc']
    #
    # if equationP == "
    #     df_stats=P_Petrelli2020_Cpx_only_noCr(cpx_comps=cpx_comps_c)
    #     P_kbar=df_stats['P_kbar_calc']
    if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
        df_stats=func(cpx_comps=cpx_comps_c)

    elif ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" in equationP:
        P_kbar=func(cpx_comps=cpx_comps_c)


    elif equationP=="P_Nimis1999_BA":
        calc_Nimis=calculate_P_Nimmis_BA(cpx_comps)

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


    if eq_tests is False:
        if equationP =="P_Nimis1999_BA":
            return calc_Nimis['P_kbar_calc']
        if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
            P_kbar=df_stats['P_kbar_calc']
            return df_stats
        else:
            return P_kbar

    # if return_input is True:
    #     if equationP =="P_Nimis1999_BA":
    #         return calc_Nimis
    #     if equationP == "P_Petrelli2020_Cpx_only" or equationP == "P_Petrelli2020_Cpx_only_withH2O" or equationP == "P_Petrelli2020_Cpx_only_noCr" or ("Jorgenson" in equationP and "onnx" not in equationP):
    #         out=pd.concat([df_stats, cpx_comps],axis=1)
    #         return out
    #     else:
    #         cpx_comps_c.insert(0, 'P_kbar_calc', P_kbar)
    #         return cpx_comps_c


    if eq_tests is True:
        if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
            out=pd.concat([df_stats, cpx_components],axis=1)
            return out
        else:
            cpx_components.insert(0, 'P_kbar', P_kbar)
            return cpx_components



def calculate_cpx_only_press_all_eqs(cpx_comps, eq_tests=True, H2O_Liq=None):
    """ This function calculates Cpx-only pressures and temperatures using all supported equations. It returns these calcs as a dataframe

    Parameters
    -------

    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    H2O_Liq: int, float, pd.Series
        Used for equations which have a term for this. Uses 0 by default and returns a warning

    eq_tests: bool
        If True, returns cpx_components as well.

    Returns
    --------------
    pd.DataFrame of calcs and Cpx comps

    """

    import warnings
    with w.catch_warnings():
        w.simplefilter('ignore')
        cpx_comps_copy=cpx_comps.reset_index(drop=True).copy()
        if isinstance(H2O_Liq, pd.Series):
            H2O_Liq=H2O_Liq.reset_index(drop=True)
        cpx_comps_c=calculate_clinopyroxene_components(cpx_comps=cpx_comps_copy)
        cpx_comps_copy['H2O_Liq']=H2O_Liq
        cpx_comps_c['H2O_Liq']=H2O_Liq
        cpx_comps_c['P_Wang21_eq1']=calculate_cpx_only_press(cpx_comps=cpx_comps_copy, equationP="P_Wang2021_eq1")

        cpx_comps_c['T_Wang21_eq2']=calculate_cpx_only_temp(cpx_comps=cpx_comps_copy, equationT="T_Wang2021_eq2", H2O_Liq=H2O_Liq)



        cpx_comps_c['T_Put_Teq32d_Peq32a']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32a", equationT="T_Put2008_eq32d").T_K_calc

        cpx_comps_c['T_Put_Teq32d_Peq32b']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32b", equationT="T_Put2008_eq32d", H2O_Liq=H2O_Liq).T_K_calc

        cpx_comps_c['T_Put_Teq32d_subsol_Peq32a']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32a", equationT="T_Put2008_eq32d_subsol").T_K_calc
        cpx_comps_c['T_Put_Teq32d_subsol_Peq32b']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32b", equationT="T_Put2008_eq32d_subsol", H2O_Liq=H2O_Liq).T_K_calc

        cpx_comps_c['P_Put_Teq32d_subsol_Peq32a']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32a", equationT="T_Put2008_eq32d_subsol").P_kbar_calc
        cpx_comps_c['P_Put_Teq32d_subsol_Peq32b']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32b", equationT="T_Put2008_eq32d_subsol", H2O_Liq=H2O_Liq).P_kbar_calc



        cpx_comps_c['P_Put_Teq32d_Peq32a']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32a", equationT="T_Put2008_eq32d").P_kbar_calc

        cpx_comps_c['P_Put_Teq32d_Peq32b']=calculate_cpx_only_press_temp(cpx_comps=cpx_comps_copy,
        equationP="P_Put2008_eq32b", equationT="T_Put2008_eq32d", H2O_Liq=H2O_Liq).P_kbar_calc





        # Check if marchine learning package is installed


        try:
            import Thermobar_onnx  # Try importing the package

            cpx_comps_c['P_Petrelli20']=calculate_cpx_only_press(cpx_comps=cpx_comps_copy,
            equationP="P_Petrelli2020_Cpx_only_onnx")

            cpx_comps_c['P_Jorgenson22']=calculate_cpx_only_press(cpx_comps=cpx_comps_copy,
            equationP="P_Jorgenson2022_Cpx_only_onnx")

            cpx_comps_c['T_Petrelli20']=calculate_cpx_only_temp(cpx_comps=cpx_comps_copy,
            equationT="T_Petrelli2020_Cpx_only_onnx")

            cpx_comps_c['T_Jorgenson22']=calculate_cpx_only_temp(cpx_comps=cpx_comps_copy,
            equationT="T_Jorgenson2022_Cpx_only_onnx")


            cols_to_move = ['P_Wang21_eq1', 'T_Wang21_eq2', 'T_Jorgenson22', 'P_Jorgenson22', 'T_Petrelli20', 'T_Put_Teq32d_Peq32a', 'T_Put_Teq32d_Peq32b', 'P_Petrelli20',
            'P_Put_Teq32d_Peq32a', 'P_Put_Teq32d_Peq32b', 'Jd_from 0=Na, 1=Al']





        except ImportError:  # If the package is not installed
            # Print a warning message
            print('Warning: Thermobar_onnx is not installed - this is required to perform calculations using Petrelli and Jorgenson ML method. See the ReadME for further instructions')

            cols_to_move = ['P_Wang21_eq1', 'T_Wang21_eq2', 'T_Put_Teq32d_Peq32a', 'T_Put_Teq32d_Peq32b',
            'P_Put_Teq32d_Peq32a', 'P_Put_Teq32d_Peq32b', 'Jd_from 0=Na, 1=Al']


        cpx_comps_c_move = cpx_comps_c[cols_to_move + [
        col for col in cpx_comps_c.columns if col not in cols_to_move]]




    if eq_tests is True:
        return cpx_comps_c_move
    if eq_tests is False:
        return cpx_comps_c_move.iloc[:, 0:13]

## Function for calculating Cpx-only temperature
Cpx_only_T_funcs = {T_Put2008_eq32d, T_Put2008_eq32d_subsol,
T_Wang2021_eq2, T_Petrelli2020_Cpx_only, T_Jorgenson2022_Cpx_only,
 T_Jorgenson2022_Cpx_only_onnx,
 T_Jorgenson2022_Cpx_only_Norm, T_Petrelli2020_Cpx_only_onnx, T_Petrelli2020_Cpx_only_withH2O, T_Put2008_eq32dH_Wang2021adap}
Cpx_only_T_funcs_by_name = {p.__name__: p for p in Cpx_only_T_funcs}


def calculate_cpx_only_temp(*, cpx_comps=None, equationT=None, P=None,
H2O_Liq=None, eq_tests=False, return_input=None):
    '''
    Clinopyroxene only thermometer. Enter a panda dataframe with Cpx compositions,
    returns a temperature in Kelvin.

    Parameters
    -------

    cpx_comps: pandas.DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    equationT: str
        | T_Put2008_eq32d (P-dependent)
        | T_Put2008_eq32d_subsol (P-dependent)
        | T_Put2008_eq32dH_Wang2021adap (P-dependent, H2O-dependent)
        | T_Petrelli2020_Cpx_only (P-independent, H2O-independent)
        | T_Petrelli2020_Cpx_only_withH2O (P-independent, H2O-dependent)
        | T_Wang2021_eq2 (P-independent, H2O-dependent)
        | T_Jorgenson2022_Cpx_only (P-independent, H2O-independent)
        | T_Jorgenson2022_Cpx_only_Norm (P-independent, H2O-independent)*As above, uses python scalar



    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool
        if True, also returns cpx_components. No actual equilibrium tests.

    Returns
    -------
    pandas series
       Temperature in Kelvin

    '''
    if return_input is not None:
        raise TypeError('For consistency with other functions, please now use eq_tests=, not return_input=....')

    cpx_comps_c=cpx_comps.copy()

    if 'Petrelli' in equationT or 'Jorgenson' in equationT:

        if  check_consecative(cpx_comps_c)==False:
            cpx_comps_c=cpx_comps_c.reset_index(drop=True)

            print('Weve reset the index of your Cpx compositions for Petrelli ML calculations, as non-consecative indexes or missing indices cause problems for iteration' )

    cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)

    if H2O_Liq is None:
        if equationT == "T_Petrelli2020_Cpx_only_withH2O" or  equationT == "T_Wang2021_eq2" or equationT == "T_Wang2021_eq4" or equationT=="T_Put2008_eq32dH_Wang2021adap":
            w.warn('This Cpx-only thermometer is sensitive to H2O content of the liquid. '
        ' By default, this function uses H2O=0 wt%, else you can enter a value of H2O_Liq in the function')
            cpx_components['H2O_Liq']=0
            cpx_comps_c['H2O_Liq']=0
    if H2O_Liq is not None:
        if equationT == "T_Petrelli2020_Cpx_only_withH2O" or  equationT == "T_Wang2021_eq2" or equationT == "T_Wang2021_eq4" or equationT=="T_Put2008_eq32dH_Wang2021adap":
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


    if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:

        df_stats=func(cpx_comps=cpx_comps_c)
        T_K=df_stats['T_K_calc']

    elif ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" in equationT:

        T_K=func(cpx_comps=cpx_comps_c)



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

    if eq_tests is False:
        if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
            return df_stats
        else:
            return T_K
    if eq_tests is True:
        if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
            out=pd.concat([df_stats, cpx_components],axis=1)
            return out
        else:
            cpx_components.insert(0, 'T_K_calc', T_K)
            return cpx_components


## Iterating PT- Cpx only
def calculate_cpx_only_press_temp(*, cpx_comps=None, equationP=None,
equationT=None, iterations=30, T_K_guess=1300, H2O_Liq=None, eq_tests=True, return_input=None):


    '''
    Solves simultaneous equations for temperature and pressure using
    clinopyroxene-lonly thermometers and barometers.


    Parameters
    -------

    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    equationP: str
        | P_Put2008_eq32a (T-dependent)
        | P_Put2008_eq32b (T-dependent, H2O dependent)

    equationT: str
        | T_Put2008_eq32d (P-dependent)
        | T_Put2008_eq32d_subsol (P-dependent)

    H2O_Liq: float, int, pandas.Series, optional
        Needed if you select P_Put2008_eq32b, which is H2O-dependent.

    Optional:


    iterations: int, default=30
         Number of iterations used to converge to solution.

    T_K_guess: int or float. Default is 1300 K
         Initial guess of temperature.

    eq_tests: bool
        if True, also returns cpx_components


    Returns:
    -------
    pandas.DataFrame: Pressure in kbar, Temperature in K
    '''
    if return_input is not None:
        raise TypeError('For consistency with other functions, please now use eq_tests=, not return_input=....')


    cpx_comps_c=cpx_comps.copy()
    if ('Petrelli' in equationT or "Petrelli" in equationP) or ('Jorgenson' in equationT or "Jorgenson" in equationP):

        if  check_consecative(cpx_comps_c)==False:
            cpx_comps_c=cpx_comps_c.reset_index(drop=True)

            print('Weve reset the index of your Cpx compositions for Petrelli ML calculations, as non-consecative indexes or missing indices cause problems for iteration' )

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

    if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:
        P_func2=P_func.copy()
        P_func = P_func2.P_kbar_calc
        Median_P=P_func2.Median_Trees
        Std_P=P_func2.Std_Trees
        IQR_P=P_func2.IQR_Trees


    if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
        T_func2=T_func.copy()
        T_func=T_func2.T_K_calc
        Median_T=T_func2.Median_Trees
        Std_T=T_func2.Std_Trees
        IQR_T=T_func2.IQR_Trees

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
        count=0
        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)
            if count==iterations-2:
                # On the second last step, save the pressure
                P_out_loop=P_guess.values
                T_out_loop=T_K_guess.values
            count=count+1

        DeltaP=P_guess-P_out_loop
        DeltaT=T_K_guess-T_out_loop

    else:
        DeltaP=0
        DeltaT=0

    PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess,
                                    'Delta_P_kbar_Iter': DeltaP,
                                    'Delta_T_K_Iter': DeltaT})
    PT_out.replace([np.inf, -np.inf], np.nan, inplace=True)

    if ('Petrelli' in equationP or "Jorgenson" in equationP) and "onnx" not in equationP:

        PT_out.insert(4, "Median_Trees_P", Median_P)
        PT_out.insert(5, "Std_Trees_P", Std_P)
        PT_out.insert(6, "IQR_Trees_P", Std_P)

    if ('Petrelli' in equationT or "Jorgenson" in equationT) and "onnx" not in equationT:
        PT_out.insert(len(PT_out.columns), "Median_Trees_T", Median_T)
        PT_out.insert(len(PT_out.columns), "Std_Trees_T", Std_T)
        PT_out.insert(len(PT_out.columns), "IQR_Trees_T", Std_T)





    if eq_tests is False:
        return PT_out
    if eq_tests is True:
        # Lets calcuate Cpx components
        cpx_components=calculate_clinopyroxene_components(cpx_comps=cpx_comps_c)
        out=pd.concat([PT_out, cpx_components],axis=1)
        return out


