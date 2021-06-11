import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
w.filterwarnings(
    "ignore", message="rubicon.objc.ctypes_patch has only been tested ")
w.filterwarnings("ignore", message="The handle")
w.simplefilter("once")

from Thermobar.core import *

# Liquid-Only Thermometers

# First, we define all the relevant equations as functions. P is in kbar,
# T is in Kelvin.


def T_Put2008_eq13(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 13 of Putirka et al. (2008)
    SEE=Â±72 Â°C
    '''
    return (26.3 * MgO_Liq + 994.4 + 273.15)


def T_Put2008_eq14(P=None, *, Mg_Number_Liq_NoFe3, MgO_Liq,
                   FeOt_Liq, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 14 of Putirka et al. (2008)
    SEE=Â±58 Â°C
    '''
    return (754 + 190.6 * Mg_Number_Liq_NoFe3 + 25.52 * MgO_Liq + 9.585 *
            FeOt_Liq + 14.87 * (Na2O_Liq + K2O_Liq) - 9.176 * H2O_Liq + 273.15)


def T_Put2008_eq15(P=None, *, Mg_Number_Liq_NoFe3, MgO_Liq,
                   FeOt_Liq, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 15 of Putirka et al. (2008). Pressure-dependent form of Equation 14.
    SEE=Â±46 Â°C
    '''
    return (815.3 + 265.5 * Mg_Number_Liq_NoFe3 + 15.37 * MgO_Liq + 8.61 * FeOt_Liq +
            6.646 * (Na2O_Liq + K2O_Liq) + 39.16 * 0.1 * P - 12.83 * H2O_Liq + 273.15)


def T_Put2008_eq16(P, *, SiO2_Liq_mol_frac,
                   Al2O3_Liq_mol_frac, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer for ol-cpx-plag-saturated liquids: Equation 16 of Putirka et al. (2008). Adapted from Yang et al (1996).
    SEE=Â±19 Â°C
    '''
    return (- 583 + 3141 * SiO2_Liq_mol_frac + 15779 * Al2O3_Liq_mol_frac + 1338.6 *
            MgO_Liq_mol_frac - 31440 * (Al2O3_Liq_mol_frac) * (SiO2_Liq_mol_frac) + 77.67 * 0.1 * P + 273.15)


def T_Helz1987_MgO(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 1 (MgO thermometer) of Helz and Thornber (1987)
    SEE=Â±10 Â°C
    '''
    return (20.1 * MgO_Liq + 1014 + 273.15)


def T_Montierth1995_MgO(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: update of Helz and Thornber (1987) by Montrieth et al. (1995).
    SEE=Â±10 Â°C
    '''
    return (23.0 * MgO_Liq + 1012 + 273.15)


def T_Helz1987_CaO(P=None, *, CaO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 2 (CaO thermometer) of Helz and Thornber (1987)
    SEE=Â±10 Â°C
    '''
    return (16.6 * CaO_Liq + 968 + 273.15)


def T_Beatt93_NoOl(P=None, *, Den_Beat93):
    '''
    Liquid-only thermometer  Re-arrangement of Beattie (1993) by Putirka (2008) such that an olivine composition isn't required (DMg ol-melt is calculated rather than measured.

    '''
    return (((113.1 * 1000) / 8.3144 + (0.1 * P * 10**9 - 10**5)
            * 4.11 * (10**(-6)) / 8.3144) / Den_Beat93)


def T_Beatt93_NoOl_HerzCorr(P=None, *, Den_Beat93):
    '''
    Liquid-only thermometer. Herzberg and O'Hara (2002) correction to the olivine-free re-arrangment (T_Beatt93_NoOl) of Beattie (1993) by Putirka. Eliminates systematic error at high pressures

    '''
    return (((113.1 * 1000 / 8.3144 + (0.0001 * (10**9) - 10**5) * 4.11 *
            (10**(-6)) / 8.3144) / Den_Beat93) + 54 * (0.1 * P) + 2 * (0.1 * P)**2)


def T_Put2008_eq19_BeattDMg(
        P=None, *, CalcDMg_Beat93, Beat_CNML, Beat_CSiO2L, Beat_NF):
    '''
    Liquid-only thermometer. Combining terms from Beattie et al. (1993) by Putirka (2008). This function uses calculated DMg from Beattie, so you don't need a measured olivine composition.
    '''
    return ((13603) + (4.943 * 10**(-7)) * ((0.1 * P) - 10**(-5))) / (6.26 + 2 *
                                                                      np.log(CalcDMg_Beat93) + 2 * np.log(1.5 * Beat_CNML) + 2 * np.log(3 * Beat_CSiO2L) - Beat_NF)


def T_Sug2000_eq1(P=None, *, MgO_Liq_mol_frac):
    return (1316 + 12.95 * MgO_Liq_mol_frac * 100)
    '''
    Liquid-only thermometer. Equation 1 of Sugawara et al. (2000) for olivine-saturated liquids at 0.1 MPa
    '''
# Coefficients form their Table 2 for olivine-liquid


def T_Sug2000_eq3_ol(P=None, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for olivine-saturated liquids. Unlike eq 1, this is pressure-dependent
    '''
    A = 1293
    B = 14.60
    C = 0.0055
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

# Coefficients form their Table 2 for opx-liquid


def T_Sug2000_eq3_opx(P=None, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for opx-saturated liquids.
    '''
    A = 1324
    B = 12.22
    C = 0.0052
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

# Coefficients form their Table 2 for cpx-liquid


def T_Sug2000_eq3_cpx(P=None, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for cpx-saturated liquids.
    '''
    A = 1289
    B = 12.35
    C = 0.0086
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

# Coefficients form their Table 2 for pig-liquid


def T_Sug2000_eq3_pig(P=None, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for pig-saturated liquids.
    '''
    A = 1294
    B = 13.47
    C = 0.0052
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))


def T_Sug2000_eq6a(P=None, *, SiO2_Liq_mol_frac,
                   FeOt_Liq_mol_frac, MgO_Liq_mol_frac, CaO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 6a of Sugawara et al. (2000) for olivine-saturated liquids. Adds terms for CaO, SiO2 and FeO relative to equation 1 and 3.
    '''
    return (1446 - 1.440 * SiO2_Liq_mol_frac * 100 - 0.5 * FeOt_Liq_mol_frac * 100 +
            12.32 * MgO_Liq_mol_frac * 100 - 3.899 * CaO_Liq_mol_frac * 100 + 0.0043 * (P * 1000))


def T_Sug2000_eq6b(P=None, *, SiO2_Liq_mol_frac,
                   FeOt_Liq_mol_frac, MgO_Liq_mol_frac, CaO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 6b of Sugawara et al. (2000) for cpx-saturated liquids. Adds terms for CaO, SiO2 and FeO relative to equation 3
    '''
    return (1202 + 1.511 * SiO2_Liq_mol_frac * 100 - 1.426 * FeOt_Liq_mol_frac * 100 +
            8.780 * MgO_Liq_mol_frac * 100 + 5.537 * CaO_Liq_mol_frac * 100 + 0.0081 * (P * 1000))

# These two functions allow equations 21 and 22 to be iterated for
# temperature, using DMg from 21, and Temp from 22


def calculate_DMg_eq21(T, *, P, H2O_Liq, Na2O_Liq, K2O_Liq):
    return (np.exp(-2.158 + 55.09 * (P / 10) / (T - 273.15) - 6.213 * 10**(-2) * H2O_Liq + 4430 / (T - 273.15)
                   + 5.115 * 10**(-2) * (Na2O_Liq + K2O_Liq)))


def T_Put2008_eq22_DMgIter(DMg, *, P, MgO_Liq_cat_frac, FeOt_Liq_cat_frac, CaO_Liq_cat_frac,
                           MnO_Liq_cat_frac, SiO2_Liq_cat_frac, Al2O3_Liq_cat_frac, TiO2_Liq_cat_frac, H2O_Liq):
    '''
    Liquid-only thermometer (adapted from ol-liq thermometer using calc DMg from Beattie) Putirka (2008), equation 22 (originally Putirka et al., 2007,  Eq 4). Recalibration of Beattie (1993) to account for the pressures sensitivity noted by Herzberg and O'Hara (2002), and eliminates the systematic error of Beattie (1993) for hydrous compositions.
    '''
    CNML = MgO_Liq_cat_frac + FeOt_Liq_cat_frac + CaO_Liq_cat_frac + MnO_Liq_cat_frac
    CSiO2L = SiO2_Liq_cat_frac
    NF = (7 / 2) * np.log(1 - Al2O3_Liq_cat_frac +
                          7 * np.log(1 - TiO2_Liq_cat_frac))

    return ((15294.6 + 1318.8 * 0.1 * P + 2.48348 * ((0.1 * P)**2)) / (8.048 + 2.8352 * np.log(DMg)
                                                                       + 2.097 * np.log(1.5 * CNML) + 2.575 * np.log(3 * CSiO2L) - 1.41 * NF + 0.222 * H2O_Liq + 0.5 * (0.1 * P)) + 273.15)

# Equation 21 and 22 - Using DMg Beattie


def T_Put2008_eq21_BeattDMg(
        P=None, *, CalcDMg_Beat93, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Liquid-only thermometer (adapted from ol-liq thermometer using calc DMg from Beattie) Putirka (2008), equation 21 (originally Putirka et al., 2007,  Eq 2). Recalibration of Beattie (1993) to account for the pressures sensitivity noted by Herzberg and O'Hara (2002), and eliminates the systematic error of Beattie (1993) for hydrous compositions.
    '''
    return (1 / ((np.log(CalcDMg_Beat93) + 2.158 - 5.115 * 10**(-2) * (Na2O_Liq +
            K2O_Liq) + 6.213 * 10**(-2) * H2O_Liq) / (55.09 * 0.1 * P + 4430)) + 273.15)


def T_Put2008_eq22_BeattDMg(
        P=None, *, CalcDMg_Beat93, Beat_CNML, Beat_CSiO2L, Beat_NF, H2O_Liq):
    '''
    Liquid-only thermometer (adapted from ol-liq thermometer using calc DMg from Beattie): Putirka (2008), equation 22 (originally Putirka et al., 2007,  Eq 4). Recalibration of Beattie (1993) to account for the pressures sensitivity noted by Herzberg and O'Hara (2002), and esliminates the systematic error of Beattie (1993) for hydrous compositions.
    '''
    return ((15294.6 + 1318.8 * 0.1 * P + 2.48348 * ((0.1 * P)**2)) / (8.048 + 2.8352 * np.log(CalcDMg_Beat93) + 2.097 *
            np.log(1.5 * Beat_CNML) + 2.575 * np.log(3 * Beat_CSiO2L) - 1.41 * Beat_NF + 0.222 * H2O_Liq + 0.5 * (0.1 * P)) + 273.15)

# Functions for saturation surfaces of minerals based just on liquids





def T_Molina2015_amp_sat(P=None, *, MgO_Liq_cat_frac,
                         CaO_Liq_cat_frac, Al2O3_Liq_cat_frac):
    return (273.15 + 107 * np.log(MgO_Liq_cat_frac) - 108 *
            np.log(CaO_Liq_cat_frac / (CaO_Liq_cat_frac + Al2O3_Liq_cat_frac)) + 1184)


# Note, can't find the exact eq. in 1999 - so have exaclty mimiced putirka
# spreadsheet, so even though 0.87 log term is really Mg#, not using Fe3+
def T_Put1999_cpx_sat(P, *, MgO_Liq_cat_frac, FeOt_Liq_cat_frac,
                      CaO_Liq_cat_frac, SiO2_Liq_cat_frac, Al2O3_Liq_cat_frac):
    '''
    Liquid-only thermometer, Putirka (1999). temperature at which a liquid is saturated in clinopyroxene (for a given P).
    '''
    return (10 ** 4 / (3.12 - 0.0259 * P - 0.37 * np.log(MgO_Liq_cat_frac / (MgO_Liq_cat_frac + FeOt_Liq_cat_frac))
                       + 0.47 * np.log(CaO_Liq_cat_frac * (MgO_Liq_cat_frac +
                                                           FeOt_Liq_cat_frac) * (SiO2_Liq_cat_frac)**2)
                       - 0.78 * np.log((MgO_Liq_cat_frac + FeOt_Liq_cat_frac)
                                       ** 2 * (SiO2_Liq_cat_frac)**2)
                       - 0.34 * np.log(CaO_Liq_cat_frac * (Al2O3_Liq_cat_frac)**2 * SiO2_Liq_cat_frac)))


def T_Put2008_eq34_cpx_sat(
        P, *, H2O_Liq, CaO_Liq_cat_frac, SiO2_Liq_cat_frac, MgO_Liq_cat_frac):
    '''
    Liquid-only thermometer- temperature at which a liquid is saturated in clinopyroxene (for a given P). Equation 34 of Putirka et al. (2008)
    '''
    return (10 ** 4 / (6.39 + 0.076 * H2O_Liq - 5.55 * (CaO_Liq_cat_frac * SiO2_Liq_cat_frac)
            - 0.386 * np.log(MgO_Liq_cat_frac) - 0.046 * P + 2.2 * (10 ** (-4)) * P**2))


def T_Put2008_eq28b_opx_sat(P, *, H2O_Liq, MgO_Liq_cat_frac, CaO_Liq_cat_frac, K2O_Liq_cat_frac, MnO_Liq_cat_frac,
                            FeOt_Liq_cat_frac, FeOt_Opx_cat_6ox, Al2O3_Liq_cat_frac, TiO2_Liq_cat_frac, Mg_Number_Liq_NoFe3):
    '''
    Liquid-only thermometer- temperature at which a liquid is saturated in orhopyroxene (for a given P). Equation 28b of Putirka et al. (2008)
    '''
    Cl_NM = MgO_Liq_cat_frac + FeOt_Liq_cat_frac + \
        CaO_Liq_cat_frac + MnO_Liq_cat_frac
    NF = (7 / 2) * np.log(1 - Al2O3_Liq_cat_frac) + \
        7 * np.log(1 - TiO2_Liq_cat_frac)
    return (273.15 + (5573.8 + 587.9 * (P / 10) - 61 * (P / 10)**2) / (5.3 - 0.633 * np.log(Mg_Number_Liq_NoFe3) - 3.97 * Cl_NM +
            0.06 * NF + 24.7 * CaO_Liq_cat_frac**2 + 0.081 * H2O_Liq + 0.156 * (P / 10)))


def T_Put2008_eq22_DMgIter2(P, *, DMg, MgO_Liq_cat_frac, FeOt_Liq_cat_frac, CaO_Liq_cat_frac,
                            MnO_Liq_cat_frac, SiO2_Liq_cat_frac, Al2O3_Liq_cat_frac, TiO2_Liq_cat_frac, H2O_Liq):
    CNML = MgO_Liq_cat_frac + FeOt_Liq_cat_frac + CaO_Liq_cat_frac + MnO_Liq_cat_frac
    CSiO2L = SiO2_Liq_cat_frac
    NF = (7 / 2) * np.log(1 - Al2O3_Liq_cat_frac +
                          7 * np.log(1 - TiO2_Liq_cat_frac))
    return ((15294.6 + 1318.8 * 0.1 * P + 2.48348 * ((0.1 * P)**2)) / (8.048 + 2.8352 * np.log(DMg)
                                                                       + 2.097 * np.log(1.5 * CNML) + 2.575 * np.log(3 * CSiO2L) - 1.41 * NF + 0.222 * H2O_Liq + 0.5 * (0.1 * P)) + 273.15)
