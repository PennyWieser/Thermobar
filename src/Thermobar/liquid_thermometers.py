import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from Thermobar.core import *


# Liquid-only thermometry functions

def T_Put2008_eq13(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 13 of Putirka et al. (2008)
    :cite:`putirka2008thermometers`

    SEE=±72 °C
    '''
    return (26.3 * MgO_Liq + 994.4 + 273.15)


def T_Shi_Test(P, *, MgO_Liq, SiO2_Liq, H2O_Liq):
    '''
    Testing the thermometer of Shi et al. (in prep)
    '''
    return (
    845.54899873 + (25.85511871*MgO_Liq)
+ (3.08839706*SiO2_Liq)
- (61.64200793*(np.log(1+H2O_Liq)))
+ (61.13449845*P/10)
)+273.15


def T_Put2008_eq14(P=None, *, Mg_Number_Liq_NoFe3, MgO_Liq,
                   FeOt_Liq, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 14 of Putirka et al. (2008)
    :cite:`putirka2008thermometers`

    SEE=±58 °C
    '''
    return (754 + 190.6 * Mg_Number_Liq_NoFe3 + 25.52 * MgO_Liq + 9.585 *
            FeOt_Liq + 14.87 * (Na2O_Liq + K2O_Liq) - 9.176 * H2O_Liq + 273.15)


def T_Put2008_eq15(P, *, Mg_Number_Liq_NoFe3, MgO_Liq,
                   FeOt_Liq, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 15 of Putirka et al. (2008).
    Pressure-dependent form of Equation 14.
    :cite:`putirka2008thermometers`

    SEE=±46 °C
    '''
    return (815.3 + 265.5 * Mg_Number_Liq_NoFe3 + 15.37 * MgO_Liq + 8.61 * FeOt_Liq +
            6.646 * (Na2O_Liq + K2O_Liq) + 39.16 * 0.1 * P - 12.83 * H2O_Liq + 273.15)


def T_Put2008_eq16(P, *, SiO2_Liq_mol_frac,
                   Al2O3_Liq_mol_frac, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer for ol-cpx-plag-saturated liquids:
    Equation 16 of Putirka et al. (2008). Adapted from Yang et al (1996).
    :cite:`putirka2008thermometers`

    SEE=±19 °C
    '''
    return (- 583 + 3141 * SiO2_Liq_mol_frac + 15779 * Al2O3_Liq_mol_frac + 1338.6 *
            MgO_Liq_mol_frac - 31440 * (Al2O3_Liq_mol_frac) * (SiO2_Liq_mol_frac) + 77.67 * 0.1 * P + 273.15)


def T_Helz1987_MgO(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids:
    Equation 1 (MgO thermometer) of Helz and Thornber (1987)
    :cite:``

    SEE=±10 °C
    '''
    return (20.1 * MgO_Liq + 1014 + 273.15)


def T_Shea2022_MgO(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids:
    Shea et al. (2022) adaptation of Helz and Thornber
    :cite:``

    SEE=±10 °C
    '''
    return (21.2 * MgO_Liq + 1017 + 273.15)

def T_Montierth1995_MgO(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids:
    update of Helz and Thornber (1987) by Montrieth et al. (1995).
    :cite:``

    SEE=±10 °C
    '''
    return (23.0 * MgO_Liq + 1012 + 273.15)


def T_Helz1987_CaO(P=None, *, CaO_Liq):
    '''
    Liquid-only thermometer for glasses with Ol+Augite+Plag
    Equation 2 (CaO thermometer) of Helz and Thornber (1987)
    :cite:``

    SEE=±10 °C
    '''
    return (16.6 * CaO_Liq + 968 + 273.15)


def T_Beatt93_BeattDMg(P, *, Den_Beat93):
    '''
    Liquid-only thermometer.
    Re-arrangement of Beattie (1993) by Putirka (2008) such that
    an olivine composition isn't required (DMg ol-melt is calculated rather than measured).
    :cite:``

    '''
    return (((113.1 * 1000) / 8.3144 + (0.1 * P * 10**9 - 10**5)
            * 4.11 * (10**(-6)) / 8.3144) / Den_Beat93)


def T_Beatt93_BeattDMg_HerzCorr(P, *, Den_Beat93):
    '''
    Liquid-only thermometer. Herzberg and O'Hara (2002) correction
    to the olivine-free re-arrangment (T_Beatt93_BeattDMg) of Beattie (1993) by Putirka.
    Eliminates systematic error at high pressures
    :cite:``

    '''
    return (((113.1 * 1000 / 8.3144 + (0.0001 * (10**9) - 10**5) * 4.11 *
            (10**(-6)) / 8.3144) / Den_Beat93) + 54 * (0.1 * P) + 2 * (0.1 * P)**2)




def T_Sug2000_eq1(P=None, *, MgO_Liq_mol_frac):

    '''
    Liquid-only thermometer. Equation 1 of Sugawara et al. (2000) for olivine-saturated liquids at 0.1 MPa
    :cite:``

    '''
    return (1316 + 12.95 * MgO_Liq_mol_frac * 100)
# Coefficients form their Table 2 for olivine-liquid


def T_Sug2000_eq3_ol(P, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for olivine-saturated liquids.
    Unlike eq 1, this is pressure-dependent
    :cite:``
    '''
    A = 1293
    B = 14.60
    C = 0.0055
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

# Coefficients form their Table 2 for opx-liquid


def T_Sug2000_eq3_opx(P, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for opx-saturated liquids.
    :cite:``
    '''
    A = 1324
    B = 12.22
    C = 0.0052
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

# Coefficients form their Table 2 for cpx-liquid


def T_Sug2000_eq3_cpx(P, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for cpx-saturated liquids.
    :cite:``
    '''
    A = 1289
    B = 12.35
    C = 0.0086
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

# Coefficients form their Table 2 for pig-liquid


def T_Sug2000_eq3_pig(P, *, MgO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 3 of Sugawara et al. (2000) for pig-saturated liquids.
    :cite:``
    '''
    A = 1294
    B = 13.47
    C = 0.0052
    return (A + B * MgO_Liq_mol_frac * 100 + C * (P * 1000))

def T_Sug2000_eq6a_H7a(P, *, SiO2_Liq_mol_frac,
                   FeOt_Liq_mol_frac, MgO_Liq_mol_frac, CaO_Liq_mol_frac, H2O_mol_frac):
    '''
    Liquid-only thermometer. Equation 6a of Sugawara et al. (2000) for
    olivine-saturated liquids. Adds terms for CaO, SiO2 and FeO relative to equation 1 and 3.
    Included corrections for H2O given in their equation 7a.
    :cite:``
    '''
    return ((1446 - 1.440 * SiO2_Liq_mol_frac * 100 - 0.5 * FeOt_Liq_mol_frac * 100 +
            12.32 * MgO_Liq_mol_frac * 100 - 3.899 * CaO_Liq_mol_frac * 100 + 0.0043 * (P * 1000))
            -5.403*100*H2O_mol_frac)

def T_Sug2000_eq6a(P, *, SiO2_Liq_mol_frac,
                   FeOt_Liq_mol_frac, MgO_Liq_mol_frac, CaO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 6a of Sugawara et al. (2000) for
    olivine-saturated liquids. Adds terms for CaO, SiO2 and FeO relative to equation 1 and 3.
    :cite:``

    '''
    return (1446 - 1.440 * SiO2_Liq_mol_frac * 100 - 0.5 * FeOt_Liq_mol_frac * 100 +
            12.32 * MgO_Liq_mol_frac * 100 - 3.899 * CaO_Liq_mol_frac * 100 + 0.0043 * (P * 1000))


def T_Sug2000_eq6b(P, *, SiO2_Liq_mol_frac,
                   FeOt_Liq_mol_frac, MgO_Liq_mol_frac, CaO_Liq_mol_frac):
    '''
    Liquid-only thermometer. Equation 6b of Sugawara et al. (2000) for cpx-saturated liquids.
    Adds terms for CaO, SiO2 and FeO relative to equation 3
    :cite:``
    '''
    return (1202 + 1.511 * SiO2_Liq_mol_frac * 100 - 1.426 * FeOt_Liq_mol_frac * 100 +
            8.780 * MgO_Liq_mol_frac * 100 + 5.537 * CaO_Liq_mol_frac * 100 + 0.0081 * (P * 1000))

def T_Sug2000_eq6b_H7b(P, *, SiO2_Liq_mol_frac,
                   FeOt_Liq_mol_frac, MgO_Liq_mol_frac, CaO_Liq_mol_frac, H2O_mol_frac):
    '''
    Liquid-only thermometer. Equation 6b of Sugawara et al. (2000) for cpx-saturated liquids.
    Adds terms for CaO, SiO2 and FeO relative to equation 3.
    Included corrections for H2O given in their equation 7b.
    :cite:``
    '''
    return ((1202 + 1.511 * SiO2_Liq_mol_frac * 100 - 1.426 * FeOt_Liq_mol_frac * 100 +
            8.780 * MgO_Liq_mol_frac * 100 + 5.537 * CaO_Liq_mol_frac * 100 + 0.0081 * (P * 1000))
            -5.674*100*H2O_mol_frac)

def T_Put2008_eq19_BeattDMg(P, *, calcDMg_Beat93, Beat_CNML, Beat_CSiO2L, Beat_NF):
    '''
    Liquid-only thermometer. Combining terms from Beattie et al. (1993) by Putirka (2008).
    This function uses calculated DMg from Beattie, so you don't need a measured olivine composition.
    :cite:`putirka2008thermometers`
    '''
    return ((13603) + (4.943 * 10**(-7)) * ((0.1 * P)*10**9 - 10**(-5))) / (6.26 + 2 *
            np.log(calcDMg_Beat93) + 2 * np.log(1.5 * Beat_CNML) + 2 * np.log(3 * Beat_CSiO2L) - Beat_NF)



def T_Put2008_eq21_BeattDMg(P, *, calcDMg_Beat93, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Liquid-only thermometer (adapted from ol-liq thermometer using calc DMg from Beattie) Putirka (2008),
    equation 21 (originally Putirka et al., 2007,  Eq 2). Recalibration of Beattie (1993) to account for the
    pressure sensitivity noted by Herzberg and O'Hara (2002), and eliminates the systematic error of Beattie (1993) for hydrous compositions.
    :cite:`putirka2008thermometers`

    '''
    return (1 / ((np.log(calcDMg_Beat93) + 2.158 - 5.115 * 10**(-2) * (Na2O_Liq +
            K2O_Liq) + 6.213 * 10**(-2) * H2O_Liq) / (55.09 * 0.1 * P + 4430)) + 273.15)


def T_Put2008_eq22_BeattDMg(P, *, calcDMg_Beat93, Beat_CNML, Beat_CSiO2L, Beat_NF, H2O_Liq):
    '''
    Liquid-only thermometer (adapted from ol-liq thermometer using calc DMg from Beattie): Putirka (2008),
    equation 22 (originally Putirka et al., 2007,  Eq 4). Recalibration of Beattie (1993) to account for the pressure
    sensitivity noted by Herzberg and O'Hara (2002), and eliminates the systematic error of Beattie (1993) for hydrous compositions.
    :cite:`putirka2008thermometers`
    '''
    return ((15294.6 + 1318.8 * 0.1 * P + 2.48348 * ((0.1 * P)**2)) / (8.048 + 2.8352 * np.log(calcDMg_Beat93) + 2.097 *
            np.log(1.5 * Beat_CNML) + 2.575 * np.log(3 * Beat_CSiO2L) - 1.41 * Beat_NF + 0.222 * H2O_Liq + 0.5 * (0.1 * P)) + 273.15)

## Functions for saturation surfaces of minerals based just on liquids


def T_Molina2015_amp_sat(P=None, *, Mg_Liq_cat_frac,
                         Ca_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Amphibole-saturation thermometer from Molina et al. (2015)
    :cite:``
    '''

    return (273.15 + 107 * np.log(Mg_Liq_cat_frac) - 108 *
            np.log(Ca_Liq_cat_frac / (Ca_Liq_cat_frac + Al_Liq_cat_frac)) + 1184)

def T_Put2016_eq3_amp_sat(P=None, *, FeOt_Liq_mol_frac_hyd, CaO_Liq_mol_frac_hyd, SiO2_Liq_mol_frac_hyd,
                          TiO2_Liq_mol_frac_hyd, MgO_Liq_mol_frac_hyd, MnO_Liq_mol_frac_hyd, Al2O3_Liq_mol_frac_hyd):
    '''
    Equation 3 of Putirka et al. (2016)
    Amphibole-Liquid thermometer- temperature at which a liquid is saturated in amphibole.
    :cite:``
    '''
    return (273.15 + (24429.2) / (2.31 + 42.1 * FeOt_Liq_mol_frac_hyd
    + 32.2 * CaO_Liq_mol_frac_hyd + 2.21 * np.log(SiO2_Liq_mol_frac_hyd) -
    1.4 * np.log(TiO2_Liq_mol_frac_hyd)
    - 2.666 * np.log((FeOt_Liq_mol_frac_hyd + MgO_Liq_mol_frac_hyd) * Al2O3_Liq_mol_frac_hyd)))


def T_Put1999_cpx_sat(P, *, Mg_Liq_cat_frac, Fet_Liq_cat_frac,
                      Ca_Liq_cat_frac, Si_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Liquid-only thermometer, Putirka (1999). temperature at which a liquid is saturated
    in clinopyroxene (for a given P).
    '''
    return (10 ** 4 / (3.12 - 0.0259 * P - 0.37 * np.log(Mg_Liq_cat_frac / (Mg_Liq_cat_frac + Fet_Liq_cat_frac))
                       + 0.47 * np.log(Ca_Liq_cat_frac * (Mg_Liq_cat_frac +
                                                           Fet_Liq_cat_frac) * (Si_Liq_cat_frac)**2)
                       - 0.78 * np.log((Mg_Liq_cat_frac + Fet_Liq_cat_frac)
                                       ** 2 * (Si_Liq_cat_frac)**2)
                       - 0.34 * np.log(Ca_Liq_cat_frac * (Al_Liq_cat_frac)**2 * Si_Liq_cat_frac)))


def T_Put2008_eq34_cpx_sat(P, *, H2O_Liq, Ca_Liq_cat_frac, Si_Liq_cat_frac, Mg_Liq_cat_frac):
    '''
    Liquid-only thermometer- temperature at which a liquid is saturated in clinopyroxene (for a given P). Equation 34 of Putirka et al. (2008)
    '''
    return (10 ** 4 / (6.39 + 0.076 * H2O_Liq - 5.55 * (Ca_Liq_cat_frac * Si_Liq_cat_frac)
            - 0.386 * np.log(Mg_Liq_cat_frac) - 0.046 * P + 2.2 * (10 ** (-4)) * P**2))


def T_Beatt1993_opx(P, *, Ca_Liq_cat_frac, Fet_Liq_cat_frac, Mg_Liq_cat_frac,
                    Mn_Liq_cat_frac, Al_Liq_cat_frac, Ti_Liq_cat_frac):
    '''
    Opx-Liquid thermometer of Beattie (1993). Only uses liquid composition.
    Putirka (2008) warn that overpredicts for hydrous compositions at <1200°C, and anhydrous compositions at <1100°C
    '''
    Num_B1993 = 125.9 * 1000 / 8.3144 + \
        ((0.1 * P) * 10**9 - 10**5) * 6.5 * (10**(-6)) / 8.3144
    D_Mg_opx_li1 = (0.5 - (-0.089 * Ca_Liq_cat_frac - 0.025 * Mn_Liq_cat_frac + 0.129 * Fet_Liq_cat_frac)) / \
        (Mg_Liq_cat_frac + 0.072 * Ca_Liq_cat_frac +
         0.352 * Mn_Liq_cat_frac + 0.264 * Fet_Liq_cat_frac)
    Cl_NM = Mg_Liq_cat_frac + Fet_Liq_cat_frac + \
        Ca_Liq_cat_frac + Mn_Liq_cat_frac
    NF = (7 / 2) * np.log(1 - Al_Liq_cat_frac) + \
        7 * np.log(1 - Ti_Liq_cat_frac)
    Den_B1993 = 67.92 / 8.3144 + 2 * \
        np.log(D_Mg_opx_li1) + 2 * np.log(2 * Cl_NM) - NF
    return Num_B1993 / Den_B1993




def T_Put2005_eqD_plag_sat(P, *, Ca_Liq_cat_frac,
                           H2O_Liq, Si_Liq_cat_frac, Al_Liq_cat_frac):
    '''
    Plagioclase-Liquid saturation temperature thermometer of Putirka (2005) eq. D
    '''
    return ((10**4 / (8.759 - 6.396 * Ca_Liq_cat_frac + 0.2147 * H2O_Liq + 1.221 *
            Si_Liq_cat_frac**3 - 1.751 * 10**-2 * P - 8.043 * Al_Liq_cat_frac)))


def T_Put2008_eq26_plag_sat(P, *, Si_Liq_cat_frac,
                            Ca_Liq_cat_frac, Al_Liq_cat_frac, K_Liq_cat_frac, H2O_Liq):
    '''
    Plagioclase-Liquid saturation temperature thermometer of Putirka (2008) eq. 26. Update of P2005_eqD
    SEE=+-37C
    '''
    return (10**4 / (10.86 - 9.7654 * Si_Liq_cat_frac + 4.241 * Ca_Liq_cat_frac
                     - 55.56 * Ca_Liq_cat_frac * Al_Liq_cat_frac +
                     37.5 * K_Liq_cat_frac * Al_Liq_cat_frac
                     + 11.206 * Si_Liq_cat_frac**3 - 3.151 * 10**(-2) * (P / 10) * 10 + 0.1709 * H2O_Liq))

def T_Put2008_eq24c_kspar_sat(P, *, Al_Liq_cat_frac, Na_Liq_cat_frac,
                  K_Liq_cat_frac, Si_Liq_cat_frac, Ca_Liq_cat_frac, H2O_Liq):
    '''
    Alkali Felspar saturation thermometer of Putirka (2008) eq. 24b.
    SEE=+-23 C (Calibration data)
    SEE=+-25 C (All data)

    '''
    return (10**4 / (14.6 + 0.055 * H2O_Liq - 0.06 * (P / 10) / 10 - 99.6 * Na_Liq_cat_frac * Al_Liq_cat_frac
    - 2313 * Ca_Liq_cat_frac *Al_Liq_cat_frac + 395 * K_Liq_cat_frac * Al_Liq_cat_frac
    - 151 * K_Liq_cat_frac * Si_Liq_cat_frac + 15037 * Ca_Liq_cat_frac**2))


def T_Put2008_eq28b_opx_sat(P, *, H2O_Liq, Mg_Liq_cat_frac, Ca_Liq_cat_frac, K_Liq_cat_frac, Mn_Liq_cat_frac,
                            Fet_Liq_cat_frac, Fet_Opx_cat_6ox, Al_Liq_cat_frac, Ti_Liq_cat_frac, Mg_Number_Liq_NoFe3):
    '''
    Equation 28b of Putirka et al. (2008). Orthopyroxene-liquid thermometer- temperature at which a liquid is saturated in orhopyroxene (for a given P).
    :cite:`putirka2008thermometers`
    '''
    Cl_NM = Mg_Liq_cat_frac + Fet_Liq_cat_frac + \
        Ca_Liq_cat_frac + Mn_Liq_cat_frac
    NF = (7 / 2) * np.log(1 - Al_Liq_cat_frac.astype(float)) + \
        7 * np.log(1 - Ti_Liq_cat_frac.astype(float))
    return (273.15 + (5573.8 + 587.9 * (P / 10) - 61 * (P / 10)**2) / (5.3 - 0.633 * np.log(Mg_Number_Liq_NoFe3.astype(float)) - 3.97 * Cl_NM +
            0.06 * NF + 24.7 * Ca_Liq_cat_frac**2 + 0.081 * H2O_Liq + 0.156 * (P / 10)))


## Listing them all to check for invalid inputs- add new ones into this list so they become recognised.

Liquid_only_funcs = {T_Put2008_eq13, T_Put2008_eq14, T_Put2008_eq15, T_Put2008_eq16,
T_Helz1987_MgO, T_Montierth1995_MgO, T_Helz1987_CaO, T_Beatt93_BeattDMg,
T_Beatt93_BeattDMg_HerzCorr, T_Sug2000_eq1, T_Sug2000_eq3_ol, T_Sug2000_eq3_opx,
T_Sug2000_eq3_cpx, T_Sug2000_eq3_pig,
T_Sug2000_eq6a, T_Sug2000_eq6a_H7a, T_Sug2000_eq6b, T_Sug2000_eq6b_H7b, T_Put2008_eq19_BeattDMg, T_Put2008_eq21_BeattDMg,
T_Put2008_eq22_BeattDMg, T_Molina2015_amp_sat, T_Put2016_eq3_amp_sat,
T_Put2008_eq34_cpx_sat, T_Put2008_eq28b_opx_sat,
T_Put1999_cpx_sat, T_Put2008_eq26_plag_sat, T_Put2005_eqD_plag_sat, T_Put2008_eq24c_kspar_sat, T_Beatt1993_opx, T_Shi_Test, T_Shea2022_MgO} # put on outside

Liquid_only_funcs_by_name = {p.__name__: p for p in Liquid_only_funcs}


def calculate_liq_only_temp(*, liq_comps, equationT, P=None, H2O_Liq=None, print=False):

    '''
    Liquid-only thermometery. Returns a temperature in Kelvin.

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    equationT: str
        If has _sat at the end, represents the saturation surface of that mineral.

        Equations from Putirka et al. (2016).
            | T_Put2016_eq3_amp_sat (saturation surface of amphibole)

        Equations from Putirka (2008) and older studies:

            | T_Put2008_eq13
            | T_Put2008_eq14
            | T_Put2008_eq15
            | T_Put2008_eq16
            | T_Put2008_eq34_cpx_sat
            | T_Put2008_eq28b_opx_sat
            | T_Put1999_cpx_sat
            * Following 3 thermometers are adaptations of olivine-liquid thermometers with  DMg calculated using Beattie 1993,
            This means you can use them without knowing an olivine composition. ocan be applied when you haven't measured an olivine composiiton.
            | T_Put2008_eq19_BeattDMg
            | T_Put2008_eq21_BeattDMg
            | T_Put2008_eq22_BeattDMg

        Equations from Sugawara (2000):

            | T_Sug2000_eq1
            | T_Sug2000_eq3_ol
            | T_Sug2000_eq3_opx
            | T_Sug2000_eq3_cpx
            | T_Sug2000_eq3_pig
            | T_Sug2000_eq6a
            | T_Sug2000_eq6b

        Equations from Helz and Thornber (1987):

            | T_Helz1987_MgO
            | T_Helz1987_CaO

        Equation from Molina et al. (2015)

            | T_Molina2015_amp_sat

        Equation from Montrieth 1995
           | T_Montierth1995_MgO

        Equation from Beattie (1993)
           | T_Beatt1993_opx

    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    H2O_Liq: optional.
        If None, uses H2O_Liq column from input.
        If int, float, pandas.Series, uses this instead of H2O_Liq Column


    Returns
    -------
    pandas series
       Temperature in K

    '''


# This checks if your function is one of the accepted liquid equations
    try:
        func = Liquid_only_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)


    if isinstance(P, pd.Series):
        if len(P) != len(liq_comps):
            raise ValueError('The panda series entered for pressure isnt the same length as the dataframe of liquid compositions')


    liq_comps_c = liq_comps.copy()

    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq']=H2O_Liq
        if print is True:
            print('Water content replaced with that from H2O_Liq')

# Keiths Liq-only spreadsheet doesn't use Cr2O3 and P2O5 to calc cat. frac. So have set this to zero.
    if equationT != "T_Put2008_eq26_plag_sat" and equationT != "T_Put2008_eq24c_kspar_sat" \
    and equationT !="T_Put2005_eqD_plag_sat" and equationT != "T_Molina2015_amp_sat" and equationT != "T_Put2016_eq3_amp_sat":
        liq_comps_c['Cr2O3_Liq']=0
        liq_comps_c['P2O5_Liq']=0





# Now calculate cation fractions - if using putirka, uses hydrous cat fracs.
    if equationT == "T_Put2016_eq3_amp_sat":
        anhyd_cat_frac = calculate_hydrous_mol_fractions_liquid(liq_comps=liq_comps_c)
    else:
        anhyd_cat_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)

    if equationT=="T_Sug2000_eq6a_H7a" or equationT=="T_Sug2000_eq6b_H7b":
        anhyd_cat_frac['H2O_mol_frac']=calculate_hydrous_mol_fractions_liquid(liq_comps=liq_comps_c).H2O_Liq_mol_frac_hyd



# This performs extra calculation steps for Beattie equations
    if equationT == "T_Beatt93_BeattDMg" or equationT == "T_Beatt93_BeattDMg_HerzCorr" or equationT == "T_Put2008_eq19_BeattDMg" \
    or equationT == "T_Put2008_eq21_BeattDMg" or equationT == "T_Put2008_eq22_BeattDMg":
        anhyd_cat_frac['calcDMg_Beat93'] = (0.666 - (-0.049 * anhyd_cat_frac['Mn_Liq_cat_frac']
        + 0.027 * anhyd_cat_frac['Fet_Liq_cat_frac'])) / (
        1 * anhyd_cat_frac['Mg_Liq_cat_frac'] + 0.259 * anhyd_cat_frac['Mn_Liq_cat_frac']
        + 0.299 * anhyd_cat_frac['Fet_Liq_cat_frac'])

        anhyd_cat_frac['Beat_CNML'] = (anhyd_cat_frac['Mg_Liq_cat_frac'] + anhyd_cat_frac['Fet_Liq_cat_frac'] +
            anhyd_cat_frac['Ca_Liq_cat_frac'] + anhyd_cat_frac['Mn_Liq_cat_frac'])
        anhyd_cat_frac['Beat_CSiO2L'] = anhyd_cat_frac['Si_Liq_cat_frac']
        anhyd_cat_frac['Beat_NF'] = ( 7 / 2) * np.log(1 - anhyd_cat_frac['Al_Liq_cat_frac']) + 7 * np.log(1 - anhyd_cat_frac['Ti_Liq_cat_frac'])
        anhyd_cat_frac['Den_Beat93'] = 52.05 / 8.3144 + 2 * np.log(anhyd_cat_frac['calcDMg_Beat93']) + 2 * np.log(
            1.5 * anhyd_cat_frac['Beat_CNML']) + 2 * np.log(3 * anhyd_cat_frac['Beat_CSiO2L']) - anhyd_cat_frac['Beat_NF']

# Checks if P-dependent function you have entered a P
    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')

    #else:
        # print('gothere2')
        # if P is not None:
        #     print('gothere3')
        #     print('Youve selected a P-independent function')
        #     P=None
        #     print('got here')



    kwargs = {name: anhyd_cat_frac[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:
        T_K=func(P, **kwargs)

    return T_K
