import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd



from Thermobar.core import *

## Pressure equations for two pyroxenes


def P_Put2008_eq38(T=None, *, Na_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
Ti_Opx_cat_6ox, Ca_Opx_cat_6ox, Cr_Opx_cat_6ox, Mg_Opx_cat_6ox,
Fet_Opx_cat_6ox, Mn_Opx_cat_6ox, Ca_Cpx_cat_6ox, Fm2Si2O6, En_Opx, Di_Opx):
    '''

    Two pyroxene barometer of Putirka (2008) Eq38. Calibrated on Mg#-rich systems (>0.75)
    :cite:`putirka2008thermometers`

    | SEE=+-3.7 kbar
    '''
    Lindley_Fe3_Opx = (Na_Opx_cat_6ox + Al_IV_Opx_cat_6ox - Al_VI_Opx_cat_6ox - \
        2 * Ti_Opx_cat_6ox - Cr_Opx_cat_6ox)  # This is cell FR
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    a_En_opx_mod = (((0.5 * Mg_Opx_cat_6ox / (0.5 * (Fet_Opx_cat_6ox - Lindley_Fe3_Opx)
    + 0.5 * Mg_Opx_cat_6ox + Na_Opx_cat_6ox +Ca_Opx_cat_6ox + Mn_Opx_cat_6ox)))
    * (0.5 * Mg_Opx_cat_6ox / (0.5 * Mg_Opx_cat_6ox + 0.5 * (Fet_Opx_cat_6ox - Lindley_Fe3_Opx)
    + Ti_Opx_cat_6ox + Al_VI_Opx_cat_6ox + Cr_Opx_cat_6ox + Lindley_Fe3_Opx)))
    Kf = Ca_Opx_cat_6ox / (1 - Ca_Cpx_cat_6ox)
    return (-279.8 + 293 * Al_VI_Opx_cat_6ox + 455 * Na_Opx_cat_6ox + 229 * Cr_Opx_cat_6ox +
            519 * Fm2Si2O6 - 563 * En_Opx + 371 * Di_Opx + 327 * a_En_opx_mod + 1.19 / Kf)


def P_Put2008_eq39(T, *, Na_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
Ti_Opx_cat_6ox, Cr_Opx_cat_6ox, Fet_Opx_cat_6ox, Mn_Opx_cat_6ox, Ca_Opx_cat_6ox,
Mg_Opx_cat_6ox, Na_Cpx_cat_6ox, Al_IV_cat_6ox, Al_VI_cat_6ox, Ti_Cpx_cat_6ox,
Ca_Cpx_cat_6ox, Mg_Cpx_cat_6ox, Mn_Cpx_cat_6ox, Fet_Cpx_cat_6ox, Cr_Cpx_cat_6ox,
Fm2Si2O6, En_Opx, EnFs):
    '''

    Two pyroxene barometer of Putirka (2008) Eq39. Similar to Eq38, but
    has a temperature term.
    :cite:`putirka2008thermometers`

    | SEE=+-2.8 kbar (Cpx Mg#>0.75)
    | SEE=+-3.2 kbar (all data)
    '''
    Lindley_Fe3_Opx = Na_Opx_cat_6ox + Al_IV_Opx_cat_6ox - \
        Al_VI_Opx_cat_6ox - 2 * Ti_Opx_cat_6ox - Cr_Opx_cat_6ox
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    a_En_opx_mod = (((0.5 * Mg_Opx_cat_6ox / (0.5 * (Fet_Opx_cat_6ox - Lindley_Fe3_Opx)
    + 0.5 * Mg_Opx_cat_6ox + Na_Opx_cat_6ox +Ca_Opx_cat_6ox + Mn_Opx_cat_6ox)))
    * (0.5 * Mg_Opx_cat_6ox / (0.5 * Mg_Opx_cat_6ox + 0.5 * (Fet_Opx_cat_6ox - Lindley_Fe3_Opx)
    + Ti_Opx_cat_6ox + Al_VI_Opx_cat_6ox + Cr_Opx_cat_6ox + Lindley_Fe3_Opx)))

    Lindley_Fe3_Cpx = Na_Cpx_cat_6ox + Al_IV_cat_6ox - \
        Al_VI_cat_6ox - 2 * Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    a_Di_cpx = Ca_Cpx_cat_6ox / (Ca_Cpx_cat_6ox + 0.5 * Mg_Cpx_cat_6ox + 0.5 * (
        Fet_Cpx_cat_6ox - Lindley_Fe3_Cpx) + Mn_Cpx_cat_6ox + Na_Cpx_cat_6ox)
    Kf = Ca_Opx_cat_6ox / (1 - Ca_Cpx_cat_6ox)
    return (-94.25 + 0.045 * (T - 273.15) + 187.7 * Al_VI_Opx_cat_6ox + 246.8 * Fm2Si2O6 -
            212.5 * En_Opx + 127.5 * a_En_opx_mod - 69.4 * EnFs - 133.9 * a_Di_cpx - 1.66 / Kf)

## Temperature equations for two pyroxenes


def T_Put2008_eq36(P, *, EnFs, Fm2Si2O6, Ca_Cpx_cat_6ox,
                   CrCaTs, Mn_Opx_cat_6ox, Na_Opx_cat_6ox, En_Opx, Di_Opx):
    '''
    Two-pyroxene thermometer of Putirka (2008) eq 36. Best for Cpx with Mg#>0.75,
    but calibrated using all.
    :cite:`putirka2008thermometers`

    SEE=+-45C for Cpx Mg#>0.75
    SEE=+-56C for all data
    '''
    return (273.15 + 10 ** 4 / (11.2 - 1.96 * np.log(EnFs.astype(float) / Fm2Si2O6.astype(float)) - 3.3 * Ca_Cpx_cat_6ox - 25.8 *
            CrCaTs + 33.2 * Mn_Opx_cat_6ox - 23.6 * Na_Opx_cat_6ox - 2.08 * En_Opx - 8.33 * Di_Opx - 0.05 * P))




def T_Put2008_eq37(P, *, EnFs, Di_Cpx, Fm2Si2O6, Mn_Opx_cat_6ox,
                   FmAl2SiO6, Mg_Cpx_cat_6ox, Fet_Cpx_cat_6ox):
    '''
    Two-pyroxene thermometer of Putirka (2008) eq 37.
    Calibrated on Cpx with Mg#>0.75
    :cite:`putirka2008thermometers`

    SEE=+-38C for Cpx Mg#>0.75
    SEE=+-60C for all data
    '''
    return (273.15 + 10**4 / (13.4 - 3.4 * np.log(EnFs.astype(float) / Fm2Si2O6.astype(float)) + 5.59 * np.log(Mg_Cpx_cat_6ox.astype(float))
    + 23.85 * Mn_Opx_cat_6ox +6.48 * FmAl2SiO6 - 2.38 * Di_Cpx - 0.044 * P
    - 8.8 * Mg_Cpx_cat_6ox / (Mg_Cpx_cat_6ox + Fet_Cpx_cat_6ox)))

def T_Brey1990(P, *, Fet_Cpx_cat_6ox, Ca_Cpx_cat_6ox, Mg_Cpx_cat_6ox, Na_Cpx_cat_6ox,
               Fet_Opx_cat_6ox, Mg_Opx_cat_6ox, Ca_Opx_cat_6ox, Na_Opx_cat_6ox):
    '''
    Two-pyroxene thermometer of Brey and Kohler (1990).
    :cite:`brey1990geothermobarometry`

    SEE=+-50C for Cpx Mg#>0.75
    SEE=+-70C for all data
    '''
    return ((23664 + (24.9 + 126.3 * Fet_Cpx_cat_6ox / (Fet_Cpx_cat_6ox + Mg_Cpx_cat_6ox)) * P)
    / (13.38 + (np.log((1 - Ca_Cpx_cat_6ox.astype(float) /(1 - Na_Cpx_cat_6ox.astype(float))) /
    (1 - Ca_Opx_cat_6ox.astype(float) / (1 - Na_Opx_cat_6ox.astype(float)))))**2
    + 11.59 * Fet_Opx_cat_6ox / (Fet_Opx_cat_6ox + Mg_Opx_cat_6ox)))


def T_Wood1973(P=None, *, Mg_Opx_cat_6ox, Ca_Opx_cat_6ox, Mn_Opx_cat_6ox,
Fet_Opx_cat_6ox, Na_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
Ti_Opx_cat_6ox, Cr_Opx_cat_6ox, Mg_Cpx_cat_6ox, Ca_Cpx_cat_6ox,
Mn_Cpx_cat_6ox, Fet_Cpx_cat_6ox, Na_Cpx_cat_6ox, Al_IV_cat_6ox,
Al_VI_cat_6ox, Ti_Cpx_cat_6ox, Cr_Cpx_cat_6ox):
    '''
    Two-pyroxene thermometer of Wood and Banno (1973)
    :cite:`wood1973garnet`

    '''
    # Opx parts
    Lindley_Fe3_Opx = Na_Opx_cat_6ox + Al_IV_Opx_cat_6ox - Al_VI_Opx_cat_6ox - \
        2 * Ti_Opx_cat_6ox - Cr_Opx_cat_6ox  # This is cell FR
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    MgNo_WB_Opx = Mg_Opx_cat_6ox / \
        (Mg_Opx_cat_6ox + (Fet_Opx_cat_6ox - Lindley_Fe3_Opx))
    X_Mg_M2_Opx = (1 - Ca_Opx_cat_6ox - Na_Opx_cat_6ox -
                   Mn_Opx_cat_6ox) * MgNo_WB_Opx  # FL
    X_Fe_M2_Opx = (1 - Ca_Opx_cat_6ox - Na_Opx_cat_6ox -
                   Mn_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FM
    X_Mg_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   Ti_Opx_cat_6ox - Cr_Opx_cat_6ox) * MgNo_WB_Opx  # FJ
    X_Fe_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   Ti_Opx_cat_6ox - Cr_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FK
    MgNo_WB_Opx = Mg_Opx_cat_6ox / \
        (Mg_Opx_cat_6ox + (Fet_Opx_cat_6ox - Lindley_Fe3_Opx))
    a_opx_En = (X_Mg_M2_Opx / (X_Mg_M2_Opx + X_Fe_M2_Opx + Ca_Opx_cat_6ox +
    Na_Opx_cat_6ox + Mn_Opx_cat_6ox)) * \
        (X_Mg_M1_Opx / (Lindley_Fe3_Opx + Ti_Opx_cat_6ox +
         Al_VI_Opx_cat_6ox + Cr_Opx_cat_6ox + X_Mg_M1_Opx + X_Fe_M1_Opx))
    # Cpx parts
    Lindley_Fe3_Cpx = Na_Cpx_cat_6ox + Al_IV_cat_6ox - Al_VI_cat_6ox - \
        2 * Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox  # This is cell FR
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    Fe2_WB_Cpx = Fet_Cpx_cat_6ox - Lindley_Fe3_Cpx
    MgNo_WB_Cpx = Mg_Cpx_cat_6ox / (Mg_Cpx_cat_6ox + Fe2_WB_Cpx)
    X_Mg_M2_Cpx = (1 - Ca_Cpx_cat_6ox - Na_Cpx_cat_6ox -
                   Mn_Cpx_cat_6ox) * MgNo_WB_Cpx  # FL
    X_Fe_M2_Cpx = (1 - Ca_Cpx_cat_6ox - Na_Cpx_cat_6ox -
                   Mn_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FM
    X_Mg_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox) * MgNo_WB_Cpx  # FJ
    X_Fe_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FK
    a_cpx_En = (X_Mg_M2_Cpx / (X_Mg_M2_Cpx + X_Fe_M2_Cpx + Ca_Cpx_cat_6ox
    + Na_Cpx_cat_6ox + Mn_Cpx_cat_6ox)) * \
        (X_Mg_M1_Cpx / (Lindley_Fe3_Cpx + Ti_Cpx_cat_6ox +
         Al_VI_cat_6ox + Cr_Cpx_cat_6ox + X_Mg_M1_Cpx + X_Fe_M1_Cpx))

    return ((-10202 / (np.log(a_cpx_En.astype(float) / a_opx_En.astype(float)) - 7.65 *
            (1 - MgNo_WB_Opx) + 3.88 * (1 - MgNo_WB_Opx)**2 - 4.6)))


def T_Wells1977(P=None, *, Mg_Opx_cat_6ox, Ca_Opx_cat_6ox, Mn_Opx_cat_6ox,
Fet_Opx_cat_6ox, Na_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
 Ti_Opx_cat_6ox, Cr_Opx_cat_6ox, Mg_Cpx_cat_6ox, Ca_Cpx_cat_6ox,
 Mn_Cpx_cat_6ox, Fet_Cpx_cat_6ox, Na_Cpx_cat_6ox, Al_IV_cat_6ox,
 Al_VI_cat_6ox, Ti_Cpx_cat_6ox, Cr_Cpx_cat_6ox):
    '''
    Two-pyroxene thermometer of Wells 1977
    :cite:`wells1977pyroxene`

    '''
    # Opx parts
    Lindley_Fe3_Opx = Na_Opx_cat_6ox + Al_IV_Opx_cat_6ox - Al_VI_Opx_cat_6ox - \
        2 * Ti_Opx_cat_6ox - Cr_Opx_cat_6ox  # This is cell FR
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    MgNo_WB_Opx = Mg_Opx_cat_6ox / \
        (Mg_Opx_cat_6ox + (Fet_Opx_cat_6ox - Lindley_Fe3_Opx))
    X_Mg_M2_Opx = (1 - Ca_Opx_cat_6ox - Na_Opx_cat_6ox -
                   Mn_Opx_cat_6ox) * MgNo_WB_Opx  # FL
    X_Fe_M2_Opx = (1 - Ca_Opx_cat_6ox - Na_Opx_cat_6ox -
                   Mn_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FM
    X_Mg_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   Ti_Opx_cat_6ox - Cr_Opx_cat_6ox) * MgNo_WB_Opx  # FJ
    X_Fe_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   Ti_Opx_cat_6ox - Cr_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FK
    MgNo_WB_Opx = Mg_Opx_cat_6ox / \
        (Mg_Opx_cat_6ox + (Fet_Opx_cat_6ox - Lindley_Fe3_Opx))
    a_opx_En = (X_Mg_M2_Opx / (X_Mg_M2_Opx + X_Fe_M2_Opx + Ca_Opx_cat_6ox
    + Na_Opx_cat_6ox + Mn_Opx_cat_6ox)) * \
        (X_Mg_M1_Opx / (Lindley_Fe3_Opx + Ti_Opx_cat_6ox +
         Al_VI_Opx_cat_6ox + Cr_Opx_cat_6ox + X_Mg_M1_Opx + X_Fe_M1_Opx))
    # Cpx parts
    Lindley_Fe3_Cpx = Na_Cpx_cat_6ox + Al_IV_cat_6ox - Al_VI_cat_6ox - \
        2 * Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox  # This is cell FR
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    Fe2_WB_Cpx = Fet_Cpx_cat_6ox - Lindley_Fe3_Cpx
    MgNo_WB_Cpx = Mg_Cpx_cat_6ox / (Mg_Cpx_cat_6ox + Fe2_WB_Cpx)
    X_Mg_M2_Cpx = (1 - Ca_Cpx_cat_6ox - Na_Cpx_cat_6ox -
                   Mn_Cpx_cat_6ox) * MgNo_WB_Cpx  # FL
    X_Fe_M2_Cpx = (1 - Ca_Cpx_cat_6ox - Na_Cpx_cat_6ox -
                   Mn_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FM
    X_Mg_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox) * MgNo_WB_Cpx  # FJ
    X_Fe_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   Ti_Cpx_cat_6ox - Cr_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FK
    a_cpx_En = (X_Mg_M2_Cpx / (X_Mg_M2_Cpx + X_Fe_M2_Cpx + Ca_Cpx_cat_6ox + Na_Cpx_cat_6ox + Mn_Cpx_cat_6ox)) * \
        (X_Mg_M1_Cpx / (Lindley_Fe3_Cpx + Ti_Cpx_cat_6ox +
         Al_VI_cat_6ox + Cr_Cpx_cat_6ox + X_Mg_M1_Cpx + X_Fe_M1_Cpx))

    return ((7341 / (3.355 + 2.44 * (1 - MgNo_WB_Opx) - np.log(a_cpx_En.astype(float) / a_opx_En.astype(float)))))

## Function for calculating Cpx-Opx pressure

Cpx_Opx_P_funcs = {P_Put2008_eq38, P_Put2008_eq39} # put on outside

Cpx_Opx_P_funcs_by_name = {p.__name__: p for p in Cpx_Opx_P_funcs}

def calculate_cpx_opx_press(*, cpx_comps=None, opx_comps=None,
Two_Px_Match=None, equationP=None, eq_tests=False, T=None):
    '''
    calculates pressure in kbar for Opx-Cpx pairs

    The function requires inputs of cpx_comps and opx_comps, or input of a
    combined dataframe of cpx-opx compositions (this is used for the
    calculate_cpx_opx_press_temp_matching function)

    Parameters
    -----------

    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    opx_comps: pandas.DataFrame
       Opx compositions with column headings SiO2_Opx, MgO_Opx etc.

    Two_Px_Match: pandas.DataFrame
        Combined Cpx-Opx compositions instead of separate dataframes.
        Used for calculate Cpx_Opx_press_temp_matching function.

    equationP: str
        Choose from:

        |  P_Put2008_eq38 (T-independent)
        |  P_Put2008_eq39 (T-dependent)

    T: float, int, pd.Series, str  ("Solve")
        Temperature in Kelvin to perform calculations at.
        Only needed for T-sensitive barometers.
        If T="Solve", returns a partial function.
        Else, enter an integer, float, or panda series.

    eq_tests: bool
        If False, just returns temperature in K (default) as a panda series.
        If True, returns pressure in kbar, Kd Fe-Mg for opx-cpx,
        and the user-entered cpx and opx comps as a panda dataframe.


    Returns
    -------
    If eq_tests is False
        pandas.Series: Pressure in kbar
    If eq_tests is True
        pandas.DataFrame: Pressure in kbar + Kd-Fe-Mg + cpx+opx comps

    '''
    try:
        func = Cpx_Opx_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T, or specify T="Solve"')
    # else:
    #     if T is not None:
    #         print('Youve selected a T-independent function')

    if isinstance(T, pd.Series):
        if cpx_comps is not None:
            if len(T) != len(cpx_comps):
                raise ValueError('The panda series entered for temperature isnt the'
                ' same length as the dataframe of Cpx compositions')

    if Two_Px_Match is None:
        if len(opx_comps)!=len(cpx_comps):
            raise ValueError('Opx comps need to be same length as Cpx comps. use the _matching function calculate_cpx_opx_press_temp_matching() instead if you want to consider all pairs')


    if Two_Px_Match is not None:
        two_pyx = Two_Px_Match
    if cpx_comps is not None:
        cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps)
        opx_components = calculate_orthopyroxene_components(opx_comps=opx_comps)
        two_pyx = pd.concat([cpx_components, opx_components], axis=1)



    kwargs = {name: two_pyx[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(T, str) or T is None:
        if T == "Solve":
            P_kbar = partial(func, **kwargs)
        if T is None:
            P_kbar=func(**kwargs)

    else:
        P_kbar=func(T, **kwargs)

    if eq_tests is False:
        return P_kbar
    else:
        two_pyx = calculate_cpx_opx_eq_tests(
            cpx_comps=cpx_comps, opx_comps=opx_comps)
        two_pyx.insert(0, "P_kbar_calc", P_kbar)
        two_pyx.insert(2, "Equation Choice (P)", str(equationP))
        two_pyx.replace([np.inf, -np.inf], np.nan, inplace=True)
        return two_pyx



#--------------------Function for solving for temperature for two pyroxenes-----------------------------------------------------#
Cpx_Opx_T_funcs = {T_Put2008_eq36, T_Brey1990, T_Put2008_eq37, T_Wood1973, T_Wells1977} # put on outside

Cpx_Opx_T_funcs_by_name = {p.__name__: p for p in Cpx_Opx_T_funcs}


def calculate_cpx_opx_temp(*, cpx_comps=None, opx_comps=None,
                           Two_Px_Match=None, equationT=None, P=None, eq_tests=False):
    '''
    calculates temperature in Kelvin for Opx-Cpx pairs.

    The function requires inputs of cpx_comps and opx_comps, or input of a
    combined dataframe of cpx-opx compositions (this is used for the
    calculate_cpx_opx_press_temp_matching function).

    Parameters
    ------------

    cpx_comps: pandas.DataFrame
        Cpx compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    opx_comps: pandas.DataFrame
       Opx compositions with column headings SiO2_Opx, MgO_Opx etc.

    Two_Px_Match: pandas.DataFrame
        Combined Cpx-Opx compositions. Used for "melt match" functionality.

    equationT: str
        Choose from:

        |  T_Put2008_Eq36  (P-dependent)
        |  T_Put2008_Eq37 (P-dependent)
        |  T_Brey1990 (P-dependent)
        |  T_Wood1973 (P-independent)
        |  T_Wells1977 (P-independent)

    P: int, float, pandas.Series, str ("Solve")
        Pressure in kbar to perform calculations at.
        Can enter float or int to use same P for all calculations
        If "Solve", returns partial if function is P-dependent

    eq_tests: bool
        If False, just returns pressure in kbar (default) as a panda series
        If True, returns pressure in kbar, Kd Fe-Mg for opx-cpx, and the user-entered cpx and opx comps as a panda dataframe.


    Returns
    -------
    If eq_tests is False
        pandas.Series: Temperature in K
    If eq_tests is True
        pandas.DataFrame: Temperature in K + Kd-Fe-Mg + cpx + opx comps

    '''
    if Two_Px_Match is None:
        if len(opx_comps)!=len(cpx_comps):
            raise ValueError('Opx comps need to be same length as Cpx comps. use the _matching function calculate_cpx_opx_press_temp_matching() instead if you want to consider all pairs')

    try:
        func = Cpx_Opx_T_funcs_by_name[equationT]
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
        if cpx_comps is not None:
            if len(P) != len(cpx_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of cpx compositions')



    if Two_Px_Match is not None:
        two_pyx = Two_Px_Match
    if cpx_comps is not None:
        cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps)
        opx_components = calculate_orthopyroxene_components(opx_comps=opx_comps)
        two_pyx = pd.concat([cpx_components, opx_components], axis=1)

    kwargs = {name: two_pyx[name] for name, p in sig.parameters.items()
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
            return T_K

    else:
        two_pyx = calculate_cpx_opx_eq_tests(
            cpx_comps=cpx_comps, opx_comps=opx_comps)
        two_pyx.insert(0, "T_K_calc", T_K)
        two_pyx.insert(1, "Equation Choice (T)", str(equationT))

    return two_pyx

## Iterative calculations of P and T


def calculate_cpx_opx_press_temp(*, cpx_comps=None, opx_comps=None, Two_Px_Match=None,
                              equationP=None, equationT=None, iterations=30, T_K_guess=1300, eq_tests=False):
    '''
    Solves simultaneous equations for temperature and pressure using clinopyroxene-orthopyroxene
    thermometers and barometers.

    The function requires inputs of cpx_comps and opx_comps, or input of a
    combined dataframe of cpx-opx compositions (this is used for the
    calculate_cpx_opx_press_temp_matching function).

    Parameters
    -----------

    opx_comps: pandas.DataFrame
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    cpx_comps: pandas.DataFrame
        Cpx compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    meltmatch: pandas.DataFrame
        Combined dataframe of Opx-Cpx compositions (headings SiO2_Cpx, SiO2_Opx etc.).
        Used for calculate cpx_opx_press_temp_matching function.

    equationP: str
        Choose from:

        |  P_Put2008_eq38 (T-independent)
        |  P_Put2008_eq39 (T-dependent)

    equationT: str
        Choose from:

        |  T_Put2008_Eq36  (P-dependent)
        |  T_Put2008_Eq37 (P-dependent)
        |  T_Brey1990 (P-dependent)
        |  T_Wood1973 (P-independent)
        |  T_Wells1977 (P-independent)

    Optional:

    iterations: int, Default = 20
         Number of iterations used to converge to solution

    T_K_guess: int or float. Default = 1300K
         Initial guess of temperature to start iterations at

    eq_tests: bool
        If False, just returns pressure (default) as a panda series
        If True, returns pressure, Values of Eq tests,
        as well as user-entered opx and cpx comps and components.



    Returns
    -------
    If eq_tests is False
        pandas.DataFrame: Temperature in Kelvin, pressure in kbar
    If eq_tests is True
        pandas.DataFrame: Temperature in Kelvin, pressure in kbar, eq Tests + opx+cpx comps + components

    '''
    # Gives users flexibility to reduce or increase iterations

    if Two_Px_Match is None:
        if len(opx_comps)!=len(cpx_comps):
            raise ValueError('Opx comps need to be same length as Cpx comps. use the _matching function calculate_cpx_opx_press_temp_matching() instead if you want to consider all pairs')



    if Two_Px_Match is None:
        T_func = calculate_cpx_opx_temp(
            cpx_comps=cpx_comps, opx_comps=opx_comps, equationT=equationT, P="Solve")
        P_func = calculate_cpx_opx_press(
            cpx_comps=cpx_comps, opx_comps=opx_comps, equationP=equationP, T="Solve")
    if Two_Px_Match is not None:
        T_func = calculate_cpx_opx_temp(
            Two_Px_Match=Two_Px_Match, equationT=equationT, P="Solve")
        P_func = calculate_cpx_opx_press(
            Two_Px_Match=Two_Px_Match, equationP=equationP, T="Solve")

    # This bit checks if temperature is already a series - e.g., equations
    # with no pressure dependence
    if isinstance(T_func, pd.Series) and isinstance(P_func, pd.Series):
        P_guess = P_func
        T_K_guess = T_func

    if isinstance(T_func, pd.Series) and isinstance(P_func, partial):
        P_guess = P_func(T_func)
        T_K_guess = T_func

    if isinstance(P_func, pd.Series) and isinstance(T_func, partial):
        T_K_guess = T_func(P_func)
        P_guess = P_func

    if isinstance(P_func, partial) and isinstance(T_func, partial):

        # Gives users flexibility to add a different guess temperature
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

    T_K_guess_is_bad = (T_K_guess == 0) | (T_K_guess == 273.15) | (T_K_guess ==  -np.inf) | (T_K_guess ==  np.inf)
    T_K_guess[T_K_guess_is_bad] = np.nan
    P_guess[T_K_guess_is_bad] = np.nan


    if eq_tests is False:
        PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess,
                                    'Delta_P_kbar_Iter': DeltaP,
                                    'Delta_T_K_Iter': DeltaT})

        return PT_out
    if eq_tests is True:
        two_pyx = calculate_cpx_opx_eq_tests(
            cpx_comps=cpx_comps, opx_comps=opx_comps)
        DeltaKd_HighT= np.abs(
            1.09 - two_pyx['Kd_Fe_Mg_Cpx_Opx'])
        DeltaKd_SubSol= np.abs(
            0.7 - two_pyx['Kd_Fe_Mg_Cpx_Opx'])
        two_pyx.insert(0, "T_K_calc", T_K_guess)
        two_pyx.insert(1, "P_kbar_calc", P_guess)
        two_pyx.insert(2, "Equation Choice (T)", str(equationT))
        two_pyx.insert(3, "Equation Choice (P)", str(equationP))
        two_pyx.insert(4, 'Delta Kd High T', DeltaKd_HighT)
        two_pyx.insert(6, 'Delta Kd Low T', DeltaKd_SubSol)
        two_pyx.replace([np.inf, -np.inf], np.nan, inplace=True)
    return two_pyx
   # return P_func

## Two pyroxene matching

def calculate_cpx_opx_press_temp_matching(*, opx_comps, cpx_comps, equationT=None, equationP=None,
                                  Kd_Match=None, Kd_Err=None, Cpx_Quality=False, Opx_Quality=False, P=None, T=None,
                                  return_all_pairs=False, iterations=30):
    '''
    Evaluates all possible Cpx-Opx pairs for user supplied dataframes of opx and cpx
    comps (can be different lengths). Returns P (kbar) and T (K) for those in Kd Fe-Mg equilibrium.

    Parameters
    -----------

    opx_comps: pandas.DataFrame
        Panda DataFrame of opx compositions with column headings SiO2_Opx etc.

    cpx_comps: pandas.DataFrame
        Panda DataFrame of cpx compositions with column headings SiO2_Cpx etc.

    equationP: str
        Choose from:

        |  P_Put2008_eq38 (T-independent)
        |  P_Put2008_eq39 (T-dependent)

    equationT: str
        Choose from:

        |  T_Put2008_Eq36  (P-dependent)
        |  T_Put2008_Eq37 (P-dependent)
        |  T_Brey1990 (P-dependent)
        |  T_Wood1973 (P-independent)
        |  T_Wells1977 (P-independent)

    P, T: float, int, pandas.Series. Instead of specifying equationP or equationT

        P is pressure in kbar to perform calculations at if equationP not specified

        T is temperature in Kelvin to perform calculations at if equationT not specified

    Kd_Match: str
        |  If None, returns all cpx-opx pairs.
        |  If "HighTemp", returns all cpxs-opxs within Kd cpx-opx=1.09+-0.14 suggested by Putirka (2008)
        |  If "Subsolidus" returns all cpxs-opxs within Kd cpx-opx=0.7+-0.2 suggested by Putirka (2008)
        |  If int or float, also need to specify Kd_Err. Returns all matches within Kd_Match +- Kd_Err
    Or specify return_all_pairs=True to get all matches

    Kd_Err: float or int (defaults given in Kd_Match)
        Optional input to change defaults. Returns all cpx-opx pairs within Kd_Match+-Kd_Err

    Cpx_Quality: bool
        Default False, no filter
        If True, filters out clinopyroxenes with cation sums outside of 4.02-3.99 (after Neave et al. 2017)

    Opx_Quality: bool
        Default False, no filter
        If True, filters out orthopyroxenes with cation sums outside of 4.02-3.99

    Returns
    ----------
        Dict with keys: 'Av_PTs_per_Cpx', 'All_PTs'.

        'Av_PTs_perCpx': Average P and T for each cpx, e.g., if cpx1 matches Opx1,
        Opx4, Opx6, Opx10, returns mean and 1 sigma for each cpx.

        'All_PTs': Returns output parameters for all matches, e.g, cpx1-opx1,
        cpx1-opx4 without any averaging.

    '''
    if (Kd_Err is None and isinstance(Kd_Match, int)) or (Kd_Err is None and isinstance(Kd_Match, float)):
        raise Exception(
            'You have entered a numerical value for Kd_Match, but have not'
            'specified a Kd_Err to accept matches within Kd_Match+-Kd_Err')

    # calculating Cpx and opx components. Do before duplication to save
    # computation time
    myCPXs1_concat = calculate_clinopyroxene_components(cpx_comps=cpx_comps)
    myOPXs1_concat = calculate_orthopyroxene_components(opx_comps=opx_comps)

    # Adding an ID label to help with melt-cpx rematching later
    myCPXs1_concat['ID_CPX'] = myCPXs1_concat.index
    myOPXs1_concat['ID_OPX'] = myOPXs1_concat.index
    if "Sample_ID_Cpx" not in myCPXs1_concat:
        myCPXs1_concat['Sample_ID_Cpx'] = myCPXs1_concat.index
    if "Sample_ID_Opx" not in myOPXs1_concat:
        myOPXs1_concat['Sample_ID_Opx'] = myOPXs1_concat.index
    # Duplicate cpxs and opxs so end up with panda of all possible opx-cpx
    # matches

    # This duplicates CPXs, repeats cpx1-cpx1*N, cpx2-cpx2*N etc.
    DupCPXs = pd.DataFrame(np.repeat(myCPXs1_concat.values, np.shape(
        myOPXs1_concat)[0], axis=0))  # .astype('float64')
    DupCPXs.columns = myCPXs1_concat.columns

    # This duplicates opxs like opx1-opx2-opx3 for cpx1, opx1-opx2-opx3 for
    # cpx2 etc.
    DupOPXs = pd.concat([myOPXs1_concat] * np.shape(myCPXs1_concat)[0]).reset_index(drop=True)
    # Combines these merged opx and cpx dataframes
    Combo_opxs_cpxs = pd.concat([DupOPXs, DupCPXs], axis=1)

    Combo_opxs_cpxs_1 = Combo_opxs_cpxs.copy()
    LenCombo = str(np.shape(Combo_opxs_cpxs)[0])

    LenCpx=len(cpx_comps)
    LenOpx=len(opx_comps)
    print("Considering N=" + str(LenCpx) + " Cpx & N=" + str(LenOpx) +" Opx, which is a total of N="+ str(LenCombo) +
          " Cpx-Opx pairs, be patient if this is >>1 million!")



    # calculate Kd for these pairs
    En = (Combo_opxs_cpxs.Fm2Si2O6 * (Combo_opxs_cpxs.Mg_Opx_cat_6ox /
    (Combo_opxs_cpxs.Mg_Opx_cat_6ox +Combo_opxs_cpxs.Fet_Cpx_cat_6ox + Combo_opxs_cpxs.Mn_Cpx_cat_6ox)))

    Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'] = ((Combo_opxs_cpxs['Fet_Cpx_cat_6ox']
    / Combo_opxs_cpxs['Mg_Cpx_cat_6ox'])) / (Combo_opxs_cpxs['Fet_Opx_cat_6ox']
     / Combo_opxs_cpxs['Mg_Opx_cat_6ox'])

    if Kd_Match == "Subsolidus":
        #print('made it here')
        Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx'] = np.abs(
            0.7 - Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'])
        print(len(Combo_opxs_cpxs))

        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.loc[np.abs(
            Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx']) < 0.2]  # +- 0.2 suggested by Putirka spreadsheet
        #print(len(Combo_opxs_cpxs_1))
    if Kd_Match == "HighTemp":
        Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx'] = np.abs(
            1.09 - Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'])
        # +- 0.14 suggested by Putirka spreadsheet
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.loc[np.abs(
            Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx']) < 0.14]
    if isinstance(Kd_Match, int) or isinstance(
            Kd_Match, float) and Kd_Err is not None:
        Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx'] = np.abs(
            Kd_Match - Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'])
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.loc[np.abs(
            Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx']) < Kd_Err]  # +- 0.14 suggested by Putirka

    if return_all_pairs is True:
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.copy()
        print('No Kd selected, all matches are shown')

    if return_all_pairs is False and Kd_Match is None:
        raise Exception('You havent specified what Kd filter you want. Either enter Kd_Match= "Subsolidus", "HighTemp", or Kd_Match=value and Kd_Err=val, or  return_all_pairs=True')


    if len(Combo_opxs_cpxs_1) == 0:
        raise Exception('No matches found to the choosen Kd criteria.')

    Combo_opxs_cpxs_2 = Combo_opxs_cpxs_1.copy()
    # Uses neave method, filters out cation summs not between 4.02 and 3.99
    if Cpx_Quality is True and Opx_Quality is False:
        Combo_opxs_cpxs_2 = Combo_opxs_cpxs_1.loc[(Combo_opxs_cpxs_1['Cation_Sum_Cpx'] < 4.02) & (
            Combo_opxs_cpxs_1['Cation_Sum_Cpx'] > 3.99)]

    if Cpx_Quality is False and Opx_Quality is True:
        Combo_opxs_cpxs_2 = Combo_opxs_cpxs_1.loc[(Combo_opxs_cpxs_1['Cation_Sum_Opx'] < 4.02) & (
            Combo_opxs_cpxs_1['Cation_Sum_Opx'] > 3.99)]

    if Cpx_Quality is True and Opx_Quality is True:
        Combo_opxs_cpxs_2 = Combo_opxs_cpxs1.loc[(Combo_opxs_cpxs_1['Cation_Sum_Opx'] < 4.02) & (Combo_opxs_cpxs_1['Cation_Sum_Opx'] > 3.99)
                                                 & (Combo_opxs_cpxs_1['Cation_Sum_Cpx'] < 4.02) & (Combo_opxs_cpxs_1['Cation_Sum_Cpx'] > 3.99)]

    # Combo_opxs_cpxs_names= Combo_opxs_cpxs.copy() # Making a copy to keep
    # the names

    if len(Combo_opxs_cpxs_2) == 0:
        raise Exception('No matches found after Kd and quality filter.')


    Combo_opxs_cpxs_2_names = Combo_opxs_cpxs_2.copy()
    Combo_opxs_cpxs_2 = Combo_opxs_cpxs_2.drop(
        ['Sample_ID_Cpx', 'Sample_ID_Opx'], axis=1).astype('float64')

    # This gives users flexibility to input a constant P or T as well as
    # choosing equations
    if equationP is not None and P is not None:
        raise ValueError('You have entered an equation for P and specified a pressure. '
        ' The code doesnt know what you want it to do. Either enter an equation, or choose a pressure. ')
    if equationT is not None and T is not None:
        raise ValueError('You have entered an equation for T and specified a temperature. '
        'The code doesnt know what you want it to do. Either enter an equation, or choose a temperature.')
    if equationP is not None and equationT is not None:
        PT_out = calculate_cpx_opx_press_temp(
            Two_Px_Match=Combo_opxs_cpxs_2, equationP=equationP, equationT=equationT, iterations=iterations)
        Combo_opxs_cpxs_2.insert(0, "T_K_calc", PT_out['T_K_calc'])
        Combo_opxs_cpxs_2.insert(1, "P_kbar_calc", PT_out['P_kbar_calc'])
        Combo_opxs_cpxs_2.insert(2, "Delta_T_K_Iter", PT_out['Delta_T_K_Iter'].astype(float))
        Combo_opxs_cpxs_2.insert(3, "Delta_P_kbar_Iter", PT_out['Delta_P_kbar_Iter'].astype(float))


    if P is not None:
        T_K_calc = calculate_cpx_opx_temp(
            Two_Px_Match=Combo_opxs_cpxs_2, equationT=equationT, P=P)
        Combo_opxs_cpxs_2.insert(0, "T_K_calc", T_K_calc)
        Combo_opxs_cpxs_2.insert(1, "P_kbar_input", P)
        Combo_opxs_cpxs_2.insert(2, "Delta_T_K_Iter", 0)
        Combo_opxs_cpxs_2.insert(3, "Delta_P_kbar_Iter", 0)

    if T is not None:
        P_kbar_calc = calculate_cpx_opx_press(
            Two_Px_Match=Combo_opxs_cpxs_2, equationP=equationP, T=T)
        Combo_opxs_cpxs_2.insert(0, "T_K_input", T)
        Combo_opxs_cpxs_2.insert(1, "P_kbar_calc", P_kbar_calc)
        Combo_opxs_cpxs_2.insert(2, "Delta_T_K_Iter", 0)
        Combo_opxs_cpxs_2.insert(3, "Delta_P_kbar_Iter", 0)


    cols_to_move = ['Kd_Fe_Mg_Cpx_Opx']
    Combo_opxs_cpxs_2 = Combo_opxs_cpxs_2[cols_to_move + [
        col for col in Combo_opxs_cpxs_2.columns if col not in cols_to_move]]

    Combo_opxs_cpxs_2.insert(
        0, "Sample_ID_Opx", Combo_opxs_cpxs_2_names['Sample_ID_Opx'])
    Combo_opxs_cpxs_2.insert(
        1, "Sample_ID_Cpx", Combo_opxs_cpxs_2_names['Sample_ID_Cpx'])


        # Final step, calcuate a 3rd output which is the average and standard
        # deviation for each CPx (e.g., CPx1-Melt1, CPx1-melt3 etc. )
    CpxNumbers = Combo_opxs_cpxs_2['ID_CPX'].unique()
    Opx_sample_ID=Combo_opxs_cpxs_2["Sample_ID_Opx"]
    Combo_opxs_cpxs_2.drop(["Sample_ID_Opx"], axis=1, inplace=True)

    if len(CpxNumbers) > 0:
        df1_Mean_nopref=Combo_opxs_cpxs_2.groupby(['ID_CPX', 'Sample_ID_Cpx'], as_index=False).mean()
        df1_Std_nopref=Combo_opxs_cpxs_2.groupby(['ID_CPX', 'Sample_ID_Cpx'], as_index=False).std()
        count=Combo_opxs_cpxs_2.groupby('ID_CPX').count()
        Sample_ID_Cpx_Mean=df1_Mean_nopref['Sample_ID_Cpx']
        Sample_ID_Cpx_Std=df1_Std_nopref['Sample_ID_Cpx']
        df1_Mean=df1_Mean_nopref.add_prefix('Mean_')
        df1_Std=df1_Std_nopref.add_prefix('Std_')
        df1_Mean=df1_Mean.drop(['Mean_Sample_ID_Cpx'], axis=1)
        df1_Std=df1_Std.drop(['Std_Sample_ID_Cpx'], axis=1)
        df1_Mean.rename(columns={"Mean_ID_CPX": "ID_CPX"}, inplace=True)
        df1_Std.rename(columns={"Std_ID_CPX": "ID_CPX"}, inplace=True)

        df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_CPX'])
        df1_M['Sample_ID_Cpx']=Sample_ID_Cpx_Mean
        if equationT is not None and equationP is not None:
            cols_to_move = ['Sample_ID_Cpx',
                        'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']
        if equationT is not None and equationP is None:
            cols_to_move = ['Sample_ID_Cpx',
                        'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_input',
                        'Std_P_kbar_input']
        if equationT is None and equationP is not None:
            cols_to_move = ['Sample_ID_Cpx',
                        'Mean_T_K_input', 'Std_T_K_input', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']

        df1_M = df1_M[cols_to_move +
                    [col for col in df1_M.columns if col not in cols_to_move]]

    opxNumbers = Combo_opxs_cpxs_2['ID_OPX'].unique()
    Combo_opxs_cpxs_2['Sample_ID_Opx']=Opx_sample_ID
    Combo_opxs_cpxs_2.drop(["Sample_ID_Cpx"], axis=1, inplace=True)


    if len(opxNumbers) > 0:
        df1_2Mean_nopref=Combo_opxs_cpxs_2.groupby(['ID_OPX', 'Sample_ID_Opx'], as_index=False).mean()
        df1_2Std_nopref=Combo_opxs_cpxs_2.groupby(['ID_OPX', 'Sample_ID_Opx'], as_index=False).std()
        count=Combo_opxs_cpxs_2.groupby('ID_OPX').count()
        Sample_ID_Opx_Mean=df1_2Mean_nopref['Sample_ID_Opx']
        Sample_ID_Opx_Std=df1_2Std_nopref['Sample_ID_Opx']
        df1_2Mean=df1_2Mean_nopref.add_prefix('Mean_')
        df1_2Std=df1_2Std_nopref.add_prefix('Std_')
        df1_2Mean=df1_2Mean.drop(['Mean_Sample_ID_Opx'], axis=1)
        df1_2Std=df1_2Std.drop(['Std_Sample_ID_Opx'], axis=1)
        df1_2Mean.rename(columns={"Mean_ID_OPX": "ID_OPX"}, inplace=True)
        df1_2Std.rename(columns={"Std_ID_OPX": "ID_OPX"}, inplace=True)

        df1_2M=pd.merge(df1_2Mean, df1_2Std, on=['ID_OPX'])
        df1_2M['Sample_ID_Opx']=Sample_ID_Opx_Mean


        if equationT is not None and equationP is not None:
            cols_to_move = ['Sample_ID_Opx',
                        'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']
        if equationT is not None and equationP is None:
            cols_to_move = ['Sample_ID_Opx',
                        'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_input',
                        'Std_P_kbar_input']
        if equationT is None and equationP is not None:
            cols_to_move = ['Sample_ID_Opx',
                        'Mean_T_K_input', 'Std_T_K_input', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']



        df1_2M = df1_2M[cols_to_move +
                    [col for col in df1_2M.columns if col not in cols_to_move]]

    else:
        raise Exception(
            'No Matches - you may need to set less strict filters, e.g.,'
            'you could edit Kd_Match is None and Kd_Err to get more matches')



    print('Done!!! I found a total of N='+str(len(Combo_opxs_cpxs_2)) + ' Cpx-Opx matches using the specified filter. ')
    print('N=' + str(len(df1_M)) + ' Cpx out of the N='+str(LenCpx)+' Cpx that you input matched to 1 or more Opx')
    print( 'N=' + str(len(df1_2M)) + ' Opx out of the N='+str(LenOpx)+' Opx that you input matched to 1 or more Cpx')
    print('Done!')

    if equationT is not None:
        df1_M.insert(4, "Equation Choice (T)", str(equationT))
        df1_2M.insert(4, "Equation Choice (T)", str(equationT))
        Combo_opxs_cpxs_2.insert(4, "Equation Choice (T)", str(equationT))


    if equationP is not None:
        df1_M.insert(5, "Equation Choice (P)", str(equationP))
        df1_2M.insert(5, "Equation Choice (P)", str(equationP))
        Combo_opxs_cpxs_2.insert(5, "Equation Choice (P)", str(equationP))

    Combo_opxs_cpxs_2['Sample_ID_Opx']=Opx_sample_ID


    return {'Av_PTs_perCPX': df1_M, 'Av_PTs_perOPX': df1_2M, 'All_PTs': Combo_opxs_cpxs_2}



