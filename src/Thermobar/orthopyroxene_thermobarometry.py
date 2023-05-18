import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd


from Thermobar.core import *
from Thermobar.liquid_thermometers import*

## Opx-Liquid barometers



def P_Put2008_eq29a(T, *, Si_Liq_cat_frac, Mg_Liq_cat_frac, Fet_Opx_cat_6ox, FmAl2SiO6,
                    Na_Liq_cat_frac, Al_Liq_cat_frac, K_Liq_cat_frac, H2O_Liq, NaAlSi2O6):
    '''
    Orthopyroxene-Liquid barometer of Putirka, (2008) eq 29a. Global calibration of experiments.
    :cite:`putirka2008thermometers`

    SEE=+-2.6 kbar (all data)
    SEE=+-2.1 kbar for hydrous data

    '''
    Na_Si_Al_Na=(NaAlSi2O6 / (Si_Liq_cat_frac**2 * Al_Liq_cat_frac * Na_Liq_cat_frac)).astype(float)
    log_Na_Si_Al_Na=np.log(Na_Si_Al_Na)
    return (-13.97 + 0.0129 * (T - 273.15) - 19.64 * Si_Liq_cat_frac + 47.49 * Mg_Liq_cat_frac + 6.99 * Fet_Opx_cat_6ox
            + 37.37 * FmAl2SiO6 + 0.748 * H2O_Liq + 79.67 * (Na_Liq_cat_frac + K_Liq_cat_frac) +
            0.001416 * (T - 273.15)*log_Na_Si_Al_Na)



def P_Put2008_eq29b(T, *, ln_FmAl2SiO6_liq, Al_Liq_cat_frac, Mg_Liq_cat_frac,
Fet_Liq_cat_frac, Si_Opx_cat_6ox, Fet_Opx_cat_6ox,
Na_Liq_cat_frac, K_Liq_cat_frac, H2O_Liq):
    '''
    Orthopyroxene-Liquid barometer of Putirka, (2008) eq 29b. Global calibration of experiments.
    :cite:`putirka2008thermometers`

    Exact SEE not given, but ~2-3 kbar.

    '''
    return (1.788 + 0.0375 * (T - 273.15) + 0.001295 * (T - 273.15) * ln_FmAl2SiO6_liq - 33.42 * Al_Liq_cat_frac
            + 9.795 * Mg_Liq_cat_frac /
            (Mg_Liq_cat_frac + Fet_Liq_cat_frac) - 26.2 *
            Si_Opx_cat_6ox + 14.21 * Fet_Opx_cat_6ox
            + 36.08 * (Na_Liq_cat_frac + K_Liq_cat_frac) + 0.784 * H2O_Liq)


def P_Put_Global_Opx(T=None, *, MgO_Liq, Al2O3_Opx, Al2O3_Liq, Na2O_Liq, K2O_Liq):
    '''
    New Opx-Liquid barometer released in Putirka spreadsheets.
    Addresses problem in low pressure Opxs that Al(VI)=0. Uses the Al2O3 content of the Opx instead.
    :cite:`putirka2008thermometers`

    SEE=+-3.2 kbar

    '''
    return ((-8.51 + 0.856 * MgO_Liq - 1.14 * Al2O3_Opx + 45.474 *
            Al2O3_Opx / Al2O3_Liq + 1.067 * (Na2O_Liq + K2O_Liq)))


def P_Put_Felsic_Opx(T=None, *, Al2O3_Opx, Al2O3_Liq):
    '''
    New Opx_Liq barometer released in Putirka spreadsheets.
    Addresses problem in low pressure Opxs that Al(VI)=0. Uses the Al2O3 content of the Opx instead.
    Felsic regression.
    :cite:`putirka2008thermometers`

    |  SEE=+-1.2 kbar

    '''
    return ((-0.892 + 31.81 * Al2O3_Opx / Al2O3_Liq))



## Opx-Only Barometers

def P_Put2008_eq29c(T, *, Al_Opx_cat_6ox,
                    Ca_Opx_cat_6ox, Cr_Opx_cat_6ox):
    '''
    Orthopyroxene-only barometer of Putirka, (2008) eq 29c. Doesn't require liquid composition.
    Global calibration of experiments, has systematic error for hydrous data.
    :cite:`putirka2008thermometers`

    SEE=+-3 kbar (anhydrous)

    SEE=+-4.1 kbar (hydrous)

    '''
    logCr2O3 = np.log(Cr_Opx_cat_6ox.astype(float))

    #print(logCr2O3)
    #print(logCr2O3)


    return (2064 + 0.321 * (T - 273.15) - 343.4 * np.log((T - 273.15)) + 31.52 * Al_Opx_cat_6ox - 12.28 * Ca_Opx_cat_6ox
            - 290 * Cr_Opx_cat_6ox - 177.2 *
            (Al_Opx_cat_6ox - 0.1715)**2 - 372 *
            (Al_Opx_cat_6ox - 0.1715) * (Ca_Opx_cat_6ox - 0.0736)
            + 1.54 * logCr2O3)


def P_Put2008_eq29cnoCr(T, *, Al_Opx_cat_6ox,
                    Ca_Opx_cat_6ox, Cr_Opx_cat_6ox):
    '''
    Orthopyroxene-only barometer of Putirka, (2008) eq 29c. Doesn't require liquid composition.
    Global calibration of experiments, has systematic error for hydrous data.
    :cite:`putirka2008thermometers`

    SEE=+-3 kbar (anhydrous)

    SEE=+-4.1 kbar (hydrous)

    '''
    logCr2O3 = np.log(Cr_Opx_cat_6ox.astype(float))
    #print(logCr2O3)


    return (2064 + 0.321 * (T - 273.15) - 343.4 * np.log((T - 273.15)) + 31.52 * Al_Opx_cat_6ox - 12.28 * Ca_Opx_cat_6ox
            - 290 * Cr_Opx_cat_6ox - 177.2 *
            (Al_Opx_cat_6ox - 0.1715)**2 - 372 *
            (Al_Opx_cat_6ox - 0.1715) * (Ca_Opx_cat_6ox - 0.0736))

## Opx-Liquid thermometers



def T_Put2008_eq28a(P, *, H2O_Liq, ln_Fm2Si2O6_liq, Mg_Liq_cat_frac,
                    K_Liq_cat_frac, Fet_Liq_cat_frac, Fet_Opx_cat_6ox):
    """
    Putirka (2008) Equation 28a.
    Global calibration: T=750-1600°C, SiO2=33-77 wt%, P=atm-11 GPa. H2O=0-14.2 wt%.
    :cite:`putirka2008thermometers`

    SEE= ±26°C for calibration data

    SEE=± 41°C for testing data
    """
    return (273.15 + 10**4 / (4.07 - 0.329 * (0.1 * P) + 0.12 * H2O_Liq +
    0.567 * ln_Fm2Si2O6_liq.astype(float) - 3.06 * Mg_Liq_cat_frac -
    6.17 * K_Liq_cat_frac + 1.89 * Mg_Liq_cat_frac /
    (Mg_Liq_cat_frac + Fet_Liq_cat_frac) + 2.57 * Fet_Opx_cat_6ox))

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


def T_Beatt1993_opx(P, *, Ca_Liq_cat_frac, Fet_Liq_cat_frac, Mg_Liq_cat_frac,
                    Mn_Liq_cat_frac, Al_Liq_cat_frac, Ti_Liq_cat_frac):
    '''
    Opx-Liquid thermometer of Beattie (1993). Only uses liquid composition.
    Putirka (2008) warn that overpredicts for hydrous compositions at <1200°C, and anhydrous compositions at <1100°C
    :cite:`beattie1993olivine`
    '''
    Num_B1993 = 125.9 * 1000 / 8.3144 + \
        ((0.1 * P) * 10**9 - 10**5) * 6.5 * (10**(-6)) / 8.3144
    D_Mg_opx_li1 = (0.5 - (-0.089 * Ca_Liq_cat_frac - 0.025 * Mn_Liq_cat_frac + 0.129 * Fet_Liq_cat_frac)) / \
        (Mg_Liq_cat_frac + 0.072 * Ca_Liq_cat_frac +
         0.352 * Mn_Liq_cat_frac + 0.264 * Fet_Liq_cat_frac)
    Cl_NM = Mg_Liq_cat_frac + Fet_Liq_cat_frac + \
        Ca_Liq_cat_frac + Mn_Liq_cat_frac
    NF = (7 / 2) * np.log(1 - Al_Liq_cat_frac.astype(float)) + \
        7 * np.log(1 - Ti_Liq_cat_frac.astype(float))
    Den_B1993 = 67.92 / 8.3144 + 2 * \
        np.log(D_Mg_opx_li1.astype(float)) + 2 * np.log(2 * Cl_NM.astype(float)) - NF
    return Num_B1993 / Den_B1993

##  Opx-Only barometry function
Opx_only_P_funcs = {P_Put2008_eq29c, P_Put2008_eq29cnoCr} # put on outside

Opx_only_P_funcs_by_name = {p.__name__: p for p in Opx_only_P_funcs}


def calculate_opx_only_press(*, opx_comps, equationP, T=None):
    '''
    Orthopyroxene only barometry. Enter a panda dataframe with orthopyroxene compositions, returns a pressure in kbar.

   Parameters
    -------

    opx_comps: pandas.DataFrame
        orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    equationP: str
        | P_Put2008_eq29c

    T: float, int, pandas.Series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas series
       Pressure in kbar


    '''
    try:
        func = Opx_only_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None

    sig=inspect.signature(func)

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T, or specify T="Solve"')
    # else:
    #     if T is not None:
    #         print('Youve selected a T-independent function')

    opx_comps = calculate_orthopyroxene_components(opx_comps=opx_comps)
    if equationP != "P_Put2008_eq29c":
        raise ValueError('Equation not recognised, at the moment the only choice is P_Put2008_eq29c')


    P_func = P_Put2008_eq29c
    if any(opx_comps['Cr_Opx_cat_6ox'] == 0):
        w.warn('The selected barometer uses the log of Cr2O3 component of '
        'Opx, which is zero for some of your compositions. '
         'This means the function will return infinity.')

    kwargs = {name: opx_comps[name] for name, p in inspect.signature(
        P_func).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

    if isinstance(T, str) or T is None:
        if T == "Solve" or T is None:
            P_kbar = partial(P_func, **kwargs)

        if T == "input":
            T = liq_comps['T_K']
            P_kbar = P_func(T, **kwargs)
    else:
        T = T
        P_kbar = P_func(T, **kwargs)

    P_kbar.replace([np.inf, -np.inf], np.nan,inplace=True)



    return P_kbar

## Orthopyroxene-Liquid pressure

Opx_Liq_P_funcs = {P_Put2008_eq29a, P_Put2008_eq29b, P_Put_Global_Opx, P_Put_Felsic_Opx, P_Put2008_eq29c, P_Put2008_eq29cnoCr} # put on outside

Opx_Liq_P_funcs_by_name = {p.__name__: p for p in Opx_Liq_P_funcs}

def calculate_opx_liq_press(*, equationP, opx_comps=None, liq_comps=None, meltmatch=None,
                            T=None, eq_tests=False, H2O_Liq=None, Fe3Fet_Liq=None):
    '''
    Orthopyroxene-Liquid barometer, user specifies equation, and calculates pressure in kbar.
    Also has option to calculate equilibrium tests.

    Parameters
    -------

    opx_comps: pandas.DataFrame
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    Or:

    meltmatch: pandas.DataFrame
        Combined Opx-Liquid compositions.
        Used for calculate_opx_liq_press_temp_matching.

    EquationP: str

        choose from:

        |  P_Put2008_eq28a
        |  P_Put2008_eq28b
        |  P_Put2008_eq28c
        |  P_Put_Global_Opx
        |  P_Put_Felsic_Opx


    T: float, int, pandas.Series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool
        If False, just returns pressure (default) as a panda series
        If True, returns pressure, Values of Eq tests,
        as well as user-entered opx and liq comps and components.

    Returns
    -------
    If eq_tests=False
        pandas.Series: Pressure in kbar (if eq_tests=False)
    If eq_tests=True
        pandas.DataFrame: Pressure in kbar + Kd-Fe-Mg + opx+liq comp

    '''
# This checks if your equation is one of the accepted equations
    try:
        func = Opx_Liq_P_funcs_by_name[equationP]
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

    if meltmatch is None and liq_comps is not None and opx_comps is not None:
        if len(liq_comps)!=len(opx_comps):
            raise ValueError('opx comps need to be same length as liq comps. use a _matching function instead if you want to consider all pairs: calculate_opx_liq_press_temp_matching')


    if meltmatch is not None:
        Combo_liq_opxs = meltmatch
    if liq_comps is not None:
        Combo_liq_opxs = calculate_orthopyroxene_liquid_components(
            liq_comps=liq_comps_c, opx_comps=opx_comps)

    kwargs = {name: Combo_liq_opxs[name] for name, p in sig.parameters.items() \
    if p.kind == inspect.Parameter.KEYWORD_ONLY}



    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T')
    # else:
    #     if T is not None:
    #         print('Youve selected a T-independent function')


    kwargs = {name: Combo_liq_opxs[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
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
            P_kbar.replace([np.inf, -np.inf], np.nan,inplace=True)
            return P_kbar

    if eq_tests is True:
        P_kbar.replace([np.inf, -np.inf], np.nan, inplace=True)
        Combo_liq_opxs.insert(1, "P_kbar_calc", P_kbar)
        Combo_liq_opxs.insert(2, "eq_tests_Kd_Fe_Mg_Fet",
                              Combo_liq_opxs['Kd_Fe_Mg_Fet'])
        Combo_liq_opxs.insert(3, "eq_tests_Kd_Fe_Mg_Fe2",
                              Combo_liq_opxs['Kd_Fe_Mg_Fe2'])
        Combo_liq_opxs.replace([np.inf, -np.inf], np.nan, inplace=True)

        return Combo_liq_opxs


## Opx-Liquid temperature

Opx_Liq_T_funcs = {T_Put2008_eq28a, T_Put2008_eq28b_opx_sat, T_Beatt1993_opx}

Opx_Liq_T_funcs_by_name = {p.__name__: p for p in Opx_Liq_T_funcs}
def calculate_opx_liq_temp(*, equationT, opx_comps=None, liq_comps=None, meltmatch=None,
                           P=None, eq_tests=False, Fe3Fet_Liq=None, H2O_Liq=None):
    '''
    Orthopyroxene-Liquid thermometer, user specifies equation,
    and calculates temperature in Kelvin.  Also has option to calculate equilibrium tests.

    Parameters
    -------

    opx_comps: pandas.DataFrame
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    meltmatch: pandas.DataFrame
        Combined Opx-Liquid compositions. Used for "melt match" functionality.

    EquationT: str
        Choice of equation:
        |  T_Opx_Beatt1993
        |  T_Put2008_eq28a
        |  T_Put2008_eq28b_opx_sat


    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool
        If False (default), returns temperature as a panda series
        If True, returns prsesure, Kd Fe-Mg for liq-opx,
        as well as user-entered opx and liq comps as a panda dataframe.

    Returns
    -------
    If eq_tests=False
        pandas.Series: Pressure in kbar (if eq_tests=False)
    If eq_tests=True
        pandas.DataFrame: Pressure in kbar + Kd-Fe-Mg + opx+liq comps


    '''
    try:
        func = Opx_Liq_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if meltmatch is None and liq_comps is not None and opx_comps is not None:
        if len(liq_comps)!=len(opx_comps):
            raise ValueError('opx comps need to be same length as liq comps. use a _matching function instead if you want to consider all pairs: calculate_opx_liq_press_temp_matching')

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
        Combo_liq_opxs = meltmatch

    if liq_comps is not None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None and not isinstance(H2O_Liq, str):
            liq_comps_c['H2O_Liq'] = H2O_Liq
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
        Combo_liq_opxs = calculate_orthopyroxene_liquid_components(
            liq_comps=liq_comps_c, opx_comps=opx_comps)


    kwargs = {name: Combo_liq_opxs[name] for name, p in sig.parameters.items()
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


    if eq_tests is True:

        Combo_liq_opxs.insert(0, "T_K_calc", T_K)
        Combo_liq_opxs.insert(1, "eq_tests_Kd_Fe_Mg_Fet",
                              Combo_liq_opxs['Kd_Fe_Mg_Fet'])
        Combo_liq_opxs.insert(2, "eq_tests_Kd_Fe_Mg_Fe2",
                              Combo_liq_opxs['Kd_Fe_Mg_Fe2'])
        Combo_liq_opxs.replace([np.inf, -np.inf], np.nan, inplace=True)

        return Combo_liq_opxs

    return T_K

## Iterating P and T when you don't know either
def calculate_opx_liq_press_temp(*, liq_comps=None, opx_comps=None, meltmatch=None, equationP=None, equationT=None,
                              iterations=30, T_K_Guess=1300, eq_tests=False, H2O_Liq=None, Fe3Fet_Liq=None):
    '''
    Solves simultaneous equations for temperature and pressure using
    orthopyroxene-liquid thermometers and barometers.

    Parameters
    -------

    opx_comps: pandas.DataFrame
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    EquationP: str
        Barometer
        |  P_Put2008_eq28a
        |  P_Put2008_eq28b
        |  P_Put2008_eq28c
        |  P_Put_Global
        |  P_Put_Felsic

    EquationT: str
        Thermometer
        |  T_Opx_Beatt1993
        |  T_Put2008_eq28a
        |  T_Put2008_eq28b_opx_sat

    Optional:


    iterations: int (default=30)
         Number of iterations used to converge to solution.

    T_K_guess: int or float  (default=1300K)
         Initial guess of temperature. Default is 1300K

    eq_tests: bool
        If False, just returns pressure in Kbar, temp in Kelvin as a dataframe
        If True, returns pressure, temperature, Values of Eq tests,
        as well as user-entered opx and liq comps and components.


    Returns
    -------
    If eq_tests=False
        pandas.DataFrame: Temperature in Kelvin, pressure in Kbar
    If eq_tests=True
        pandas.DataFrame: Temperature in Kelvin, pressure in Kbar
        Eq Tests + opx+liq comps + components

    '''
    # Gives users flexibility to reduce or increase iterations

    if meltmatch is None and liq_comps is not None and opx_comps is not None:
        if len(liq_comps)!=len(opx_comps):
            raise ValueError('opx comps need to be same length as liq comps. use a _matching function instead if you want to consider all pairs: calculate_opx_liq_press_temp_matching')

    if iterations is not None:
        iterations = iterations
    else:
        iterations = 30

    if T_K_Guess is not None:
        T_K_guess = T_K_Guess
    else:
        T_K_guess = 1300

    if meltmatch is None:
        liq_comps_c = liq_comps.copy()

        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq

        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

        T_func = calculate_opx_liq_temp(
            opx_comps=opx_comps, liq_comps=liq_comps_c, equationT=equationT, P="Solve")
        P_func = calculate_opx_liq_press(
            opx_comps=opx_comps, liq_comps=liq_comps_c, equationP=equationP, T="Solve")

    if meltmatch is not None:

        T_func = calculate_opx_liq_temp(
            meltmatch=meltmatch, equationT=equationT, P="Solve")
        P_func = calculate_opx_liq_press(
            meltmatch=meltmatch, equationP=equationP, T="Solve")

        # Gives users flexibility to add a different guess temperature

    if isinstance(P_func, pd.Series) and isinstance(T_func, partial):
        P_guess = P_func
        T_K_guess = T_func(P_guess)
    if isinstance(T_func, pd.Series) and isinstance(P_func, partial):
        T_K_guess = T_func
        P_guess = P_func(T_K_guess)
    if isinstance(T_func, pd.Series) and isinstance(P_func, pd.Series):
        T_K_guess = T_func
        P_gues = P_func


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

# This gets rid of any stray Nans, 0s, 0C etc.
    T_K_guess_is_bad = (T_K_guess == 0) | (T_K_guess == 273.15) | (T_K_guess ==  -np.inf) | (T_K_guess ==  np.inf)
    T_K_guess[T_K_guess_is_bad] = np.nan
    P_guess[T_K_guess_is_bad] = np.nan


    # calculates Kd Fe-Mg if eq_tests="True"
    if eq_tests is False:
        PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess,
                                    'Delta_P_kbar_Iter': DeltaP,
                                    'Delta_T_K_Iter': DeltaT})

        return PT_out
    if eq_tests is True and meltmatch is None:
        Combo_liq_opxs = calculate_orthopyroxene_liquid_components(
            opx_comps=opx_comps, liq_comps=liq_comps_c)
        Combo_liq_opxs.insert(0, "P_kbar_calc", P_guess)
        Combo_liq_opxs.insert(1, "T_K_calc", T_K_guess)
        Combo_liq_opxs.insert(2, 'Delta_P_kbar_Iter', DeltaP)
        Combo_liq_opxs.insert(3, 'Delta_T_K_Iter',  DeltaT)
        Combo_liq_opxs.insert(4, "eq_tests_Kd_Fe_Mg_Fet",
                              Combo_liq_opxs['Kd_Fe_Mg_Fet'])
        Combo_liq_opxs.insert(5, "eq_tests_Kd_Fe_Mg_Fe2",
                              Combo_liq_opxs['Kd_Fe_Mg_Fe2'])
    if eq_tests is True and meltmatch is not None:
        Combo_liq_opxs = meltmatch.copy()
        Combo_liq_opxs.insert(0, "P_kbar_calc", P_guess)
        Combo_liq_opxs.insert(1, "T_K_calc", T_K_guess)
        Combo_liq_opxs.insert(2, 'Delta_P_kbar_Iter', DeltaP)
        Combo_liq_opxs.insert(3, 'Delta_T_K_Iter',  DeltaT)
        Combo_liq_opxs.insert(4, "eq_tests_Kd_Fe_Mg_Fet",
                              meltmatch['Kd_Fe_Mg_Fet'])
        Combo_liq_opxs.insert(5, "eq_tests_Kd_Fe_Mg_Fe2",
                              meltmatch['Kd_Fe_Mg_Fe2'])

    return Combo_liq_opxs

## Considering all possible Orthopyroxene-melt pairs, and iterating P and T

def calculate_opx_liq_press_temp_matching(*, liq_comps, opx_comps, equationT=None,
equationP=None, P=None, T=None, eq_crit=False, Fe3Fet_Liq=None, H2O_Liq=None,
 Kd_Match=None, Kd_Err=None, Opx_Quality=False, return_all_pairs=False, iterations=30):

    '''
    Evaluates all possible Opx-Liq pairs from  N Liquids, M opx compositions
    returns P (kbar) and T (K) for those in equilibrium.

    Parameters
    -----------

    liq_comps: pandas.DataFrame
        Panda DataFrame of liquid compositions with column headings SiO2_Liq etc.

    opx_comps: pandas.DataFrame
        Panda DataFrame of opx compositions with column headings SiO2_Opx etc.

    EquationP: str
        Barometer
        |  P_Put2008_eq28a
        |  P_Put2008_eq28b
        |  P_Put2008_eq28c
        |  P_Put_Global
        |  P_Put_Felsic

    EquationT: str
        Thermometer
        |  T_Opx_Beatt1993
        |  T_Put2008_eq28a
        |  T_Put2008_eq28b_opx_sat

    Or:

    P: int, float
        Can also specify a pressure to run calculations at, rather than iterating
        using an equation for pressure. E.g., specify an equationT, but no equationP

    T: int, float
        Can also specify a temperature to run calculations at, rather than iterating
        using an equation for temperature.  E.g., specify an equationP, but no equationT

    Optional:


    Kd_Match: int of float, optional
        Allows users to ovewrite the default where Kd is calculated from the
        expression in Putirka (2008) based on the Si content of the liquid.

    Kd_Err: int or float, optional
        Allows users to override the defualt 1 sigma on Kd matches of +-0.06


    Opx Quality: bool, optional
        If True, filters out orthopyroxenes with cation sums outside of 4.02-3.99 (after Neave et al. 2017)

   Fe3Fet_Liq: int or float, optional
        Fe3FeT ratio used to assess Kd Fe-Mg equilibrium between opx and melt.
        If users don't specify, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    Returns: dict

        Av_PTs: Average P and T for each opx.
        E.g., if opx1 matches Liq1, Liq4, Liq6, Liq10, averages outputs for all 4 of those liquids.
        Returns mean and 1 sigma of these averaged parameters for each Opx.

        All_PTs: Returns output parameters for all matches (e.g, opx1-Liq1, opx1-Liq4) without any averaging.

    '''
    # This checks that inputs are consistent, and not contradictory
    if equationP is not None and P is not None:
        raise ValueError('You have entered an equation for P and specified a pressure. '
        ' The code doesnt know what you want it to do. Either enter an equation, or choose a pressure. ')
    if equationT is not None and T is not None:
        raise ValueError('You have entered an equation for T and specified a temperature. '
        'The code doesnt know what you want it to do. Either enter an equation, or choose a temperature.  ')

    # This over-writes inputted Fe3Fet_Liq and H2O_Liq inputs.
    liq_comps_c = liq_comps.copy()
    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
    if H2O_Liq is not None and not isinstance(H2O_Liq, str):
        liq_comps_c['H2O_Liq'] = H2O_Liq
    if "Fe3Fet_Liq" not in liq_comps:
        liq_comps_c['Fe3Fet_Liq'] = 0


    # Adding sample names if there aren't any
    if "Sample_ID_Liq" not in liq_comps:
        liq_comps_c['Sample_ID_Liq'] = liq_comps_c.index
    if "Sample_ID_Opx" not in opx_comps:
        opx_comps['Sample_ID_Opx'] = opx_comps.index




    # calculating Opx and liq components. Do before duplication to save
    # computation time
    myOPXs1_concat = calculate_orthopyroxene_components(opx_comps=opx_comps)
    myLiquids1_concat = calculate_anhydrous_cat_fractions_liquid(
        liq_comps=liq_comps_c)

    # Adding an ID label to help with melt-opx rematching later
    myOPXs1_concat['Sample_ID_Opx'] = opx_comps['Sample_ID_Opx']
    myLiquids1_concat['Sample_ID_Liq'] = liq_comps_c['Sample_ID_Liq']
    myOPXs1_concat['ID_OPX'] = myOPXs1_concat.index
    myLiquids1_concat['ID_Liq'] = myLiquids1_concat.index


    # This duplicates OPXs, repeats opx1-opx1*N, opx2-opx2*N etc.
    DupOPXs = pd.DataFrame(
        np.repeat(myOPXs1_concat.values, np.shape(myLiquids1_concat)[0], axis=0))
    DupOPXs.columns = myOPXs1_concat.columns

    # This duplicates liquids like liq1-liq2-liq3 for opx1, liq1-liq2-liq3 for
    # opx2 etc.
    DupLiqs = pd.concat([myLiquids1_concat] *
                        np.shape(myOPXs1_concat)[0]).reset_index(drop=True)
    # Combines these merged liquids and opx dataframes
    Combo_liq_opxs = pd.concat([DupLiqs, DupOPXs], axis=1)

    # calculate clinopyroxene-liquid components for this merged dataframe
    Combo_liq_opxs = calculate_orthopyroxene_liquid_components(
        meltmatch=Combo_liq_opxs)
   # Combo_liq_opxs.drop(['Kd Eq (Put2008+-0.06)'], axis=1, inplace=True)

    #Combo_liq_opxs = Combo_liq_opxs.convert_objects(convert_numeric=True)
    LenCombo = str(np.shape(Combo_liq_opxs)[0])

    LenOpx=len(opx_comps)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenOpx) + " Opx & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(LenCombo) +
          " Liq-Opx pairs, be patient if this is >>1 million!")


    if return_all_pairs is False:


        # Filters using the method of Neave et al. 2017
        if Opx_Quality is True:
            Combo_liq_opxs_2 = Combo_liq_opxs.loc[(Combo_liq_opxs['Cation_Sum_Opx'] < 4.02) & (
                Combo_liq_opxs['Cation_Sum_Opx'] > 3.99)]
        if Opx_Quality is False:
            Combo_liq_opxs_2 = Combo_liq_opxs

        # Filtering out matches which don't fit default, or user-specified Kd_Match
        # and Kd_Err values.
        if Kd_Match is None and Kd_Err is None:
            Combo_liq_opx_fur_filt = Combo_liq_opxs_2.loc[Combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'] < 0.06].reset_index(drop=True)
            Kd = Combo_liq_opx_fur_filt['Delta_Kd_Fe_Mg_Fe2']


        if Kd_Match is not None and Kd_Err is None:
            Combo_liq_opx_fur_filt = Combo_liq_opxs_2.loc[abs(Kd_Match -
            Combo_liq_opxs['Kd_Fe_Mg_Fe2']) < 0.06].reset_index(drop=True)
            Kd = Kd_Match - Combo_liq_opx_fur_filt['Kd_Fe_Mg_Fe2']
        if Kd_Match is not None and Kd_Err is not None:
            Combo_liq_opx_fur_filt = Combo_liq_opxs_2.loc[abs(Kd_Match -
            Combo_liq_opxs['Kd_Fe_Mg_Fe2']) < Kd_Err].reset_index(drop=True)
            Kd = Kd_Match - Combo_liq_opx_fur_filt['Kd_Fe_Mg_Fe2']
        if Kd_Match is None and Kd_Err is not None:
            Combo_liq_opx_fur_filt = Combo_liq_opxs_2.loc[Combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'] < Kd_Err].reset_index(drop=True)
            Kd = Combo_liq_opx_fur_filt['Delta_Kd_Fe_Mg_Fe2']

        if len(Combo_liq_opx_fur_filt) == 0:
            raise Exception('No matches found to the choosen Kd criteria.')



        # Replace automatically calculated one with various user-options.
        Combo_liq_opx_fur_filt.drop(['Delta_Kd_Fe_Mg_Fe2'], axis=1, inplace=True)
        Combo_liq_opx_fur_filt.insert(0, "Delta_Kd_Fe_Mg_Fe2", Kd)


    if return_all_pairs is True:
        Combo_liq_opx_fur_filt=Combo_liq_opxs

        # Now we have reduced down the number of calculations, we solve for P and T iteratively

        # If users want to melt match specifying an equation for both T and P
    if equationP is not None and equationT is not None:

        PT_out = calculate_opx_liq_press_temp(meltmatch=Combo_liq_opx_fur_filt, equationP=equationP, equationT=equationT, iterations=iterations)
        #print(PT_out)
        P_guess = PT_out['P_kbar_calc'].astype('float64')
        T_K_guess = PT_out['T_K_calc'].astype('float64')
        Delta_T_K_Iter=PT_out['Delta_T_K_Iter'].astype(float)
        Delta_P_kbar_Iter=PT_out['Delta_P_kbar_Iter'].astype(float)
        Combo_liq_opx_fur_filt.insert(0, "P_kbar_calc", P_guess.astype(float))
        Combo_liq_opx_fur_filt.insert(1, "T_K_calc", T_K_guess.astype(float))


    # Users may already know their pressure, rather than choosing an equation.
    if equationT is not None and equationP is None:
        P_guess = P
        T_K_guess = calculate_opx_liq_temp(meltmatch=Combo_liq_opx_fur_filt, equationT=equationT, P=P_guess)
        Combo_liq_opx_fur_filt.insert(0, "P_kbar_input", P_guess)
        Combo_liq_opx_fur_filt.insert(1, "T_K_calc", T_K_guess.astype(float))
        Delta_T_K_Iter=0
        Delta_P_kbar_Iter=0
# Users may already know their temperature, rather than using an equation
    if equationP is not None and equationT is None:
        T_K_guess = T
        P_guess = calculate_opx_liq_press(meltmatch=Combo_liq_opx_fur_filt, equationP=equationP, T=T_K_guess)
        Combo_liq_opx_fur_filt.insert(0, "P_kbar_calc", P_guess.astype(float))
        Combo_liq_opx_fur_filt.insert(1, "T_K_input", T_K_guess)
        Delta_T_K_Iter=0
        Delta_P_kbar_Iter=0


    print('Finished calculating Ps and Ts, now just averaging the results. Almost there!')
    Combo_liq_opx_fur_filt.insert(2, "Delta_T_K_Iter", Delta_P_kbar_Iter)
    Combo_liq_opx_fur_filt.insert(3, "Delta_P_kbar_Iter",  Delta_T_K_Iter)

    Liquid_sample_ID=Combo_liq_opx_fur_filt["Sample_ID_Liq"]
    Combo_liq_opx_fur_filt.drop(["Sample_ID_Liq", "Kd Eq (Put2008+-0.06)"], axis=1, inplace=True)


    # # This bit averages all the matches for a given Opx (e.g, Opx1-Liq1,
    opxNumbers = Combo_liq_opx_fur_filt['ID_OPX'].unique()
    if len(opxNumbers) > 0:
        df1_Mean_nopref=Combo_liq_opx_fur_filt.groupby(['ID_OPX', 'Sample_ID_Opx'], as_index=False).mean()
        df1_Std_nopref=Combo_liq_opx_fur_filt.groupby(['ID_OPX', 'Sample_ID_Opx'], as_index=False).std()
        count=Combo_liq_opx_fur_filt.groupby('ID_OPX').count()
        Sample_ID_Opx_Mean=df1_Mean_nopref['Sample_ID_Opx']
        Sample_ID_Opx_Std=df1_Std_nopref['Sample_ID_Opx']
        df1_Mean=df1_Mean_nopref.add_prefix('Mean_')
        df1_Std=df1_Std_nopref.add_prefix('Std_')
        df1_Mean=df1_Mean.drop(['Mean_Sample_ID_Opx'], axis=1)
        df1_Std=df1_Std.drop(['Std_Sample_ID_Opx'], axis=1)
        df1_Mean.rename(columns={"Mean_ID_OPX": "ID_OPX"}, inplace=True)
        df1_Std.rename(columns={"Std_ID_OPX": "ID_OPX"}, inplace=True)

        df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_OPX'])
        df1_M['Sample_ID_Opx']=Sample_ID_Opx_Mean

        if equationT is not None and equationP is not None:
            cols_to_move = ['Sample_ID_Opx',
                        'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']

        if equationT is not None and equationP is None:
            cols_to_move = ['Sample_ID_Opx',
                        'Mean_P_kbar_input',
                        'Std_P_kbar_input', 'Mean_T_K_calc', 'Std_T_K_calc']

        if equationT is None and equationP is not None:
            cols_to_move = ['Sample_ID_Opx',
                        'Mean_T_K_input', 'Std_T_K_input', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']



        df1_M = df1_M[cols_to_move +
                        [col for col in df1_M.columns if col not in cols_to_move]]


    else:
        raise Exception(
            'No Matches - you may need to set less strict filters, e.g.,'
            'you could edit Kd_Match is None and Kd_Err to get more matches')


    # Returns all opxs-liquids that went through 1st Kd filter with
    # equilibrium parameters, averaged matches, and all matches (not averaged)


    print('Done!!! I found a total of N='+str(len(Combo_liq_opx_fur_filt)) +
    ' Opx-Liq matches using the specified filter. N=' + str(len(df1_M)) +
     ' Opx out of the N='+str(LenOpx)
     +' Opx that you input matched to 1 or more liquids')

    Combo_liq_opx_fur_filt['Sample_ID_Liq']=Liquid_sample_ID

    cols_to_move = ['Sample_ID_Opx', 'Sample_ID_Liq']

    Combo_liq_opx_fur_filt = Combo_liq_opx_fur_filt[cols_to_move +
                        [col for col in Combo_liq_opx_fur_filt.columns if col not in cols_to_move]]



    return {'Av_PTs': df1_M, 'All_PTs': Combo_liq_opx_fur_filt}
        # return Combo_liq_opx_fur_filt