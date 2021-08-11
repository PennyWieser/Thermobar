import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd



from Thermobar.core import *

## Pressure equations for two pyroxenes


def P_Put2008_eq38(T=None, *, Na2O_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
TiO2_Opx_cat_6ox, CaO_Opx_cat_6ox, Cr2O3_Opx_cat_6ox, MgO_Opx_cat_6ox,
FeOt_Opx_cat_6ox, MnO_Opx_cat_6ox, CaO_Cpx_cat_6ox, Fm2Si2O6, En_Opx, Di_Opx):
    '''

    Two pyroxene barometer of Putirka (2008) Eq38. Calibrated on Mg#-rich systems (>0.75)

    | SEE=+-3.7 kbar
    '''
    Lindley_Fe3_Opx = (Na2O_Opx_cat_6ox + Al_IV_Opx_cat_6ox - Al_VI_Opx_cat_6ox - \
        2 * TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox)  # This is cell FR
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    a_En_opx_mod = (((0.5 * MgO_Opx_cat_6ox / (0.5 * (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx)
    + 0.5 * MgO_Opx_cat_6ox + Na2O_Opx_cat_6ox +CaO_Opx_cat_6ox + MnO_Opx_cat_6ox)))
    * (0.5 * MgO_Opx_cat_6ox / (0.5 * MgO_Opx_cat_6ox + 0.5 * (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx)
    + TiO2_Opx_cat_6ox + Al_VI_Opx_cat_6ox + Cr2O3_Opx_cat_6ox + Lindley_Fe3_Opx)))
    Kf = CaO_Opx_cat_6ox / (1 - CaO_Cpx_cat_6ox)
    return (-279.8 + 293 * Al_VI_Opx_cat_6ox + 455 * Na2O_Opx_cat_6ox + 229 * Cr2O3_Opx_cat_6ox +
            519 * Fm2Si2O6 - 563 * En_Opx + 371 * Di_Opx + 327 * a_En_opx_mod + 1.19 / Kf)


def P_Put2008_eq39(T, *, Na2O_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
TiO2_Opx_cat_6ox, Cr2O3_Opx_cat_6ox, FeOt_Opx_cat_6ox, MnO_Opx_cat_6ox, CaO_Opx_cat_6ox,
MgO_Opx_cat_6ox, Na2O_Cpx_cat_6ox, Al_IV_cat_6ox, Al_VI_cat_6ox, TiO2_Cpx_cat_6ox,
CaO_Cpx_cat_6ox, MgO_Cpx_cat_6ox, MnO_Cpx_cat_6ox, FeOt_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox,
Fm2Si2O6, En_Opx, EnFs):
    '''

    Two pyroxene barometer of Putirka (2008) Eq39. As for Eq38, but also a function of temperature.

    | SEE=+-2.8 kbar (Cpx Mg#>0.75)
    | SEE=+-3.2 kbar (all data)
    '''
    Lindley_Fe3_Opx = Na2O_Opx_cat_6ox + Al_IV_Opx_cat_6ox - \
        Al_VI_Opx_cat_6ox - 2 * TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    a_En_opx_mod = (((0.5 * MgO_Opx_cat_6ox / (0.5 * (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx)
    + 0.5 * MgO_Opx_cat_6ox + Na2O_Opx_cat_6ox +CaO_Opx_cat_6ox + MnO_Opx_cat_6ox)))
    * (0.5 * MgO_Opx_cat_6ox / (0.5 * MgO_Opx_cat_6ox + 0.5 * (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx)
    + TiO2_Opx_cat_6ox + Al_VI_Opx_cat_6ox + Cr2O3_Opx_cat_6ox + Lindley_Fe3_Opx)))

    Lindley_Fe3_Cpx = Na2O_Cpx_cat_6ox + Al_IV_cat_6ox - \
        Al_VI_cat_6ox - 2 * TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    a_Di_cpx = CaO_Cpx_cat_6ox / (CaO_Cpx_cat_6ox + 0.5 * MgO_Cpx_cat_6ox + 0.5 * (
        FeOt_Cpx_cat_6ox - Lindley_Fe3_Cpx) + MnO_Cpx_cat_6ox + Na2O_Cpx_cat_6ox)
    Kf = CaO_Opx_cat_6ox / (1 - CaO_Cpx_cat_6ox)
    return (-94.25 + 0.045 * (T - 273.15) + 187.7 * Al_VI_Opx_cat_6ox + 246.8 * Fm2Si2O6 -
            212.5 * En_Opx + 127.5 * a_En_opx_mod - 69.4 * EnFs - 133.9 * a_Di_cpx - 1.66 / Kf)

## Temperature equations for two pyroxenes


def T_Put2008_eq36(P, *, EnFs, Fm2Si2O6, CaO_Cpx_cat_6ox,
                   CrCaTs, MnO_Opx_cat_6ox, Na2O_Opx_cat_6ox, En_Opx, Di_Opx):
    '''
    Two-pyroxene thermometer of Putirka (2008) eq 36. Best for Cpx with Mg#>0.75
    SEE=+-45C for Cpx Mg#>0.75
    SEE=+-56C for all data
    '''
    return (273.15 + 10 ** 4 / (11.2 - 1.96 * np.log(EnFs.astype(float) / Fm2Si2O6.astype(float)) - 3.3 * CaO_Cpx_cat_6ox - 25.8 *
            CrCaTs + 33.2 * MnO_Opx_cat_6ox - 23.6 * Na2O_Opx_cat_6ox - 2.08 * En_Opx - 8.33 * Di_Opx - 0.05 * P))


def T_Brey1990(P, *, FeOt_Cpx_cat_6ox, CaO_Cpx_cat_6ox, MgO_Cpx_cat_6ox, Na2O_Cpx_cat_6ox,
               FeOt_Opx_cat_6ox, MgO_Opx_cat_6ox, CaO_Opx_cat_6ox, Na2O_Opx_cat_6ox):
    '''
    Two-pyroxene thermometer of Brey and Kohler (1990).
    SEE=+-50C for Cpx Mg#>0.75
    SEE=+-70C for all data
    '''
    return ((23664 + (24.9 + 126.3 * FeOt_Cpx_cat_6ox / (FeOt_Cpx_cat_6ox + MgO_Cpx_cat_6ox)) * P)
    / (13.38 + (np.log((1 - CaO_Cpx_cat_6ox.astype(float) /(1 - Na2O_Cpx_cat_6ox.astype(float))) /
    (1 - CaO_Opx_cat_6ox.astype(float) / (1 - Na2O_Opx_cat_6ox.astype(float)))))**2
    + 11.59 * FeOt_Opx_cat_6ox / (FeOt_Opx_cat_6ox + MgO_Opx_cat_6ox)))


def T_Put2008_eq37(P, *, EnFs, Di_Cpx, Fm2Si2O6, MnO_Opx_cat_6ox,
                   FmAl2SiO6, MgO_Cpx_cat_6ox, FeOt_Cpx_cat_6ox):
    '''
    Two-pyroxene thermometer of Putirka (2008) eq 37. For Cpx with Mg#<0.75
    SEE=+-38C for Cpx Mg#>0.75
    SEE=+-60C for all data
    '''
    return (273.15 + 10**4 / (13.4 - 3.4 * np.log(EnFs.astype(float) / Fm2Si2O6.astype(float)) + 5.59 * np.log(MgO_Cpx_cat_6ox.astype(float))
    + 23.85 * MnO_Opx_cat_6ox +6.48 * FmAl2SiO6 - 2.38 * Di_Cpx - 0.044 * P
    - 8.8 * MgO_Cpx_cat_6ox / (MgO_Cpx_cat_6ox + FeOt_Cpx_cat_6ox)))


def T_Wood1973(P=None, *, MgO_Opx_cat_6ox, CaO_Opx_cat_6ox, MnO_Opx_cat_6ox,
FeOt_Opx_cat_6ox, Na2O_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
TiO2_Opx_cat_6ox, Cr2O3_Opx_cat_6ox, MgO_Cpx_cat_6ox, CaO_Cpx_cat_6ox,
MnO_Cpx_cat_6ox, FeOt_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, Al_IV_cat_6ox,
Al_VI_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox):
    '''
    Two-pyroxene thermometer of Wood and Banno (1973)
    '''
    # Opx parts
    Lindley_Fe3_Opx = Na2O_Opx_cat_6ox + Al_IV_Opx_cat_6ox - Al_VI_Opx_cat_6ox - \
        2 * TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox  # This is cell FR
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    MgNo_WB_Opx = MgO_Opx_cat_6ox / \
        (MgO_Opx_cat_6ox + (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx))
    X_Mg_M2_Opx = (1 - CaO_Opx_cat_6ox - Na2O_Opx_cat_6ox -
                   MnO_Opx_cat_6ox) * MgNo_WB_Opx  # FL
    X_Fe_M2_Opx = (1 - CaO_Opx_cat_6ox - Na2O_Opx_cat_6ox -
                   MnO_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FM
    X_Mg_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox) * MgNo_WB_Opx  # FJ
    X_Fe_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FK
    MgNo_WB_Opx = MgO_Opx_cat_6ox / \
        (MgO_Opx_cat_6ox + (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx))
    a_opx_En = (X_Mg_M2_Opx / (X_Mg_M2_Opx + X_Fe_M2_Opx + CaO_Opx_cat_6ox +
    Na2O_Opx_cat_6ox + MnO_Opx_cat_6ox)) * \
        (X_Mg_M1_Opx / (Lindley_Fe3_Opx + TiO2_Opx_cat_6ox +
         Al_VI_Opx_cat_6ox + Cr2O3_Opx_cat_6ox + X_Mg_M1_Opx + X_Fe_M1_Opx))
    # Cpx parts
    Lindley_Fe3_Cpx = Na2O_Cpx_cat_6ox + Al_IV_cat_6ox - Al_VI_cat_6ox - \
        2 * TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox  # This is cell FR
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    Fe2_WB_Cpx = FeOt_Cpx_cat_6ox - Lindley_Fe3_Cpx
    MgNo_WB_Cpx = MgO_Cpx_cat_6ox / (MgO_Cpx_cat_6ox + Fe2_WB_Cpx)
    X_Mg_M2_Cpx = (1 - CaO_Cpx_cat_6ox - Na2O_Cpx_cat_6ox -
                   MnO_Cpx_cat_6ox) * MgNo_WB_Cpx  # FL
    X_Fe_M2_Cpx = (1 - CaO_Cpx_cat_6ox - Na2O_Cpx_cat_6ox -
                   MnO_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FM
    X_Mg_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox) * MgNo_WB_Cpx  # FJ
    X_Fe_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FK
    a_cpx_En = (X_Mg_M2_Cpx / (X_Mg_M2_Cpx + X_Fe_M2_Cpx + CaO_Cpx_cat_6ox
    + Na2O_Cpx_cat_6ox + MnO_Cpx_cat_6ox)) * \
        (X_Mg_M1_Cpx / (Lindley_Fe3_Cpx + TiO2_Cpx_cat_6ox +
         Al_VI_cat_6ox + Cr2O3_Cpx_cat_6ox + X_Mg_M1_Cpx + X_Fe_M1_Cpx))

    return ((-10202 / (np.log(a_cpx_En.astype(float) / a_opx_En.astype(float)) - 7.65 *
            (1 - MgNo_WB_Opx) + 3.88 * (1 - MgNo_WB_Opx)**2 - 4.6)))


def T_Wells1977(P=None, *, MgO_Opx_cat_6ox, CaO_Opx_cat_6ox, MnO_Opx_cat_6ox,
FeOt_Opx_cat_6ox, Na2O_Opx_cat_6ox, Al_IV_Opx_cat_6ox, Al_VI_Opx_cat_6ox,
 TiO2_Opx_cat_6ox, Cr2O3_Opx_cat_6ox, MgO_Cpx_cat_6ox, CaO_Cpx_cat_6ox,
 MnO_Cpx_cat_6ox, FeOt_Cpx_cat_6ox, Na2O_Cpx_cat_6ox, Al_IV_cat_6ox,
 Al_VI_cat_6ox, TiO2_Cpx_cat_6ox, Cr2O3_Cpx_cat_6ox):
    '''
    Two-pyroxene thermometer of Wells 1977
    '''
    # Opx parts
    Lindley_Fe3_Opx = Na2O_Opx_cat_6ox + Al_IV_Opx_cat_6ox - Al_VI_Opx_cat_6ox - \
        2 * TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox  # This is cell FR
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    MgNo_WB_Opx = MgO_Opx_cat_6ox / \
        (MgO_Opx_cat_6ox + (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx))
    X_Mg_M2_Opx = (1 - CaO_Opx_cat_6ox - Na2O_Opx_cat_6ox -
                   MnO_Opx_cat_6ox) * MgNo_WB_Opx  # FL
    X_Fe_M2_Opx = (1 - CaO_Opx_cat_6ox - Na2O_Opx_cat_6ox -
                   MnO_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FM
    X_Mg_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox) * MgNo_WB_Opx  # FJ
    X_Fe_M1_Opx = (1 - Lindley_Fe3_Opx - Al_VI_Opx_cat_6ox -
                   TiO2_Opx_cat_6ox - Cr2O3_Opx_cat_6ox) * (1 - MgNo_WB_Opx)  # FK
    MgNo_WB_Opx = MgO_Opx_cat_6ox / \
        (MgO_Opx_cat_6ox + (FeOt_Opx_cat_6ox - Lindley_Fe3_Opx))
    a_opx_En = (X_Mg_M2_Opx / (X_Mg_M2_Opx + X_Fe_M2_Opx + CaO_Opx_cat_6ox
    + Na2O_Opx_cat_6ox + MnO_Opx_cat_6ox)) * \
        (X_Mg_M1_Opx / (Lindley_Fe3_Opx + TiO2_Opx_cat_6ox +
         Al_VI_Opx_cat_6ox + Cr2O3_Opx_cat_6ox + X_Mg_M1_Opx + X_Fe_M1_Opx))
    # Cpx parts
    Lindley_Fe3_Cpx = Na2O_Cpx_cat_6ox + Al_IV_cat_6ox - Al_VI_cat_6ox - \
        2 * TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox  # This is cell FR
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    Fe2_WB_Cpx = FeOt_Cpx_cat_6ox - Lindley_Fe3_Cpx
    MgNo_WB_Cpx = MgO_Cpx_cat_6ox / (MgO_Cpx_cat_6ox + Fe2_WB_Cpx)
    X_Mg_M2_Cpx = (1 - CaO_Cpx_cat_6ox - Na2O_Cpx_cat_6ox -
                   MnO_Cpx_cat_6ox) * MgNo_WB_Cpx  # FL
    X_Fe_M2_Cpx = (1 - CaO_Cpx_cat_6ox - Na2O_Cpx_cat_6ox -
                   MnO_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FM
    X_Mg_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox) * MgNo_WB_Cpx  # FJ
    X_Fe_M1_Cpx = (1 - Lindley_Fe3_Cpx - Al_VI_cat_6ox -
                   TiO2_Cpx_cat_6ox - Cr2O3_Cpx_cat_6ox) * (1 - MgNo_WB_Cpx)  # FK
    a_cpx_En = (X_Mg_M2_Cpx / (X_Mg_M2_Cpx + X_Fe_M2_Cpx + CaO_Cpx_cat_6ox + Na2O_Cpx_cat_6ox + MnO_Cpx_cat_6ox)) * \
        (X_Mg_M1_Cpx / (Lindley_Fe3_Cpx + TiO2_Cpx_cat_6ox +
         Al_VI_cat_6ox + Cr2O3_Cpx_cat_6ox + X_Mg_M1_Cpx + X_Fe_M1_Cpx))

    return ((7341 / (3.355 + 2.44 * (1 - MgNo_WB_Opx) - np.log(a_cpx_En.astype(float) / a_opx_En.astype(float)))))

## Function for calculating Cpx-Opx pressure

Cpx_Opx_P_funcs = {P_Put2008_eq38, P_Put2008_eq39} # put on outside

Cpx_Opx_P_funcs_by_name = {p.__name__: p for p in Cpx_Opx_P_funcs}

def calculate_cpx_opx_press(*, cpx_comps=None, opx_comps=None,
Two_Px_Match=None, equationP=None, eq_tests=False, T=None):
    '''
    calculates pressure in kbar for Opx-Cpx pairs

    Parameters
    -------

    cpx_comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    opx_comps: DataFrame
       Opx compositions with column headings SiO2_Opx, MgO_Opx etc.

    Or:

    Two_Px_Match: DataFrame
        Combined Cpx-Opx compositions.
        Used for calculate Cpx_Opx_press_temp_matching function.

    EquationP: str
        Choice of equation:
        |  P_Put2008_eq38
        |  P_Put2008_eq39

    T: float, int, series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool
        If False, just returns temperature in K (default) as a panda series
        If True, returns pressure in kbar, Kd Fe-Mg for opx-cpx,
        and the user-entered cpx and opx comps as a panda dataframe.


    Returns
    -------
    If eq_tests=False
        pandas.series: Pressure in kbar (if eq_tests=False)
    If eq_tests=True
        panda.dataframe: Pressure in kbar + Kd-Fe-Mg + cpx+opx comps

    '''
    try:
        func = Cpx_Opx_P_funcs_by_name[equationP]
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
        if cpx_comps is not None:
            if len(T) != len(cpx_comps):
                raise ValueError('The panda series entered for temperature isnt the'
                ' same length as the dataframe of Cpx compositions')



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
    calculates Temperature in K for Opx-Cpx pairs

   Parameters
    -------

    cpx_comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    opx_comps: DataFrame
       Opx compositions with column headings SiO2_Opx, MgO_Opx etc.

    Or:

    Two_Px_Match: DataFrame
        Combined Cpx-Opx compositions. Used for "melt match" functionality.

    EquationT: str
        Choice of equation:
        T_Put2008_Eq36  (P-dependent)
        T_Put2008_Eq37 (P-dependent)
        T_Brey1990 (P-dependent)
        T_Wood1973 (P-independent)
        T_Wells1977 (P-independent)

     P: int, float, series, str ("Solve")
        Pressure in kbar.
        Can enter float or int to use same P for all calculations
        If "Solve", returns partial if function is P-dependent

    eq_tests: bool
        If False, just returns pressure in kbar (default) as a panda series
        If True, returns pressure in kbar, Kd Fe-Mg for opx-cpx, and the user-entered cpx and opx comps as a panda dataframe.

    '''

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
        two_pyx.insert(2, "Equation Choice (T)", str(equationT))

    return two_pyx

## Iterative calculations of P and T


def calculate_cpx_opx_press_temp(*, cpx_comps=None, opx_comps=None, Two_Px_Match=None,
                              equationP=None, equationT=None, iterations=30, T_K_guess=1300, eq_tests=False):
    '''
    Solves simultaneous equations for temperature and pressure using clinopyroxene-orthopyroxene thermometers and barometers.


   Parameters
    -------

    opx_comps: DataFrame (opt, either specify opx_comps AND cpx_comps or meltmatch)
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    cpx_comps: DataFrame (not required for P_Put2008_eq29c)
        Cpxuid compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Or:

    meltmatch: DataFrame
        Combined dataframe of Opx-Cpx compositions (headings SiO2_Cpx, SiO2_Opx etc.). S
        Used for calculate Cpx_Opx_press_temp_matching function.


    EquationP: str
        Barometer
        P_Put2008_eq38
        P_Put2008_eq39

    EquationT: str
        Thermometer
        T_Put2008_eq36
        T_Put2008_eq37
        T_Brey1990
        T_Wood1973
        T_Wells1977

    Optional:

    iterations: int (optional). Default is 20.
         Number of iterations used to converge to solution

    T_K_guess: int or float. Default is 1300K
         Initial guess of temperature.


    eq_tests: bool
        If False, just returns pressure (default) as a panda series
        If True, returns pressure, Values of Eq tests,
        as well as user-entered opx and cpx comps and components.



    Returns
    -------
    If eq_tests=False
        pandas.DataFrame: Temperature in Kelvin, pressure in Kbar
    If eq_tests=True
        panda.dataframe: Temperature in Kelvin, pressure in Kbar
        Eq Tests + opx+cpx comps + components

    '''
    # Gives users flexibility to reduce or increase iterations



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

        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)

        T_K_guess_is_bad = (T_K_guess == 0) | (T_K_guess == 273.15) | (T_K_guess ==  -np.inf) | (T_K_guess ==  np.inf)
        T_K_guess[T_K_guess_is_bad] = np.nan
        P_guess[T_K_guess_is_bad] = np.nan


    if eq_tests is False:
        PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess})

        return PT_out
    if eq_tests is True:
        two_pyx = calculate_cpx_opx_eq_tests(
            cpx_comps=cpx_comps, opx_comps=opx_comps)

        two_pyx.insert(0, "T_K_calc", T_K_guess)
        two_pyx.insert(1, "P_kbar_calc", P_guess)
        two_pyx.insert(2, "Equation Choice (T)", str(equationT))
        two_pyx.insert(3, "Equation Choice (P)", str(equationP))
        two_pyx.replace([np.inf, -np.inf], np.nan, inplace=True)
    return two_pyx
   # return P_func

## Two pyroxene matching

def calculate_cpx_opx_press_temp_matching(*, opx_comps, cpx_comps, equationT=None, equationP=None,
                                  KdMatch=None, KdErr=None, Cpx_Quality=False, Opx_Quality=False, P=None, T=None):
    '''
    Evaluates all possible Cpx-Opx pairs,
    returns P (kbar) and T (K) for those in Kd Fe-Mg equilibrium.


   Parameters
    -------

    opx_comps: DataFrame
        Panda DataFrame of opx compositions with column headings SiO2_Opx etc.


    cpx_comps: DataFrame
        Panda DataFrame of cpx compositions with column headings SiO2_Cpx etc.

    EquationP: str
        |  P_Put2008_eq38
        |  P_Put2008_eq39

    EquationT: str
        |  T_Put2008_eq36
        |  T_Put2008_eq37
        |  T_Brey1990
        |  T_Wood1973
        |  T_Wells1977


    Or: User sets one of P or T
        pressure in kbar or temperature in Kelvin. Doesn't need to iterate, e.g.,
        if set pressure, calculates temperature for that pressure, and
        returns temperature for equilibrium pairs.


    KdMatch: str
        |  If None, returns all cpx-opx pairs.
        |  If "HighTemp", returns all cpxs-opxs within Kd cpx-opx=1.09+-0.14 suggested by Putirka (2008)
        |  If "Subsolidus" returns all cpxs-opxs within Kd cpx-opx=0.7+-0.2 suggested by Putirka (2008)
        |  If int or float, also need to specify KdErr. returns all matches within KdMatch +- KdErr


    KdErr: float or int (optional)
        returns all cpx-opx pairs within KdMatch+-KdErr

    Cpx Quality: bool (optional)
        If True, filters out clinopyroxenes with cation sums outside of 4.02-3.99 (after Neave et al. 2017)

    Opx Quality: bool (optional)
        If True, filters out orthopyroxenes with cation sums outside of 4.02-3.99


   Returns
    -------
        dict

        Av_PTs_perCpx: Average P and T for each cpx.
        E.g., if cpx1 matches Opx1, Opx4, Opx6, Opx10, averages outputs for all 4 of those opxs.
        Returns mean and 1 sigma of these averaged parameters for each Cpx.

        All_PTs: Returns output parameters for all matches (e.g, cpx1-opx1, cpx1-opx4) without any averaging.

    '''
    if (KdErr is None and isinstance(KdMatch, int)) or (KdErr is None and isinstance(KdMatch, float)):
        raise Exception(
            'You have entered a numerical value for KdMatch, but have not'
            'specified a KdErr to accept matches within KdMatch+-KdErr')

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
    print("Considering " + LenCombo +
          " Opx-Cpx pairs, be patient if this is >>1 million!")

    # calculate Kd for these pairs
    En = (Combo_opxs_cpxs.Fm2Si2O6 * (Combo_opxs_cpxs.MgO_Opx_cat_6ox /
    (Combo_opxs_cpxs.MgO_Opx_cat_6ox +Combo_opxs_cpxs.FeOt_Cpx_cat_6ox + Combo_opxs_cpxs.MnO_Cpx_cat_6ox)))
    Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'] = ((Combo_opxs_cpxs['FeOt_Cpx_cat_6ox']
    / Combo_opxs_cpxs['MgO_Cpx_cat_6ox'])) / (Combo_opxs_cpxs['FeOt_Opx_cat_6ox']
     / Combo_opxs_cpxs['MgO_Opx_cat_6ox'])

    if KdMatch == "Subsolidus":
        Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx'] = np.abs(
            0.7 - Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'])
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.loc[np.abs(
            Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx']) < 0.2]  # +- 0.2 suggested by Putirka spreadsheet
    if KdMatch == "HighTemp":
        Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx'] = np.abs(
            1.09 - Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'])
        # +- 0.14 suggested by Putirka spreadsheet
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.loc[np.abs(
            Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx']) < 0.14]
    if isinstance(KdMatch, int) or isinstance(
            KdMatch, float) and KdErr is not None:
        Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx'] = np.abs(
            KdMatch - Combo_opxs_cpxs['Kd_Fe_Mg_Cpx_Opx'])
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.loc[np.abs(
            Combo_opxs_cpxs['Delta_Kd_Fe_Mg_Cpx_Opx']) < KdErr]  # +- 0.14 suggested by Putirka

    if KdMatch is None and KdMatch is None and KdErr is None:
        Combo_opxs_cpxs_1 = Combo_opxs_cpxs.copy()
        print('No Kd selected, all matches are shown')


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
            Two_Px_Match=Combo_opxs_cpxs_2, equationP=equationP, equationT=equationT)
        Combo_opxs_cpxs_2.insert(0, "P_kbar_calc", PT_out['P_kbar_calc'])
        Combo_opxs_cpxs_2.insert(1, "T_K_calc", PT_out['T_K_calc'])
        Combo_opxs_cpxs_2.insert(2, "Equation Choice (T)", str(equationT))
        Combo_opxs_cpxs_2.insert(3, "Equation Choice (P)", str(equationP))
    if P is not None:
        T_K_calc = calculate_cpx_opx_temp(
            Two_Px_Match=Combo_opxs_cpxs_2, equationT=equationT, P=P)
        Combo_opxs_cpxs_2.insert(0, "P_kbar_input", P)
        Combo_opxs_cpxs_2.insert(1, "T_K_calc", T_K_calc)
        Combo_opxs_cpxs_2.insert(2, "Equation Choice (T)", str(equationT))
    if T is not None:
        P_kbar_calc = calculate_cpx_opx_press(
            Two_Px_Match=Combo_opxs_cpxs_2, equationP=equationP, T=T)
        Combo_opxs_cpxs_2.insert(0, "P_kbar_calc", P_kbar_calc)
        Combo_opxs_cpxs_2.insert(1, "T_K_input", T)
        Combo_opxs_cpxs_2.insert(3, "Equation Choice (P)", str(equationP))

    cols_to_move = ['Kd_Fe_Mg_Cpx_Opx']
    Combo_opxs_cpxs_2 = Combo_opxs_cpxs_2[cols_to_move + [
        col for col in Combo_opxs_cpxs_2.columns if col not in cols_to_move]]

    Combo_opxs_cpxs_2.insert(
        0, "Sample_ID_Opx", Combo_opxs_cpxs_2_names['Sample_ID_Opx'])
    Combo_opxs_cpxs_2.insert(
        1, "Sample_ID_Cpx", Combo_opxs_cpxs_2_names['Sample_ID_Cpx'])

    print('Finished calculating Ps and Ts, now just averaging the results. Almost there!')
#     #Final step, calcuate a 3rd output which is the average and standard deviation for each CPx (e.g., CPx1-Melt1, CPx1-melt3 etc. )
    CpxNumbers = Combo_opxs_cpxs_2['ID_CPX'].unique()
    if len(CpxNumbers) > 0:
        df1_M = pd.DataFrame()
        df1_S = pd.DataFrame()
        for cpx in CpxNumbers:
            dff_M = pd.DataFrame(
                Combo_opxs_cpxs_2.loc[Combo_opxs_cpxs_2['ID_CPX'] == cpx].mean(axis=0)).T
            dff_M['Sample_ID_Cpx'] = Combo_opxs_cpxs_2.loc[Combo_opxs_cpxs_2['ID_CPX']
                                                           == cpx, "Sample_ID_Cpx"].iloc[0]

            if cpx == CpxNumbers[0]:
                df1_M = dff_M
                df1_M['Sample_ID_Cpx'] = Combo_opxs_cpxs_2.loc[Combo_opxs_cpxs_2['ID_CPX']
                                                               == cpx, "Sample_ID_Cpx"].iloc[0]
            else:
                df1_M = pd.concat([df1_M, dff_M], sort=False)

        df1_M = df1_M.add_prefix('Mean_')
        if equationP is not None and equationT is not None:
            cols_to_move = ['Mean_Sample_ID_Cpx',
                            'Mean_T_K_calc', 'Mean_P_kbar_calc']
        if equationT is not None and P is not None:
            cols_to_move = ['Mean_Sample_ID_Cpx',
                            'Mean_T_K_calc', 'Mean_P_kbar_input']
        if equationP is not None and T is not None:
            cols_to_move = ['Mean_Sample_ID_Cpx',
                            'Mean_T_K_input', 'Mean_P_kbar_calc']

        df1_M.rename(columns={'Mean_Sample_ID_Cpx': 'Sample_ID_Cpx'})
        df1_M = df1_M[cols_to_move +
                      [col for col in df1_M.columns if col not in cols_to_move]]

        for cpx in CpxNumbers:
            dff_S = pd.DataFrame(
                Combo_opxs_cpxs_2.loc[Combo_opxs_cpxs_2['ID_CPX'] == cpx].std(axis=0)).T
            # This tells us if there is only 1, in which case std will return
            # Nan
            if np.shape(Combo_opxs_cpxs_2.loc[Combo_opxs_cpxs_2['ID_CPX'] == cpx])[
                    0] == 1:
                dff_S = dff_S.fillna(0)
            else:
                dff_S = dff_S
            if cpx == CpxNumbers[0]:
                df1_S = dff_S
            else:
                df1_S = pd.concat([df1_S, dff_S])

        df1_S = df1_S.add_prefix('st_dev_')
        if equationP is not None and equationT is not None:
            df1_M.insert(1, "st_dev_T_K_calc", df1_S['st_dev_T_K_calc'])
            df1_M.insert(3, "st_dev_P_kbar_calc", df1_S['st_dev_P_kbar_calc'])
            df1_M.insert(4, "Equation Choice (T)", str(equationT))
            df1_M.insert(5, "Equation Choice (P)", str(equationP))
        if equationT is not None and P is not None:
            df1_M.insert(1, "st_dev_T_K_calc", df1_S['st_dev_T_K_calc'])
            df1_M.insert(3, "Equation Choice (T)", str(equationT))
        if equationP is not None and T is not None:
            df1_M.insert(1, "st_dev_P_kbar_calc", df1_S['st_dev_P_kbar_calc'])
            df1_M.insert(3, "Equation Choice (P)", str(equationP))

    else:
        raise Exception(
            'No Matches - you may need to set less strict filters, e.g.,'
            'you could edit KdMatch is None and KdErr to get more matches')

    print('Done!')
    return {'Av_PTs_perCPX': df1_M, 'All_PTs': Combo_opxs_cpxs_2}
