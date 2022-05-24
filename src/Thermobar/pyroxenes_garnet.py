import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *

## Pressure equations for pyroxene garnets

def P_Brey1990(T ,*, Al_Opx_cat_6ox, Cr_Opx_cat_6ox, Ti_Opx_cat_6ox,
    Na_Opx_cat_6ox, Ca_Opx_cat_6ox, Mn_Opx_cat_6ox,
    Mg_Opx_cat_6ox, Fet_Opx_cat_6ox, Ca_Gt_Cat, Mg_Gt_Cat, Fe_Gt_Cat, Mn_Gt_Cat,
    Al_AlCr_Gt, Cr_AlCr_Gt):

    X_m1_Al = (Al_Opx_cat_6ox + Na_Opx_cat_6ox - Cr_Opx_cat_6ox - (2*Ti_Opx_cat_6ox)) / 2.0 #OK

    X_Jadeite = Na_Opx_cat_6ox - Cr_Opx_cat_6ox - (2*Ti_Opx_cat_6ox) #OK
    X_m1_alTS = np.zeros(len(X_Jadeite))
    for i in range(0,len(X_Jadeite)):
        if X_Jadeite[i] <0:
            X_m1_alTS[i] = ((Al_Opx_cat_6ox[i] + X_Jadeite[i]) / 2.0)
        else:
            X_m1_alTS[i] = ((Al_Opx_cat_6ox[i] - X_Jadeite[i]) / 2.0)
    # X_m1_alTS = (Al_Opx_cat_6ox - abs(X_Jadeite)) / 2.0
    X_m1_mf = 1 - X_m1_Al - Cr_Opx_cat_6ox - Ti_Opx_cat_6ox #OK
    X_m2_mf = 1 - Ca_Opx_cat_6ox - Na_Opx_cat_6ox - Mn_Opx_cat_6ox #OK
    X_mf = (Mg_Opx_cat_6ox / (Mg_Opx_cat_6ox + Fet_Opx_cat_6ox))
    X_m1_mg = X_m1_mf * X_mf
    X_m1_fe =  X_m1_mf * (1-X_mf)
    X_m2_mg = X_m2_mf * X_mf

    Ca_CaMgFeMn_Gt = (Ca_Gt_Cat / (Ca_Gt_Cat + Mg_Gt_Cat + Fe_Gt_Cat + Mn_Gt_Cat))
    Fe_CaMgFeMn_Gt = (Fe_Gt_Cat / (Ca_Gt_Cat + Mg_Gt_Cat + Fe_Gt_Cat + Mn_Gt_Cat))

    R = 8.3143

    Kd = (((1 - Ca_CaMgFeMn_Gt)**3) * (Al_AlCr_Gt**2)) /\
        (X_m1_mf * (X_m2_mf**2) * X_m1_alTS)

    C1 = -R * T * np.log(Kd) - 5510.0 + (88.91*T) - (19 * T**1.2) + (3.0*(Ca_CaMgFeMn_Gt**2.0) *\
        82458.0) + (X_m1_mg * X_m1_fe *(80942 - (46.7*T))) -\
        (3*Fe_CaMgFeMn_Gt * Ca_CaMgFeMn_Gt * 17793.0) -\
        (Ca_CaMgFeMn_Gt * Cr_AlCr_Gt * (1.164e6 - (420.4*T))) -\
        (Fe_CaMgFeMn_Gt * Cr_AlCr_Gt * (-1.25e6 + (565.0*T)))

    C2 = -0.832 - (8.78e-5 * (T - 298)) + (3.305*3*Ca_CaMgFeMn_Gt**2) -\
        (13.45 * Ca_CaMgFeMn_Gt * Cr_AlCr_Gt) +\
        (10.5 * Fe_CaMgFeMn_Gt * Cr_AlCr_Gt)

    C3 = 16.6e-4

    P_kbar = (-C2 - np.sqrt(C2**2 + ((4.0*C3*C1) / 1e3))) / (2.0 * C3)

    return P_kbar

def P_Nickel1985(T, *, Al_Opx_cat_6ox, Cr_Opx_cat_6ox, Ti_Opx_cat_6ox,
    Na_Opx_cat_6ox, Ca_Opx_cat_6ox, Mn_Opx_cat_6ox,
    Mg_Opx_cat_6ox, Fet_Opx_cat_6ox, Ca_Gt_Cat, Mg_Gt_Cat, Fe_Gt_Cat,
    Mn_Gt_Cat,Al_AlCr_Gt, Cr_AlCr_Gt):

    X_m1_Al = (Al_Opx_cat_6ox - Cr_Opx_cat_6ox - (2*Ti_Opx_cat_6ox) +\
    Na_Opx_cat_6ox) / 2.0
    X_m1_mg_fe = 1.0 - X_m1_Al - Cr_Opx_cat_6ox - Ti_Opx_cat_6ox

    X_m2_mg_fe = 1.0 - Ca_Opx_cat_6ox - Na_Opx_cat_6ox - Mn_Opx_cat_6ox

    X_m1_mg = (Mg_Opx_cat_6ox / (Mg_Opx_cat_6ox + Fet_Opx_cat_6ox)) * X_m1_mg_fe

    X_m1_cr = 1 - X_m1_mg_fe - X_m1_Al - Ti_Opx_cat_6ox

    Ca_CaMgFeMn = (Ca_Gt_Cat / (Ca_Gt_Cat + Mg_Gt_Cat + Fe_Gt_Cat + Mn_Gt_Cat))
    Fe_CaMgFeMn = (Fe_Gt_Cat / (Ca_Gt_Cat + Mg_Gt_Cat + Fe_Gt_Cat + Mn_Gt_Cat))

    Kd = (((1 - Ca_CaMgFeMn)**3) * Al_AlCr_Gt**2) *\
        ((X_m1_mg_fe * (X_m2_mg_fe**2) * X_m1_Al)**-1)


    P_kbar = (1.0 / -(183.3 + 178.98 * X_m1_Al * (1 - X_m1_Al))) *\
        (-R * T * np.log(Kd) - (9000 * Ca_CaMgFeMn**2) -\
        3400 * (2*(Cr_AlCr_Gt**2) - (X_m1_mg * X_m1_cr)) -\
        (Ca_CaMgFeMn * Cr_AlCr_Gt * (90853.0 - 52.1*T)) -\
        (7590.0 * Fe_CaMgFeMn * Ca_CaMgFeMn) +\
        6047.0 - (3.23*T))

    return P_kbar
"""
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
"""
def T_Brey1990(P, *, Fet_Cpx_cat_6ox, Mg_Cpx_cat_6ox, Ca_Cpx_cat_6ox, Na_Cpx_cat_6ox,
    Ca_Opx_cat_6ox, Na_Opx_cat_6ox, Fet_Opx_cat_6ox, Mg_Opx_cat_6ox):
    '''
    Two-pyroxene thermometer of Brey and Kohler (1990).
    :cite:`brey1990geothermobarometry`

    SEE=+-50C for Cpx Mg#>0.75
    SEE=+-70C for all data
    '''

    Kd = (1 - (Ca_Cpx_cat_6ox / (1 - Na_Cpx_cat_6ox))) / (1 - (Ca_Opx_cat_6ox / (1 - Na_Opx_cat_6ox)))

    x_cpx_fe = (Fet_Cpx_cat_6ox) / (Fet_Cpx_cat_6ox + Mg_Cpx_cat_6ox)
    x_opx_fe = (Fet_Opx_cat_6ox) / (Fet_Opx_cat_6ox + Mg_Opx_cat_6ox)

    T_K = (23664.0 + ((24.9 + (126.3 * x_cpx_fe)) * P)) /\
        (13.38 + (np.log(Kd)**2.0) + (11.59 * x_opx_fe))

    return T_K

# def T_Brey1990_Ca_in_Opx(P, *, Ca_Cpx_cat_6ox):
#
#     T_K = (6425.0 + (26.4 * P)) / (-np.log(Ca_Cpx_cat_6ox) + 1.843)
#
#     return T_K
#--------------------Function for solving for temperature for two pyroxenes-----------------------------------------------------#
Px_Gt_P_funcs = {P_Brey1990, P_Nickel1985} # put on outside

Px_Gt_P_funcs_by_name = {p.__name__: p for p in Px_Gt_P_funcs}

def calculate_pyroxenes_garnet_press(*, opx_comps = None, cpx_comps = None, gt_comps = None, equationP = None, T = None):

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

    gt_comps: pandas.DataFrame
        Gt compositions with column headings SiO2_Gt, MgO_Gt etc.

    equationP: str
        Choose from:

        |  P_Brey1990 (T-dependent)


    Returns
    -------
    If eq_tests is False
        pandas.DataFrame: Temperature in Kelvin, pressure in kbar
    If eq_tests is True
        pandas.DataFrame: Temperature in Kelvin, pressure in kbar, eq Tests + opx+cpx comps + components

    '''

    try:
        func = Px_Gt_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationT} requires you to enter T, or specify T="Solve"')

    if isinstance(T, pd.Series):
        if cpx_comps is not None:
            if len(T) != len(cpx_comps):
                raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of cpx compositions')
        if opx_comps is not None:
            if len(T) != len(opx_comps):
                raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of opx compositions')
        if gt_comps is not None:
            if len(T) != len(gt_comps):
                raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of gt compositions')

    data_array = []
    if cpx_comps is not None:
        cpx_comps = calculate_clinopyroxene_components(cpx_comps)
        data_array.append(cpx_comps)
    if opx_comps is not None:
        opx_comps = calculate_orthopyroxene_components(opx_comps)
        data_array.append(opx_comps)
    if gt_comps is not None:
        gt_comps = calculate_garnet_components(gt_comps)
        data_array.append(gt_comps)

    if len(data_array) > 1:
        data_array = pd.concat(data_array, axis = 1)
    else:
        data_array = data_array[0]

    kwargs = {name: data_array[name] for name, p in sig.parameters.items()
    if p.kind == inspect.Parameter.KEYWORD_ONLY}

    if isinstance(T, str) or T is None:
        if T == "Solve":
            P_kbar = partial(func, **kwargs)
        if T is None:
            P_kbar = func(**kwargs)
    else:
        P_kbar = func(T, **kwargs)

    return P_kbar

Px_Gt_T_funcs = {T_Brey1990} # put on outside

Px_Gt_T_funcs_by_name = {p.__name__: p for p in Px_Gt_T_funcs}

def calculate_pyroxenes_garnet_temp(*, opx_comps = None, cpx_comps = None, gt_comps = None, equationT = None, P = None):

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

    gt_comps: pandas.DataFrame
        Gt compositions with column headings SiO2_Gt, MgO_Gt etc.

    equationP: str
        Choose from:

        |  P_Brey1990 (T-dependent)


    Returns
    -------

    pandas.DataFrame: Temperature in Kelvin, pressure in kbar

    '''

    try:
        func = Px_Gt_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')

    if isinstance(P, pd.Series):
        if cpx_comps is not None:
            if len(P) != len(cpx_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of cpx compositions')
        if opx_comps is not None:
            if len(P) != len(opx_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of opx compositions')
        if gt_comps is not None:
            if len(P) != len(gt_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of opx compositions')

    data_array = []
    if cpx_comps is not None:
        cpx_comps = calculate_clinopyroxene_components(cpx_comps)
        data_array.append(cpx_comps)
    if opx_comps is not None:
        opx_comps = calculate_orthopyroxene_components(opx_comps)
        data_array.append(opx_comps)
    if gt_comps is not None:
        gt_comps = calculate_garnet_components(gt_comps)
        data_array.append(gt_comps)

    if len(data_array) >1:
        data_array = pd.concat(data_array, axis = 1)
    else:
        data_array = data_array[0]

    kwargs = {name: data_array[name] for name, p in sig.parameters.items()
    if p.kind == inspect.Parameter.KEYWORD_ONLY}

    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)

        if P is None:
            T_K=func(**kwargs)
    else:
        T_K=func(P, **kwargs)

    return T_K

def calculate_pyroxenes_garnet_press_temp(*, cpx_comps=None, opx_comps=None, gt_comps = None,
                              equationP=None, equationT=None, iterations=30, T_K_guess=1300):
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

    gt_comps: pandas.DataFrame
        Gt compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    meltmatch: pandas.DataFrame
        Combined dataframe of Opx-Cpx compositions (headings SiO2_Cpx, SiO2_Opx etc.).
        Used for calculate cpx_opx_press_temp_matching function.

    equationP: str
        Choose from:

        |  P_Brey1990 (T-dependent)
        |  P_Nickel1985 (T-dependent)

    equationT: str
        Choose from:

        |  T_Brey1990 (P-dependent)


    Optional:

    iterations: int, Default = 20
         Number of iterations used to converge to solution

    T_K_guess: int or float. Default = 1300K
         Initial guess of temperature to start iterations at


    Returns
    -------
        pandas.DataFrame: Temperature in Kelvin, pressure in kba

    '''
    # Gives users flexibility to reduce or increase iterations
    T_func = calculate_pyroxenes_garnet_temp(
        cpx_comps=cpx_comps, opx_comps=opx_comps, gt_comps = gt_comps, equationT=equationT, P="Solve")
    P_func = calculate_pyroxenes_garnet_press(
        cpx_comps=cpx_comps, opx_comps=opx_comps, gt_comps = gt_comps, equationP=equationP, T="Solve")

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

        # T_K_guess_is_bad = (T_K_guess == 0) | (T_K_guess == 273.15) | (T_K_guess ==  -np.inf) | (T_K_guess ==  np.inf)
        # T_K_guess[T_K_guess_is_bad] = np.nan
        # P_guess[T_K_guess_is_bad] = np.nan


    PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess})

    return PT_out
