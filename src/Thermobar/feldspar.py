import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from Thermobar.core import *
from tqdm import tqdm


## Equations: Plag-Liquid thermometers
def T_Put2008_eq23(P, *, An_Plag, Si_Liq_cat_frac,
                   Al_Liq_cat_frac, Ca_Liq_cat_frac, Ab_Plag, H2O_Liq):
    '''
    Plagioclase-Liquid thermometer of Putirka (2008) eq. 23
    :cite:`putirka2008thermometers`
    SEE=+-43C
    '''
    return ((10**4 / (6.12 + 0.257 * np.log(An_Plag.astype(float) / (Si_Liq_cat_frac.astype(float)**2 * Al_Liq_cat_frac.astype(float)**2 * Ca_Liq_cat_frac.astype(float)))
                      - 3.166 * Ca_Liq_cat_frac - 3.137 *
                      (Al_Liq_cat_frac / (Al_Liq_cat_frac + Si_Liq_cat_frac))
                      + 1.216 * Ab_Plag**2 - 2.475 * 10**-2 * (P / 10) * 10 + 0.2166 * H2O_Liq)))




def T_Put2008_eq24a(P, *, An_Plag, Si_Liq_cat_frac, Al_Liq_cat_frac,
                    Ca_Liq_cat_frac, K_Liq_cat_frac, Ab_Plag, H2O_Liq):
    '''
    Plagioclase-Liquid thermometer of Putirka (2008) eq. 24a
    :cite:`putirka2008thermometers`

    Global regression, improves equation 23 SEE by 6C
    SEE=+-36C
    '''
    return ((10**4 / (6.4706 + 0.3128 * (np.log(An_Plag.astype(float) / (Si_Liq_cat_frac.astype(float)**2 * Al_Liq_cat_frac.astype(float)**2 * Ca_Liq_cat_frac.astype(float))))
                      - 8.103 * Si_Liq_cat_frac + 4.872 *
                      K_Liq_cat_frac + 8.661 * Si_Liq_cat_frac**2
                      + 1.5346 * Ab_Plag**2 - 3.341 * 10**-2 * (P / 10) * 10 + 0.18047 * H2O_Liq)))


## Equation: Plag-Liquid barometer

def P_Put2008_eq25(T, *, Ab_Plag, Al_Liq_cat_frac, Ca_Liq_cat_frac,
                   Na_Liq_cat_frac, Si_Liq_cat_frac, An_Plag, K_Liq_cat_frac):
    '''
    Plagioclase-Liquid barometer of Putirka (2008) eq. 26.
    :cite:`putirka2008thermometers`

    SEE=+-2.2 kbar *But in spreadsheet, Putirka warns plag is not a good barometer*
    '''
    return (-42.2 + (4.94 * (10**-2) * T) + (1.16 * (10**-2) * T) *
    (np.log((Ab_Plag.astype(float) * Al_Liq_cat_frac.astype(float) * Ca_Liq_cat_frac.astype(float)) / (Na_Liq_cat_frac.astype(float) * Si_Liq_cat_frac.astype(float) * An_Plag.astype(float))))
            - 19.6 * np.log(Ab_Plag.astype(float)) - 382.3 * Si_Liq_cat_frac**2 +
            514.2 * Si_Liq_cat_frac**3 - 139.8 * Ca_Liq_cat_frac
            + 287.2 * Na_Liq_cat_frac + 163.9 * K_Liq_cat_frac)

## Equations - Kspar-Liquid thermometry

def T_Put2008_eq24b(P, *, Ab_Kspar, Al_Liq_cat_frac, Na_Liq_cat_frac,
                  K_Liq_cat_frac, Si_Liq_cat_frac, Ca_Liq_cat_frac):
    '''
    Alkali Felspar-Liquid thermometer of Putirka (2008) eq. 24b.
    :cite:`putirka2008thermometers`

    SEE=+-23 C (Calibration data)

    SEE=+-25 C (All data)

    '''
    return (10**4 / (17.3 - 1.03 * np.log(Ab_Kspar.astype(float) / (Na_Liq_cat_frac.astype(float) * Al_Liq_cat_frac.astype(float) * Si_Liq_cat_frac.astype(float)**3))
    - 200 * Ca_Liq_cat_frac - 2.42 * Na_Liq_cat_frac -29.8 * K_Liq_cat_frac + 13500 * (Ca_Liq_cat_frac - 0.0037)**2
    - 550 * (K_Liq_cat_frac - 0.056) * (Na_Liq_cat_frac - 0.089) - 0.078 * (P / 10) / 10))





## Function: Fspar-Liquid temperature (works for alkali and plag feldspars)

plag_liq_T_funcs = {T_Put2008_eq23, T_Put2008_eq24a}
plag_liq_T_funcs_by_name = {p.__name__: p for p in plag_liq_T_funcs}

Kspar_Liq_T_funcs = {T_Put2008_eq24b}
Kspar_Liq_T_funcs_by_name = {p.__name__: p for p in Kspar_Liq_T_funcs}



def calculate_fspar_liq_temp(*, plag_comps=None, kspar_comps=None, meltmatch_plag=None,
meltmatch_kspar=None,
    liq_comps=None, equationT=None, P=None, H2O_Liq=None, eq_tests=False, warnAn=False):
    '''
    Liquid-Feldspar thermometery (same function for Plag and Kspar),
    returns temperature in Kelvin.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    kspar_comps or plag_comps (pandas.DataFrame)

        Specify kspar_comps=... for Kspar-Liquid thermometry (with column headings SiO2_Kspar, MgO_Kspar) etc

        Specify plag_comps=... for Plag-Liquid thermometry (with column headings SiO2_Plag, MgO_Plag) etc

    EquationT: str

        choose from:

            |   T_Put2008_eq24b (Kspar-Liq, P-dependent, H2O-independent
            |   T_Put2008_eq23 (Plag-Liq, P-dependent, H2O-dependent)
            |   T_Put2008_eq24a (Plag-Liq, P-dependent, H2O-dependent)

    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar to perform calculations at
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    H2O_Liq: optional.
        If None, uses H2O_Liq column from input.
        If int, float, pandas.Series, uses this instead of H2O_Liq Column

    Returns
    -------

        Temperature in Kelvin: pandas.Series
            If eq_tests is False

        Temperature in Kelvin + eq Tests + input compositions: pandas.DataFrame
            If eq_tests is True

    '''
    if meltmatch_plag is None and meltmatch_kspar is None and plag_comps is not None and liq_comps is not None:
        if len(plag_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as plag comps. If you want to match up all possible pairs, use the _matching functions instead')

    if meltmatch_plag is None and meltmatch_kspar is None and kspar_comps is not None and liq_comps is not None:
        if len(kspar_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as kspar comps. If you want to match up all possible pairs, use the _matching functions instead')

    if kspar_comps is not None:
        kspar_comps_c=kspar_comps.copy()
        kspar_comps_c=kspar_comps_c.reset_index(drop=True)
    if plag_comps is not None:
        plag_comps_c=plag_comps.copy()
        plag_comps_c=plag_comps_c.reset_index(drop=True)
    if liq_comps is not None:
        liq_comps_c=liq_comps.copy()
        liq_comps_c=liq_comps_c.reset_index(drop=True)
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq


    if plag_comps is not None or meltmatch_plag is not None:
        try:
            func = plag_liq_T_funcs_by_name[equationT]
        except KeyError:
            raise ValueError(f'{equationT} is not a valid equation for Plag-Liquid') from None
        sig=inspect.signature(func)


    if kspar_comps is not None or meltmatch_kspar is not None:
        try:
            func = Kspar_Liq_T_funcs_by_name[equationT]
        except KeyError:
            raise ValueError(f'{equationT} is not a valid equation for Kspar-Liquid') from None
        sig=inspect.signature(func)



    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')
    else:
        if P is not None:
            print('Youve selected a P-independent function')



    if plag_comps is not None:
        cat_plags = calculate_cat_fractions_plagioclase(plag_comps=plag_comps_c)
        cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
        combo_fspar_Liq = pd.concat([cat_plags, cat_liqs], axis=1)
        if any(combo_fspar_Liq['An_Plag'] < 0.05):
            if warnAn is True:
                w.warn('Some inputted feldspars have An<0.05, but you have selected a plagioclase-liquid thermometer'
            '. If these are actually alkali felspars, please use T_P2008_eq24b or T_P2008_24c instead', stacklevel=2)
        Kd_Ab_An = (combo_fspar_Liq['Ab_Plag'] * combo_fspar_Liq['Al_Liq_cat_frac'] * combo_fspar_Liq['Ca_Liq_cat_frac'] /
                    (combo_fspar_Liq['An_Plag'] * combo_fspar_Liq['Na_Liq_cat_frac'] * combo_fspar_Liq['Si_Liq_cat_frac']))
        combo_fspar_Liq['Kd_Ab_An'] = Kd_Ab_An


    if kspar_comps is not None:
        cat_kspars = calculate_cat_fractions_kspar(kspar_comps=kspar_comps_c)
        cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
        combo_fspar_Liq = pd.concat([cat_kspars, cat_liqs], axis=1)

        Kd_Ab_An = (combo_fspar_Liq['Ab_Kspar'] * combo_fspar_Liq['Al_Liq_cat_frac'] * combo_fspar_Liq['Ca_Liq_cat_frac'] /
                    (combo_fspar_Liq['An_Kspar'] * combo_fspar_Liq['Na_Liq_cat_frac'] * combo_fspar_Liq['Si_Liq_cat_frac']))
        combo_fspar_Liq['Kd_Ab_An'] = Kd_Ab_An

        if np.min(combo_fspar_Liq['An_Kspar'] > 0.05):
            w.warn('Some inputted feldspars have An>0.05, but you have selected a Kspar-liquid thermometer'
            '. If these are actually Plagioclase feldspars, please use T_P2008_eq23 or _eq24a instead', stacklevel=2)


    if meltmatch_kspar is not None:
        combo_fspar_Liq=meltmatch_kspar
    if meltmatch_plag is not None:
        combo_fspar_Liq=meltmatch_plag

    kwargs = {name: combo_fspar_Liq[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:
        T_K=func(P, **kwargs)

    if eq_tests is False:
        return T_K

    if eq_tests is True:
        if kspar_comps is not None:
            print('Sorry, no equilibrium tests implemented for Kspar-Liquid')
            return T_K
        if plag_comps is not None:
            eq_tests=calculate_plag_liq_eq_tests(liq_comps=liq_comps_c,
            plag_comps=plag_comps_c, P=P, T=T_K)
            eq_tests.insert(0, "T_K_calc", T_K)
            return eq_tests


## Function: Fspar-Liq pressure
plag_liq_P_funcs = {P_Put2008_eq25}
plag_liq_P_funcs_by_name = {p.__name__: p for p in plag_liq_P_funcs}

Kspar_Liq_P_funcs = {}
Kspar_Liq_P_funcs_by_name = {p.__name__: p for p in Kspar_Liq_P_funcs}



def calculate_fspar_liq_press(*, plag_comps=None, kspar_comps=None, liq_comps=None, equationP=None,
                              T=None, H2O_Liq=None, eq_tests=False):

    '''
    Liquid-Feldspar barometer (at the moment, only options for Plag).
    Note, Putirka warns that plagioclase is not a reliable barometer!
    For a user-selected equation returns a pressure in kbar.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    plag_comps: pandas.DataFrame
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc.

    EquationP: str

        Choose from

            |   P_Put2008_eq25 (Plag-Liq, P-dependent, H2O-independent)

    T: float, int, pandas.Series, str  ("Solve")
        Temperature in Kelvin to perform calculations at.
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function,
        else, enter an integer, float, or panda series.

    H2O_Liq: optional.
        If None, uses H2O_Liq column from input.
        If int, float, pandas.Series, uses this instead of H2O_Liq Column

    Returns
    -------
    If eq_tests is False: pandas.Series
        Pressure in kbar
    If eq_tests is True: pandas.DataFrame
        Pressure in kbar + eq Tests + input compositions


    '''
    if  plag_comps is not None and liq_comps is not None:
        if len(plag_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as plag comps. We dont currently having a matching function for plag-liq barometry, as it is so bad, but if you really need one, reach out!')

    if kspar_comps is not None and liq_comps is not None:
        if len(kspar_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as kspar comps. We dont currently having a matching function for plag-liq barometry, as it is so bad, but if you really need one, reach out!')

    if kspar_comps is not None:
        kspar_comps_c=kspar_comps.copy()
        kspar_comps_c=kspar_comps_c.reset_index(drop=True)
    if plag_comps is not None:
        plag_comps_c=plag_comps.copy()
        plag_comps_c=plag_comps_c.reset_index(drop=True)
    if liq_comps is not None:
        liq_comps_c=liq_comps.copy()
        liq_comps_c=liq_comps_c.reset_index(drop=True)
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

    if kspar_comps is not None:
        raise TypeError('Sorry, this tool doesnt contain any alkali-fspar-liquid barometers,'
        ' this option is here as a place holder incase a great one comes along!')

    try:
        func = plag_liq_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None
    sig=inspect.signature(func)







    cat_plags = calculate_cat_fractions_plagioclase(plag_comps=plag_comps_c)
    cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
    combo_plag_liq = pd.concat([cat_plags, cat_liqs], axis=1)

    Kd_Ab_An = (combo_plag_liq['Ab_Plag'] * combo_plag_liq['Al_Liq_cat_frac'] * combo_plag_liq['Ca_Liq_cat_frac'] /
                (combo_plag_liq['An_Plag'] * combo_plag_liq['Na_Liq_cat_frac'] * combo_plag_liq['Si_Liq_cat_frac']))

    combo_plag_liq['Kd_Ab_An'] = Kd_Ab_An


    kwargs = {name: combo_plag_liq[name] for name, p in sig.parameters.items() \
    if p.kind == inspect.Parameter.KEYWORD_ONLY}

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T')
    else:
        if T is not None:
            print('Youve selected a T-independent function')


    kwargs = {name: combo_plag_liq[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(T, str) or T is None:
        if T == "Solve":
            P_kbar = partial(func, **kwargs)
        if T is None:
            P_kbar=func(**kwargs)

    else:
        P_kbar=func(T, **kwargs)

    if eq_tests is False:
        return P_kbar
    if eq_tests is True:

        eq_tests=calculate_plag_liq_eq_tests(liq_comps=liq_comps_c,
        plag_comps=plag_comps_c, P=P_kbar, T=T)
        eq_tests.insert(1, "P_kbar_calc", P_kbar)
        return eq_tests
    return P_kbar

## Function for iterating pressure and temperature - (probably not recomended)

def calculate_fspar_liq_press_temp(*, liq_comps=None, plag_comps=None, kspar_comps=None,
                                meltmatch=None, equationP=None, equationT=None, iterations=30, T_K_guess=1300,
                                H2O_Liq=None, eq_tests=False):
    '''
    Solves simultaneous equations for temperature and pressure using
    feldspar-liquid thermometers and barometers. Currently no Kspar barometers exist.

    Parameters
    -----------

    plag_comps: pandas.DataFrame
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc.

    liq_comps: pandas.DataFrame (not required for P_Put2008_eq29c)
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    EquationP: str

        choose from;

            |   P_Put2008_eq25 (Plag-Liq, P-dependent, H2O-independent)

    EquationT: str

        choose from:

            |   T_Put2008_eq24b (Kspar-Liq, P-dependent, H2O-independent)
            |   T_Put2008_eq23 (Plag-Liq, P-dependent, H2O-dependent)
            |   T_Put2008_eq24a (Plag-Liq, P-dependent, H2O-dependent)

    iterations: int (default=30)
         Number of iterations used to converge to solution.

    T_K_guess: int or float (Default = 1300K)
         Initial guess of temperature.

    eq_tests: bool


    Returns
    ---------
    pandas.DataFrame: Pressure in kbar, Temperature in K + fspar+liq comps (if eq_tests=True)
    '''
    # Gives users flexibility to reduce or increase iterations


    if plag_comps is not None:
        plag_comps_c=plag_comps.copy()
        plag_comps_c=plag_comps_c.reset_index(drop=True)
    if liq_comps is not None:
        liq_comps_c=liq_comps.copy()
        liq_comps_c=liq_comps_c.reset_index(drop=True)
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

    if  plag_comps is not None and liq_comps is not None:
        if len(plag_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as plag comps. We dont currently having a matching function for plag-liq barometry, as it is so bad, but if you really need one, reach out!')





    if kspar_comps is not None:
        raise Exception(
            'Sorry, we currently dont have any kspar barometers coded up, so you cant iteratively solve for Kspars')


    if meltmatch is None:
        T_func = calculate_fspar_liq_temp(
            plag_comps=plag_comps_c, liq_comps=liq_comps_c, equationT=equationT, P="Solve")
        P_func = calculate_fspar_liq_press(
            plag_comps=plag_comps_c, liq_comps=liq_comps_c, equationP=equationP,  T="Solve")
    if meltmatch is not None:
        T_func = calculate_fspar_liq_temp(
            meltmatch=meltmatch, equationT=equationT)
        P_func = calculate_fspar_liq_press(
            meltmatch=meltmatch, equationP=equationP)

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
        with w.catch_warnings():
            w.simplefilter('ignore')

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

    # calculates Kd Fe-Mg if eq_tests="True"

    else:
        DeltaP=0
        DeltaT=0

    if eq_tests is True:

        eq_tests=calculate_plag_liq_eq_tests(liq_comps=liq_comps_c,
        plag_comps=plag_comps_c, P=P_guess, T=T_K_guess)
        eq_tests.insert(1, "P_kbar_calc", P_guess)
        eq_tests.insert(2, "T_K_calc", T_K_guess)
        eq_tests.insert(3, "Delta_P_kbar_Iter", DeltaP)
        eq_tests.insert(4, 'Delta_T_K_Iter',DeltaT)
        return eq_tests

    else:
         PT_out = PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess,
                                    'Delta_P_kbar_Iter': DeltaP,
                                    'Delta_T_K_Iter': DeltaT})
         return PT_out




## Equations: Plag-Liquid hygrometry

def H_Put2008_eq25b(T, *, P, An_Plag, Si_Liq_cat_frac,
                    Al_Liq_cat_frac, Ca_Liq_cat_frac, K_Liq_cat_frac, Mg_Liq_cat_frac):
    '''
    Plagioclase-Hygrometer of  Putirka (2008) eq. 25b
    :cite:`putirka2008thermometers`


    Global calibration improving estimate of Putirka (2005) eq H
    SEE=+-1.1 wt%
    '''
    LnKAn=np.log((An_Plag.astype(float))/((Si_Liq_cat_frac.astype(float)**2)*(Al_Liq_cat_frac.astype(float)**2) * Ca_Liq_cat_frac.astype(float)))
    return (25.95-0.0032*(T-273.15)*LnKAn-18.9*K_Liq_cat_frac+14.5*Mg_Liq_cat_frac-40.3*Ca_Liq_cat_frac+5.7*An_Plag**2+0.108*(P))

def H_Masotta2019(T, *, An_Plag, Ab_Plag, Si_Liq_cat_frac, Al_Liq_cat_frac, Na_Liq_cat_frac,
                  Ca_Liq_cat_frac,  K_Liq_cat_frac):
    '''
    Plagioclase-Hygrometer of  Masotta et al. (2019), for trachytic systems.
    :cite:`masotta2019new`
    '''

    return (46.2207233262084
    -0.329007908696103* (np.log(An_Plag.astype(float) / (Si_Liq_cat_frac.astype(float)**2 * Al_Liq_cat_frac.astype(float)**2 * Ca_Liq_cat_frac.astype(float))))
    -0.0348279402544078*(T-273.15) -12.306919163926*Ab_Plag
    -1.30868665306982* (Na_Liq_cat_frac/(Na_Liq_cat_frac+K_Liq_cat_frac)))


def H_Put2005_eqH(T, *, An_Plag, Si_Liq_cat_frac, Al_Liq_cat_frac, Na_Liq_cat_frac,
                  Ca_Liq_cat_frac, Pred_Ab_EqF, Pred_An_EqE, K_Liq_cat_frac, Mg_Liq_cat_frac):
    '''
    Plagioclase-Hygrometer of  Putirka (2005) eq. H
    Tends to overpredict water in global datasets
    :cite:`putirka2005igneous`
    '''

    return (24.757 - 2.26 * 10**(-3) * (T) * (np.log(An_Plag.astype(float) / (Si_Liq_cat_frac.astype(float)**2 * Al_Liq_cat_frac.astype(float)**2 * Ca_Liq_cat_frac.astype(float))))
            - 3.847 * Pred_Ab_EqF + 1.927 * Pred_An_EqE / (Ca_Liq_cat_frac / (Ca_Liq_cat_frac + Na_Liq_cat_frac)))


def H_Waters2015(T, *, liq_comps=None, plag_comps=None,
                 P, XAn=None, XAb=None):
    '''
    Plagioclase-Hygrometer of  Waters and Lange (2015),
    update from Lange et al. (2009).
    :cite:`waters2015updated`

    SEE=+-0.35 wt%

    '''
    T=T+0.000000000001 # This stops it being an integer
    # 1st bit calculates mole fractions in the same way as Waters and Lange.
    # Note, to match excel, all Fe is considered as FeOT.
    mol_prop = calculate_anhydrous_mol_proportions_liquid(liq_comps=liq_comps)
    mol_prop['Al2O3_Liq_mol_prop_v2'] = mol_prop['Al2O3_Liq_mol_prop'] - \
        mol_prop['K2O_Liq_mol_prop']
    # These are set to zero, so the sum of molar proportions is the same as in
    # the Waters and Lange Matlab script
    mol_prop['Al2O3_Liq_mol_prop'] = 0
    mol_prop['P2O5_Liq_mol_prop'] = 0
    mol_prop['Cr2O3_Liq_mol_prop'] = 0
    mol_prop['MnO_Liq_mol_prop'] = 0
    # This sums the remaining non-zero columns
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    Liq_anhyd_mol_frac = mol_prop.div(mol_prop['sum'], axis='rows')
    Liq_anhyd_mol_frac.drop(['sum'], axis='columns', inplace=True)
    Liq_anhyd_mol_frac.columns = [str(col).replace(
        'prop', 'frac') for col in Liq_anhyd_mol_frac.columns]

    if XAn is not None and XAb is not None:
        XAn_Plag = XAn
        XAb_Plag = XAb
    if isinstance(plag_comps, pd.DataFrame):
        feld_cat_frac = calculate_cat_fractions_plagioclase(
            plag_comps=plag_comps)
        XAn_Plag = feld_cat_frac['An_Plag']
        XAb_Plag = feld_cat_frac['Ab_Plag']

    # The other option is they just enter Xan and Xab: Then we use the
    # existing functions to calculate plag cat fractions, which are then used
    # to calculate An and Ab.

    # Returns warning

    Al2O3_Liq_mol_frac_v2 = Liq_anhyd_mol_frac['Al2O3_Liq_mol_frac_v2']
    FeOt_Liq_mol_frac = Liq_anhyd_mol_frac['FeOt_Liq_mol_frac']
    Na2O_Liq_mol_frac = Liq_anhyd_mol_frac['Na2O_Liq_mol_frac']
    CaO_Liq_mol_frac = Liq_anhyd_mol_frac['CaO_Liq_mol_frac']
    SiO2_Liq_mol_frac = Liq_anhyd_mol_frac['SiO2_Liq_mol_frac']
    TiO2_Liq_mol_frac = Liq_anhyd_mol_frac['TiO2_Liq_mol_frac']
    MgO_Liq_mol_frac = Liq_anhyd_mol_frac['MgO_Liq_mol_frac']
    K2O_Liq_mol_frac = Liq_anhyd_mol_frac['K2O_Liq_mol_frac']

    AnTerm1 =( (-0.000000000010696 * ((T - 273.15)**4)) + (0.00000004945054 * ((T - 273.15)**3))
    - (0.00008207491 * ((T - 273.15)**2)) + (0.05275448 * (T - 273.15)) - 7.914622)

    AnTerm2 = ((0.000000000000219441 * ((T - 273.15)**5)) - (0.00000000106298 * ((T - 273.15)**4)) +
        (0.00000200317 * ((T - 273.15)**3)) - (0.00182716 *((T - 273.15)**2)) + (0.811309 * (T - 273.15)) - 146.951)

    AnTerm3 = ((-0.0000000000000002443527 * ((T - 273.15)**6)) + (0.0000000000013185282 * ((T - 273.15)**5))
    - (0.0000000028995878 *((T - 273.15)**4)) + (0.0000033377181 * ((T - 273.15)**3))
    - (0.0021408319 * ((T - 273.15)**2)) + (0.73454132 * (T - 273.15)) - 104.25573)

    AnTerm4 = ((0.00000000000000009642821 * ((T - 273.15)**6)) - (0.0000000000005482202 * ((T - 273.15)**5))
    + (0.000000001280331 *((T - 273.15)**4)) - (0.00000157613 * ((T - 273.15)**3))
    + (0.001084872 * ((T - 273.15)**2)) - (0.3997902 * (T - 273.15)) + 63.39784)


    AnConstant = ((-0.000000000000000006798238 * ((T - 273.15)**6)) + (0.00000000000004008407 * ((T - 273.15)**5))
    - (0.0000000000972437 *((T - 273.15)**4)) + (0.0000001242399 * ((T - 273.15)**3))
    - (0.00008824071 * ((T - 273.15)**2)) + (0.03310392 * (T - 273.15)) - 5.136283)
    #
    AbTerm1 = ((0.0000000000311852 * ((T - 273.15)**4)) - (0.000000124156 * ((T - 273.15)**3))
    + (0.000181664 * ((T - 273.15)**2)) - (0.110128 * (T - 273.15)) + 19.9603)

    AbTerm2 = ((-0.00000000006683324 * ((T - 273.15)**4)) + (0.000000264682 * ((T - 273.15)**3))
    - (0.0003813719 * ((T - 273.15)**2)) + (0.2230349 * (T - 273.15)) - 35.75847)

    AbTerm3 = ((0.000000000045560722 * ((T - 273.15)**4)) - (0.00000017904633 * ((T - 273.15)**3))
    + (0.00025203154 * ((T - 273.15)**2)) - (0.13902056 * (T - 273.15)) + 17.479424)

    AbTerm4 = ((-0.00000000001038026 * ((T - 273.15)**4)) + (0.00000004046137 * ((T - 273.15)**3))
     - (0.00005510837 * ((T - 273.15)**2)) + (0.02766904 * (T - 273.15)) - 0.9465208)

    AbConstant = ((0.00000000000000003664241 * ((T - 273.15)**6)) - (0.0000000000002106807 * ((T - 273.15)**5))
    + (0.0000000004975989 *((T - 273.15)**4)) - (0.0000006178773 * ((T - 273.15)**3))
    + (0.0004252932 * ((T - 273.15)**2)) - (0.1537074 * (T - 273.15)) + 22.74359)

    Aab = (AbTerm1 * (XAb_Plag**4)) + (AbTerm2 * (XAb_Plag**3)) + \
        (AbTerm3 * (XAb_Plag**2)) + (AbTerm4 * XAb_Plag) + AbConstant
    Aan = (AnTerm1 * (XAn_Plag**4)) + (AnTerm2 * (XAn_Plag**3)) + \
        (AnTerm3 * (XAn_Plag**2)) + (AnTerm4 * XAn_Plag) + AnConstant

    # volume of albite crystal
    VolAbxtal = 100.570 * np.exp(0.0000268 * (T - 298))
    # volume of anorthite crystal
    VolAnxtal = 100.610 * np.exp(0.0000141 * (T - 298))
    VliqAb = 112.715 + (0.00382 * (T - 1373))  # volume of liquid at 1 bar
    VliqAn = 106.300 + (0.003708 * (T - 1673))  # volume of liquid at 1 bar
    dVdPliqAb = (0.75 * (-1.843) + 0.125 * (0.685 + 0.0024 * (T - 1673)) + 0.125 * (-2.384 -
                 0.0035 * (T - 1673))) * 4 / 10000  # Change in volume of albite liquid with temperature
    # Change in volume of anorthite liquid with temperature
    dVdPliqAn = (0.50 * (-1.906) + 0.250 * (-1.665) + 0.25 *
                 (0.295 - (0.00101 * (T - 1673)))) * 4 / 10000
    # 1st part of solution to the pressure integral for albite AKA
    # deltaV(P-1)albite(J)
    deltaValbite_J = 0.1 * (VliqAb - VolAbxtal) * ((P * 1000) - 1)
    # 2nd part of the solution to the pressure integral for albite AKA
    # dV/dP**2albite(J)
    deltaValbite2_J = 0.1 * ((dVdPliqAb - 0.0000167) / 2) * ((P * 1000)**2)
    # 1st part of the solution to the pressure integral for anorthite AKA
    # deltaV(P-1)anorth(J)
    deltaVanorthite_J = 0.1 * (VliqAn - VolAnxtal) * ((P * 1000) - 1)
    # second part of the solution to the pressure integral for anorthiteAKA
    # dV/dP**2anorthtie(J)
    deltaVanorthite2_J = 0.1 * ((dVdPliqAn - 0.0000116) / 2) * ((P * 1000)**2)
    # the volume change between anorthite liquid and cyrstal at temperature
    # and pressure
    intdeltaVAn = deltaVanorthite_J + deltaVanorthite2_J
    # the volume change between albite liquid and cyrstal at temperature and
    # pressure
    intdeltaVAb = deltaValbite_J + deltaValbite2_J
    # the total change in volume between anorthite and albite crystal and
    # liquid
    intdeltaV = - deltaValbite_J - deltaValbite2_J + \
        deltaVanorthite_J + deltaVanorthite2_J

    XAbliq = ((Na2O_Liq_mol_frac**0.5) * (Al2O3_Liq_mol_frac_v2**0.5) *
              (SiO2_Liq_mol_frac**3)) * 18.963     # mol fraction of albite in the liquid
    # mol fraction of anorthite in the liquid
    XAnliq = (CaO_Liq_mol_frac * Al2O3_Liq_mol_frac_v2 *
              (SiO2_Liq_mol_frac**2)) * 64
    XAn_AnAb_liquid = XAnliq / (XAbliq + XAnliq) * 100  # liquid An #
    lnK = (np.log(Aab) - np.log(Aan)) + \
        (np.log(XAnliq) - np.log(XAbliq))  # equilibrium constant
    # overall change in volume considering the effects of temperature and
    # pressure divided by the constant R and T in kelvin
    volterm = intdeltaV / (8.3144 * T)

    intdeltaCp_An = ((-5.77) * (T - 1830)) - ((-3734 / 0.5) * (T**0.5 - 1830**0.5)) - ((317020000 / -2)
     * (T**-2 - 1830**-2))  # AKA the integral of the heat capacity of anorthite crystal
    deltaHfus_An = 142406 + intdeltaCp_An
    intdeltaCp_Ab = ((-35.64) * (T - 1373)) - ((-2415.5 / 0.5) * (T**0.5 - 1373**0.5)) - \
        ((789280) * ((T**-1) - (1373**-1))) - \
        ((1070640000 / -2) * ((T**-2) - (1373**-2)))
    deltaHfus_Ab = 64500 + intdeltaCp_Ab
    Cap_deltaH_AnAb = deltaHfus_An - deltaHfus_Ab
    Cap_deltaH_RT = Cap_deltaH_AnAb / (8.3144 * T)
    intCp_divdedTfus_An = (-5.77 * (np.log(T) - np.log(1830)) + (3734 / -0.5) * ((T**-0.5) - (
        1830**-0.5)) + (0 / -2) * ((T**-2) - (1830**-2)) - (317020000 / -3) * ((T**-3) - (1830**-3)))
    deltaSfus_An = 77.82 + intCp_divdedTfus_An
    intCp_divdedTfus_Ab = -35.64 * (np.log(T) - np.log(1373)) + (2415.5 / -0.5) * ((T**-0.5) - (
        1373**-0.5)) + (7892800 / -2) * (T**-2 - 1373**-2) - (1070640000 / -3) * ((T**-3) - (1373**-3))
    deltaSfus_Ab = 47 + intCp_divdedTfus_Ab
    CapDeltaS_R = (deltaSfus_An - deltaSfus_Ab) / 8.3144
    H_TdeltaS_An = (deltaHfus_An - (T * deltaSfus_An)) / 1000
    H_TdeltaS_Ab = (deltaHfus_Ab - (T * deltaSfus_Ab)) / 1000
    Gibbs_exchange = H_TdeltaS_An - H_TdeltaS_Ab
    Gibbs_exchange_RT = 1000 * Gibbs_exchange / (8.31441 * T)
    P_T = (P * 1000) / T
    H_RTminusS_R = Cap_deltaH_RT - CapDeltaS_R
    neglnK_V_G = -1 * (lnK + volterm + Gibbs_exchange_RT)
    T_term = 10000 / T
    negSiO2_Liq_mol_frac = -1 * SiO2_Liq_mol_frac
    negTiO2_Liq_mol_frac = -1 * TiO2_Liq_mol_frac
    negAl2O3_Liq_mol_frac_v2 = -1 * Al2O3_Liq_mol_frac_v2
    negFeOt_Liq_mol_frac = -1 * FeOt_Liq_mol_frac
    negMgO_Liq_mol_frac = -1 * MgO_Liq_mol_frac
    negCaO_Liq_mol_frac = -1 * CaO_Liq_mol_frac
    negNa2O_Liq_mol_frac = -1 * Na2O_Liq_mol_frac
    negK2O_Liq_mol_frac = -1 * K2O_Liq_mol_frac

    wtH2Ocalc = (- 17.3204203938587 + (0.389218669342048 * neglnK_V_G) + (2.98588693659695 * T_term)
    + (7.8289140199477 * negSiO2_Liq_mol_frac) - (50.1063951084878 * negAl2O3_Liq_mol_frac_v2) +
     (14.114740308799 * negFeOt_Liq_mol_frac) + (23.9996276026497 * negMgO_Liq_mol_frac)
     - (15.9003801663855 * negCaO_Liq_mol_frac) + (18.6326909831708 * negNa2O_Liq_mol_frac)
     + (24.0180473651546 * negK2O_Liq_mol_frac))

    if XAn is not None and XAb is not None and plag_comps is None:
        Combo_Out = pd.DataFrame(
            data={'H2O_calc': wtH2Ocalc, 'An': XAn, 'Ab': XAb})
        return Combo_Out
    else:
        Combo_Out = pd.concat(
            [feld_cat_frac, liq_comps, Liq_anhyd_mol_frac], axis=1)
        cols_to_move = ['An_Plag', 'Ab_Plag']
        Combo_Out = Combo_Out[cols_to_move +
                              [col for col in Combo_Out.columns if col not in cols_to_move]]
        Combo_Out.insert(0, "H2O_calc", wtH2Ocalc)
    return Combo_Out

## Function for Plag-Liq hygrometry

plag_liq_H_funcs = {H_Put2008_eq25b, H_Put2005_eqH, H_Waters2015, H_Masotta2019}
plag_liq_H_funcs_by_name = {p.__name__: p for p in plag_liq_H_funcs}



def calculate_fspar_liq_hygr(*, liq_comps, plag_comps=None, kspar_comps=None,
                            equationH=None, P=None, T=None, XAn=None, XAb=None, XOr=0):
    '''
    calculates H2O content (wt%) from composition of equilibrium plagioclase
    and liquid.

    Parameters
    ------------

    liq_comps: pandas.DataFrame
        Liq compositions with column headings SiO2_Liq, MgO_Liq etc.

    plag_comps: pandas.DataFrame
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc. Or users
        can enter XAn, XAb, without the other oxides.

    XAn+XAb: float, int, pandas.Series
        If plag_comps is None, enter XAn= and XAb= for plagioclases instead.
        XOr is set to zero by default, but can be overwritten for equilibrium tests

    equationH: str
        choose from:

        |  H_Waters2015 (T-dependent, P-dependent)
        |  H_Masotta2019 (T-dependent, P-independent, for Trachytes)
        |  H_Put2005_eqH (T-dependent, P-independent)
        |  H_Put2008_eq25b (T-dependent, P-dependent)


    P: float, int, pandas.Series
        Pressure in kbar to perform calculations at

    T: float, int, pandas.Series
        Temperature in Kelvin to perform calculations at

    Returns
    ---------

    Calculated H2O, eq tests, and input plag and liq parameters: pandas.DataFrame

    '''
    if  plag_comps is not None and liq_comps is not None:
        if len(plag_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as plag comps. We dont currently having a matching function for plag-liq hygrometry, but if you need one, reach out!')



    if kspar_comps is not None:
        raise ValueError('Sorry, no k-fspar hygrometers implemented in this tool. You must enter plag_comps=')
    try:
        func = plag_liq_H_funcs_by_name[equationH]
    except KeyError:
        raise ValueError(f'{equationH} is not a valid equation') from None
    sig=inspect.signature(func)

    if equationH=="H_Put2008_eq25b" or equationH=="H_Waters2015":
        if P is None:
            raise ValueError(f'{equationH} requires you to enter P. If you dont know this'
            ' Waters and Lange (2015) suggesting entering P=1')

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationH} requires you to enter T')
    else:
        if T is not None:
            print('Youve selected a T-independent function')


    # Then, warn if enter contradicting inputs;
    if plag_comps is not None and XAn is not None:
        raise Exception(
            'You have entered both a plag composition, and an XAn value'
            ', so the code doesnt know which to use. Either enter a full feld composition,'
            ' OR XAn and XAb values')
    if plag_comps is not None and XAb is not None:
        raise Exception(
            'You have entered both a feld composition, and an XAb value,'
            ' so the code doesnt know what to use. Either enter a full feld composition,'
            ' OR XAn and XAb values')

    if plag_comps is None and XAb is None:
        raise Exception('You must enter either plag_comps=, or an XAb and XAn content')


    if plag_comps is not None:
        plag_comps_c=plag_comps.copy()
        plag_comps_c=plag_comps_c.reset_index(drop=True)
    if liq_comps is not None:
        liq_comps_c=liq_comps.copy()
        liq_comps_c=liq_comps_c.reset_index(drop=True)



    if equationH == "H_Waters2015":
        if XAn is not None and XAb is not None and plag_comps is None:
            Combo_Out = H_Waters2015(
                liq_comps=liq_comps_c, P=P, T=T, XAn=XAn, XAb=XAb)
            eq_tests=calculate_plag_liq_eq_tests(liq_comps=liq_comps_c, XAn=XAn, XAb=XAb,
                P=P, T=T)



        if plag_comps is not None:

            Combo_Out = H_Waters2015(
                liq_comps=liq_comps_c, plag_comps=plag_comps_c, P=P, T=T)
            eq_tests=calculate_plag_liq_eq_tests(plag_comps=plag_comps_c,
                liq_comps=liq_comps_c, P=P, T=T)

        eq_tests.insert(0, 'H2O_calc', Combo_Out['H2O_calc'])

        return eq_tests


    if equationH == "H_Put2008_eq25b" or equationH == "H_Put2005_eqH" or equationH == "H_Masotta2019":
        if P is None:
            raise TypeError('even if the equation doesnt require a P to be entered'
            ' because we want to return you eq tests, you need to enter a P')
        if plag_comps is not None:
            combo_plag_liq = calculate_plag_liq_eq_tests(
                liq_comps=liq_comps_c, plag_comps=plag_comps_c, T=T, P=P)
        if plag_comps is None:
            combo_plag_liq = calculate_plag_liq_eq_tests(
                liq_comps=liq_comps_c, T=T, P=P, XAn=XAn, XAb=XAb)
            combo_plag_liq['An_Plag'] = XAn
            combo_plag_liq['Ab_Plag'] = XAb

        kwargs = {name: combo_plag_liq[name] for name,
        p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

        H2O_out = func(T, **kwargs)
        combo_plag_liq.insert(0, "H2O_calc", H2O_out)
        return combo_plag_liq


## Function for iterating temperature and water content

# plag_liq_T_funcs = {T_Put2008_eq23, T_Put2008_eq24a}
# plag_liq_T_funcs_by_name = {p.__name__: p for p in plag_liq_T_funcs}

# plag_liq_H_funcs = {H_Put2008_eq25b, H_Put2005_eqH, H_Waters2015}
# plag_liq_H_funcs_by_name = {p.__name__: p for p in plag_liq_H_funcs}

def calculate_fspar_liq_temp_hygr(*, liq_comps, plag_comps, equationT, equationH, iterations=20,
                                P=None, kspar_comps=None, eq_tests=True, H2O_estimate=1):

    '''
    Iterates temperature and water content for Plag-liquid pairs for a user-specified number of
    iterations. Returns calculated T and H2O, as well as change in T and H with the # of iterations.

    Parameters
    -------
    liq_comps: pandas.DataFrame
        liquids compositions with column headings SiO2_Liq, MgO_Liq etc.

    plag_comps: pandas.DataFrame (optional)
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc.

    equationH: str
        choose from:

        |  H_Waters2015 (T-dependent, P-dependent)
        |  H_Put2005_eqH (T-dependent, P-independent)
        |  H_Put2008_eq25b (T-dependent, P-dependent)

    equationT: str
        choose from:

        |   T_Put2008_eq23 (Plag-Liq, P-dependent, H2O-dependent)
        |   T_Put2008_eq24a (Plag-Liq, P-dependent, H2O-dependent)
    H2O_estimate: float, int
        Initial estimate of H2O content. Can help convergence.

    P: float, int, pandas.Series
        Pressure (kbar) to perform calculations at

    iterations: int
        number of times to iterate temperature and H2O. Default 20.



    Returns
    -------
        Dictionary with keys 'T_H_calc' and 'T_H_Evolution': Dictionary

        'T_H_calc': pandas.DataFrame
            Calculated H2O, calculated T, plag-liq equilibrium parameters,
            and change in T and H between second last and last iteration

        'T_H_Evolution': pandas.DataFrame
            Pandas dataframes, where rows are number of iterations, and column headings show the T and H2O
            calculated for each sample.


    '''
    if kspar_comps is not None:
        raise ValueError('Sorry, no k-fspar hygrometers implemented in this tool. You must enter plag_comps=')

    if  plag_comps is not None and liq_comps is not None:
        if len(plag_comps)!=len(liq_comps):
            raise ValueError('Liq comps need to be same length as plag comps. IF you want to consider all pairs, use the matching function: calculate_fspar_liq_temp_hygr_matching')



    if eq_tests is False:
        print('eq_tests=False? too bad, we return the equilibrium tests anyway, as you really need to look at them!')
    #Check valid equation for T
    try:
        func = plag_liq_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation for Plag-Liquid') from None
    sig=inspect.signature(func)

    # Check entered valid equation for H
    try:
        func = plag_liq_H_funcs_by_name[equationH]
    except KeyError:
        raise ValueError(f'{equationH} is not a valid equation') from None
    sig=inspect.signature(func)

    if plag_comps is not None:
        plag_comps_c=plag_comps.copy()
        plag_comps_c=plag_comps_c.reset_index(drop=True)
    if liq_comps is not None:
        liq_comps_c=liq_comps.copy()
        liq_comps_c=liq_comps_c.reset_index(drop=True)




    if P is None:
        raise ValueError('Please enter a pressure in kbar (P=...)')

    H2O_iter=np.empty(len(liq_comps), dtype=float)
    T_iter_23_W2015=calculate_fspar_liq_temp(liq_comps=liq_comps_c, plag_comps=plag_comps_c, equationT=equationT,
                               P=P, H2O_Liq=H2O_estimate)
    T_sample=np.empty([len(T_iter_23_W2015), iterations], dtype=float)
    H2O_sample=np.empty([len(T_iter_23_W2015), iterations], dtype=float)
    It_sample=np.empty( iterations)

    for i in range (0, iterations):

        H2O_iter_23_W2015=calculate_fspar_liq_hygr(liq_comps=liq_comps_c, plag_comps=plag_comps_c,
                                            equationH=equationH,
                                               P=P, T=T_iter_23_W2015)
        H2O_sample[:, i]=H2O_iter_23_W2015['H2O_calc']

        T_iter_23_W2015=calculate_fspar_liq_temp(liq_comps=liq_comps_c, plag_comps=plag_comps_c, equationT=equationT,
                               P=P, H2O_Liq=H2O_iter_23_W2015['H2O_calc'])
        T_sample[:, i]=T_iter_23_W2015
        It_sample[i]=i


    Combined_output=H2O_iter_23_W2015.copy()
    Combined_output.insert(0, '# of iterations', iterations)
    Combined_output.insert(1, 'T_K_calc', T_iter_23_W2015)

    # Calculating delta T
    DeltaT=T_sample[:, -1]-T_sample[:, -2]
    DeltaH=H2O_sample[:, -1]-H2O_sample[:, -2]
    Combined_output.insert(2, 'Delta T (last 2 iters)', DeltaT)
    Combined_output.insert(4, 'Delta H (last 2 iters)', DeltaH)

    # Calculating a dataframe showing the evolution of temperature and H2O vs. number of iterations
    Iter=It_sample

    df_t=pd.DataFrame(T_sample.T)
    df_Temp=df_t.add_prefix('Sample_')
    df_Temp2=df_Temp.add_suffix('_T_calc')
    df_Temp2

    print(len(df_Temp2))

    df_h=pd.DataFrame(H2O_sample.T)
    df_H2O=df_h.add_prefix('Sample_')
    df_H2O2=df_H2O.add_suffix('_H_calc')
    df_H2O2

    T_evol=pd.concat([df_Temp2, df_H2O2], axis=1)
    T_evol['Iteration']=Iter

    # for i in range(0, len(liq_comps)):
    #     if i==0:
    #         T_evol=pd.DataFrame(data={'Iteration': Iter, 'Sample_0_T_calc': T_sample[0, :]})
    #         T_evol.insert(i+1, 'Sample_0_H_calc', H2O_sample[0, :])
    #     else:
    #         new_col_name_T=('Sample_'+str(i)+ "_T_calc")
    #         new_col_name_H=('Sample_'+str(i)+ "_H_calc")
    #         T_evol.insert(2*i, new_col_name_T, T_sample[i, :])
    #         T_evol.insert(2*i+1, new_col_name_H, H2O_sample[i, :])
    #


    return {'T_H_calc': Combined_output, 'T_H_Evolution':  T_evol, 'T_sample': T_sample, 'H2O_sample': H2O_sample}



## Equations: Two feldspar thermometers


def T_Put2008_eq27a(P, *, K_Barth, Si_Kspar_cat_frac,
                    Ca_Kspar_cat_frac, An_Kspar, An_Plag, Ab_Plag):
    '''
    Two feldspar thermometer: Equation 27a of Putirka (2008).
    :cite:`putirka2008thermometers`

    SEE±23°C for calibration

    SEE±44°C for test data
    '''

    return (273.15 + 10**4 / (9.8 - 0.0976 * P - 2.46 * np.log(K_Barth) - 14.2 * Si_Kspar_cat_frac +
            423.4 * Ca_Kspar_cat_frac - 2.42 * np.log(An_Kspar) - 11.4 * An_Plag * Ab_Plag))


def T_Put2008_eq27b(P, *, K_Barth, Si_Kspar_cat_frac,
                    Ca_Kspar_cat_frac, An_Kspar, An_Plag, Ab_Plag):
    '''
    Two feldspar thermometer: Equation 27b of Putirka (2008).
    Putirka recomends using this thermometer over 27a, with preference
    being 27b>global>27a
    Global calibration (unlike 27a, which is calibrated on a smaller dataset)
    :cite:`putirka2008thermometers`

    SEE±30°C for calibration
    '''
    return (273.15 + (-442 - 3.72 * P) / (-0.11 + 0.1 * np.log(K_Barth) -
            3.27 * An_Kspar + 0.098 * np.log(An_Kspar) + 0.52 * An_Plag * Ab_Plag))


def T_Put_Global_2Fspar(P, *, K_Barth, Si_Kspar_cat_frac, Ca_Kspar_cat_frac,
Ab_Kspar, Or_Kspar, Na_Plag_cat_frac, An_Kspar, An_Plag, Ab_Plag):
    '''
    Two feldspar thermometer from supporting spreadsheet of Putirka (2008)
    Global calibration
    :cite:`putirka2008thermometers`

    '''
    return (273.15 + 10**4 / (100.638 - 0.0975 * P - 4.9825 * An_Plag
    - 115.03 * Ab_Kspar - 95.745 * Or_Kspar
    + 45.68 * Na_Plag_cat_frac - 2.5182 * np.log(An_Kspar)
    + 3.7522 * np.log(K_Barth) - 15.503 * (Ab_Kspar - 0.3136) * (Or_Kspar - 0.6625)))




## Function for calculating 2 feldspar temperatures
Plag_Kspar_T_funcs = {T_Put2008_eq27a, T_Put2008_eq27b, T_Put_Global_2Fspar}

Plag_Kspar_T_funcs_by_name = {p.__name__: p for p in Plag_Kspar_T_funcs}


def calculate_plag_kspar_temp(*, plag_comps=None, kspar_comps=None, Two_Fspar_Match=None,
equationT=None, P=None, eq_tests=False):
    '''

    Two feldspar thermometer (Kspar-Plag), returns temperature in Kelvin

    Parameters
    ----------

    plag_comps: pandas.DataFrame
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc.

    kspar_comps: pandas.DataFrame
        Kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    EquationT: str
        choose from:

            |   T_Put2008_eq27a (P-dependent, H2O-independent)
            |   T_Put2008_eq27b (P-dependent, H2O-independent)
            |   T_Put_Global_2Fspar (P-dependent, H2O-independent)

    P: float, int, pandas.Series, str
        Pressure in kbar to perform calculations at.
        Only needed for P-sensitive thermometers.
        If P="Solve", returns a partial function,
        else, enter an integer, float, or panda series.

    Returns
    -------
    If eq_tests is False
        pandas.Series: Temperature in Kelvin
    If eq_tests is True
        pandas.DataFrame: Temperature in Kelvin+eq Tests + input compositions

    '''
# These are checks that our inputs are okay
    try:
        func = Plag_Kspar_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if Two_Fspar_Match is None:
        if len(plag_comps)!=len(kspar_comps):
            raise ValueError('Kspar comps need to be same length as plag comps. use a _matching function instead if you want to consider all pairs')
        if isinstance(P, pd.Series):
            if len(P) != len(plag_comps):
                raise ValueError('The panda series entered for pressure isnt the same length '
                'as the dataframe of Plag compositions')




        cat_plag = calculate_cat_fractions_plagioclase(plag_comps=plag_comps)
        cat_kspar = calculate_cat_fractions_kspar(kspar_comps=kspar_comps)
        combo_fspars = pd.concat([cat_plag, cat_kspar], axis=1)
        combo_fspars['K_Barth'] = combo_fspars['Ab_Kspar'] / \
        combo_fspars['Ab_Plag']

    else:
        combo_fspars=Two_Fspar_Match

    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')
    else:
        if P is not None:
            print('Youve selected a P-independent function')


    kwargs = {name: combo_fspars[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:
        T_K=func(P, **kwargs)

    if eq_tests is False:
        return T_K
    if eq_tests is True:
        func=calculate_fspar_activity_components
        combo_fspars['T']=T_K
        combo_fspars['P']=P
        kwargs = {name: combo_fspars[name] for name, p in inspect.signature(func).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        eq_tests=func(**kwargs)

        eq_tests.insert(0, "T_K_calc", T_K)
        eq_tests_final=pd.concat([eq_tests, combo_fspars], axis=1)
        return eq_tests_final


## Plag Kspar matching

def calculate_plag_kspar_temp_matching(*, kspar_comps, plag_comps, equationT=None,
                                 P=None):
    '''
    Evaluates all possible Plag-Kspar pairs,
    returns T (K) and equilibrium tests

    Parameters
    ----------------

    plag_comps: pandas.DataFrame
        Panda DataFrame of plag compositions with column headings SiO2_Plag, CaO_Plag etc.

    kspar_comps: pandas.DataFrame
        Panda DataFrame of kspar compositions with column headings SiO2_Kspar etc.

    EquationT: str
        Choose from:

            |   T_Put2008_eq27a (P-dependent, H2O-independent)
            |   T_Put2008_eq27b (P-dependent, H2O-independent)
            |   T_Put_Global_2Fspar (P-dependent, H2O-independent)

    P: float, int, pandas.Series
        Pressure in kbar to perform calculations at.

    Returns
    -------
    pandas.DataFrame
        T in K for all posible plag-kspar matches, along with equilibrium
        tests, components and input mineral compositions
    '''


    # calculating Plag and plag components. Do before duplication to save
    # computation time
    cat_plag = calculate_cat_fractions_plagioclase(plag_comps=plag_comps)
    cat_kspar = calculate_cat_fractions_kspar(kspar_comps=kspar_comps)

    # Adding an ID label to help with melt-kspar rematching later
    cat_plag['ID_Plag'] = cat_plag.index
    cat_kspar['ID_Kspar'] = cat_kspar.index.astype('float64')
    if "Sample_ID_Plag" not in cat_plag:
        cat_plag['Sample_ID_Plag'] = cat_plag.index.astype('float64')
    if "Sample_ID_Kspar" not in cat_kspar:
        cat_kspar['Sample_ID_Kspar'] = cat_kspar.index.astype('float64')
    # Duplicate kspars and liquids so end up with panda of all possible liq-kspar
    # matches

    # This duplicates Plags, repeats kspar1-kspar1*N, kspar2-kspar2*N etc.
    DupPlags = pd.DataFrame(np.repeat(cat_plag.values, np.shape(
        cat_kspar)[0], axis=0))  # .astype('float64')
    DupPlags.columns = cat_plag.columns

    # This duplicates liquids like liq1-liq2-liq3 for kspar1, liq1-liq2-liq3 for
    # kspar2 etc.
    DupKspars = pd.concat([cat_kspar] * np.shape(cat_plag)[0]).reset_index(drop=True)
    # Combines these merged liquids and kspar dataframes
    Combo_plags_kspars = pd.concat([DupKspars, DupPlags], axis=1)

    Combo_plags_kspars_1 = Combo_plags_kspars.copy()
    Combo_plags_kspars_1['K_Barth'] = Combo_plags_kspars_1['Ab_Kspar'] / \
        Combo_plags_kspars_1['Ab_Plag']
    LenCombo = str(np.shape(Combo_plags_kspars)[0])
    LenPlag=len(plag_comps)
    LenKspar=len(kspar_comps)
    print("Considering N=" + str(LenPlag) + " Plag & N=" + str(LenKspar) +" Kspar, which is a total of N="+ str(LenCombo) +
          " Plag-Kspar pairs, be patient if this is >>1 million!")



    T_K_calc=calculate_plag_kspar_temp(Two_Fspar_Match=Combo_plags_kspars_1.astype(float),
    equationT=equationT, P=P)
    Combo_plags_kspars_1.insert(0, "T_K_calc", T_K_calc)

    func=calculate_fspar_activity_components
    Combo_plags_kspars_1['T']=T_K_calc
    Combo_plags_kspars_1['P']=P
    kwargs = {name: Combo_plags_kspars_1[name].astype('float64') for name, p in inspect.signature(func).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    eq_tests=func(**kwargs)
    out=pd.concat([eq_tests, Combo_plags_kspars_1], axis=1)

    return out# Combo_plags_kspars_1


## Matching functions for fspar-Liq
def calculate_fspar_liq_temp_hygr_matching(liq_comps, plag_comps, equationT, equationH, iterations=20,
                                P=None, kspar_comps=None, Ab_An_P2008=True ):

    '''
    Evaluates all possible Plag-liq pairs, iterates T and H2O
    returns T (K) and equilibrium test values. Users must investigate correct values for eq tests.

    Parameters
    ----------------

    plag_comps: pandas.DataFrame
        Panda DataFrame of plag compositions with column headings SiO2_Plag, CaO_Plag etc.

    liq_comps: pandas.DataFrame
        Panda DataFrame of liq compositions with column headings SiO2_Liq etc.

    EquationT: str
        Choose from:

            |   T_Put2008_eq24b (Kspar-Liq, P-dependent, H2O-independent
            |   T_Put2008_eq23 (Plag-Liq, P-dependent, H2O-dependent)
            |   T_Put2008_eq24a (Plag-Liq, P-dependent, H2O-dependent)


    P: float, int, pandas.Series
        Pressure in kbar to perform calculations at.

    iterations: int
        number of times to iterate temperature and H2O. Default 20.

    Returns
    -------
    dict
        'Av_HTs': df of averaged T and H for each Plag, and all the liquids it matches (+eq tests etc)
        'All_HTs: df of all T and H for all possible Plag-Liq combinations (+eq tests etc)
        'T_H_Evolution': dataframe of T-H evolution against number of iterations.


    '''


    if kspar_comps is not None:
        raise Exception('This function isnt able to use KSpar yet as no hygrometers exist')

    plag_comps_c=plag_comps.copy()
    liq_comps_c=liq_comps.copy()

    #Check valid equation for T
    try:
        func = plag_liq_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation for Plag-Liquid') from None
    sig=inspect.signature(func)

    # Check entered valid equation for H
    try:
        func = plag_liq_H_funcs_by_name[equationH]
    except KeyError:
        raise ValueError(f'{equationH} is not a valid equation') from None
    sig=inspect.signature(func)


    if P is None:
        raise ValueError('Please enter a pressure in kbar (P=...)')
    # calculating Plag and Liq components. Do before duplication to save
    # computation time
    liq_comps_c=liq_comps.copy()

    plag_comps_c['ID_Fspar'] = plag_comps_c.index
    liq_comps_c['ID_liq'] = liq_comps_c.index.astype('float64')

    if "Sample_ID_Plag" not in plag_comps:
        plag_comps_c['Sample_ID_Plag'] = plag_comps.index.astype('float64')
    if "Sample_ID_liq" not in liq_comps:
        liq_comps_c['Sample_ID_liq'] = liq_comps.index.astype('float64')



    # This duplicates Plags, repeats liq1-liq1*N, liq2-liq2*N etc.
    DupFspars = pd.DataFrame(np.repeat(plag_comps_c.values, np.shape(
        liq_comps_c)[0], axis=0))  # .astype('float64')
    DupFspars.columns = plag_comps_c.columns

    # This duplicates liquids like liq1-liq2-liq3 for liq1, liq1-liq2-liq3 for
    # liq2 etc.
    Dupliqs = pd.concat([liq_comps_c] * np.shape(plag_comps_c)[0]).reset_index(drop=True)
    # Combines these merged liquids and liq dataframes
    Combo_fspar_liqs = pd.concat([Dupliqs, DupFspars], axis=1)

    Combo_fspar_liqs_1 = Combo_fspar_liqs.copy()
    # Combo_plags_liqs_1['K_Barth'] = Combo_plags_liqs_1['Ab_liq'] / \
    #     Combo_plags_liqs_1['Ab_Plag']
    LenCombo = str(np.shape(Combo_fspar_liqs)[0])

    LenFspar=len(plag_comps)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenFspar) + " Fspar & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(LenCombo) +
          " Liq- Fspar pairs, be patient if this is >>1 million!")



    H2O_iter=np.empty(len(Dupliqs), dtype=float)
    T_iter_23_W2015=calculate_fspar_liq_temp(liq_comps=Dupliqs, plag_comps=DupFspars, equationT=equationT,
                               P=P, H2O_Liq=0)
    T_sample=np.empty([len(T_iter_23_W2015), iterations], dtype=float)
    H2O_sample=np.empty([len(T_iter_23_W2015), iterations], dtype=float)
    It_sample=np.empty( iterations)

    for i in tqdm(range(0, iterations)):



        H2O_iter_23_W2015=calculate_fspar_liq_hygr(liq_comps=Dupliqs, plag_comps=DupFspars,
                                            equationH=equationH,
                                            P=P, T=T_iter_23_W2015)
        H2O_sample[:, i]=H2O_iter_23_W2015['H2O_calc']

        if i==0:


            T_iter_23_W2015=calculate_fspar_liq_temp(liq_comps=Dupliqs, plag_comps=DupFspars, equationT=equationT,
                                P=P, H2O_Liq=H2O_iter_23_W2015['H2O_calc'], warnAn=True)
        else:
            T_iter_23_W2015=calculate_fspar_liq_temp(liq_comps=Dupliqs, plag_comps=DupFspars, equationT=equationT,
                                P=P, H2O_Liq=H2O_iter_23_W2015['H2O_calc'])


        T_sample[:, i]=T_iter_23_W2015
        It_sample[i]=i

    Combined_output=H2O_iter_23_W2015.copy()
    Combined_output.insert(0, '# of iterations', iterations)
    Combined_output.insert(1, 'T_K_calc', T_iter_23_W2015)

    # Calculating delta T
    DeltaT=T_sample[:, -1]-T_sample[:, -2]
    DeltaH=H2O_sample[:, -1]-H2O_sample[:, -2]
    Combined_output.insert(2, 'Delta T (last 2 iters)', DeltaT)
    Combined_output.insert(4, 'Delta H (last 2 iters)', DeltaH)

    Iter=It_sample

    df_t=pd.DataFrame(T_sample.T)
    df_Temp=df_t.add_prefix('Sample_')
    df_Temp2=df_Temp.add_suffix('_T_calc')
    df_Temp2



    df_h=pd.DataFrame(H2O_sample.T)
    df_H2O=df_h.add_prefix('Sample_')
    df_H2O2=df_H2O.add_suffix('_T_calc')
    df_H2O2

    T_evol=pd.concat([df_Temp2, df_H2O2], axis=1)
    T_evol['Iteration']=Iter
    # Calculating a dataframe showing the evolution of temperature and H2O vs. number of iterations
    # Iter=It_sample
    # for i in range(0, len(Dupliqs)):
    #     if i==0:
    #         T_evol=pd.DataFrame(data={'Iteration': Iter, 'Sample_0_T_calc': T_sample[0, :]})
    #         T_evol.insert(i+1, 'Sample_0_H_calc', H2O_sample[0, :])
    #     else:
    #         new_col_name_T=('Sample_'+str(i)+ "_T_calc")
    #         new_col_name_H=('Sample_'+str(i)+ "_H_calc")
    #         T_evol.insert(2*i, new_col_name_T, T_sample[i, :])
    #         T_evol.insert(2*i+1, new_col_name_H, H2O_sample[i, :])
    #
    #
    #
    #

    if Ab_An_P2008 is True:
        print('Applying filter to only average those that pass the An-Ab eq test of Putirka, 2008')
        Combo_fspar_liqs2=Combined_output.loc[Combined_output['Pass An-Ab Eq Test Put2008?'].str.contains('Yes')].reset_index(drop=True)

    if Ab_An_P2008 is False:
        print('We are returning all pairs, if you want to use the Ab-An equilibrium test of Putirka (2008), enter Ab_An_P2008=True to filter prior to averaging')
        Combo_fspar_liqs2=Combined_output




    FsparNumbers = Combo_fspar_liqs2['ID_Fspar'].unique()

    Combo_fspar_liqs3=Combo_fspar_liqs2.drop(['Pass An-Ab Eq Test Put2008?', 'T_K_calc'], axis=1)
    Combo_fspar_liqs3['T_K_calc']=Combo_fspar_liqs2['T_K_calc'].astype(float)
    if len(FsparNumbers) > 0:
        if plag_comps is not None:
            df1_Mean_nopref=Combo_fspar_liqs3.groupby(['ID_Fspar', 'Sample_ID_Plag'], as_index=False).mean()
            df1_Std_nopref=Combo_fspar_liqs3.groupby(['ID_Fspar', 'Sample_ID_Plag'], as_index=False).std()
            count=Combo_fspar_liqs2.groupby('ID_Fspar',as_index=False).count().iloc[:, 1]
            df1_Mean_nopref['# of Liqs Averaged']=count

            Sample_ID_Fspar_Mean=df1_Mean_nopref['Sample_ID_Plag']
            Sample_ID_Fspar_Std=df1_Std_nopref['Sample_ID_Plag']
            df1_Mean=df1_Mean_nopref.add_prefix('Mean_')
            df1_Std=df1_Std_nopref.add_prefix('Std_')

            df1_Mean.rename(columns={"Mean_ID_Fspar": "ID_Fspar"}, inplace=True)
            df1_Mean.rename(columns={"Mean_# of Liqs Averaged": "# of Liqs Averaged"}, inplace=True)
            df1_Std.rename(columns={"Std_ID_Fspar": "ID_Fspar"}, inplace=True)



            df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_Fspar'])
            df1_M['Sample_ID_Plag']=Sample_ID_Fspar_Mean

    else:
        raise Exception(
            'No Matches - to set less strict filters, change our Kd filter')


    print('Done!!! I found a total of N='+str(len(Combo_fspar_liqs3)) + ' Fspar-Liq matches using the specified filter. N=' + str(len(df1_M)) + ' Fspar out of the N='+str(LenFspar)+' Fspar that you input matched to 1 or more liquids')

    return {'Av_HTs': df1_M, 'All_HTs': Combo_fspar_liqs2, 'T_H_Evolution':  T_evol}
    # Now we do the averaging step for each feldspar crystal








def calculate_fspar_liq_temp_matching(*, liq_comps, plag_comps=None,
kspar_comps=None, H2O_Liq=None, equationT=None,
P=None, Ab_An_P2008=False):
    '''
    Evaluates all possible Plag-liq or kspar-liq pairs,
    returns T (K) and equilibrium test values. Users must investigate correct values for eq tests.

    Parameters
    ----------------

    plag_comps: pandas.DataFrame
        Panda DataFrame of plag compositions with column headings SiO2_Plag, CaO_Plag etc.

    kspar_comps: pandas.DataFrame
        Panda DataFrame of kspar compositions with column headings SiO2_Kspar, CaO_Kspar etc.

    liq_comps: pandas.DataFrame
        Panda DataFrame of liq compositions with column headings SiO2_Liq etc.

    EquationT: str
        Choose from:

            |   T_Put2008_eq24b (Kspar-Liq, P-dependent, H2O-independent
            |   T_Put2008_eq23 (Plag-Liq, P-dependent, H2O-dependent)
            |   T_Put2008_eq24a (Plag-Liq, P-dependent, H2O-dependent)


    P: float, int, pandas.Series
        Pressure in kbar to perform calculations at.


    Returns
    -------
    dict
        'Av_PTs': df of averaged T for each Plag, and all the liquids it matches (+eq tests etc)
        'All_PTs: df of all T for all possible Plag-Liq combinations (+eq tests etc)



    '''


    # calculating Plag and plag components. Do before duplication to save
    # computation time
    liq_comps_c=liq_comps.copy()
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq']=H2O_Liq

    if plag_comps is not None:
        try:
            func = plag_liq_T_funcs_by_name[equationT]
        except KeyError:
            raise ValueError(f'{equationT} is not a valid equation for Plag-Liquid') from None
        sig=inspect.signature(func)

    if plag_comps is not None:
        cat_fspar = calculate_cat_fractions_plagioclase(plag_comps=plag_comps)
        cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
        combo_fspar_Liq = pd.concat([cat_fspar, cat_liqs], axis=1)

        Kd_Ab_An = (combo_fspar_Liq['Ab_Plag'] * combo_fspar_Liq['Al_Liq_cat_frac'] * combo_fspar_Liq['Ca_Liq_cat_frac'] /
                    (combo_fspar_Liq['An_Plag'] * combo_fspar_Liq['Na_Liq_cat_frac'] * combo_fspar_Liq['Si_Liq_cat_frac']))
        cat_fspar['Kd_Ab_An'] = Kd_Ab_An

        if np.min(combo_fspar_Liq['An_Plag'] < 0.05):
            w.warn('Some inputted feldspars have An<0.05, but you have selected a plagioclase-liquid thermometer'
            '. If these are actually alkali felspars, please use T_P2008_eq24b or T_P2008_24c instead', stacklevel=2)

    if kspar_comps is not None:
        try:
            func = Kspar_Liq_T_funcs_by_name[equationT]
        except KeyError:
            raise ValueError(f'{equationT} is not a valid equation for Kspar-Liquid') from None
        sig=inspect.signature(func)

    if kspar_comps is not None:
        cat_fspar = calculate_cat_fractions_kspar(kspar_comps=kspar_comps)
        cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
        combo_fspar_Liq = pd.concat([cat_fspar, cat_liqs], axis=1)

        Kd_Ab_An = (combo_fspar_Liq['Ab_Kspar'] * combo_fspar_Liq['Al_Liq_cat_frac'] * combo_fspar_Liq['Ca_Liq_cat_frac'] /
                    (combo_fspar_Liq['An_Kspar'] * combo_fspar_Liq['Na_Liq_cat_frac'] * combo_fspar_Liq['Si_Liq_cat_frac']))
        cat_fspar['Kd_Ab_An'] = Kd_Ab_An

        if np.min(cat_fspar['An_Kspar'] > 0.05):
            w.warn('Some inputted feldspars have An>0.05, but you have selected a Kspar-liquid thermometer'
            '. If these are actually Plagioclase feldspars, please use T_P2008_eq23 or _eq24a instead', stacklevel=2)


    # Adding an ID label to help with melt-liq rematching later
    cat_fspar['ID_Fspar'] = cat_fspar.index
    cat_liqs['ID_liq'] = cat_liqs.index.astype('float64')
    if "Sample_ID_Plag" not in cat_fspar:
        cat_fspar['Sample_ID_Plag'] = cat_fspar.index.astype('float64')
    if "Sample_ID_liq" not in cat_liqs:
        cat_liqs['Sample_ID_liq'] = cat_liqs.index.astype('float64')
    # Duplicate liqs and liquids so end up with panda of all possible liq-liq
    # matches

    # This duplicates Plags, repeats liq1-liq1*N, liq2-liq2*N etc.
    DupFspars = pd.DataFrame(np.repeat(cat_fspar.values, np.shape(
        cat_liqs)[0], axis=0))  # .astype('float64')
    DupFspars.columns = cat_fspar.columns

    # This duplicates liquids like liq1-liq2-liq3 for liq1, liq1-liq2-liq3 for
    # liq2 etc.
    Dupliqs = pd.concat([cat_liqs] * np.shape(cat_fspar)[0]).reset_index(drop=True)
    # Combines these merged liquids and liq dataframes
    Combo_fspar_liqs = pd.concat([Dupliqs, DupFspars], axis=1)

    Combo_fspar_liqs_1 = Combo_fspar_liqs.copy()
    # Combo_plags_liqs_1['K_Barth'] = Combo_plags_liqs_1['Ab_liq'] / \
    #     Combo_plags_liqs_1['Ab_Plag']
    LenCombo = str(np.shape(Combo_fspar_liqs)[0])

    LenFspar=len(cat_fspar)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenFspar) + " Fspar & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(LenCombo) +
          " Liq-Fspar pairs, be patient if this is >>1 million!")


    if plag_comps is not None:
        T_K_calc=calculate_fspar_liq_temp(meltmatch_plag=Combo_fspar_liqs_1,
        equationT=equationT, P=P)
        Combo_fspar_liqs_1.insert(0, "T_K_calc", T_K_calc)
    if kspar_comps is not None:

        T_K_calc=calculate_fspar_liq_temp(meltmatch_kspar=Combo_fspar_liqs_1,
        equationT=equationT, P=P)
        Combo_fspar_liqs_1.insert(0, "T_K_calc", T_K_calc)


    if kspar_comps is not None:
        print('Sorry, no equilibrium tests implemented for Kspar-Liquid. Weve returned all possible pairs, you will have to filter them yourselves')
        Combo_fspar_liqs.insert(0, "T_K_calc", T_K_calc)
        Combo_fspar_liqs2=Combo_fspar_liqs



    if plag_comps is not None:
        eq_tests=calculate_plag_liq_eq_tests(meltmatch=Combo_fspar_liqs_1,
        P=P, T=T_K_calc)


        cols_to_move = ['T_K_calc']
        eq_testsN = eq_tests[cols_to_move + [
            col for col in eq_tests.columns if col not in cols_to_move]]
        if Ab_An_P2008 is True:
            Combo_fspar_liqs2=eq_testsN.loc[eq_testsN['Pass An-Ab Eq Test Put2008?'].str.contains('Yes')].reset_index(drop=True)
            print('Applying filter to only average those that pass the An-Ab eq test of Putirka, 2008')

        if Ab_An_P2008 is False:
            print('We are returning all pairs, if you want to use the Ab-An equilibrium test of Putirka (2008), enter Ab_An_P2008=True')
            Combo_fspar_liqs2=eq_testsN

        Combo_fspar_liqs3=Combo_fspar_liqs2.drop(['Pass An-Ab Eq Test Put2008?', 'T_K_calc'], axis=1)

    else:
        Combo_fspar_liqs3=Combo_fspar_liqs_1




    FsparNumbers = Combo_fspar_liqs2['ID_Fspar'].unique()

    Sample_ID_Liq=Combo_fspar_liqs2['Sample_ID_Liq']
    Combo_fspar_liqs3.drop(["Sample_ID_Liq"], axis=1, inplace=True)



    Combo_fspar_liqs3['T_K_calc']=Combo_fspar_liqs2['T_K_calc'].astype(float)
    if len(FsparNumbers) > 0:
        if plag_comps is not None:
            df1_Mean_nopref=Combo_fspar_liqs3.groupby(['ID_Fspar', 'Sample_ID_Plag'], as_index=False).mean()
            df1_Std_nopref=Combo_fspar_liqs3.groupby(['ID_Fspar', 'Sample_ID_Plag'], as_index=False).std()
            count=Combo_fspar_liqs2.groupby('ID_Fspar',as_index=False).count().iloc[:, 1]
            df1_Mean_nopref['# of Liqs Averaged']=count

            Sample_ID_Fspar_Mean=df1_Mean_nopref['Sample_ID_Plag']
            Sample_ID_Fspar_Std=df1_Std_nopref['Sample_ID_Plag']
            df1_Mean=df1_Mean_nopref.add_prefix('Mean_')
            df1_Std=df1_Std_nopref.add_prefix('Std_')

            df1_Mean.rename(columns={"Mean_ID_Fspar": "ID_Fspar"}, inplace=True)
            df1_Mean.rename(columns={"Mean_# of Liqs Averaged": "# of Liqs Averaged"}, inplace=True)
            df1_Std.rename(columns={"Std_ID_Fspar": "ID_Fspar"}, inplace=True)



            df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_Fspar'])
            df1_M['Sample_ID_Plag']=Sample_ID_Fspar_Mean



            # cols_to_move = ['Sample_ID_Plag',
            #                 'Mean_T_K_calc', 'Std_T_K_calc']
            #
            # df1_M = df1_M[cols_to_move +
            #             [col for col in df1_M.columns if col not in cols_to_move]]



        if kspar_comps is not None:
            df1_Mean_nopref=Combo_fspar_liqs3.groupby(['ID_Fspar', 'Sample_ID_Kspar'], as_index=False).mean()
            df1_Std_nopref=Combo_fspar_liqs3.groupby(['ID_Fspar', 'Sample_ID_Kspar'], as_index=False).std()
            count=Combo_fspar_liqs2.groupby('ID_Fspar',as_index=False).count().iloc[:, 1]
            df1_Mean_nopref['# of Liqs Averaged']=count

            Sample_ID_Fspar_Mean=df1_Mean_nopref['Sample_ID_Kspar']
            Sample_ID_Fspar_Std=df1_Std_nopref['Sample_ID_Kspar']
            df1_Mean=df1_Mean_nopref.add_prefix('Mean_')
            df1_Std=df1_Std_nopref.add_prefix('Std_')

            df1_Mean.rename(columns={"Mean_ID_Fspar": "ID_Fspar"}, inplace=True)
            df1_Mean.rename(columns={"Mean_# of Liqs Averaged": "# of Liqs Averaged"}, inplace=True)
            df1_Std.rename(columns={"Std_ID_Fspar": "ID_Fspar"}, inplace=True)



            df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_Fspar'])
            df1_M['Sample_ID_Kspar']=Sample_ID_Fspar_Mean


    else:
        raise Exception(
            'No Matches - to set less strict filters, change our Kd filter')


    print('Done!!! I found a total of N='+str(len(Combo_fspar_liqs3)) + ' Fspar-Liq matches using the specified filter. N=' + str(len(df1_M)) + ' Fspar out of the N='+str(LenFspar)+' Fspar that you input matched to 1 or more liquids')

    Combo_fspar_liqs2['Sample_ID_Liq']=Sample_ID_Liq

    return {'Av_PTs': df1_M, 'All_PTs': Combo_fspar_liqs2}
    # Now we do the averaging step for each feldspar crystal




