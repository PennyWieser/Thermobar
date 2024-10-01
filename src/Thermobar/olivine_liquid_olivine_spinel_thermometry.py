import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd


from Thermobar.core import *
from Thermobar.mineral_equilibrium import *

## Functions for olivine-liquid hygrometry
def H_Gavr2016(*, CaO_Liq, MgO_Liq, CaO_Ol):
    '''
    Olivine-Liquid hygrometer of Gavrilenko et al. (2016).
    '''
    DCa_dry_HighMgO=0.00042*MgO_Liq+0.0196
    DCa_dry_LowMgO=-0.0043*MgO_Liq+0.072
    DCa_dry_calc=np.empty(len(MgO_Liq), dtype=float)
    DeltaCa=np.empty(len(MgO_Liq), dtype=float)

    H2O_Calc=np.empty(len(MgO_Liq), dtype=float)

    DCa_Meas=CaO_Ol/CaO_Liq
    DCa_Divider=0.00462*MgO_Liq-0.027
    for i in range(0, len(MgO_Liq)):
        if DCa_Meas[i]<=DCa_Divider[i]:
            DCa_dry_calc[i]=DCa_dry_HighMgO[i]
            DeltaCa[i]=DCa_dry_calc[i]-DCa_Meas[i]
            H2O_Calc[i]=397*DeltaCa[i]
        else:
            DCa_dry_calc[i]=DCa_dry_LowMgO[i]
            DeltaCa[i]=DCa_dry_calc[i]-DCa_Meas[i]
            H2O_Calc[i]=188*DeltaCa[i]
    H2O_Calc[CaO_Ol==0]=np.nan
    return H2O_Calc

Liquid_olivine_hygr_funcs = {H_Gavr2016}
Liquid_olivine_hygr_funcs_by_name = {p.__name__: p for p in Liquid_olivine_hygr_funcs}


def calculate_ol_liq_hygr(*, liq_comps=None, ol_comps=None, equationH=None, eq_tests=False,
P=None, T=None, meltmatch=None, equationT=None, Fe3Fet_Liq=None):
    '''
    Olivine-liquid hygrometer. Returns the estimated H2O content
    in wt%

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    ol_comps: pandas.DataFrame
        Olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    equationH: str
        H_Gavr2016 (P-independent, H2O_independent)

    eq_tests: bool
        if true, calculates Kd for olivine-liquid pairs.
        Other inputs for these tests:

        P: int, flt, pandas.Series (needed for Toplis Kd calculation).
        If nothing inputted, P set to 1 kbar

        T: int, flt, pandas.Series (needed for Toplis KD calculation).
        Can also specify equationT="" to calculate temperature
        using an olivine-liquid thermometer, using the calculated H2O content
        from the hygrometer

        Fe3Fet_Liq: int, flt, pandas.Series. As Kd calculated using only Fe2 in the Liq.


    Returns
    -------
    pandas.core.series.Series
        H2O content in wt%.
 '''
# These are checks that our inputs are okay
    if meltmatch is None:
        if len(liq_comps)!=len(ol_comps):
            raise ValueError('Ol comps need to be same length as Liq comps. use a _matching function calculate_ol_liq_hygr_matching instead if you want to consider all pairs')



    try:
        func = Liquid_olivine_hygr_funcs_by_name[equationH]
    except KeyError:
        raise ValueError(f'{equationH} is not a valid equation') from None
    sig=inspect.signature(func)

    if meltmatch is None:

        ol_comps_c=ol_comps.copy()
        liq_comps_c=liq_comps.copy()

        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq

        if len(liq_comps) != len(ol_comps):
            raise ValueError('The panda series entered for olivine isnt the same length as for liquids')

        anhyd_cat_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
        ol_cat_frac = calculate_cat_fractions_olivine(ol_comps=ol_comps_c)
        Liq_Ols = pd.concat([ol_comps_c, anhyd_cat_frac, ol_cat_frac], axis=1)

    if meltmatch is not None:
        Liq_Ols=meltmatch



    kwargs = {name: Liq_Ols[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    H2O_Calc_np=func(**kwargs)
    H2O_Calc=pd.Series(H2O_Calc_np)

    if eq_tests is False and meltmatch is None:
         return H2O_Calc

    if eq_tests is False and meltmatch is not None:
        if 'H2O_calc' in Liq_Ols.columns:
            print("Column already exists in dataframe. Have ovewritten")
            Liq_Ols['H2O_calc']=H2O_Calc

        else:
            Liq_Ols.insert(0, 'H2O_calc', H2O_Calc)

    if eq_tests is True:
        if P is None:
            P = 1
            print(
                'You have not selected a pressure, so we have calculated Toplis Kd at 1kbar')
        if T is None and equationT is None:
            raise ValueError('Temperature needed to calculate Delta Kd using Toplis.'
             'Either enter T, or specify equationT=""')

        if equationT is not None:
            # If not melt match, do separatly
            #if meltmatch is None:

            T=calculate_ol_liq_temp(meltmatch=Liq_Ols,
            equationT=equationT, H2O_Liq=H2O_Calc, P=P, eq_tests=eq_tests).T_K_calc
            # If melt match do on combined dataframe
            # if meltmatch is not None:
            #     T=calculate_ol_liq_temp(ol_comps=ol_comps, liq_comps=liq_comps,
            # equationT=equationT, H2O_Liq=H2O_Calc, P=P).T_K_calc

        ol_fo = (Liq_Ols['MgO_Ol'] / 40.3044) / \
            ((Liq_Ols['MgO_Ol'] / 40.3044) + Liq_Ols['FeOt_Ol'] / 71.844)
        KdFeMg_Meas = (
            ((Liq_Ols['FeOt_Ol'] / 71.844) / (Liq_Ols['MgO_Ol'] / 40.3044)) /
            ((Liq_Ols['FeOt_Liq'] * (1 - Liq_Ols['Fe3Fet_Liq']
                                         ) / 71.844) / (Liq_Ols['MgO_Liq'] / 40.3044))
        )
        Kd_func = partial(calculate_toplis2005_kd, SiO2_mol=Liq_Ols['SiO2_Liq_mol_frac'],
        Na2O_mol=Liq_Ols['Na2O_Liq_mol_frac'], K2O_mol=Liq_Ols['Na2O_Liq_mol_frac'],
        P=P, H2O=Liq_Ols['H2O_Liq'], T=T)
        Kd_Toplis_calc = Kd_func(ol_fo)

        DeltaKd_Toplis = KdFeMg_Meas - Kd_Toplis_calc
        DeltaKd_Roeder = KdFeMg_Meas - 0.3
        DeltaKd_Matzen = KdFeMg_Meas - 0.34
        if equationT is None:
            df = pd.DataFrame(data={'H2O_calc': H2O_Calc, 'Temp used for calcs': T, 'P used for calcs': P, 'Kd Meas': KdFeMg_Meas, 'Kd calc (Toplis)': Kd_Toplis_calc,'ΔKd, Toplis (M-P)': DeltaKd_Toplis, 'ΔKd, Roeder (M-P)': DeltaKd_Roeder, 'ΔKd, Matzen (M-P)': DeltaKd_Matzen})
        if equationT is not None:
            df = pd.DataFrame(data={'H2O_calc': H2O_Calc, 'T_K_calc': T, 'P used for calcs': P, 'Kd Meas': KdFeMg_Meas, 'Kd calc (Toplis)': Kd_Toplis_calc,'ΔKd, Toplis (M-P)': DeltaKd_Toplis, 'ΔKd, Roeder (M-P)': DeltaKd_Roeder, 'ΔKd, Matzen (M-P)': DeltaKd_Matzen})


        df_out = pd.concat([df, Liq_Ols], axis=1)

        return df_out

def calculate_ol_liq_hygr_matching(*, liq_comps, ol_comps, equationH, eq_tests=False,
T=None, equationT=None, P=None,  Fe3Fet_Liq=None, iterations=30):

    '''
    Evaluates all possible Ol-liq pairs for H2O
    returns H2O and equilibrium test values.

    Parameters
    ----------------

    ol_comps: pandas.DataFrame
        Panda DataFrame of Ol compositions with column headings SiO2_Ol, CaO_Ol etc.

    liq_comps: pandas.DataFrame
        Panda DataFrame of liq compositions with column headings SiO2_Liq etc.

    EquationH: str
        Choose from:

            |   H_Gavr2016 (P-independent, T-independent)




    iterations: int
        number of times to iterate temperature and H2O. Default 30.

    Returns
    -------
    pandas.DataFrame
        H2O (wt%) for all posible ol-liq matches, along with equilibrium
        tests, components and input mineral compositions
    '''

    ol_comps_c=ol_comps.copy()
    liq_comps_c=liq_comps.copy()

    #Check valid equation for T
    try:
        func = Liquid_olivine_hygr_funcs_by_name[equationH]
    except KeyError:
        raise ValueError(f'{equationH} is not a valid equation for Ol-Liquid Hygrometry') from None
    sig=inspect.signature(func)

    # This is a duplication step, to pair up all possible mathes. First we give a unique identifier

    ol_comps_c['ID_Ol'] = ol_comps_c.index
    liq_comps_c['ID_liq'] = liq_comps_c.index.astype('float64')

    if "Sample_ID_Ol" not in ol_comps:
        ol_comps_c['Sample_ID_Ol'] = ol_comps.index.astype('float64')
    if "Sample_ID_liq" not in liq_comps:
        liq_comps_c['Sample_ID_liq'] = liq_comps.index.astype('float64')


    anhyd_liq_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
    ol_cat_frac1 = calculate_cat_fractions_olivine(ol_comps=ol_comps_c)
    ol_cat_frac= pd.concat([ol_comps_c, ol_cat_frac1], axis=1)


    # This duplicates Ols, repeats liq1-liq1*N, liq2-liq2*N etc.
    DupOls = pd.DataFrame(np.repeat(ol_cat_frac.values, np.shape(
        anhyd_liq_frac)[0], axis=0))  # .astype('float64')
    DupOls.columns = ol_cat_frac.columns

    # This duplicates liquids like liq1-liq2-liq3 for liq1, liq1-liq2-liq3 for
    # liq2 etc.
    Dupliqs = pd.concat([anhyd_liq_frac ] * np.shape(ol_cat_frac)[0]).reset_index(drop=True)
    # Combines these merged liquids and liq dataframes
    Combo_Ol_liqs = pd.concat([Dupliqs, DupOls], axis=1)

    Combo_Ol_liqs_1 = Combo_Ol_liqs.copy()
    # Combo_Ols_liqs_1['K_Barth'] = Combo_Ols_liqs_1['Ab_liq'] / \
    #     Combo_Ols_liqs_1['Ab_Ol']
    LenCombo = str(np.shape(Combo_Ol_liqs)[0])

    LenOl=len(ol_comps)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenOl) + " Ol & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(LenCombo) +
          " Liq- Ol pairs, be patient if this is >>1 million!")



    if equationT is None:
        CalcH2O=calculate_ol_liq_hygr(meltmatch=Combo_Ol_liqs,
    equationH=equationH, T=T, P=P, eq_tests=eq_tests,
    Fe3Fet_Liq=Fe3Fet_Liq)

    if equationT is not None:
        CalcH2O=calculate_ol_liq_hygr(meltmatch=Combo_Ol_liqs,
    equationH=equationH, equationT=equationT, P=P, eq_tests=eq_tests,
    Fe3Fet_Liq=Fe3Fet_Liq)




    return CalcH2O





## Functions for olivine-Liquid thermometry
# Defining functions where DMg is measured.
def T_Beatt93_ol(P, *, Den_Beat93):
    '''
    Olivine-Liquid thermometer: Beattie (1993). Putirka (2008) suggest this is the best olivine-liquid thermometer for anhydrous conditions at relatively low pressures.
    '''
    return (((113.1 * 1000) / 8.3144 + (0.1 * P * 10**9 - 10**5)
            * 4.11 * (10**(-6)) / 8.3144) / Den_Beat93)
# This is equation 19 from Putirka 2008


def T_Beatt93_ol_HerzCorr(P, *, Den_Beat93):
    '''
    Olivine-Liquid thermometer: Beattie (1993) with correction of Herzberg and O'Hara (2002), eliminating systematic error at higher pressures
    Anhydrous SEE=±44°C
    Hydrous SEE=±53°C
    '''

    return (((113.1 * 1000 / 8.3144 + (0.0001 * (10**9) - 10**5) * 4.11 *
            (10**(-6)) / 8.3144) / Den_Beat93) + 54 * (0.1 * P) + 2 * (0.1 * P)**2)


def T_Put2008_eq19(P, *, DMg_Meas, CNML, CSiO2L, NF):
    '''
    Olivine-Liquid thermometer originally from Beattie, (1993), form from Putirka (2008)

    '''
    return ((13603) + (4.943 * 10**(-7)) * ((0.1 * P)*10**9 - 10**(-5))) / (6.26 + 2 *
            np.log(DMg_Meas) + 2 * np.log(1.5 * CNML) + 2 * np.log(3 * CSiO2L) - NF)



def T_Put2008_eq21(P, *, DMg_Meas, Na2O_Liq, K2O_Liq, H2O_Liq):
    '''
    Olivine-Liquid thermometer: Putirka (2008), equation 21 (originally Putirka et al., 2007,  Eq 2). Recalibration of Beattie (1993) to account for the pressures sensitivity noted by Herzberg and O'Hara (2002), and esliminates the systematic error of Beattie (1993) for hydrous compositions.
    Anhydrous SEE=±53°C
    Hydrous SEE=±36°C
    '''
    return (1 / ((np.log(DMg_Meas) + 2.158 - 5.115 * 10**(-2) * (Na2O_Liq +
            K2O_Liq) + 6.213 * 10**(-2) * H2O_Liq) / (55.09 * 0.1 * P + 4430)) + 273.15)


def T_Put2008_eq22(P, *, DMg_Meas, CNML, CSiO2L, NF, H2O_Liq):
    '''
    Olivine-Liquid thermometer: Putirka (2008), equation 22 (originally Putirka et al., 2007,  Eq 4). Recalibration of Beattie (1993) to account for the pressures sensitivity noted by Herzberg and O'Hara (2002), and esliminates the systematic error of Beattie (1993) for hydrous compositions. Putirka (2008) suggest this is the best olivine-liquid thermometer for hydrous melts.
    Anhydrous SEE±45°C
    Hydrous SEE±29°C
    '''
    return ((15294.6 + 1318.8 * 0.1 * P + 2.48348 * ((0.1 * P)**2)) / (8.048 + 2.8352 * np.log(DMg_Meas) + 2.097 *
            np.log(1.5 * CNML) + 2.575 * np.log(3 * CSiO2L) - 1.41 * NF + 0.222 * H2O_Liq + 0.5 * (0.1 * P)) + 273.15)


def T_Sisson1992(P, *, KdMg_TSG1992):
    '''
    Olivine-Liquid thermometer: Sisson and Grove (1992). Putirka (2008) suggests this thermometer is best in peridotitic systems containing 2-25 wt% CO2.
    '''
    return ((4129 + 0.0146 * (P*1000-1)) /
            (np.log10(KdMg_TSG1992) + 2.082))


def T_Pu2017(P=None, *, NiO_Ol_mol_frac, FeOt_Liq_mol_frac, MnO_Liq_mol_frac, MgO_Liq_mol_frac,
             CaO_Liq_mol_frac, NiO_Liq_mol_frac, Al2O3_Liq_mol_frac, TiO2_Liq_mol_frac, SiO2_Liq_mol_frac):
    '''
    Olivine-Liquid thermometer: Pu et al. (2017), Eq 1  Uses D Ni (ol-melt) rather than D Mg (ol-melt),
    meaning this thermometer has far less sensitivity to H2O or pressure at 0-1 GPa.
    SEE=±29°C
    '''
    # Check it has Ni
    NiO_Ol_mol_frac = np.asarray(NiO_Ol_mol_frac, dtype=float)
    NiO_Liq_mol_frac = np.asarray(NiO_Liq_mol_frac, dtype=float)

    hasNi=(NiO_Ol_mol_frac>0) & (NiO_Liq_mol_frac>0)

    D_Ni_Mol = np.divide(NiO_Ol_mol_frac, NiO_Liq_mol_frac, out=np.full_like(NiO_Ol_mol_frac, np.nan), where=hasNi)
    XNm = FeOt_Liq_mol_frac + MnO_Liq_mol_frac + \
        MgO_Liq_mol_frac + CaO_Liq_mol_frac + NiO_Liq_mol_frac
    NFX = 3.5 * np.log(1 - Al2O3_Liq_mol_frac) + 7 * \
        np.log(1 - TiO2_Liq_mol_frac)


 # Calculate the log terms only for valid entries
    valid_indices = (D_Ni_Mol > 0) & (XNm > 0) & (SiO2_Liq_mol_frac > 0)

    # Calculate the temperature for valid values

    temperature = np.where(valid_indices,
                       9416 / (np.log(D_Ni_Mol.astype(float)) + 0.71 * np.log(XNm.astype(float)) -
                                0.349 * NFX - 0.532 * np.log(SiO2_Liq_mol_frac.astype(float)) + 4.319),
                       np.nan)



    return temperature


def T_Pu2021(P, *, NiO_Ol_mol_frac, FeOt_Liq_mol_frac, MnO_Liq_mol_frac, MgO_Liq_mol_frac,
             CaO_Liq_mol_frac, NiO_Liq_mol_frac, Al2O3_Liq_mol_frac, TiO2_Liq_mol_frac, SiO2_Liq_mol_frac):
    '''
    Olivine-Liquid thermometer: Pu et al. (2017), with the pressure correction of Pu et al. (2021).
    Uses D Ni (ol-melt) rather than D Mg (ol-melt), meaning this thermometer has far less sensitivity to melt H2O than other olivine-liquid thermometers.
    SEE=±45°C (for the 2017 expression).


    '''

    # Get P to the right format

    if isinstance(P, (float, int)):
            P_values = pd.Series([P] * len(NiO_Ol_mol_frac))
    elif isinstance(P, pd.Series):
        P_values = P
    else:
        raise ValueError("P must be either a float, integer, or a pandas Series.")

    NiO_Ol_mol_frac = np.asarray(NiO_Ol_mol_frac, dtype=float)
    NiO_Liq_mol_frac = np.asarray(NiO_Liq_mol_frac, dtype=float)

    hasNi=(NiO_Ol_mol_frac>0) & (NiO_Liq_mol_frac>0)

    D_Ni_Mol = np.divide(NiO_Ol_mol_frac, NiO_Liq_mol_frac, out=np.full_like(NiO_Ol_mol_frac, np.nan), where=hasNi)

    XNm = FeOt_Liq_mol_frac + MnO_Liq_mol_frac + \
        MgO_Liq_mol_frac + CaO_Liq_mol_frac + NiO_Liq_mol_frac
    NFX = 3.5 * np.log(1 - Al2O3_Liq_mol_frac) + 7 * \
        np.log(1 - TiO2_Liq_mol_frac)
# Initialize P_Corr to zero for all rows
    P_Corr = pd.Series(np.zeros(len(P_values)), index=P_values.index)

    # Calculate P_Corr only where P is between 10 and 30 GPa
    mask = (P_values >= 10) & (P_values <= 30)
    P_Corr[mask] = - 70 + 110 * (P_values[mask] * 0.1) - 18 * (P_values[mask] * 0.1)**2



 # Calculate the log terms only for valid entries
    valid_indices = (D_Ni_Mol > 0) & (XNm > 0) & (SiO2_Liq_mol_frac > 0)

    # Calculate the temperature for valid values
    temperature = np.where(valid_indices,
                       9416 / (np.log(D_Ni_Mol.astype(float)) + 0.71 * np.log(XNm.astype(float)) - 0.349 * NFX - 0.532 * np.log(SiO2_Liq_mol_frac.astype(float)) + 4.319),
                       np.nan)

    return temperature




## Listing all equation options
Liquid_olivine_funcs = {T_Beatt93_ol, T_Beatt93_ol_HerzCorr, T_Put2008_eq19, T_Put2008_eq21,
T_Put2008_eq22, T_Sisson1992, T_Pu2017, T_Pu2021}

Liquid_olivine_funcs_by_name = {p.__name__: p for p in Liquid_olivine_funcs}


## Function for calculating olivine-liquid temperature using various equations.

def calculate_ol_liq_temp_matching(*, liq_comps, ol_comps, eq_tests=False,
equationT=None, P=None, H2O_Liq=None, Fe3Fet_Liq=None, iterations=30):

    '''
    Evaluates all possible Ol-liq pairs for temperature, and calculates
    equilibrium tests

    Parameters
    ----------------

    ol_comps: pandas.DataFrame
        Panda DataFrame of Ol compositions with column headings SiO2_Ol, CaO_Ol etc.

    liq_comps: pandas.DataFrame
        Panda DataFrame of liq compositions with column headings SiO2_Liq etc.

    equationT: str
        T_Beatt93_ol (P-dependent, H2O_independent)
        T_Beatt93_ol_HerzCorr (P-dependent, H2O_independent)
        T_Put2008_eq21 (P-dependent, H2O-dependent)
        T_Put2008_eq22 (P-dependent, H2O-dependent)
        T_Sisson1992 (P-dependent, H2O_independent)
        T_Pu2017 (P-independent, H2O_independent)
        T_Pu2021 (P-dependent, H2O_independent)

    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    H2O_Liq: optional.
        If None, uses H2O_Liq column from input.
        If int, float, pandas.Series, uses this instead of H2O_Liq Column


    iterations: int
        number of times to iterate temperature and H2O. Default 30.

    Returns
    -------
    pandas.DataFrame
        Temp (K) for all posible ol-liq matches, along with equilibrium
        tests, components and input mineral compositions. At the moment, doesnt average per olivine.. But could upon request.
    '''

    ol_comps_c=ol_comps.copy()
    liq_comps_c=liq_comps.copy()

    #Check valid equation for T
    try:
        func = Liquid_olivine_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation for Ol-Liquid Hygrometry') from None
    sig=inspect.signature(func)

    # This is a duplication step, to pair up all possible mathes. First we give a unique identifier

    ol_comps_c['ID_Ol'] = ol_comps_c.index
    liq_comps_c['ID_liq'] = liq_comps_c.index.astype('float64')

    if "Sample_ID_Ol" not in ol_comps:
        ol_comps_c['Sample_ID_Ol'] = ol_comps.index.astype('float64')
    if "Sample_ID_liq" not in liq_comps:
        liq_comps_c['Sample_ID_liq'] = liq_comps.index.astype('float64')

    # Uses mole fractions, not cation fractions
    if equationT == "T_Pu2017" or equationT == "T_Pu2021":
        # For liquid
        anhyd_mol_frac_Ni = calculate_anhydrous_mol_fractions_liquid_Ni(
                liq_comps=liq_comps_c)
        # For olivine
        ol_mol_frac_Ni = calculate_mol_fractions_olivine_ni(
                    ol_comps=ol_comps_c)
        # So names stay the same
        ol_cat_frac=pd.concat([ol_mol_frac_Ni, ol_comps_c], axis=1)
        anhyd_liq_frac =pd.concat([anhyd_mol_frac_Ni, liq_comps_c], axis=1)



    else:

        anhyd_liq_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
        ol_cat_frac1 = calculate_cat_fractions_olivine(ol_comps=ol_comps_c)
        ol_cat_frac= pd.concat([ol_comps_c, ol_cat_frac1], axis=1)


    # This duplicates Ols, repeats liq1-liq1*N, liq2-liq2*N etc.
    DupOls = pd.DataFrame(np.repeat(ol_cat_frac.values, np.shape(
        anhyd_liq_frac)[0], axis=0))  # .astype('float64')
    DupOls.columns = ol_cat_frac.columns

    # This duplicates liquids like liq1-liq2-liq3 for liq1, liq1-liq2-liq3 for
    # liq2 etc.
    Dupliqs = pd.concat([anhyd_liq_frac ] * np.shape(ol_cat_frac)[0]).reset_index(drop=True)
    # Combines these merged liquids and liq dataframes
    Combo_Ol_liqs = pd.concat([Dupliqs, DupOls], axis=1)

    Combo_Ol_liqs_1 = Combo_Ol_liqs.copy()
    # Combo_Ols_liqs_1['K_Barth'] = Combo_Ols_liqs_1['Ab_liq'] / \
    #     Combo_Ols_liqs_1['Ab_Ol']
    LenCombo = str(np.shape(Combo_Ol_liqs)[0])

    LenOl=len(ol_comps)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenOl) + " Ol & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(LenCombo) +
          " Liq-Ol pairs, be patient if this is >>1 million!")

    CalcT=calculate_ol_liq_temp(equationT=equationT, meltmatch=Combo_Ol_liqs, Fe3Fet_Liq=Fe3Fet_Liq,
    H2O_Liq=H2O_Liq, eq_tests=eq_tests, P=P)


    Fo=calculate_ol_fo(ol_comps=DupOls)

    CalcT.insert(1, 'Ol_Fo_Meas', Fo, )

    df_out=pd.concat([CalcT, Combo_Ol_liqs_1], axis=1)

    return df_out





def calculate_ol_liq_temp(*, equationT, liq_comps=None, ol_comps=None, meltmatch=None, P=None,
                          NiO_Ol_Mol=None, H2O_Liq=None, Fe3Fet_Liq=None, eq_tests=False):
    '''
    Olivine-liquid thermometers. Returns the temperature in Kelvin,
    along with calculations of Kd-Fe-Mg equilibrium tests.

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    ol_comps: pandas.DataFrame
        Olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    equationT: str
        T_Beatt93_ol (P-dependent, H2O_independent)
        T_Beatt93_ol_HerzCorr (P-dependent, H2O_independent)
        T_Put2008_eq21 (P-dependent, H2O-dependent)
        T_Put2008_eq22 (P-dependent, H2O-dependent)
        T_Sisson1992 (P-dependent, H2O_independent)
        T_Pu2017 (P-independent, H2O_independent) Eq 1
        T_Pu2021 (P-dependent, H2O_independent) - only applies P corr for 10-30 GPa

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
    pandas.core.series.Series
        Temperatures in kelvin.


    '''
# These are checks that our inputs are okay
    if NiO_Ol_Mol is not None:
        raise TypeError('Sorry, this functionality was lost in the latest diadfit version to allow for melt matching using Pu. Please enter oliivne and liquid compositions instead')


    try:
        func = Liquid_olivine_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if meltmatch is not None:
            Liq_Ols=meltmatch


    else:
        if len(liq_comps)!=len(ol_comps):
            raise ValueError('Ol comps need to be same length as Liq comps. use a _matching function calculate_ol_liq_temp_matching instead if you want to consider all pairs')


        ol_comps_c=ol_comps.copy()
        liq_comps_c=liq_comps.copy()

        if isinstance(P, pd.Series):
            if len(P) != len(liq_comps):
                raise ValueError('The panda series entered for pressure isnt the same length '
                'as the dataframe of liquid compositions')
            if len(liq_comps) != len(ol_comps):
                raise ValueError('The panda series entered for olivine isnt the same length as for liquids')
        liq_comps_c = liq_comps.copy()
        ol_comps_c=ol_comps.copy()


        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq

# Allows different calculation scheme for Ni-bearing equations



        else:
        # Keiths spreadsheets dont use Cr2O3 and P2O5. So have set this to zero.
            liq_comps_c['Cr2O3_Liq']=0
            liq_comps_c['P2O5_Liq']=0
            ol_comps_c['Cr2O3_Ol']=0
            ol_comps_c['P2O5_Ol']=0
    # Now calculate cation fractions


        if equationT == "T_Pu2017" or equationT == "T_Pu2021":
            anhyd_mol_frac_Ni = calculate_anhydrous_mol_fractions_liquid_Ni(
                liq_comps=liq_comps_c)
            # Calculate the liquid mole fraction of NiO
            ol_mol_frac_Ni = calculate_mol_fractions_olivine_ni(
                    ol_comps=ol_comps_c)

            Liq_Ols = pd.concat([anhyd_mol_frac_Ni, ol_mol_frac_Ni, ol_comps_c, liq_comps_c], axis=1)

        else:


            anhyd_cat_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
            ol_cat_frac = calculate_cat_fractions_olivine(ol_comps=ol_comps_c)
            Liq_Ols = pd.concat([anhyd_cat_frac, ol_cat_frac, ol_comps_c], axis=1)




    # This performs extra calculation steps for Beattie equations
    if equationT == "T_Put2008_eq22" or equationT == "T_Put2008_eq21" or \
    equationT == "T_Beatt93_ol" or equationT == "T_Beatt93_ol_HerzCorr" or equationT=="T_Put2008_eq19":



        Liq_Ols['DMg_Meas'] = Liq_Ols['Mg_Ol_cat_frac'].astype(float) /Liq_Ols['Mg_Liq_cat_frac'].astype(float)
        Liq_Ols['CNML'] = (Liq_Ols['Mg_Liq_cat_frac'] + Liq_Ols['Fet_Liq_cat_frac'] +
                                Liq_Ols['Ca_Liq_cat_frac'] + Liq_Ols['Mn_Liq_cat_frac'])
        Liq_Ols['CSiO2L'] = Liq_Ols['Si_Liq_cat_frac']
        Liq_Ols['NF'] = (7 / 2) * np.log(1 - Liq_Ols['Al_Liq_cat_frac']
                                            ) + 7 * np.log(1 - Liq_Ols['Ti_Liq_cat_frac'])
        Liq_Ols['Den_Beat93'] = 52.05 / 8.3144 + 2 * np.log(Liq_Ols['DMg_Meas']) + 2 * np.log(
            1.5 * Liq_Ols['CNML']) + 2 * np.log(3 * Liq_Ols['CSiO2L']) - Liq_Ols['NF']

    if equationT == "T_Sisson1992":
        Liq_Ols['KdMg_TSG1992'] = (Liq_Ols['Mg_Ol_cat_frac'] /
            (Liq_Ols['Mg_Liq_cat_frac'] *
                (Liq_Ols['Si_Liq_cat_frac']**(0.5))))


# Checks if P-dependent function you have entered a P
    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P')
    else:
        if P is not None:
            print('Youve selected a P-independent function')


    kwargs = {name: Liq_Ols[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:

        T_K=func(P, **kwargs)

    if eq_tests is False:

        KdFeMg_Meas = (
            ((Liq_Ols['FeOt_Ol'] / 71.844) / (Liq_Ols['MgO_Ol'] / 40.3044)) /
            ((Liq_Ols['FeOt_Liq'] * (1 - Liq_Ols['Fe3Fet_Liq']
                                            ) / 71.844) / (Liq_Ols['MgO_Liq'] / 40.3044))
        )
        df = pd.DataFrame(
            data={'T_K_calc': T_K, 'Kd (Fe-Mg) Meas': KdFeMg_Meas})
        return df

    if eq_tests is True:
        if NiO_Ol_Mol is not None:
            raise Exception(
                'No olivine composition, so cannot calculate equilibrium test. Set eq_tests=False')
        if P is None:
            P = 1
            print(
                'You have not selected a pressure, so we have calculated Toplis Kd at 1kbar')
        ol_fo = (Liq_Ols['MgO_Ol'] / 40.3044) / \
            ((Liq_Ols['MgO_Ol'] / 40.3044) + Liq_Ols['FeOt_Ol'] / 71.844)
        KdFeMg_Meas = (
            ((Liq_Ols['FeOt_Ol'] / 71.844) / (Liq_Ols['MgO_Ol'] / 40.3044)) /
            ((Liq_Ols['FeOt_Liq'] * (1 - Liq_Ols['Fe3Fet_Liq']
                                         ) / 71.844) / (Liq_Ols['MgO_Liq'] / 40.3044))
        )
        Kd_func = partial(calculate_toplis2005_kd, SiO2_mol=Liq_Ols['SiO2_Liq_mol_frac'], Na2O_mol=Liq_Ols[
                          'Na2O_Liq_mol_frac'], K2O_mol=Liq_Ols['Na2O_Liq_mol_frac'], P=P, H2O=Liq_Ols['H2O_Liq'], T=T_K)
        Kd_Toplis_calc = Kd_func(ol_fo)

        DeltaKd_Toplis = KdFeMg_Meas - Kd_Toplis_calc
        DeltaKd_Roeder = KdFeMg_Meas - 0.3
        DeltaKd_Matzen = KdFeMg_Meas - 0.34
        DeltaKd_Shea= KdFeMg_Meas-0.335
        df = pd.DataFrame(data={'T_K_calc': T_K, 'Kd Meas': KdFeMg_Meas, 'Kd calc (Toplis)': Kd_Toplis_calc,
                                'ΔKd, Toplis (M-P)': DeltaKd_Toplis, 'ΔKd, Roeder (M-P)': DeltaKd_Roeder, 'ΔKd, Matzen (M-P)': DeltaKd_Matzen, 'ΔKd, Shea (M-P)': DeltaKd_Shea})
        df_out = pd.concat([df, Liq_Ols], axis=1)

        return df_out

## Functions for olivine-spinel thermometry equations

def T_Coogan2014(P=None, *, Cr_No_sp, Al2O3_Ol, Al2O3_Sp):
    '''
    Aluminum-in-olivine thermometer from Coogan et al. 2014. doi: 10.1016/j.chemgeo.2014.01.004
    Uses the Al2O3 content in Olivine, Al2O3 content of Spinel, and the Cr number of the spinel
    '''

    return (10000 / ((0.575) + 0.884 * Cr_No_sp -
            0.897 * np.log(Al2O3_Ol / Al2O3_Sp)))


def T_Wan2008(P=None, *, Cr_No_sp, Al2O3_Ol, Al2O3_Sp):
    '''
    Aluminum-in-olivine thermometer from Wan et al. (2008)  - doi: 10.2138/am.2008.2758
    Uses the Al2O3 content in Olivine, Al2O3 content of Spinel, and the Cr number of the spinel
    '''

    return (10000 / ((0.512) + 0.873 * Cr_No_sp -
            0.91 * np.log(Al2O3_Ol / Al2O3_Sp)))

##  Olivine-spinel thermometry function

def calculate_ol_sp_temp(ol_comps, sp_comps, equationT):
    ''' calculates temperatures from olivine-spinel pairs.


   Parameters
    -------
    ol_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Ol, MgO_Ol etc

    sp_comps: pandas.DataFrame
        spinel compositions with column headings SiO2_Sp, MgO_Sp etc

    equationT: str
        Equation choices:
            |   T_Wan2008
            |   T_Coogan2014

    Returns
    -------
    pandas series
       Temperature in K

    '''

    if len(sp_comps)!=len(ol_comps):
        raise ValueError('Ol comps need to be same length as Sp comps.we dont have a matching function yet if want to consider all pairs, but could make one on request!')



    combo = pd.concat([ol_comps, sp_comps], axis=1)
    Cr_No = (sp_comps['Cr2O3_Sp'] / 151.99) / \
        (sp_comps['Cr2O3_Sp'] / 151.9 + sp_comps['Al2O3_Sp'] / 101.96)


    combo.insert(1, "Cr_No_sp", Cr_No)
    if equationT == "T_Coogan2014":
        T_func = T_Coogan2014
    if equationT == "T_Wan2008":
        T_func = T_Wan2008
    kwargs = {name: combo[name] for name, p in inspect.signature(
        T_func).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    T_K = T_func(**kwargs)
    df_T_K = pd.DataFrame(data={'T_K_calc' + str(equationT.strip('T_')): T_K})

    return T_K
