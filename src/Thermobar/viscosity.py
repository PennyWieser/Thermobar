import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
from pathlib import Path
Thermobar_dir=Path(__file__).parent
from Thermobar.import_export import *
from Thermobar.core import *

def convert_F2O_to_F_ppm(F2O_wt=None):
    """ Converts from F2O in wt% to F in ppm"""
    F2O_mass=53.99621
    F_mass=18.998403
    F_calc=2*F_mass*(F2O_wt/F2O_mass)
    return F_calc*10000

def convert_F_to_F2O(F_ppm=None):
    """ Converts from F in ppm to F2O in wt%"""
    F=F_ppm/10000
    F2O_mass=53.99621
    F_mass=18.998403
    F2O_calc=(F/F_mass)*0.5*F2O_mass
    return F2O_calc


def normalize_anhydrous_to_100_incF_mol_prop(liq_comps, F2O_content=0):
    '''
    Calculates normalized hydrous molar proportions following the normalization scheme
    of Giordano et al. (2008) in their excel spreadsheet.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    F2O_content: int, pd.series
        F2O content of the liquid (wt%), by default is set at zero.

    Returns
    -------
    pd.DataFrame
        column headings of the form Liq_mol_frac_hyd, ...
    '''


    if isinstance(F2O_content, pd.Series):
        F2O_content.fillna(0, inplace=True)
    liq_comps_C=liq_comps.copy()
    liq_comps_C['NiO_Liq']=0
    liq_comps_C['Cr2O3_Liq']=0
    liq_comps_C['CoO_Liq']=0
    liq_comps_C['CO2_Liq']=0

    liq_comps_sum=(liq_comps_C['SiO2_Liq']+liq_comps_C['TiO2_Liq']+
    liq_comps_C['Al2O3_Liq']+liq_comps_C['FeOt_Liq']+liq_comps_C['MnO_Liq']+liq_comps_C['MgO_Liq']+
    liq_comps_C['CaO_Liq']+liq_comps_C['Na2O_Liq']+liq_comps_C['K2O_Liq']+
    liq_comps_C['P2O5_Liq']+F2O_content)
    Norm_Factor=(100-liq_comps_C['H2O_Liq'])/(liq_comps_sum)


    if 'Sample_ID_Liq' in liq_comps_C.columns:
        liq_comps_C_nolabel= liq_comps_C.drop('Sample_ID_Liq', axis=1)
    else:
        liq_comps_C_nolabel=liq_comps_C
    liq_comps_c_N=liq_comps_C_nolabel.multiply(Norm_Factor, axis=0)

    liq_comps_c_N=liq_comps_C_nolabel.multiply(Norm_Factor, axis=0)
    # Don't normalize water
    liq_comps_c_N['H2O_Liq']=liq_comps_C['H2O_Liq']
    liq_comps_N_mol=calculate_hydrous_mol_proportions_liquid(liq_comps=liq_comps_c_N)

    liq_comps_N_mol['F2O_Liq_mol_prop_hyd']=F2O_content*Norm_Factor/37.9968
    liq_comps_N_mol['F2O_Liq_mol_prop_hyd'].fillna(0, inplace=True)
    mols_sum = 100/(liq_comps_N_mol.sum(axis='columns'))
    cat_frac_anhyd = liq_comps_N_mol.multiply(mols_sum, axis='rows')
    cat_frac_anhyd.columns = [str(col).replace('mol_prop', 'mol_frac')
                              for col in cat_frac_anhyd.columns]

    #print(cat_frac_anhyd)


    return cat_frac_anhyd  #mols_sum #liq_comps_N_mol['sum']



def calculate_viscosity_giordano_2008(liq_comps, T=None, T_K=None, H2O_Liq=None, F2O_content=0):
    '''
    Calculates viscosity parameters A, B, C from Giordano et al. 2008.
    If a temperature is supplied, calculates viscosity.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    F2O_content: int, pd.series
        F2O content of the liquid (wt%), by default is set at zero.

    Optional:

    T or T_K: int, flt, pd.Series
        Temperature in Kelvin.
        If specified, returns viscosity and log viscosity

    H2O_Liq: int, flt, pd.Series
        Water content in wt%, overwrites that in input spreadsheet

    Returns
    -------
    pd.DataFrame
        viscosity parameters, viscosity (if T supplied), as well user-entered
        liquid composition.
    '''
    if T_K is not None:
        T=T_K
    liq_comps_c=liq_comps.copy()
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq']=H2O_Liq
        print('Water content from input overridden')




    mol_percents=normalize_anhydrous_to_100_incF_mol_prop(liq_comps=liq_comps_c,
    F2O_content=F2O_content)

    V = mol_percents["H2O_Liq_mol_frac_hyd"] + mol_percents["F2O_Liq_mol_frac_hyd"]
    TA = mol_percents["TiO2_Liq_mol_frac_hyd"] + mol_percents["Al2O3_Liq_mol_frac_hyd"]
    FM = mol_percents["FeOt_Liq_mol_frac_hyd"] + mol_percents["MnO_Liq_mol_frac_hyd"] + mol_percents["MgO_Liq_mol_frac_hyd"]
    NK = mol_percents["Na2O_Liq_mol_frac_hyd"] + mol_percents["K2O_Liq_mol_frac_hyd"]

    b1 = 159.5600 * (mol_percents["SiO2_Liq_mol_frac_hyd"] + mol_percents["TiO2_Liq_mol_frac_hyd"])
    b2 = -173.34 * mol_percents["Al2O3_Liq_mol_frac_hyd"]
    b3 = 72.13 * (mol_percents["FeOt_Liq_mol_frac_hyd"] + mol_percents["MnO_Liq_mol_frac_hyd"] + mol_percents["P2O5_Liq_mol_frac_hyd"])
    b4 = 75.69 * mol_percents["MgO_Liq_mol_frac_hyd"]
    b5 = -38.98 * mol_percents["CaO_Liq_mol_frac_hyd"]
    b6 = -84.08 * (mol_percents["Na2O_Liq_mol_frac_hyd"] + V)
    b7 = 141.54 * (V + np.log(1 + mol_percents["H2O_Liq_mol_frac_hyd"]))
    b11 = -2.43 * ((mol_percents["SiO2_Liq_mol_frac_hyd"] + mol_percents["TiO2_Liq_mol_frac_hyd"]) * FM)
    b12 = (-0.91 * ((mol_percents["SiO2_Liq_mol_frac_hyd"] + TA + mol_percents["P2O5_Liq_mol_frac_hyd"])
                    * (NK + mol_percents["H2O_Liq_mol_frac_hyd"])))
    b13 = 17.62 * (mol_percents["Al2O3_Liq_mol_frac_hyd"] * NK)

    c1 = 2.75 * mol_percents["SiO2_Liq_mol_frac_hyd"]
    c2 = 15.72 * TA
    c3 = 8.32 * FM
    c4 = 10.20 * mol_percents["CaO_Liq_mol_frac_hyd"]
    c5 = -12.29 * NK
    c6 = -99.54 * np.log(1 + V)
    c11 = 0.30 * ((mol_percents["Al2O3_Liq_mol_frac_hyd"] + FM + mol_percents["CaO_Liq_mol_frac_hyd"] - mol_percents["P2O5_Liq_mol_frac_hyd"]) * (NK + V))

    df_B = pd.DataFrame(np.array([b1,b2,b3,b4,b5,b6,b7,b11,b12,b13]).T,
                        columns = ['b1','b2','b3','b4','b5','b6','b7','b11','b12','b13']
                       )
    df_C = pd.DataFrame(np.array([c1,c2,c3,c4,c5,c6,c11]).T,
                        columns = ['c1','c2','c3','c4','c5','c6','c11']
                       )

    df_params = pd.concat([df_B, df_C],axis = 'columns')





    A = -4.55
    B = df_params.loc[:,'b1':'b13'].sum(axis = 'columns')
    C = df_params.loc[:,'c1':'c11'].sum(axis = 'columns')


    liq_comps_c.insert(0, 'A', A)
    liq_comps_c.insert(1, 'B', B.values)
    liq_comps_c.insert(2, 'C', C.values)

    if T is not None:
        df_params['T_K'] = T
        logn_melt = A + (B / (df_params['T_K'] - C))
        n_melt=10**(logn_melt)
        df_params['logn_melt'] = logn_melt
        liq_comps_c.insert(0, 'T_K', T)
        liq_comps_c.insert(0, 'logn_melt', logn_melt.values)
        liq_comps_c.insert(0, 'n_melt', n_melt.values)
    combo=pd.concat([liq_comps_c, df_params], axis=1)

    return combo

