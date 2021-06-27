import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *

## These functions calculate the additional amphibole site things needed for certain barometers

def get_amp_sites(amp_apfu_df):
    """
    get_amp_sites takes generalized atom per formula unit calculations from
    calculate_23oxygens_amphibole and puts them in the proper cation sites
    according to Leake et al., 1997.

    Parameters
    ----------
    amp_apfu_df : pandas DataFrame
        This is the dataframe output from calculate_23oxygens_amphibole. You should
        not have to modify this dataframe at all.


    Returns
    -------
    sites_df : pandas DataFrame
        a samples by cation sites dimension dataframe where each column corresponds
        to a cation site in amphibole. The suffix at the end corresponds to which site
        the cation is in:
            T = tetrahedral sites (8 total)
            C = octahedral sites (5 total)
            B  = M4 sites (2 total)
            A = A site (0 - 1 total)
    """

# new column names to drop the amp_cat_23ox. Can make this more flexible or we can just force the formatting
    # of the output inside the function. Mostly I didnt want to type it out a bunch so we can omit this little
    # loop if we want to formalize things
    newnames = []
    for name in amp_apfu_df.columns.tolist():
        newnames.append(name.split('_')[0])

    norm_cations = amp_apfu_df.copy()
    norm_cations.columns = newnames

    samples = norm_cations.index.tolist()

    # containers to fill later
    Si_T = np.empty(len(samples))
    Al_T = np.empty(len(samples))
    Al_C = np.empty(len(samples))
    Ti_C = np.empty(len(samples))
    Mg_C = np.empty(len(samples))
    Fe_C = np.empty(len(samples))
    Mn_C = np.empty(len(samples))
    Mg_B = np.empty(len(samples))
    Fe_B = np.empty(len(samples))
    Mn_B = np.empty(len(samples))
    Na_B = np.empty(len(samples))
    Ca_B = np.empty(len(samples))
    Na_A = np.empty(len(samples))
    K_A = np.empty(len(samples))
    Cr_C = np.empty(len(samples))

    for sample, i in zip(samples, range(len(samples))):
        # these are all the cations that have no site ambiguity
        Si_T[i] = norm_cations.loc[sample, 'SiO2']
        K_A[i] = norm_cations.loc[sample, 'K2O']
        Ti_C[i] = norm_cations.loc[sample, 'TiO2']
        Ca_B[i] = norm_cations.loc[sample, 'CaO']
        Cr_C[i] = norm_cations.loc[sample, 'Cr2O3']

        # site ambiguous cations. Follows Leake et al., (1997) logic
        if Si_T[i] + norm_cations.loc[sample, 'Al2O3'] > 8:
            Al_T[i] = 8 - norm_cations.loc[sample, 'SiO2']
            Al_C[i] = norm_cations.loc[sample, 'SiO2'] + \
                norm_cations.loc[sample, 'Al2O3'] - 8
        else:
            Al_T[i] = norm_cations.loc[sample, 'Al2O3']
            Al_C[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + norm_cations.loc[sample, 'MgO'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                norm_cations.loc[sample, 'MgO'] - 5
        else:
            Mg_C[i] = norm_cations.loc[sample, 'MgO']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] > 5:
            Fe_C[i] = 0
            Fe_B[i] = norm_cations.loc[sample, 'FeOt']
        else:
            Fe_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i])
            Fe_B[i] = norm_cations.loc[sample, 'FeOt'] - Fe_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = norm_cations.loc[sample, 'MnO']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe_C[i])
            Mn_B[i] = norm_cations.loc[sample, 'MnO'] - Mn_C[i]

        if Mg_B[i] + Fe_B[i] + Mn_B[i] + Ca_B[i] > 2:
            Na_B[i] = 0
            Na_A[i] = norm_cations.loc[sample, 'Na2O']
        else:
            Na_B[i] = 2 - (Mg_B[i] + Fe_B[i] + Mn_B[i] + Ca_B[i])
            # Euan has as if Na A >0, set as 0, otherwise, =Na cations 23 O -
            # Na from A site. Ask jordan where he got this from.
            Na_A[i] = norm_cations.loc[sample, 'Na2O'] - Na_B[i]

    # making the dataframe for the output
    site_vals = np.array([Si_T, Al_T, Al_C, Ti_C, Mg_C, Fe_C, Mn_C, Cr_C, Mg_B,
    Fe_B, Mn_B, Na_B, Ca_B, Na_A, K_A])
    sites_df = pd.DataFrame(site_vals.T, columns=['Si_T', 'Al_T', 'Al_C', 'Ti_C',
     'Mg_C', 'Fe_C', 'Mn_C', 'Cr_C', 'Mg_B', 'Fe_B', 'Mn_B', 'Na_B', 'Ca_B', 'Na_A', 'K_A'],
                            index=amp_apfu_df.index
                            )
    return sites_df


def amp_components_ferric_ferrous(sites_df, norm_cations):
    """
    amp_components_ferric_ferrous calculates the Fe3+ and Fe2+ apfu values of
    amphibole and adjusts the generic stoichiometry such that charge balance is
    maintained. This is based off the "f parameters" listed in Holland and Blundy
    (1994).

    Parameters
    ----------
    sites_df : pandas DataFrame
        output from the get_amp_sites function. you do not need to modify this at all
    norm_cations : pandas DataFrame
        This is the dataframe output from calculate_23oxygens_amphibole. You should
        not have to modify this dataframe at all.


    Returns
    -------
    norm_cations_hb : pandas DataFrame
        amphibole apfu values for each cation, however two things are different:
            1) FeOt is replaced by individual Fe2O3 and FeO columns
            2) all apfu values from the generic mineral recalculation have been
                adjusted by the "f" parameter from Holland and Blundy (1994)
                to maintain charge balance and stoichiometry

    """
    # A group
    f1 = 16 / sites_df.sum(axis='columns')
    f2 = 8 / sites_df['Si_T']
    f3 = 15 / (sites_df.sum(axis='columns') -
               (sites_df['Na_B'] + sites_df['Na_A']) - sites_df['K_A'] + sites_df['Mn_C'])
    f4 = 2 / sites_df['Ca_B']
    f5 = 1
    fa = pd.DataFrame({'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, })

    # B group
    f6 = 8 / (sites_df['Si_T'] + sites_df['Al_T'] + sites_df['Al_C'])
    f7 = 15 / (sites_df.sum(axis='columns') - sites_df['K_A'])
    f8 = 12.9 / (sites_df.sum(axis='columns') - (sites_df['Na_A'] + sites_df['K_A'] +
    sites_df['K_A']) - sites_df['Ca_B'] - (sites_df['Mn_C'] + sites_df['Mn_B']))
    f9 = 36 / (46 - (sites_df['Al_T'] + sites_df['Al_C']
                     ) - sites_df['Si_T'] - sites_df['Ti_C'])
    f10 = 46 / ((sites_df['Fe_C'] + sites_df['Fe_B']) + 46)
    fb = pd.DataFrame({'f6': f6, 'f7': f7, 'f8': f8, 'f9': f9, 'f10': f10, })

    f_ave = (fa.min(axis='columns') + fb.max(axis='columns')) / 2
    # f_ave = (2/3)*fa.min(axis = 'columns') + (1/3)*fb.max(axis = 'columns')

    norm_cations_hb = norm_cations.multiply(f_ave, axis='rows')
    norm_cations_hb['Fe2O3'] = 46 * (1 - f_ave)
    norm_cations_hb['FeO'] = norm_cations_hb['FeOt_Amp_cat_23ox'] - \
        norm_cations_hb['Fe2O3']
    norm_cations_hb.drop(columns=['FeOt_Amp_cat_23ox', 'oxy_renorm_factor',
                         'cation_sum_Si_Mg', 'cation_sum_Si_Ca', 'cation_sum_All'], inplace=True)
    newnames = []
    for name in norm_cations_hb.columns.tolist():
        newnames.append(name.split('_')[0])

    norm_cations_hb.columns = newnames

    return norm_cations_hb


def get_amp_sites_ferric_ferrous(amp_apfu_df):
    """
    get_amp_sites_ferric_ferrous is very similar to get_amp_sites, however it now
    incorporates the newly calculated Fe2O3 and FeO apfu values such that all
    Fe2O3 gets incorporated into the octahedral sites before any FeO. For more
    information see Leake et al., 1997 Appendix A.

    Parameters
    ----------
    amp_apfu_df :pandas DataFrame
        amphibole apfu values for each cation, now reflecting Fe2O3 and FeO
        values. This is the output from amp_components_ferric_ferrous and does
        not need to be modified at all.

    Returns
    -------
    sites_df : pandas DataFrame
    a samples by cation sites dimension dataframe where each column corresponds
        to a cation site in amphibole. The suffix at the end corresponds to which site
        the cation is in:
            T = tetrahedral sites (8 total)
            C = octahedral sites (5 total)
            B  = M4 sites (2 total)
            A = A site (0 - 1 total)
        See Leake et al., 1997 for a discussion on cation site prioritization

    """
    samples = amp_apfu_df.index.tolist()
    Si_T = np.empty(len(samples))
    Al_T = np.empty(len(samples))
    Al_C = np.empty(len(samples))
    Ti_C = np.empty(len(samples))
    Mg_C = np.empty(len(samples))
    Fe3_C = np.empty(len(samples))
    Fe2_C = np.empty(len(samples))
    Mn_C = np.empty(len(samples))
    Mg_B = np.empty(len(samples))
    Fe2_B = np.empty(len(samples))
    Mn_B = np.empty(len(samples))
    Na_B = np.empty(len(samples))
    Ca_B = np.empty(len(samples))
    Na_A = np.empty(len(samples))
    K_A = np.empty(len(samples))
    Cr_C = np.empty(len(samples))
    Fe3_C = np.empty(len(samples))

    for sample, i in zip(samples, range(len(samples))):
        Si_T[i] = amp_apfu_df.loc[sample, 'SiO2']
        K_A[i] = amp_apfu_df.loc[sample, 'K2O']
        Ti_C[i] = amp_apfu_df.loc[sample, 'TiO2']
        Ca_B[i] = amp_apfu_df.loc[sample, 'CaO']
        Cr_C[i] = amp_apfu_df.loc[sample, 'Cr2O3']
        Fe3_C[i] = amp_apfu_df.loc[sample, 'Fe2O3']

        if Si_T[i] + amp_apfu_df.loc[sample, 'Al2O3'] > 8:
            Al_T[i] = 8 - amp_apfu_df.loc[sample, 'SiO2']
            Al_C[i] = amp_apfu_df.loc[sample, 'SiO2'] + \
                amp_apfu_df.loc[sample, 'Al2O3'] - 8
        else:
            Al_T[i] = amp_apfu_df.loc[sample, 'Al2O3']
            Al_C[i] = 0
        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + \
                amp_apfu_df.loc[sample, 'MgO'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                Fe3_C[i] + amp_apfu_df.loc[sample, 'MgO'] - 5
        else:
            Mg_C[i] = amp_apfu_df.loc[sample, 'MgO']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i] > 5:
            Fe2_C[i] = 0
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO']
        else:
            Fe2_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i])
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO'] - Fe2_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe3_C[i] + Fe2_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = amp_apfu_df.loc[sample, 'MnO']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] +
                           Mg_C[i] + Fe2_C[i] + Fe3_C[i])
            Mn_B[i] = amp_apfu_df.loc[sample, 'MnO'] - Mn_C[i]

        if Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i] > 2:
            Na_B[i] = 0
            Na_A[i] = amp_apfu_df.loc[sample, 'Na2O']
        else:
            Na_B[i] = 2 - (Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i])
            Na_A[i] = amp_apfu_df.loc[sample, 'Na2O'] - Na_B[i]

    site_vals = np.array([Si_T, Al_T, Al_C, Ti_C, Mg_C, Fe3_C, Fe2_C, Mn_C, Cr_C,
    Mg_B, Fe2_B, Mn_B, Na_B, Ca_B, Na_A, K_A])
    sites_df = pd.DataFrame(site_vals.T, columns=['Si_T', 'Al_T', 'Al_C', 'Ti_C', 'Mg_C', 'Fe3_C',
    'Fe2_C','Mn_C', 'Cr_C', 'Mg_B','Fe2_B','Mn_B', 'Na_B','Ca_B', 'Na_A', 'K_A'],
                            index=amp_apfu_df.index)
    return sites_df


def get_amp_sites_mutch(amp_apfu_df):
    """
    get_amp_sites takes generalized atom per formula unit calculations from
    calculate_23oxygens_amphibole and puts them in the proper cation sites
    according to the spreasheet of Mutch et al. Gives negative numbers for Na. .

    Parameters
    ----------
    amp_apfu_df : pandas DataFrame
        This is the dataframe output from calculate_23oxygens_amphibole. You should
        not have to modify this dataframe at all.


    Returns
    -------
    sites_df : pandas DataFrame
        a samples by cation sites dimension dataframe where each column corresponds
        to a cation site in amphibole. The suffix at the end corresponds to which site
        the cation is in:
            T = tetrahedral sites (8 total)
            C = octahedral sites (5 total)
            B  = M4 sites (2 total)
            A = A site (0 - 1 total)
    """

# new column names to drop the amp_cat_23ox. Can make this more flexible or we can just force the formatting
    # of the output inside the function. Mostly I didnt want to type it out a bunch so we can omit this little
    # loop if we want to formalize things
    newnames = []
    for name in amp_apfu_df.columns.tolist():
        newnames.append(name.split('_')[0])

    norm_cations = amp_apfu_df.copy()
    norm_cations.columns = newnames

    samples = norm_cations.index.tolist()

    # containers to fill later
    Si_T = np.empty(len(samples))
    Al_T = np.empty(len(samples))
    Al_C = np.empty(len(samples))
    Ti_C = np.empty(len(samples))
    Mg_C = np.empty(len(samples))
    Fe_C = np.empty(len(samples))
    Mn_C = np.empty(len(samples))
    Mg_B = np.empty(len(samples))
    Fe_B = np.empty(len(samples))
    Mn_B = np.empty(len(samples))
    Na_B = np.empty(len(samples))
    Ca_B = np.empty(len(samples))
    Na_A = np.empty(len(samples))
    K_A = np.empty(len(samples))
    Cr_C = np.empty(len(samples))

    for sample, i in zip(samples, range(len(samples))):
        # these are all the cations that have no site ambiguity
        Si_T[i] = norm_cations.loc[sample, 'SiO2']
        K_A[i] = norm_cations.loc[sample, 'K2O']
        Ti_C[i] = norm_cations.loc[sample, 'TiO2']
        Ca_B[i] = norm_cations.loc[sample, 'CaO']
        Cr_C[i] = norm_cations.loc[sample, 'Cr2O3']

        # site ambiguous cations. Follows Leake et al., (1997) logic
        if Si_T[i] + norm_cations.loc[sample, 'Al2O3'] > 8:
            Al_T[i] = 8 - norm_cations.loc[sample, 'SiO2']
            Al_C[i] = norm_cations.loc[sample, 'SiO2'] + \
                norm_cations.loc[sample, 'Al2O3'] - 8
        else:
            Al_T[i] = norm_cations.loc[sample, 'Al2O3']
            Al_C[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + norm_cations.loc[sample, 'MgO'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                norm_cations.loc[sample, 'MgO'] - 5
        else:
            Mg_C[i] = norm_cations.loc[sample, 'MgO']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] > 5:
            Fe_C[i] = 0
            Fe_B[i] = norm_cations.loc[sample, 'FeOt']
        else:
            Fe_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i])
            Fe_B[i] = norm_cations.loc[sample, 'FeOt'] - Fe_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = norm_cations.loc[sample, 'MnO']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe_C[i])
            Mn_B[i] = norm_cations.loc[sample, 'MnO'] - Mn_C[i]

        if Mg_B[i] + Fe_B[i] + Mn_B[i] + Ca_B[i] + \
                amp_apfu_df['Na2O_Amp_cat_23ox'].iloc[i] > 2:
            Na_B[i] = 2 - (Mg_B[i] + Fe_B[i] + Mn_B[i] + Ca_B[i])
            Na_A[i] = amp_apfu_df['Na2O_Amp_cat_23ox'].iloc[i] - Na_B[i]
        else:
            Na_B[i] = amp_apfu_df['Na2O_Amp_cat_23ox'].iloc[i]
            # Euan has as if Na A >0, set as 0, otherwise, =Na cations 23 O -
            # Na from A site. Ask jordan where he got this from.
            Na_A[i] = amp_apfu_df['Na2O_Amp_cat_23ox'].iloc[i] - Na_B[i]


    site_vals = np.array([Si_T, Al_T, Al_C, Ti_C, Mg_C, Fe_C, Mn_C, Cr_C, Mg_B,
    Fe_B, Mn_B, Na_B, Ca_B, Na_A, K_A])
    sites_df = pd.DataFrame(site_vals.T, columns=['Si_T', 'Al_T', 'Al_C', 'Ti_C',
     'Mg_C', 'Fe_C', 'Mn_C', 'Cr_C', 'Mg_B', 'Fe_B', 'Mn_B', 'Na_B', 'Ca_B', 'Na_A', 'K_A'],
    index=amp_apfu_df.index)
    return sites_df


def amp_components_ferric_ferrous_mutch(sites_df, norm_cations):
    """
    amp_components_ferric_ferrous calculates the Fe3+ and Fe2+ apfu values of
    amphibole and adjusts the generic stoichiometry such that charge balance is
    maintained. This is based off the "f parameters" listed in Holland and Blundy
    (1994), using the averaging in the spreadsheet supplied by Euan Mutch

    Parameters
    ----------
    sites_df : pandas DataFrame
        output from the get_amp_sites function. you do not need to modify this at all
    norm_cations : pandas DataFrame
        This is the dataframe output from calculate_23oxygens_amphibole. You should
        not have to modify this dataframe at all.


    Returns
    -------
    norm_cations_hb : pandas DataFrame
        amphibole apfu values for each cation, however two things are different:
            1) FeOt is replaced by individual Fe2O3 and FeO columns
            2) all apfu values from the generic mineral recalculation have been
                adjusted by the "f" parameter from Holland and Blundy (1994)
                to maintain charge balance and stoichiometry

    """
    # A group
    f1 = 16 / sites_df.sum(axis='columns')
    f2 = 8 / sites_df['Si_T']
    #f3 = 15/(sites_df.sum(axis = 'columns') - (sites_df['Na_B'] + sites_df['Na_A']) - sites_df['K_A'] + sites_df['Mn_C'])
    f3 = 15 / (sites_df.sum(axis='columns') -
               sites_df['K_A'] - (sites_df['Na_A'] + sites_df['Na_B']) + sites_df['Mn_B'])

    f4 = 2 / sites_df['Ca_B']
    f5 = 1
    fa = pd.DataFrame({'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, })

    # B group
    f6 = 8 / (sites_df['Si_T'] + sites_df['Al_T'] + sites_df['Al_C'])
    f7 = 15 / (sites_df.sum(axis='columns') - sites_df['K_A'])
    f8 = 12.9 / (sites_df.sum(axis='columns') - (sites_df['Na_A'] + sites_df['K_A'] +
                                                 sites_df['K_A']) - sites_df['Ca_B'] - (sites_df['Mn_C'] + sites_df['Mn_B']))
    f8 = (13 / (sites_df.sum(axis='columns') - (sites_df['K_A'] + sites_df['Na_A'] + sites_df['Na_B'])
                - (sites_df['Mn_B'] + sites_df['Mn_C']) - sites_df['Ca_B']))

    f9 = 36 / (46 - (sites_df['Al_T'] + sites_df['Al_C']
                     ) - sites_df['Si_T'] - sites_df['Ti_C'])
    f10 = 46 / ((sites_df['Fe_C'] + sites_df['Fe_B']) + 46)
    fb = pd.DataFrame({'f6': f6, 'f7': f7, 'f8': f8, 'f9': f9, 'f10': f10, })

    #f_ave = (fa.min(axis = 'columns') + fb.max(axis = 'columns'))/2
    f_ave = (2 / 3) * fa.min(axis='columns') + (1 / 3) * fb.max(axis='columns')

    norm_cations_hb = norm_cations.multiply(f_ave, axis='rows')
    norm_cations_hb['Fe2O3'] = 46 * (1 - f_ave)
    norm_cations_hb['FeO'] = norm_cations_hb['FeOt_Amp_cat_23ox'] - \
        norm_cations_hb['Fe2O3']
    norm_cations_hb.drop(columns=['FeOt_Amp_cat_23ox', 'oxy_renorm_factor',
                         'cation_sum_Si_Mg', 'cation_sum_Si_Ca', 'cation_sum_All'], inplace=True)
    newnames = []
    for name in norm_cations_hb.columns.tolist():
        newnames.append(name.split('_')[0])

    norm_cations_hb.columns = newnames

    return norm_cations_hb

##


def get_amp_sites_ferric_ferrous_mutch(amp_apfu_df):
    """
    get_amp_sites_ferric_ferrous is very similar to get_amp_sites, however it now
    incorporates the newly calculated Fe2O3 and FeO apfu values such that all
    Fe2O3 gets incorporated into the octahedral sites before any FeO. For more
    information see Leake et al., 1997 Appendix A.

    Parameters
    ----------
    amp_apfu_df :pandas DataFrame
        amphibole apfu values for each cation, now reflecting Fe2O3 and FeO
        values. This is the output from amp_components_ferric_ferrous and does
        not need to be modified at all.

    Returns
    -------
    sites_df : pandas DataFrame
    a samples by cation sites dimension dataframe where each column corresponds
        to a cation site in amphibole. The suffix at the end corresponds to which site
        the cation is in:
            T = tetrahedral sites (8 total)
            C = octahedral sites (5 total)
            B  = M4 sites (2 total)
            A = A site (0 - 1 total)
        See Leake et al., 1997 for a discussion on cation site prioritization

    """
    samples = amp_apfu_df.index.tolist()
    Si_T = np.empty(len(samples))
    Al_T = np.empty(len(samples))
    Al_C = np.empty(len(samples))
    Ti_C = np.empty(len(samples))
    Mg_C = np.empty(len(samples))
    Fe3_C = np.empty(len(samples))
    Fe2_C = np.empty(len(samples))
    Mn_C = np.empty(len(samples))
    Mg_B = np.empty(len(samples))
    Fe2_B = np.empty(len(samples))
    Mn_B = np.empty(len(samples))
    Na_B = np.empty(len(samples))
    Ca_B = np.empty(len(samples))
    Na_A = np.empty(len(samples))
    K_A = np.empty(len(samples))
    Cr_C = np.empty(len(samples))
    Fe3_C = np.empty(len(samples))

    for sample, i in zip(samples, range(len(samples))):
        Si_T[i] = amp_apfu_df.loc[sample, 'SiO2']
        K_A[i] = amp_apfu_df.loc[sample, 'K2O']
        Ti_C[i] = amp_apfu_df.loc[sample, 'TiO2']
        Ca_B[i] = amp_apfu_df.loc[sample, 'CaO']
        Cr_C[i] = amp_apfu_df.loc[sample, 'Cr2O3']
        Fe3_C[i] = amp_apfu_df.loc[sample, 'Fe2O3']

        if Si_T[i] + amp_apfu_df.loc[sample, 'Al2O3'] > 8:
            Al_T[i] = 8 - amp_apfu_df.loc[sample, 'SiO2']
            Al_C[i] = amp_apfu_df.loc[sample, 'SiO2'] + \
                amp_apfu_df.loc[sample, 'Al2O3'] - 8
        else:
            Al_T[i] = amp_apfu_df.loc[sample, 'Al2O3']
            Al_C[i] = 0
        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + \
                amp_apfu_df.loc[sample, 'MgO'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                Fe3_C[i] + amp_apfu_df.loc[sample, 'MgO'] - 5
        else:
            Mg_C[i] = amp_apfu_df.loc[sample, 'MgO']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i] > 5:
            Fe2_C[i] = 0
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO']
        else:
            Fe2_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i])
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO'] - Fe2_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe3_C[i] + Fe2_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = amp_apfu_df.loc[sample, 'MnO']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] +
                           Mg_C[i] + Fe2_C[i] + Fe3_C[i])
            Mn_B[i] = amp_apfu_df.loc[sample, 'MnO'] - Mn_C[i]

        if Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i] + \
                amp_apfu_df.loc[sample, 'Na2O'] > 2:
            Na_B[i] = 2 - (Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i])
            Na_A[i] = amp_apfu_df.loc[sample, 'Na2O'] - Na_B[i]

        else:
            Na_B[i] = amp_apfu_df.loc[sample, 'Na2O']
            # Euan has as if Na A >0, set as 0, otherwise, =Na cations 23 O -
            # Na from A site. Ask jordan where he got this from.
            Na_A[i] = amp_apfu_df.loc[sample, 'Na2O'] - Na_B[i]

    site_vals = np.array([Si_T, Al_T, Al_C, Ti_C, Mg_C, Fe3_C, Fe2_C, Mn_C, Cr_C, Mg_B,
    Fe2_B, Mn_B, Na_B, Ca_B, Na_A, K_A])
    sites_df = pd.DataFrame(site_vals.T, columns=['Si_T', 'Al_T', 'Al_C', 'Ti_C',
     'Mg_C', 'Fe3_C', 'Fe2_C', 'Mn_C', 'Cr_C', 'Mg_B', 'Fe2_B', 'Mn_B', 'Na_B', 'Ca_B', 'Na_A', 'K_A'],
    index=amp_apfu_df.index)

    return sites_df



## Equations: Amphibole-only Barometers

def P_Ridolfi2012_1a(T=None, *, SiO2_Amp_13_cat, TiO2_Amp_13_cat, FeOt_Amp_13_cat,
                     MgO_Amp_13_cat, CaO_Amp_13_cat, K2O_Amp_13_cat, Na2O_Amp_13_cat, Al2O3_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1a of Ridolfi and Renzulli (2012). Calibrated between 1.3-22 kbars
    '''
    return 0.01 * (np.exp(125.9332115 - 9.587571403 * SiO2_Amp_13_cat - 10.11615567 * TiO2_Amp_13_cat
    - 8.173455128 * Al2O3_Amp_13_cat- 9.226076274 * FeOt_Amp_13_cat - 8.793390507 * MgO_Amp_13_cat
    - 1.6658613 * CaO_Amp_13_cat + 2.48347198 * Na2O_Amp_13_cat + 2.519184959 * K2O_Amp_13_cat))


def P_Ridolfi2012_1b(T=None, *, SiO2_Amp_13_cat, TiO2_Amp_13_cat, FeOt_Amp_13_cat,
                     MgO_Amp_13_cat, CaO_Amp_13_cat, K2O_Amp_13_cat, Na2O_Amp_13_cat, Al2O3_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1b of Ridolfi and Renzulli (2012). Calibrated between 1.3-5 kbars
    '''
    return (0.01 * (np.exp(38.722545085 - 2.695663047 * SiO2_Amp_13_cat - 2.35647038717941 * TiO2_Amp_13_cat
            - 1.30063975020919 * Al2O3_Amp_13_cat - 2.7779767369382 * FeOt_Amp_13_cat
            - 2.48384821395444 * MgO_Amp_13_cat- 0.661386638563983 * CaO_Amp_13_cat
            - 0.270530207793162 * Na2O_Amp_13_cat + 0.111696322092308 * K2O_Amp_13_cat)))


def P_Ridolfi2012_1c(T=None, *, SiO2_Amp_13_cat, TiO2_Amp_13_cat, FeOt_Amp_13_cat, MgO_Amp_13_cat,
                     CaO_Amp_13_cat, K2O_Amp_13_cat, Na2O_Amp_13_cat, Al2O3_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1c of Ridolfi and Renzulli (2012). Calibrated between 1.3-5 kbars
    '''
    return (0.01 * (24023.367332 - 1925.298250* SiO2_Amp_13_cat
    - 1720.63250944418 * TiO2_Amp_13_cat - 1478.53847391822 * Al2O3_Amp_13_cat
    - 1843.19249824537 * FeOt_Amp_13_cat - 1746.94437497404 * MgO_Amp_13_cat
    - 158.279055907371 * CaO_Amp_13_cat - 40.4443246813322 * Na2O_Amp_13_cat
    + 253.51576430265 * K2O_Amp_13_cat))


def P_Ridolfi2012_1d(T=None, *, SiO2_Amp_13_cat, TiO2_Amp_13_cat, FeOt_Amp_13_cat, MgO_Amp_13_cat, CaO_Amp_13_cat,
                     K2O_Amp_13_cat, Na2O_Amp_13_cat, Al2O3_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1d of Ridolfi and Renzulli (2012). . Calibrated between 4-15 kbars
    '''
    return (0.01 * (26105.7092067 - 1991.93398583468 * SiO2_Amp_13_cat
    - 3034.9724955129 * TiO2_Amp_13_cat - 1472.2242262718 * Al2O3_Amp_13_cat - 2454.76485311127 * FeOt_Amp_13_cat
    - 2125.79095875747 * MgO_Amp_13_cat - 830.644984403603 * CaO_Amp_13_cat
    + 2708.82902160291 * Na2O_Amp_13_cat + 2204.10480275638 * K2O_Amp_13_cat))


def P_Ridolfi2012_1e(T=None, *, SiO2_Amp_13_cat, TiO2_Amp_13_cat, FeOt_Amp_13_cat,
                     MgO_Amp_13_cat, CaO_Amp_13_cat, K2O_Amp_13_cat, Na2O_Amp_13_cat, Al2O3_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1e of Ridolfi and Renzulli (2012).  Calibrated between 930-2200 kbars
    '''
    return (0.01 * np.exp(26.5426319326957 - 1.20851740386237 * SiO2_Amp_13_cat
    - 3.85930939071001 * TiO2_Amp_13_cat - 1.10536070667051 * Al2O3_Amp_13_cat
    - 2.90677947035468 * FeOt_Amp_13_cat - 2.64825741548332 *MgO_Amp_13_cat
    + 0.513357584438019 * CaO_Amp_13_cat
    + 2.9751971464851 * Na2O_Amp_13_cat + 1.81467032749331 * K2O_Amp_13_cat))


def P_Ridolfi2010(T=None, *, Al2O3_Amp_cat_23ox, cation_sum_Si_Ca):
    '''
    Amphibole-only (Al) barometer: Ridolfi et al. (2010)
    '''
    return (10 * (19.209 * np.exp(1.438 * Al2O3_Amp_cat_23ox *
            13 / cation_sum_Si_Ca)) / 1000)


def P_Hammerstrom1986_eq1(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hammerstrom and Zen, 1986 eq.1
    '''
    return (-3.92 + 5.03 * Al2O3_Amp_cat_23ox)


def P_Hammerstrom1986_eq2(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hammerstrom and Zen, 1986 eq.2
    '''
    return (1.27 * (Al2O3_Amp_cat_23ox**2.01))


def P_Hammerstrom1986_eq3(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hammerstrom and Zen, 1986 eq.3
    '''
    return (0.26 * np.exp(1.48 * Al2O3_Amp_cat_23ox))


def P_Hollister1987(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hollister et al. 1987
    '''
    return (-4.76 + 5.64 * Al2O3_Amp_cat_23ox)


def P_Johnson1989(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Johnson and Rutherford, 1989
    '''
    return (-3.46 + 4.23 * Al2O3_Amp_cat_23ox)


def P_Anderson1995(T, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Anderson and Smith (1995)
    '''
    return (4.76 * Al2O3_Amp_cat_23ox - 3.01 - (((T - 273.15 - 675) / 85)
            * (0.53 * Al2O3_Amp_cat_23ox + 0.005294 * (T - 273.15 - 675))))


def P_Blundy1990(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Blundy et al. 1990
    '''
    return (5.03 * Al2O3_Amp_cat_23ox - 3.53)


def P_Schmidt1992(T=None, *, Al2O3_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Schmidt 1992
    '''
    return (-3.01 + 4.76 * Al2O3_Amp_cat_23ox)



## Function: Amphibole-only barometry

Amp_only_P_funcs = { P_Ridolfi2012_1a, P_Ridolfi2012_1b, P_Ridolfi2012_1c, P_Ridolfi2012_1d,
P_Ridolfi2012_1e, P_Ridolfi2010, P_Hammerstrom1986_eq1, P_Hammerstrom1986_eq2, P_Hammerstrom1986_eq3, P_Hollister1987,
P_Johnson1989, P_Blundy1990, P_Schmidt1992, P_Anderson1995} # put on outside

Amp_only_P_funcs_by_name= {p.__name__: p for p in Amp_only_P_funcs}

Amp_only_P_funcs_sim = {P_Hammerstrom1986_eq1, P_Hammerstrom1986_eq2, P_Hammerstrom1986_eq3,
P_Hollister1987, P_Johnson1989, P_Blundy1990, P_Schmidt1992, P_Anderson1995} # put on outside

Amp_only_P_funcs_by_name_sim = {p.__name__: p for p in Amp_only_P_funcs_sim}

def calculate_amp_only_press(amp_comps=None, equationP=None, T=None):
    """
    Amphibole-only barometry, returns pressure in kbar.

    amp_comps: DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.


    EquationP: str
        | P_Mutch2016 (T-independent)
        | P_Ridolfi2012_1a (T-independent)
        | P_Ridolfi2012_1b (T-independent)
        | P_Ridolfi2012_1c (T-independent)
        | P_Ridolfi2012_1d (T-independent)
        | P_Ridolfi2012_1e (T-independent)
        | P_Ridolfi2021 - (T-independent)- Uses new algorithm in 2021 paper to
        select pressures from equations 1a-e.

        | P_Ridolfi2010  (T-independent)
        | P_Hammerstrom1986_eq1  (T-independent)
        | P_Hammerstrom1986_eq2 (T-independent)
        | P_Hammerstrom1986_eq3 (T-independent)
        | P_Hollister1987 (T-independent)
        | P_Johnson1989 (T-independent)
        | P_Blundy1990 (T-independent)
        | P_Schmidt1992 (T-independent)
        | P_Anderson1995 (*T-dependent*)

    T: float, int, series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas series
       Pressure in kbar
    """
    if equationP !="P_Ridolfi2021" and equationP !="P_Mutch2016":
        try:
            func = Amp_only_P_funcs_by_name[equationP]
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
            if amp_comps is not None:
                if len(T) != len(amp_comps):
                    raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of amphibole compositions')



    if "Sample_ID_Amp" not in amp_comps:
        amp_comps['Sample_ID_Amp'] = amp_comps.index

    if equationP == "P_Mutch2016":
        ox23 = calculate_23oxygens_amphibole(amp_comps)
        Amp_sites_initial = get_amp_sites_mutch(ox23)
        norm_cat = amp_components_ferric_ferrous_mutch(Amp_sites_initial, ox23)
        final_cat = get_amp_sites_ferric_ferrous_mutch(norm_cat)
        final_cat['Al_tot'] = final_cat['Al_T'] + final_cat['Al_C']
        P_kbar = 0.5 + 0.331 * \
            final_cat['Al_tot'] + 0.995 * (final_cat['Al_tot'])**2
        final_cat.insert(0, "P_kbar_calc", P_kbar)
        return final_cat

    if 'Ridolfi2012' in equationP or equationP == "P_Ridolfi2021":
        cat13 = calculate_13cations_amphibole_ridolfi(amp_comps)
        myAmps1_label = amp_comps.drop(['Sample_ID_Amp'], axis='columns')
        Sum_input = myAmps1_label.sum(axis='columns')

        kwargs_1a = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1a).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1b = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1b).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1c = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1c).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1d = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1d).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1e = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1e).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

        P_MPa_1a = 100 * partial(P_Ridolfi2012_1a, **kwargs_1a)(0)
        P_MPa_1b = 100 * partial(P_Ridolfi2012_1b, **kwargs_1b)(0)
        P_MPa_1c = 100 * partial(P_Ridolfi2012_1c, **kwargs_1c)(0)
        P_MPa_1d = 100 * partial(P_Ridolfi2012_1d, **kwargs_1d)(0)
        P_MPa_1e = 100 * partial(P_Ridolfi2012_1e, **kwargs_1e)(0)

        if equationP == "P_Ridolfi2021":

            XPae = (P_MPa_1a - P_MPa_1e) / P_MPa_1a
            deltaPdb = P_MPa_1d - P_MPa_1b

            P_MPa = np.empty(len(P_MPa_1a))
            name=np.empty(len(P_MPa_1a), dtype=np.dtype('U100'))
            for i in range(0, len(P_MPa_1a)):
                if P_MPa_1b[i] < 335:
                    P_MPa[i] = P_MPa_1b[i]
                    name[i]="1b"
                elif P_MPa_1b[i] < 399:
                    P_MPa[i] = (P_MPa_1b[i] + P_MPa_1c[i]) / 2
                    name[i]="(1b+1c)/2"
                elif P_MPa_1c[i] < 415:
                    P_MPa[i] = (P_MPa_1c[i])
                    name[i]="1c"
                elif P_MPa_1d[i] < 470:
                    P_MPa[i] = (P_MPa_1c[i])
                    name[i]="1c"
                elif XPae[i] > 0.22:
                    P_MPa[i] = (P_MPa_1c[i] + P_MPa_1d[i]) / 2
                    name[i]="1c+1d"
                elif deltaPdb[i] > 350:
                    P_MPa[i] = P_MPa_1e[i]
                    name[i]="1e"
                elif deltaPdb[i] > 210:
                    P_MPa[i] = P_MPa_1d[i]
                    name[i]="1d"
                elif deltaPdb[i] < 75:
                    P_MPa[i] = P_MPa_1c[i]
                    name[i]="1c"
                elif XPae[i] < -0.2:
                    P_MPa[i] = (P_MPa_1b[i] + P_MPa_1c[i]) / 2
                    name[i]="(1b+1c)/2"
                elif XPae[i] > 0.05:
                    P_MPa[i] = (P_MPa_1c[i] + P_MPa_1d[i]) / 2
                    name[i]="(1c+1d)/2"
                else:
                    P_MPa[i] = P_MPa_1a[i]
                    name[i]="1a"
                if Sum_input[i] < 90:
                    P_MPa[i] = np.nan
                P_kbar = pd.DataFrame(data={"P_kbar_calc": (P_MPa / 100), "equation": name})

            if any(Sum_input) < 90:
                print('The sum of Si, Ti, Al, Fe, Mg, Ca, Na and K based on 13 cations for some '
                'input amphiboles is <90; P=nan is returned for these analyses')

        if equationP == "P_Ridolfi2012_1a":
            P_kbar = P_MPa_1a / 100

        if equationP == "P_Ridolfi2012_1b":
            P_kbar = P_MPa_1b / 100

        if equationP == "P_Ridolfi2012_1c":
            P_kbar = P_MPa_1c / 100

        if equationP == "P_Ridolfi2012_1d":
            P_kbar = P_MPa_1d / 100

        if equationP == "P_Ridolfi2012_1e":
            P_kbar = P_MPa_1e / 100

        return P_kbar

    if equationP != "Mutch2016" and 'Ridolfi2012' not in equationP and equationP != "P_Ridolfi2021":
        ox23_amp = calculate_23oxygens_amphibole(amp_comps=amp_comps)

    kwargs = {name: ox23_amp[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(T, str) or T is None:
        if T == "Solve":
            P_kbar = partial(func, **kwargs)
        if T is None:
            P_kbar=func(**kwargs)

    else:
        P_kbar=func(T, **kwargs)

    return P_kbar



## Amphibole-only thermometers


def T_Put2016_eq5(P=None, *, SiO2_Amp_cat_23ox,
                  TiO2_Amp_cat_23ox, FeOt_Amp_cat_23ox, Na2O_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Equation 5 of Putirka et al. (2016)
    '''
    return (273.15 + 1781 - 132.74 * SiO2_Amp_cat_23ox + 116.6 *
            TiO2_Amp_cat_23ox - 69.41 * FeOt_Amp_cat_23ox + 101.62 * Na2O_Amp_cat_23ox)


def T_Put2016_eq6(P, *, SiO2_Amp_cat_23ox,
                  TiO2_Amp_cat_23ox, FeOt_Amp_cat_23ox, Na2O_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Equation 6 of Putirka et al. (2016)
    '''
    return (273.15 + 1687 - 118.7 * SiO2_Amp_cat_23ox + 131.56 * TiO2_Amp_cat_23ox -
            71.41 * FeOt_Amp_cat_23ox + 86.13 * Na2O_Amp_cat_23ox + 22.44 * P / 10)


def T_Put2016_SiHbl(P=None, *, SiO2_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Si in Hbl, Putirka et al. (2016)
    '''
    return (273.15 + 2061 - 178.4 * SiO2_Amp_cat_23ox)

def T_Ridolfi2012(P, *, SiO2_Amp_13_cat, TiO2_Amp_13_cat, FeOt_Amp_13_cat,
                  MgO_Amp_13_cat, CaO_Amp_13_cat, K2O_Amp_13_cat, Na2O_Amp_13_cat, Al2O3_Amp_13_cat):
    '''
    Amphibole-only thermometer of Ridolfi and Renzuli, 2012
    '''
    return (273.15 + 8899.682 - 691.423 * SiO2_Amp_13_cat - 391.548 * TiO2_Amp_13_cat - 666.149 * Al2O3_Amp_13_cat
    - 636.484 * FeOt_Amp_13_cat -584.021 * MgO_Amp_13_cat - 23.215 * CaO_Amp_13_cat
    + 79.971 * Na2O_Amp_13_cat - 104.134 * K2O_Amp_13_cat + 78.993 * np.log(P * 100))

def T_Put2016_eq8(P, *, SiO2_Amp_cat_23ox, TiO2_Amp_cat_23ox,
                  MgO_Amp_cat_23ox, Na2O_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Eq8,  Putirka et al. (2016)
    '''
    return (273.15+1201.4 - 97.93 * SiO2_Amp_cat_23ox + 201.82 * TiO2_Amp_cat_23ox +
            72.85 * MgO_Amp_cat_23ox + 88.9 * Na2O_Amp_cat_23ox + 40.65 * P / 10)
## Equations: Amphibole-Liquid barometers

def P_Put2016_eq7a(T=None, *, Al2O3_Amp_cat_23ox, Na2O_Amp_cat_23ox,
K2O_Amp_cat_23ox, Al2O3_Liq_mol_frac_hyd, Na2O_Liq_mol_frac_hyd,
H2O_Liq_mol_frac_hyd, P2O5_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid barometer: Equation 7a of Putirka et al. (2016)
    '''
    return (10 * (-3.093 - 4.274 * np.log(Al2O3_Amp_cat_23ox / Al2O3_Liq_mol_frac_hyd)
    - 4.216 * np.log(Al2O3_Liq_mol_frac_hyd) + 63.3 * P2O5_Liq_mol_frac_hyd +
    1.264 * H2O_Liq_mol_frac_hyd + 2.457 * Al2O3_Amp_cat_23ox + 1.86 * K2O_Amp_cat_23ox
    + 0.4 * np.log(Na2O_Amp_cat_23ox / Na2O_Liq_mol_frac_hyd)))


def P_Put2016_eq7b(T=None, *, Al2O3_Liq_mol_frac_hyd, P2O5_Liq_mol_frac_hyd, Al2O3_Amp_cat_23ox,
    SiO2_Liq_mol_frac_hyd, Na2O_Liq_mol_frac_hyd, K2O_Liq_mol_frac_hyd, CaO_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid barometer: Equation 7b of Putirka et al. (2016)
    '''
    return (-64.79 - 6.064 * np.log(Al2O3_Amp_cat_23ox / Al2O3_Liq_mol_frac_hyd)
    + 61.75 * SiO2_Liq_mol_frac_hyd + 682 * P2O5_Liq_mol_frac_hyd
    - 101.9 *CaO_Liq_mol_frac_hyd + 7.85 * Al2O3_Amp_cat_23ox
    - 46.46 * np.log(SiO2_Liq_mol_frac_hyd)
    - 4.81 * np.log(Na2O_Liq_mol_frac_hyd + K2O_Liq_mol_frac_hyd))


def P_Put2016_eq7c(T=None, *, Al2O3_Amp_cat_23ox, K2O_Amp_cat_23ox,
                   P2O5_Liq_mol_frac, Al2O3_Liq_mol_frac, Na2O_Amp_cat_23ox, Na2O_Liq_mol_frac):
    '''
    Amphibole-Liquid barometer: Equation 7c of Putirka et al. (2016)
    '''
    return (-45.55 + 26.65 * Al2O3_Amp_cat_23ox + 22.52 * K2O_Amp_cat_23ox
    + 439 * P2O5_Liq_mol_frac - 51.1 * np.log(Al2O3_Liq_mol_frac) -
    46.3 * np.log(Al2O3_Amp_cat_23ox / (Al2O3_Liq_mol_frac))
    + 5.231 * np.log(Na2O_Amp_cat_23ox / (Na2O_Liq_mol_frac)))

## Equations: Amphibole-Liquid thermometers


def T_Put2016_eq4b(P=None, *, H2O_Liq_mol_frac_hyd, FeOt_Amp_cat_23ox, FeOt_Liq_mol_frac_hyd, MgO_Liq_mol_frac_hyd,
                   MnO_Liq_mol_frac_hyd, Al2O3_Liq_mol_frac_hyd, TiO2_Amp_cat_23ox, TiO2_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid thermometer: Eq4b,  Putirka et al. (2016)
    '''
    return (273.15 + (8037.85 / (3.69 - 2.62 * H2O_Liq_mol_frac_hyd + 0.66 * FeOt_Amp_cat_23ox
    - 0.416 * np.log(TiO2_Liq_mol_frac_hyd) + 0.37 * np.log(MgO_Liq_mol_frac_hyd)
    -1.05 * np.log((FeOt_Liq_mol_frac_hyd + MgO_Liq_mol_frac_hyd
    + MnO_Liq_mol_frac_hyd) * Al2O3_Liq_mol_frac_hyd)
    - 0.462 * np.log(TiO2_Amp_cat_23ox / TiO2_Liq_mol_frac_hyd))))


def T_Put2016_eq4a_amp_sat(P=None, *, FeOt_Liq_mol_frac_hyd, TiO2_Liq_mol_frac_hyd, Al2O3_Liq_mol_frac_hyd,
                           MnO_Liq_mol_frac_hyd, MgO_Liq_mol_frac_hyd, Na2O_Amp_cat_23ox, Na2O_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid thermometer Saturation surface of amphibole, Putirka et al. (2016)
    '''
    return (273.15 + (6383.4 / (-12.07 + 45.4 * Al2O3_Liq_mol_frac_hyd + 12.21 * FeOt_Liq_mol_frac_hyd -
    0.415 * np.log(TiO2_Liq_mol_frac_hyd) - 3.555 * np.log(Al2O3_Liq_mol_frac_hyd)
     - 0.832 * np.log(Na2O_Liq_mol_frac_hyd) -0.481 * np.log((FeOt_Liq_mol_frac_hyd
     + MgO_Liq_mol_frac_hyd + MnO_Liq_mol_frac_hyd) * Al2O3_Liq_mol_frac_hyd)
     - 0.679 * np.log(Na2O_Amp_cat_23ox / Na2O_Liq_mol_frac_hyd))))


def T_Put2016_eq9(P=None, *, SiO2_Amp_cat_23ox, TiO2_Amp_cat_23ox, MgO_Amp_cat_23ox,
FeOt_Amp_cat_23ox, Na2O_Amp_cat_23ox,  FeOt_Liq_mol_frac_hyd, Al2O3_Amp_cat_23ox, Al2O3_Liq_mol_frac_hyd,
K2O_Amp_cat_23ox, CaO_Amp_cat_23ox, Na2O_Liq_mol_frac_hyd, K2O_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid thermometer: Eq9,  Putirka et al. (2016)
    '''
    NaM4_1=2-FeOt_Amp_cat_23ox-CaO_Amp_cat_23ox
    NaM4=np.empty(len(NaM4_1))
    for i in range(0, len(NaM4)):
        if NaM4_1[i]<=0.1:
            NaM4[i]=0
        else:
            NaM4[i]=NaM4_1[i]

    HelzA=Na2O_Amp_cat_23ox-NaM4
    ln_KD_Na_K=np.log((K2O_Amp_cat_23ox/HelzA)*(Na2O_Liq_mol_frac_hyd/K2O_Liq_mol_frac_hyd))

    return (273.15+(10073.5/(9.75+0.934*SiO2_Amp_cat_23ox-1.454*TiO2_Amp_cat_23ox
    -0.882*MgO_Amp_cat_23ox-1.123*Na2O_Amp_cat_23ox-0.322*np.log(FeOt_Liq_mol_frac_hyd)
    -0.7593*np.log(Al2O3_Amp_cat_23ox/Al2O3_Liq_mol_frac_hyd)-0.15*ln_KD_Na_K)))



## Function: Amphibole-only temperature

Amp_only_T_funcs = {T_Put2016_eq5, T_Put2016_eq6, T_Put2016_SiHbl, T_Put2016_eq8,
 T_Ridolfi2012, T_Put2016_eq4a_amp_sat, T_Put2016_eq8} # put on outside

Amp_only_T_funcs_by_name= {p.__name__: p for p in Amp_only_T_funcs}




def calculate_amp_only_temp(amp_comps, equationT, P=None):
    '''
    Amphibole-only thermometry, calculates temperature in Kelvin.

    equationT: str
        |   T_Put2016_eq5 (P-independent)
        |   T_Put2016_eq6 (P-dependent)
        |   T_Put2016_SiHbl (P-independent)
        |   T_Ridolfi2012 (P-dependent)
        |   T_Put2016_eq8 (P-dependent)


    P: float, int, series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas.series: Pressure in kbar (if eq_tests=False

    '''
    try:
        func = Amp_only_T_funcs_by_name[equationT]
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
        if amp_comps is not None:
            if len(P) != len(amp_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of amphibole compositions')


    if equationT == "T_Ridolfi2012":
        if P is None:
            raise Exception(
                'You have selected a P-dependent thermometer, please enter an option for P')
        cat13 = calculate_13cations_amphibole_ridolfi(amp_comps)
        myAmps1_label = amp_comps.drop(['Sample_ID_Amp'], axis='columns')
        Sum_input = myAmps1_label.sum(axis='columns')

        kwargs = {name: cat13[name] for name, p in inspect.signature(
            T_Ridolfi2012).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

    else:
        amp_comps =calculate_23oxygens_amphibole(amp_comps=amp_comps)
        kwargs = {name: amp_comps[name] for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY}


    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:
        T_K=func(P, **kwargs)

    return T_K


## Function: PT Iterate Amphibole - only

def calculate_amp_only_press_temp(amp_comps, equationT, equationP, iterations=30, T_K_guess=1300):
    '''
    Solves simultaneous equations for temperature and pressure using
    amphibole only thermometers and barometers.


   Parameters
    -------

    amp_comps: DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    EquationP: str

        | P_Mutch2016 (T-independent)
        | P_Ridolfi2012_1a (T-independent)
        | P_Ridolfi2012_1b (T-independent)
        | P_Ridolfi2012_1c (T-independent)
        | P_Ridolfi2012_1d (T-independent)
        | P_Ridolfi2012_1e (T-independent)
        | P_Ridolfi2021 - (T-independent)- Uses new algorithm in 2021 paper to
        select pressures from equations 1a-e.

        | P_Ridolfi2010  (T-independent)
        | P_Hammerstrom1986_eq1  (T-independent)
        | P_Hammerstrom1986_eq2 (T-independent)
        | P_Hammerstrom1986_eq3 (T-independent)
        | P_Hollister1987 (T-independent)
        | P_Johnson1989 (T-independent)
        | P_Blundy1990 (T-independent)
        | P_Schmidt1992 (T-independent)
        | P_Anderson1995 (*T-dependent*)

    equationT: str
        |   T_Put2016_eq5 (P-independent)
        |   T_Put2016_eq6 (P-dependent)
        |   T_Put2016_SiHbl (P-independent)
        |   T_Ridolfi2012 (P-dependent)
        |   T_Put2016_eq8 (P-dependent)

    H2O_Liq: float, int, series, optional
        Needed if you select P_Put2008_eq32b, which is H2O-dependent.

    Optional:

     iterations: int, default=30
         Number of iterations used to converge to solution.

     T_K_guess: int or float. Default is 1300 K
         Initial guess of temperature.


    Returns:
    -------
    panda.dataframe: Pressure in Kbar, Temperature in K
    '''
    T_func = calculate_amp_only_temp(amp_comps=amp_comps, equationT=equationT, P="Solve")
    if equationP !="P_Ridolfi2021" and equationP != "P_Mutch2016":
        P_func = calculate_amp_only_press(amp_comps=amp_comps, equationP=equationP, T="Solve")
    else:
        P_func = calculate_amp_only_press(amp_comps=amp_comps, equationP=equationP, T="Solve").P_kbar_calc
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

        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)


    PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess, 'T_K_calc': T_K_guess})
    return PT_out

## Function: Amphibole-Liquid barometer
Amp_Liq_P_funcs = {P_Put2016_eq7a, P_Put2016_eq7b, P_Put2016_eq7c}

Amp_Liq_P_funcs_by_name = {p.__name__: p for p in Amp_Liq_P_funcs}


def calculate_amp_liq_press(*, amp_comps=None, liq_comps=None,
                            meltmatch=None, equationP=None, T=None,
                             eq_tests=False, H2O_Liq=None):
    '''
    Amphibole-liquid barometer. Returns pressure in kbar

   Parameters
    -------

    amp_comps: pandas DataFrame
        amphibole compositions (SiO2_Amp, TiO2_Amp etc.)

    liq_comps: pandas DataFrame
        liquid compositions (SiO2_Liq, TiO2_Liq etc.)

    equationP: str
        | P_Put2016_eq7a (T-independent, H2O-dependent)
        | P_Put2016_eq7b (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7c (T-independent, H2O-dependent (as hyd frac))

    P: float, int, series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series


    Eq_Test: bool. Default False
        If True, also calcualtes Kd Fe-Mg, which Putirka (2016) suggest
        as an equilibrium test.


    Returns
    -------
    pandas.core.series.Series (for simple barometers)
        Pressure in kbar
    pandas DataFrame for barometers like P_Ridolfi_2021, P_Mutch2016

    '''
    try:
        func = Amp_Liq_P_funcs_by_name[equationP]
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


    if meltmatch is not None:
        Combo_liq_amps = meltmatch
    if liq_comps is not None and amp_comps is not None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

        amp_comps_23 = calculate_23oxygens_amphibole(amp_comps=amp_comps)
        liq_comps_hy = calculate_hydrous_cat_fractions_liquid(
            liq_comps=liq_comps_c)
        liq_comps_an = calculate_anhydrous_cat_fractions_liquid(
            liq_comps=liq_comps_c)
        Combo_liq_amps = pd.concat(
            [amp_comps_23, liq_comps_hy, liq_comps_an], axis=1)


    kwargs = {name: Combo_liq_amps[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
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
        MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
        Kd=(MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/(liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd'])

        b = np.empty(len(MolProp), dtype=str)
        for i in range(0, len(MolProp)):

            if Kd[i] >= 0.17 and Kd[i] <= 0.39:
                b[i] = str("Yes")
            else:
                b[i] = str("No")
        Out=pd.DataFrame(data={'P_kbar_calc': P_kbar, 'Kd-Fe-Mg': Kd, "Eq Putirka 2016?": b})
    return Out
## Function: Amp-Liq temp

Amp_Liq_T_funcs = {T_Put2016_eq4b,  T_Put2016_eq4a_amp_sat, T_Put2016_eq9}

Amp_Liq_T_funcs_by_name = {p.__name__: p for p in Amp_Liq_T_funcs}

def calculate_amp_liq_temp(*, amp_comps=None, liq_comps=None, equationT=None,
P=None, H2O_Liq=None, eq_tests=False):
    '''
    Amphibole-liquid thermometers. Returns temperature in Kelvin.

   Parameters
    -------
    amp_comps: pandas DataFrame
        amphibole compositions (SiO2_Amp, TiO2_Amp etc.)

    liq_comps: pandas DataFrame
        liquid compositions (SiO2_Liq, TiO2_Liq etc.)

    equationT: str
        T_Put2016_eq4a_amp_sat (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq4b (P-independent, H2O-dep)
        T_Put2016_eq9 (P-independent, H2O-dep through hydrous fractions)

    P: float, int, series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Eq_Test: bool. Default False
        If True, also calcualtes Kd Fe-Mg, which Putirka (2016) suggest
        as an equilibrium test.


    Returns
    -------
    pandas.core.series.Series
        Temperature in Kelvin
    '''
    try:
        func = Amp_Liq_T_funcs_by_name[equationT]
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


    if liq_comps is not None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

    amp_comps_23 = calculate_23oxygens_amphibole(amp_comps=amp_comps)
    liq_comps_hy = calculate_hydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
    liq_comps_an = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
    Combo_liq_amps = pd.concat([amp_comps_23, liq_comps_hy, liq_comps_an], axis=1)
    kwargs = {name: Combo_liq_amps[name] for name, p in sig.parameters.items()
    if p.kind == inspect.Parameter.KEYWORD_ONLY}


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
        MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
        Kd=((MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/
        (liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd']))

        b = np.empty(len(MolProp), dtype=str)
        for i in range(0, len(MolProp)):

            if Kd[i] >= 0.17 and Kd[i] <= 0.39:
                b[i] = str("Yes")
            else:
                b[i] = str("No")
        Out=pd.DataFrame(data={'T_K_calc': T_K, 'Kd-Fe-Mg': Kd, "Eq Putirka 2016?": b})
    return Out

## Function for amphibole-liquid PT iter (although technically not needed)


def calculate_amp_liq_press_temp(liq_comps, amp_comps, equationT, equationP, iterations=30,
T_K_guess=1300, H2O_Liq=None, eq_tests=False):
    '''
    Solves simultaneous equations for temperature and pressure using
    amphibole only thermometers and barometers.


   Parameters
    -------

    amp_comps: DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    equationP: str
        | P_Put2016_eq7a (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7b (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7c (T-independent, H2O-dependent (as hyd frac))

    equationT: str
        T_Put2016_eq4a_amp_sat (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq4b (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq9 (P-independent, H2O-dep through hydrous fractions)


    H2O_Liq: float, int, series, optional
        Needed if you select P_Put2008_eq32b, which is H2O-dependent.

    Optional:

    iterations: int, default=30
         Number of iterations used to converge to solution.

    T_K_guess: int or float. Default is 1300 K
         Initial guess of temperature.

    Eq_Test: bool. Default False
        If True, also calcualtes Kd Fe-Mg, which Putirka (2016) suggest
        as an equilibrium test.



    Returns:
    -------
    panda.dataframe: Pressure in Kbar, Temperature in K, Kd-Fe-Mg if Eq_Test=True
    '''
    liq_comps_c=liq_comps.copy()
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq']=H2O_Liq

    T_func = calculate_amp_liq_temp(liq_comps=liq_comps_c,
    amp_comps=amp_comps, equationT=equationT, P="Solve")

    P_func = calculate_amp_liq_press(liq_comps=liq_comps_c,
    amp_comps=amp_comps, equationP=equationP, T="Solve")

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

        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)


    PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess, 'T_K_calc': T_K_guess})

    if eq_tests is False:
        return PT_out
    if eq_tests is True:
        liq_comps_hy = calculate_hydrous_cat_fractions_liquid(
            liq_comps=liq_comps_c)
        MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
        Kd=(MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/(liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd'])
        PT_out['Kd-Fe-Mg']=Kd

        b = np.empty(len(MolProp), dtype=str)
        for i in range(0, len(MolProp)):

            if Kd[i] >= 0.17 and Kd[i] <= 0.39:
                b[i] = str("Yes")
            else:
                b[i] = str("No")

        PT_out["Eq Putirka 2016?"]=b

    return PT_out
