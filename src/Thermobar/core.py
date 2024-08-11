import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
from Thermobar.import_export import *
from pickle import load
import pickle
from pathlib import Path
Thermobar_dir=Path(__file__).parent
np.seterr(divide='ignore', invalid='ignore')



## This specifies the default order for each dataframe type used in calculations
df_ideal_liq = pd.DataFrame(columns=['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq',
'FeOt_Liq', 'MnO_Liq', 'MgO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq',
'Cr2O3_Liq', 'P2O5_Liq', 'H2O_Liq', 'Fe3Fet_Liq', 'NiO_Liq', 'CoO_Liq',
 'CO2_Liq'])



df_ideal_oxide = pd.DataFrame(columns=['SiO2', 'TiO2', 'Al2O3',
'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O',
'Cr2O3', 'P2O5'])

df_ideal_cpx = pd.DataFrame(columns=['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx',
'FeOt_Cpx','MnO_Cpx', 'MgO_Cpx', 'CaO_Cpx', 'Na2O_Cpx', 'K2O_Cpx',
'Cr2O3_Cpx'])

df_ideal_ol = pd.DataFrame(columns=['SiO2_Ol', 'TiO2_Ol', 'Al2O3_Ol',
'FeOt_Ol', 'MnO_Ol', 'MgO_Ol', 'CaO_Ol', 'Na2O_Ol', 'K2O_Ol', 'Cr2O3_Ol',
'NiO_Ol'])

df_ideal_sp = pd.DataFrame(columns=['SiO2_Sp', 'TiO2_Sp', 'Al2O3_Sp',
'FeOt_Sp', 'MnO_Sp', 'MgO_Sp', 'CaO_Sp', 'Na2O_Sp', 'K2O_Sp', 'Cr2O3_Sp',
'NiO_Sp'])

df_ideal_opx = pd.DataFrame(columns=['SiO2_Opx', 'TiO2_Opx', 'Al2O3_Opx',
'FeOt_Opx', 'MnO_Opx', 'MgO_Opx', 'CaO_Opx', 'Na2O_Opx', 'K2O_Opx',
'Cr2O3_Opx'])

df_ideal_gt = pd.DataFrame(columns=['SiO2_Gt', 'TiO2_Gt', 'Al2O3_Gt',
'Cr2O3_Gt', 'FeOt_Gt', 'MnO_Gt', 'MgO_Gt', 'CaO_Gt', 'K2O_Gt', 'Na2O_Gt', 'NiO_Gt',
'Ni_Gt', 'Ti_Gt', 'Zr_Gt', 'Zn_Gt', 'Ga_Gt', 'Sr_Gt', 'Y_Gt'])

df_ideal_plag = pd.DataFrame(columns=['SiO2_Plag', 'TiO2_Plag', 'Al2O3_Plag',
'FeOt_Plag', 'MnO_Plag', 'MgO_Plag', 'CaO_Plag', 'Na2O_Plag', 'K2O_Plag',
'Cr2O3_Plag'])

df_ideal_kspar = pd.DataFrame(columns=['SiO2_Kspar', 'TiO2_Kspar',
 'Al2O3_Kspar', 'FeOt_Kspar','MnO_Kspar', 'MgO_Kspar', 'CaO_Kspar',
 'Na2O_Kspar', 'K2O_Kspar', 'Cr2O3_Kspar'])

df_ideal_amp = pd.DataFrame(columns=['SiO2_Amp', 'TiO2_Amp', 'Al2O3_Amp',
 'FeOt_Amp', 'MnO_Amp', 'MgO_Amp', 'CaO_Amp', 'Na2O_Amp', 'K2O_Amp',
 'Cr2O3_Amp', 'F_Amp', 'Cl_Amp'])

# Same, but order for errors
df_ideal_liq_Err = pd.DataFrame(columns=['SiO2_Liq_Err', 'TiO2_Liq_Err',
'Al2O3_Liq_Err', 'FeOt_Liq_Err', 'MnO_Liq_Err', 'MgO_Liq_Err', 'CaO_Liq_Err',
'Na2O_Liq_Err', 'K2O_Liq_Err', 'Cr2O3_Liq_Err', 'P2O5_Liq_Err', 'H2O_Liq_Err',
 'Fe3Fet_Liq_Err', 'NiO_Liq_Err', 'CoO_Liq_Err', 'CO2_Liq_Err'])

df_ideal_cpx_Err = pd.DataFrame(columns=['SiO2_Cpx_Err', 'TiO2_Cpx_Err',
 'Al2O3_Cpx_Err', 'FeOt_Cpx_Err', 'MnO_Cpx_Err', 'MgO_Cpx_Err', 'CaO_Cpx_Err',
  'Na2O_Cpx_Err', 'K2O_Cpx_Err', 'Cr2O3_Cpx_Err', 'P_kbar_Err', 'T_K_Err'])

df_ideal_ol_Err = pd.DataFrame(columns=['SiO2_Ol_Err', 'TiO2_Ol_Err',
'Al2O3_Ol_Err', 'FeOt_Ol_Err', 'MnO_Ol_Err', 'MgO_Ol_Err', 'CaO_Ol_Err',
 'Na2O_Ol_Err', 'K2O_Ol_Err', 'Cr2O3_Ol_Err', 'NiO_Ol_Err', 'P_kbar_Err',
 'T_K_Err'])

df_ideal_sp_Err = pd.DataFrame(columns=['SiO2_Sp_Err', 'TiO2_Sp_Err',
'Al2O3_Sp_Err', 'FeOt_Sp_Err', 'MnO_Sp_Err', 'MgO_Sp_Err', 'CaO_Sp_Err',
'Na2O_Sp_Err', 'K2O_Sp_Err', 'Cr2O3_Sp_Err', 'NiO_Sp_Err', 'P_kbar_Err',
'T_K_Err'])

df_ideal_opx_Err = pd.DataFrame(columns=['SiO2_Opx_Err', 'TiO2_Opx_Err',
'Al2O3_Opx_Err', 'FeOt_Opx_Err', 'MnO_Opx_Err', 'MgO_Opx_Err', 'CaO_Opx_Err',
 'Na2O_Opx_Err', 'K2O_Opx_Err', 'Cr2O3_Opx_Err', 'P_kbar_Err', 'T_K_Err'])

df_ideal_Gt_Err = pd.DataFrame(columns = ['SiO2_Gt_Err', 'TiO2_Gt_Err', 'Al2O3_Gt_Err',
'Cr2O3_Gt_Err', 'FeOt_Gt_Err', 'MnO_Gt_Err', 'MgO_Gt_Err', 'CaO_Gt_Err', 'K2O_Gt_Err', 'Na2O_Gt_Err', 'NiO_Gt_Err',
'Ni_Gt_Err', 'Ti_Gt_Err', 'Zr_Gt_Err', 'Zn_Gt_Err', 'Ga_Gt_Err', 'Sr_Gt_Err', 'Y_Gt_Err'])

df_ideal_plag_Err = pd.DataFrame(columns=['SiO2_Plag_Err', 'TiO2_Plag_Err',
'Al2O3_Plag_Err', 'FeOt_Plag_Err', 'MnO_Plag_Err', 'MgO_Plag_Err',
'CaO_Plag_Err', 'Na2O_Plag_Err', 'K2O_Plag_Err', 'Cr2O3_Plag_Err',
'P_kbar_Err', 'T_K_Err'])

df_ideal_kspar_Err = pd.DataFrame(columns=['SiO2_Kspar_Err', 'TiO2_Kspar_Err',
'Al2O3_Kspar_Err', 'FeOt_Kspar_Err', 'MnO_Kspar_Err', 'MgO_Kspar_Err',
'CaO_Kspar_Err', 'Na2O_Kspar_Err', 'K2O_Kspar_Err', 'Cr2O3_Kspar_Err',
 'P_kbar_Err', 'T_K_Err'])

df_ideal_amp_Err = pd.DataFrame(columns=['SiO2_Amp_Err', 'TiO2_Amp_Err',
 'Al2O3_Amp_Err', 'FeOt_Amp_Err', 'MnO_Amp_Err', 'MgO_Amp_Err',
'CaO_Amp_Err', 'Na2O_Amp_Err', 'K2O_Amp_Err', 'Cr2O3_Amp_Err',
'F_Amp_Err', 'Cl_Amp_Err', 'P_kbar_Err', 'T_K_Err'])

# Used to store variables.


df_ideal_exp = pd.DataFrame(columns=['P_kbar', 'T_K'])
df_ideal_exp_Err = pd.DataFrame(columns=['P_kbar_Err', 'T_K_Err'])


# Hydrous Liquids: Specifying Cation numbers, oxide masses etc.
cation_num_liq_hyd = {'SiO2_Liq': 1, 'MgO_Liq': 1, 'MnO_Liq': 1, 'FeOt_Liq': 1,
 'CaO_Liq': 1, 'Al2O3_Liq': 2, 'Na2O_Liq': 2, 'K2O_Liq': 2, 'TiO2_Liq': 1,
  'P2O5_Liq': 2, 'Cr2O3_Liq': 2, 'H2O_Liq': 2}
cation_num_liq_hyd_df = pd.DataFrame.from_dict(
    cation_num_liq_hyd, orient='index').T
cation_num_liq_hyd_df['Sample_ID_Liq'] = 'CatNum'
cation_num_liq_hyd_df.set_index('Sample_ID_Liq', inplace=True)

oxide_mass_liq_hyd = {'SiO2_Liq': 60.0843, 'MgO_Liq': 40.3044,
'MnO_Liq': 70.9375, 'FeOt_Liq': 71.844, 'CaO_Liq': 56.0774,
'Al2O3_Liq': 101.961,'Na2O_Liq': 61.9789, 'K2O_Liq': 94.196,
 'TiO2_Liq': 79.8788, 'P2O5_Liq': 141.937, 'Cr2O3_Liq': 151.9982,
  'H2O_Liq': 18.01528}
# Turns dictionary into a dataframe so pandas matrix math functions can be used
oxide_mass_liq_hyd_df = pd.DataFrame.from_dict(
    oxide_mass_liq_hyd, orient='index').T
oxide_mass_liq_hyd_df['Sample_ID_Liq'] = 'MolWt'
oxide_mass_liq_hyd_df.set_index('Sample_ID_Liq', inplace=True)

# Anydrous Liquids: Specifying Cation numbers, oxide masses etc.

cation_num_liq_anhyd = {'SiO2_Liq': 1, 'MgO_Liq': 1, 'MnO_Liq': 1,
'FeOt_Liq': 1, 'CaO_Liq': 1, 'Al2O3_Liq': 2, 'Na2O_Liq': 2,
'K2O_Liq': 2, 'TiO2_Liq': 1, 'P2O5_Liq': 2, 'Cr2O3_Liq': 2}

# Turns dictionary into a dataframe so pandas matrix math functions can be used
cation_num_liq_anhyd_df = pd.DataFrame.from_dict(
    cation_num_liq_anhyd, orient='index').T
cation_num_liq_anhyd_df['Sample_ID_Liq'] = 'CatNum'
cation_num_liq_anhyd_df.set_index('Sample_ID_Liq', inplace=True)

oxide_mass_liq_anhyd = {'SiO2_Liq': 60.0843, 'MgO_Liq': 40.3044,
'MnO_Liq': 70.9375, 'FeOt_Liq': 71.8464, 'CaO_Liq': 56.0774,
'Al2O3_Liq': 101.961, 'Na2O_Liq': 61.9789, 'K2O_Liq': 94.196,
'TiO2_Liq': 79.8788, 'P2O5_Liq': 141.937, 'Cr2O3_Liq': 151.9982}
# Turns dictionary into a dataframe so pandas matrix math functions can be used
oxide_mass_liq_anhyd_df = pd.DataFrame.from_dict(
    oxide_mass_liq_anhyd, orient='index').T
oxide_mass_liq_anhyd_df['Sample_ID_Liq'] = 'MolWt'
oxide_mass_liq_anhyd_df.set_index('Sample_ID_Liq', inplace=True)

# Component for Ni and Co too for the Pu et al. 2017, and 2019 thermometers

oxide_mass_liq_anhyd_Ni = {'SiO2_Liq': 60.0843, 'MgO_Liq': 40.3044,
'MnO_Liq': 70.9375, 'FeOt_Liq': 71.8464, 'CaO_Liq': 56.0774,
'Al2O3_Liq': 101.961,'Na2O_Liq': 61.9789, 'K2O_Liq': 94.196,
'TiO2_Liq': 79.8788, 'P2O5_Liq': 141.937, 'Cr2O3_Liq': 151.9982,
'NiO_Liq': 74.692, 'CoO_Liq': 74.932}

# Turns dictionary into a dataframe so pandas matrix math functions can be used
oxide_mass_liq_anhyd_df_Ni = pd.DataFrame.from_dict(
    oxide_mass_liq_anhyd_Ni, orient='index').T
oxide_mass_liq_anhyd_df_Ni['Sample_ID_Liq'] = 'MolWt'
oxide_mass_liq_anhyd_df_Ni.set_index('Sample_ID_Liq', inplace=True)


# Olivines: Specifying Cation numbers, oxide masses etc.
cation_num_ol = {'SiO2_Ol': 1, 'MgO_Ol': 1, 'FeOt_Ol': 1, 'CaO_Ol': 1,
                 'Al2O3_Ol': 2, 'Na2O_Ol': 2, 'K2O_Ol': 2, 'MnO_Ol': 1,
                 'TiO2_Ol': 1, 'P2O5_Ol': 2}
cation_num_ol_df = pd.DataFrame.from_dict(cation_num_ol, orient='index').T
cation_num_ol_df['Sample_ID_Ol'] = 'CatNum'
cation_num_ol_df.set_index('Sample_ID_Ol', inplace=True)

oxide_mass_ol = {'SiO2_Ol': 60.0843, 'MgO_Ol': 40.3044, 'FeOt_Ol': 71.8464,
'CaO_Ol': 56.0774,'Al2O3_Ol': 101.961, 'Na2O_Ol': 61.9789, 'K2O_Ol': 94.196,
 'MnO_Ol': 70.9375, 'TiO2_Ol': 79.7877}
oxide_mass_ol_df = pd.DataFrame.from_dict(oxide_mass_ol, orient='index').T
oxide_mass_ol_df['Sample_ID_Ol'] = 'MolWt'
oxide_mass_ol_df.set_index('Sample_ID_Ol', inplace=True)

# Things for olivine using the Ni thermometers of Pu et al. 2017, 2019
oxide_mass_ol_Ni = {'SiO2_Ol': 60.0843, 'MgO_Ol': 40.3044, 'FeOt_Ol': 71.8464,
'CaO_Ol': 56.0774, 'Al2O3_Ol': 101.961, 'Na2O_Ol': 61.9789, 'K2O_Ol': 94.196,
'MnO_Ol': 70.9375, 'TiO2_Ol': 79.7877, 'NiO_Ol': 74.692}
oxide_mass_ol_df_Ni = pd.DataFrame.from_dict(
    oxide_mass_ol_Ni, orient='index').T
oxide_mass_ol_df_Ni['Sample_ID_Ol'] = 'MolWt'
oxide_mass_ol_df_Ni.set_index('Sample_ID_Ol', inplace=True)

# Clinopyrones: Specifying Cation numbers, oxide masses etc.
cation_num_cpx = {'SiO2_Cpx': 1, 'MgO_Cpx': 1, 'FeOt_Cpx': 1, 'CaO_Cpx': 1,
'Al2O3_Cpx': 2, 'Na2O_Cpx': 2, 'K2O_Cpx': 2, 'MnO_Cpx': 1,'TiO2_Cpx': 1,
'Cr2O3_Cpx': 2}
cation_num_cpx_df = pd.DataFrame.from_dict(cation_num_cpx, orient='index').T
cation_num_cpx_df['Sample_ID_Cpx'] = 'CatNum'
cation_num_cpx_df.set_index('Sample_ID_Cpx', inplace=True)
# From Putirka Cpx-liq spreadsheet for consistency
oxide_mass_cpx = {'SiO2_Cpx': 60.0843, 'MgO_Cpx': 40.3044, 'FeOt_Cpx': 71.8464,
 'CaO_Cpx': 56.0774, 'Al2O3_Cpx': 101.961, 'Na2O_Cpx': 61.9789,
'K2O_Cpx': 94.196, 'MnO_Cpx': 70.9375, 'TiO2_Cpx': 79.8788,
'Cr2O3_Cpx': 151.9982}

oxide_mass_cpx_df = pd.DataFrame.from_dict(oxide_mass_cpx, orient='index').T
oxide_mass_cpx_df['Sample_ID_Cpx'] = 'MolWt'
oxide_mass_cpx_df.set_index('Sample_ID_Cpx', inplace=True)


oxygen_num_cpx = {'SiO2_Cpx': 2, 'MgO_Cpx': 1, 'FeOt_Cpx': 1, 'CaO_Cpx': 1,
'Al2O3_Cpx': 3, 'Na2O_Cpx': 1, 'K2O_Cpx': 1, 'MnO_Cpx': 1, 'TiO2_Cpx': 2,
'Cr2O3_Cpx': 3}
oxygen_num_cpx_df = pd.DataFrame.from_dict(oxygen_num_cpx, orient='index').T
oxygen_num_cpx_df['Sample_ID_Cpx'] = 'OxNum'
oxygen_num_cpx_df.set_index('Sample_ID_Cpx', inplace=True)

# Orthopyroxenes: Specifying Cation numbers, oxide masses etc.
cation_num_opx = {'SiO2_Opx': 1, 'MgO_Opx': 1, 'FeOt_Opx': 1, 'CaO_Opx': 1,
'Al2O3_Opx': 2, 'Na2O_Opx': 2, 'K2O_Opx': 2, 'MnO_Opx': 1, 'TiO2_Opx': 1,
'Cr2O3_Opx': 2}
cation_num_opx_df = pd.DataFrame.from_dict(cation_num_opx, orient='index').T
cation_num_opx_df['Sample_ID_Opx'] = 'CatNum'
cation_num_opx_df.set_index('Sample_ID_Opx', inplace=True)

# From Putirka Opx-liq spreadsheet for consistency
oxide_mass_opx = {'SiO2_Opx': 60.0843, 'MgO_Opx': 40.3044, 'FeOt_Opx': 71.8464,
 'CaO_Opx': 56.0774, 'Al2O3_Opx': 101.961,'Na2O_Opx': 61.9789,
 'K2O_Opx': 94.196, 'MnO_Opx': 70.9375, 'TiO2_Opx': 79.8788,
 'Cr2O3_Opx': 151.9982}
oxide_mass_opx_df = pd.DataFrame.from_dict(oxide_mass_opx, orient='index').T
oxide_mass_opx_df['Sample_ID_Opx'] = 'MolWt'
oxide_mass_opx_df.set_index('Sample_ID_Opx', inplace=True)


oxygen_num_opx = {'SiO2_Opx': 2, 'MgO_Opx': 1, 'FeOt_Opx': 1, 'CaO_Opx': 1,
                  'Al2O3_Opx': 3, 'Na2O_Opx': 1, 'K2O_Opx': 1, 'MnO_Opx': 1, 'TiO2_Opx': 2, 'Cr2O3_Opx': 3}
oxygen_num_opx_df = pd.DataFrame.from_dict(oxygen_num_opx, orient='index').T
oxygen_num_opx_df['Sample_ID_Opx'] = 'OxNum'
oxygen_num_opx_df.set_index('Sample_ID_Opx', inplace=True)

#Pyrope Garnet

oxide_mass_gt = {'SiO2_Gt': 60.0843, 'MgO_Gt': 40.3044, 'FeOt_Gt': 71.8464,
 'CaO_Gt': 56.0774, 'Al2O3_Gt': 101.961,'Na2O_Gt': 61.9789,
 'K2O_Gt': 94.196, 'MnO_Gt': 70.9375, 'TiO2_Gt': 79.8788,
 'Cr2O3_Gt': 151.9982, 'Ni_Gt':  74.6994}

oxide_mass_gt_df = pd.DataFrame.from_dict(oxide_mass_gt, orient='index').T
oxide_mass_gt_df['Sample_ID_Gt'] = 'MolWt'
oxide_mass_gt_df.set_index('Sample_ID_Gt', inplace=True)

oxygen_num_gt = {'SiO2_Gt': 2, 'MgO_Gt': 1, 'FeOt_Gt': 1, 'CaO_Gt': 1,
'Al2O3_Gt': 3, 'Na2O_Gt': 1, 'K2O_Gt': 1, 'MnO_Gt': 1, 'TiO2_Gt': 2,
'Cr2O3_Gt': 3, 'Ni_Gt': 1}
oxygen_num_gt_df = pd.DataFrame.from_dict(oxygen_num_gt, orient = 'index').T
oxygen_num_gt_df['Sample_ID_Gt'] = 'OxNum'
oxygen_num_gt_df.set_index('Sample_ID_Gt', inplace=True)

cation_num_gt = {'SiO2_Gt': 1, 'MgO_Gt': 1, 'FeOt_Gt': 1, 'CaO_Gt': 1,
'Al2O3_Gt': 2, 'Na2O_Gt': 2, 'K2O_Gt': 2, 'MnO_Gt': 1, 'TiO2_Gt': 1,
'Cr2O3_Gt': 2, 'Ni_Gt': 1}
cation_num_gt_df = pd.DataFrame.from_dict(cation_num_gt, orient = 'index').T
cation_num_gt_df['Sample_ID_Gt'] = 'CatNum'
cation_num_gt_df.set_index('Sample_ID_Gt', inplace=True)

# Plagioclase: Specifying Cation numbers, oxide masses etc.
cation_num_plag = {'SiO2_Plag': 1, 'MgO_Plag': 1, 'FeOt_Plag': 1, 'CaO_Plag': 1, 'Al2O3_Plag': 2, 'Na2O_Plag': 2,
                   'K2O_Plag': 2, 'MnO_Plag': 1, 'TiO2_Plag': 1, 'Cr2O3_Plag': 2}

cation_num_plag_df = pd.DataFrame.from_dict(cation_num_plag, orient='index').T
cation_num_plag_df['Sample_ID_Plag'] = 'CatNum'
cation_num_plag_df.set_index('Sample_ID_Plag', inplace=True)

oxide_mass_plag = {'SiO2_Plag': 60.0843, 'MgO_Plag': 40.3044, 'FeOt_Plag': 71.8464, 'CaO_Plag': 56.0774, 'Al2O3_Plag': 101.961,
                   'Na2O_Plag': 61.9789, 'K2O_Plag': 94.196, 'MnO_Plag': 70.9375, 'TiO2_Plag': 79.8788, 'Cr2O3_Plag': 151.9982}
oxide_mass_plag_df = pd.DataFrame.from_dict(oxide_mass_plag, orient='index').T
oxide_mass_plag_df['Sample_ID_Plag'] = 'MolWt'
oxide_mass_plag_df.set_index('Sample_ID_Plag', inplace=True)


oxygen_num_plag = {'SiO2_Plag': 2, 'MgO_Plag': 1, 'FeOt_Plag': 1, 'CaO_Plag': 1, 'Al2O3_Plag': 3, 'Na2O_Plag': 1,
                   'K2O_Plag': 1, 'MnO_Plag': 1, 'TiO2_Plag': 2, 'Cr2O3_Plag': 3}
oxygen_num_plag_df = pd.DataFrame.from_dict(oxygen_num_plag, orient='index').T
oxygen_num_plag_df['Sample_ID_Plag'] = 'OxNum'
oxygen_num_plag_df.set_index('Sample_ID_Plag', inplace=True)

# Alkali Feldspar: Specifying Cation numbers, oxide masses etc.
cation_num_kspar = {'SiO2_Kspar': 1, 'MgO_Kspar': 1, 'FeOt_Kspar': 1, 'CaO_Kspar': 1, 'Al2O3_Kspar': 2, 'Na2O_Kspar': 2,
                    'K2O_Kspar': 2, 'MnO_Kspar': 1, 'TiO2_Kspar': 1, 'Cr2O3_Kspar': 2}

cation_num_kspar_df = pd.DataFrame.from_dict(
    cation_num_kspar, orient='index').T
cation_num_kspar_df['Sample_ID_Kspar'] = 'CatNum'
cation_num_kspar_df.set_index('Sample_ID_Kspar', inplace=True)

oxide_mass_kspar = {'SiO2_Kspar': 60.0843, 'MgO_Kspar': 40.3044, 'FeOt_Kspar': 71.8464, 'CaO_Kspar': 56.0774, 'Al2O3_Kspar': 101.961,
                    'Na2O_Kspar': 61.9789, 'K2O_Kspar': 94.196, 'MnO_Kspar': 70.9375, 'TiO2_Kspar': 79.8788, 'Cr2O3_Kspar': 151.9982}
oxide_mass_kspar_df = pd.DataFrame.from_dict(
    oxide_mass_kspar, orient='index').T
oxide_mass_kspar_df['Sample_ID_Kspar'] = 'MolWt'
oxide_mass_kspar_df.set_index('Sample_ID_Kspar', inplace=True)


oxygen_num_kspar = {'SiO2_Kspar': 2, 'MgO_Kspar': 1, 'FeOt_Kspar': 1, 'CaO_Kspar': 1, 'Al2O3_Kspar': 3, 'Na2O_Kspar': 1,
                    'K2O_Kspar': 1, 'MnO_Kspar': 1, 'TiO2_Kspar': 2, 'Cr2O3_Kspar': 3}
oxygen_num_kspar_df = pd.DataFrame.from_dict(
    oxygen_num_kspar, orient='index').T
oxygen_num_kspar_df['Sample_ID_Kspar'] = 'OxNum'
oxygen_num_kspar_df.set_index('Sample_ID_Kspar', inplace=True)


# Amphiboles specifying cation umbers etc.
cation_num_amp = {'SiO2_Amp': 1, 'MgO_Amp': 1, 'FeOt_Amp': 1, 'CaO_Amp': 1, 'Al2O3_Amp': 2, 'Na2O_Amp': 2,
                  'K2O_Amp': 2, 'MnO_Amp': 1, 'TiO2_Amp': 1, 'Cr2O3_Amp': 2}


cation_num_amp_df = pd.DataFrame.from_dict(cation_num_amp, orient='index').T
cation_num_amp_df['Sample_ID_Amp'] = 'CatNum'
cation_num_amp_df.set_index('Sample_ID_Amp', inplace=True)

oxide_mass_amp = {'SiO2_Amp': 60.0843, 'MgO_Amp': 40.3044, 'FeOt_Amp': 71.8464, 'CaO_Amp': 56.0774, 'Al2O3_Amp': 101.961,
                  'Na2O_Amp': 61.9789, 'K2O_Amp': 94.196, 'MnO_Amp': 70.9375, 'TiO2_Amp': 79.8788, 'Cr2O3_Amp': 151.9982}
oxide_mass_amp_df = pd.DataFrame.from_dict(oxide_mass_amp, orient='index').T
oxide_mass_amp_df['Sample_ID_Amp'] = 'MolWt'
oxide_mass_amp_df.set_index('Sample_ID_Amp', inplace=True)


oxygen_num_amp = {'SiO2_Amp': 2, 'MgO_Amp': 1, 'FeOt_Amp': 1, 'CaO_Amp': 1, 'Al2O3_Amp': 3, 'Na2O_Amp': 1,
                  'K2O_Amp': 1, 'MnO_Amp': 1, 'TiO2_Amp': 2, 'Cr2O3_Amp': 3}
oxygen_num_amp_df = pd.DataFrame.from_dict(oxygen_num_amp, orient='index').T
oxygen_num_amp_df['Sample_ID_Amp'] = 'OxNum'
oxygen_num_amp_df.set_index('Sample_ID_Amp', inplace=True)

# Ridolfi amphiboles, with F and Cl
cation_num_amp_Ridolfi = {'SiO2_Amp': 1, 'MgO_Amp': 1, 'FeOt_Amp': 1, 'CaO_Amp': 1, 'Al2O3_Amp': 2, 'Na2O_Amp': 2,
                          'K2O_Amp': 2, 'MnO_Amp': 1, 'TiO2_Amp': 1, 'Cr2O3_Amp': 2, 'F_Amp': 1, 'Cl_Amp': 1}

cation_num_amp_df_Ridolfi = pd.DataFrame.from_dict(
    cation_num_amp_Ridolfi, orient='index').T
cation_num_amp_df_Ridolfi['Sample_ID_Amp'] = 'CatNum'
cation_num_amp_df_Ridolfi.set_index('Sample_ID_Amp', inplace=True)

oxide_mass_amp_Ridolfi = {'SiO2_Amp': 60.084, 'MgO_Amp': 40.304, 'FeOt_Amp': 71.846, 'CaO_Amp': 56.079, 'Al2O3_Amp': 101.961,
                          'Na2O_Amp': 61.979, 'K2O_Amp': 94.195, 'MnO_Amp': 70.937, 'TiO2_Amp': 79.898, 'Cr2O3_Amp': 151.9902, 'F_Amp': 18.998, 'Cl_Amp': 35.453}
oxide_mass_amp_df_Ridolfi = pd.DataFrame.from_dict(
    oxide_mass_amp_Ridolfi, orient='index').T
oxide_mass_amp_df_Ridolfi['Sample_ID_Amp'] = 'MolWt'
oxide_mass_amp_df_Ridolfi.set_index('Sample_ID_Amp', inplace=True)

# For oxide to wt% function
oxide_mass_all = {'SiO2': 60.084, 'MgO': 40.304, 'FeOt': 71.846, 'CaO': 56.079, 'Al2O3': 101.961,
                          'Na2O': 61.979, 'K2O': 94.195, 'MnO': 70.937, 'TiO2': 79.898,
                          'Cr2O3': 151.9902, 'P2O5':141.944, 'F': 18.998, 'Cl': 35.453, 'H2O': 18.01528}
oxide_mass_all = pd.DataFrame.from_dict(
    oxide_mass_all, orient='index').T
oxide_mass_all['Sample_ID'] = 'MolWt'
oxide_mass_all.set_index('Sample_ID', inplace=True)

elemental_mass_mult_all = {'SiO2': 28.0855, 'MgO': 24.305, 'FeOt': 55.845, 'CaO': 40.078, 'Al2O3': 26.981539*2,
                          'Na2O': 22.989769*2, 'K2O': 39.0983*2, 'MnO': 54.938044, 'TiO2': 47.867,
                          'Cr2O3': 51.9961*2, 'P2O5': 2*30.973762, 'F': 18.998, 'Cl': 35.453, 'H2O':1.00794*2}
elemental_mass_mult_all = pd.DataFrame.from_dict(
    elemental_mass_mult_all, orient='index').T
elemental_mass_mult_all['Sample_ID'] = 'ElWt'
elemental_mass_mult_all.set_index('Sample_ID', inplace=True)



df_ideal_all2 = pd.DataFrame(columns=['SiO2', 'TiO2', 'Al2O3',
'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O',
'Cr2O3', 'P2O5', 'F', 'Cl', 'H2O'])



def convert_oxide_percent_to_element_weight_percent(df, suffix=None,
 without_oxygen=False, anhydrous=True):
    """
    Converts oxide wt% to elemental wt% including oxygen by default

   Parameters
    -------

    df: pandas.DataFrame
        Data frame of oxide compositions. Can have suffixes like "_Amp"
        in which case you need to specify suffix="_Amp"

    without_oxygen: str
        default False, element wt% doesnt sum to 100.]
        if true, all elements sum to 100 w/o oxygen.

    returns: pandas.DataFrame
    wt% of elements
    """
    df_c=df.copy()
    if suffix is not None:
        df_c.columns = df_c.columns.str.rstrip(suffix)

    if anhydrous==True:
        df_c['H2O']=0


    df_oxides=df_c.reindex(df_ideal_all2.columns, axis=1).fillna(0)

    liq_wt_combo = pd.concat([oxide_mass_all, df_oxides],)


    mol_prop_anhyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])

    el_combo=pd.concat([elemental_mass_mult_all, mol_prop_anhyd ],)
    wt_perc = el_combo.multiply(
        el_combo.loc['ElWt', :], axis='columns').drop(['ElWt'])


    wt_perc2=pd.DataFrame(data={'Si_wt': wt_perc['SiO2'],
                                'Mg_wt': wt_perc['MgO'],
                                'Fe_wt':wt_perc['FeOt'],
                                'Ca_wt':wt_perc['CaO'],
                                'Al_wt':wt_perc['Al2O3'],
                                'Na_wt':wt_perc['Na2O'],
                                'K_wt':wt_perc['K2O'],
                                'Mn_wt':wt_perc['MnO'],
                                'Ti_wt':wt_perc['TiO2'],
                                'Cr_wt':wt_perc['Cr2O3'],
                                'P_wt':wt_perc['P2O5'],
                                'F_wt':wt_perc['F'],
                                'H_wt': wt_perc['H2O'],
                                'Cl_wt':wt_perc['Cl']


                                })
    sum_element=wt_perc2.sum(axis=1)





    if without_oxygen is True:
        wt_perc3=wt_perc2.div(sum_element/100,  axis=0)
        wt_perc3=wt_perc3.add_suffix('_noO2')
        return wt_perc3
    if without_oxygen is False:
        Oxy=100-sum_element
        wt_perc2['O_wt_make_to_100']=Oxy
        return wt_perc2









## Anhydrous mole proportions, mole fractions, cation proportions, cation fractions

def calculate_anhydrous_mol_proportions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        Panda DataFrame of liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the form SiO2_Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = liq_comps.reindex(
        oxide_mass_liq_anhyd_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_anhyd_df, liq_wt],)
    # Drop the calculation column
    mol_prop_anhyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_anhydrous_mol_fractions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    liq_comps: pandas.DataFrame
                Panda DataFrame of liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_anhydrous_mol_proportions_liquid(liq_comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd


def calculate_anhydrous_cat_proportions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous cation proportions (e.g., mole proportions * no of cations)

   Parameters
    -------

    liq_comps: pandas.DataFrame
                Panda DataFrame of liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.


    Returns
    -------
    pandas DataFrame
        anhydrous cation proportions for the liquid with column headings of the form S_Liq_cat_prop

    '''
    mol_prop_no_cat_num = calculate_anhydrous_mol_proportions_liquid(liq_comps)
    mol_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in mol_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_liq_anhyd_df.reindex(
        oxide_mass_liq_anhyd_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]



    return cation_prop_anhyd



def calculate_anhydrous_cat_fractions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous cation fractions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.


    Returns
    -------
    pandas DataFrame
        anhydrous cation fractions for the liquid with column headings of the form _Liq_cat_frac,
        as well as the initial dataframe of liquid compositions.


    '''
    cat_prop = calculate_anhydrous_cat_proportions_liquid(liq_comps=liq_comps)
    mol_prop = calculate_anhydrous_mol_fractions_liquid(liq_comps=liq_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd = pd.concat([liq_comps, mol_prop, cat_frac_anhyd], axis=1)

    if "Fe3Fet_Liq" in cat_frac_anhyd:
        cat_frac_anhyd['Mg_Number_Liq_NoFe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] / 71.844))
        cat_frac_anhyd['Mg_Number_Liq_Fe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] * (1 - cat_frac_anhyd['Fe3Fet_Liq']) / 71.844))
    if "Fe3Fet_Liq" not in cat_frac_anhyd:
        cat_frac_anhyd['Mg_Number_Liq_Fe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] / 71.844))
        cat_frac_anhyd['Mg_Number_Liq_NoFe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] / 71.844))


    cation_frac_anhyd2=cat_frac_anhyd.rename(columns={
                        'SiO2_Liq_cat_frac': 'Si_Liq_cat_frac',
                        'TiO2_Liq_cat_frac': 'Ti_Liq_cat_frac',
                        'Al2O3_Liq_cat_frac': 'Al_Liq_cat_frac',
                        'FeOt_Liq_cat_frac': 'Fet_Liq_cat_frac',
                        'MnO_Liq_cat_frac': 'Mn_Liq_cat_frac',
                        'MgO_Liq_cat_frac': 'Mg_Liq_cat_frac',
                        'CaO_Liq_cat_frac': 'Ca_Liq_cat_frac',
                        'Na2O_Liq_cat_frac': 'Na_Liq_cat_frac',
                        'K2O_Liq_cat_frac': 'K_Liq_cat_frac',
                        'Cr2O3_Liq_cat_frac': 'Cr_Liq_cat_frac',
                        'P2O5_Liq_cat_frac': 'P_Liq_cat_frac',

                        })

    return cation_frac_anhyd2

# Liquid Mgno function

def calculate_liq_mgno(liq_comps, Fe3Fet_Liq=None):
    '''
    calculates Liquid Mg#

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Fe3FeT: opt, float, pandas.Series, int
        Overwrites Fe3Fet_Liq column if specified

    Returns
    -------
    pandas Series
        Mg# of liquid
    '''
    liq_comps_c=liq_comps.copy()
    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq

    Liq_Mgno = (liq_comps_c['MgO_Liq'] / 40.3044) / ((liq_comps_c['MgO_Liq'] / 40.3044) +
            (liq_comps_c['FeOt_Liq'] * (1 - liq_comps_c['Fe3Fet_Liq']) / 71.844))
    return Liq_Mgno

## Hydrous mole proportions, mole fractions, cation proportions, cation fractions


def calculate_hydrous_mol_proportions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------


    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the ..Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = liq_comps.reindex(oxide_mass_liq_hyd_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_hyd_df, liq_wt],)
    # Drop the calculation column
    mol_prop_hyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_hyd.columns = [
        str(col) + '_mol_prop_hyd' for col in mol_prop_hyd.columns]
    return mol_prop_hyd


def calculate_hydrous_mol_fractions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.



    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_hydrous_mol_proportions_liquid(liq_comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_hyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_hyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_hyd.columns = [str(col).replace('prop', 'frac')
                            for col in mol_frac_hyd.columns]
    return mol_frac_hyd


def calculate_hydrous_cat_proportions_liquid(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous cation proportions (e.g., mole proportions * no of cations)

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.


    Returns
    -------
    pandas DataFrame
        hydrous cation proportions for the liquid with column headings of the form ...Liq_cat_prop_hyd

    '''
    mol_prop_no_cat_num = calculate_hydrous_mol_proportions_liquid(liq_comps)
    mol_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop_hyd', '') for col in mol_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_liq_hyd_df.reindex(
        oxide_mass_liq_hyd_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_no_cat_num])
    cation_prop_hyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_hyd.columns = [
        str(col) + '_cat_prop_hyd' for col in cation_prop_hyd.columns]


    cation_prop_hyd2=cation_prop_hyd.rename(columns={
                        'SiO2_Liq_cat_prop_hyd': 'Si_Liq_cat_prop_hyd',
                        'TiO2_Liq_cat_prop_hyd': 'Ti_Liq_cat_prop_hyd',
                        'Al2O3_Liq_cat_prop_hyd': 'Al_Liq_cat_prop_hyd',
                        'FeOt_Liq_cat_prop_hyd': 'Fet_Liq_cat_prop_hyd',
                        'MnO_Liq_cat_prop_hyd': 'Mn_Liq_cat_prop_hyd',
                        'MgO_Liq_cat_prop_hyd': 'Mg_Liq_cat_prop_hyd',
                        'CaO_Liq_cat_prop_hyd': 'Ca_Liq_cat_prop_hyd',
                        'Na2O_Liq_cat_prop_hyd': 'Na_Liq_cat_prop_hyd',
                        'K2O_Liq_cat_prop_hyd': 'K_Liq_cat_prop_hyd',
                        'Cr2O3_Liq_cat_prop_hyd': 'Cr_Liq_cat_prop_hyd',
                        'P2O5_Liq_cat_prop_hyd': 'P_Liq_cat_prop_hyd',
                        })

    return cation_prop_hyd2

def calculate_hydrous_cat_fractions_liquid(liq_comps, oxide_headers=False):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous cation fractions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    oxide_headers=False
        oxide_headers: bool
            default=False, returns as Ti_Liq_cat_prop.
            =True returns Ti_Liq_cat_prop.
            This is used for rapid matrix division for
            pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        anhydrous cation fractions for the liquid with column headings of the form ..Liq_cat_frac

    '''
    cat_prop = calculate_hydrous_cat_proportions_liquid(liq_comps=liq_comps)
    mol_prop = calculate_hydrous_mol_fractions_liquid(liq_comps=liq_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_hyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_hyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_hyd.columns = [str(col).replace('prop', 'frac')
                            for col in cat_frac_hyd.columns]
    cat_frac_hyd = pd.concat([liq_comps, mol_prop, cat_frac_hyd], axis=1)


    cat_frac_hyd['Fe2_Liq_cat_frac_hyd'] = cat_frac_hyd['Fet_Liq_cat_frac_hyd'] * \
        (1 - liq_comps['Fe3Fet_Liq'])
    if "Fe3Fet_Liq" in cat_frac_hyd:
        cat_frac_hyd['Mg_Number_Liq_NoFe3'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] / 71.844))
        cat_frac_hyd['Mg_Number_Liq_Fe3'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] * (1 - cat_frac_hyd['Fe3Fet_Liq']) / 71.844))
    if "Fe3Fet_Liq" not in cat_frac_hyd:
        cat_frac_hyd['Mg_Number_Liq'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] / 71.844))

        cat_frac_hyd['Mg_Number_Liq_NoFe3'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] / 71.844))



    return cat_frac_hyd

# calculating Liquid mole and cation fractions including Ni for Pu et al.
# 2017 and 2019


## Anhydrous mole proportions, mole fractions, cation proportions, cation fractions including Ni for the thermometers of Pu et al.

def calculate_anhydrous_mol_proportions_liquid_Ni(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the form SiO2_Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = liq_comps.reindex(
        oxide_mass_liq_anhyd_df_Ni.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_anhyd_df_Ni, liq_wt],)
    # Drop the calculation column
    mol_prop_anhyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_anhydrous_mol_fractions_liquid_Ni(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_anhydrous_mol_proportions_liquid_Ni(liq_comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd


def calculate_mol_proportions_olivine_ni(ol_comps):
    '''Import Olivine compositions using ol_comps=My_Olivines, returns mole proportions

   Parameters
    -------

    ol_comps: pandas.DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for olivines with column headings of the form SiO2_Ol_mol_prop

    '''
    # This makes it match the columns in the oxide mass dataframe
    ol_wt = ol_comps.reindex(oxide_mass_ol_df_Ni.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ol_wt_combo = pd.concat([oxide_mass_ol_df_Ni, ol_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ol_wt_combo.div(
        ol_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_mol_fractions_olivine_ni(ol_comps):
    '''Import Olivine compositions using ol_comps=My_Olivines, returns mole fractions

   Parameters
    -------

    ol_comps: pandas.DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for olivines with column headings of the form SiO2_Ol_mol_frac

    '''
    mol_prop = calculate_mol_proportions_olivine_ni(ol_comps=ol_comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd

## Olivine mole proportions, fractions, cation proportions and fractions


def calculate_mol_proportions_olivine(ol_comps):
    '''Import Olivine compositions using ol_comps=My_Olivines, returns mole proportions

   Parameters
    -------

    ol_comps: pandas.DataFrame
            Panda DataFrame of olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for olivines with column headings of the form SiO2_Ol_mol_prop

    '''
    # This makes it match the columns in the oxide mass dataframe
    ol_wt = ol_comps.reindex(oxide_mass_ol_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ol_wt_combo = pd.concat([oxide_mass_ol_df, ol_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ol_wt_combo.div(
        ol_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_mol_fractions_olivine(ol_comps):
    '''Import Olivine compositions using ol_comps=My_Olivines, returns mole fractions

   Parameters
    -------

    ol_comps: pandas.DataFrame
            olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for olivines with column headings of the form SiO2_Ol_mol_frac

    '''
    mol_prop = calculate_mol_proportions_olivine(ol_comps=ol_comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd


def calculate_cat_proportions_olivine(ol_comps):
    '''Import Olivine compositions using ol_comps=My_Olivines, returns cation proportions

   Parameters
    -------

    ol_comps: pandas.DataFrame
            olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for olivine with column headings of the form SiO2_Ol_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Ol_cat_prop rather than Na_Ol_cat_prop.
    '''

    mol_prop_no_cat_num = calculate_mol_proportions_olivine(ol_comps=ol_comps)
    mol_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in mol_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_ol_df.reindex(
        oxide_mass_ol_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]

    cation_prop_anhyd2=cation_prop_anhyd.rename(columns={

                        'SiO2_Ol_cat_prop': 'Si_Ol_cat_prop',
                        'TiO2_Ol_cat_prop': 'Ti_Ol_cat_prop',
                        'Al2O3_Ol_cat_prop': 'Al_Ol_cat_prop',
                        'FeOt_Ol_cat_prop': 'Fet_Ol_cat_prop',
                        'MnO_Ol_cat_prop': 'Mn_Ol_cat_prop',
                        'MgO_Ol_cat_prop': 'Mg_Ol_cat_prop',
                        'CaO_Ol_cat_prop': 'Ca_Ol_cat_prop',
                        'Na2O_Ol_cat_prop': 'Na_Ol_cat_prop',
                        'K2O_Ol_cat_prop': 'K_Ol_cat_prop',
                        'Cr2O3_Ol_cat_prop': 'Cr_Ol_cat_prop',
                        'P2O5_Ol_cat_prop': 'P_Ol_cat_prop',
                        })

    return cation_prop_anhyd2


def calculate_cat_fractions_olivine(ol_comps):
    '''Import Olivine compositions using ol_comps=My_Olivines, returns cation proportions

   Parameters
    -------

    ol_comps: pandas.DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for olivine with column headings of the form SiO2_Ol_cat_frac.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Ol_cat_frac rather than Na_Ol_cat_frac.

    '''

    cat_prop = calculate_cat_proportions_olivine(ol_comps=ol_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]

    return cat_frac_anhyd

## Garnet stuff
def calculate_mol_proportions_garnet(gt_comps):

    #Exchanging oxide measurements with element one if it exists.
    pd.options.mode.chained_assignment = None
    for i in range(0,len(gt_comps['SiO2_Gt'])):

        try:
            gt_comps['TiO2_Gt'][i] = (gt_comps['Ti_Gt'][i] * 1.6685) / 1e4
        except KeyError:
            pass
        ni_pass = True
        try:
            gt_comps['Ni_Gt'][i]
        except KeyError:
            ni_pass = False
            try:
                gt_comps['NiO_Gt'][i]
            except KeyError:
                gt_comps['Ni_Gt'][i] = (gt_comps['NiO_Gt'][i] * 1e4) / 1.27259

        if ni_pass == True:
            try:
                gt_comps['NiO_Gt'][i] = (gt_comps['Ni_Gt'][i] * 1.27259) / 1e4
            except KeyError:
                pass


    pd.options.mode.chained_assignment = 'warn'
    # This makes the input match the columns in the oxide mass dataframe
    gt_wt = gt_comps.reindex(oxide_mass_gt_df.columns, axis=1).fillna(0)
    gt_wt_combo = pd.concat([oxide_mass_gt_df, gt_wt],)

    mol_prop_anhyd = gt_wt_combo.div(
        gt_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]

    return mol_prop_anhyd

def calculate_oxygens_garnet(gt_comps):

    mol_prop = calculate_mol_proportions_garnet(gt_comps=gt_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_gt_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_moles_garnet(gt_comps):

    mol_prop = calculate_mol_proportions_garnet(gt_comps=gt_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    mole_num_reindex = cation_num_gt_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([mole_num_reindex, mol_prop])
    moles_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    moles_anhyd.columns = [str(col) + '_Mole' for col in moles_anhyd.columns]

    return moles_anhyd

def calculate_garnet_components(gt_comps):

    no_oxygen = 12.0
    oxy_prop = calculate_oxygens_garnet(gt_comps = gt_comps)
    cat_prop = calculate_moles_garnet(gt_comps = gt_comps)

    gt_calc = pd.concat([oxy_prop, cat_prop],axis = 1, join = 'inner')

    gt_calc['Oxygen_Sum_Gt'] = gt_calc['SiO2_Gt_ox'] + gt_calc['TiO2_Gt_ox'] +\
    gt_calc['Al2O3_Gt_ox'] + gt_calc['Cr2O3_Gt_ox'] + gt_calc['FeOt_Gt_ox'] +\
    gt_calc['MnO_Gt_ox'] + gt_calc['MgO_Gt_ox'] + gt_calc['CaO_Gt_ox'] +\
    gt_calc['Na2O_Gt_ox'] + gt_calc['K2O_Gt_ox']

    gt_calc['Si_Gt_Cat'] = (no_oxygen * gt_calc['SiO2_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Ti_Gt_Cat'] = (no_oxygen * gt_calc['TiO2_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Al_Gt_Cat'] = (no_oxygen * gt_calc['Al2O3_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Cr_Gt_Cat'] = (no_oxygen * gt_calc['Cr2O3_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Fe_Gt_Cat'] = (no_oxygen * gt_calc['FeOt_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Mn_Gt_Cat'] = (no_oxygen * gt_calc['MnO_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Mg_Gt_Cat'] = (no_oxygen * gt_calc['MgO_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Ca_Gt_Cat'] = (no_oxygen * gt_calc['CaO_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['Na_Gt_Cat'] = (no_oxygen * gt_calc['Na2O_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']
    gt_calc['K_Gt_Cat'] = (no_oxygen * gt_calc['K2O_Gt_Mole']) / gt_calc['Oxygen_Sum_Gt']

    gt_calc['Mg_MgFeCa_Gt'] = gt_calc['Mg_Gt_Cat'] / (gt_calc['Mg_Gt_Cat'] +\
     gt_calc['Fe_Gt_Cat'] + gt_calc['Ca_Gt_Cat'])
    gt_calc['Fe_MgFeCa_Gt'] = gt_calc['Fe_Gt_Cat'] / (gt_calc['Mg_Gt_Cat'] +\
     gt_calc['Fe_Gt_Cat'] + gt_calc['Ca_Gt_Cat'])
    gt_calc['Ca_MgFeCa_Gt'] = gt_calc['Ca_Gt_Cat'] / (gt_calc['Mg_Gt_Cat'] +\
     gt_calc['Fe_Gt_Cat'] + gt_calc['Ca_Gt_Cat'])

    gt_calc['Al_AlCr_Gt'] = gt_calc['Al_Gt_Cat'] / (gt_calc['Al_Gt_Cat'] +\
     gt_calc['Cr_Gt_Cat'])
    gt_calc['Cr_AlCr_Gt'] = gt_calc['Cr_Gt_Cat'] / (gt_calc['Al_Gt_Cat'] +\
     gt_calc['Cr_Gt_Cat'])

    return gt_calc

## Orthopyroxene mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_orthopyroxene(opx_comps):
    '''Import orthopyroxene compositions using opx_comps=My_Opxs, returns mole proportions

   Parameters
    -------

    opx_comps: pandas.DataFrame
        orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for orthopyroxene with column headings of the form SiO2_Opx_mol_prop

    '''

    # This makes the input match the columns in the oxide mass dataframe
    opx_wt = opx_comps.reindex(oxide_mass_opx_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    opx_wt_combo = pd.concat([oxide_mass_opx_df, opx_wt],)
    # Drop the calculation column
    mol_prop_anhyd = opx_wt_combo.div(
        opx_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_oxygens_orthopyroxene(opx_comps):
    '''Import orthopyroxene compositions using opx_comps=My_Opxs, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------

    opx_comps: pandas.DataFrame
        Panda DataFrame of orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Opx_ox

    '''

    mol_prop = calculate_mol_proportions_orthopyroxene(opx_comps=opx_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_opx_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]
    return oxygens_anhyd


def calculate_6oxygens_orthopyroxene(opx_comps):
    '''Import orthopyroxene compositions using opx_comps=My_Opxs, returns cations on the basis of 6 oxygens.

   Parameters
    -------

    opx_comps: pandas.DataFrame
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 6 oxygens, with column headings of the form ...Opx_cat_6ox.

    '''

    oxygens = calculate_oxygens_orthopyroxene(opx_comps=opx_comps)
    renorm_factor = 6 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_orthopyroxene(opx_comps=opx_comps)
    mol_prop['oxy_renorm_factor_opx'] = renorm_factor
    mol_prop_6 = mol_prop.multiply(mol_prop['oxy_renorm_factor_opx'], axis='rows')
    mol_prop_6.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_6.columns]

    ox_num_reindex = cation_num_opx_df.reindex(
        mol_prop_6.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_6])
    cation_6 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_6.columns = [str(col).replace('_mol_prop', '_cat_6ox')
                        for col in mol_prop.columns]
    cation_6['Al_IV_Opx_cat_6ox'] = 2 - cation_6['SiO2_Opx_cat_6ox']
    cation_6['Al_VI_Opx_cat_6ox'] = cation_6['Al2O3_Opx_cat_6ox'] - \
        cation_6['Al_IV_Opx_cat_6ox']
    cation_6.loc[cation_6['Al_VI_Opx_cat_6ox'] < 0, 'Al_VI_Opx_cat_6ox'] = 0
    cation_6['Si_Ti_Opx_cat_6ox'] = cation_6['SiO2_Opx_cat_6ox'] + \
        cation_6['TiO2_Opx_cat_6ox']

    cation_6_2=cation_6.rename(columns={
                            'SiO2_Opx_cat_6ox': 'Si_Opx_cat_6ox',
                            'TiO2_Opx_cat_6ox': 'Ti_Opx_cat_6ox',
                            'Al2O3_Opx_cat_6ox': 'Al_Opx_cat_6ox',
                            'FeOt_Opx_cat_6ox': 'Fet_Opx_cat_6ox',
                            'MnO_Opx_cat_6ox': 'Mn_Opx_cat_6ox',
                            'MgO_Opx_cat_6ox': 'Mg_Opx_cat_6ox',
                            'CaO_Opx_cat_6ox': 'Ca_Opx_cat_6ox',
                            'Na2O_Opx_cat_6ox': 'Na_Opx_cat_6ox',
                            'K2O_Opx_cat_6ox': 'K_Opx_cat_6ox',
                            'Cr2O3_Opx_cat_6ox': 'Cr_Opx_cat_6ox',
                            'P2O5_Opx_cat_6ox': 'P_Opx_cat_6ox_frac',
                            })

    cation_6_2['En_Simple_MgFeCa_Opx']=(cation_6_2['Mg_Opx_cat_6ox']/(cation_6_2['Mg_Opx_cat_6ox']
    +cation_6_2['Fet_Opx_cat_6ox']+cation_6_2['Ca_Opx_cat_6ox']))

    cation_6_2['Fs_Simple_MgFeCa_Opx']=(cation_6_2['Fet_Opx_cat_6ox']/(cation_6_2['Mg_Opx_cat_6ox']
    +cation_6_2['Fet_Opx_cat_6ox']+cation_6_2['Ca_Opx_cat_6ox']))

    cation_6_2['Wo_Simple_MgFeCa_Opx']=(cation_6_2['Ca_Opx_cat_6ox']/(cation_6_2['Mg_Opx_cat_6ox']
    +cation_6_2['Fet_Opx_cat_6ox']+cation_6_2['Ca_Opx_cat_6ox']))

    return cation_6_2



def calculate_orthopyroxene_components(opx_comps):
    '''Import orthopyroxene compositions using opx_comps=My_Opxs, returns orthopyroxene components along with entered Cpx compositions

   Parameters
    -------

    opx_comps: pandas.DataFrame
        Panda DataFrame of orthopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.


    Returns
    -------
    pandas DataFrame
        -orthopyroxene compositions (MgO_Opx etc.)
        -cations on bases of 6 oxygens (column headings of form  Cr2O3_Opx_cat_6ox, as well as Cation_Sum_Opx)
        -orthopyroxene components (NaAlSi2O6, FmTiAlSiO6, CrAl2SiO6, FmAl2SiO6, CaFmSi2O6, Fm2Si2O6, En_Opx, Di_Opx, Mgno_Opx)

    '''

    opx_calc = calculate_6oxygens_orthopyroxene(opx_comps=opx_comps)
    # Sum of cations, used to filter bad analyses
    opx_calc['Cation_Sum_Opx'] = (opx_calc['Si_Opx_cat_6ox'] + opx_calc['Ti_Opx_cat_6ox']
    + opx_calc['Al_Opx_cat_6ox'] + opx_calc['Fet_Opx_cat_6ox']
    + opx_calc['Mn_Opx_cat_6ox'] + opx_calc['Mg_Opx_cat_6ox'] +
    opx_calc['Ca_Opx_cat_6ox'] +opx_calc['Na_Opx_cat_6ox']
    + opx_calc['K_Opx_cat_6ox'] + opx_calc['Cr_Opx_cat_6ox'])

    opx_calc['Ca_CaMgFe']=opx_calc['Ca_Opx_cat_6ox']/(opx_calc['Ca_Opx_cat_6ox']
    +opx_calc['Fet_Opx_cat_6ox']+opx_calc['Mg_Opx_cat_6ox'])

    opx_calc['NaAlSi2O6'] = opx_calc['Na_Opx_cat_6ox']
    opx_calc['FmTiAlSiO6'] = opx_calc['Ti_Opx_cat_6ox']
    opx_calc['CrAl2SiO6'] = opx_calc['Cr_Opx_cat_6ox']
    opx_calc['FmAl2SiO6'] = opx_calc['Al_VI_Opx_cat_6ox'] - \
        opx_calc['NaAlSi2O6'] - opx_calc['CrAl2SiO6']

    opx_calc.loc[opx_calc['FmAl2SiO6'] < 0, 'FmAl2SiO6'] = 0

    opx_calc['CaFmSi2O6'] = opx_calc['Ca_Opx_cat_6ox']
    opx_calc['Fm2Si2O6'] = (((opx_calc['Fet_Opx_cat_6ox'] + opx_calc['Mg_Opx_cat_6ox']
    + opx_calc['Mn_Opx_cat_6ox'])- opx_calc['FmTiAlSiO6'] - opx_calc['FmAl2SiO6'] - opx_calc['CaFmSi2O6']) / 2)
    opx_calc['En_Opx'] = opx_calc['Fm2Si2O6'] * (opx_calc['Mg_Opx_cat_6ox'] / (
        opx_calc['Mg_Opx_cat_6ox'] + opx_calc['Fet_Opx_cat_6ox'] + opx_calc['Mn_Opx_cat_6ox']))
    opx_calc['Di_Opx'] = opx_calc['CaFmSi2O6'] * (opx_calc['Mg_Opx_cat_6ox'] / (
        opx_calc['Mg_Opx_cat_6ox'] + opx_calc['Fet_Opx_cat_6ox'] + opx_calc['Mn_Opx_cat_6ox']))
    opx_calc['Mgno_OPX'] = (opx_comps['MgO_Opx'] / 40.3044) / \
        (opx_comps['MgO_Opx'] / 40.3044 + opx_comps['FeOt_Opx'] / 71.844)

    opx_calc = pd.concat([opx_comps, opx_calc], axis=1)
    return opx_calc


def calculate_orthopyroxene_liquid_components(
        *, opx_comps=None, liq_comps=None, meltmatch=None, Fe3Fet_Liq=None):
    '''Import orthopyroxene compositions using opx_comps=My_Opxs and liquid compositions using liq_comps=My_Liquids,
        returns orthopyroxene and liquid components.

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.
    AND
     opx_comps: pandas.DataFrame
        orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.
    OR
    meltmatch: pandas.DataFrame
       merged orthopyroxene and liquid compositions used for melt matching

    Returns
    -------
    pandas DataFrame
       Merged dataframe of inputted liquids, liquid mole fractions, liquid cation fractions,
       inputted opx compositions, opx cations on 6 oxygen basis, opx components and opx-liquid components.

    '''
    # For when users enter a combined dataframe meltmatch=""
    if meltmatch is not None:
        combo_liq_opxs = meltmatch
    if liq_comps is not None and opx_comps is not None:
        liq_comps_c=liq_comps.copy()
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq']=Fe3Fet_Liq

        if len(liq_comps) != len(opx_comps):
            raise Exception(
                "inputted dataframes for liq_comps and opx_comps need to have the same number of rows")
        else:
            myOPXs1_comp = calculate_orthopyroxene_components(
                opx_comps=opx_comps)
            myLiquids1_comps = calculate_anhydrous_cat_fractions_liquid(
                liq_comps=liq_comps_c)
            combo_liq_opxs = pd.concat(
                [myLiquids1_comps, myOPXs1_comp], axis=1)

    combo_liq_opxs['ln_Fm2Si2O6_liq'] = (np.log(combo_liq_opxs['Fm2Si2O6'].astype('float64') /
    (combo_liq_opxs['Si_Liq_cat_frac']**2 * (combo_liq_opxs['Fet_Liq_cat_frac']
    + combo_liq_opxs['Mn_Liq_cat_frac'] + combo_liq_opxs['Mg_Liq_cat_frac'])**2)))

    combo_liq_opxs['ln_FmAl2SiO6_liq'] = (np.log(combo_liq_opxs['FmAl2SiO6'].astype('float64') /
    (combo_liq_opxs['Si_Liq_cat_frac'] * combo_liq_opxs['Al_Liq_cat_frac']**2 *
    (combo_liq_opxs['Fet_Liq_cat_frac']+ combo_liq_opxs['Mn_Liq_cat_frac'] + combo_liq_opxs['Mg_Liq_cat_frac']))))
    combo_liq_opxs['Kd_Fe_Mg_Fet'] = ((combo_liq_opxs['FeOt_Opx'] / 71.844) /
    (combo_liq_opxs['MgO_Opx'] / 40.3044)) / (
        (combo_liq_opxs['FeOt_Liq'] / 71.844) / (combo_liq_opxs['MgO_Liq'] / 40.3044))

    combo_liq_opxs['Kd_Fe_Mg_Fe2'] = ((combo_liq_opxs['FeOt_Opx'] / 71.844) /
    (combo_liq_opxs['MgO_Opx'] / 40.3044)) / (
        ((1 - combo_liq_opxs['Fe3Fet_Liq']) * combo_liq_opxs['FeOt_Liq'] / 71.844) /
        (combo_liq_opxs['MgO_Liq'] / 40.3044))

    combo_liq_opxs['Ideal_Kd'] = 0.4805 - 0.3733 * \
        combo_liq_opxs['Si_Liq_cat_frac']
    combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'] = abs(
        combo_liq_opxs['Ideal_Kd'] - combo_liq_opxs['Kd_Fe_Mg_Fe2'])
    b = np.empty(len(combo_liq_opxs), dtype=str)
    for i in range(0, len(combo_liq_opxs)):

        if combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'].iloc[i] > 0.06:
            b[i] = str("No")
        if combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'].iloc[i] < 0.06:
            b[i] = str("Yes")
    combo_liq_opxs.insert(1, "Kd Eq (Put2008+-0.06)", b)


    combo_liq_opxs['Mgno_Liq_noFe3']= ((combo_liq_opxs['MgO_Liq'] / 40.3044) /
    ((combo_liq_opxs['MgO_Liq'] / 40.3044) +
    (combo_liq_opxs['FeOt_Liq']) / 71.844))


    combo_liq_opxs['Mgno_Liq_Fe2']=((combo_liq_opxs['MgO_Liq'] / 40.3044) /
    ((combo_liq_opxs['MgO_Liq'] / 40.3044) +
    (combo_liq_opxs['FeOt_Liq'] * (1 - combo_liq_opxs['Fe3Fet_Liq']) / 71.844)))

    return combo_liq_opxs

## Clinopyroxene mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_clinopyroxene(cpx_comps):
    '''Import clinopyroxene compositions using cpx_comps=My_Cpxs, returns mole proportions

   Parameters
    -------

    cpx_comps: pandas.DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for clinopyroxene with column headings of the form SiO2_Cpx_mol_prop

    '''

    # This makes the input match the columns in the oxide mass dataframe
    cpx_wt = cpx_comps.reindex(oxide_mass_cpx_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    cpx_wt_combo = pd.concat([oxide_mass_cpx_df, cpx_wt],)
    # Drop the calculation column
    mol_prop_anhyd = cpx_wt_combo.div(
        cpx_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_oxygens_clinopyroxene(cpx_comps):
    '''Import clinopyroxene compositions using cpx_comps=My_Cpxs, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------

    cpx_comps: pandas.DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.



    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Cpx_ox

    '''

    mol_prop = calculate_mol_proportions_clinopyroxene(cpx_comps=cpx_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_cpx_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]
    return oxygens_anhyd


def calculate_6oxygens_clinopyroxene(cpx_comps):
    '''Import clinopyroxene compositions using cpx_comps=My_Cpxs, returns cations on the basis of 6 oxygens.

   Parameters
    -------

    cpx_comps: pandas.DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.


    Returns
    -------
    pandas DataFrame
        cations on the basis of 6 oxygens, with column headings of the form ...Cpx_cat_6ox.

    '''

    oxygens = calculate_oxygens_clinopyroxene(cpx_comps=cpx_comps)
    renorm_factor = 6 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_clinopyroxene(cpx_comps=cpx_comps)
    mol_prop['oxy_renorm_factor'] = renorm_factor
    mol_prop_6 = mol_prop.multiply(mol_prop['oxy_renorm_factor'], axis='rows')
    mol_prop_6.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_6.columns]

    ox_num_reindex = cation_num_cpx_df.reindex(
        mol_prop_6.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_6])
    cation_6 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_6.columns = [str(col).replace('_mol_prop', '_cat_6ox')
                        for col in mol_prop.columns]
    cation_6['Al_IV_cat_6ox'] = 2 - cation_6['SiO2_Cpx_cat_6ox']
    cation_6['Al_VI_cat_6ox'] = cation_6['Al2O3_Cpx_cat_6ox'] - \
        cation_6['Al_IV_cat_6ox']
    cation_6.loc[cation_6['Al_VI_cat_6ox'] < 0, 'Al_VI_cat_6ox'] = 0


    cation_6_2=cation_6.rename(columns={
                        'SiO2_Cpx_cat_6ox': 'Si_Cpx_cat_6ox',
                        'TiO2_Cpx_cat_6ox': 'Ti_Cpx_cat_6ox',
                        'Al2O3_Cpx_cat_6ox': 'Al_Cpx_cat_6ox',
                        'FeOt_Cpx_cat_6ox': 'Fet_Cpx_cat_6ox',
                        'MnO_Cpx_cat_6ox': 'Mn_Cpx_cat_6ox',
                        'MgO_Cpx_cat_6ox': 'Mg_Cpx_cat_6ox',
                        'CaO_Cpx_cat_6ox': 'Ca_Cpx_cat_6ox',
                        'Na2O_Cpx_cat_6ox': 'Na_Cpx_cat_6ox',
                        'K2O_Cpx_cat_6ox': 'K_Cpx_cat_6ox',
                        'Cr2O3_Cpx_cat_6ox': 'Cr_Cpx_cat_6ox',
                        'P2O5_Cpx_cat_6ox': 'P_Cpx_cat_6ox_frac',
                        })

    cation_6_2['En_Simple_MgFeCa_Cpx']=(cation_6_2['Mg_Cpx_cat_6ox']/(cation_6_2['Mg_Cpx_cat_6ox']
    +cation_6_2['Fet_Cpx_cat_6ox']+cation_6_2['Ca_Cpx_cat_6ox']))

    cation_6_2['Fs_Simple_MgFeCa_Cpx']=(cation_6_2['Fet_Cpx_cat_6ox']/(cation_6_2['Mg_Cpx_cat_6ox']
    +cation_6_2['Fet_Cpx_cat_6ox']+cation_6_2['Ca_Cpx_cat_6ox']))

    cation_6_2['Wo_Simple_MgFeCa_Cpx']=(cation_6_2['Ca_Cpx_cat_6ox']/(cation_6_2['Mg_Cpx_cat_6ox']
    +cation_6_2['Fet_Cpx_cat_6ox']+cation_6_2['Ca_Cpx_cat_6ox']))
    return cation_6_2

# calculating Clinopyroxene components following Putirka spreadsheet


def calculate_clinopyroxene_components(cpx_comps):
    '''Import clinopyroxene compositions using cpx_comps=My_Cpxs, returns clinopyroxene components along with entered Cpx compositions

   Parameters
    -------

    cpx_comps: pandas.DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Returns
    -------
    pandas DataFrame
        Clinopyroxene components (column headings: Cation_Sum, CrCaTs, a_cpx_En, Mgno_Cpx, Jd, CaTs, CaTi, DiHd_1996, DiHd_2003, En_Fs), cations on bases of 6 oxygens (column headings of form  Cr2O3_Cpx_cat_6ox), as well as inputted Cpx compositions (column headings of form MgO_Cpx)

    '''
    cpx_calc = calculate_6oxygens_clinopyroxene(cpx_comps=cpx_comps)

    # Sum of cations, used by Neave and Putirka (2017) to filter out bad
    # clinopyroxene analyses
    cpx_calc['Cation_Sum_Cpx'] = (cpx_calc['Si_Cpx_cat_6ox'] + cpx_calc['Ti_Cpx_cat_6ox']
    + cpx_calc['Al_Cpx_cat_6ox'] + cpx_calc['Fet_Cpx_cat_6ox']
    + cpx_calc['Mn_Cpx_cat_6ox'] + cpx_calc['Mg_Cpx_cat_6ox'] +
    cpx_calc['Ca_Cpx_cat_6ox'] + cpx_calc['Na_Cpx_cat_6ox']
    + cpx_calc['K_Cpx_cat_6ox'] + cpx_calc['Cr_Cpx_cat_6ox'])

    cpx_calc['Ca_CaMgFe']=cpx_calc['Ca_Cpx_cat_6ox']/(cpx_calc['Ca_Cpx_cat_6ox']+cpx_calc['Fet_Cpx_cat_6ox']
    +cpx_calc['Mg_Cpx_cat_6ox'])


    cpx_calc['Lindley_Fe3_Cpx'] = (cpx_calc['Na_Cpx_cat_6ox'] + cpx_calc['Al_IV_cat_6ox'] - cpx_calc['Al_VI_cat_6ox'] -
        2 * cpx_calc['Ti_Cpx_cat_6ox'] - cpx_calc['Cr_Cpx_cat_6ox'])  # This is cell FR


    cpx_calc.loc[(cpx_calc['Lindley_Fe3_Cpx'] < 0.0000000001),  'Lindley_Fe3_Cpx'] = 0
    cpx_calc.loc[(cpx_calc['Lindley_Fe3_Cpx'] >= cpx_calc['Fet_Cpx_cat_6ox'] ),  'Lindley_Fe3_Cpx'] = (
    cpx_calc['Fet_Cpx_cat_6ox'])




    cpx_calc['Lindley_Fe2_Cpx']=cpx_calc['Fet_Cpx_cat_6ox']-cpx_calc['Lindley_Fe3_Cpx']
    cpx_calc['Lindley_Fe3_Cpx_prop']=cpx_calc['Lindley_Fe3_Cpx']/cpx_calc['Fet_Cpx_cat_6ox']

    # Cpx Components that don't nee if and else statements and don't rely on
    # others.
    cpx_calc['CrCaTs'] = 0.5 * cpx_calc['Cr_Cpx_cat_6ox']
    cpx_calc['a_cpx_En'] = ((1 - cpx_calc['Ca_Cpx_cat_6ox'] - cpx_calc['Na_Cpx_cat_6ox'] - cpx_calc['K_Cpx_cat_6ox'])
     * (1 - 0.5 * (cpx_calc['Al_Cpx_cat_6ox']+ cpx_calc['Cr_Cpx_cat_6ox'] + cpx_calc['Na_Cpx_cat_6ox']
      + cpx_calc['K_Cpx_cat_6ox'])))
    cpx_calc['Mgno_Cpx'] = (cpx_comps['MgO_Cpx'] / 40.3044) / \
        (cpx_comps['MgO_Cpx'] / 40.3044 + cpx_comps['FeOt_Cpx'] / 71.844)

    # cpx_calc['Jd'] = np.empty(len(cpx_calc), dtype=float)
    # cpx_calc['CaTs'] = np.empty(len(cpx_calc), dtype=float)
    # cpx_calc['CaTi'] = np.empty(len(cpx_calc), dtype=float)
    # cpx_calc['DiHd_1996'] = np.empty(len(cpx_calc), dtype=float)


    AlVI_minus_Na=cpx_calc['Al_VI_cat_6ox']-cpx_calc['Na_Cpx_cat_6ox']
    cpx_calc['Jd']=cpx_calc['Na_Cpx_cat_6ox']
    cpx_calc['Jd_from 0=Na, 1=Al']=0
    cpx_calc['CaTs'] = cpx_calc['Al_VI_cat_6ox'] -cpx_calc['Na_Cpx_cat_6ox']

    # If value of AlVI<Na cat frac
    cpx_calc.loc[(AlVI_minus_Na<0), 'Jd_from 0=Na, 1=Al']=1
    cpx_calc.loc[(AlVI_minus_Na<0), 'Jd']=cpx_calc['Al_VI_cat_6ox']
    cpx_calc.loc[(AlVI_minus_Na<0), 'CaTs']=0

    # If value of AlIV>CaTs
    AlVI_minus_CaTs=cpx_calc['Al_IV_cat_6ox']-cpx_calc['CaTs']
    #default, if is bigger
    cpx_calc['CaTi']= (cpx_calc['Al_IV_cat_6ox'] - cpx_calc['CaTs']) / 2
    cpx_calc.loc[(AlVI_minus_CaTs<0), 'CaTi']=0


    #  If CaO-CaTs-CaTi-CrCaTs is >0
    Ca_CaTs_CaTi_CrCaTs=(cpx_calc['Ca_Cpx_cat_6ox'] - cpx_calc['CaTs'] -
                cpx_calc['CaTi'] - cpx_calc['CrCaTs'])

    cpx_calc['DiHd_1996']= (cpx_calc['Ca_Cpx_cat_6ox'] - cpx_calc['CaTs']
- cpx_calc['CaTi'] - cpx_calc['CrCaTs'] )

    cpx_calc.loc[(Ca_CaTs_CaTi_CrCaTs<0), 'DiHd_1996']=0



    cpx_calc['EnFs'] = ((cpx_calc['Fet_Cpx_cat_6ox'] +
                        cpx_calc['Mg_Cpx_cat_6ox']) - cpx_calc['DiHd_1996']) / 2
    cpx_calc['DiHd_2003'] = (cpx_calc['Ca_Cpx_cat_6ox'] -
                             cpx_calc['CaTs'] - cpx_calc['CaTi'] - cpx_calc['CrCaTs'])
    cpx_calc['Di_Cpx'] = cpx_calc['DiHd_2003'] * (cpx_calc['Mg_Cpx_cat_6ox'] / (
        cpx_calc['Mg_Cpx_cat_6ox'] + cpx_calc['Mn_Cpx_cat_6ox'] + cpx_calc['Fet_Cpx_cat_6ox']))
    cpx_calc['DiHd_1996'] = cpx_calc['DiHd_1996'].clip(lower=0)
    cpx_calc['DiHd_2003'] = cpx_calc['DiHd_2003'].clip(lower=0)
    cpx_calc['Jd'] = cpx_calc['Jd'].clip(lower=0)

    cpx_calc['FeIII_Wang21']=(cpx_calc['Na_Cpx_cat_6ox']+cpx_calc['Al_IV_cat_6ox']
    -cpx_calc['Al_VI_cat_6ox']-2*cpx_calc['Ti_Cpx_cat_6ox']-cpx_calc['Cr_Cpx_cat_6ox'])
    cpx_calc['FeII_Wang21']=cpx_calc['Fet_Cpx_cat_6ox']-cpx_calc['FeIII_Wang21']


    # Merging new Cpx compnoents with inputted cpx composition
    cpx_combined = pd.concat([cpx_comps, cpx_calc], axis='columns')

    return cpx_combined



def calculate_clinopyroxene_liquid_components(
        *, cpx_comps=None, liq_comps=None, meltmatch=None, Fe3Fet_Liq=None):
    '''Import clinopyroxene compositions using cpx_comps=My_Cpxs and liquid compositions using liq_comps=My_Liquids,
        returns clinopyroxene and liquid components.

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.
    AND
     cpx_comps: pandas.DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.
    OR
    meltmatch: pandas.DataFrame
        Panda DataFrame of merged clinopyroxene and liquid compositions used for melt matching

    Fe3Fet_Liq: opt, int, float, pandas.Series
        overwrites Fe3FeT ratio inliquid input
    Returns
    -------
    pandas DataFrame
       Merged dataframe of inputted liquids, liquid mole fractions, liquid cation fractions,
       inputted cpx compositions, cpx cations on 6 oxygen basis, cpx components and cpx-liquid components.

    '''
    # For when users enter a combined dataframe meltmatch=""
    if meltmatch is not None:
        combo_liq_cpxs = meltmatch
        if "Sample_ID_Cpx" in combo_liq_cpxs:
            combo_liq_cpxs = combo_liq_cpxs.drop(
                ['Sample_ID_Cpx'], axis=1) #.astype('float64')
        if  "Sample_ID_Liq" in combo_liq_cpxs:
            combo_liq_cpxs = combo_liq_cpxs.drop(
                ['Sample_ID_Liq'], axis=1)#.astype('float64')


    if liq_comps is not None and cpx_comps is not None:
        liq_comps_c=liq_comps.copy()
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq']=Fe3Fet_Liq



        if len(liq_comps) != len(cpx_comps):
            raise Exception(
                "inputted dataframes for liq_comps and cpx_comps need to have the same number of rows")
        else:
            myCPXs1_comp = calculate_clinopyroxene_components(
                cpx_comps=cpx_comps).reset_index(drop=True)
            myLiquids1_comps = calculate_anhydrous_cat_fractions_liquid(
                liq_comps=liq_comps_c).reset_index(drop=True)
            combo_liq_cpxs = pd.concat(
                [myLiquids1_comps, myCPXs1_comp], axis=1)
            if "Sample_ID_Cpx" in combo_liq_cpxs:
                combo_liq_cpxs=combo_liq_cpxs.drop(['Sample_ID_Cpx'], axis=1)
            if "Sample_ID_Liq" in combo_liq_cpxs:
                combo_liq_cpxs=combo_liq_cpxs.drop(['Sample_ID_Liq'], axis=1)



# Measured Kd Fe-Mg (using 2+)
    combo_liq_cpxs['Kd_Fe_Mg_Fe2'] = ((combo_liq_cpxs['Fet_Cpx_cat_6ox'] / combo_liq_cpxs['Mg_Cpx_cat_6ox']) / (
        (combo_liq_cpxs['Fet_Liq_cat_frac'] * (1 - combo_liq_cpxs['Fe3Fet_Liq']) / combo_liq_cpxs['Mg_Liq_cat_frac'])))

# Measured Kd Fe-Mg using 2+ in the liquid and Cpx based on Lindley
    combo_liq_cpxs['Kd_Fe_Mg_Fe2_Lind'] = ((combo_liq_cpxs['Lindley_Fe2_Cpx'] / combo_liq_cpxs['Mg_Cpx_cat_6ox']) / (
        (combo_liq_cpxs['Fet_Liq_cat_frac'] * (1 - combo_liq_cpxs['Fe3Fet_Liq']) / combo_liq_cpxs['Mg_Liq_cat_frac'])))


# Measured Kd Fe-Mg using Fet
    combo_liq_cpxs['Kd_Fe_Mg_Fet'] = ((combo_liq_cpxs['Fet_Cpx_cat_6ox'] / combo_liq_cpxs['Mg_Cpx_cat_6ox']) / (
        (combo_liq_cpxs['Fet_Liq_cat_frac'] / combo_liq_cpxs['Mg_Liq_cat_frac'])))

    combo_liq_cpxs['lnK_Jd_liq'] = np.log((combo_liq_cpxs['Jd'].astype(float)) / ((combo_liq_cpxs['Na_Liq_cat_frac']) * (
        combo_liq_cpxs['Al_Liq_cat_frac']) * ((combo_liq_cpxs['Si_Liq_cat_frac'])**2)))

    combo_liq_cpxs['lnK_Jd_DiHd_liq_1996'] = np.log((combo_liq_cpxs['Jd'].astype(float))
    * (combo_liq_cpxs['Ca_Liq_cat_frac'].astype(float)) * ((combo_liq_cpxs['Fet_Liq_cat_frac'].astype(float)) + (
        combo_liq_cpxs['Mg_Liq_cat_frac'].astype(float))) / ((combo_liq_cpxs['DiHd_1996'].astype(float)) * (combo_liq_cpxs['Na_Liq_cat_frac'].astype(float)) * (combo_liq_cpxs['Al_Liq_cat_frac'].astype(float))))

    combo_liq_cpxs['lnK_Jd_DiHd_liq_2003'] = np.log((combo_liq_cpxs['Jd'].astype(float)) * (combo_liq_cpxs['Ca_Liq_cat_frac'].astype(float)) * ((combo_liq_cpxs['Fet_Liq_cat_frac'].astype(float)) + (
        combo_liq_cpxs['Mg_Liq_cat_frac'].astype(float))) / ((combo_liq_cpxs['DiHd_2003'].astype(float)) * (combo_liq_cpxs['Na_Liq_cat_frac'].astype(float)) * (combo_liq_cpxs['Al_Liq_cat_frac'].astype(float))))

    combo_liq_cpxs['Kd_Fe_Mg_IdealWB'] = 0.109 + 0.186 * \
        combo_liq_cpxs['Mgno_Cpx']  # equation 35 of wood and blundy

    combo_liq_cpxs['Mgno_Liq_noFe3']= (combo_liq_cpxs['MgO_Liq'] / 40.3044) / ((combo_liq_cpxs['MgO_Liq'] / 40.3044) +
            (combo_liq_cpxs['FeOt_Liq']) / 71.844)


    combo_liq_cpxs['Mgno_Liq_Fe2']=(combo_liq_cpxs['MgO_Liq'] / 40.3044) / ((combo_liq_cpxs['MgO_Liq'] / 40.3044) +
            (combo_liq_cpxs['FeOt_Liq'] * (1 - combo_liq_cpxs['Fe3Fet_Liq']) / 71.844))


# Different ways to calculate DeltaFeMg

    combo_liq_cpxs['DeltaFeMg_WB'] = abs(
        combo_liq_cpxs['Kd_Fe_Mg_Fe2'] - combo_liq_cpxs['Kd_Fe_Mg_IdealWB'])
    # Adding back in sample names
    if meltmatch is not None and "Sample_ID_Cpx" in meltmatch:
        combo_liq_cpxs['Sample_ID_Cpx'] = meltmatch['Sample_ID_Cpx']
        combo_liq_cpxs['Sample_ID_Liq'] = meltmatch['Sample_ID_Liq']
    if meltmatch is not None and "Sample_ID_Cpx" not in meltmatch:
        combo_liq_cpxs['Sample_ID_Cpx'] = meltmatch.index
        combo_liq_cpxs['Sample_ID_Liq'] = meltmatch.index
    if liq_comps is not None and "Sample_ID_Liq" in liq_comps:
        combo_liq_cpxs['Sample_ID_Liq'] = liq_comps['Sample_ID_Liq']
    if liq_comps is not None and "Sample_ID_Liq" not in liq_comps:
        combo_liq_cpxs['Sample_ID_Liq'] = liq_comps.index
    if cpx_comps is not None and "Sample_ID_Cpx" not in cpx_comps:
        combo_liq_cpxs['Sample_ID_Cpx'] = cpx_comps.index
    if cpx_comps is not None and "Sample_ID_Cpx" in cpx_comps:
        combo_liq_cpxs['Sample_ID_Cpx'] = cpx_comps['Sample_ID_Cpx']

    combo_liq_cpxs.replace([np.inf, -np.inf], np.nan, inplace=True)

    return combo_liq_cpxs

## Feldspar mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_plagioclase(*, plag_comps=None):
    '''Import plagioclase compositions using plag_comps=My_plagioclases, returns mole proportions

   Parameters
    -------

    plag_comps: pandas.DataFrame
            Panda DataFrame of plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for plagioclases with column headings of the form SiO2_Plag_mol_prop

    '''

    if plag_comps is not None:
        plag_comps = plag_comps
        # This makes it match the columns in the oxide mass dataframe
        plag_wt = plag_comps.reindex(
            oxide_mass_plag_df.columns, axis=1).fillna(0)
        # Combine the molecular weight and weight percent dataframes
        plag_wt_combo = pd.concat([oxide_mass_plag_df, plag_wt],)
        # Drop the calculation column
        plag_prop_anhyd = plag_wt_combo.div(
            plag_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
        plag_prop_anhyd.columns = [
            str(col) + '_mol_prop' for col in plag_prop_anhyd.columns]
        return plag_prop_anhyd


def calculate_mol_fractions_plagioclase(*, plag_comps=None):
    '''Import plagioclase compositions using plag_comps=My_plagioclases, returns mole fractions

   Parameters
    -------

    plag_comps: pandas.DataFrame
            plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for plagioclases with column headings of the form SiO2_Plag_mol_frac


    '''

    if plag_comps is not None:
        plag_comps = plag_comps
        plag_prop = calculate_mol_proportions_plagioclase(
            plag_comps=plag_comps)
        plag_prop['sum'] = plag_prop.sum(axis='columns')
        plag_frac_anhyd = plag_prop.div(plag_prop['sum'], axis='rows')
        plag_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
        plag_frac_anhyd.columns = [str(col).replace(
            'prop', 'frac') for col in plag_frac_anhyd.columns]
        return plag_frac_anhyd


def calculate_cat_proportions_plagioclase(*, plag_comps=None, oxide_headers=False):
    '''Import plagioclase compositions using plag_comps=My_plagioclases, returns cation proportions

   Parameters
    -------

    plag_comps: pandas.DataFrame
            plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    oxide_headers: bool
        default=False, returns as Ti_Plag_cat_prop.
        =True returns Ti_Plag_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc


    Returns
    -------
    pandas DataFrame
        cation proportions for plagioclase with column headings of the form ...Plag_cat_prop
    '''

    plag_prop_no_cat_num = calculate_mol_proportions_plagioclase(
        plag_comps=plag_comps)
    plag_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in plag_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_plag_df.reindex(
        oxide_mass_plag_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, plag_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={

                            'SiO2_Plag_cat_prop': 'Si_Plag_cat_prop',
                            'TiO2_Plag_cat_prop': 'Ti_Plag_cat_prop',
                            'Al2O3_Plag_cat_prop': 'Al_Plag_cat_prop',
                            'FeOt_Plag_cat_prop': 'Fet_Plag_cat_prop',
                            'MnO_Plag_cat_prop': 'Mn_Plag_cat_prop',
                            'MgO_Plag_cat_prop': 'Mg_Plag_cat_prop',
                            'CaO_Plag_cat_prop': 'Ca_Plag_cat_prop',
                            'Na2O_Plag_cat_prop': 'Na_Plag_cat_prop',
                            'K2O_Plag_cat_prop': 'K_Plag_cat_prop',
                            'Cr2O3_Plag_cat_prop': 'Cr_Plag_cat_prop',
                            'P2O5_Plag_cat_prop': 'P_Plag_cat_prop',
                            })

        return cation_prop_anhyd2


def calculate_cat_fractions_plagioclase(plag_comps):
    '''Import plagioclase compositions using plag_comps=My_plagioclases, returns cation fractions

   Parameters
    -------

    plag_comps: pandas.DataFrame
        plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for plagioclase with column headings of the form ...Plag_cat_frac.


    '''

    cat_prop = calculate_cat_proportions_plagioclase(plag_comps=plag_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd['An_Plag'] = cat_frac_anhyd['Ca_Plag_cat_frac'] / \
        (cat_frac_anhyd['Ca_Plag_cat_frac'] +
         cat_frac_anhyd['Na_Plag_cat_frac'] + cat_frac_anhyd['K_Plag_cat_frac'])
    cat_frac_anhyd['Ab_Plag'] = cat_frac_anhyd['Na_Plag_cat_frac'] / \
        (cat_frac_anhyd['Ca_Plag_cat_frac'] +
         cat_frac_anhyd['Na_Plag_cat_frac'] + cat_frac_anhyd['K_Plag_cat_frac'])
    cat_frac_anhyd['Or_Plag'] = 1 - \
        cat_frac_anhyd['An_Plag'] - cat_frac_anhyd['Ab_Plag']
    cat_frac_anhyd2 = pd.concat([plag_comps, cat_prop, cat_frac_anhyd], axis=1)
    return cat_frac_anhyd2



# calculating alkali feldspar components


def calculate_mol_proportions_kspar(*, kspar_comps=None):
    '''Import AlkaliFspar compositions using kspar_comps=My_kspars, returns mole proportions

   Parameters
    -------

    kspar_comps: pandas.DataFrame
            Panda DataFrame of AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for AlkaliFspars with column headings of the form SiO2_Kspar_mol_prop

    '''

    if kspar_comps is not None:
        alk_comps = kspar_comps
        # This makes it match the columns in the oxide mass dataframe
        alk_wt = alk_comps.reindex(
            oxide_mass_kspar_df.columns, axis=1).fillna(0)
        # Combine the molecular weight and weight percent dataframes
        alk_wt_combo = pd.concat([oxide_mass_kspar_df, alk_wt],)
        # Drop the calculation column
        alk_prop_anhyd = alk_wt_combo.div(
            alk_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
        alk_prop_anhyd.columns = [
            str(col) + '_mol_prop' for col in alk_prop_anhyd.columns]
        return alk_prop_anhyd


def calculate_mol_fractions_kspar(kspar_comps):
    '''Import AlkaliFspar compositions using kspar_comps=My_kspars, returns mole fractions

   Parameters
    -------

    kspar_comps: pandas.DataFrame
            AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for AlkaliFspars with column headings of the form ...Kspar_mol_frac


    '''


    alk_comps = kspar_comps
    alk_prop = calculate_mol_proportions_kspar(kspar_comps=alk_comps)
    alk_prop['sum'] = alk_prop.sum(axis='columns')
    alk_frac_anhyd = alk_prop.div(alk_prop['sum'], axis='rows')
    alk_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    alk_frac_anhyd.columns = [str(col).replace(
        'prop', 'frac') for col in alk_frac_anhyd.columns]


    return alk_frac_anhyd


def calculate_cat_proportions_kspar(*, kspar_comps=None, oxide_headers=False):
    '''Import kspar compositions using kspar_comps=My_kspars, returns cation proportions

   Parameters
    -------

    kspar_comps: pandas.DataFrame
            kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    oxide_headers: bool
        default=False, returns as Ti_Kspar_cat_prop.
        =True returns Ti_Kspar_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc


    Returns
    -------
    pandas DataFrame
        cation proportions for kspar with column headings of the form SiO2_Kspar_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Kspar_cat_prop rather than Na_Kspar_cat_prop.
    '''

    kspar_prop_no_cat_num = calculate_mol_proportions_kspar(
        kspar_comps=kspar_comps)
    kspar_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in kspar_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_kspar_df.reindex(
        oxide_mass_kspar_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, kspar_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={

                            'SiO2_Kspar_cat_prop': 'Si_Kspar_cat_prop',
                            'TiO2_Kspar_cat_prop': 'Ti_Kspar_cat_prop',
                            'Al2O3_Kspar_cat_prop': 'Al_Kspar_cat_prop',
                            'FeOt_Kspar_cat_prop': 'Fet_Kspar_cat_prop',
                            'MnO_Kspar_cat_prop': 'Mn_Kspar_cat_prop',
                            'MgO_Kspar_cat_prop': 'Mg_Kspar_cat_prop',
                            'CaO_Kspar_cat_prop': 'Ca_Kspar_cat_prop',
                            'Na2O_Kspar_cat_prop': 'Na_Kspar_cat_prop',
                            'K2O_Kspar_cat_prop': 'K_Kspar_cat_prop',
                            'Cr2O3_Kspar_cat_prop': 'Cr_Kspar_cat_prop',
                            'P2O5_Kspar_cat_prop': 'P_Kspar_cat_prop',
                            })

        return cation_prop_anhyd2

def calculate_cat_fractions_kspar(*, kspar_comps=None):
    '''Import AlkaliFspar compositions using kspar_comps=My_kspars, returns cation fractions

   Parameters
    -------

    kspar_comps: pandas.DataFrame
        AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for AlkaliFspar with column headings of the form SiO2_Kspar_cat_frac.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Kspar_cat_frac rather than Na_Kspar_cat_frac.

    '''

    cat_prop = calculate_cat_proportions_kspar(kspar_comps=kspar_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd['An_Kspar'] = cat_frac_anhyd['Ca_Kspar_cat_frac'] / \
        (cat_frac_anhyd['Ca_Kspar_cat_frac'] +
         cat_frac_anhyd['Na_Kspar_cat_frac'] + cat_frac_anhyd['K_Kspar_cat_frac'])
    cat_frac_anhyd['Ab_Kspar'] = cat_frac_anhyd['Na_Kspar_cat_frac'] / \
        (cat_frac_anhyd['Ca_Kspar_cat_frac'] +
         cat_frac_anhyd['Na_Kspar_cat_frac'] + cat_frac_anhyd['K_Kspar_cat_frac'])
    cat_frac_anhyd['Or_Kspar'] = 1 - \
        cat_frac_anhyd['An_Kspar'] - cat_frac_anhyd['Ab_Kspar']
    cat_frac_anhyd2 = pd.concat(
        [kspar_comps, cat_prop, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

## Amphibole mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_amphibole(amp_comps):
    '''Import amphibole compositions using amp_comps=My_amphiboles, returns mole proportions

   Parameters
    -------

    amp_comps: pandas.DataFrame
            Panda DataFrame of amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for amphiboles with column headings of the form SiO2_Amp_mol_prop

    '''

    # This makes it match the columns in the oxide mass dataframe
    amp_wt = amp_comps.reindex(oxide_mass_amp_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    amp_wt_combo = pd.concat([oxide_mass_amp_df, amp_wt],)
    # Drop the calculation column
    mol_prop_anhyd = amp_wt_combo.div(
        amp_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_mol_fractions_amphibole(amp_comps):
    '''Import amphibole compositions using amp_comps=My_amphiboles, returns mole fractions

   Parameters
    -------

    amp_comps: pandas.DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for amphiboles with column headings of the form SiO2_Amp_mol_frac

    '''


def calculate_cat_proportions_amphibole(*, amp_comps=None, oxide_headers=False):
    '''Import amphibole compositions using amp_comps=My_amphiboles, returns cation proportions

   Parameters
    -------

    amp_comps: pandas.DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.


    Returns
    -------
    pandas DataFrame
        cation proportions for amphibole with column headings of the form ...Amp_cat_prop
    '''

    amp_prop_no_cat_num = calculate_mol_proportions_amphibole(
        amp_comps=amp_comps)
    amp_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in amp_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_amp_df.reindex(
        oxide_mass_amp_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, amp_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]

    cation_prop_anhyd2=cation_prop_anhyd.rename(columns={

                        'SiO2_Amp_cat_prop': 'Si_Amp_cat_prop',
                        'TiO2_Amp_cat_prop': 'Ti_Amp_cat_prop',
                        'Al2O3_Amp_cat_prop': 'Al_Amp_cat_prop',
                        'FeOt_Amp_cat_prop': 'Fet_Amp_cat_prop',
                        'MnO_Amp_cat_prop': 'Mn_Amp_cat_prop',
                        'MgO_Amp_cat_prop': 'Mg_Amp_cat_prop',
                        'CaO_Amp_cat_prop': 'Ca_Amp_cat_prop',
                        'Na2O_Amp_cat_prop': 'Na_Amp_cat_prop',
                        'K2O_Amp_cat_prop': 'K_Amp_cat_prop',
                        'Cr2O3_Amp_cat_prop': 'Cr_Amp_cat_prop',
                        'P2O5_Amp_cat_prop': 'P_Amp_cat_prop',
                        })





    return cation_prop_anhyd2


def calculate_oxygens_amphibole(amp_comps):
    '''Import amphiboles compositions using amp_comps=My_Amps, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------

    amp_comps: pandas.DataFrame
        amphiboles compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Amp_ox

    '''

    mol_prop = calculate_mol_proportions_amphibole(amp_comps=amp_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_amp_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]
    return oxygens_anhyd


def calculate_23oxygens_amphibole(amp_comps):
    '''Import amphibole compositions using amp_comps=My_Amps, returns cations on the basis of 23 oxygens.

   Parameters
    -------

    amp_comps: pandas.DataFrame
        amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 23 oxygens, with column headings of the form Si_Amp_cat_23ox.
        For simplicity, and consistency of column labelling to aid calculations, oxide names are preserved,
        so outputs are Na_Amp_cat_23ox rather than Na_Amp_cat_23ox.

    '''

    oxygens = calculate_oxygens_amphibole(amp_comps=amp_comps)
    renorm_factor = 23 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_amphibole(amp_comps=amp_comps)
    mol_prop['oxy_renorm_factor'] = renorm_factor
    mol_prop_23 = mol_prop.multiply(mol_prop['oxy_renorm_factor'], axis='rows')
    mol_prop_23.columns = [str(col).replace('_mol_prop', '')
                           for col in mol_prop_23.columns]

    ox_num_reindex = cation_num_amp_df.reindex(
        mol_prop_23.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_23])
    cation_23_ox = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_23_ox.columns = [str(col).replace('_mol_prop', '_cat_23ox')
                         for col in mol_prop.columns]

    cation_23=cation_23_ox.rename(columns={
                            'SiO2_Amp_cat_23ox': 'Si_Amp_cat_23ox',
                            'TiO2_Amp_cat_23ox': 'Ti_Amp_cat_23ox',
                            'Al2O3_Amp_cat_23ox': 'Al_Amp_cat_23ox',
                            'FeOt_Amp_cat_23ox': 'Fet_Amp_cat_23ox',
                            'MnO_Amp_cat_23ox': 'Mn_Amp_cat_23ox',
                            'MgO_Amp_cat_23ox': 'Mg_Amp_cat_23ox',
                            'CaO_Amp_cat_23ox': 'Ca_Amp_cat_23ox',
                            'Na2O_Amp_cat_23ox': 'Na_Amp_cat_23ox',
                            'K2O_Amp_cat_23ox': 'K_Amp_cat_23ox',
                            'Cr2O3_Amp_cat_23ox': 'Cr_Amp_cat_23ox',
                            'P2O5_Amp_cat_23ox': 'P_Amp_cat_23ox_frac',})

    cation_23['cation_sum_Si_Mg'] = (cation_23['Si_Amp_cat_23ox']
    + cation_23['Ti_Amp_cat_23ox'] + cation_23['Al_Amp_cat_23ox'] +
        cation_23['Cr_Amp_cat_23ox'] + cation_23['Fet_Amp_cat_23ox'] +
        cation_23['Mn_Amp_cat_23ox'] + cation_23['Mg_Amp_cat_23ox'])
    cation_23['cation_sum_Si_Ca'] = (cation_23['cation_sum_Si_Mg'] +
        cation_23['Ca_Amp_cat_23ox'])
    cation_23['cation_sum_All'] = cation_23['cation_sum_Si_Ca'] + \
        cation_23['Na_Amp_cat_23ox'] + +cation_23['K_Amp_cat_23ox']
    cation_23['Mgno_Amp']=cation_23['Mg_Amp_cat_23ox']/(cation_23['Mg_Amp_cat_23ox']
    +cation_23['Fet_Amp_cat_23ox'])

    return cation_23
# Ridolfi Amphiboles, using Cl and F, does on 13 cations.


def calculate_mol_proportions_amphibole_ridolfi(amp_comps):
    '''Import amphibole compositions using amp_comps=My_amphiboles, returns mole proportions

   Parameters
    -------

    amp_comps: pandas.DataFrame
            Panda DataFrame of amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for amphiboles with column headings of the form SiO2_Amp_mol_prop

    '''

    # This makes it match the columns in the oxide mass dataframe
    amp_wt = amp_comps.reindex(
        oxide_mass_amp_df_Ridolfi.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    amp_wt_combo = pd.concat([oxide_mass_amp_df_Ridolfi, amp_wt],)
    # Drop the calculation column
    mol_prop_anhyd = amp_wt_combo.div(
        amp_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_cat_proportions_amphibole_ridolfi(*, amp_comps=None):
    '''Import amphibole compositions using amp_comps=My_amphiboles, returns cation proportions

   Parameters
    -------

    amp_comps: pandas.DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for amphibole with column headings of the form SiO2_Amp_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Amp_cat_prop rather than Na_Amp_cat_prop.

    '''

    amp_prop_no_cat_num = calculate_mol_proportions_amphibole_ridolfi(
        amp_comps=amp_comps)
    amp_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in amp_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_amp_df_Ridolfi.reindex(
        oxide_mass_amp_df_Ridolfi.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, amp_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    return cation_prop_anhyd


def calculate_13cations_amphibole_ridolfi(amp_comps):
    '''Import amphibole compositions using amp_comps=My_amphiboles, returns
    components calculated on basis of 13 cations following Ridolfi supporting information

   Parameters
    -------

    amp_comps: pandas.DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for amphiboles with column headings of the form SiO2_Amp_13_cat...

    '''
    cats = calculate_cat_proportions_amphibole_ridolfi(amp_comps=amp_comps)
    cats['cation_sum_Si_Mg'] = (cats['SiO2_Amp_cat_prop'] + cats['TiO2_Amp_cat_prop'] + cats['Al2O3_Amp_cat_prop'] +
                                cats['Cr2O3_Amp_cat_prop'] + cats['FeOt_Amp_cat_prop'] + cats['MnO_Amp_cat_prop'] + cats['MgO_Amp_cat_prop'])

    sum_SiMg = cats['cation_sum_Si_Mg']
    cats.drop(['cation_sum_Si_Mg'], axis=1, inplace=True)
    cat_13 = 13 * cats.divide(sum_SiMg, axis='rows')
    cat_13.columns = [str(col).replace('_cat_prop', '_13_cat')
                      for col in cat_13.columns]
    cat_13['cation_sum_Si_Mg'] = sum_SiMg
    cation_13_noox=cat_13.rename(columns={
                            'SiO2_Amp_13_cat': 'Si_Amp_13_cat',
                            'TiO2_Amp_13_cat': 'Ti_Amp_13_cat',
                            'Al2O3_Amp_13_cat': 'Al_Amp_13_cat',
                            'FeOt_Amp_13_cat': 'Fet_Amp_13_cat',
                            'MnO_Amp_13_cat': 'Mn_Amp_13_cat',
                            'MgO_Amp_13_cat': 'Mg_Amp_13_cat',
                            'CaO_Amp_13_cat': 'Ca_Amp_13_cat',
                            'Na2O_Amp_13_cat': 'Na_Amp_13_cat',
                            'K2O_Amp_13_cat': 'K_Amp_13_cat',
                            'Cr2O3_Amp_13_cat': 'Cr_Amp_13_cat',
                            'P2O5_Amp_13_cat': 'P_Amp_13_cat_frac'})



    cat_13_out = pd.concat([cats, cation_13_noox], axis=1)
    return cat_13_out

def calculate_sites_ridolfi(amp_comps):

    amp_comps_c=amp_comps.copy()
    norm_cations=calculate_13cations_amphibole_ridolfi(amp_comps_c)

    # Ridolfi T sites
    norm_cations['Si_T']=norm_cations['Si_Amp_13_cat']
    norm_cations['Al_IV_T']=0
    norm_cations['Ti_T']=0
    # Ridolfi C Sites


    norm_cations['Cr_C']=0
    norm_cations['Fe3_C']=0
    norm_cations['Mg_C']=0
    norm_cations['Fe2_C']=0
    norm_cations['Mn_C']=0
    # Ridolfi B sites
    norm_cations['Ca_B']=norm_cations['Ca_Amp_13_cat']
    norm_cations['Na_B']=0
    # Ridolfi A sites
    norm_cations['Na_A']=0
    norm_cations['K_A']=norm_cations['K_Amp_13_cat']

    # if sum greater than 8, equal to difference
    norm_cations['Al_IV_T']=8-norm_cations['Si_T']
    # But if SiTi grater than 8
    Si_Al_less8=(norm_cations['Si_Amp_13_cat']+norm_cations['Al_Amp_13_cat'])<8
    norm_cations.loc[(Si_Al_less8), 'Al_IV_T']=norm_cations['Al_Amp_13_cat']

    # Ti, If Si + Al (IV)<8, 8-Si-AlIV
    Si_Al_sites_less8=(norm_cations['Si_T']+norm_cations['Al_IV_T'])<8
    norm_cations.loc[(Si_Al_sites_less8), 'Ti_T']=8-norm_cations['Si_T']-norm_cations['Al_IV_T']

    #  AL VI, any AL left
    norm_cations['Al_VI_C']=norm_cations['Al_Amp_13_cat']-norm_cations['Al_IV_T']

    # Ti C Sites, any Ti left
    norm_cations['Ti_C']=norm_cations['Ti_Amp_13_cat']-norm_cations['Ti_T']

    # CR site, All Cr
    norm_cations['Cr_C']=norm_cations['Cr_Amp_13_cat']

    # Calculate charge for Fe
    norm_cations['Charge']=(norm_cations['Si_Amp_13_cat']*4+norm_cations['Ti_Amp_13_cat']*4+norm_cations['Al_Amp_13_cat']*3+
    norm_cations['Cr_Amp_13_cat']*3+norm_cations['Fet_Amp_13_cat']*2+norm_cations['Mn_Amp_13_cat']*2+norm_cations['Mg_Amp_13_cat']*2
    +norm_cations['Ca_Amp_13_cat']*2+norm_cations['Na_Amp_13_cat']+norm_cations['K_Amp_13_cat'])

    # If DG2 (charge)>46, set Fe3 to zero, else set to 46-charge
    norm_cations['Fe3_C']=46-norm_cations['Charge']
    High_Charge=norm_cations['Charge']>46
    norm_cations.loc[(High_Charge), 'Fe3_C']=0

    norm_cations['Fe2_C']=norm_cations['Fet_Amp_13_cat']-norm_cations['Fe3_C']

    #  Allocate all Mg
    norm_cations['Mg_C']=norm_cations['Mg_Amp_13_cat']

    # Allocate all Mn
    norm_cations['Mn_C']=norm_cations['Mn_Amp_13_cat']

    # Na B site,

    norm_cations['Na_B']=2-norm_cations['Ca_Amp_13_cat']
    Ca_greaterthanNa=norm_cations['Na_Amp_13_cat']<(2-norm_cations['Ca_Amp_13_cat'])
    norm_cations.loc[(Ca_greaterthanNa), 'Na_B']=norm_cations['Na_Amp_13_cat']

    # All Na left after B
    norm_cations['Na_A']=norm_cations['Na_Amp_13_cat']-norm_cations['Na_B']
    if "Sample_ID_Amp" in amp_comps.columns:
        myAmps1_label = amp_comps_c.drop(['Sample_ID_Amp'], axis='columns')
    else:
        myAmps1_label = amp_comps_c
    norm_cations['Sum_input'] = myAmps1_label.sum(axis='columns')
    Sum_input=norm_cations['Sum_input']
    Low_sum=norm_cations['Sum_input'] <90

   # Other checks in Ridolfi's spreadsheet
    norm_cations['H2O_calc']=(2-norm_cations['F_Amp_13_cat']-norm_cations['Cl_Amp_13_cat'])*norm_cations['cation_sum_Si_Mg']*17/13/2
    norm_cations.loc[(Low_sum), 'H2O_calc']=0

    norm_cations['Charge']=(norm_cations['Si_Amp_13_cat']*4+norm_cations['Ti_Amp_13_cat']*4+norm_cations['Al_Amp_13_cat']*3+
    norm_cations['Cr_Amp_13_cat']*3+norm_cations['Fet_Amp_13_cat']*2+norm_cations['Mn_Amp_13_cat']*2+norm_cations['Mg_Amp_13_cat']*2
    +norm_cations['Ca_Amp_13_cat']*2+norm_cations['Na_Amp_13_cat']+norm_cations['K_Amp_13_cat'])

    norm_cations['Fe3_calc']=46-norm_cations['Charge']
    High_Charge=norm_cations['Charge']>46
    norm_cations.loc[(High_Charge), 'Fe3_calc']=0

    norm_cations['Fe2_calc']=norm_cations['Fet_Amp_13_cat']-norm_cations['Fe3_calc']


    norm_cations['Fe2O3_calc']=norm_cations['Fe3_calc']*norm_cations['cation_sum_Si_Mg']*159.691/13/2
    norm_cations.loc[(Low_sum), 'Fe2O3_calc']=0

    norm_cations['FeO_calc']=norm_cations['Fe2_calc']*norm_cations['cation_sum_Si_Mg']*71.846/13
    norm_cations.loc[(Low_sum), 'Fe2O3_calc']=0

    norm_cations['O=F,Cl']=-(amp_comps_c['F_Amp']*0.421070639014633+amp_comps_c['Cl_Amp']*0.225636758525372)
    norm_cations.loc[(Low_sum), 'O=F,Cl']=0

    norm_cations['Total_recalc']=(Sum_input-amp_comps_c['FeOt_Amp']+norm_cations['H2O_calc']+norm_cations['Fe2O3_calc']
    +norm_cations['FeO_calc']+norm_cations['O=F,Cl'])
    norm_cations.loc[(Low_sum), 'Total']=0

    # Set up a column for a fail message
    norm_cations['Fail Msg']=""
    norm_cations['Input_Check']=True

    # Check that old total isn't <90

    norm_cations.loc[(Low_sum), 'Input_Check']=False
    norm_cations.loc[(Low_sum), 'Fail Msg']="Cation oxide Total<90"

    # First check, that new total is >98.5 (e.g with recalculated H2O etc).

    Low_total_Recalc=norm_cations['Total_recalc']<98.5
    norm_cations.loc[(Low_total_Recalc), 'Input_Check']=False
    norm_cations.loc[(Low_total_Recalc), 'Fail Msg']="Recalc Total<98.5"

    # Next, check that new total isn't >102
    High_total_Recalc=norm_cations['Total_recalc']>102
    norm_cations.loc[(High_total_Recalc), 'Input_Check']=False
    norm_cations.loc[(High_total_Recalc), 'Fail Msg']="Recalc Total>102"

    # Next, check that charge isn't >46.5 ("unbalanced")
    Unbalanced_Charge=norm_cations['Charge']>46.5
    norm_cations.loc[(Unbalanced_Charge), 'Input_Check']=False
    norm_cations.loc[(Unbalanced_Charge), 'Fail Msg']="unbalanced charge (>46.5)"

    # Next check that Fe2+ is greater than 0, else unbalanced
    Negative_Fe2=norm_cations['Fe2_calc']<0
    norm_cations.loc[(Negative_Fe2), 'Input_Check']=False
    norm_cations.loc[(Negative_Fe2), 'Fail Msg']="unbalanced charge (Fe2<0)"

    # Check that Mg# calculated using just Fe2 is >54, else low Mg
    norm_cations['Mgno_Fe2']=norm_cations['Mg_Amp_13_cat']/(norm_cations['Mg_Amp_13_cat']+norm_cations['Fe2_calc'])
    norm_cations['Mgno_FeT']=norm_cations['Mg_Amp_13_cat']/(norm_cations['Mg_Amp_13_cat']+norm_cations['Fet_Amp_13_cat'])

    Low_Mgno=100*norm_cations['Mgno_Fe2']<54
    norm_cations.loc[(Low_Mgno), 'Input_Check']=False
    norm_cations.loc[(Low_Mgno), 'Fail Msg']="Low Mg# (<54)"

    #Only ones that matter are low Ca, high Ca, BJ3>60, low B cations"



    # If Column CU<1.5,"low Ca"
    Ca_low=norm_cations['Ca_Amp_13_cat']<1.5
    norm_cations.loc[(Ca_low), 'Input_Check']=False
    norm_cations.loc[(Ca_low), 'Fail Msg']="Low Ca (<1.5)"

    # If Column CU>2.05, "high Ca"
    Ca_high=norm_cations['Ca_Amp_13_cat']>2.05
    norm_cations.loc[(Ca_high), 'Input_Check']=False
    norm_cations.loc[(Ca_high), 'Fail Msg']="High Ca (>2.05)"

    # Check that CW<1.99, else "Low B cations"
    norm_cations['Na_calc']=2-norm_cations['Ca_Amp_13_cat']
    Ca_greaterthanNa=norm_cations['Na_Amp_13_cat']<(2-norm_cations['Ca_Amp_13_cat'])
    norm_cations.loc[(Ca_greaterthanNa), 'Na_calc']=norm_cations['Na_Amp_13_cat']
    norm_cations['B_Sum']=norm_cations['Na_calc']+norm_cations['Ca_Amp_13_cat']


    Low_B_Cations=norm_cations['B_Sum']<1.99
    norm_cations.loc[(Low_B_Cations), 'Input_Check']=False
    norm_cations.loc[(Low_B_Cations), 'Fail Msg']="Low B Cations"

    # Printing composition
    norm_cations['A_Sum']=norm_cations['Na_A']+norm_cations['K_A']
    norm_cations['class']="N/A"
    # Fix to be explicit
    norm_cations['classification'] = ""
    # If <1.5, low Ca,
    lowCa=norm_cations['Ca_B']<1.5
    norm_cations.loc[(lowCa), 'classification']="low-Ca"



    # Else, if high Ca, If Mgno<5, its low Mg
    LowMgno=norm_cations['Mgno_Fe2']<0.5
    norm_cations.loc[((~lowCa)&(LowMgno)), 'classification']="low-Mg"
    # Else, if Si>6.5, its a Mg-hornblende
    MgHbl=norm_cations['Si_T']>=6.5
    norm_cations.loc[((~lowCa)&(~LowMgno)&(MgHbl)), 'classification']="Mg-Hornblende"
    # Else if Ti_C >0.5, Kaerstiertei
    Kaer=norm_cations['Ti_C']>0.5
    norm_cations.loc[((~lowCa)&(~LowMgno)&(~MgHbl)&(Kaer)), 'classification']="kaersutite"
    # Else if A_Sum<0.5, Tschermakitic pargasite
    Tsh=norm_cations['A_Sum']<0.5
    norm_cations.loc[((~lowCa)&(~LowMgno)&(~MgHbl)&(~Kaer)&(Tsh)), 'classification']="Tschermakitic pargasite"
    # Else if Fe3+>AL VI C sum, Mg-hastingsite
    MgHast=(norm_cations['Fe3_calc']>norm_cations['Al_VI_C'])
    norm_cations.loc[((~lowCa)&(~LowMgno)&(~MgHbl)&(~Kaer)&(~Tsh)&(MgHast)), 'classification']="Mg-hastingsite"
    # Else, its Pargasite
    norm_cations.loc[((~lowCa)&(~LowMgno)&(~MgHbl)&(~Kaer)&(~Tsh)&(~MgHast)), 'classification']="Pargasite"






    return norm_cations










def calculate_amp_liq_mgno_hyd(liq_comps, amp_comps):
    liq_comps_hy = calculate_hydrous_cat_fractions_liquid(liq_comps=liq_comps)
    MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
    Kd=((MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/
    (liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd']))
    return Kd

def calculate_amp_liq_mgno_anhyd(liq_comps, amp_comps):
    liq_comps_hy = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps)
    MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
    Kd=((MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/
    (liq_comps_hy['FeOt_Liq_mol_frac']/liq_comps_hy['MgO_Liq_mol_frac']))
    return Kd


## These functions calculate the additional amphibole site things needed for certain barometers

def get_amp_sites_from_input(amp_comps):
    """
    get amp_sites from amp_comps input from import_excel() function.
    """
    amp_amfu_df=calculate_23oxygens_amphibole(amp_comps)
    amp_sites=get_amp_sites_leake(amp_amfu_df)
 #   out=pd.concat([amp_sites], axis=1)

    return amp_sites

def calculate_cpx_sites_from_input_not_cpx(dfin, col_drop):
    cpx_comps_lie=dfin.copy()
    cpx_comps_lie.columns = [col.replace(col_drop, '_Cpx') for col in cpx_comps_lie.columns]
    cpx_sites=calculate_clinopyroxene_components(cpx_comps=cpx_comps_lie)
    return cpx_sites

def get_amp_sites_from_input_not_amp(dfin, col_drop):
    """
    get amp_sites from amp_comps input from import_excel() function.
    """
    amp_comps_lie=dfin.copy()
    amp_comps_lie.columns = [col.replace(col_drop, '_Amp') for col in amp_comps_lie.columns]
    amp_amfu_df=calculate_23oxygens_amphibole(amp_comps_lie)
    amp_sites=get_amp_sites_leake(amp_amfu_df)
 #   out=pd.concat([amp_sites], axis=1)

    return amp_sites


def get_amp_sites_leake(amp_apfu_df):
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



    norm_cations = amp_apfu_df.copy()

    # Take unambigous ones and allocate them, set everything else to zero


    norm_cations['Si_T']=norm_cations['Si_Amp_cat_23ox']
    norm_cations['Al_T']=0
    norm_cations['Al_C']=0
    norm_cations['Ti_C']=norm_cations['Ti_Amp_cat_23ox']
    norm_cations['Mg_C']=0
    norm_cations['Fe_C']=0
    norm_cations['Mn_C']=0
    norm_cations['Cr_C']=norm_cations['Cr_Amp_cat_23ox']
    norm_cations['Mg_B']=0
    norm_cations['Fe_B']=0
    norm_cations['Mn_B']=0
    norm_cations['Na_B']=0
    norm_cations['Ca_B']=norm_cations['Ca_Amp_cat_23ox']
    norm_cations['Na_A']=0
    norm_cations['K_A']=norm_cations['K_Amp_cat_23ox']
    norm_cations['Ca_A']=0 # This is very ambigous, Leake have Ca A in some of their plots, but no way to put it in A based on workflow of site allocation.



    # 5a) Leake T Sites. Place all Si here, if Si<8, fill rest of T with Al.

    #If Si + Al is greater than 8, need to split Al between Al_T and Al_C as it can't all go here
    Si_Ti_sum_gr8=(norm_cations['Si_Amp_cat_23ox']+norm_cations['Al_Amp_cat_23ox'])>8
    # Calculate the amount of Ti sites left to fill after putting Si in. dont need to worry
    # about overallocating Ti, as already know they sum to more than 8.
    Al_T_Si_Ti_sum_gr8=8-norm_cations['Si_Amp_cat_23ox']
    # Got to be careful here, if Si on 23ox is >8, Al_T_Si_Ti_sum_gr8 ends up negative. Add another check
    Si_l_8=(norm_cations['Si_Amp_cat_23ox']<=8)
    # If Si is less than 8 already, set Ti to the difference
    norm_cations.loc[(Si_Ti_sum_gr8&Si_l_8), 'Al_T']=Al_T_Si_Ti_sum_gr8
    # If Si is greater than 8, set Al_T to zero
    norm_cations.loc[(Si_Ti_sum_gr8&(~Si_l_8)), 'Al_T']=0


    # Put remaining Al
    norm_cations.loc[(Si_Ti_sum_gr8), 'Al_C']=norm_cations['Al_Amp_cat_23ox']-norm_cations['Al_T']

    #If Si+Al<8, put all Al in tetrahedlra sites
    Si_Ti_sum_less8=(norm_cations['Si_Amp_cat_23ox']+norm_cations['Al_Amp_cat_23ox'])<8
    norm_cations.loc[(Si_Ti_sum_less8), 'Al_T']=norm_cations['Al_Amp_cat_23ox']
    norm_cations.loc[(Si_Ti_sum_less8), 'Al_C']=0

    # 5b) Leake Octaherdal C sites.
    #already filled some with Al in lines above. Place Ti (unambg), Cr (unamb).
    # Fill sites with Mg, Fe, and Mn to bring total to 5.

    # If Sites sum to less than 5, place all elements here
    Al_Ti_Cr_Mg_Fe_Mn_less5=(norm_cations['Al_C']+norm_cations['Ti_Amp_cat_23ox']+norm_cations['Fet_Amp_cat_23ox']
    +norm_cations['Cr_Amp_cat_23ox']+norm_cations['Mn_Amp_cat_23ox']+norm_cations['Mg_Amp_cat_23ox'])<5

    norm_cations.loc[(Al_Ti_Cr_Mg_Fe_Mn_less5), 'Fe_C']=norm_cations['Fet_Amp_cat_23ox']
    norm_cations.loc[(Al_Ti_Cr_Mg_Fe_Mn_less5), 'Mn_C']=norm_cations['Mn_Amp_cat_23ox']
    norm_cations.loc[(Al_Ti_Cr_Mg_Fe_Mn_less5), 'Mg_C']=norm_cations['Mg_Amp_cat_23ox']


    #If sites sum to more than 5, after placing Ti and Cr here, fill rest with Mg, Fe and Mn
    Al_Ti_Cr_Mg_Fe_Mn_more5=(norm_cations['Al_C']+norm_cations['Ti_Amp_cat_23ox']+norm_cations['Fet_Amp_cat_23ox']
    +norm_cations['Cr_Amp_cat_23ox']+norm_cations['Mn_Amp_cat_23ox']+norm_cations['Mg_Amp_cat_23ox'])>5

    # First, check if Al+Cr+Ti sum to 5. If not, allocate Mg
    sum_C_Al_Cr_Ti=norm_cations['Al_C']+norm_cations['Cr_C']+norm_cations['Ti_C']
    check_C_Al_Cr_Ti=sum_C_Al_Cr_Ti<5
    norm_cations.loc[((Al_Ti_Cr_Mg_Fe_Mn_more5)&(check_C_Al_Cr_Ti)), 'Mg_C']=norm_cations['Mg_Amp_cat_23ox']

    # Now check you haven't added too much MgO to take you over 5
    sum_C_Al_Cr_Ti_Mg=norm_cations['Al_C']+norm_cations['Cr_C']+norm_cations['Ti_C']+norm_cations['Mg_C']
    check_C_Al_Cr_Ti_Mg_low=sum_C_Al_Cr_Ti_Mg<=5
    check_C_Al_Cr_Ti_Mg_high=sum_C_Al_Cr_Ti_Mg>5
    # If sum is >5, replace Mg with only the magnesium left needed to get to 5.
    norm_cations.loc[((Al_Ti_Cr_Mg_Fe_Mn_more5)&(check_C_Al_Cr_Ti_Mg_high)), 'Mg_C']=5-sum_C_Al_Cr_Ti

    # Now check if you are back under 5 again,  ready to allocate Fe
    sum_C_Al_Cr_Ti_Mg2=norm_cations['Al_C']+norm_cations['Cr_C']+norm_cations['Ti_C']+norm_cations['Mg_C']
    check_C_Al_Cr_Ti_Mg_low2=sum_C_Al_Cr_Ti_Mg2<5
    norm_cations.loc[((Al_Ti_Cr_Mg_Fe_Mn_more5)&(check_C_Al_Cr_Ti_Mg_low2)), 'Fe_C']=norm_cations['Fet_Amp_cat_23ox']

    # Now check you haven't added too much FeO to take you over 5
    sum_C_Al_Cr_Ti_Mg_Fe=(norm_cations['Al_C']+norm_cations['Cr_C']
    +norm_cations['Ti_C']+norm_cations['Mg_C']+norm_cations['Fe_C'])
    check_C_Al_Cr_Ti_Mg_Fe_low=sum_C_Al_Cr_Ti_Mg_Fe<=5
    check_C_Al_Cr_Ti_Mg_Fe_high=sum_C_Al_Cr_Ti_Mg_Fe>5
    # If sum is >5, replace Fe with only the Fe left needed to get to 5.
    norm_cations.loc[((Al_Ti_Cr_Mg_Fe_Mn_more5)&(check_C_Al_Cr_Ti_Mg_Fe_high)), 'Fe_C']=5-sum_C_Al_Cr_Ti_Mg2

    # Now check if you are back under 5 again,  ready to allocate Mn
    sum_C_Al_Cr_Ti_Mg_Fe2=(norm_cations['Al_C']+norm_cations['Cr_C']+norm_cations['Ti_C']
    +norm_cations['Mg_C']+norm_cations['Fe_C'])
    check_C_Al_Cr_Ti_Mg_Fe_low2=sum_C_Al_Cr_Ti_Mg_Fe2<5
    norm_cations.loc[((Al_Ti_Cr_Mg_Fe_Mn_more5)&(check_C_Al_Cr_Ti_Mg_Fe_low2)), 'Mn_C']=norm_cations['Mn_Amp_cat_23ox']

    # Now check you haven't added too much Mn to take you over 5
    sum_C_Al_Cr_Ti_Mg_Fe_Mn=(norm_cations['Al_C']+norm_cations['Cr_C']
    +norm_cations['Ti_C']+norm_cations['Mg_C']+norm_cations['Fe_C']+norm_cations['Mn_C'])
    check_C_Al_Cr_Ti_Mg_Fe_Mn_low=sum_C_Al_Cr_Ti_Mg_Fe_Mn<=5
    check_C_Al_Cr_Ti_Mg_Fe_Mn_high=sum_C_Al_Cr_Ti_Mg_Fe_Mn>5
    # If sum is >5, replace Mn with only the Mn left needed to get to 5.
    norm_cations.loc[((Al_Ti_Cr_Mg_Fe_Mn_more5)&(check_C_Al_Cr_Ti_Mg_Fe_Mn_high)), 'Mn_C']=5-sum_C_Al_Cr_Ti_Mg_Fe2



    # 5c) Leake. B Sites-
    # if any Mg, Fe, Mn or Ca remaining put here;
    Mg_remaining=norm_cations['Mg_Amp_cat_23ox']-norm_cations['Mg_C']
    Fe_remaining=norm_cations['Fet_Amp_cat_23ox']-norm_cations['Fe_C']
    Mn_remaining=norm_cations['Mn_Amp_cat_23ox']-norm_cations['Mn_C']

    norm_cations['Mg_B']= Mg_remaining
    norm_cations['Fe_B']= Fe_remaining
    norm_cations['Mn_B']= Mn_remaining

    # If B sites sum to less than 2, fill sites with Ca to bring total to 2

    # If B sites sum to less than 2, fill sites with Na to bring total to 2

    Sum_B=norm_cations['Mg_B']+norm_cations['Fe_B']+norm_cations['Mn_B']+norm_cations['Ca_B']
    Left_to_fill_B=2- Sum_B
    # Check there is actually enough Na to fill B this way fully, if so allocate the amount you need
    Enough_Na=norm_cations['Na_Amp_cat_23ox']>=Left_to_fill_B
    norm_cations.loc[((Sum_B<2)&(Enough_Na)), 'Na_B']=Left_to_fill_B
    #  If there isn't enough Na2O, allocate all Na2O you have
    norm_cations.loc[((Sum_B<2)&(~Enough_Na)), 'Na_B']=norm_cations['Na_Amp_cat_23ox']
    Na_left_AfterB=norm_cations['Na_Amp_cat_23ox']-norm_cations['Na_B']

    # A sites
    norm_cations['K_A']=norm_cations['K_Amp_cat_23ox']
    norm_cations['Na_A']=Na_left_AfterB

    norm_cations['Sum_T']=norm_cations['Al_T']+norm_cations['Si_T']
    norm_cations['Sum_C']=(norm_cations['Al_C']+norm_cations['Cr_C']+norm_cations['Mg_C']
    +norm_cations['Fe_C']+norm_cations['Mn_C'])
    norm_cations['Sum_B']=(norm_cations['Mg_B']+norm_cations['Fe_B']+norm_cations['Mn_B']
    +norm_cations['Ca_B']+norm_cations['Na_B'])
    norm_cations['Sum_A']=norm_cations['K_A']+norm_cations['Na_A']

    norm_cations['cation_sum_All'] = norm_cations['cation_sum_Si_Ca'] + \
        norm_cations['Na_Amp_cat_23ox'] + +norm_cations['K_Amp_cat_23ox']
    norm_cations['Mgno_Amp']=norm_cations['Mg_Amp_cat_23ox']/(norm_cations['Mg_Amp_cat_23ox']
    +norm_cations['Fet_Amp_cat_23ox'])

    #norm_cations['factor']=np.max()
    # Ones being used, uses FW for Si, Fy,
    return norm_cations

def get_amp_sites_avferric_zhang(amp_comps):
    norm_cations=get_amp_sites_from_input(amp_comps)

    # Taken from Zhang et al. 2017 Supplement
    norm_cations['factor_8SiAl']=8/(norm_cations['Si_Amp_cat_23ox']+norm_cations['Al_Amp_cat_23ox'])

    norm_cations['factor_15eK']=(15/(norm_cations['Si_Amp_cat_23ox']+norm_cations['Ti_Amp_cat_23ox']
    +norm_cations['Al_Amp_cat_23ox']+norm_cations['Cr_Amp_cat_23ox']++norm_cations['Fet_Amp_cat_23ox']
    +norm_cations['Mg_Amp_cat_23ox']+norm_cations['Ca_Amp_cat_23ox']+norm_cations['Mn_Amp_cat_23ox']
    +norm_cations['Na_Amp_cat_23ox']))

    norm_cations['factor_13eCNK']=13/((norm_cations['Si_Amp_cat_23ox']+norm_cations['Ti_Amp_cat_23ox']
    +norm_cations['Al_Amp_cat_23ox']+norm_cations['Cr_Amp_cat_23ox']+norm_cations['Fet_Amp_cat_23ox']
    +norm_cations['Mg_Amp_cat_23ox']+norm_cations['Ca_Amp_cat_23ox']+norm_cations['Mn_Amp_cat_23ox']))

    norm_cations['All ferric']=23/(23+(0.5*norm_cations['Fet_Amp_cat_23ox']))

    # Minimum Factors
    norm_cations['8Si_Min']=8/norm_cations['Si_Amp_cat_23ox']

    norm_cations['16CAT_Min']=(16/(norm_cations['Si_Amp_cat_23ox']+norm_cations['Ti_Amp_cat_23ox']
    +norm_cations['Al_Amp_cat_23ox']+norm_cations['Cr_Amp_cat_23ox']++norm_cations['Fet_Amp_cat_23ox']
    +norm_cations['Mg_Amp_cat_23ox']+norm_cations['Ca_Amp_cat_23ox']+norm_cations['Mn_Amp_cat_23ox']
    +norm_cations['Na_Amp_cat_23ox']+norm_cations['K_Amp_cat_23ox']))

    norm_cations['15eNK_Min']=15/((norm_cations['Si_Amp_cat_23ox']+norm_cations['Ti_Amp_cat_23ox']
    +norm_cations['Al_Amp_cat_23ox']+norm_cations['Cr_Amp_cat_23ox']+norm_cations['Fet_Amp_cat_23ox']
    +norm_cations['Mg_Amp_cat_23ox']+norm_cations['Ca_Amp_cat_23ox']+norm_cations['Mn_Amp_cat_23ox']))

    norm_cations['Min_MinFactor']=norm_cations[['8Si_Min', '16CAT_Min', '15eNK_Min']].min(axis=1)


    #If Min_MinFactor<1, allocate to min factor
    norm_cations['Min_factor']=norm_cations['Min_MinFactor']
    Min_Min_Factor_g1=norm_cations['Min_MinFactor']>1
    norm_cations.loc[Min_Min_Factor_g1, 'Min_factor']=1

    norm_cations['Max_factor']=norm_cations[['factor_8SiAl', 'factor_15eK', 'factor_13eCNK', 'All ferric']].max(axis=1)


    norm_cations['Av_factor']=0.5*(norm_cations['Max_factor']+norm_cations['Min_factor'])


    # Then things times by factors
    Si_factor=norm_cations['Av_factor']*norm_cations['Si_Amp_cat_23ox']
    Ti_factor=norm_cations['Av_factor']*norm_cations['Ti_Amp_cat_23ox']
    Al_factor=norm_cations['Av_factor']*norm_cations['Al_Amp_cat_23ox']
    Cr_factor=norm_cations['Av_factor']*norm_cations['Cr_Amp_cat_23ox']
    Fe_factor=norm_cations['Av_factor']*norm_cations['Fet_Amp_cat_23ox']
    Mg_factor=norm_cations['Av_factor']*norm_cations['Mg_Amp_cat_23ox']
    Ca_factor=norm_cations['Av_factor']*norm_cations['Ca_Amp_cat_23ox']
    Mn_factor=norm_cations['Av_factor']*norm_cations['Mn_Amp_cat_23ox']
    Na_factor=norm_cations['Av_factor']*norm_cations['Na_Amp_cat_23ox']
    K_factor=norm_cations['Av_factor']*norm_cations['K_Amp_cat_23ox']


    norm_cations['Si_T_ideal']=norm_cations['Av_factor']*norm_cations['Si_Amp_cat_23ox']
    # Allocate Al to Tetrahedral. If 8-Si_T< Al_factor, equal to 8-Si_T

    norm_cations['Al_IV_T_ideal']=8-norm_cations['Si_T_ideal']
    # But if 8-SiT>Al factor, allocate all Al_Factor
    eight_m_siT_g=(8-norm_cations['Si_T_ideal'])>Al_factor
    norm_cations.loc[ eight_m_siT_g, 'Al_IV_T_ideal']=Al_factor
    # And if SiT>8, allocate to zero
    Si_T_g_8= norm_cations['Si_T_ideal']>8
    norm_cations.loc[Si_T_g_8, 'Al_IV_T_ideal']=0

    # Ti sites, if Si + Al>8, allocate 0 (as all filled up)
    norm_cations['Ti_T_ideal']=0
    # If Si and Al<8, if 8-Si-Al is > Ti_factor, Ti Factor (as all Ti can go in site)
    Si_Al_l8=(norm_cations['Si_T_ideal']+norm_cations['Al_IV_T_ideal'])<8
    eight_Si_Ti_gTiFactor=(8-norm_cations['Si_T_ideal']-norm_cations['Al_IV_T_ideal'])>Ti_factor
    norm_cations.loc[(( Si_Al_l8)&(eight_Si_Ti_gTiFactor)), 'Ti_T_ideal']=Ti_factor
    norm_cations.loc[(( Si_Al_l8)&(~eight_Si_Ti_gTiFactor)), 'Ti_T_ideal']=(8-norm_cations['Si_T_ideal']-norm_cations['Al_IV_T_ideal'])

    # Al VI C site
    norm_cations['Al_VI_C_ideal']=Al_factor-norm_cations['Al_IV_T_ideal']
    # Unless Alfactor-Al_VI equal to or less than zero, in which case none left
    Alfactor_Al_l0=(Al_factor-norm_cations['Al_IV_T_ideal'])<=0
    norm_cations.loc[(Alfactor_Al_l0), 'Al_VI_C_ideal']=0

    # Ti C site. If Ti Factor + Al_VI_C_ideal<5, equal to Ti Factor
    norm_cations['Ti_C_ideal']=Ti_factor
    # Else if >5, equal to 5-Al VI
    Ti_fac_Al_VI_g5=(Ti_factor+ norm_cations['Al_VI_C_ideal'])>5
    norm_cations.loc[(Ti_fac_Al_VI_g5), 'Ti_C_ideal']=5-norm_cations['Al_VI_C_ideal']

    # Cr C site. If Al_C + Ti_C + Cr Factor . Equal to Cr factor
    norm_cations['Cr_C_ideal']=Cr_factor
    C_Al_Ti_Cr_g5=(norm_cations['Ti_C_ideal']+ norm_cations['Al_VI_C_ideal']+Cr_factor)>5
    norm_cations.loc[(C_Al_Ti_Cr_g5), 'Cr_C_ideal']=5-norm_cations['Al_VI_C_ideal']-norm_cations['Ti_C_ideal']

    #  Fe3 C site
    NewSum=(Si_factor*2+Ti_factor*2+Al_factor*3/2+Cr_factor*3/2+Fe_factor
    +Mg_factor+Ca_factor+Mn_factor+Na_factor/2+K_factor/2)
    norm_cations['Fe3_C_ideal']=(23-NewSum)*2

    # Mg C site - If Al, Ti, Fe3, Cr already 5, set to zero
    norm_cations['Mg_C_ideal']=0

    # If sum not 5, if sum + Mg<5, allocate Mg factor, else do 5- sum
    Sum_beforeMgC=(norm_cations['Ti_C_ideal']+ norm_cations['Al_VI_C_ideal']
    +norm_cations['Cr_C_ideal']+norm_cations['Fe3_C_ideal'])
    Sum_beforeMgC_l5=Sum_beforeMgC<5
    Sum_beforeMgC_Mg_l5=(Sum_beforeMgC+Mg_factor)<5
    norm_cations.loc[((Sum_beforeMgC_l5)&(Sum_beforeMgC_Mg_l5)), 'Mg_C_ideal']=Mg_factor
    norm_cations.loc[((Sum_beforeMgC_l5)&(~Sum_beforeMgC_Mg_l5)), 'Mg_C_ideal']=5-Sum_beforeMgC_l5

    # Fe2_C site. If revious sum>5, alocate zero
    norm_cations['Fe2_C_ideal']=0
    # If previous sum<5
    Sum_beforeFeC=(norm_cations['Ti_C_ideal']+ norm_cations['Al_VI_C_ideal']
    +norm_cations['Cr_C_ideal']+norm_cations['Fe3_C_ideal']+norm_cations['Mg_C_ideal'])
    Sum_beforeFeC_l5=Sum_beforeFeC<5
    Sum_beforeFeC_Fe_l5=(Sum_beforeFeC+(Fe_factor-norm_cations['Fe3_C_ideal']))<5
    norm_cations.loc[((Sum_beforeFeC_l5)&(Sum_beforeFeC_Fe_l5)), 'Fe2_C_ideal']=Fe_factor-norm_cations['Fe3_C_ideal']
    norm_cations.loc[((Sum_beforeFeC_l5)&(~Sum_beforeFeC_Fe_l5)), 'Fe2_C_ideal']=5- Sum_beforeFeC

    # Mn Site, if sum>=5, set to zero
    norm_cations['Mn_C_ideal']=0
    # IF previous sum <5
    Sum_beforeMnC=(norm_cations['Ti_C_ideal']+ norm_cations['Al_VI_C_ideal']
    +norm_cations['Cr_C_ideal']+norm_cations['Fe3_C_ideal']+norm_cations['Mg_C_ideal']+norm_cations['Fe2_C_ideal'])
    Sum_beforeMnC_l5=Sum_beforeMnC<5
    Sum_beforeMnC_Mn_l5=(Sum_beforeMnC+Mn_factor)<5
    norm_cations.loc[((Sum_beforeMnC_l5)&(Sum_beforeMnC_Mn_l5)), 'Mn_C_ideal']=Mn_factor
    norm_cations.loc[((Sum_beforeMnC_l5)&(~Sum_beforeMnC_Mn_l5)), 'Mn_C_ideal']=5-Sum_beforeMnC

    # Mg B site, if any Mg left, put here.
    norm_cations['Mg_B_ideal']=0
    Mg_left_B=(Mg_factor-norm_cations['Mg_C_ideal'])>0
    norm_cations.loc[(Mg_left_B), 'Mg_B_ideal']=Mg_factor-norm_cations['Mg_C_ideal']

    # Fe B site, if any Fe2 left, but here
    norm_cations['Fe2_B_ideal']=0
    Fe2_left_B=(Fe_factor-norm_cations['Fe2_C_ideal']-norm_cations['Fe3_C_ideal'])>0
    norm_cations.loc[(Fe2_left_B), 'Fe2_B_ideal']=Fe_factor-norm_cations['Fe2_C_ideal']-norm_cations['Fe3_C_ideal']


    # Mn B site, if any Mn left, put here.
    norm_cations['Mn_B_ideal']=0
    Mn_left_B=(Mn_factor-norm_cations['Mn_C_ideal'])>0
    norm_cations.loc[(Mn_left_B), 'Mn_B_ideal']=Mn_factor-norm_cations['Mn_C_ideal']

    # Ca B site, all Ca
    norm_cations['Ca_B_ideal']=Ca_factor

    # Na, if Mg+Fe+Mn+Ca B >2, 0
    norm_cations['Na_B_ideal']=0
    Sum_beforeNa=(norm_cations['Mn_B_ideal']+norm_cations['Fe2_B_ideal']+norm_cations['Mg_B_ideal']+norm_cations['Ca_B_ideal'])
    Sum_beforeNa_l2=Sum_beforeNa<2
    Sum_before_Na_Na_l2=(Sum_beforeNa+Na_factor)<2
    norm_cations.loc[((Sum_beforeNa_l2)&(Sum_before_Na_Na_l2)), 'Na_B_ideal']=Na_factor
    norm_cations.loc[((Sum_beforeNa_l2)&(~Sum_before_Na_Na_l2)), 'Na_B_ideal']=2-Sum_beforeNa

    # Na_A - any Na left
    norm_cations['Na_A_ideal']=Na_factor-norm_cations['Na_B_ideal']
    norm_cations['K_A_ideal']=K_factor


















    return norm_cations

def get_amp_sites2(amp_apfu_df):
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
        newnames.append(norm_cations.split('_')[0])

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
    Sum_T=Al_T+Si_T
    Sum_C=Al_C+Cr_C+Mg_C+Fe_C+Mn_C
    Sum_B=Mg_B+Fe_B+Mn_B+Ca_B+Na_B
    Sum_A=K_A+Na_A


    site_vals = np.array([Si_T, Al_T, Al_C, Ti_C, Mg_C, Fe_C, Mn_C, Cr_C, Mg_B,
    Fe_B, Mn_B, Na_B, Ca_B, Na_A, K_A])
    sites_df = pd.DataFrame(site_vals.T, columns=['Si_T', 'Al_T', 'Al_C', 'Ti_C',
     'Mg_C', 'Fe_C', 'Mn_C', 'Cr_C', 'Mg_B', 'Fe_B', 'Mn_B', 'Na_B', 'Ca_B', 'Na_A', 'K_A'],
                            index=amp_apfu_df.index
                            )
    sites_df['Sum_T'] =   Sum_T
    sites_df['Sum_C'] =   Sum_C
    sites_df['Sum_B'] =   Sum_B
    sites_df['Sum_A'] =   Sum_A

    return sites_df


def amp_components_ferric_ferrous(sites_df, norm_cations):
    """
    amp_components_ferric_ferrous calculates the Fe3+ and Fe2+ apfu values of
    amphibole and adjusts the generic stoichiometry such that charge balance is
    maintained. This is based off the "f parameters" listed in Holland and Blundy
    (1994). Following Mutch et al. (2016) spreadsheet.

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
    f8 = 12.9 / (sites_df.sum(axis='columns') - (sites_df['Na_A'] + sites_df['Na_B'] +
    sites_df['K_A']) - (sites_df['Mn_B'] + sites_df['Mn_C']) - sites_df['Ca_B'])
    f9 = 36 / (46 - (sites_df['Al_T'] + sites_df['Al_C']
                     ) - sites_df['Si_T'] - sites_df['Ti_C'])
    f10 = 46 / ((sites_df['Fe_C'] + sites_df['Fe_B']) + 46)
    fb = pd.DataFrame({'f6': f6, 'f7': f7, 'f8': f8, 'f9': f9, 'f10': f10, })

    f_ave = (fa.min(axis='columns') + fb.max(axis='columns')) / 2
    # f_ave = (2/3)*fa.min(axis = 'columns') + (1/3)*fb.max(axis = 'columns')

    norm_cations_hb = norm_cations.multiply(f_ave, axis='rows')
    norm_cations_hb['Fe2O3_Amp_cat_23ox'] = 46 * (1 - f_ave)
    norm_cations_hb['FeO_Amp_cat_23ox'] = norm_cations_hb['Fet_Amp_cat_23ox'] - \
        norm_cations_hb['Fe2O3_Amp_cat_23ox']
    norm_cations_hb.drop(columns=['Fet_Amp_cat_23ox', 'oxy_renorm_factor',
                         'cation_sum_Si_Mg', 'cation_sum_Si_Ca', 'cation_sum_All'], inplace=True)
    # newnames = []
    # for name in norm_cations_hb.columns.tolist():
    #     newnames.append(norm_cations.split('_')[0])

    #norm_cations_hb.columns = newnames

    return norm_cations_hb, f_ave


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
        Si_T[i] = amp_apfu_df.loc[sample, 'Si_Amp_cat_23ox']
        K_A[i] = amp_apfu_df.loc[sample, 'K_Amp_cat_23ox']
        Ti_C[i] = amp_apfu_df.loc[sample, 'Ti_Amp_cat_23ox']
        Ca_B[i] = amp_apfu_df.loc[sample, 'Ca_Amp_cat_23ox']
        Cr_C[i] = amp_apfu_df.loc[sample, 'Cr_Amp_cat_23ox']
        Fe3_C[i] = amp_apfu_df.loc[sample, 'Fe2O3_Amp_cat_23ox']

        if Si_T[i] + amp_apfu_df.loc[sample, 'Al_Amp_cat_23ox'] > 8:
            Al_T[i] = 8 - amp_apfu_df.loc[sample, 'Si_Amp_cat_23ox']
            Al_C[i] = amp_apfu_df.loc[sample, 'Si_Amp_cat_23ox'] + \
                amp_apfu_df.loc[sample, 'Al_Amp_cat_23ox'] - 8
        else:
            Al_T[i] = amp_apfu_df.loc[sample, 'Al_Amp_cat_23ox']
            Al_C[i] = 0
        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + \
                amp_apfu_df.loc[sample, 'Mg_Amp_cat_23ox'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                Fe3_C[i] + amp_apfu_df.loc[sample, 'Mg_Amp_cat_23ox'] - 5
        else:
            Mg_C[i] = amp_apfu_df.loc[sample, 'Mg_Amp_cat_23ox']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i] > 5:
            Fe2_C[i] = 0
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO_Amp_cat_23ox']
        else:
            Fe2_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i])
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO_Amp_cat_23ox'] - Fe2_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe3_C[i] + Fe2_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = amp_apfu_df.loc[sample, 'Mn_Amp_cat_23ox']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] +
                           Mg_C[i] + Fe2_C[i] + Fe3_C[i])
            Mn_B[i] = amp_apfu_df.loc[sample, 'Mn_Amp_cat_23ox'] - Mn_C[i]

        if Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i] > 2:
            Na_B[i] = 0
            Na_A[i] = amp_apfu_df.loc[sample, 'Na_Amp_cat_23ox']
        else:
            Na_B[i] = 2 - (Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i])
            Na_A[i] = amp_apfu_df.loc[sample, 'Na_Amp_cat_23ox'] - Na_B[i]

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
    norm_cations = amp_apfu_df.copy()
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
        Si_T[i] = norm_cations.loc[sample, 'Si_Amp_cat_23ox']
        K_A[i] = norm_cations.loc[sample, 'K_Amp_cat_23ox']
        Ti_C[i] = norm_cations.loc[sample, 'Ti_Amp_cat_23ox']
        Ca_B[i] = norm_cations.loc[sample, 'Ca_Amp_cat_23ox']
        Cr_C[i] = norm_cations.loc[sample, 'Cr_Amp_cat_23ox']

        # site ambiguous cations. Follows Leake et al., (1997) logic
        if Si_T[i] + norm_cations.loc[sample, 'Al_Amp_cat_23ox'] > 8:
            Al_T[i] = 8 - norm_cations.loc[sample, 'Si_Amp_cat_23ox']
            Al_C[i] = norm_cations.loc[sample, 'Si_Amp_cat_23ox'] + \
                norm_cations.loc[sample, 'Al_Amp_cat_23ox'] - 8
        else:
            Al_T[i] = norm_cations.loc[sample, 'Al_Amp_cat_23ox']
            Al_C[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + norm_cations.loc[sample, 'Mg_Amp_cat_23ox'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                norm_cations.loc[sample, 'Mg_Amp_cat_23ox'] - 5
        else:
            Mg_C[i] = norm_cations.loc[sample, 'Mg_Amp_cat_23ox']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] > 5:
            Fe_C[i] = 0
            Fe_B[i] = norm_cations.loc[sample, 'Fet_Amp_cat_23ox']
        else:
            Fe_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i])
            Fe_B[i] = norm_cations.loc[sample, 'Fet_Amp_cat_23ox'] - Fe_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = norm_cations.loc[sample, 'Mn_Amp_cat_23ox']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe_C[i])
            Mn_B[i] = norm_cations.loc[sample, 'Mn_Amp_cat_23ox'] - Mn_C[i]

        if Mg_B[i] + Fe_B[i] + Mn_B[i] + Ca_B[i] + \
                amp_apfu_df['Na_Amp_cat_23ox'].iloc[i] > 2:
            Na_B[i] = 2 - (Mg_B[i] + Fe_B[i] + Mn_B[i] + Ca_B[i])
            Na_A[i] = amp_apfu_df['Na_Amp_cat_23ox'].iloc[i] - Na_B[i]
        else:
            Na_B[i] = amp_apfu_df['Na_Amp_cat_23ox'].iloc[i]
            # Euan has as if Na A >0, set as 0, otherwise, =Na cations 23 O -
            # Na from A site. Ask jordan where he got this from.
            Na_A[i] = amp_apfu_df['Na_Amp_cat_23ox'].iloc[i] - Na_B[i]


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
    norm_cations_hb['Fe2O3_Amp_cat_23ox'] = 46 * (1 - f_ave)
    norm_cations_hb['FeO_Amp_cat_23ox'] = norm_cations_hb['Fet_Amp_cat_23ox'] - \
        norm_cations_hb['Fe2O3_Amp_cat_23ox']
    norm_cations_hb.drop(columns=['Fet_Amp_cat_23ox', 'oxy_renorm_factor',
                         'cation_sum_Si_Mg', 'cation_sum_Si_Ca', 'cation_sum_All'], inplace=True)


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
        Si_T[i] = amp_apfu_df.loc[sample, 'Si_Amp_cat_23ox']
        K_A[i] = amp_apfu_df.loc[sample, 'K_Amp_cat_23ox']
        Ti_C[i] = amp_apfu_df.loc[sample, 'Ti_Amp_cat_23ox']
        Ca_B[i] = amp_apfu_df.loc[sample, 'Ca_Amp_cat_23ox']
        Cr_C[i] = amp_apfu_df.loc[sample, 'Cr_Amp_cat_23ox']
        Fe3_C[i] = amp_apfu_df.loc[sample, 'Fe2O3_Amp_cat_23ox']

        if Si_T[i] + amp_apfu_df.loc[sample, 'Al_Amp_cat_23ox'] > 8:
            Al_T[i] = 8 - amp_apfu_df.loc[sample, 'Si_Amp_cat_23ox']
            Al_C[i] = amp_apfu_df.loc[sample, 'Si_Amp_cat_23ox'] + \
                amp_apfu_df.loc[sample, 'Al_Amp_cat_23ox'] - 8
        else:
            Al_T[i] = amp_apfu_df.loc[sample, 'Al_Amp_cat_23ox']
            Al_C[i] = 0
        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + \
                amp_apfu_df.loc[sample, 'Mg_Amp_cat_23ox'] > 5:
            Mg_C[i] = 5 - Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i]
            Mg_B[i] = Al_C[i] + Ti_C[i] + Cr_C[i] + \
                Fe3_C[i] + amp_apfu_df.loc[sample, 'Mg_Amp_cat_23ox'] - 5
        else:
            Mg_C[i] = amp_apfu_df.loc[sample, 'Mg_Amp_cat_23ox']
            Mg_B[i] = 0

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i] > 5:
            Fe2_C[i] = 0
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO_Amp_cat_23ox']
        else:
            Fe2_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] + Fe3_C[i] + Mg_C[i])
            Fe2_B[i] = amp_apfu_df.loc[sample, 'FeO_Amp_cat_23ox'] - Fe2_C[i]

        if Al_C[i] + Ti_C[i] + Cr_C[i] + Mg_C[i] + Fe3_C[i] + Fe2_C[i] > 5:
            Mn_C[i] = 0
            Mn_B[i] = amp_apfu_df.loc[sample, 'Mn_Amp_cat_23ox']
        else:
            Mn_C[i] = 5 - (Al_C[i] + Ti_C[i] + Cr_C[i] +
                           Mg_C[i] + Fe2_C[i] + Fe3_C[i])
            Mn_B[i] = amp_apfu_df.loc[sample, 'Mn_Amp_cat_23ox'] - Mn_C[i]

        if Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i] + \
                amp_apfu_df.loc[sample, 'Na_Amp_cat_23ox'] > 2:
            Na_B[i] = 2 - (Mg_B[i] + Fe2_B[i] + Mn_B[i] + Ca_B[i])
            Na_A[i] = amp_apfu_df.loc[sample, 'Na_Amp_cat_23ox'] - Na_B[i]

        else:
            Na_B[i] = amp_apfu_df.loc[sample, 'Na_Amp_cat_23ox']
            # Euan has as if Na A >0, set as 0, otherwise, =Na cations 23 O -
            # Na from A site. Ask jordan where he got this from.
            Na_A[i] = amp_apfu_df.loc[sample, 'Na_Amp_cat_23ox'] - Na_B[i]

    site_vals = np.array([Si_T, Al_T, Al_C, Ti_C, Mg_C, Fe3_C, Fe2_C, Mn_C, Cr_C, Mg_B,
    Fe2_B, Mn_B, Na_B, Ca_B, Na_A, K_A])
    sites_df = pd.DataFrame(site_vals.T, columns=['Si_T', 'Al_T', 'Al_C', 'Ti_C',
     'Mg_C', 'Fe3_C', 'Fe2_C', 'Mn_C', 'Cr_C', 'Mg_B', 'Fe2_B', 'Mn_B', 'Na_B', 'Ca_B', 'Na_A', 'K_A'],
    index=amp_apfu_df.index)

    return sites_df





## Equilibrium tests clinopyroxene

def calculate_cpx_liq_eq_tests(*, meltmatch=None, liq_comps=None, cpx_comps=None,
                           Fe3Fet_Liq=None, P=None, T=None, sigma=1, Kd_Err=0.03):
    '''
    calculates Kd Fe-Mg, EnFs, DiHd, CaTs for cpx-liquid pairs

   Parameters
    -------

    cpx_comps: pandas.DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    meltmatch: pandas.DataFrame
        Combined Cpx-Liquid compositions. Used for "melt match" functionality.


    P_kbar: int, float, pandas.Series
        Pressure in kbar at which to evaluate equilibrium tests

    T: int, float, pandas.Series
        Temprature in K at which to evaluate equilibrium tests

    Fe3Fet_Liq: int, float, pandas.Series (optional)
        Fe3FeT ratio used to assess Kd Fe-Mg equilibrium between cpx and melt.
        If None, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    sigma: int or float
        determines sigma level at which to consider the DiHd, EnFs and CaTs tests laid out by Neave et al. (2019)

    Kd_Err: int or float
        deviation from the Kd value of Putirka (2008) for the equilibrium tests of Neave et al. (2019)

   Returns
    -------

    DataFrame
        Equilibrium tests, cation fractions, and inputted liquid and cpx compositions.


    '''

    if meltmatch is not None:
        Combo_liq_cpxs = meltmatch
    if liq_comps is not None and cpx_comps is not None:
        liq_comps_c = liq_comps.copy()
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
        else:
            print('Using Fe3FeT from input file to calculate Kd Fe-Mg')
        Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(
            cpx_comps=cpx_comps, liq_comps=liq_comps_c)

    Combo_liq_cpxs['P_kbar_calc'] = P
    Combo_liq_cpxs['T_K_calc'] = T

    # First up, Kd Fe-MG

     # Equation 35 of Putirka 2008 - theoretical Kd-Mg exchange coefficient
    Combo_liq_cpxs['Kd_Ideal_Put'] = np.exp(-0.107 - 1719 / T)  # eq 35
    ratioMasotta = Combo_liq_cpxs['Na_Liq_cat_frac'] / (
        Combo_liq_cpxs['Na_Liq_cat_frac'] + Combo_liq_cpxs['K_Liq_cat_frac'])

    # Masotta et al. For Alkali basalts
    Combo_liq_cpxs['Kd_Ideal_Masotta'] = np.exp(
        1.735 - 3056 / T - 1.668 * ratioMasotta)  # eq35 alk, for trachytes and phonolites
    Combo_liq_cpxs['Delta_Kd_Put2008'] = abs(
        Combo_liq_cpxs['Kd_Ideal_Put'] - Combo_liq_cpxs['Kd_Fe_Mg_Fe2'])
    Combo_liq_cpxs['Delta_Kd_Put2008_I_M'] = Combo_liq_cpxs['Kd_Ideal_Put'] - Combo_liq_cpxs['Kd_Fe_Mg_Fe2']
    Combo_liq_cpxs['Delta_Kd_Mas2013'] = abs(
        Combo_liq_cpxs['Kd_Ideal_Masotta'] - Combo_liq_cpxs['Kd_Fe_Mg_Fe2'])

    # Di Hd equilibrium

    # Putirka (1999) DiHd Eq3.1a
    Combo_liq_cpxs['DiHd_Pred_Put1999']=(np.exp(-9.8
    + 0.24*np.log(Combo_liq_cpxs['Ca_Liq_cat_frac']*(Combo_liq_cpxs['Fet_Liq_cat_frac']+Combo_liq_cpxs['Mg_Liq_cat_frac'])*
    Combo_liq_cpxs['Si_Liq_cat_frac']**2)+17558/T+8.7*np.log(T/1670)-4.61*10**3*(Combo_liq_cpxs['EnFs']**2/T))
    )
    Combo_liq_cpxs['Delta_DiHd_Put1999'] = abs(
        Combo_liq_cpxs['DiHd_1996'] - Combo_liq_cpxs['DiHd_Pred_Put1999'])
    Combo_liq_cpxs['Delta_DiHd_I_M_Put1999']=Combo_liq_cpxs['DiHd_Pred_Put1999']-Combo_liq_cpxs['DiHd_1996']

    # One labelled "new", presume from Putirka (2008)
    Combo_liq_cpxs['DiHd_Pred_P2008']=(np.exp(-0.482-0.439*np.log(Combo_liq_cpxs['Si_Liq_cat_frac'])
    +101.03*(Combo_liq_cpxs['Na_Liq_cat_frac']+Combo_liq_cpxs['K_Liq_cat_frac'])**3
    -51.69*P/T-3742.5*Combo_liq_cpxs['EnFs']**2/T) )
    Combo_liq_cpxs['Delta_DiHd_P2008'] = abs(
        Combo_liq_cpxs['DiHd_1996'] - Combo_liq_cpxs['DiHd_Pred_P2008'])



    # Mollo 2013 for DiHd - Eq7
    Combo_liq_cpxs['DiHd_Pred_Mollo13'] = (np.exp(-2.18 - 3.16 * Combo_liq_cpxs['Ti_Liq_cat_frac']
    - 0.365 * np.log(Combo_liq_cpxs['Al_Liq_cat_frac'].astype(float))
     + 0.05 * np.log(Combo_liq_cpxs['Mg_Liq_cat_frac']) - 3858.2 * (
                                                    Combo_liq_cpxs['EnFs']**2 / T) + (2107.4 / T)
                                                - 17.64 * P / T))
    Combo_liq_cpxs['Delta_DiHd_Mollo13'] = abs(
        Combo_liq_cpxs['DiHd_1996'] - Combo_liq_cpxs['DiHd_Pred_Mollo13'])
    Combo_liq_cpxs['Delta_DiHd_I_M_Mollo13']=Combo_liq_cpxs['DiHd_Pred_Mollo13']-Combo_liq_cpxs['DiHd_1996']

    # En Fs equilibrium


    # Putirka En Fs
    Combo_liq_cpxs['EnFs_Pred_Put1999']=( np.exp(-6.96+18438/T+8*np.log(T/1670)
    +0.66*np.log((Combo_liq_cpxs['Fet_Liq_cat_frac']+Combo_liq_cpxs['Mg_Liq_cat_frac'])**2*Combo_liq_cpxs['Si_Liq_cat_frac']**2)
    -5.1*10**3*(Combo_liq_cpxs['DiHd_1996']**2/T)+1.81*np.log(Combo_liq_cpxs['Si_Liq_cat_frac']) ))
    Combo_liq_cpxs['Delta_EnFs_Put1999'] = abs(
        Combo_liq_cpxs['EnFs'] - Combo_liq_cpxs['EnFs_Pred_Put1999'])
    Combo_liq_cpxs['Delta_EnFs_I_M_Put1999'] = Combo_liq_cpxs['EnFs_Pred_Put1999']-Combo_liq_cpxs['EnFs']




    # En Fs Mollo eq6

    Combo_liq_cpxs['EnFs_Pred_Mollo13'] = (np.exp(0.018 - 9.61 * Combo_liq_cpxs['Ca_Liq_cat_frac'] +
                                                7.46 *
                                                Combo_liq_cpxs['Mg_Liq_cat_frac'] *
                                                Combo_liq_cpxs['Si_Liq_cat_frac']
                                                - 0.34 *np.log(Combo_liq_cpxs['Al_Liq_cat_frac'].astype(float))
                                                - 3.78 * (Combo_liq_cpxs['Na_Liq_cat_frac'] + Combo_liq_cpxs['K_Liq_cat_frac']) -
                                                3737.3 * (Combo_liq_cpxs['DiHd_1996']**2) / T - 46.8 * P / T))
    Combo_liq_cpxs['Delta_EnFs_Mollo13'] = abs(
        Combo_liq_cpxs['EnFs'] - Combo_liq_cpxs['EnFs_Pred_Mollo13'])
    Combo_liq_cpxs['Delta_EnFs_I_M_Mollo13'] = Combo_liq_cpxs['EnFs_Pred_Mollo13']-Combo_liq_cpxs['EnFs']


    #  CaTs equilibrium (Mollo didnt release on of these)
    # This is equation 3.4 of Putirka
    Combo_liq_cpxs['CaTs_Pred_Put1999'] = (np.exp(2.58 + 0.12 * P / T - 9 * 10**(-7) * P**2 / T
    + 0.78 * np.log(Combo_liq_cpxs['Ca_Liq_cat_frac'].astype(float)
    * Combo_liq_cpxs['Al_Liq_cat_frac'].astype(float)**2
    * Combo_liq_cpxs['Si_Liq_cat_frac'].astype(float)) - 4.3 * 10**3 * (Combo_liq_cpxs['DiHd_1996']**2 / T)))

    Combo_liq_cpxs['Delta_CaTs_Put1999'] = abs(
        Combo_liq_cpxs['CaTs'] - Combo_liq_cpxs['CaTs_Pred_Put1999'])
    Combo_liq_cpxs['Delta_CaTs_I_M_Put1999'] =Combo_liq_cpxs['CaTs_Pred_Put1999']-Combo_liq_cpxs['CaTs']


    #CrCaTs component

    Combo_liq_cpxs['CrCaTs_Pred_Put1999'] = (np.exp(12.8) * Combo_liq_cpxs['Ca_Liq_cat_frac'] * (
        Combo_liq_cpxs['Cr_Liq_cat_frac']**2) * Combo_liq_cpxs['Si_Liq_cat_frac'])
    Combo_liq_cpxs['Delta_CrCaTs_Put1999']=abs(Combo_liq_cpxs['CrCaTs']-Combo_liq_cpxs['CrCaTs_Pred_Put1999'])
    Combo_liq_cpxs['Delta_CrCaTs_I_M_Put1999']=abs(Combo_liq_cpxs['CrCaTs']-Combo_liq_cpxs['CrCaTs_Pred_Put1999'])

    # CaTi component - Eq 3.6

    Combo_liq_cpxs['CaTi_Pred_Put1999']=( np.exp(5.1 + 0.52*np.log(Combo_liq_cpxs['Ca_Liq_cat_frac']*Combo_liq_cpxs['Ti_Liq_cat_frac']*Combo_liq_cpxs['Al_Liq_cat_frac']**2)
    +2.04*10**(3)* (Combo_liq_cpxs['DiHd_1996']**2 / T)- 6.2* Combo_liq_cpxs['Si_Liq_cat_frac']
    +42.5*Combo_liq_cpxs['Na_Liq_cat_frac']*Combo_liq_cpxs['Al_Liq_cat_frac']
    - 45.1*(Combo_liq_cpxs['Fet_Liq_cat_frac']
    +Combo_liq_cpxs['Mg_Liq_cat_frac'])*Combo_liq_cpxs['Al_Liq_cat_frac'] ))

    Combo_liq_cpxs['Delta_CaTi_Put1999']=abs(Combo_liq_cpxs['CaTi']-Combo_liq_cpxs['CaTi_Pred_Put1999'])
    Combo_liq_cpxs['Delta_CaTi_I_M_Put1999']=abs(Combo_liq_cpxs['CaTi']-Combo_liq_cpxs['CaTi_Pred_Put1999'])


    # Jd component Eq3.5
    Combo_liq_cpxs['Jd_Pred_Put1999']=(np.exp(-1.06+0.23*P/T-6*10**(-7)*P**2/T
    +1.02*np.log(Combo_liq_cpxs['Na_Liq_cat_frac']*Combo_liq_cpxs['Al_Liq_cat_frac']*Combo_liq_cpxs['Si_Liq_cat_frac']**2)
    -0.8*np.log(Combo_liq_cpxs['Al_Liq_cat_frac'])-2.2*np.log(Combo_liq_cpxs['Si_Liq_cat_frac'])))
    Combo_liq_cpxs['Delta_Jd_Put1999']=abs(Combo_liq_cpxs['Jd']-Combo_liq_cpxs['Jd_Pred_Put1999'])
    Combo_liq_cpxs['Delta_Jd_I_M_Put1999']=abs(Combo_liq_cpxs['Jd']-Combo_liq_cpxs['Jd_Pred_Put1999'])


    b = np.empty(len(Combo_liq_cpxs), dtype=bool)
    for i in range(0, len(Combo_liq_cpxs)):

        if ((Combo_liq_cpxs['Delta_Kd_Put2008'].iloc[i] < Kd_Err) & (Combo_liq_cpxs['Delta_EnFs_Mollo13'].iloc[i] < 0.05 * sigma) &
                (Combo_liq_cpxs['Delta_CaTs_Put1999'].iloc[i] < 0.03 * sigma)
                & (Combo_liq_cpxs['Delta_DiHd_Mollo13'].iloc[i] < 0.06 * sigma)):
            b[i] = True
        else:
            b[i] = False
    Combo_liq_cpxs.insert(1, "Eq Tests Neave2017?", b)

    cols_to_move = ['P_kbar_calc', 'T_K_calc', "Eq Tests Neave2017?",
                    'Delta_Kd_Put2008', 'Delta_Kd_Mas2013', 'Delta_EnFs_Mollo13', 'Delta_EnFs_Put1999',
                    'Delta_CaTs_Put1999', 'Delta_DiHd_Mollo13', 'Delta_DiHd_Put1999', 'Delta_CrCaTs_Put1999', 'Delta_CaTi_Put1999']
    Combo_liq_cpxs = Combo_liq_cpxs[cols_to_move +
                                    [col for col in Combo_liq_cpxs.columns if col not in cols_to_move]]

    return Combo_liq_cpxs.copy()


def calculate_cpx_opx_eq_tests(cpx_comps, opx_comps):
    '''
    Import Cpx and Opx compositions, assesses degree of Fe-Mg disequilibrium.
    Parameters
    -------

    cpx_comps: pandas.DataFrame
            Cpx compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    opx_comps: pandas.DataFrame
            Cpx compositions with column headings SiO2_Opx, MgO_Opx etc.
    Returns
    -------
    pandas DataFrame
        Return all opx and cpx components, as well as values for Kd-Fe Opx-Cpx.
    '''
    cpx_components = calculate_clinopyroxene_components(cpx_comps=cpx_comps)
    opx_components = calculate_orthopyroxene_components(opx_comps=opx_comps)
    two_pyx = pd.concat([cpx_components, opx_components], axis=1)
    two_pyx['En'] = (two_pyx['Fm2Si2O6'] * (two_pyx['Mg_Opx_cat_6ox'] / (two_pyx['Mg_Opx_cat_6ox'] +
                    two_pyx['Fet_Cpx_cat_6ox'] + two_pyx['Mn_Cpx_cat_6ox'])))
    two_pyx['Kd_Fe_Mg_Cpx_Opx'] = ((two_pyx['Fet_Cpx_cat_6ox'] / two_pyx['Mg_Cpx_cat_6ox'])) / (
        two_pyx['Fet_Opx_cat_6ox'] / two_pyx['Mg_Opx_cat_6ox'])

    Lindley_Fe3_Opx = (two_pyx['Na_Opx_cat_6ox'] + two_pyx['Al_IV_Opx_cat_6ox'] -
        two_pyx['Al_VI_Opx_cat_6ox'] - 2 * two_pyx['Ti_Opx_cat_6ox'] - two_pyx['Cr_Opx_cat_6ox'])
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    two_pyx['Lindley_Fe3_Opx']=Lindley_Fe3_Opx
    a_En_opx_mod = (((0.5 * two_pyx['Mg_Opx_cat_6ox'] / (0.5 * (two_pyx['Fet_Opx_cat_6ox'] - two_pyx['Lindley_Fe3_Opx'])
    + 0.5 * two_pyx['Mg_Opx_cat_6ox'] + two_pyx['Na_Opx_cat_6ox'] +two_pyx['Ca_Opx_cat_6ox'] + two_pyx['Mn_Opx_cat_6ox'])))
    * (0.5 * two_pyx['Mg_Opx_cat_6ox'] / (0.5 * two_pyx['Mg_Opx_cat_6ox'] + 0.5 * (two_pyx['Fet_Opx_cat_6ox']
    - two_pyx['Lindley_Fe3_Opx'])+ two_pyx['Ti_Opx_cat_6ox'] + two_pyx['Al_VI_Opx_cat_6ox']
    + two_pyx['Cr_Opx_cat_6ox'] + two_pyx['Lindley_Fe3_Opx'])))

    Lindley_Fe3_Cpx = two_pyx['Na_Cpx_cat_6ox'] + two_pyx['Al_IV_cat_6ox'] - \
        two_pyx['Al_VI_cat_6ox'] - 2 * two_pyx['Ti_Cpx_cat_6ox'] - two_pyx['Cr_Cpx_cat_6ox']
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    two_pyx['Lindley_Fe3_Cpx']=Lindley_Fe3_Cpx
    two_pyx['a_Di_cpx'] = two_pyx['Ca_Cpx_cat_6ox'] / (two_pyx['Ca_Cpx_cat_6ox'] + 0.5 * two_pyx['Mg_Cpx_cat_6ox']
    + 0.5 * (two_pyx['Fet_Cpx_cat_6ox'] - two_pyx['Lindley_Fe3_Cpx']) + two_pyx['Mn_Cpx_cat_6ox']
    + two_pyx['Na_Cpx_cat_6ox'])
    two_pyx['Kf'] = two_pyx['Ca_Opx_cat_6ox'] / (1 - two_pyx['Ca_Cpx_cat_6ox'])

    two_pyx['a_En_opx_mod'] = (((0.5 * two_pyx['Mg_Opx_cat_6ox'] / (0.5 * (two_pyx['Fet_Opx_cat_6ox']
    - two_pyx['Lindley_Fe3_Opx'])+ 0.5 * two_pyx['Mg_Opx_cat_6ox'] + two_pyx['Na_Opx_cat_6ox']
    +two_pyx['Ca_Opx_cat_6ox'] + two_pyx['Mn_Opx_cat_6ox'])))
    * (0.5 * two_pyx['Mg_Opx_cat_6ox'] / (0.5 * two_pyx['Mg_Opx_cat_6ox'] + 0.5 * (two_pyx['Fet_Opx_cat_6ox']
    - two_pyx['Lindley_Fe3_Opx'])+ two_pyx['Ti_Opx_cat_6ox'] + two_pyx['Al_VI_Opx_cat_6ox']
    + two_pyx['Cr_Opx_cat_6ox'] + two_pyx['Lindley_Fe3_Opx'])))



    a = np.empty(len(two_pyx['Kd_Fe_Mg_Cpx_Opx']), dtype=str)
    b = np.empty(len(two_pyx['Kd_Fe_Mg_Cpx_Opx']), dtype=str)
    for i in range(0, len(two_pyx['Kd_Fe_Mg_Cpx_Opx'])):
        if abs(0.7 - two_pyx['Kd_Fe_Mg_Cpx_Opx'].iloc[i] < 0.2):
            b[i] = str("Yes")
        if abs(0.7 - two_pyx['Kd_Fe_Mg_Cpx_Opx'].iloc[i] > 0.2):
            b[i] = str("No")
        if abs(1.09 - two_pyx['Kd_Fe_Mg_Cpx_Opx'].iloc[i]) < 0.14:
            a[i] = str("Yes")
        if abs(1.09 - two_pyx['Kd_Fe_Mg_Cpx_Opx'].iloc[i]) > 0.14:
            a[i] = str("No")

    two_pyx.insert(0, "High T Kd Eq?", a)
    two_pyx.insert(1, "Low T Kd Eq?", b)
    return two_pyx


def calculate_plag_liq_eq_tests(*, plag_comps=None, liq_comps=None, meltmatch=None, XAb=None, XAn=None, XOr=0,
P, T):

    '''
    Import Plag and Liq compositions, assesses An, Ab and Or equilibrium using the
    equations of Putirka (2005)
    Parameters
    -------

    liq_comps: pandas.DataFrame
            Cpx compositions with column headings SiO2_Liq, MgO_Liq etc.

    One of:

    1) plag_comps: pandas.DataFrame (optional)
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc.


    2) XAn, XAb, XOr, float, int, pandas.Series
        If plag_comps is None, enter XAn and XAb for plagioclases instead.
        XOr is set to zero by default, but can be overwritten for equilibrium tests

    3) meltmatch (pandas dataframe), dont need Liq compositions.
    Used for plag-liq melt matching code.

    P: int, float, pandas.Series
        Pressure in kbar

    T: int, float, pandas.Series
        Temperature in Kelvin

    Returns
    -------
    pandas DataFrame
        Return all plag components, liq compnoents, and  and cpx components,
        as well as values for Observed Kd (Ab-An), which Putirka (2008)
        suggests is the best equilibrium test. Also calculate Delta_An, which
        is the absolute value of measured An - predicted An from Putirka (2008).
    '''
    if liq_comps is not None:
        cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps)
    if plag_comps is not None and liq_comps is not None:
        cat_plags = calculate_cat_fractions_plagioclase(plag_comps=plag_comps)
        combo_plag_liq = pd.concat([cat_plags, cat_liqs], axis=1)
    elif meltmatch is not None:
        combo_plag_liq=meltmatch

    else:
        combo_plag_liq = cat_liqs
        combo_plag_liq['An_Plag']=XAn
        combo_plag_liq['Ab_Plag']=XAb
        combo_plag_liq['Or_Plag']=XOr

    # if type(P)==int:
    #     P=float(P)
    # if type(T)==int:
    #     T=float(T)
    combo_plag_liq['P'] = P
    combo_plag_liq['T'] = T
    Pred_An_EqE = (np.exp(-3.485 + 22.93 * combo_plag_liq['Ca_Liq_cat_frac'].astype(float)
     + 0.0805 * combo_plag_liq['H2O_Liq'].astype(float)
    + 1.0925 * combo_plag_liq['Ca_Liq_cat_frac'].astype(float)
     / (combo_plag_liq['Ca_Liq_cat_frac'].astype(float) + combo_plag_liq['Na_Liq_cat_frac'].astype(float))
     +13.11 * combo_plag_liq['Al_Liq_cat_frac'].astype(float) / (
    combo_plag_liq['Al_Liq_cat_frac'].astype(float) + combo_plag_liq['Si_Liq_cat_frac'].astype(float))
    + 5.59258 *combo_plag_liq['Si_Liq_cat_frac'].astype(float)**3 -
    38.786 * combo_plag_liq['P'].astype(float)  / (combo_plag_liq['T'].astype(float))- 125.04 *combo_plag_liq['Ca_Liq_cat_frac'].astype(float)
    *combo_plag_liq['Al_Liq_cat_frac'].astype(float)
    + 8.958 * combo_plag_liq['Si_Liq_cat_frac'].astype(float) * combo_plag_liq['K_Liq_cat_frac'].astype(float)
     - 2589.27 / (combo_plag_liq['T'].astype(float))))

    Pred_Ab_EqF = (np.exp(-2.748 - 0.1553 * combo_plag_liq['H2O_Liq'].astype(float) + 1.017 * combo_plag_liq['Mg_Number_Liq_NoFe3'].astype(float) - 1.997 * combo_plag_liq['Si_Liq_cat_frac'].astype(float)**3
     + 54.556 * combo_plag_liq['P'].astype(float)  / combo_plag_liq['T'].astype(float)- 67.878 *combo_plag_liq['K_Liq_cat_frac'].astype(float) *
     combo_plag_liq['Al_Liq_cat_frac'].astype(float)
    - 99.03 * combo_plag_liq['Ca_Liq_cat_frac'].astype(float) * combo_plag_liq['Al_Liq_cat_frac'].astype(float) + 4175.307 / combo_plag_liq['T'].astype(float)))

    Pred_Or_EqG = (np.exp(19.42 - 12.5 * combo_plag_liq['Mg_Liq_cat_frac'].astype(float)
     - 161.4 * combo_plag_liq['Na_Liq_cat_frac'].astype(float)
     - 16.65 * combo_plag_liq['Ca_Liq_cat_frac'].astype(float) / (
    combo_plag_liq['Ca_Liq_cat_frac'].astype(float) + combo_plag_liq['Na_Liq_cat_frac'].astype(float))
     - 528.1 * combo_plag_liq['K_Liq_cat_frac'] * combo_plag_liq['Al_Liq_cat_frac'] -
    19.38 * combo_plag_liq['Si_Liq_cat_frac']**3
     + 168.2 *combo_plag_liq['Si_Liq_cat_frac'] *
     combo_plag_liq['Na_Liq_cat_frac']
     - 1951.2 * combo_plag_liq['Ca_Liq_cat_frac'] * combo_plag_liq['K_Liq_cat_frac'] - 10190 / combo_plag_liq['T'].astype(float)))

    Obs_Kd_Ab_An=(combo_plag_liq['Ab_Plag']*combo_plag_liq['Al_Liq_cat_frac']*combo_plag_liq['Ca_Liq_cat_frac']/
    (combo_plag_liq['An_Plag']*combo_plag_liq['Na_Liq_cat_frac']*combo_plag_liq['Si_Liq_cat_frac']))

    relevantT=np.empty(len(combo_plag_liq), dtype=object)
    if isinstance(T, pd.Series):
        T=T.values
    if isinstance(T, float) or isinstance(T, int):
        T=np.empty(len(combo_plag_liq))+T
    for i in range(0, len(combo_plag_liq)):
        if T[i] < (1050+273.15):
            if Obs_Kd_Ab_An.iloc[i]>=0.05 and Obs_Kd_Ab_An.iloc[i]<=0.15:
                relevantT[i] = 'Low T: Yes'
            else:
                relevantT[i] = 'Low T: No'
        elif T[i] >= (1050+273.15):
            if Obs_Kd_Ab_An.iloc[i]>=0.17 and Obs_Kd_Ab_An.iloc[i]<=0.39:
                relevantT[i] = 'High T: Yes'
            else:
                relevantT[i] = 'High T: No'

        else:
            relevantT[i]='Temp Nan, test cant be performed'


    Delta_An=abs(Pred_An_EqE-combo_plag_liq['An_Plag'])
    Delta_Ab=abs(Pred_Ab_EqF-combo_plag_liq['Ab_Plag'])
    Delta_Or=abs(Pred_Or_EqG-combo_plag_liq['Or_Plag'])

    combo_plag_liq.insert(0, "Pass An-Ab Eq Test Put2008?", relevantT)
    combo_plag_liq.insert(1, 'Delta_An', Delta_An)
    combo_plag_liq.insert(2, 'Delta_Ab', Delta_Ab)
    combo_plag_liq.insert(3, 'Delta_Or', Delta_Or)
    combo_plag_liq.insert(4, 'Pred_An_EqE', Pred_An_EqE)
    combo_plag_liq.insert(5, 'Pred_Ab_EqF', Pred_Ab_EqF)
    combo_plag_liq.insert(6, 'Pred_Or_EqG', Pred_Or_EqG)
    combo_plag_liq.insert(7, 'Obs_Kd_Ab_An', Obs_Kd_Ab_An)

    return combo_plag_liq

## Feldspar compnents of Elkins and Grove, adapted directly from Putirka (2008) supplementay spreadsheet

def calculate_fspar_activity_components(*, Ab_Plag, Or_Plag, An_Plag, Ab_Kspar, Or_Kspar, An_Kspar, T, P):

    E_G_1990_Kspar_aAb1=(22820-T*6.3+0.461*1000*P)*(2*Ab_Kspar*Or_Kspar*(1-Ab_Kspar)+Or_Kspar*An_Kspar*(0.5-Ab_Kspar))
    E_G_1990_Kspar_aAb2=(19550-T*10.5+0.327*1000*P)*(Or_Kspar**2*(1-2*Ab_Kspar)+Or_Kspar*An_Kspar*(0.5-Ab_Kspar))
    E_G_1990_Kspar_aAb3=(31000-T*4.5+0.069*1000*P)*(An_Kspar**2*(1-2*Ab_Kspar)+Or_Kspar*An_Kspar*(0.5-Ab_Kspar))
    E_G_1990_Kspar_aAb4=(9800-T*-1.7-0.049*1000*P)*(2*An_Kspar*Ab_Kspar*(1-Ab_Kspar)+Or_Kspar*An_Kspar*(0.5-Ab_Kspar))
    E_G_1990_Kspar_aAb5=(90600-T*29.5-0.257*1000*P)*(Or_Kspar*An_Kspar*(0.5-Ab_Kspar-2*An_Kspar))
    E_G_1990_Kspar_aAb6=(60300-T*11.2-0.21*1000*P)*(Or_Kspar*An_Kspar*(0.5-Ab_Kspar-2*Or_Kspar))
    E_G_1990_Kspar_aAb7=(8000-T*0-0.467*1000*P)*(Or_Kspar*An_Kspar*(1-2*Ab_Kspar))
    E_G_1990_Kspar_Absum17=E_G_1990_Kspar_aAb1+E_G_1990_Kspar_aAb2+E_G_1990_Kspar_aAb3+E_G_1990_Kspar_aAb4+E_G_1990_Kspar_aAb5+E_G_1990_Kspar_aAb6+E_G_1990_Kspar_aAb7
    a_Ab_kspar=Ab_Kspar*np.exp(E_G_1990_Kspar_Absum17/(8.3144*T))

    E_G_1990_Kspar_aOr1=(22820-6.3*T+0.461*1000*P)*(Ab_Kspar**2*(1-2*Or_Kspar)+Ab_Kspar*An_Kspar*(0.5-Or_Kspar))
    E_G_1990_Kspar_aOr2=(19550-10.5*T+0.327*P*1000)*(2*Ab_Kspar*Or_Kspar*(1-Or_Kspar)+Ab_Kspar*An_Kspar*(0.5-Or_Kspar))
    E_G_1990_Kspar_aOr3=(31000-4.5*T+0.069*1000*P)*(Ab_Kspar*An_Kspar*(0.5-Or_Kspar-2*An_Kspar))
    E_G_1990_Kspar_aOr4=(9800--1.7*T-0.049*P*1000)*(Ab_Kspar*An_Kspar*(0.5-Or_Kspar-2*Ab_Kspar))
    E_G_1990_Kspar_aOr5=(90600-29.5*T-0.257*1000*P)*(An_Kspar**2*(1-2*Or_Kspar)+Ab_Kspar*An_Kspar*(0.5-Or_Kspar))
    E_G_1990_Kspar_aOr6=(60300-11.2*T-0.21*1000*P)*(2*Or_Kspar*An_Kspar*(1-Or_Kspar)+Ab_Kspar*An_Kspar*(0.5-Or_Kspar))
    E_G_1990_Kspar_aOr7=(8000-0*T-0.467*1000*P)*(An_Kspar*Ab_Kspar*(1-2*Or_Kspar))
    E_G_1990_Kspar_Orsum17=E_G_1990_Kspar_aOr1+E_G_1990_Kspar_aOr2+E_G_1990_Kspar_aOr3+E_G_1990_Kspar_aOr4+E_G_1990_Kspar_aOr5+E_G_1990_Kspar_aOr6+E_G_1990_Kspar_aOr7
    a_Or_kspar=Or_Kspar*np.exp(E_G_1990_Kspar_Orsum17/(8.3144*T))


    E_G_1990_Kspar_aAn1=(22820-T*6.3+0.461*1000*P)*Ab_Kspar*Or_Kspar*(0.5-An_Kspar-2*Ab_Kspar)
    E_G_1990_Kspar_aAn2=(19550-10.5*T+0.327*1000*P)*(Ab_Kspar*Or_Kspar*(0.5-An_Kspar-2*Or_Kspar))
    E_G_1990_Kspar_aAn3=(31000-T*4.5+0.069*1000*P)*(2*Ab_Kspar*An_Kspar*(1-An_Kspar)+Ab_Kspar*Or_Kspar*(0.5-An_Kspar))
    E_G_1990_Kspar_aAn4=(9800-T*-1.7-0.049*1000*P)*(Ab_Kspar**2*(1-2*An_Kspar)+Ab_Kspar*Or_Kspar*(0.5-An_Kspar))
    E_G_1990_Kspar_aAn5=(90600-T*29.5-0.257*P*1000)*(2*Or_Kspar*An_Kspar*(1-An_Kspar)+Ab_Kspar*Or_Kspar*(0.5-An_Kspar))
    E_G_1990_Kspar_aAn6=(60300-T*11.2-0.21*1000*P)*(Or_Kspar**2*(1-2*An_Kspar)+Ab_Kspar*Or_Kspar*(0.5-An_Kspar))
    E_G_1990_Kspar_aAn7=(8000-T*0-0.467*1000*P)*(Or_Kspar*Ab_Kspar*(1-2*An_Kspar))
    E_G_1990_Kspar_Ansum17=(E_G_1990_Kspar_aAn1+E_G_1990_Kspar_aAn2+E_G_1990_Kspar_aAn3+E_G_1990_Kspar_aAn4+E_G_1990_Kspar_aAn5+E_G_1990_Kspar_aAn6+E_G_1990_Kspar_aAn7)
    a_An_kspar=An_Kspar*np.exp(E_G_1990_Kspar_Ansum17/(8.3144*T))



    E_G_1990_Plag_aAb1=(22820-T*6.3+0.461*1000*P)*(2*Ab_Plag*Or_Plag*(1-Ab_Plag)+Or_Plag*An_Plag*(0.5-Ab_Plag))
    E_G_1990_Plag_aAb2=(19550-T*10.5+0.327*1000*P)*(Or_Plag**2*(1-2*Ab_Plag)+Or_Plag*An_Plag*(0.5-Ab_Plag))
    E_G_1990_Plag_aAb3=(31000-T*4.5+0.069*1000*P)*(An_Plag**2*(1-2*Ab_Plag)+Or_Plag*An_Plag*(0.5-Ab_Plag))
    E_G_1990_Plag_aAb4=(9800-T*-1.7-0.049*1000*P)*(2*An_Plag*Ab_Plag*(1-Ab_Plag)+Or_Plag*An_Plag*(0.5-Ab_Plag))
    E_G_1990_Plag_aAb5=(90600-T*29.5-0.257*1000*P)*(Or_Plag*An_Plag*(0.5-Ab_Plag-2*An_Plag))
    E_G_1990_Plag_aAb6=(60300-T*11.2-0.21*1000*P)*(Or_Plag*An_Plag*(0.5-Ab_Plag-2*Or_Plag))
    E_G_1990_Plag_aAb7=(8000-T*0-0.467*1000*P)*(Or_Plag*An_Plag*(1-2*Ab_Plag))
    E_G_1990_Plag_Absum17=E_G_1990_Plag_aAb1+E_G_1990_Plag_aAb2+E_G_1990_Plag_aAb3+E_G_1990_Plag_aAb4+E_G_1990_Plag_aAb5+E_G_1990_Plag_aAb6+E_G_1990_Plag_aAb7
    a_Ab_plg=Ab_Plag*np.exp(E_G_1990_Plag_Absum17/(8.3144*T))

    E_G_1990_Plag_aOr1=(22820-6.3*T+0.461*1000*P)*(Ab_Plag**2*(1-2*Or_Plag)+Ab_Plag*An_Plag*(0.5-Or_Plag))
    E_G_1990_Plag_aOr2=(19550-10.5*T+0.327*P*1000)*(2*Ab_Plag*Or_Plag*(1-Or_Plag)+Ab_Plag*An_Plag*(0.5-Or_Plag))
    E_G_1990_Plag_aOr3=(31000-4.5*T+0.069*1000*P)*(Ab_Plag*An_Plag*(0.5-Or_Plag-2*An_Plag))
    E_G_1990_Plag_aOr4=(9800--1.7*T-0.049*P*1000)*(Ab_Plag*An_Plag*(0.5-Or_Plag-2*Ab_Plag))
    E_G_1990_Plag_aOr5=(90600-29.5*T-0.257*1000*P)*(An_Plag**2*(1-2*Or_Plag)+Ab_Plag*An_Plag*(0.5-Or_Plag))
    E_G_1990_Plag_aOr6=(60300-11.2*T-0.21*1000*P)*(2*Or_Plag*An_Plag*(1-Or_Plag)+Ab_Plag*An_Plag*(0.5-Or_Plag))
    E_G_1990_Plag_aOr7=(8000-0*T-0.467*1000*P)*(An_Plag*Ab_Plag*(1-2*Or_Plag))
    E_G_1990_Plag_Orsum17=E_G_1990_Plag_aOr1+E_G_1990_Plag_aOr2+E_G_1990_Plag_aOr3+E_G_1990_Plag_aOr4+E_G_1990_Plag_aOr5+E_G_1990_Plag_aOr6+E_G_1990_Plag_aOr7
    a_Or_plg=Or_Plag*np.exp(E_G_1990_Plag_Orsum17/(8.3144*T))


    E_G_1990_Plag_aAn1=(22820-T*6.3+0.461*1000*P)*Ab_Plag*Or_Plag*(0.5-An_Plag-2*Ab_Plag)
    E_G_1990_Plag_aAn2=(19550-10.5*T+0.327*1000*P)*(Ab_Plag*Or_Plag*(0.5-An_Plag-2*Or_Plag))
    E_G_1990_Plag_aAn3=(31000-T*4.5+0.069*1000*P)*(2*Ab_Plag*An_Plag*(1-An_Plag)+Ab_Plag*Or_Plag*(0.5-An_Plag))
    E_G_1990_Plag_aAn4=(9800-T*-1.7-0.049*1000*P)*(Ab_Plag**2*(1-2*An_Plag)+Ab_Plag*Or_Plag*(0.5-An_Plag))
    E_G_1990_Plag_aAn5=(90600-T*29.5-0.257*P*1000)*(2*Or_Plag*An_Plag*(1-An_Plag)+Ab_Plag*Or_Plag*(0.5-An_Plag))
    E_G_1990_Plag_aAn6=(60300-T*11.2-0.21*1000*P)*(Or_Plag**2*(1-2*An_Plag)+Ab_Plag*Or_Plag*(0.5-An_Plag))
    E_G_1990_Plag_aAn7=(8000-T*0-0.467*1000*P)*(Or_Plag*Ab_Plag*(1-2*An_Plag))
    E_G_1990_Plag_Ansum17=(E_G_1990_Plag_aAn1+E_G_1990_Plag_aAn2+E_G_1990_Plag_aAn3+E_G_1990_Plag_aAn4+E_G_1990_Plag_aAn5+E_G_1990_Plag_aAn6+E_G_1990_Plag_aAn7)
    a_An_plg=An_Plag*np.exp(E_G_1990_Plag_Ansum17/(8.3144*T))

    DeltaAn_Plag_Kspar= a_An_plg-a_An_kspar
    DeltaOr_Plag_Kspar= a_Or_plg-a_Or_kspar
    DeltaAb_Plag_Kspar= a_Ab_plg-a_Ab_kspar


    Components=pd.DataFrame(data={'Delta_An': DeltaAn_Plag_Kspar,  'Delta_Ab': DeltaAb_Plag_Kspar, 'Delta_Or': DeltaOr_Plag_Kspar,
                                  'a_Ab_plg': a_Ab_plg, 'a_An_plg':a_An_plg, 'a_Or_plg': a_Or_plg,
                                 'a_Ab_kspar': a_Ab_kspar, 'a_An_kspar':a_An_kspar, 'a_Or_kspar': a_Or_kspar})
    return Components


def calculate_plag_components(*, Ca_Liq_cat_frac, H2O_Liq, Na_Liq_cat_frac, Al_Liq_cat_frac,
    Si_Liq_cat_frac, K_Liq_cat_frac, T, P, An_Plag, Ab_Plag, Mg_Number_Liq_NoFe3, Mg_Liq_cat_frac):



    An_Pred=np.exp(-3.485+22.93*Ca_Liq_cat_frac+0.0805*H2O_Liq+1.0925*Ca_Liq_cat_frac/(Ca_Liq_cat_frac+Na_Liq_cat_frac)
    +13.11*Al_Liq_cat_frac/(Al_Liq_cat_frac+Si_Liq_cat_frac)+5.59258*Si_Liq_cat_frac**3-
    38.786*P/(T)-125.04*Ca_Liq_cat_frac*Al_Liq_cat_frac+8.958*Si_Liq_cat_frac*K_Liq_cat_frac-2589.27/(T))
    Ab_Pred=np.exp(-2.748-0.1553*H2O_Liq+1.017*Mg_Number_Liq_NoFe3-1.997*Si_Liq_cat_frac**3+54.556*P/T-67.878*K_Liq_cat_frac*Al_Liq_cat_frac
    -99.03*Ca_Liq_cat_frac*Al_Liq_cat_frac+4175.307/T)
    Or_Pred=np.exp(19.42-12.5*Mg_Liq_cat_frac--161.4*Na_Liq_cat_frac-16.65*Ca_Liq_cat_frac/(Ca_Liq_cat_frac+Na_Liq_cat_frac)
    -528.1*K_Liq_cat_frac*Al_Liq_cat_frac-19.38*Si_Liq_cat_frac**3+168.2*Si_Liq_cat_frac*Na_Liq_cat_frac
    -1951.2*Ca_Liq_cat_frac*K_Liq_cat_frac-10190/T)
    Obs_Kd_Ab_An=Ab_Plag*Al_Liq_cat_frac*Ca_Liq_cat_frac/(An_Plag*Na_Liq_cat_frac*Si_Liq_cat_frac)
    Components=pd.DataFrame(data={'An_Pred': An_Pred, 'Ab_Pred': Ab_Pred, 'Or_Pred': Or_Pred})

    return Components

def calculate_eq_plag_components(liq_comps, H2O_Liq, T,P):

    cfs=calculate_anhydrous_cat_fractions_liquid(liq_comps)



    cfs['An_Pred']=np.exp(-3.485+22.93*cfs['Ca_Liq_cat_frac']+0.0805*H2O_Liq
    +1.0925*cfs['Ca_Liq_cat_frac']/(cfs['Ca_Liq_cat_frac']+cfs['Na_Liq_cat_frac'])
    +13.11*cfs['Al_Liq_cat_frac']/(cfs['Al_Liq_cat_frac']+cfs['Si_Liq_cat_frac'])+5.59258*cfs['Si_Liq_cat_frac']**3-
    38.786*P/(T)-125.04*cfs['Ca_Liq_cat_frac']*cfs['Al_Liq_cat_frac']+8.958*cfs['Si_Liq_cat_frac']*cfs['K_Liq_cat_frac']-2589.27/(T))
    cfs['Ab_Pred']=np.exp(-2.748-0.1553*H2O_Liq+1.017*cfs['Mg_Number_Liq_NoFe3']-1.997*cfs['Si_Liq_cat_frac']**3+54.556*P/T-67.878*cfs['K_Liq_cat_frac']*cfs['Al_Liq_cat_frac']
    -99.03*cfs['Ca_Liq_cat_frac']*cfs['Al_Liq_cat_frac']+4175.307/T)
    cfs['Or_Pred']=np.exp(19.42-12.5*cfs['Mg_Liq_cat_frac']-161.4*cfs['Na_Liq_cat_frac']-16.65*cfs['Ca_Liq_cat_frac']/(cfs['Ca_Liq_cat_frac']+cfs['Na_Liq_cat_frac'])
    -528.1*cfs['K_Liq_cat_frac']*cfs['Al_Liq_cat_frac']-19.38*cfs['Si_Liq_cat_frac']**3
    +168.2*cfs['Si_Liq_cat_frac']*cfs['Na_Liq_cat_frac']
    -1951.2*cfs['Ca_Liq_cat_frac']*cfs['K_Liq_cat_frac']-10190/T)

    return cfs


## Tool to get Fe3Fet from logfo2 or buffer value.
def convert_fo2_to_fe_partition(*, liq_comps, T_K, P_kbar,  model="Kress1991", fo2, renorm=False, fo2_offset=0):
    '''
    Calculates Fe3Fet_Liq, FeO and Fe2O3 based on user-specified buffer or fo2 value (in bars)

   Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    T_K:  int, flt, pandas.Series
        Temperature in Kelvin (buffer positions are very T-sensitive)

    P_kbar: int, flt, pandas.Series
        Pressure in Kbar (Buffer positions are slightly sensitive to pressure)

    fo2:  str ("QFM", "NNO") or int, flt, pandas.Series
        Either a value of fo2 (enter 10*logfo2), or buffer position as a string
        So far, includes QFM or NNO
        fo2 in bars.

    fo2_offset: int, flt, pandas.Series
        log units offset from buffer, e.g., could specify fo2=QFM, fo2_offset=1
        to perform calculations at QFM+1

    model: str
        "Kress1991" - Uses Kress and Carmichael 1991 to calculate XFe2Fe3 from fo2
        "Put2016_eq6b" - Uses Putirka (2016) expression to calculate XFe2Fe3 from fo2

    renorm: bool
        Following excel code of K. Iacovino.
        If True, renormalizes other oxide concentrations
        to account for change in total following partitioning of Fe into FeO and Fe2O3.

    Returns
    -------

    liquid compositions with calculated Fe3Fet_Liq, FeO_Liq, Fe2O3_Liq, and XFe3Fe2.

    '''
    if any(liq_comps.columns=="Sample_ID_Liq"):
        liq_comps_c=liq_comps.copy()
    else:

        liq_comps_c=liq_comps.copy()
        liq_comps_c['Sample_ID_Liq']=liq_comps_c.index
    if isinstance(fo2, str):
        fo2_int=fo2
        if fo2_int=="NNO":
        # Buffer position from frost (1991)
            logfo2=(-24930/T_K) + 9.36 + 0.046 * ((P_kbar*1000)-1)/T_K+fo2_offset


        if fo2_int=="QFM":

        # Buffer position from frost (1991)
            logfo2_QFM_highT=(-25096.3/T_K) + 8.735 + 0.11 * ((P_kbar*1000)-1)/T_K
            T_Choice='HighT Beta Qtz'

            logfo2_QFM_lowT=(-26455.3/T_K) +10.344 + 0.092 * ((P_kbar*1000)-1)/T_K
            T_Choice='Low T alpha Qtz'

            Cut_off_T=573+273.15+0.025*(P_kbar*1000)

            if isinstance(logfo2_QFM_lowT, float) or isinstance(logfo2_QFM_lowT, int):
                if T_K<Cut_off_T:
                    logfo2_QFM=logfo2_QFM_lowT
                if T_K>=Cut_off_T:
                    logfo2_QFM=logfo2_QFM_highT

            else:
                logfo2_QFM=pd.Series(logfo2_QFM_highT)

                T_K=pd.Series(T_K).fillna(0)

                lowT = pd.Series(T_K)<Cut_off_T
                # nanmask=np.isnan(T_K)
                # final_mask=np.logical_or(lowT, nanmask)
                # print(np.shape(lowT))
                # print(sum(lowT))
                # print(lowT)


                if sum(lowT)>0:

                    logfo2_QFM.loc[lowT]=logfo2_QFM_lowT




            logfo2=logfo2_QFM


        fo2=10**logfo2




    mol_frac_hyd_short=calculate_hydrous_mol_fractions_liquid(liq_comps_c)
    mol_frac_hyd=pd.concat([mol_frac_hyd_short, liq_comps_c], axis=1)
    To=1673.15

    if model=="Kress1991":
        ln_XFe2FeO3_XFeO=((0.196*np.log(fo2))+(11492/T_K)-6.675+((-2.243*mol_frac_hyd['Al2O3_Liq_mol_frac_hyd'])+(-1.828*mol_frac_hyd['FeOt_Liq_mol_frac_hyd'])
        +(3.201*mol_frac_hyd['CaO_Liq_mol_frac_hyd'])+(5.854*mol_frac_hyd['Na2O_Liq_mol_frac_hyd'])+(6.215*mol_frac_hyd['K2O_Liq_mol_frac_hyd']))
        -3.36*(1-(To/T_K) - np.log(T_K/To)) -0.000000701*((P_kbar*100000000)/T_K)
         + -0.000000000154*(((T_K-1673)*(P_kbar*100000000))/T_K) + 0.0000000000000000385*((P_kbar*100000000)**2/T_K))
        #print(ln_XFe2FeO3_XFeO)
        #print(fo2)

    if model=="Put2016_eq6b":
        ln_XFe2FeO3_XFeO=(-6.53+10813.8/T_K + 0.19*np.log(fo2)+ 12.4*(mol_frac_hyd['Na2O_Liq_mol_frac_hyd']
         +mol_frac_hyd['K2O_Liq_mol_frac_hyd'])
        -3.44*(mol_frac_hyd['Al2O3_Liq_mol_frac_hyd']/(mol_frac_hyd['Al2O3_Liq_mol_frac_hyd']+mol_frac_hyd['SiO2_Liq_mol_frac_hyd']))
        +4.15*mol_frac_hyd['CaO_Liq_mol_frac_hyd'])

    X_Fe2O3_X_FeO=np.exp(ln_XFe2FeO3_XFeO)
    X_Fe2O3=X_Fe2O3_X_FeO*mol_frac_hyd['FeOt_Liq_mol_frac_hyd']/(2*X_Fe2O3_X_FeO+1)

    #X_FeO=mol_frac_hyd['FeOt_Liq_mol_frac_hyd']/(2*X_Fe2O3_X_FeO+1) Kayla's way
    X_FeO=mol_frac_hyd['FeOt_Liq_mol_frac_hyd']-2*X_Fe2O3
    Sum_all_mol_frac_hyd=(mol_frac_hyd['SiO2_Liq_mol_frac_hyd']+mol_frac_hyd['TiO2_Liq_mol_frac_hyd']+mol_frac_hyd['Al2O3_Liq_mol_frac_hyd']+mol_frac_hyd['MnO_Liq_mol_frac_hyd']
                      +mol_frac_hyd['MgO_Liq_mol_frac_hyd']+mol_frac_hyd['CaO_Liq_mol_frac_hyd']+mol_frac_hyd['Na2O_Liq_mol_frac_hyd']+mol_frac_hyd['K2O_Liq_mol_frac_hyd']
                      +mol_frac_hyd['P2O5_Liq_mol_frac_hyd']+X_FeO+X_Fe2O3)

    Fe2O3_unnorm=X_Fe2O3*159.6
    FeO_unnorm=X_FeO*71.844
    Sum_All_mol=(mol_frac_hyd['SiO2_Liq_mol_frac_hyd']*60.0843+mol_frac_hyd['TiO2_Liq_mol_frac_hyd']*79.8788
    +mol_frac_hyd['Al2O3_Liq_mol_frac_hyd']*101.961+mol_frac_hyd['MnO_Liq_mol_frac_hyd']*70.9375
    +mol_frac_hyd['MgO_Liq_mol_frac_hyd']*40.3044+mol_frac_hyd['CaO_Liq_mol_frac_hyd']*56.0774+mol_frac_hyd['Na2O_Liq_mol_frac_hyd']*61.9789+mol_frac_hyd['K2O_Liq_mol_frac_hyd']*94.196
    +mol_frac_hyd['P2O5_Liq_mol_frac_hyd']*141.937+X_Fe2O3*159.6+X_FeO*71.844)
    New_Fe2O3_wt=(100*X_Fe2O3*159.6)/Sum_All_mol
    New_FeO_wt=(100*X_FeO*71.844)/Sum_All_mol

    New_Oxide_out_nonorm=liq_comps.copy()
    New_Oxide_out_nonorm['FeO_Liq']=New_FeO_wt
    New_Oxide_out_nonorm['Fe2O3_Liq']=New_Fe2O3_wt
    New_Oxide_out_nonorm['XFe3Fe2']=X_Fe2O3_X_FeO
    New_Oxide_out_nonorm['Fe3Fet_Liq']=New_Fe2O3_wt*0.8998/(New_FeO_wt+New_Fe2O3_wt*0.8998)


    New_Oxide_out_norm=pd.DataFrame(data={'SiO2_Liq': 100*mol_frac_hyd['SiO2_Liq_mol_frac_hyd']*60.084/Sum_All_mol,
                                         'TiO2_Liq': 100*mol_frac_hyd['TiO2_Liq_mol_frac_hyd']*79.8788/Sum_All_mol,
                                         'Al2O3_Liq':100*mol_frac_hyd['Al2O3_Liq_mol_frac_hyd']*101.961/Sum_All_mol,
                                          'Fe2O3_Liq': (100*X_Fe2O3*159.6)/Sum_All_mol,
                                          'FeO_Liq': (100*X_FeO*71.844)/Sum_All_mol,
                                          'MnO_Liq': 100*mol_frac_hyd['MnO_Liq_mol_frac_hyd']*70.9375/Sum_All_mol,
                                          'MgO_Liq': 100*mol_frac_hyd['MgO_Liq_mol_frac_hyd']*40.3044/Sum_All_mol,
                                         'CaO_Liq': 100*mol_frac_hyd['CaO_Liq_mol_frac_hyd']*56.0774/Sum_All_mol,
                                          'Na2O_Liq': 100*mol_frac_hyd['Na2O_Liq_mol_frac_hyd']*61.9789/Sum_All_mol,
                                          'K2O_Liq': 100*mol_frac_hyd['K2O_Liq_mol_frac_hyd']*94.196/Sum_All_mol,
                                         'P2O5_Liq':  100*mol_frac_hyd['P2O5_Liq_mol_frac_hyd']*141.937/Sum_All_mol,
                                         })
    Old_Sum=(100/liq_comps_c.drop(['Sample_ID_Liq'], axis=1).sum(axis=1))
    New_Oxide_out_New_old_total=New_Oxide_out_norm.div(Old_Sum, axis=0)
    New_Oxide_out_New_old_total['Fe3Fet_Liq']=(New_Oxide_out_norm['Fe2O3_Liq']*0.8998/(New_Oxide_out_norm['FeO_Liq']+New_Oxide_out_norm['Fe2O3_Liq']*0.8998)).fillna(0)



    if renorm==False:
        New_Oxide_out_nonorm['ln_XFe2FeO3_XFeO']=ln_XFe2FeO3_XFeO
        return New_Oxide_out_nonorm
    else:
        New_Oxide_out_New_old_total['ln_XFe2FeO3_XFeO']=ln_XFe2FeO3_XFeO
        return New_Oxide_out_New_old_total

## Need some functions for calculating mole proportions with Fe partition
oxide_mass_liq_hyd_redox = {'SiO2_Liq': 60.0843, 'MgO_Liq': 40.3044,
'MnO_Liq': 70.9375, 'FeO_Liq': 71.844, 'Fe2O3_Liq': 159.69, 'CaO_Liq': 56.0774,
'Al2O3_Liq': 101.961,'Na2O_Liq': 61.9789, 'K2O_Liq': 94.196,
 'TiO2_Liq': 79.8788, 'P2O5_Liq': 141.937, 'Cr2O3_Liq': 151.9982,
  'H2O_Liq': 18.01528}
# Turns dictionary into a dataframe so pandas matrix math functions can be used
oxide_mass_liq_hyd_df_redox = pd.DataFrame.from_dict(
    oxide_mass_liq_hyd_redox, orient='index').T
oxide_mass_liq_hyd_df_redox['Sample_ID_Liq'] = 'MolWt'
oxide_mass_liq_hyd_df_redox.set_index('Sample_ID_Liq', inplace=True)

def calculate_hydrous_mol_proportions_liquid_redox(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------


    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the ..Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = liq_comps.reindex(oxide_mass_liq_hyd_df_redox.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_hyd_df_redox, liq_wt],)
    # Drop the calculation column
    mol_prop_hyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_hyd.columns = [
        str(col) + '_mol_prop_hyd' for col in mol_prop_hyd.columns]
    return mol_prop_hyd

def calculate_hydrous_mol_fractions_liquid_redox(liq_comps):
    '''Import Liq compositions using liq_comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    liq_comps: pandas.DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.



    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_hydrous_mol_proportions_liquid_redox(liq_comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_hyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_hyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_hyd.columns = [str(col).replace('prop', 'frac')
                            for col in mol_frac_hyd.columns]
    return mol_frac_hyd


def convert_fo2_to_buffer(fo2=None, T_K=None, P_kbar=None):

    """ Converts fo2 in bars to deltaNNO and delta QFM using Frost 1991, ONeill et al. (2018)
    and the equation in Petrolog3.
    based on user-entered T in Kelvin and P in kbar
    """
    logfo2=np.log10(fo2)

    # NNO Buffer position from frost (1991)
    logfo2_NNO_Frost=(-24930/T_K) + 9.36 + 0.046 * ((P_kbar*1000)-1)/T_K

    DeltaNNO_Frost=logfo2-logfo2_NNO_Frost



#  QFM Buffer position from frost (1991)

    # Calculates cut off T for alpha-beta qtz transition that determins QFM
    Cut_off_T=573+273.15+0.025*(P_kbar*1000)

    logfo2_QFM_highT=(-25096.3/T_K) + 8.735 + 0.11 * ((P_kbar*1000)-1)/T_K
    T_Choice='HighT Beta Qtz'

    logfo2_QFM_lowT=(-26455.3/T_K) +10.344 + 0.092 * ((P_kbar*1000)-1)/T_K
    T_Choice='Low T alpha Qtz'


    fo2_QFM_highT=10**logfo2_QFM_highT
    fo2_QFM_lowT=10**logfo2_QFM_lowT

    Delta_QFM_highT=logfo2-logfo2_QFM_highT
    Delta_QFM_lowT=logfo2-logfo2_QFM_highT

    if isinstance(fo2, float) or isinstance(fo2, int):
        if T_K<Cut_off_T:
            DeltaQFM=Delta_QFM_lowT
        if T_K>=Cut_off_T:
            DeltaQFM=Delta_QFM_highT
        out=pd.DataFrame(data={'DeltaNNO_Frost1991':  DeltaNNO_Frost,
                               'DeltaQFM_Frost1991':  DeltaQFM,
                               'QFM_equation_Choice': 'High T',
                               'T_K': T_K,
                               'P_kbar': P_kbar,
                               'fo2': fo2}, index=[0])
    else:

        out=pd.DataFrame(data={'DeltaNNO_Frost1991':  DeltaNNO_Frost,
                               'DeltaQFM_Frost1991': Delta_QFM_highT,
                               'QFM_equation_Choice': 'High T',
                               'T_K': T_K,
                               'P_kbar': P_kbar,
                               'fo2': fo2})
        out.loc[(T_K<Cut_off_T), 'DeltaQFM_Frost1991']=Delta_QFM_lowT
        out.loc[(T_K<Cut_off_T), 'QFM_equation_Choice']='Low T'

    # Other buffer positions
    logfo2_QFM_Oneill=8.58-25050/T_K
    Delta_QFM_Oneill=logfo2-logfo2_QFM_Oneill
    out['DeltaQFM_ONeill1987']=Delta_QFM_Oneill


    out['Cut off T (K)']=Cut_off_T

    return out




def convert_fe_partition_to_fo2(*, liq_comps,  T_K, P_kbar,  model="Kress1991", Fe3Fet_Liq=None,
 renorm=False):
    '''
    Calculates delta fo2 relative to QFM and NNO buffer for liq compositions with FeO and Fe2O3

   Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq, FeO_Liq and Fe2O3_Liq etc.
        Or, have FeOt_Liq and Fe3Fet_Liq terms. In which case, you can overwrite with

    T_K:  int, flt, pandas.Series
        Temperature in Kelvin (buffer positions are very T-sensitive)

    P_kbar: int, flt, pandas.Series
        Pressure in Kbar (Buffer positions are slightly sensitive to pressure)




    model: str
        "Kress1991" - Uses Kress and Carmichael 1991 to calculate XFe2Fe3 from fo2
        "Put2016_eq6b" - Uses Putirka (2016) expression to calculate XFe2Fe3 from fo2

    renorm: bool
        Following excel code of K. Iacovino.
        If True, renormalizes other oxide concentrations
        to account for change in total following partitioning of Fe into FeO and Fe2O3.

    Returns
    -------

    liquid compositions with calculated Fe3FeT_Liq, FeO_Liq, Fe2O3_Liq, and XFe3Fe2.

    '''
    liq_comps_c=liq_comps.copy()
    # If the input has FeO and Fe2O3 contents already
    if any(liq_comps.columns=="FeO_Liq") and any(liq_comps.columns=="Fe2O3_Liq"):
        print('using inputted FeO and Fe2O3 contents')
        if Fe3Fet_Liq is not None:
            print('sorry, you entered FeO and Fe2O3, so you cant overwrite Fe3Fet_Liq')
        #liq_comps_c['FeOt_Liq']=liq_comps_c['FeO_Liq']+c['Fe2O3_Liq']*0.8998

    # If the person specifies Fe3Fet in the function itself, overwrite the input
    else:
        if Fe3Fet_Liq is not None:
            liq_comps_c['Fe3Fet_Liq']=Fe3Fet_Liq
            print('overwriting Fe3Fet_Liq to that specified in the function input')

    # If any of the columns contain Fe3Fet_Liq
    if any(liq_comps_c.columns=="Fe3Fet_Liq") and any(liq_comps_c.columns=="FeOt_Liq"):
        liq_comps_c['FeO_Liq']=liq_comps['FeOt_Liq']*(1-liq_comps_c['Fe3Fet_Liq'])
        liq_comps_c['Fe2O3_Liq']=liq_comps['FeOt_Liq']*(liq_comps_c['Fe3Fet_Liq'])*1.11111


    mol_frac_hyd_redox=calculate_hydrous_mol_fractions_liquid_redox(liq_comps=liq_comps_c)
    if any(liq_comps_c.columns=="FeOt_Liq"):
        liq_comps_FeOt=liq_comps_c.copy()
    else:
        liq_comps_c['FeOt_Liq']=liq_comps_c['FeO_Liq']+liq_comps_c['Fe2O3_Liq']*0.8998
        liq_comps_FeOt=liq_comps_c.copy()

    hyd_mol_frac_test=calculate_hydrous_mol_fractions_liquid(liq_comps=liq_comps_FeOt)

    # Calculating buffer positions from Frost 1991

    To= 1673.15

    logfo2_NNO=(-24930/T_K) + 9.36 + 0.046 * ((P_kbar*1000)-1)/T_K
    fo2_NNO=10**logfo2_NNO

#  QFM Buffer position from frost (1991)

    # Calculates cut off T for alpha-beta qtz transition that determins QFM
    Cut_off_T=573+273.15+0.025*(P_kbar*1000)

    logfo2_QFM_highT=(-25096.3/T_K) + 8.735 + 0.11 * ((P_kbar*1000)-1)/T_K
    T_Choice='HighT Beta Qtz'

    logfo2_QFM_lowT=(-26455.3/T_K) +10.344 + 0.092 * ((P_kbar*1000)-1)/T_K
    T_Choice='Low T alpha Qtz'


    fo2_QFM_highT=10**logfo2_QFM_highT
    fo2_QFM_lowT=10**logfo2_QFM_lowT



    if isinstance(logfo2_QFM_lowT, float) or isinstance(logfo2_QFM_lowT, int):
        if T_K<Cut_off_T:
            logfo2_QFM=logfo2_QFM_lowT
        if T_K>=Cut_off_T:
            logfo2_QFM=logfo2_QFM_highT

    else:
        logfo2_QFM=pd.Series(logfo2_QFM_highT)

        T_K=pd.Series(T_K).fillna(0)

        lowT = pd.Series(T_K)<Cut_off_T
        # nanmask=np.isnan(T_K)
        # final_mask=np.logical_or(lowT, nanmask)
        # print(np.shape(lowT))
        # print(sum(lowT))
        # print(lowT)

        print(np.shape(logfo2_QFM))
        if sum(lowT)>0:

            logfo2_QFM.loc[lowT]=logfo2_QFM_lowT





    fo2_QFM=10**logfo2_QFM




    # This is Ln (XFe2O3/XFeO) from the Kress and Carmichael 1991 paper
    Z=np.log(mol_frac_hyd_redox['Fe2O3_Liq_mol_frac_hyd']/
         (mol_frac_hyd_redox['FeO_Liq_mol_frac_hyd']))

    # We've simplified the equatoin down to Z= a ln fo2 + rightside

    rightside=( (11492/T_K)-6.675+((-2.243*mol_frac_hyd_redox['Al2O3_Liq_mol_frac_hyd'])+(-1.828*hyd_mol_frac_test['FeOt_Liq_mol_frac_hyd'])
    +(3.201*mol_frac_hyd_redox['CaO_Liq_mol_frac_hyd'])+(5.854*mol_frac_hyd_redox['Na2O_Liq_mol_frac_hyd'])+(6.215*mol_frac_hyd_redox['K2O_Liq_mol_frac_hyd']))
    -3.36*(1-(To/T_K) - np.log(T_K.astype(float)/To)) -0.000000701*((P_kbar*100000000)/T_K)
    + -0.000000000154*(((T_K-1673)*(P_kbar*100000000))/T_K) + 0.0000000000000000385*((P_kbar*100000000)**2/T_K)
    )

    ln_fo2_calc=(Z-rightside)/0.196
    fo2_calc=np.exp(ln_fo2_calc.astype(float))

    # and back to log base 10
    log_fo2_calc=np.log10(fo2_calc)
    DeltaQFM=log_fo2_calc-logfo2_QFM
    DeltaNNO=log_fo2_calc-logfo2_NNO


    liq_comps_c.insert(0, 'DeltaQFM_Frost1991', DeltaQFM)
    liq_comps_c.insert(1, 'DeltaNNO_Frost1991', DeltaNNO)
    liq_comps_c.insert(2, 'fo2_calc', fo2_calc)
    return liq_comps_c

## Machine Learning Voting using the old format where things were a pickle
def get_voting_ExtraTreesRegressor(X, reg):
    voting = []
    for tree in reg.estimators_:
        voting.append(tree.predict(X).tolist())
    voting = np.asarray(voting)
    return voting

def get_voting_stats_ExtraTreesRegressor(X, reg, central_tendency='aritmetic_mean', dispersion='dev_std'):

    voting = get_voting_ExtraTreesRegressor(X, reg)

    voting_central_tendency_mean = voting.mean(axis=0)
    voting_central_tendency_median = np.median(voting, axis=0)
    voting_dispersion_std = voting.std(axis=0)
    voting_dispersion_IQR = np.percentile(voting, 75, axis=0) - np.percentile(voting, 25, axis=0)
    df_stats=pd.DataFrame(data={
                          'Median_Trees': voting_central_tendency_median,
                          'Std_Trees': voting_dispersion_std,
                          'IQR_Trees': voting_dispersion_IQR})
    df_voting=pd.DataFrame(voting).T.add_prefix('Tree_')

    return  df_stats, df_voting

## Machine learning voting using new pickle format




def classify_phases(filename=None, sheet_name=None, df=None, return_end_members=False, str_to_drop=None):
    """
    Function in progress


    """
    if filename is not None:
        Excel_In=import_excel(filename, sheet_name)
        if str_to_drop is None:
            my_input_copy=Excel_In['my_oxides']
        if str_to_drop is not None:
            my_input=Excel_In['my_input']
            my_input_copy=my_input.copy()
            my_input_copy.columns = [col.replace(str_to_drop, '') for col in my_input_copy.columns]
    if df is not None:
        my_input=df
        my_input_copy=my_input.copy()

        if str_to_drop is not None:
            my_input_copy.columns = [col.replace(str_to_drop, '') for col in my_input_copy.columns]


    myOxides1 = my_input_copy.reindex(df_ideal_oxide.columns, axis=1).fillna(0)
    myOxides1 = myOxides1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myOxides1[myOxides1 < 0] = 0
    Oxides=myOxides1

    Oxides_prefix=Oxides.drop(columns=['Cr2O3', 'P2O5'])
    Oxides_prefix=Oxides_prefix.add_suffix('_input')



    Oxides_amp_sites=get_amp_sites_from_input_not_amp(Oxides_prefix, "_input")
    Oxides_cpx_sites=calculate_cpx_sites_from_input_not_cpx(Oxides_prefix, "_input")
    Oxides_prefix['Ca_B']=Oxides_amp_sites['Ca_B']
    Oxides_prefix['Na_K_A']=Oxides_amp_sites['Na_A']+Oxides_amp_sites['K_A']
    Oxides_prefix['Sum_Amp_Cat_Sites']=Oxides_amp_sites['cation_sum_All']
    Oxides_prefix['Cation_Sum_Cpx']=Oxides_cpx_sites['Cation_Sum_Cpx']
    Oxides_prefix['Ca_CaMgFe']=Oxides_cpx_sites['Ca_CaMgFe']
    Oxides_prefix.replace([np.nan, -np.nan], 0, inplace=True)



    with open(Thermobar_dir/'svc_model_linear_MinClass.pkl', 'rb') as f:
        svc_model=load(f)
    with open(Thermobar_dir/'scaler_MinClass.pkl', 'rb') as f:
        scaler=load(f)

    # Dropping things which are often missing

    X_in=Oxides_prefix.values


    # This does the machine learning classification
    X_in_scaled= scaler.transform(X_in)
    svc_predictions=svc_model.predict(X_in_scaled)
    Oxides_out=Oxides_prefix.copy()
    Oxides_out['Sum_Oxides']=Oxides.sum(axis=1)
    Oxides_out['Phase_Min_Group_ML']=svc_predictions
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(0, "Amp")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(1, "Px")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(10, "Px")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(11, "Px")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(2, "Fspar")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(3, "Ol")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(4, "Sp")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(5, "Ox")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(6, "Ap")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(7, "Bt")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(8, "Qz")
    Oxides_out[['Phase_Min_Group_ML']]=Oxides_out[['Phase_Min_Group_ML']].replace(9, "Gt")

    #print(' # of Amps = ' + str(len(Oxides_out['Phase_Min_Group_ML']=="Amp")) + '# of Opxs = ')

    Oxides_out.loc[Oxides_out['Sum_Oxides']<60, 'Phase_Min_Group_ML'] = "Not Classified - Total<60"
    Oxides_out.loc[Oxides_out['Sum_Oxides']>110, 'Phase_Min_Group_ML'] = "Not Classified - Total>110"
    Oxides_out['Phase_Mineral']=Oxides_out['Phase_Min_Group_ML']
    # Classification for amphibole names based on Ridolfi
    Oxides_Amp1=Oxides_out.copy()
    Oxides_Amp1.columns = [col.replace("_input", "_Amp") for col in Oxides_Amp1.columns]
    Oxides_Amp=Oxides_Amp1.reindex(df_ideal_amp.columns, axis=1)
    amp_names = calculate_sites_ridolfi(amp_comps=Oxides_Amp).classification
    Oxides_out.loc[Oxides_out['Phase_Min_Group_ML']=="Amp", 'Phase_Mineral']=amp_names

    #This does manual classification for feldsdpars.
    Fspar=Oxides_out['Phase_Min_Group_ML']=="Fspar" # This checks we are only doing it for felspar
    Oxides_Fspar=Oxides_out.copy()
    Oxides_Fspar.columns = [col.replace("_input", "_Plag") for col in Oxides_Fspar.columns]


    Fspar_components=calculate_cat_fractions_plagioclase(plag_comps=Oxides_Fspar)
    Fspar_An=Fspar_components['An_Plag']
    Fspar_Ab=Fspar_components['Ab_Plag']
    Fspar_Or=Fspar_components['Or_Plag']
    if return_end_members==True:
        Oxides_out['Ab']=Fspar_Ab

        Oxides_out.loc[Oxides_out['Phase_Min_Group_ML']!="Fspar", 'Ab']="N/A"
        Oxides_out['An']=Fspar_An
        Oxides_out.loc[Oxides_out['Phase_Min_Group_ML']!="Fspar", 'An']="N/A"
        Oxides_out['Or']=Fspar_Or
        Oxides_out.loc[Oxides_out['Phase_Min_Group_ML']!="Fspar", 'Or']="N/A"

    Oxides_out.loc[( (Fspar) & (Fspar_An>0.1) & (Fspar_Or<0.1) ), 'Phase_Mineral'] = "Plag"
    Oxides_out.loc[( (Fspar) & (Fspar_Or>0.1) & (Fspar_An<0.1)), 'Phase_Mineral'] = "Kspar"
    Oxides_out.loc[(Fspar & (Fspar_An<0.1) &  (Fspar_Or<0.1)), 'Phase_Mineral'] = "Albite"

    Oxides_out['Fspar_Class']=Oxides_out['Phase_Min_Group_ML']

    Oxides_out.loc[( (Fspar) & (Fspar_An<0.1) &  (Fspar_Or<0.1) ), 'Fspar_Class'] = "Albite"
    Oxides_out.loc[( (Fspar) & (Fspar_Or.between(0.1, 0.37) ) &  (Fspar_An<0.2)), 'Fspar_Class'] = "Anorthoclase"
    Oxides_out.loc[( (Fspar) & (Fspar_Or>0.37) &  (Fspar_An<0.2) ), 'Fspar_Class'] = "Sanidine"
    Oxides_out.loc[( (Fspar) & (Fspar_An.between(0.1, 0.3)) &  (Fspar_Or<0.1) ), 'Fspar_Class'] = "Oligoclase"
    Oxides_out.loc[( (Fspar) & (Fspar_An.between(0.3, 0.5)) &  (Fspar_Or<0.1) ), 'Fspar_Class'] = "Andesine"
    Oxides_out.loc[( (Fspar) & (Fspar_An.between(0.5, 0.7)) &  (Fspar_Or<0.1) ), 'Fspar_Class'] = "Labradorite"
    Oxides_out.loc[( (Fspar) & (Fspar_An.between(0.7, 0.9)) &  (Fspar_Or<0.1) ), 'Fspar_Class'] = "Bytownite"

    Oxides_out.loc[( (Fspar) & (Fspar_An>0.9) &  (Fspar_Or<0.1) ), 'Fspar_Class'] = "Anorthite"
    Oxides_out.loc[Oxides_out['Phase_Min_Group_ML']!="Fspar", 'Fspar_Class']="N/A"

    # This does Manual Classification for Pyroxenes
    Oxides_Pyroxenes=Oxides_out.add_suffix('_Opx')
    Px=Oxides_out['Phase_Min_Group_ML']=="Px"

    Px_CaMgFe=Oxides_out['Ca_CaMgFe']
    Oxides_out.loc[( (Px) & (Px_CaMgFe<0.05) ), 'Phase_Mineral'] = "Opx"
    Oxides_out.loc[( (Px) & (Px_CaMgFe.between(0.05, 0.2)) ), 'Phase_Mineral'] = "Pig"
    Oxides_out.loc[( (Px) & (Px_CaMgFe>0.2) ), 'Phase_Mineral'] = "Cpx"
    if return_end_members==True:
        Oxides_out['Ca/CaMgFe']=Px_CaMgFe
        Oxides_out.loc[Oxides_out['Phase_Min_Group_ML']!="Px", 'Ca/CaMgFe']="N/A"

    # This does manual classification for amphiboles, based on Leake (1997)


    return Oxides_out


def check_consecative(df):
    idx = df.index
    diffs = np.diff(idx)
    diff_one_check = (diffs == 1).all()
    index0=df.index[0]==0
    if index0 == True and diff_one_check == True:
        return True
    else:
        return False

def normalize_liquid_jorgenson(liq_comps):
    """ Normalizes for Jorgenson Thermometers, rounds to 2 dp"""
    print('Im normalizing using the Jorgenson method, e.g. 100 total, 2dp')

    Liq_test=liq_comps.copy()
    Liq_no_H2O=Liq_test.drop(labels=['Sample_ID_Liq', 'Fe3Fet_Liq', 'NiO_Liq',
                                   'CoO_Liq', 'CO2_Liq', 'H2O_Liq'], axis=1)
    if 'Sample_ID_Liq_Num' in Liq_no_H2O:
        Liq_no_H2O.drop('Sample_ID_Liq_Num', axis=1, inplace=True)

    Liq_no_H2O
    sum_row= 0.01*Liq_no_H2O.sum(axis=1)
    Liq_norm1=Liq_no_H2O.divide(sum_row, axis='rows')
    Liq_norm=Liq_norm1.round(decimals=2)

    Liq_norm['Fe3Fet_Liq']=liq_comps['Fe3Fet_Liq']
    Liq_norm['Sample_ID_Liq']=liq_comps['Sample_ID_Liq']
    Liq_norm['NiO_Liq']=liq_comps['NiO_Liq']
    Liq_norm['CoO_Liq']=liq_comps['CoO_Liq']
    Liq_norm['CO2_Liq']=liq_comps['CO2_Liq']
    Liq_norm['H2O_Liq']=liq_comps['H2O_Liq']
    return Liq_norm

def normalize_liquid_100_anhydrous_chompi(liq_comps):
    """ Normalizes liquid to 100% anhydrous for CHOMPI and other calcs

    """
    Liq_test=liq_comps.copy()
    Liq_no_H2O=Liq_test.drop(labels=['Sample_ID_Liq', 'Fe3Fet_Liq', 'MnO_Liq', 'Cr2O3_Liq', 'P2O5_Liq', 'NiO_Liq',
                                   'CoO_Liq', 'CO2_Liq', 'H2O_Liq'], axis=1)
    if 'Sample_ID_Liq_Num' in Liq_no_H2O:
        Liq_no_H2O.drop('Sample_ID_Liq_Num', axis=1, inplace=True)
    sum_row= 0.01*Liq_no_H2O.sum(axis=1)
    Liq_norm1=Liq_no_H2O.divide(sum_row, axis='rows')
    Liq_norm=Liq_norm1.copy()
    Liq_norm['Fe3Fet_Liq']=liq_comps['Fe3Fet_Liq']
    Liq_norm['Sample_ID_Liq']=liq_comps['Sample_ID_Liq']
    Liq_norm['NiO_Liq']=liq_comps['NiO_Liq']
    Liq_norm['CoO_Liq']=liq_comps['CoO_Liq']
    Liq_norm['CO2_Liq']=liq_comps['CO2_Liq']
    Liq_norm['H2O_Liq']=liq_comps['H2O_Liq']
    return Liq_norm

df_ideal_liq = pd.DataFrame(columns=['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq',
'FeOt_Liq', 'MnO_Liq', 'MgO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq',
'Cr2O3_Liq', 'P2O5_Liq', 'H2O_Liq', 'Fe3Fet_Liq', 'NiO_Liq', 'CoO_Liq',
 'CO2_Liq'])


