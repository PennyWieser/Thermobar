import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers

## This specifies the default order for each dataframe type used in calculations
df_ideal_liq = pd.DataFrame(columns=['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq',
'FeOt_Liq', 'MnO_Liq', 'MgO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq',
'Cr2O3_Liq', 'P2O5_Liq', 'H2O_Liq', 'Fe3FeT_Liq', 'NiO_Liq', 'CoO_Liq',
 'CO2_Liq'])

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
 'Fe3FeT_Liq_Err', 'NiO_Liq_Err', 'CoO_Liq_Err', 'CO2_Liq_Err'])

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



## Anhydrous mole proportions, mole fractions, cation proportions, cation fractions

def calculate_anhydrous_mol_proportions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------

    Liq_Comps: DataFrame
        Panda DataFrame of liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the form SiO2_Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = Liq_Comps.reindex(
        oxide_mass_liq_anhyd_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_anhyd_df, liq_wt],)
    # Drop the calculation column
    mol_prop_anhyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_anhydrous_mol_fractions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    Liq_Comps: DataFrame
                Panda DataFrame of liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_anhydrous_mol_proportions_liquid(Liq_Comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd


def calculate_anhydrous_cat_proportions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous cation proportions (e.g., mole proportions * no of cations)

   Parameters
    -------

    Liq_Comps: DataFrame
                Panda DataFrame of liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous cation proportions for the liquid with column headings of the form SiO2_Liq_cat_prop

    '''
    mol_prop_no_cat_num = calculate_anhydrous_mol_proportions_liquid(Liq_Comps)
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


def calculate_anhydrous_cat_fractions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous cation fractions

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous cation fractions for the liquid with column headings of the form SiO2_Liq_cat_frac, as well as the initial dataframe of liquid compositions.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Liq_cat_frac rather than Na_Liq_cat_frac.
    '''
    cat_prop = calculate_anhydrous_cat_proportions_liquid(Liq_Comps=Liq_Comps)
    mol_prop = calculate_anhydrous_mol_fractions_liquid(Liq_Comps=Liq_Comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd = pd.concat([Liq_Comps, mol_prop, cat_frac_anhyd], axis=1)
    cat_frac_anhyd['FeO_Liq_cat_frac'] = cat_frac_anhyd['FeOt_Liq_cat_frac'] * \
        (1 - Liq_Comps['Fe3FeT_Liq'])
    if "Fe3FeT_Liq" in cat_frac_anhyd:
        cat_frac_anhyd['Mg_Number_Liq_NoFe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] / 71.844))
        cat_frac_anhyd['Mg_Number_Liq_Fe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] * (1 - cat_frac_anhyd['Fe3FeT_Liq']) / 71.844))
    if "Fe3FeT_Liq" not in cat_frac_anhyd:
        cat_frac_anhyd['Mg_Number_Liq_Fe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] / 71.844))
        cat_frac_anhyd['Mg_Number_Liq_NoFe3'] = (cat_frac_anhyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_anhyd['MgO_Liq'] / 40.3044) + (cat_frac_anhyd['FeOt_Liq'] / 71.844))
    return cat_frac_anhyd

# Liquid Mgno function

def calculate_Liq_Mgno(Liq_Comps, Fe3FeT_Liq=None):
    '''
    Calculates Liquid Mg#

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Fe3FeT: opt, float, series, int
        Overwrites Fe3FeT_Liq column if specified

    Returns
    -------
    pandas Series
        Mg# of liquid
    '''

    if Fe3FeT_Liq is not None:
        Liq_Comps['Fe3FeT_Liq'] = Fe3FeT_Liq

    Liq_Mgno = (Liq_Comps['MgO_Liq'] / 40.3044) / ((Liq_Comps['MgO_Liq'] / 40.3044) +
                                                   (Liq_Comps['FeOt_Liq'] * (1 - Liq_Comps['Fe3FeT_Liq']) / 71.844))
    return Liq_Mgno

## Hydrous mole proportions, mole fractions, cation proportions, cation fractions


def calculate_hydrous_mol_proportions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------


    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the form SiO2_Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = Liq_Comps.reindex(oxide_mass_liq_hyd_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_hyd_df, liq_wt],)
    # Drop the calculation column
    mol_prop_hyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_hyd.columns = [
        str(col) + '_mol_prop_hyd' for col in mol_prop_hyd.columns]
    return mol_prop_hyd


def calculate_hydrous_mol_fractions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_hydrous_mol_proportions_liquid(Liq_Comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_hyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_hyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_hyd.columns = [str(col).replace('prop', 'frac')
                            for col in mol_frac_hyd.columns]
    return mol_frac_hyd


def calculate_hydrous_cat_proportions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous cation proportions (e.g., mole proportions * no of cations)

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous cation proportions for the liquid with column headings of the form SiO2_Liq_cat_prop

    '''
    mol_prop_no_cat_num = calculate_hydrous_mol_proportions_liquid(Liq_Comps)
    mol_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop_hyd', '') for col in mol_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_liq_hyd_df.reindex(
        oxide_mass_liq_hyd_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_no_cat_num])
    cation_prop_hyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_hyd.columns = [
        str(col) + '_cat_prop_hyd' for col in cation_prop_hyd.columns]
    return cation_prop_hyd


def calculate_hydrous_cat_fractions_liquid(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous cation fractions

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous cation fractions for the liquid with column headings of the form SiO2_Liq_cat_frac, as well as the initial dataframe of liquid compositions.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Liq_cat_frac rather than Na_Liq_cat_frac.
    '''
    cat_prop = calculate_hydrous_cat_proportions_liquid(Liq_Comps=Liq_Comps)
    mol_prop = calculate_hydrous_mol_fractions_liquid(Liq_Comps=Liq_Comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_hyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_hyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_hyd.columns = [str(col).replace('prop', 'frac')
                            for col in cat_frac_hyd.columns]
    cat_frac_hyd = pd.concat([Liq_Comps, mol_prop, cat_frac_hyd], axis=1)
    cat_frac_hyd['FeO_Liq_cat_frac_hyd'] = cat_frac_hyd['FeOt_Liq_cat_frac_hyd'] * \
        (1 - Liq_Comps['Fe3FeT_Liq'])
    if "Fe3FeT_Liq" in cat_frac_hyd:
        cat_frac_hyd['Mg_Number_Liq_NoFe3'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] / 71.844))
        cat_frac_hyd['Mg_Number_Liq_Fe3'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] * (1 - cat_frac_hyd['Fe3FeT_Liq']) / 71.844))
    if "Fe3FeT_Liq" not in cat_frac_hyd:
        cat_frac_hyd['Mg_Number_Liq'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] / 71.844))

        cat_frac_hyd['Mg_Number_Liq_NoFe3'] = (cat_frac_hyd['MgO_Liq'] / 40.3044) / (
            (cat_frac_hyd['MgO_Liq'] / 40.3044) + (cat_frac_hyd['FeOt_Liq'] / 71.844))

        #         myLiquids1['Mg_Number_Liq']=(myLiquids1['MgO_Liq']/40.3044)/( (myLiquids1['MgO_Liq']/40.3044)+ (myLiquids1['FeO_Liq']/71.844))
#         myLiquids1['Mg_Number_Liq_NoFe3']=(myLiquids1['MgO_Liq']/40.3044)/( (myLiquids1['MgO_Liq']/40.3044)+ (myLiquids1['FeOt_Liq']/71.844))
#     else:
#         myLiquids1['Mg_Number_Liq']=(myLiquids1['MgO_Liq']/40.3044)/( (myLiquids1['MgO_Liq']/40.3044)+ (myLiquids1['FeOt_Liq']/71.844))

    return cat_frac_hyd

    return cat_frac_anhyd
# Calculating Liquid mole and cation fractions including Ni for Pu et al.
# 2017 and 2019


## Anhydrous mole proportions, mole fractions, cation proportions, cation fractions including Ni for the thermometers of Pu et al.

def calculate_anhydrous_mol_proportions_liquid_Ni(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous mole proportions

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole proportions for the liquid with column headings of the form SiO2_Liq_mol_prop

    '''
    # This makes the input match the columns in the oxide mass dataframe
    liq_wt = Liq_Comps.reindex(
        oxide_mass_liq_anhyd_df_Ni.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    liq_wt_combo = pd.concat([oxide_mass_liq_anhyd_df_Ni, liq_wt],)
    # Drop the calculation column
    mol_prop_anhyd = liq_wt_combo.div(
        liq_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_anhydrous_mol_fractions_liquid_Ni(Liq_Comps):
    '''Import Liq compositions using Liq_Comps=My_Liquids, returns anhydrous mole fractions

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, TiO2_Liq etc.

    Returns
    -------
    pandas DataFrame
        anhydrous mole fractions for the liquid with column headings of the form SiO2_Liq_mol_frac

    '''
    mol_prop = calculate_anhydrous_mol_proportions_liquid_Ni(Liq_Comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd


def calculate_mol_proportions_olivine_Ni(Ol_Comps):
    '''Import Olivine compositions using Ol_Comps=My_Olivines, returns mole proportions

   Parameters
    -------

    Ol_Comps: DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for olivines with column headings of the form SiO2_Ol_mol_prop

    '''
    # This makes it match the columns in the oxide mass dataframe
    ol_wt = Ol_Comps.reindex(oxide_mass_ol_df_Ni.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ol_wt_combo = pd.concat([oxide_mass_ol_df_Ni, ol_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ol_wt_combo.div(
        ol_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_mol_fractions_olivine_Ni(Ol_Comps):
    '''Import Olivine compositions using Ol_Comps=My_Olivines, returns mole fractions

   Parameters
    -------

    Ol_Comps: DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for olivines with column headings of the form SiO2_Ol_mol_frac

    '''
    mol_prop = calculate_mol_proportions_olivine_Ni(Ol_Comps=Ol_Comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd

## Olivine mole proportions, fractions, cation proportions and fractions


def calculate_mol_proportions_olivine(Ol_Comps):
    '''Import Olivine compositions using Ol_Comps=My_Olivines, returns mole proportions

   Parameters
    -------

    Ol_Comps: DataFrame
            Panda DataFrame of olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for olivines with column headings of the form SiO2_Ol_mol_prop

    '''
    # This makes it match the columns in the oxide mass dataframe
    ol_wt = Ol_Comps.reindex(oxide_mass_ol_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ol_wt_combo = pd.concat([oxide_mass_ol_df, ol_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ol_wt_combo.div(
        ol_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_mol_fractions_olivine(Ol_Comps):
    '''Import Olivine compositions using Ol_Comps=My_Olivines, returns mole fractions

   Parameters
    -------

    Ol_Comps: DataFrame
            olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for olivines with column headings of the form SiO2_Ol_mol_frac

    '''
    mol_prop = calculate_mol_proportions_olivine(Ol_Comps=Ol_Comps)
    mol_prop['sum'] = mol_prop.sum(axis='columns')
    mol_frac_anhyd = mol_prop.div(mol_prop['sum'], axis='rows')
    mol_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    mol_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in mol_frac_anhyd.columns]
    return mol_frac_anhyd


def calculate_cat_proportions_olivine(Ol_Comps):
    '''Import Olivine compositions using Ol_Comps=My_Olivines, returns cation proportions

   Parameters
    -------

    Ol_Comps: DataFrame
            olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for olivine with column headings of the form SiO2_Ol_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Ol_cat_prop rather than Na_Ol_cat_prop.
    '''

    mol_prop_no_cat_num = calculate_mol_proportions_olivine(Ol_Comps=Ol_Comps)
    mol_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in mol_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_ol_df.reindex(
        oxide_mass_ol_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    return cation_prop_anhyd


def calculate_cat_fractions_olivine(Ol_Comps):
    '''Import Olivine compositions using Ol_Comps=My_Olivines, returns cation proportions

   Parameters
    -------

    Ol_Comps: DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for olivine with column headings of the form SiO2_Ol_cat_frac.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Ol_cat_frac rather than Na_Ol_cat_frac.

    '''

    cat_prop = calculate_cat_proportions_olivine(Ol_Comps=Ol_Comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]

    return cat_frac_anhyd

## Orthopyroxene mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_orthopyroxene(Opx_Comps):
    '''Import orthopyroxene compositions using Opx_Comps=My_Opxs, returns mole proportions

   Parameters
    -------

    Opx_Comps: DataFrame
        orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for orthopyroxene with column headings of the form SiO2_Opx_mol_prop

    '''

    # This makes the input match the columns in the oxide mass dataframe
    opx_wt = Opx_Comps.reindex(oxide_mass_opx_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    opx_wt_combo = pd.concat([oxide_mass_opx_df, opx_wt],)
    # Drop the calculation column
    mol_prop_anhyd = opx_wt_combo.div(
        opx_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_oxygens_orthopyroxene(Opx_Comps):
    '''Import orthopyroxene compositions using Opx_Comps=My_Opxs, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------

    Opx_Comps: DataFrame
        Panda DataFrame of orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Opx_ox

    '''

    mol_prop = calculate_mol_proportions_orthopyroxene(Opx_Comps=Opx_Comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_opx_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]
    return oxygens_anhyd


def calculate_orthopyroxene_6oxygens(Opx_Comps):
    '''Import orthopyroxene compositions using Opx_Comps=My_Opxs, returns cations on the basis of 6 oxygens.

   Parameters
    -------

    Opx_Comps: DataFrame
        Orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 6 oxygens, with column headings of the form SiO2_Opx_cat_6ox.
        For simplicity, and consistency of column labelling to aid calculations, oxide names are preserved,
        so outputs are Na2O_Opx_cat_6ox rather than Na_Opx_cat_6ox.

    '''

    oxygens = calculate_oxygens_orthopyroxene(Opx_Comps=Opx_Comps)
    renorm_factor = 6 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_orthopyroxene(Opx_Comps=Opx_Comps)
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
    cation_6.Al_VI_Opx_cat_6ox[cation_6.Al_VI_Opx_cat_6ox < 0] = 0
    cation_6['Si_Ti_Opx_cat_6ox'] = cation_6['SiO2_Opx_cat_6ox'] + \
        cation_6['TiO2_Opx_cat_6ox']

    return cation_6


def calculate_orthopyroxene_components(Opx_Comps):
    '''Import orthopyroxene compositions using Opx_Comps=My_Opxs, returns orthopyroxene components along with entered Cpx compositions

   Parameters
    -------

    Opx_Comps: DataFrame
        Panda DataFrame of orthopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.


    Returns
    -------
    pandas DataFrame
        -orthopyroxene compositions (MgO_Opx etc.)
        -cations on bases of 6 oxygens (column headings of form  Cr2O3_Opx_cat_6ox, as well as Cation_Sum_Opx)
        -orthopyroxene components (NaAlSi2O6, FmTiAlSiO6, CrAl2SiO6, FmAl2SiO6, CaFmSi2O6, Fm2Si2O6, En_Opx, Di_Opx, Mgno_Opx)

    '''

    opx_calc = calculate_orthopyroxene_6oxygens(Opx_Comps=Opx_Comps)
    # Sum of cations, used to filter bad analyses
    opx_calc['Cation_Sum_Opx'] = (opx_calc['SiO2_Opx_cat_6ox'] + opx_calc['TiO2_Opx_cat_6ox'] + opx_calc['Al2O3_Opx_cat_6ox'] + opx_calc['FeOt_Opx_cat_6ox']
                                  + opx_calc['MnO_Opx_cat_6ox'] + opx_calc['MgO_Opx_cat_6ox'] +
                                  opx_calc['CaO_Opx_cat_6ox'] +
                                  opx_calc['Na2O_Opx_cat_6ox']
                                  + opx_calc['K2O_Opx_cat_6ox'] + opx_calc['Cr2O3_Opx_cat_6ox'])

    opx_calc['NaAlSi2O6'] = opx_calc['Na2O_Opx_cat_6ox']
    opx_calc['FmTiAlSiO6'] = opx_calc['TiO2_Opx_cat_6ox']
    opx_calc['CrAl2SiO6'] = opx_calc['Cr2O3_Opx_cat_6ox']
    opx_calc['FmAl2SiO6'] = opx_calc['Al_VI_Opx_cat_6ox'] - \
        opx_calc['NaAlSi2O6'] - opx_calc['CrAl2SiO6']
    opx_calc.FmAl2SiO6[opx_calc.FmAl2SiO6 < 0] = 0
    opx_calc['CaFmSi2O6'] = opx_calc['CaO_Opx_cat_6ox']
    opx_calc['Fm2Si2O6'] = (((opx_calc['FeOt_Opx_cat_6ox'] + opx_calc['MgO_Opx_cat_6ox'] + opx_calc['MnO_Opx_cat_6ox'])
                             - opx_calc['FmTiAlSiO6'] - opx_calc['FmAl2SiO6'] - opx_calc['CaFmSi2O6']) / 2)
    opx_calc['En_Opx'] = opx_calc['Fm2Si2O6'] * (opx_calc['MgO_Opx_cat_6ox'] / (
        opx_calc['MgO_Opx_cat_6ox'] + opx_calc['FeOt_Opx_cat_6ox'] + opx_calc['MnO_Opx_cat_6ox']))
    opx_calc['Di_Opx'] = opx_calc['CaFmSi2O6'] * (opx_calc['MgO_Opx_cat_6ox'] / (
        opx_calc['MgO_Opx_cat_6ox'] + opx_calc['FeOt_Opx_cat_6ox'] + opx_calc['MnO_Opx_cat_6ox']))
    opx_calc['Mgno_OPX'] = (Opx_Comps['MgO_Opx'] / 40.3044) / \
        (Opx_Comps['MgO_Opx'] / 40.3044 + Opx_Comps['FeOt_Opx'] / 71.844)

    opx_calc = pd.concat([Opx_Comps, opx_calc], axis=1)
    return opx_calc


def calculate_orthopyroxene_liquid_components(
        *, Opx_Comps=None, Liq_Comps=None, MeltMatch=None):
    '''Import orthopyroxene compositions using Opx_Comps=My_Opxs and liquid compositions using Liq_Comps=My_Liquids,
        returns orthopyroxene and liquid components.

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.
    AND
     Opx_Comps: DataFrame
        orthopyroxene compositions with column headings SiO2_Opx, MgO_Opx etc.
    OR
    MeltMatch: DataFrame
       merged orthopyroxene and liquid compositions used for melt matching

    Returns
    -------
    pandas DataFrame
       Merged dataframe of inputted liquids, liquid mole fractions, liquid cation fractions,
       inputted opx compositions, opx cations on 6 oxygen basis, opx components and opx-liquid components.

    '''
    # For when users enter a combined dataframe MeltMatch=""
    if MeltMatch is not None:
        combo_liq_opxs = MeltMatch
    if Liq_Comps is not None and Opx_Comps is not None:
        if len(Liq_Comps) != len(Opx_Comps):
            raise Exception(
                "inputted dataframes for Liq_Comps and Opx_Comps need to have the same number of rows")
        else:
            myOPXs1_comp = calculate_orthopyroxene_components(
                Opx_Comps=Opx_Comps)
            myLiquids1_comps = calculate_anhydrous_cat_fractions_liquid(
                Liq_Comps=Liq_Comps)
            combo_liq_opxs = pd.concat(
                [myLiquids1_comps, myOPXs1_comp], axis=1)

    combo_liq_opxs['ln_Fm2Si2O6_liq'] = (np.log(combo_liq_opxs['Fm2Si2O6'].astype('float64') /
                                                (combo_liq_opxs['SiO2_Liq_cat_frac']**2 * (combo_liq_opxs['FeOt_Liq_cat_frac'] + combo_liq_opxs['MnO_Liq_cat_frac'] + combo_liq_opxs['MgO_Liq_cat_frac'])**2)))

    combo_liq_opxs['ln_FmAl2SiO6_liq'] = (np.log(combo_liq_opxs['FmAl2SiO6'].astype('float64') /
                                                 (combo_liq_opxs['SiO2_Liq_cat_frac'] * combo_liq_opxs['Al2O3_Liq_cat_frac']**2 * (combo_liq_opxs['FeOt_Liq_cat_frac']
                                                                                                                                   + combo_liq_opxs['MnO_Liq_cat_frac'] + combo_liq_opxs['MgO_Liq_cat_frac']))))
    combo_liq_opxs['Kd_Fe_Mg_Fet'] = ((combo_liq_opxs['FeOt_Opx'] / 71.844) / (combo_liq_opxs['MgO_Opx'] / 40.3044)) / (
        (combo_liq_opxs['FeOt_Liq'] / 71.844) / (combo_liq_opxs['MgO_Liq'] / 40.3044))
    combo_liq_opxs['Kd_Fe_Mg_Fe2'] = ((combo_liq_opxs['FeOt_Opx'] / 71.844) / (combo_liq_opxs['MgO_Opx'] / 40.3044)) / (
        ((1 - combo_liq_opxs['Fe3FeT_Liq']) * combo_liq_opxs['FeOt_Liq'] / 71.844) / (combo_liq_opxs['MgO_Liq'] / 40.3044))
    combo_liq_opxs['Ideal_Kd'] = 0.4805 - 0.3733 * \
        combo_liq_opxs['SiO2_Liq_cat_frac']
    combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'] = abs(
        combo_liq_opxs['Ideal_Kd'] - combo_liq_opxs['Kd_Fe_Mg_Fe2'])
    b = np.empty(len(combo_liq_opxs), dtype=str)
    for i in range(0, len(combo_liq_opxs)):

        if combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'].iloc[i] > 0.06:
            b[i] = str("No")
        if combo_liq_opxs['Delta_Kd_Fe_Mg_Fe2'].iloc[i] < 0.06:
            b[i] = str("Yes")
    combo_liq_opxs.insert(1, "Kd Eq (Put2008+-0.06)", b)

    return combo_liq_opxs

## Clinopyroxene mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_clinopyroxene(Cpx_Comps):
    '''Import clinopyroxene compositions using Cpx_Comps=My_Cpxs, returns mole proportions

   Parameters
    -------

    Cpx_Comps: DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for clinopyroxene with column headings of the form SiO2_Cpx_mol_prop

    '''

    # This makes the input match the columns in the oxide mass dataframe
    cpx_wt = Cpx_Comps.reindex(oxide_mass_cpx_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    cpx_wt_combo = pd.concat([oxide_mass_cpx_df, cpx_wt],)
    # Drop the calculation column
    mol_prop_anhyd = cpx_wt_combo.div(
        cpx_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_oxygens_clinopyroxene(Cpx_Comps):
    '''Import clinopyroxene compositions using Cpx_Comps=My_Cpxs, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------

    Cpx_Comps: DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Cpx_ox

    '''

    mol_prop = calculate_mol_proportions_clinopyroxene(Cpx_Comps=Cpx_Comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_cpx_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]
    return oxygens_anhyd


def calculate_clinopyroxene_6oxygens(Cpx_Comps):
    '''Import clinopyroxene compositions using Cpx_Comps=My_Cpxs, returns cations on the basis of 6 oxygens.

   Parameters
    -------

    Cpx_Comps: DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 6 oxygens, with column headings of the form SiO2_Cpx_cat_6ox.
        For simplicity, and consistency of column labelling to aid calculations, oxide names are preserved,
        so outputs are Na2O_Cpx_cat_6ox rather than Na_Cpx_cat_6ox.

    '''

    oxygens = calculate_oxygens_clinopyroxene(Cpx_Comps=Cpx_Comps)
    renorm_factor = 6 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_clinopyroxene(Cpx_Comps=Cpx_Comps)
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
    cation_6.Al_VI_cat_6ox[cation_6.Al_VI_cat_6ox < 0] = 0

    return cation_6

# Calculating Clinopyroxene components following Putirka spreadsheet


def calculate_clinopyroxene_components(Cpx_Comps):
    '''Import clinopyroxene compositions using Cpx_Comps=My_Cpxs, returns clinopyroxene components along with entered Cpx compositions

   Parameters
    -------

    Cpx_Comps: DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Returns
    -------
    pandas DataFrame
        Clinopyroxene components (column headings: Cation_Sum, CrCaTs, a_cpx_En, Mgno_CPX, Jd, CaTs, CaTi, DiHd_1996, DiHd_2003, En_Fs), cations on bases of 6 oxygens (column headings of form  Cr2O3_Cpx_cat_6ox), as well as inputted Cpx compositions (column headings of form MgO_Cpx)

    '''
    cpx_calc = calculate_clinopyroxene_6oxygens(Cpx_Comps=Cpx_Comps)

    # Sum of cations, used by Neave and Putirka (2017) to filter out bad
    # clinopyroxene analyses
    cpx_calc['Cation_Sum_Cpx'] = (cpx_calc['SiO2_Cpx_cat_6ox'] + cpx_calc['TiO2_Cpx_cat_6ox'] + cpx_calc['Al2O3_Cpx_cat_6ox'] + cpx_calc['FeOt_Cpx_cat_6ox']
                                  + cpx_calc['MnO_Cpx_cat_6ox'] + cpx_calc['MgO_Cpx_cat_6ox'] +
                                  cpx_calc['CaO_Cpx_cat_6ox'] +
                                  cpx_calc['Na2O_Cpx_cat_6ox']
                                  + cpx_calc['K2O_Cpx_cat_6ox'] + cpx_calc['Cr2O3_Cpx_cat_6ox'])

    # Cpx Components that don't nee if and else statements and don't rely on
    # others.
    cpx_calc['CrCaTs'] = 0.5 * cpx_calc['Cr2O3_Cpx_cat_6ox']
    cpx_calc['a_cpx_En'] = (1 - cpx_calc['CaO_Cpx_cat_6ox'] - cpx_calc['Na2O_Cpx_cat_6ox'] - cpx_calc['K2O_Cpx_cat_6ox']) * (1 - 0.5 * (cpx_calc['Al2O3_Cpx_cat_6ox']
                                                                                                                                        + cpx_calc['Cr2O3_Cpx_cat_6ox'] + cpx_calc['Na2O_Cpx_cat_6ox'] + cpx_calc['K2O_Cpx_cat_6ox']))
    cpx_calc['Mgno_CPX'] = (Cpx_Comps['MgO_Cpx'] / 40.3044) / \
        (Cpx_Comps['MgO_Cpx'] / 40.3044 + Cpx_Comps['FeOt_Cpx'] / 71.844)

    cpx_calc['Jd'] = np.empty(len(cpx_calc), dtype=float)
    cpx_calc['CaTs'] = np.empty(len(cpx_calc), dtype=float)
    cpx_calc['CaTi'] = np.empty(len(cpx_calc), dtype=float)
    cpx_calc['DiHd_1996'] = np.empty(len(cpx_calc), dtype=float)
    for i in range(0, len(cpx_calc)):

        if (cpx_calc['Al_VI_cat_6ox'].iloc[i]) > (
                cpx_calc['Na2O_Cpx_cat_6ox'].iloc[i]):
            cpx_calc['Jd'].iloc[i] = cpx_calc['Na2O_Cpx_cat_6ox'].iloc[i]
            cpx_calc['CaTs'].iloc[i] = cpx_calc['Al_VI_cat_6ox'].iloc[i] - \
                cpx_calc['Na2O_Cpx_cat_6ox'].iloc[i]
        else:
            cpx_calc['Jd'].iloc[i] = cpx_calc['Al_VI_cat_6ox'].iloc[i]
            cpx_calc['CaTs'].iloc[i] = 0

        if (cpx_calc['Al_IV_cat_6ox'].iloc[i]) > (cpx_calc['CaTs'].iloc[i]):
            cpx_calc['CaTi'].iloc[i] = (
                cpx_calc['Al_IV_cat_6ox'].iloc[i] - cpx_calc['CaTs'].iloc[i]) / 2
        else:
            cpx_calc['CaTi'].iloc[i] = 0
        if (cpx_calc['CaO_Cpx_cat_6ox'].iloc[i] - cpx_calc['CaTs'].iloc[i] -
                cpx_calc['CaTi'].iloc[i] - cpx_calc['CrCaTs'].iloc[i] > 0):
            cpx_calc['DiHd_1996'].iloc[i] = (
                cpx_calc['CaO_Cpx_cat_6ox'].iloc[i] - cpx_calc['CaTs'].iloc[i] - cpx_calc['CaTi'].iloc[i] - cpx_calc['CrCaTs'].iloc[i])
        else:
            cpx_calc['DiHd_1996'].iloc[i] = 0

    cpx_calc['EnFs'] = ((cpx_calc['FeOt_Cpx_cat_6ox'] +
                        cpx_calc['MgO_Cpx_cat_6ox']) - cpx_calc['DiHd_1996']) / 2
    cpx_calc['DiHd_2003'] = (cpx_calc['CaO_Cpx_cat_6ox'] -
                             cpx_calc['CaTs'] - cpx_calc['CaTi'] - cpx_calc['CrCaTs'])
    cpx_calc['Di_Cpx'] = cpx_calc['DiHd_2003'] * (cpx_calc['MgO_Cpx_cat_6ox'] / (
        cpx_calc['MgO_Cpx_cat_6ox'] + cpx_calc['MnO_Cpx_cat_6ox'] + cpx_calc['FeOt_Cpx_cat_6ox']))
    cpx_calc['DiHd_1996'] = cpx_calc['DiHd_1996'].clip(lower=0)
    cpx_calc['DiHd_2003'] = cpx_calc['DiHd_2003'].clip(lower=0)
    cpx_calc['Jd'] = cpx_calc['Jd'].clip(lower=0)
    # Merging new Cpx compnoents with inputted cpx composition
    cpx_combined = pd.concat([Cpx_Comps, cpx_calc], axis='columns')

    return cpx_combined



def calculate_clinopyroxene_liquid_components(
        *, Cpx_Comps=None, Liq_Comps=None, MeltMatch=None):
    '''Import clinopyroxene compositions using Cpx_Comps=My_Cpxs and liquid compositions using Liq_Comps=My_Liquids,
        returns clinopyroxene and liquid components.

   Parameters
    -------

    Liq_Comps: DataFrame
        liquid compositions with column headings SiO2_Liq, MgO_Liq etc.
    AND
     Cpx_Comps: DataFrame
        clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.
    OR
    MeltMatch: DataFrame
        Panda DataFrame of merged clinopyroxene and liquid compositions used for melt matching

    Returns
    -------
    pandas DataFrame
       Merged dataframe of inputted liquids, liquid mole fractions, liquid cation fractions,
       inputted cpx compositions, cpx cations on 6 oxygen basis, cpx components and cpx-liquid components.

    '''
    # For when users enter a combined dataframe MeltMatch=""
    if MeltMatch is not None:
        combo_liq_cpxs = MeltMatch
        if "Sample_ID_Cpx" and "Sample_ID_Liq" in combo_liq_cpxs:
            combo_liq_cpxs = combo_liq_cpxs.drop(
                ['Sample_ID_Cpx', 'Sample_ID_Liq'], axis=1).astype('float64')

    if Liq_Comps is not None and Cpx_Comps is not None:
        if len(Liq_Comps) != len(Cpx_Comps):
            raise Exception(
                "inputted dataframes for Liq_Comps and Cpx_Comps need to have the same number of rows")
        else:
            myCPXs1_comp = calculate_clinopyroxene_components(
                Cpx_Comps=Cpx_Comps).reset_index(drop=True)
            myLiquids1_comps = calculate_anhydrous_cat_fractions_liquid(
                Liq_Comps=Liq_Comps).reset_index(drop=True)
            combo_liq_cpxs = pd.concat(
                [myLiquids1_comps, myCPXs1_comp], axis=1)
            if "Sample_ID_Cpx" in combo_liq_cpxs:
                combo_liq_cpxs.drop(['Sample_ID_Cpx'], axis=1)
            if "Sample_ID_Liq" in combo_liq_cpxs:
                combo_liq_cpxs.drop(['Sample_ID_Liq'], axis=1)


# Measured Kd Fe-Mg (using 2+)
    combo_liq_cpxs['Kd_Fe_Mg'] = ((combo_liq_cpxs['FeOt_Cpx_cat_6ox'] / combo_liq_cpxs['MgO_Cpx_cat_6ox']) / (
        (combo_liq_cpxs['FeOt_Liq_cat_frac'] * (1 - combo_liq_cpxs['Fe3FeT_Liq']) / combo_liq_cpxs['MgO_Liq_cat_frac'])))
# Measured Kd Fe-Mg using Fet
    combo_liq_cpxs['Kd_Fe_Mg_NoFe3'] = ((combo_liq_cpxs['FeOt_Cpx_cat_6ox'] / combo_liq_cpxs['MgO_Cpx_cat_6ox']) / (
        (combo_liq_cpxs['FeOt_Liq_cat_frac'] / combo_liq_cpxs['MgO_Liq_cat_frac'])))

    combo_liq_cpxs['lnK_Jd_liq'] = np.log((combo_liq_cpxs['Jd']) / ((combo_liq_cpxs['Na2O_Liq_cat_frac']) * (
        combo_liq_cpxs['Al2O3_Liq_cat_frac']) * ((combo_liq_cpxs['SiO2_Liq_cat_frac'])**2)))

    combo_liq_cpxs['lnK_Jd_DiHd_liq_1996'] = np.log((combo_liq_cpxs['Jd']) * (combo_liq_cpxs['CaO_Liq_cat_frac']) * ((combo_liq_cpxs['FeOt_Liq_cat_frac']) + (
        combo_liq_cpxs['MgO_Liq_cat_frac'])) / ((combo_liq_cpxs['DiHd_1996']) * (combo_liq_cpxs['Na2O_Liq_cat_frac']) * (combo_liq_cpxs['Al2O3_Liq_cat_frac'])))

    combo_liq_cpxs['lnK_Jd_DiHd_liq_2003'] = np.log((combo_liq_cpxs['Jd']) * (combo_liq_cpxs['CaO_Liq_cat_frac']) * ((combo_liq_cpxs['FeOt_Liq_cat_frac']) + (
        combo_liq_cpxs['MgO_Liq_cat_frac'])) / ((combo_liq_cpxs['DiHd_2003']) * (combo_liq_cpxs['Na2O_Liq_cat_frac']) * (combo_liq_cpxs['Al2O3_Liq_cat_frac'])))

    combo_liq_cpxs['Kd_Fe_Mg_IdealWB'] = 0.109 + 0.186 * \
        combo_liq_cpxs['Mgno_CPX']  # equation 35 of wood and blundy


# Different ways to calculate DeltaFeMg

    combo_liq_cpxs['DeltaFeMg_WB'] = abs(
        combo_liq_cpxs['Kd_Fe_Mg'] - combo_liq_cpxs['Kd_Fe_Mg_IdealWB'])
    # Adding back in sample names
    if MeltMatch is not None and "Sample_ID_Cpx" in MeltMatch:
        combo_liq_cpxs['Sample_ID_Cpx'] = MeltMatch['Sample_ID_Cpx']
        combo_liq_cpxs['Sample_ID_Liq'] = MeltMatch['Sample_ID_Liq']
    if MeltMatch is not None and "Sample_ID_Cpx" not in MeltMatch:
        combo_liq_cpxs['Sample_ID_Cpx'] = MeltMatch.index
        combo_liq_cpxs['Sample_ID_Liq'] = MeltMatch.index
    if Liq_Comps is not None and "Sample_ID_Liq" in Liq_Comps:
        combo_liq_cpxs['Sample_ID_Liq'] = Liq_Comps['Sample_ID_Liq']
    if Liq_Comps is not None and "Sample_ID_Liq" not in Liq_Comps:
        combo_liq_cpxs['Sample_ID_Liq'] = Liq_Comps.index
    if Cpx_Comps is not None and "Sample_ID_Cpx" not in Cpx_Comps:
        combo_liq_cpxs['Sample_ID_Cpx'] = Cpx_Comps.index
    if Cpx_Comps is not None and "Sample_ID_Cpx" in Cpx_Comps:
        combo_liq_cpxs['Sample_ID_Cpx'] = Cpx_Comps['Sample_ID_Cpx']

    combo_liq_cpxs.replace([np.inf, -np.inf], np.nan, inplace=True)

    return combo_liq_cpxs

## Feldspar mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_plagioclase(*, Plag_Comps=None):
    '''Import plagioclase compositions using Plag_Comps=My_plagioclases, returns mole proportions

   Parameters
    -------

    Plag_Comps: DataFrame
            Panda DataFrame of plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for plagioclases with column headings of the form SiO2_Plag_mol_prop

    '''

    if Plag_Comps is not None:
        plag_comps = Plag_Comps
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


def calculate_mol_fractions_plagioclase(*, Plag_Comps=None):
    '''Import plagioclase compositions using Plag_Comps=My_plagioclases, returns mole fractions

   Parameters
    -------

    Plag_Comps: DataFrame
            plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for plagioclases with column headings of the form SiO2_Plag_mol_frac


    '''

    if Plag_Comps is not None:
        plag_comps = Plag_Comps
        plag_prop = calculate_mol_proportions_plagioclase(
            Plag_Comps=plag_comps)
        plag_prop['sum'] = plag_prop.sum(axis='columns')
        plag_frac_anhyd = plag_prop.div(plag_prop['sum'], axis='rows')
        plag_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
        plag_frac_anhyd.columns = [str(col).replace(
            'prop', 'frac') for col in plag_frac_anhyd.columns]
        return plag_frac_anhyd


def calculate_cat_proportions_plagioclase(*, Plag_Comps=None):
    '''Import plagioclase compositions using Plag_Comps=My_plagioclases, returns cation proportions

   Parameters
    -------

    Plag_Comps: DataFrame
            plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for plagioclase with column headings of the form SiO2_Plag_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Plag_cat_prop rather than Na_Plag_cat_prop.
    '''

    plag_prop_no_cat_num = calculate_mol_proportions_plagioclase(
        Plag_Comps=Plag_Comps)
    plag_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in plag_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_plag_df.reindex(
        oxide_mass_plag_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, plag_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    return cation_prop_anhyd


def calculate_cat_fractions_plagioclase(*, Plag_Comps=None):
    '''Import plagioclase compositions using Plag_Comps=My_plagioclases, returns cation fractions

   Parameters
    -------

    Plag_Comps: DataFrame
        plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for plagioclase with column headings of the form SiO2_Plag_cat_frac.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Plag_cat_frac rather than Na_Plag_cat_frac.

    '''

    cat_prop = calculate_cat_proportions_plagioclase(Plag_Comps=Plag_Comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd['An_Plag'] = cat_frac_anhyd['CaO_Plag_cat_frac'] / \
        (cat_frac_anhyd['CaO_Plag_cat_frac'] +
         cat_frac_anhyd['Na2O_Plag_cat_frac'] + cat_frac_anhyd['K2O_Plag_cat_frac'])
    cat_frac_anhyd['Ab_Plag'] = cat_frac_anhyd['Na2O_Plag_cat_frac'] / \
        (cat_frac_anhyd['CaO_Plag_cat_frac'] +
         cat_frac_anhyd['Na2O_Plag_cat_frac'] + cat_frac_anhyd['K2O_Plag_cat_frac'])
    cat_frac_anhyd['Or_Plag'] = 1 - \
        cat_frac_anhyd['An_Plag'] - cat_frac_anhyd['Ab_Plag']
    cat_frac_anhyd2 = pd.concat([Plag_Comps, cat_prop, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

# Calculating alkali feldspar components


def calculate_mol_proportions_kspar(*, Kspar_Comps=None):
    '''Import AlkaliFspar compositions using Kspar_Comps=My_kspars, returns mole proportions

   Parameters
    -------

    Kspar_Comps: DataFrame
            Panda DataFrame of AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for AlkaliFspars with column headings of the form SiO2_Kspar_mol_prop

    '''

    if Kspar_Comps is not None:
        alk_comps = Kspar_Comps
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


def calculate_mol_fractions_kspar(*, Kspar_Comps=None):
    '''Import AlkaliFspar compositions using Kspar_Comps=My_kspars, returns mole fractions

   Parameters
    -------

    Kspar_Comps: DataFrame
            AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for AlkaliFspars with column headings of the form SiO2_Kspar_mol_frac


    '''

    if Kspar_Comps is not None:
        alk_comps = Kspar_Comps
        alk_prop = calculate_mol_proportions_kspar(Kspar_Comps=alk_comps)
        alk_prop['sum'] = alk_prop.sum(axis='columns')
        alk_frac_anhyd = alk_prop.div(alk_prop['sum'], axis='rows')
        alk_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
        alk_frac_anhyd.columns = [str(col).replace(
            'prop', 'frac') for col in alk_frac_anhyd.columns]
        return alk_frac_anhyd


def calculate_cat_proportions_kspar(*, Kspar_Comps=None):
    '''Import AlkaliFspar compositions using Kspar_Comps=My_kspars, returns cation proportions

   Parameters
    -------

    Kspar_Comps: DataFrame
            AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for AlkaliFspar with column headings of the form SiO2_Kspar_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Kspar_cat_prop rather than Na_Kspar_cat_prop.
    '''

    alk_prop_no_cat_num = calculate_mol_proportions_kspar(
        Kspar_Comps=Kspar_Comps)
    alk_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in alk_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_kspar_df.reindex(
        oxide_mass_kspar_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, alk_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    return cation_prop_anhyd


def calculate_cat_fractions_Kspar(*, Kspar_Comps=None):
    '''Import AlkaliFspar compositions using Kspar_Comps=My_kspars, returns cation fractions

   Parameters
    -------

    Kspar_Comps: DataFrame
        AlkaliFspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for AlkaliFspar with column headings of the form SiO2_Kspar_cat_frac.
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Kspar_cat_frac rather than Na_Kspar_cat_frac.

    '''

    cat_prop = calculate_cat_proportions_kspar(Kspar_Comps=Kspar_Comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd['An_Kspar'] = cat_frac_anhyd['CaO_Kspar_cat_frac'] / \
        (cat_frac_anhyd['CaO_Kspar_cat_frac'] +
         cat_frac_anhyd['Na2O_Kspar_cat_frac'] + cat_frac_anhyd['K2O_Kspar_cat_frac'])
    cat_frac_anhyd['Ab_Kspar'] = cat_frac_anhyd['Na2O_Kspar_cat_frac'] / \
        (cat_frac_anhyd['CaO_Kspar_cat_frac'] +
         cat_frac_anhyd['Na2O_Kspar_cat_frac'] + cat_frac_anhyd['K2O_Kspar_cat_frac'])
    cat_frac_anhyd['Or_Kspar'] = 1 - \
        cat_frac_anhyd['An_Kspar'] - cat_frac_anhyd['Ab_Kspar']
    cat_frac_anhyd2 = pd.concat(
        [Kspar_Comps, cat_prop, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

## Amphibole mole proportions, fractions, cation proportions and fractions

def calculate_mol_proportions_amphibole(Amp_Comps):
    '''Import amphibole compositions using Amp_Comps=My_amphiboles, returns mole proportions

   Parameters
    -------

    Amp_Comps: DataFrame
            Panda DataFrame of amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for amphiboles with column headings of the form SiO2_Amp_mol_prop

    '''

    # This makes it match the columns in the oxide mass dataframe
    amp_wt = Amp_Comps.reindex(oxide_mass_amp_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    amp_wt_combo = pd.concat([oxide_mass_amp_df, amp_wt],)
    # Drop the calculation column
    mol_prop_anhyd = amp_wt_combo.div(
        amp_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_mol_fractions_amphibole(Amp_Comps):
    '''Import amphibole compositions using Amp_Comps=My_amphiboles, returns mole fractions

   Parameters
    -------

    Amp_Comps: DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        mole fractions for amphiboles with column headings of the form SiO2_Amp_mol_frac

    '''


def calculate_cat_proportions_amphibole(*, Amp_Comps=None):
    '''Import amphibole compositions using Amp_Comps=My_amphiboles, returns cation proportions

   Parameters
    -------

    Amp_Comps: DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for amphibole with column headings of the form SiO2_Amp_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Amp_cat_prop rather than Na_Amp_cat_prop.

    '''

    amp_prop_no_cat_num = calculate_mol_proportions_amphibole(
        Amp_Comps=Amp_Comps)
    amp_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in amp_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_amp_df.reindex(
        oxide_mass_amp_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, amp_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    return cation_prop_anhyd


def calculate_oxygens_amphibole(Amp_Comps):
    '''Import amphiboles compositions using Amp_Comps=My_Amps, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------

    Amp_Comps: DataFrame
        amphiboles compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Amp_ox

    '''

    mol_prop = calculate_mol_proportions_amphibole(Amp_Comps=Amp_Comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_amp_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]
    return oxygens_anhyd


def calculate_amphibole_23oxygens(Amp_Comps):
    '''Import amphibole compositions using Amp_Comps=My_Amps, returns cations on the basis of 23 oxygens.

   Parameters
    -------

    Amp_Comps: DataFrame
        amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 23 oxygens, with column headings of the form SiO2_Amp_cat_23ox.
        For simplicity, and consistency of column labelling to aid calculations, oxide names are preserved,
        so outputs are Na2O_Amp_cat_23ox rather than Na_Amp_cat_23ox.

    '''

    oxygens = calculate_oxygens_amphibole(Amp_Comps=Amp_Comps)
    renorm_factor = 23 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_amphibole(Amp_Comps=Amp_Comps)
    mol_prop['oxy_renorm_factor'] = renorm_factor
    mol_prop_23 = mol_prop.multiply(mol_prop['oxy_renorm_factor'], axis='rows')
    mol_prop_23.columns = [str(col).replace('_mol_prop', '')
                           for col in mol_prop_23.columns]

    ox_num_reindex = cation_num_amp_df.reindex(
        mol_prop_23.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_23])
    cation_23 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_23.columns = [str(col).replace('_mol_prop', '_cat_23ox')
                         for col in mol_prop.columns]
    cation_23['cation_sum_Si_Mg'] = cation_23['SiO2_Amp_cat_23ox'] + cation_23['TiO2_Amp_cat_23ox'] + cation_23['Al2O3_Amp_cat_23ox'] + \
        cation_23['Cr2O3_Amp_cat_23ox'] + cation_23['FeOt_Amp_cat_23ox'] + \
        cation_23['MnO_Amp_cat_23ox'] + cation_23['MgO_Amp_cat_23ox']
    cation_23['cation_sum_Si_Ca'] = cation_23['cation_sum_Si_Mg'] + \
        cation_23['CaO_Amp_cat_23ox']
    cation_23['cation_sum_All'] = cation_23['cation_sum_Si_Ca'] + \
        cation_23['Na2O_Amp_cat_23ox'] + +cation_23['K2O_Amp_cat_23ox']

    return cation_23
# Ridolfi Amphiboles, using Cl and F, does on 13 cations.


def calculate_mol_proportions_amphibole_Ridolfi(Amp_Comps):
    '''Import amphibole compositions using Amp_Comps=My_amphiboles, returns mole proportions

   Parameters
    -------

    Amp_Comps: DataFrame
            Panda DataFrame of amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for amphiboles with column headings of the form SiO2_Amp_mol_prop

    '''

    # This makes it match the columns in the oxide mass dataframe
    amp_wt = Amp_Comps.reindex(
        oxide_mass_amp_df_Ridolfi.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    amp_wt_combo = pd.concat([oxide_mass_amp_df_Ridolfi, amp_wt],)
    # Drop the calculation column
    mol_prop_anhyd = amp_wt_combo.div(
        amp_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_cat_proportions_amphibole_Ridolfi(*, Amp_Comps=None):
    '''Import amphibole compositions using Amp_Comps=My_amphiboles, returns cation proportions

   Parameters
    -------

    Amp_Comps: DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cation proportions for amphibole with column headings of the form SiO2_Amp_cat_prop
        For simplicity, and consistency of column heading types, oxide names are preserved,
        so outputs are Na2O_Amp_cat_prop rather than Na_Amp_cat_prop.

    '''

    amp_prop_no_cat_num = calculate_mol_proportions_amphibole_Ridolfi(
        Amp_Comps=Amp_Comps)
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


def calculate_amphibole_13cations_Ridolfi(Amp_Comps):
    '''Import amphibole compositions using Amp_Comps=My_amphiboles, returns
    components calculated on basis of 13 cations following Ridolfi supporting information

   Parameters
    -------

    Amp_Comps: DataFrame
            amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    Returns
    -------
    pandas DataFrame
        cation fractions for amphiboles with column headings of the form SiO2_Amp_13_cat...

    '''
    cats = calculate_cat_proportions_amphibole_Ridolfi(Amp_Comps=Amp_Comps)
    cats['cation_sum_Si_Mg'] = (cats['SiO2_Amp_cat_prop'] + cats['TiO2_Amp_cat_prop'] + cats['Al2O3_Amp_cat_prop'] +
                                cats['Cr2O3_Amp_cat_prop'] + cats['FeOt_Amp_cat_prop'] + cats['MnO_Amp_cat_prop'] + cats['MgO_Amp_cat_prop'])
    sum_SiMg = cats['cation_sum_Si_Mg']
    cats.drop(['cation_sum_Si_Mg'], axis=1, inplace=True)
    cat_13 = 13 * cats.divide(sum_SiMg, axis='rows')
    cat_13.columns = [str(col).replace('_cat_prop', '_13_cat')
                      for col in cat_13.columns]
    cat_13['cation_sum_Si_Mg'] = sum_SiMg
    cat_13_out = pd.concat([cats, cat_13], axis=1)
    return cat_13_out

## Equilibrium tests clinopyroxene

def calculate_Cpx_Eq_Tests(*, MeltMatch=None, Liq_Comps=None, Cpx_Comps=None,
                           Fe3FeT_Liq=None, P=None, T=None, sigma=1, KdErr=0.03):
    '''
    Calculates Kd Fe-Mg, EnFs, DiHd, CaTs for cpx-liquid pairs

   Parameters
    -------

    Cpx_Comps: DataFrame
        Clinopyroxene compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Liq_Comps: DataFrame
        Liquid compositions with column headings SiO2_Liq, MgO_Liq etc.

    MeltMatch: DataFrame
        Combined Cpx-Liquid compositions. Used for "melt match" functionality.


    P_kbar: int, float, series
        Pressure in kbar at which to evaluate equilibrium tests

    T: int, float, series
        Temprature in K at which to evaluate equilibrium tests

    Fe3FeT_Liq: int, float, series (optional)
        Fe3FeT ratio used to assess Kd Fe-Mg equilibrium between cpx and melt.
        If None, uses Fe3FeT_Liq from Liq_Comps.
        If specified, overwrites the Fe3FeT_Liq column in the liquid input.

    sigma: int or float
        determines sigma level at which to consider the DiHd, EnFs and CaTs tests laid out by Neave et al. (2019)

    KdErr: int or float
        deviation from the Kd value of Putirka (2008) for the equilibrium tests of Neave et al. (2019)

   Returns
    -------

    DataFrame
        Equilibrium tests, cation fractions, and inputted liquid and cpx compositions.


    '''

    if MeltMatch is not None:
        Combo_liq_cpxs = MeltMatch
    if Liq_Comps is not None and Cpx_Comps is not None:
        Liq_Comps_c = Liq_Comps.copy()
        if Fe3FeT_Liq is not None:
            Liq_Comps_c['Fe3FeT_Liq'] = Fe3FeT_Liq
        else:
            print('Using Fe3FeT from input file to calculate Kd Fe-Mg')
        Combo_liq_cpxs = calculate_clinopyroxene_liquid_components(
            Cpx_Comps=Cpx_Comps, Liq_Comps=Liq_Comps_c)

    Combo_liq_cpxs['P_kbar_calc'] = P
    Combo_liq_cpxs['T_K_calc'] = T
    #     # Equation 35 of Putirka 2008 - theoretical Kd-Mg exchange coefficient
    Combo_liq_cpxs['Kd_Ideal_Put'] = np.exp(-0.107 - 1719 / T)  # eq 35
    ratioMasotta = Combo_liq_cpxs['Na2O_Liq_cat_frac'] / (
        Combo_liq_cpxs['Na2O_Liq_cat_frac'] + Combo_liq_cpxs['K2O_Liq_cat_frac'])
    Combo_liq_cpxs['Kd_Ideal_Masotta'] = np.exp(
        1.735 - 3056 / T - 1.668 * ratioMasotta)  # eq35 alk, for trachytes and phonolites
    Combo_liq_cpxs['Delta_Kd_Put2008'] = abs(
        Combo_liq_cpxs['Kd_Ideal_Put'] - Combo_liq_cpxs['Kd_Fe_Mg'])
    Combo_liq_cpxs['Delta_Kd_Mas2013'] = abs(
        Combo_liq_cpxs['Kd_Ideal_Masotta'] - Combo_liq_cpxs['Kd_Fe_Mg'])

    # Equation X of Mollo for DiHd and EnFs components
    Combo_liq_cpxs['DiHd_Pred_Mollo'] = (np.exp(-2.18 - 3.16 * Combo_liq_cpxs['TiO2_Liq_cat_frac'] - 0.365 * np.log(Combo_liq_cpxs['Al2O3_Liq_cat_frac'])
                                                + 0.05 * np.log(Combo_liq_cpxs['MgO_Liq_cat_frac']) - 3858.2 * (
                                                    Combo_liq_cpxs['EnFs']**2 / T) + (2107.4 / T)
                                                - 17.64 * P / T))

    Combo_liq_cpxs['EnFs_Pred_Mollo'] = (np.exp(0.018 - 9.61 * Combo_liq_cpxs['CaO_Liq_cat_frac'] +
                                                7.46 *
                                                Combo_liq_cpxs['MgO_Liq_cat_frac'] *
                                                Combo_liq_cpxs['SiO2_Liq_cat_frac']
                                                - 0.34 *
                                                np.log(
                                                    Combo_liq_cpxs['Al2O3_Liq_cat_frac'])
                                                - 3.78 * (Combo_liq_cpxs['Na2O_Liq_cat_frac'] + Combo_liq_cpxs['K2O_Liq_cat_frac']) -
                                                3737.3 * (Combo_liq_cpxs['DiHd_1996']**2) / T - 46.8 * P / T))

    #     # Putirka 1999 equations
    Combo_liq_cpxs['CaTs_Pred_P1999'] = (np.exp(2.58 + 0.12 * P / T - 9 * 10**(-7) * P**2 / T
                                                + 0.78 * np.log(Combo_liq_cpxs['CaO_Liq_cat_frac'] * Combo_liq_cpxs['Al2O3_Liq_cat_frac']**2 * Combo_liq_cpxs['SiO2_Liq_cat_frac']) - 4.3 * 10**3 * (Combo_liq_cpxs['DiHd_1996']**2 / T)))

    Combo_liq_cpxs['CrCaTS_Pred_P1999'] = (np.exp(12.8) * Combo_liq_cpxs['CaO_Liq_cat_frac'] * (
        Combo_liq_cpxs['Cr2O3_Liq_cat_frac']**2) * Combo_liq_cpxs['SiO2_Liq_cat_frac'])

    #     #Calculating deltas -e.g., absolute difference between theoreitcal and observed using Mollo and P1999
    Combo_liq_cpxs['Delta_EnFs'] = abs(
        Combo_liq_cpxs['EnFs'] - Combo_liq_cpxs['EnFs_Pred_Mollo'])
    Combo_liq_cpxs['Delta_CaTs'] = abs(
        Combo_liq_cpxs['CaTs'] - Combo_liq_cpxs['CaTs_Pred_P1999'])
    Combo_liq_cpxs['Delta_DiHd'] = abs(
        Combo_liq_cpxs['DiHd_1996'] - Combo_liq_cpxs['DiHd_Pred_Mollo'])

    b = np.empty(len(Combo_liq_cpxs), dtype=str)
    for i in range(0, len(Combo_liq_cpxs)):

        if ((Combo_liq_cpxs['Delta_Kd_Put2008'].iloc[i] < KdErr) & (Combo_liq_cpxs['Delta_EnFs'].iloc[i] < 0.05 * sigma) &
                (Combo_liq_cpxs['Delta_CaTs'].iloc[i] < 0.03 * sigma) & (Combo_liq_cpxs['Delta_DiHd'].iloc[i] < 0.06 * sigma)):
            b[i] = str("Yes")
        else:
            b[i] = str("No")
    Combo_liq_cpxs.insert(1, "Eq Tests Neave2017?", b)

    cols_to_move = ['P_kbar_calc', 'T_K_calc', "Eq Tests Neave2017?",
                    'Delta_Kd_Put2008', 'Delta_Kd_Mas2013', 'Delta_EnFs', 'Delta_CaTs', 'Delta_DiHd']
    Combo_liq_cpxs = Combo_liq_cpxs[cols_to_move +
                                    [col for col in Combo_liq_cpxs.columns if col not in cols_to_move]]

    return Combo_liq_cpxs


def calculate_Cpx_Opx_Equilibrium_Tests(Cpx_Comps, Opx_Comps):
    '''
    Import Cpx and Opx compositions, assesses degree of Fe-Mg disequilibrium.
    Parameters
    -------

    Cpx_Comps: DataFrame
            Cpx compositions with column headings SiO2_Cpx, MgO_Cpx etc.

    Opx_Comps: DataFrame
            Cpx compositions with column headings SiO2_Opx, MgO_Opx etc.
    Returns
    -------
    pandas DataFrame
        Return all opx and cpx components, as well as values for Kd-Fe Opx-Cpx.
    '''
    cpx_components = calculate_clinopyroxene_components(Cpx_Comps=Cpx_Comps)
    opx_components = calculate_orthopyroxene_components(Opx_Comps=Opx_Comps)
    two_pyx = pd.concat([cpx_components, opx_components], axis=1)
    two_pyx['En'] = (two_pyx['Fm2Si2O6'] * (two_pyx['MgO_Opx_cat_6ox'] / (two_pyx['MgO_Opx_cat_6ox'] +
                                                               two_pyx['FeOt_Cpx_cat_6ox'] + two_pyx['MnO_Cpx_cat_6ox'])))
    two_pyx['Kd_Fe_Mg_Cpx_Opx'] = ((two_pyx['FeOt_Cpx_cat_6ox'] / two_pyx['MgO_Cpx_cat_6ox'])) / (
        two_pyx['FeOt_Opx_cat_6ox'] / two_pyx['MgO_Opx_cat_6ox'])

    Lindley_Fe3_Opx = (two_pyx['Na2O_Opx_cat_6ox'] + two_pyx['Al_IV_Opx_cat_6ox'] -
        two_pyx['Al_VI_Opx_cat_6ox'] - 2 * two_pyx['TiO2_Opx_cat_6ox'] - two_pyx['Cr2O3_Opx_cat_6ox'])
    Lindley_Fe3_Opx[Lindley_Fe3_Opx < 0] = 0
    two_pyx['Lindley_Fe3_Opx']=Lindley_Fe3_Opx
    a_En_opx_mod = (((0.5 * two_pyx['MgO_Opx_cat_6ox'] / (0.5 * (two_pyx['FeOt_Opx_cat_6ox'] - two_pyx['Lindley_Fe3_Opx'])
    + 0.5 * two_pyx['MgO_Opx_cat_6ox'] + two_pyx['Na2O_Opx_cat_6ox'] +two_pyx['CaO_Opx_cat_6ox'] + two_pyx['MnO_Opx_cat_6ox'])))
    * (0.5 * two_pyx['MgO_Opx_cat_6ox'] / (0.5 * two_pyx['MgO_Opx_cat_6ox'] + 0.5 * (two_pyx['FeOt_Opx_cat_6ox'] - two_pyx['Lindley_Fe3_Opx'])
    + two_pyx['TiO2_Opx_cat_6ox'] + two_pyx['Al_VI_Opx_cat_6ox'] + two_pyx['Cr2O3_Opx_cat_6ox'] + two_pyx['Lindley_Fe3_Opx'])))

    Lindley_Fe3_Cpx = two_pyx['Na2O_Cpx_cat_6ox'] + two_pyx['Al_IV_cat_6ox'] - \
        two_pyx['Al_VI_cat_6ox'] - 2 * two_pyx['TiO2_Cpx_cat_6ox'] - two_pyx['Cr2O3_Cpx_cat_6ox']
    Lindley_Fe3_Cpx[Lindley_Fe3_Cpx < 0] = 0
    two_pyx['Lindley_Fe3_Cpx']=Lindley_Fe3_Cpx
    two_pyx['a_Di_cpx'] = two_pyx['CaO_Cpx_cat_6ox'] / (two_pyx['CaO_Cpx_cat_6ox'] + 0.5 * two_pyx['MgO_Cpx_cat_6ox'] + 0.5 * (
        two_pyx['FeOt_Cpx_cat_6ox'] - two_pyx['Lindley_Fe3_Cpx']) + two_pyx['MnO_Cpx_cat_6ox'] + two_pyx['Na2O_Cpx_cat_6ox'])
    two_pyx['Kf'] = two_pyx['CaO_Opx_cat_6ox'] / (1 - two_pyx['CaO_Cpx_cat_6ox'])

    two_pyx['a_En_opx_mod'] = (((0.5 * two_pyx['MgO_Opx_cat_6ox'] / (0.5 * (two_pyx['FeOt_Opx_cat_6ox'] - two_pyx['Lindley_Fe3_Opx'])
    + 0.5 * two_pyx['MgO_Opx_cat_6ox'] + two_pyx['Na2O_Opx_cat_6ox'] +two_pyx['CaO_Opx_cat_6ox'] + two_pyx['MnO_Opx_cat_6ox'])))
    * (0.5 * two_pyx['MgO_Opx_cat_6ox'] / (0.5 * two_pyx['MgO_Opx_cat_6ox'] + 0.5 * (two_pyx['FeOt_Opx_cat_6ox'] - two_pyx['Lindley_Fe3_Opx'])
    + two_pyx['TiO2_Opx_cat_6ox'] + two_pyx['Al_VI_Opx_cat_6ox'] + two_pyx['Cr2O3_Opx_cat_6ox'] + two_pyx['Lindley_Fe3_Opx'])))



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

    two_pyx.insert(1, "High T Kd Eq?", a)
    two_pyx.insert(2, "Low T Kd Eq?", b)
    return two_pyx


def calculate_plagioclase_liquid_components(*, Plag_Comps=None, Liq_Comps=None, XAb=None, XAn=None, XOr=0,
P, T):

    '''
    Import Plag and Liq compositions, assesses An, Ab and Or equilibrium using the
    equations of Putirka (2005)
    Parameters
    -------

    Liq_Comps: DataFrame
            Cpx compositions with column headings SiO2_Liq, MgO_Liq etc.

    One of:

    1) Plag_Comps: DataFrame (optional)
        Plag compositions with column headings SiO2_Plag, MgO_Plag etc.


    2) XAn, XAb, XOr, float, int, series
        If Plag_Comps is None, enter XAn and XAb for plagioclases instead.
        XOr is set to zero by default, but can be overwritten for equilibrium tests
    P: int, float, Series
        Pressure in kbar

    T: int, float, Series
        Temperature in Kelvin

    Returns
    -------
    pandas DataFrame
        Return all plag components, liq compnoents, and  and cpx components,
        as well as values for Observed Kd (Ab-An), which Putirka (2008)
        suggests is the best equilibrium test. Also calculate Delta_An, which
        is the absolute value of measured An - predicted An from Putirka (2008).
    '''

    cat_liqs = calculate_anhydrous_cat_fractions_liquid(Liq_Comps)
    if Plag_Comps is not None and Liq_Comps is not None:
        cat_plags = calculate_cat_fractions_plagioclase(Plag_Comps=Plag_Comps)
        combo_plag_liq = pd.concat([cat_plags, cat_liqs], axis=1)
    else:
        combo_plag_liq = cat_liqs
        combo_plag_liq['An_Plag']=XAn
        combo_plag_liq['Ab_Plag']=XAb
        combo_plag_liq['Or_Plag']=XOr


    combo_plag_liq['P'] = P
    combo_plag_liq['T'] = T
    Pred_An_EqE = (np.exp(-3.485 + 22.93 * combo_plag_liq['CaO_Liq_cat_frac'] + 0.0805 * combo_plag_liq['H2O_Liq']
                          + 1.0925 * combo_plag_liq['CaO_Liq_cat_frac'] / (combo_plag_liq['CaO_Liq_cat_frac'] + combo_plag_liq['Na2O_Liq_cat_frac']) +
                          13.11 * combo_plag_liq['Al2O3_Liq_cat_frac'] / (
                              combo_plag_liq['Al2O3_Liq_cat_frac'] + combo_plag_liq['SiO2_Liq_cat_frac'])
                          + 5.59258 *
                          combo_plag_liq['SiO2_Liq_cat_frac']**3 -
                          38.786 * P / (T)
                          - 125.04 *
                          combo_plag_liq['CaO_Liq_cat_frac'] *
                          combo_plag_liq['Al2O3_Liq_cat_frac']
                          + 8.958 * combo_plag_liq['SiO2_Liq_cat_frac'] * combo_plag_liq['K2O_Liq_cat_frac'] - 2589.27 / (T)))

    Pred_Ab_EqF = (np.exp(-2.748 - 0.1553 * combo_plag_liq['H2O_Liq'] + 1.017 * combo_plag_liq['Mg_Number_Liq_NoFe3'] - 1.997 * combo_plag_liq['SiO2_Liq_cat_frac']**3 + 54.556 * P / T
                          - 67.878 *
                          combo_plag_liq['K2O_Liq_cat_frac'] *
                          combo_plag_liq['Al2O3_Liq_cat_frac']
                          - 99.03 * combo_plag_liq['CaO_Liq_cat_frac'] * combo_plag_liq['Al2O3_Liq_cat_frac'] + 4175.307 / T))

    Pred_Or_EqG = (np.exp(19.42 - 12.5 * combo_plag_liq['MgO_Liq_cat_frac'] - 161.4 * combo_plag_liq['Na2O_Liq_cat_frac']
                          - 16.65 * combo_plag_liq['CaO_Liq_cat_frac'] / (
                              combo_plag_liq['CaO_Liq_cat_frac'] + combo_plag_liq['Na2O_Liq_cat_frac'])
                          - 528.1 * combo_plag_liq['K2O_Liq_cat_frac'] * combo_plag_liq['Al2O3_Liq_cat_frac'] -
                          19.38 * combo_plag_liq['SiO2_Liq_cat_frac']**3
                          + 168.2 *
                          combo_plag_liq['SiO2_Liq_cat_frac'] *
                          combo_plag_liq['Na2O_Liq_cat_frac']
                          - 1951.2 * combo_plag_liq['CaO_Liq_cat_frac'] * combo_plag_liq['K2O_Liq_cat_frac'] - 10190 / T))

    Obs_Kd_Ab_An=(combo_plag_liq['Ab_Plag']*combo_plag_liq['Al2O3_Liq_cat_frac']*combo_plag_liq['CaO_Liq_cat_frac']/
    (combo_plag_liq['An_Plag']*combo_plag_liq['Na2O_Liq_cat_frac']*combo_plag_liq['SiO2_Liq_cat_frac']))

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
                relevantT[i] = 'Low T: Yes'
        if T[i] >= (1050+273.15):
            if Obs_Kd_Ab_An.iloc[i]>=0.17 and Obs_Kd_Ab_An.iloc[i]<=0.39:
                relevantT[i] = 'High T: Yes'
            else:
                relevantT[i] = 'High T: No'


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

def calculate_Fspar_activity_components(*, Ab_Plag, Or_Plag, An_Plag, Ab_Kspar, Or_Kspar, An_Kspar, T, P):

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


def calculate_Plag_Components(*, CaO_Liq_cat_frac, H2O_Liq, Na2O_Liq_cat_frac, Al2O3_Liq_cat_frac,
    SiO2_Liq_cat_frac, K2O_Liq_cat_frac, T, P, An_Plag, Ab_Plag, Mg_Number_Liq_NoFe3, MgO_Liq_cat_frac):

    An_Pred=np.exp(-3.485+22.93*CaO_Liq_cat_frac+0.0805*H2O_Liq+1.0925*CaO_Liq_cat_frac/(CaO_Liq_cat_frac+Na2O_Liq_cat_frac)
    +13.11*Al2O3_Liq_cat_frac/(Al2O3_Liq_cat_frac+SiO2_Liq_cat_frac)+5.59258*SiO2_Liq_cat_frac**3-
    38.786*P/(T)-125.04*CaO_Liq_cat_frac*Al2O3_Liq_cat_frac+8.958*SiO2_Liq_cat_frac*K2O_Liq_cat_frac-2589.27/(T))
    Ab_Pred=np.exp(-2.748-0.1553*H2O_Liq+1.017*Mg_Number_Liq_NoFe3-1.997*SiO2_Liq_cat_frac**3+54.556*P/T-67.878*K2O_Liq_cat_frac*Al2O3_Liq_cat_frac
    -99.03*CaO_Liq_cat_frac*Al2O3_Liq_cat_frac+4175.307/T)
    Or_Pred=np.exp(19.42-12.5*MgO_Liq_cat_frac--161.4*Na2O_Liq_cat_frac-16.65*CaO_Liq_cat_frac/(CaO_Liq_cat_frac+Na2O_Liq_cat_frac)
    -528.1*K2O_Liq_cat_frac*Al2O3_Liq_cat_frac-19.38*SiO2_Liq_cat_frac**3+168.2*SiO2_Liq_cat_frac*Na2O_Liq_cat_frac
    -1951.2*CaO_Liq_cat_frac*K2O_Liq_cat_frac-10190/T)
    Obs_Kd_Ab_An=Ab_Plag*Al2O3_Liq_cat_frac*CaO_Liq_cat_frac/(An_Plag*Na2O_Liq_cat_frac*SiO2_Liq_cat_frac)
    Components=pd.DataFrame(data={'An_Pred': An_Pred, 'Ab_Pred': Ab_Pred, 'Or_Pred': Or_Pred})

    return Components


