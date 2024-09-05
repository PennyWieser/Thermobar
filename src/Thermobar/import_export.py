import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from pathlib import Path


from Thermobar.core import *

## This specifies the default order for each dataframe type used in calculations
df_ideal_liq = pd.DataFrame(columns=['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq',
'FeOt_Liq', 'MnO_Liq', 'MgO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq',
'Cr2O3_Liq', 'P2O5_Liq', 'H2O_Liq', 'Fe3Fet_Liq', 'NiO_Liq', 'CoO_Liq',
 'CO2_Liq'])

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

df_ideal_gt = pd.DataFrame(columns=['SiO2_Gt', 'TiO2_Gt', 'Al2O3_Gt',
'Cr2O3_Gt', 'FeOt_Gt', 'MnO_Gt', 'MgO_Gt', 'CaO_Gt', 'Na2O_Gt', 'K2O_Gt',
'Ni_Gt', 'Ti_Gt', 'Zr_Gt', 'Zn_Gt', 'Ga_Gt', 'Sr_Gt', 'Y_Gt'])

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

def import_lepr_file(filename):
    """
    Reads in data from the outputs from the Caltech LEPR site (where oxides are followed by the word "value")
    Splits the data into phases, as for the read_excel function.

   Parameters
    -------
    filename: str
        Excel filename from LEPR

    Returns
    -------
    pandas DataFrames stored in a dictionary. E.g., Access Cpxs using output.Cpxs
        my_input = pandas dataframe of the entire spreadsheet
        mylabels = sample labels
        Experimental_PT = User-entered PT
        Fluid=pandas dataframe of fluid compositions
        Liqs=pandas dataframe of liquid oxides
        Ols=pandas dataframe of olivine oxides
        Cpxs=pandas dataframe of cpx oxides
        Plags=pandas dataframe of plagioclase oxides
        Kspars=pandas dataframe of kspar oxides
        Opxs=pandas dataframe of opx oxides
        Amps=pandas dataframe of amphibole oxides
        Sps=pandas dataframe of spinel oxides


    """

    my_input = pd.ExcelFile(filename)
    sheet_names = my_input.sheet_names

    if "Experiment" in sheet_names:
        my_input_Exp = pd.read_excel(filename, sheet_name="Experiment")
        myExp = pd.DataFrame(data={'Citation': my_input_Exp['Citation'], 'Experiment': my_input_Exp['Experiment'],
                                   'T_K': my_input_Exp['T (C)']+273.15, 'P_kbar': 10*my_input_Exp['P (GPa)'],
                                   'Duration': my_input_Exp['Duration (hours)'],
                                   'Laboratory': my_input_Exp['Laboratory']
                                   }
                             )
    if "Experiment" not in sheet_names:
        myExp = pd.DataFrame()

    if "Fluid" in sheet_names:
        my_input_Fluid = pd.read_excel(filename, sheet_name="Fluid").fillna(0)
        myFluid = pd.DataFrame(data={'Experiment': my_input_Fluid['Experiment'],
                                     'H2O_Val': my_input_Fluid['H2O value'],
                                     'CO2_Val': my_input_Fluid['CO2 value']})
        myFluid_Exp = pd.merge(myFluid, myExp, on="Experiment")

    if "Fluid" not in sheet_names:
        myFluid = pd.DataFrame()
        myFluid_Exp = pd.DataFrame()

    if "Plagioclase" in sheet_names:
        my_input_Plag = pd.read_excel(
            filename, sheet_name="Plagioclase").fillna(0)
        my_input_Plag['FeOT value'] = my_input_Plag['FeO value'] + \
            0.89998*my_input_Plag['Fe2O3 value']
        myPlag = pd.DataFrame(data={'Experiment': my_input_Plag['Experiment'],
                                    'SiO2_Plag': my_input_Plag['SiO2 value'],
                                    'TiO2_Plag': my_input_Plag['TiO2 value'],
                                    'Al2O3_Plag': my_input_Plag['Al2O3 value'],
                                    'FeOt_Plag': my_input_Plag['FeOT value'],
                                    'MnO_Plag': my_input_Plag['MnO value'],
                                    'MgO_Plag': my_input_Plag['MgO value'],
                                    'CaO_Plag': my_input_Plag['CaO value'],
                                    'Na2O_Plag': my_input_Plag['Na2O value'],
                                    'K2O_Plag': my_input_Plag['K2O value'],
                                    'Cr2O3_Plag': my_input_Plag['Cr2O3 value'],
                                    'P2O5_Plag': my_input_Plag['P2O5 value']})
        myPlag_Exp = pd.merge(myPlag, myExp, on="Experiment")

    if "Plagioclase" not in sheet_names:
        myPlag = pd.DataFrame()
        myPlag_Exp = pd.DataFrame()
    if "Clinopyroxene" in sheet_names:
        my_input_Cpx = pd.read_excel(
            filename, sheet_name="Clinopyroxene").fillna(0)
        my_input_Cpx['FeOT value'] = my_input_Cpx['FeO value'] + \
            0.89998*my_input_Cpx['Fe2O3 value']
        myCpx = pd.DataFrame(data={'Experiment': my_input_Cpx['Experiment'],
                                   'SiO2_Cpx': my_input_Cpx['SiO2 value'],
                                   'TiO2_Cpx': my_input_Cpx['TiO2 value'],
                                   'TiO2_Cpx_Err': my_input_Cpx['TiO2 error'],
                                   'Al2O3_Cpx': my_input_Cpx['Al2O3 value'],
                                   'Al2O3_Cpx_Err': my_input_Cpx['Al2O3 error'],
                                   'FeOt_Cpx': my_input_Cpx['FeOT value'],
                                   'MnO_Cpx': my_input_Cpx['MnO value'],
                                   'MgO_Cpx': my_input_Cpx['MgO value'],
                                   'CaO_Cpx': my_input_Cpx['CaO value'],
                                   'Na2O_Cpx': my_input_Cpx['Na2O value'],
                                    'Na2O_Cpx_Err': my_input_Cpx['Na2O error'],
                                   'K2O_Cpx': my_input_Cpx['K2O value'],
                                   'Cr2O3_Cpx': my_input_Cpx['Cr2O3 value'],
                                   'P2O5_Cpx': my_input_Cpx['P2O5 value']})
        if 'n' in my_input_Cpx.columns:
            myCpx['N_meas_Cpx']=my_input_Cpx['n']
        if 'Number of analyses' in my_input_Cpx.columns:
            myCpx['N_meas_Cpx']=my_input_Cpx['Number of analyses']
        myCpx_Exp = pd.merge(myCpx, myExp, on="Experiment")


    if "Clinopyroxene" not in sheet_names:
        myCpx = pd.DataFrame()
        myCpx_Exp = pd.DataFrame()

    if "Orthopyroxene" in sheet_names:
        my_input_Opx = pd.read_excel(
            filename, sheet_name="Orthopyroxene").fillna(0)
        my_input_Opx['FeOT value'] = my_input_Opx['FeO value'] + \
            0.89998*my_input_Opx['Fe2O3 value']
        myOpx = pd.DataFrame(data={'Experiment': my_input_Opx['Experiment'],
                                   'SiO2_Opx': my_input_Opx['SiO2 value'],
                                   'TiO2_Opx': my_input_Opx['TiO2 value'],
                                   'Al2O3_Opx': my_input_Opx['Al2O3 value'],
                                   'FeOt_Opx': my_input_Opx['FeOT value'],
                                   'MnO_Opx': my_input_Opx['MnO value'],
                                   'MgO_Opx': my_input_Opx['MgO value'],
                                   'CaO_Opx': my_input_Opx['CaO value'],
                                   'Na2O_Opx': my_input_Opx['Na2O value'],
                                   'K2O_Opx': my_input_Opx['K2O value'],
                                   'Cr2O3_Opx': my_input_Opx['Cr2O3 value'],
                                   'P2O5_Opx': my_input_Opx['P2O5 value']})
        myOpx_Exp = pd.merge(myOpx, myExp, on="Experiment")

    if "Orthopyroxene" not in sheet_names:
        myOpx = pd.DataFrame()
        myOpx_Exp = pd.DataFrame()

    if "Liquid" in sheet_names:
        my_input_Liq = pd.read_excel(filename, sheet_name="Liquid").fillna(0)
        my_input_Liq['FeOT value'] = my_input_Liq['FeO value'] + \
            0.89998*my_input_Liq['Fe2O3 value']
        myLiq = pd.DataFrame(data={'Experiment': my_input_Liq['Experiment'],
                                   'SiO2_Liq': my_input_Liq['SiO2 value'],
                                   'TiO2_Liq': my_input_Liq['TiO2 value'],
                                   'Al2O3_Liq': my_input_Liq['Al2O3 value'],
                                   'FeOt_Liq': my_input_Liq['FeOT value'],
                                   'MnO_Liq': my_input_Liq['MnO value'],
                                   'MgO_Liq': my_input_Liq['MgO value'],
                                   'CaO_Liq': my_input_Liq['CaO value'],
                                   'Na2O_Liq': my_input_Liq['Na2O value'],
                                   'K2O_Liq': my_input_Liq['K2O value'],
                                   'Cr2O3_Liq': my_input_Liq['Cr2O3 value'],
                                   'P2O5_Liq': my_input_Liq['P2O5 value'],
                                   'H2O_Liq': my_input_Liq['H2O value'],
                                   'Total_Liq': my_input_Liq['total value']})
        if 'n' in my_input_Liq.columns:
            myLiq['N_meas_Liq']=my_input_Liq['n']

        myLiq_Exp = pd.merge(myLiq, myExp, on="Experiment")
    if "Liquid" not in sheet_names:
        myLiq = pd.DataFrame()
        myLiq_Exp = pd.DataFrame()
    if "Amphibole" in sheet_names:
        my_input_Amp = pd.read_excel(
            filename, sheet_name="Amphibole").fillna(0)
        my_input_Amp['FeOT value'] = my_input_Amp['FeO value'] + \
            0.89998*my_input_Amp['Fe2O3 value']
        myAmp = pd.DataFrame(data={'Experiment': my_input_Amp['Experiment'],
                                   'SiO2_Amp': my_input_Amp['SiO2 value'],
                                   'TiO2_Amp': my_input_Amp['TiO2 value'],
                                   'Al2O3_Amp': my_input_Amp['Al2O3 value'],
                                   'FeOt_Amp': my_input_Amp['FeOT value'],
                                   'MnO_Amp': my_input_Amp['MnO value'],
                                   'MgO_Amp': my_input_Amp['MgO value'],
                                   'CaO_Amp': my_input_Amp['CaO value'],
                                   'Na2O_Amp': my_input_Amp['Na2O value'],
                                   'K2O_Amp': my_input_Amp['K2O value'],
                                   'Cr2O3_Amp': my_input_Amp['Cr2O3 value'],
                                   'P2O5_Amp': my_input_Amp['P2O5 value']})
        myAmp_Exp = pd.merge(myAmp, myExp, on="Experiment")

    if "Amphibole" not in sheet_names:
        myAmp = pd.DataFrame()
        myAmp_Exp = pd.DataFrame()

    if "Olivine" in sheet_names:
        my_input_Ol = pd.read_excel(filename, sheet_name="Olivine").fillna(0)
        my_input_Ol['FeOT value'] = my_input_Ol['FeO value'] + \
            0.89998*my_input_Ol['Fe2O3 value']
        myOl = pd.DataFrame(data={'Experiment': my_input_Ol['Experiment'],
                            'SiO2_Ol': my_input_Ol['SiO2 value'],
                                  'TiO2_Ol': my_input_Ol['TiO2 value'],
                                  'Al2O3_Ol': my_input_Ol['Al2O3 value'],
                                  'FeOt_Ol': my_input_Ol['FeOT value'],
                                  'MnO_Ol': my_input_Ol['MnO value'],
                                  'MgO_Ol': my_input_Ol['MgO value'],
                                  'CaO_Ol': my_input_Ol['CaO value'],
                                  'Na2O_Ol': my_input_Ol['Na2O value'],
                                  'K2O_Ol': my_input_Ol['K2O value'],
                                  'Cr2O3_Ol': my_input_Ol['Cr2O3 value'],
                                  'P2O5_Ol': my_input_Ol['P2O5 value']})
        myOl_Exp = pd.merge(myOl, myExp, on="Experiment")

    if "Olivine" not in sheet_names:
        myOl = pd.DataFrame()
        myOl_Exp = pd.DataFrame()

    return {'Experimental_PT': myExp, 'Liquids': myLiq, 'Liqs_Exp': myLiq_Exp, 'Fluids': myFluid, 'Fluids_Exp': myFluid_Exp, 'Plags': myPlag, 'Plags_Exp': myPlag_Exp,
            'Cpxs': myCpx,  'Cpxs_Exp': myCpx_Exp, 'Opxs': myOpx, 'Opxs_Exp': myOpx_Exp, 'Amps': myAmp, 'Amps_Exp': myAmp_Exp, 'Ols_Exp': myOl_Exp}


# Loading Excel, returns a disctionry
def import_excel(file_name, sheet_name,  path=None, sample_label=None, GEOROC=False, suffix=None, df=None):
    '''
    Import excel sheet of oxides in wt%, headings should be of the form SiO2_Liq (for the melt/liquid), SiO2_Ol (for olivine comps), SiO2_Cpx (for clinopyroxene compositions). Order doesn't matter


   Parameters
    -------

    filename: .xlsx, .csv, .xls file
        Compositional data as an Excel spreadsheet (.xlsx, .xls) or a comma separated values (.csv) file with columns labelled SiO2_Liq, SiO2_Ol, SiO2_Cpx etc, and each row corresponding to an analysis.

    filename: str
        specifies the file name (e.g., Python_OlLiq_Thermometers_Test.xlsx)

    OR
    enter a dataframe instead of an excel file.

    Optional:

    path: provide a pathlib path to where the file is stored

    GEOROC: bool (defualt False) - reads in GEOROC files

    suffix: default None, can be '_Liq' etc. Used to add a suffix onto all column headings.

    Returns
    -------
    pandas DataFrames stored in a dictionary. E.g., Access Cpxs using output.Cpxs
        my_input = pandas dataframe of the entire spreadsheet
        mylabels = sample labels
        Experimental_press_temp = User-entered PT
        Liqs=pandas dataframe of liquid oxides
        Ols=pandas dataframe of olivine oxides
        Cpxs=pandas dataframe of cpx oxides
        Plags=pandas dataframe of plagioclase oxides
        Kspars=pandas dataframe of kspar oxides
        Opxs=pandas dataframe of opx oxides
        Amps=pandas dataframe of amphibole oxides
        Sps=pandas dataframe of spinel oxides
    '''

    if df is None:

        if path is not None:
            file_path = Path(path) / file_name
        else:
            file_path = Path(file_name)

        # Convert to string if needed
        if isinstance(file_path, Path):
            file_path = str(file_path)

        # Check the file extension and read the file accordingly
        if file_path.endswith('.csv'):
            my_input = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            if sheet_name is not None:
                my_input = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                my_input = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {Path(file_path).suffix}")
    else:
        my_input=df


    if any(my_input.columns.str.startswith(' ')):
        w.warn('your input file has some columns that start with spaces. This could cause you big problems if they are at the start of oxide names. Please ammend your file and reload.')
    if suffix is not None:
        if any(my_input.columns.str.contains(suffix)):
            w.warn('We notice you have specified a suffix, but some of your columns already have this suffix. '
        'e.g., If you already have _Liq in the file, you shouldnt specify suffix="_Liq" during the import')


    my_input_c = my_input.copy()
    if suffix is not None:
        my_input_c=my_input_c.add_suffix(suffix)

    if any(my_input.columns.str.contains("_cpx")):
        w.warn("You've got a column heading with a lower case _cpx, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Cpx)" )

    if any(my_input.columns.str.contains("_opx")):
        w.warn("You've got a column heading with a lower case _opx, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Opx)" )

    if any(my_input.columns.str.contains("_plag")):
        w.warn("You've got a column heading with a lower case _plag, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Plag)" )

    if any(my_input.columns.str.contains("_kspar")):
       w.warn("You've got a column heading with a lower case _kspar, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Kspar)" )

    if any(my_input.columns.str.contains("_sp")):
        w.warn("You've got a column heading with a lower case _sp, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Sp)" )

    if any(my_input.columns.str.contains("_ol")):
        w.warn("You've got a column heading with a lower case _ol, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Ol)" )

    if any(my_input.columns.str.contains("_amp")):
        w.warn("You've got a column heading with a lower case _amp, this is okay if this campumn is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Amp)" )

    if any(my_input.columns.str.contains("_liq")):
        w.warn("You've got a column heading with a lower case _liq, this is okay if this column is for your"
        " own use, but if its an input to Thermobar, it needs to be capitalized (_Liq)" )

    if suffix is not None:
        if any(my_input.columns.str.contains("FeO")) and (all(my_input.columns.str.contains("FeOt")==False)):
            raise ValueError("No FeOt found. You've got a column heading with FeO. To avoid errors based on common EPMA outputs"
            " thermobar only recognises columns with FeOt for all phases except liquid"
            " where you can also enter a Fe3Fet_Liq heading used for equilibrium tests")

    if any(my_input.columns.str.contains("FeO_")) and (all(my_input.columns.str.contains("FeOt_")==False)):

        if any(my_input.columns.str.contains("FeO_Liq")) and any(my_input.columns.str.contains("Fe2O3_Liq")):
            my_input_c['FeOt_Liq']=my_input_c['FeO_Liq']+my_input_c['Fe2O3_Liq']*0.89998


        else:
            raise ValueError("No FeOt found. You've got a column heading with FeO. To avoid errors based on common EPMA outputs"
        " thermobar only recognises columns with FeOt for all phases except liquid"
        " where you can also enter a Fe3Fet_Liq heading used for equilibrium tests")

    # if any(my_input.columns.str.contains("Fe2O3_")) and (all(my_input.columns.str.contains("FeOt_")==False)):
    #     raise ValueError("No FeOt column found. You've got a column heading with Fe2O3. To avoid errors based on common EPMA outputs"
    #     " thermobar only recognises columns with FeOt for all phases except liquid"
    #     " where you can also enter a Fe3Fet_Liq heading used for equilibrium tests")

    if any(my_input.columns.str.contains("FeOT_")) and (all(my_input.columns.str.contains("FeOt_")==False)):
        raise ValueError("No FeOt column found. You've got a column heading with FeOT. Change to a lower case t")



 #   myLabels=my_input.Sample_ID
    duplicates = df_ideal_exp.columns[df_ideal_exp.columns.duplicated()]

    # If duplicates are found, return them to the user
    if not duplicates.empty:
        print("Duplicate columns found:", duplicates)
        return "Duplicate columns detected. Please handle them before reindexing."

    # Proceed with reindexing if no duplicates are found
    Experimental_press_temp1 = my_input.reindex(df_ideal_exp.columns, axis=1)



    if GEOROC is True:
        my_input_c.loc[np.isnan(my_input_c['FeOt_Liq']) is True, 'FeOt_Liq'] = my_input_c.loc[np.isnan(
            my_input_c['FeOt_Liq']) is True, 'FeO_Liq'] + my_input_c.loc[np.isnan(my_input_c['FeOt_Liq']) is True, 'Fe2O3_Liq'] * 0.8999998

    myOxides1 = my_input_c.reindex(df_ideal_oxide.columns, axis=1).fillna(0)
    myOxides1 = myOxides1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myOxides1[myOxides1 < 0] = 0

    myLiquids1 = my_input_c.reindex(df_ideal_liq.columns, axis=1).fillna(0)
    myLiquids1 = myLiquids1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myLiquids1[myLiquids1 < 0] = 0

    myCPXs1 = my_input_c.reindex(df_ideal_cpx.columns, axis=1).fillna(0)
    myCPXs1 = myCPXs1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myCPXs1[myCPXs1 < 0] = 0

    myOls1 = my_input_c.reindex(df_ideal_ol.columns, axis=1).fillna(0)
    myOls1 = myOls1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myOls1[myOls1 < 0] = 0

    myPlags1 = my_input_c.reindex(df_ideal_plag.columns, axis=1).fillna(0)
    myPlags1 = myPlags1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myPlags1[myPlags1 < 0] = 0

    myKspars1 = my_input_c.reindex(df_ideal_kspar.columns, axis=1).fillna(0)
    myKspars1 = myKspars1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myKspars1[myKspars1 < 0] = 0

    myOPXs1 = my_input_c.reindex(df_ideal_opx.columns, axis=1).fillna(0)
    myOPXs1 = myOPXs1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myOPXs1[myOPXs1 < 0] = 0

    mySps1 = my_input_c.reindex(df_ideal_sp.columns, axis=1).fillna(0)
    mySps1 = mySps1.apply(pd.to_numeric, errors='coerce').fillna(0)
    mySps1[mySps1 < 0] = 0

    myAmphs1 = my_input_c.reindex(df_ideal_amp.columns, axis=1).fillna(0)
    myAmphs1 = myAmphs1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myAmphs1[myAmphs1 < 0] = 0

    myGts1 = my_input_c.reindex(df_ideal_gt.columns, axis=1).fillna(0)
    myGts1 = myGts1.apply(pd.to_numeric, errors='coerce').fillna(0)
    myGts1[myGts1 < 0] = 0

    # Adding sample Names
    if "Sample_ID_Cpx" in my_input_c:
        myCPXs1['Sample_ID_Cpx'] = my_input_c['Sample_ID_Cpx']
    else:
        myCPXs1['Sample_ID_Cpx'] = my_input.index

    if "Sample_ID_Opx" in my_input_c:
        myOPXs1['Sample_ID_Opx'] = my_input_c['Sample_ID_Opx']
    else:
        myOPXs1['Sample_ID_Opx'] = my_input.index

    if "Sample_ID_Liq" in my_input_c:
        myLiquids1['Sample_ID_Liq'] = my_input_c['Sample_ID_Liq']
    else:
        myLiquids1['Sample_ID_Liq'] = my_input.index

    if "Sample_ID_Plag" in my_input_c:
        myPlags1['Sample_ID_Plag'] = my_input_c['Sample_ID_Plag']
    else:
        myPlags1['Sample_ID_Plag'] = my_input.index

    if "Sample_ID_Amp" in my_input_c:
        myAmphs1['Sample_ID_Amp'] = my_input_c['Sample_ID_Amp']
    else:
        myAmphs1['Sample_ID_Amp'] = my_input.index

    if "Sample_ID_Gt" in my_input_c:
        myGts1['Sample_ID_Gt'] = my_input_c['Sample_ID_Gt']
    else:
        myGts1['Sample_ID_Gt'] = my_input.index

    if "Sample_ID_Ol" in my_input_c:
        myOls1['Sample_ID_Ol'] = my_input_c['Sample_ID_Ol']
    else:
        myOls1['Sample_ID_Ol'] = my_input.index


    if "Sample_ID_Kspar" in my_input_c:
        myKspars1['Sample_ID_Kspar'] = my_input_c['Sample_ID_Kspar']
    else:
        myKspars1['Sample_ID_Kspar'] = my_input.index

    if "Sample_ID_Sp" in my_input_c:
        mySps1['Sample_ID_Sp'] = my_input_c['Sample_ID_Sp']
    else:
        mySps1['Sample_ID_Sp'] = my_input.index


    # if "P_kbar" in my_input:
    #     myAmphs1['P_kbar'] = my_input['P_kbar']
    #     myPlags1['P_kbar'] = my_input['P_kbar']
    #     myOls1['P_kbar'] = my_input['P_kbar']
    #     myCPXs1['P_kbar'] = my_input['P_kbar']
    #     myOPXs1['P_kbar'] = my_input['P_kbar']
    #     mySps1['P_kbar'] = my_input['P_kbar']
    #     myLiquids1['P_kbar'] = my_input['P_kbar']
    #
    # if "T_K" in my_input:
    #     myAmphs1['T_K'] = my_input['T_K']
    #     myPlags1['T_K'] = my_input['T_K']
    #     myOls1['T_K'] = my_input['T_K']
    #     myCPXs1['T_K'] = my_input['T_K']
    #     myOPXs1['T_K'] = my_input['T_K']
    #     mySps1['T_K'] = my_input['T_K']
    #     myLiquids1['T_K'] = my_input['T_K']

    return {'my_input': my_input, 'my_oxides': myOxides1, 'Experimental_press_temp': Experimental_press_temp1, 'Cpxs': myCPXs1, 'Opxs': myOPXs1, 'Liqs': myLiquids1, 'Gts': myGts1,
            'Plags': myPlags1, 'Kspars': myKspars1, 'Amps': myAmphs1, 'Ols': myOls1, 'Sps': mySps1}  # , 'y1': y1 ,'y2': y2}


def import_excel_errors(filename, sheet_name, GEOROC=False):
    '''
    Import excel sheet of oxide errors in wt%, headings should be of the form SiO2_Liq_Err (for the melt/liquid), SiO2_Ol_Err (for olivine comps), SiO2_Cpx_Err (for clinopyroxene compositions).


   Parameters
    -------

    filename: pExcel file
                Excel file of oxides in wt% with columns labelled SiO2_Liq, SiO2_Ol, SiO2_Cpx etc.

    filename: str
        specifies the file name (e.g., Python_OlLiq_Thermometers_Test.xlsx)

    Returns
    -------
    pandas DataFrames stored in a dictionary. E.g., Access Cpxs using output.Cpxs
        my_input_Err = pandas dataframe of the entire spreadsheet
        Experimental_press_temp_Err = User-entered PT errors.
        Liqs_Err=pandas dataframe of liquid oxide errors
        Ols_Err=pandas dataframe of olivine oxide errors
        Cpxs_Err=pandas dataframe of cpx oxide  errors
        Plags_Err=pandas dataframe of plagioclase oxide errors
        Kspars_Err=pandas dataframe of kspar oxide errors
        Opxs_Err=pandas dataframe of opx oxide errors
        Amps_Err=pandas dataframe of amphibole oxide errors
        Sps_Err=pandas dataframe of spinel oxide errors
    '''
    if 'csv' in filename:
        my_input = pd.read_csv(filename)
        #my_input[my_input < 0] = 0
    elif 'xls' in filename:
        if sheet_name is not None:
            my_input = pd.read_excel(filename, sheet_name=sheet_name)
            #my_input[my_input < 0] = 0
        else:
            my_input = pd.read_excel(filename)
            #my_input[my_input < 0] = 0

    my_input_c = my_input.copy()

 #   myLabels=my_input.Sample_ID
    my_input.fillna(0)
    Experimental_press_temp1 = my_input.reindex(df_ideal_exp_Err.columns)
# This deals with the fact almost everyone will enter as FeO, but the code uses FeOt for these minerals.
# E.g., if users only enter FeO (not FeOt and Fe2O3), allocates a FeOt
# column. If enter FeO and Fe2O3, put a FeOt column

# Give warnings if no FeOt
    cols=my_input.columns
    if any(cols.str.startswith(" ")):
        w.warn('We have found some spaces  at the start of your column headings. Check these arent oxide headings, it wont read them')
    if ("FeOt_Cpx" not in my_input) and (any(cols.str.contains('_Cpx'))):
        w.warn('No FeOt_Cpx column, please check of you wanted any Fe in calcs')

    if ("FeOt_Opx" not in my_input) and (any(cols.str.contains('_Opx'))):
        w.warn('No FeOt_Opx column, please check of you wanted any Fe in calcs')

    if ("FeOt_Amp" not in my_input) and (any(cols.str.contains('_Amp'))):
        w.warn('No FeOt_Amp column, please check of you wanted any Fe in calcs')

    if ("FeOt_Sp" not in my_input) and (any(cols.str.contains('_Sp'))):
        w.warn('No FeOt_Sp column, please check of you wanted any Fe in calcs')


    if ("FeOt_Ol" not in my_input) and (any(cols.str.contains('_Ol'))):
        w.warn('No FeOt_Ol column, please check of you wanted any Fe in calcs')

    if ("FeOt_Plag" not in my_input) and (any(cols.str.contains('_Plag'))):
        w.warn('No FeOt_Plag column, please check of you wanted any Fe in calcs')

    if ("FeOt_Kspar" not in my_input) and (any(cols.str.contains('_Kspar'))):
        w.warn('No FeOt_Kspar column, please check of you wanted any Fe in calcs')

    if "FeO_Cpx_Err" in my_input and "FeOt_Cpx" not in my_input and "Fe2O3_Cpx" not in my_input:
        my_input_c['FeOt_Cpx_Err'] = my_input_c['FeO_Cpx_Err']
        print('Only FeO_Cpx found in input, the code has allocated this to FeOt_Cpx')
    if "FeO_Cpx_Err" in my_input and "Fe2O3_Cpx_Err" in my_input and 'FeOt_Cpx' not in my_input:
        my_input_c['FeOt_Cpx_Err'] = my_input_c['FeO_Cpx_Err'] + \
            0.8998 * my_input_c['Fe2O3_Cpx_Err']

    if "FeO_Opx_Err" in my_input and "FeOt_Opx" not in my_input and "Fe2O3_Opx" not in my_input:
        my_input_c['FeOt_Opx_Err'] = my_input_c['FeO_Opx_Err']
        print('Only FeO_Opx found in input, the code has allocated this to FeOt_Opx')
    if "FeO_Opx_Err" in my_input and "Fe2O3_Opx_Err" in my_input and 'FeOt_Opx' not in my_input:
        my_input_c['FeOt_Opx_Err'] = my_input_c['FeO_Opx_Err'] + \
            0.8998 * my_input_c['Fe2O3_Opx_Err']

    if "FeO_Plag_Err" in my_input and "FeOt_Plag" not in my_input and "Fe2O3_Plag" not in my_input:
        my_input_c['FeOt_Plag_Err'] = my_input_c['FeO_Plag_Err']
        print('Only FeO_Plag found in input, the code has allocated this to FeOt_Plag')
    if "FeO_Plag_Err" in my_input and "Fe2O3_Plag_Err" in my_input and 'FeOt_Plag' not in my_input:
        my_input_c['FeOt_Plag_Err'] = my_input_c['FeO_Plag_Err'] + \
            0.8998 * my_input_c['Fe2O3_Plag_Err']

    if "FeO_Kspar_Err" in my_input and "FeOt_Kspar" not in my_input and "Fe2O3_Kspar" not in my_input:
        my_input_c['FeOt_Kspar_Err'] = my_input_c['FeO_Kspar_Err']
        print('Only FeO_Kspar found in input, the code has allocated this to FeOt_Kspar')
    if "FeO_Kspar_Err" in my_input and "Fe2O3_Kspar_Err" in my_input and 'FeOt_Kspar' not in my_input:
        my_input_c['FeOt_Kspar_Err'] = my_input_c['FeO_Kspar_Err'] + \
            0.8998 * my_input_c['Fe2O3_Kspar_Err']

    if "FeO_Amp_Err" in my_input and "FeOt_Amp" not in my_input and "Fe2O3_Amp" not in my_input:
        my_input_c['FeOt_Amp_Err'] = my_input_c['FeO_Amp_Err']
        print('Only FeO_Amp found in input, the code has allocated this to FeOt_Amp')
    if "FeO_Amp_Err" in my_input and "Fe2O3_Amp_Err" in my_input and 'FeOt_Amp' not in my_input:
        my_input_c['FeOt_Amp_Err'] = my_input_c['FeO_Amp_Err'] + \
            0.8998 * my_input_c['Fe2O3_Amp_Err']

    if "FeO_Sp_Err" in my_input and "FeOt_Sp" not in my_input and "Fe2O3_Sp" not in my_input:
        my_input_c['FeOt_Sp_Err'] = my_input_c['FeO_Sp_Err']
        print('Only FeO_Sp found in input, the code has allocated this to FeOt_Sp')
    if "FeO_Sp_Err" in my_input and "Fe2O3_Sp_Err" in my_input and 'FeOt_Sp' not in my_input:
        my_input_c['FeOt_Sp_Err'] = my_input_c['FeO_Sp_Err'] + \
            0.8998 * my_input_c['Fe2O3_Sp_Err']

    if "FeO_Ol_Err" in my_input and "FeOt_Ol" not in my_input and "Fe2O3_Ol" not in my_input:
        my_input_c['FeOt_Ol_Err'] = my_input_c['FeO_Ol_Err']
        print('Only FeO_Ol found in input, the code has allocated this to FeOt_Ol')
    if "FeO_Ol_Err" in my_input and "Fe2O3_Ol_Err" in my_input and 'FeOt_Ol' not in my_input:
        my_input_c['FeOt_Ol_Err'] = my_input_c['FeO_Ol_Err'] + \
            0.8998 * my_input_c['Fe2O3_Ol_Err']

    # For liquids, users have an option to specify Fe3Fet_Liq.
    # 1st scenario, users only enter FeOt, code sets Fe3FeT to 0.
    if "FeOt_Err_Liq_Err" in my_input and "Fe2O3_Err_Liq" not in my_input and "Fe3Fet_Err_Liq" not in my_input:
        my_input_c['Fe3Fet_Liq_Err'] = 0
        print('We have set Fe3Fet_Liq_Err to zero, as you only entered FeOt. You can input a Fe3Fet_Liq_Er column to specify this value instead')
    # 2nd scenario, user only enteres FeO. Code sets FeO=FeOt, sets Fe3FeT to 0
    if "FeO_Err_Liq_Err" in my_input and "Fe2O3_Err_Liq" not in my_input and 'FeOt_Liq' not in my_input and "Fe3Fet_Err_Liq" not in my_input:
        my_input_c['FeOt_Liq_Err'] = my_input_c['FeO_Liq_Err']
        my_input_c['Fe3Fet_Liq_Err'] = 0
        print('Only FeO_Ol found in input, the code has allocated this to FeOt_Ol')
    # 3rd scenario, user only enteres Fe2O3. code calculates FeOt
    if "Fe2O3_Err_Liq_Err" in my_input and "FeO_Err_Liq" not in my_input and 'FeOt_Liq' not in my_input and "Fe3Fet_Err_Liq" not in my_input:
        my_input_c['FeOt_Liq_Err'] = 0.8998 * my_input_c['Fe2O3_Liq_Err']
        my_input_c['Fe3Fet_Liq_Err'] = 1
        print('Only Fe2O3_Ol found in input, the code has allocated this to FeOt_Ol, and set Fe3Fet_Liq=1')

    # 4th scenario, users enter only FeO and Fe2O3
    if "Fe2O3_Err_Liq_Err" in my_input and "FeO_Err_Liq_Err" in my_input and 'FeOt_Liq' not in my_input and "Fe3Fet_Err_Liq" not in my_input:
        my_input_c['FeOt_Liq_Err'] = my_input_c['FeO_Liq_Err'] + \
            0.8998 * my_input_c['Fe2O3_Liq_Err']
        my_input_c['Fe3Fet_Liq_Err'] = (0.89998 * my_input_c['Fe2O3_Liq_Err']) / (
            (0.89998 * my_input_c['Fe2O3_Liq_Err'] + my_input_c['FeO_Liq_Err']))
    if "Fe2O3_Err_Liq_Err" in my_input and "FeO_Err_Liq_Err" in my_input and 'FeOt_Liq' in my_input:
        w.warn('You have entered FeO, Fe2O3, AND FeOt. The code uses FeO and Fe2O3 to recalculate FeOt and Fe3FeT ')
        my_input_c['FeOt_Liq_Err'] = my_input_c['FeO_Liq_Err'] + \
            0.8998 * my_input_c['Fe2O3_Liq_Err']
        my_input_c['Fe3Fet_Liq_Err'] = (0.89998 * my_input_c['Fe2O3_Liq_Err']) / (
            (0.89998 * my_input_c['Fe2O3_Liq_Err'] + my_input_c['FeO_Liq_Err']))


    myLiquids1 = my_input.reindex(df_ideal_liq_Err.columns, axis=1).fillna(0)
    myLiquids1 = myLiquids1.apply(pd.to_numeric, errors='coerce').fillna(0)

    myCPXs1 = my_input_c.reindex(df_ideal_cpx_Err.columns, axis=1).fillna(0)
    myCPXs1 = myCPXs1.apply(pd.to_numeric, errors='coerce').fillna(0)

    myOls1 = my_input_c.reindex(df_ideal_ol_Err.columns, axis=1).fillna(0)
    myOls1 = myOls1.apply(pd.to_numeric, errors='coerce').fillna(0)

    myPlags1 = my_input_c.reindex(df_ideal_plag_Err.columns, axis=1).fillna(0)
    myPlags1 = myPlags1.apply(pd.to_numeric, errors='coerce').fillna(0)

    myKspars1 = my_input_c.reindex(
        df_ideal_kspar_Err.columns, axis=1).fillna(0)
    myKspars1 = myKspars1.apply(pd.to_numeric, errors='coerce').fillna(0)

    myOPXs1 = my_input_c.reindex(df_ideal_opx_Err.columns, axis=1).fillna(0)
    myOPXs1 = myOPXs1.apply(pd.to_numeric, errors='coerce').fillna(0)

    mySps1 = my_input_c.reindex(df_ideal_sp_Err.columns, axis=1).fillna(0)
    mySps1 = mySps1.apply(pd.to_numeric, errors='coerce').fillna(0)

    myAmphs1 = my_input_c.reindex(df_ideal_amp_Err.columns, axis=1).fillna(0)
    myAmphs1 = myAmphs1.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Adding sample Names
    if "Sample_ID_Cpx_Err" in my_input:
        myCPXs1['Sample_ID_Cpx_Err'] = my_input['Sample_ID_Cpx_Err']
    else:
        myCPXs1['Sample_ID_Cpx_Err'] = my_input.index

    if "Sample_ID_Opx_Err" in my_input:
        myOPXs1['Sample_ID_Opx_Err'] = my_input['Sample_ID_Opx_Err']
    else:
        myOPXs1['Sample_ID_Opx_Err'] = my_input.index

    if "Sample_ID_Err_Liq_Err" in my_input:
        myLiquids1['Sample_ID_Liq_Err'] = my_input['Sample_ID_Liq_Err']
    else:
        myLiquids1['Sample_ID_Liq_Err'] = my_input.index

    if "Sample_ID_Plag_Err" in my_input:
        myPlags1['Sample_ID_Plag_Err'] = my_input['Sample_ID_Plag_Err']
    else:
        myPlags1['Sample_ID_Plag_Err'] = my_input.index

    if "Sample_ID_Amp_Err" in my_input:
        myAmphs1['Sample_ID_Amp_Err'] = my_input['Sample_ID_Amp_Err']
    else:
        myAmphs1['Sample_ID_Amp_Err'] = my_input.index

    if "P_kbar_Err" in my_input:
        myAmphs1['P_kbar_Err'] = my_input['P_kbar_Err']
        myPlags1['P_kbar_Err'] = my_input['P_kbar_Err']
        myOls1['P_kbar_Err'] = my_input['P_kbar_Err']
        myCPXs1['P_kbar_Err'] = my_input['P_kbar_Err']
        myOPXs1['P_kbar_Err'] = my_input['P_kbar_Err']
        mySps1['P_kbar_Err'] = my_input['P_kbar_Err']
        myLiquids1['P_kbar_Err'] = my_input['P_kbar_Err']

    if "T_K_Err" in my_input:
        myAmphs1['T_K_Err'] = my_input['T_K_Err']
        myPlags1['T_K_Err'] = my_input['T_K_Err']
        myOls1['T_K_Err'] = my_input['T_K_Err']
        myCPXs1['T_K_Err'] = my_input['T_K_Err']
        myOPXs1['T_K_Err'] = my_input['T_K_Err']
        mySps1['T_K_Err'] = my_input['T_K_Err']
        myLiquids1['T_K_Err'] = my_input['T_K_Err']

    return {'my_input_Err': my_input,
    'Experimental_press_temp_Err': Experimental_press_temp1,
    'Cpxs_Err': myCPXs1,
    'Opxs_Err': myOPXs1,
    'Liqs_Err': myLiquids1,
    'Plags_Err': myPlags1,
    'Kspars_Err': myKspars1,
    'Amps_Err': myAmphs1,
    'Ols_Err': myOls1,
    'Sps_Err': mySps1}


# Gets liquid dataframe into a format that can be used in VESical. Have to
# have VESIcal installed for the final step, which we do in the script for
# simplicity
def convert_to_vesical(liq_comps, T1, unit='Kelvin', Fe3Fet_Liq=None):
    ''' Takes liquid dataframe in the format used for PyMME, and strips the _Liq string so that it can be input into VESical. Also removes the Fe3FeTcolumn, and appends temperature (converted from Kevlin to celcius)


   Parameters
    -------
    liq_comps: pandas.DataFrame
        DataFrame of liquid compositions.

    T1: Panda series, int, float
        Temperature in Kelvin by default (e.g., from a thermometer of choice)

    unit: Kelvin or Celcius
        What unit supplied temperature is in

   Returns
    -------
    DataFrame formatted so that it can be inputted into VESIcal.

    '''
    df = liq_comps.copy()
    if unit=='Kelvin':
        df['Temp'] = T1 - 273.15
    else:
        df['Temp'] = T1
    if Fe3Fet_Liq is None:
        Fe3Fet_Liq=df['Fe3Fet_Liq']
    FeOt=df['FeOt_Liq']
    df.drop(['Fe3Fet_Liq', 'FeOt_Liq'], inplace=True, axis=1)
    df.columns = [str(col).replace('_Liq', '') for col in df.columns]

    df['FeO']=FeOt*(1-Fe3Fet_Liq)
    df['Fe2O3']=FeOt*Fe3Fet_Liq*1.111111


    # This bit gets rid of Fe3Fet_Liq
    return df


def convert_from_vesical(data):
    '''
    Takes liquid compositions from VESIcal, and converts it into a panda dataframe that can be inputted into the thermobaromety functions in PyMME

   Parameters
    -------
    data:
    VESIcal data formatted, obtained from myfile.get_Data

   Returns
    -------
    pandas dataframe formatted wtih column headings suitable for PyMME
    '''
    df = data.copy()
    df = df.rename(columns={col: col + '_Liq' for col in df.columns if col in
                            ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'Fe2O3', 'MnO', 'MgO', 'CaO',
                             'K2O', 'Na2O', 'Cr2O3', 'P2O5', 'H2O', 'NiO']})
    df['FeOt_Liq'] = df['FeO_Liq'] + 0.8999998 * df['Fe2O3_Liq']
    df['Fe3Fet_Liq'] = df['Fe2O3_Liq'] * 1.111111 / \
        (df['Fe2O3_Liq'] * 1.111111 + df['FeO_Liq'])
    df['Sample_ID_Liq'] = df.index
    return df
