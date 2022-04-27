import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *



def return_cali_dataset(model=None):
    """
    This function returns the calibration dataset for different models, often with mineral components returned.
    This allows you to make your own plots rather than using the generic_cali_plot() option.


    Parameters
    -------


    model: str
        AMPHIBOLE:
        Ridolfi2021:  Ridolfi et al. (2021)
        Putirka2016:  Putirka (2016)
        Mutch2016: Mutch et al. (2016)
        Zhang2017: Zhang et al. (2017)

        CPX:
        Putirka2008: Putirka (2008) - entire database for Cpx-Liq,
        (not used for all equations).
        Masotta2013: Masotta et al. (2013)
        Neave2017:  Neave and Putirka (2017) for Cpx-Liq
        Brugman2019: Brugman and Till 2019
        Petrelli2020:  Petrelli et al. (2020) for Cpx and Cpx-Liq
        Wang2021: Wang et al. (2021) Cpx-only,but contains Liq compositions too.
        Jorgenson2022: Jorgenson et al. (2022) Cpx-Liq and Cpx-only,

        PLAGIOCLASE
        Waters2015: Waters and Lange (2015) for plag-liq hygrometry
        Masotta2019: Masotta et al. (2019) plag-liq hygrometry
    """

    # Amphibole models
    if model=="Ridolfi2021":
        with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Zhang2017":
        with open(Thermobar_dir/'Zhang17_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)
            Cali_input['F_Amp']=0
            Cali_input['Cl_Amp']=0

    if model=="Putirka2016":
        with open(Thermobar_dir/'Putirka16_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)
            Cali_input['F_Amp']=0
            Cali_input['Cl_Amp']=0


    if model=="Mutch2016":
        with open(Thermobar_dir/'Mutch_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    # Cpx model
    if model=="Wang2021":
        with open(Thermobar_dir/'Wang21_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Petrelli2020":
        with open(Thermobar_dir/'Petrelli20_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Putirka2008":
        with open(Thermobar_dir/'Putirka2008_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Neave2017":
        with open(Thermobar_dir/'NeavePutirka_2017_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Brugman2019":
        with open(Thermobar_dir/'Brugman_2019_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Masotta2013":
        with open(Thermobar_dir/'Masotta_2013_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Jorgenson2022":
        with open(Thermobar_dir/'Jorgenson2022_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    # Plag models

    if model=="Waters2015":
        with open(Thermobar_dir/'Waters_Lange2015_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    return Cali_input




def generic_cali_plot(df, model=None, x=None, y=None, P_kbar=None, T_K=None, figsize=(7, 5),
 shape_cali='o', mfc_cali='white', mec_cali='k', ms_cali=5,
 shape_data='^', mfc_data='red', alpha_cali=1, alpha_data=1, mec_data='k',
 ms_data=10, order="cali bottom", save_fig=False, fig_name=None,  dpi=200):

    """
    This function plots your compositions amongst the calibration dataset for a variety of models where we could
    obtain the exact calibration dataset. see model for option.

    Parameters
    -------

    df: pandas DataFrame
        dataframe of your compositions, e.g. a dataframe of Cpx composition

    x and y: str
        What you want to plotted against each other. E.g. x="SiO2_Cpx", y="Al2O3_Cpx"

    model: str
        AMPHIBOLE:
        Ridolfi2021:  Ridolfi et al. (2012)
        Putirka2016:  Putirka (2016)
        Mutch2016: Mutch et al. (2016)
        Zhang2017: Zhang et al. (2017)


        CPX:
        Putirka2008: Putirka (2008) - entire database for Cpx-Liq,
        (not used for all equations).
        Masotta2013: Masotta et al. (2013)
        Neave2017:  Neave and Putirka (2017) for Cpx-Liq
        Brugman2019: Brugman and Till 2019
        Petrelli2020:  Petrelli et al. (2020) for Cpx and Cpx-Liq
        Wang2021: Wang et al. (2021) Cpx-only,but contains Liq compositions too.
        Jorgenson2022: Jorgenson et al. (2022) Cpx-Liq and Cpx-only,

        PLAGIOCLASE
        Waters2015: Waters and Lange (2015) for plag-liq hygrometry
        Masotta2019: Masotta et al. (2019) plag-liq hygrometry

    P_kbar and T_K: pd.Series
        if you want to plot calculated pressure and temperature against the calibration range,
        specify those panda series here, and it will append them onto your dataframe

    Plot customization options.
    -------


    figsize: (x, y)
        figure size

    order: 'cali top' or 'cali bottom':
        whether cali or user data is plotted ontop. default, cali data bottom

    shape_cali, shape_data:
        matplotlib symbol, e.g. shape_cali='o' and shape_data='^' are the defaults

    ms_cali, ms_data:
        marker size for cali and user data

    mfc_cali, mfc_data:
        marker face color for cali and data (can be different)

    mec_cali, mfc_data:
        marker edge color for cali and data.

    alpha_cali, alpha_data:
        transparency of symbols

    save_fig: bool
        if True, saves figure
        Uses fig_name to give filename,  dpi to give dpi


    """

    #df_c=df.copy()
    if model=="Ridolfi2021" or model=="Putirka2016" or model=="Mutch2016" or model=="Zhang2017":
        Amp_sites=calculate_sites_ridolfi(amp_comps=df)
        df_c=pd.concat([df, Amp_sites], axis=1)


    if model=="Petrelli2020" or model=="Wang2021" or model=="Putirka2008" or model=="Neave2017" or model=="Brugman2019" or model=="Masotta2013" or model=="Jorgenson2022":
        if "Jd" not in df:
            Cpx_sites=calculate_clinopyroxene_components(cpx_comps=df)
            df_c=Cpx_sites
        else:
            df_c=df

    if P_kbar is not None:
        df_c['P_kbar']=P_kbar
    if T_K is not None:
        df_c['T_K']=T_K

    # Amphibole models
    if model=="Ridolfi2021":
        with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Putirka2016":
        with open(Thermobar_dir/'Putirka16_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Mutch2016":
        with open(Thermobar_dir/'Mutch_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Zhang2017":
        with open(Thermobar_dir/'Zhang17_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)
            Cali_input['F_Amp']=0

    # Cpx model
    if model=="Wang2021":
        with open(Thermobar_dir/'Wang21_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Petrelli2020":
        with open(Thermobar_dir/'Petrelli20_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Putirka2008":
        with open(Thermobar_dir/'Putirka2008_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Neave2017":
        with open(Thermobar_dir/'NeavePutirka_2017_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Brugman2019":
        with open(Thermobar_dir/'Brugman_2019_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Masotta2013":
        with open(Thermobar_dir/'Masotta_2013_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)

    if model=="Jorgenson2022":
        with open(Thermobar_dir/'Jorgenson2022_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)
            Cali_input['Cl_Amp']=0

    # Plag models

    if model=="Waters2015":
        with open(Thermobar_dir/'Waters_Lange2015_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)



    if x not in df_c:
        print(df_c.columns)
        raise TypeError('x variable no present in input dataframe. Choose one of the columns printed above instead')
    if y not in df_c:
        print(df_c.columns)
        raise TypeError('y variable no present in input dataframe')
    if x not in Cali_input:
        print(Cali_input.columns)
        raise TypeError('x variable no present in calibration dataframe')
    if y not in Cali_input:
        print(Cali_input.columns)
        raise TypeError('y variable no present in calibration dataframe')




    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)

    if order=="cali top":
        zorder_cali=0
        zorder_data=5
    if order=="cali bottom":
        zorder_cali=5
        zorder_data=0
    ax1.plot(df_c[x], df_c[y], shape_data,
    mfc=mfc_data, mec=mec_data, ms=ms_data, alpha=alpha_data, label='User Data', zorder=zorder_cali)

    ax1.plot(Cali_input[x], Cali_input[y], shape_cali,
    mfc=mfc_cali, mec=mec_cali, ms=ms_cali, alpha=alpha_cali, label=model, zorder=zorder_data)

    xlabel=x.replace('_', ' ')
    ylabel=y.replace('_', ' ')
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if save_fig is True:
        fig.savefig(fig_name, dpi=200)

    return fig


