import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *



def return_Ridolfi21_cali_dataset(all=True):
    with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
        Ridolfi_Cali_input=load(f)
    return Ridolfi_Cali_input

def return_Petrelli2020_cali_dataset(all=True):
    with open(Thermobar_dir/'Petrelli20_Cali_input.pkl', 'rb') as f:
        Petrelli20_Cali_input=load(f)
    return Petrelli20_Cali_input

def generic_cali_plot(df, model=None, P_kbar=None, T_K=None, figsize=(7, 5),x=None, y=None,
 shape_cali='o', mfc_cali='white', mec_cali='k', ms_cali=5,
 shape_data='^', mfc_data='red', alpha_cali=1, alpha_data=1, mec_data='k', ms_data=10, order="cali top"):
    df_c=df.copy()
    if P_kbar is not None:
        df_c['P_kbar']=P_kbar
    if T_K is not None:
        df_c['T_K']=T_K
    if model=="Ridolfi21":
        with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
            Cali_input=load(f)
    if model=="Petrelli20":
        with open(Thermobar_dir/'Petrelli20_Cali_input.pkl', 'rb') as f:
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
    mfc=mfc_data, mec=mec_data, ms=ms_data, alpha=alpha_data, label='Data', zorder=zorder_cali)

    ax1.plot(Cali_input[x], Cali_input[y], shape_cali,
    mfc=mfc_cali, mec=mec_cali, ms=ms_cali, alpha=alpha_cali, label='Cali', zorder=zorder_data)

    xlabel=x.replace('_', ' ')
    ylabel=y.replace('_', ' ')
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    return fig

def Ridolfi21_cali_plot(amp_comps, P_kbar=None, T_K=None, figsize=(7, 5),x=None, y=None,
 shape_cali='o', mfc_cali='white', mec_cali='k', ms_cali=5,
 shape_data='^', mfc_data='red', mec_data='k', ms_data=10):
    amp_comps_c=amp_comps.copy()
    if P_kbar is not None:
        amp_comps_c['P_kbar']=P_kbar
    if T_K is not None:
        amp_comps_c['T_K']=T_K

    with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
        Ridolfi_Cali_input=load(f)

    if x not in amp_comps_c:
        print(amp_comps_c.columns)
        raise TypeError('x variable no present in input dataframe. Choose one of the columns printed above instead')
    if y not in amp_comps_c:
        print(amp_comps_c.columns)
        raise TypeError('y variable no present in input dataframe')
    if x not in Ridolfi_Cali_input:
        print(Ridolfi_Cali_input.columns)
        raise TypeError('x variable no present in calibration dataframe')
    if y not in Ridolfi_Cali_input:
        print(Ridolfi_Cali_input.columns)
        raise TypeError('y variable no present in calibration dataframe')




    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)


    ax1.plot(amp_comps_c[x], amp_comps_c[y], shape_data,
    mfc=mfc_data, mec=mec_data, ms=ms_data, label='Data')

    ax1.plot(Ridolfi_Cali_input[x], Ridolfi_Cali_input[y], shape_cali,
    mfc=mfc_cali, mec=mec_cali, ms=ms_cali, label='Cali')

    xlabel=x.replace('_', ' ')
    ylabel=y.replace('_', ' ')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()

    return fig
