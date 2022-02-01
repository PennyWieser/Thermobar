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

def Ridolfi21_cali_plot(amp_comps, P_kbar=None, T_K=None, figsize=(7, 5),x=None, y=None,
 shape_cali='o', mfc_cali='white', mec_cali='k', ms_cali=5,
 shape_data='^', mfc_data='red', mec_data='k', ms_data=10,
)
    amp_comps_c=amp_comps.copy()
    if P_kbar is not None:
        amp_comps_c['P_kbar']=P_kbar
    if T_K is not None:
        amp_comps_c['T_K']=T_K

    with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
        Ridolfi_Cali_input=load(f)

    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    ax1.plot(Ridolfi_Cali_input[x], Ridolfi_Cali_input[y], marker=shape_cali,
    mfc=mfc_cali, mec=mec_cali, ms=ms_cali)

    ax1.plot(amp_comps_c[x], amp_comps_c[y], marker=shape_data,
    mfc=mfc_data, mec=mec_data, ms=ms_data)

    return fig
