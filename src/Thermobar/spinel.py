import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
import itertools
from Thermobar.core import *

def T_Ryan1996(sp_comps):

    '''
    Zn-in-cr-spinel thermometer of Ryan et al. (1996)
    :cite:`ryan1996`
    SEE=+-50C
    '''

    temp = 1000 / (-0.9 + (0.6 * np.log(np.array(sp_comps['Sp_Zn']))))

    return temp

#--------------------Function for solving for temperature for garnets-----------------------------------------------------#
Sp_T_funcs = {T_Ryan1996} # put on outside

Sp_T_funcs_by_name = {p.__name__: p for p in Sp_T_funcs}

def calculate_sp_temp(*, sp_comps=None, equationT=None):

    '''

    Parameters
    ------------

    sp_comps: pandas.DataFrame
        sp compositions with column headings SiO2_Sp, MgO_Sp etc.

    equationT: str
        Choose from:

        |  T_Ryan1996

    out_format: str
        Choose from:

        |  'Array' - A numpy array output
        |  'DataFrame' - Pandas dataframe output

    Returns
    -----------
        pandas.Series: Temperature in K
    '''

    try:
        func = Sp_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if sp_comps is not None:
        if sp_comps['Zn_Sp'] is not None:

            T_K = func(sp_comps)

    else:
        raise ValueError(f'{equationT} requires you to enter sp_comps that involves Zn_Sp [ppm]')

    T_K_series= pd.Series(T_K)

    return T_K_series
