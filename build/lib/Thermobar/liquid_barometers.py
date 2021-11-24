import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from Thermobar.core import *


# Liquid-only thermometry functions

def T_Put2008_eq13(P=None, *, MgO_Liq):
    '''
    Liquid-only thermometer for olivine-saturated liquids: Equation 13 of Putirka et al. (2008)
    SEE=±72 °C
    '''
    return (26.3 * MgO_Liq + 994.4 + 273.15)