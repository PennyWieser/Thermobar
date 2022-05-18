import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *

with open(Thermobar_dir/'Ridolfi_Cali_input.pkl', 'rb') as f:
    Ridolfi_Cali_input=load(f)

def Ridolfi21_cali_dataset(all=True):
    return Ridolfi_Cali_input