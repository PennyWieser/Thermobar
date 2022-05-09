#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib.pyplot as plt
core_path_ext = os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'src/Thermobar')
sys.path.append(core_path_ext)
from Thermobar.core import *
from Thermobar.import_export import import_excel
from Thermobar.garnet import *
from Thermobar.garnet_class import *
from Thermobar.geotherm import *
from Thermobar.garnet_plot import *

#Now load the data
file = "../Examples/Garnet/Group1_Kimberley.xlsx" #Wherever that /Examples/Garnet is
data = import_excel(file, sheet_name = "Sheet1")
my_input_gt = data['my_input']

#Ryan1996
T_garnet = T_Ryan1996(my_input_gt) #Calculating T_Ryan1996
P_garnet = P_Ryan_1996(my_input_gt, T_garnet)

# T, depth, p, depth_intercepts = calculate_hasterok2011_geotherm(SHF = i, BDL_T = BDL_T+273, T_0 = 0, max_depth = max_depth, moho = 38, kinked = True)

P_inp = np.arange(1.0,4,0.1)

T_C, T_K = T_Katsura_2022_Adiabat(P_inp)

T, depth, p, depth_intercepts = calculate_hasterok2011_geotherm(SHF = 40, BDL_T = 800+273, T_0 = 0, max_depth = 300, moho = 38, kinked = False, adiabat = True)


invert_generalised_mantle_geotherm(P_sample = P_garnet, T_sample = T_garnet, std_P = 0.2, std_T = 50,
 SHF_start = 35, SHF_end=45, SHF_increment=0.1, max_depth=300, kinked=False, BDL_T = 170, adiabat = True,
 plot_solution = True)
