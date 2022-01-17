#!/usr/bin/env python3

import os, sys
core_path_ext = os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'src/Thermobar')
sys.path.append(core_path_ext)
from Thermobar.core import *
from Thermobar.import_export import import_excel
from Thermobar.garnet import *
from Thermobar.garnet_class import *
from Thermobar.geotherm import *
from Thermobar.garnet_plot import *

file = "/home/sinan/src/Thermobar_Sinan/Examples/Garnet/Garnet.xlsx"
data = import_excel(file, sheet_name = "Sheet1")
my_input_gt = data['my_input']
xMg, xCa, xFe, xAl, xCr = calculate_cations_garnet(my_input_gt)

T = T_Ryan1996(my_input_gt)
T2 = T_Sudholz2021(my_input_gt, xCa, xCr)
T3 = T_Canil1999(my_input_gt)
P = P_Ryan_1996(my_input_gt, T, xMg, xCa, xFe, xAl, xCr)





#plot_CA_CR(garnet_comps = my_input_gt, T_Ni = T, P_Cr = P, BDL_T = 1250, SHF_low = 35, SHF_high = 45, SHF_chosen = 35, max_depth = 300, temp_unit = 'Celsius', plot_type = 'show')
plot_garnet_composition_section(gt_comps = my_input_gt, depth_interval = 10, min_section_depth = 50, max_section_depth = 300, T_Ni = T, P_Cr = P, BDL_T = 1250, SHF_low = 35, SHF_high = 45, SHF_chosen = 35, temp_unit = 'Celsius', plot_type = 'show', plot_path = None)
