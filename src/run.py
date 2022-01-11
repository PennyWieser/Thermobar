#!/usr/bin/env python3

import os, sys
core_path_ext = os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'src/Thermobar')
sys.path.append(core_path_ext)
from Thermobar.core import *
from Thermobar.import_export import import_excel
from Thermobar.garnet import *
from Thermobar.garnet_class import *

file = "/home/sinan/src/Thermobar_Sinan/Examples/Garnet/Garnet.xlsx"
data = import_excel(file, sheet_name = "Sheet1")
my_input_gt = data['my_input']
xMg, xCa, xFe, xAl, xCr = calculate_cations_garnet(my_input_gt)

T = T_Ryan1996(my_input_gt)
T2 = T_Sudholz2021(my_input_gt, xCa, xCr)
T3 = T_Canil1999(my_input_gt)
P = P_Ryan_1996(my_input_gt, T, xMg, xCa, xFe, xAl, xCr)

CARP_CLASS = garnet_CARP_class_Griffin2002(my_input_gt)
GRUTTER_CLASS = garnet_class_Grutter2003(my_input_gt)
CA_CR_CLASS = garnet_ca_cr_class_Griffin2002(my_input_gt)
Y_ZR_CLASS = y_zr_classification_Griffin2002(my_input_gt)
