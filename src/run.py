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

file = "../Examples/Garnet/Group1_Kimberley.xlsx"
data = import_excel(file, sheet_name = "Sheet1")
my_input_gt = data['my_input']
xMg, xCa, xFe, xAl, xCr = calculate_fractions_garnet(gt_comps = my_input_gt)

T = T_Ryan1996(my_input_gt)
P = P_Ryan_1996(my_input_gt, T)

fig = plt.figure()
# y = np.arange(0,12,1)
y = np.arange(0,1500,10)
x = y
ax1 = plt.subplot(111)
ax1.plot(x,y,color = 'r',linewidth = 0.5)

file_ext = "../Benchmarking/garnet/Group_1_PT_solution.xlsx"
data_ext = import_excel(file_ext, sheet_name = "Group_1_PT_solution")
input_gt_ext = data_ext['my_input']
p_ext = np.array(input_gt_ext['P_Kb']) / 10.0
t_ext = np.array(input_gt_ext['T_C'] + 273.0)

# ax1.plot(p_ext,P,'o')
ax1.plot(t_ext,T, 'o')
ax1.set_xlabel('T taken from Ozaydin2021')
ax1.set_ylabel('T calculated with Thermobar')
plt.show()

fig = plt.figure()
y = np.arange(0,12,1)
x = y
ax1 = plt.subplot(111)
ax1.plot(x,y,color = 'r', linewidth = 0.5)

ax1.plot(p_ext, P, 'o')
ax1.set_xlabel('P taken from Ozaydin2021')
ax1.set_ylabel('P calculated with Thermobar')
plt.show()


file_sudholz = "../Benchmarking/garnet/PT_Sudholz.xlsx"
data_sudholz = import_excel(file_sudholz, sheet_name = "PT_Sudholz")
input_sudholz_ext = data_sudholz['my_input']
t_ext_sudholz = np.array(input_sudholz_ext['T_Sudholz'])
t_ext_canil = np.array(input_sudholz_ext['T_Canil'])

T2 = T_Sudholz2021(input_sudholz_ext)
T3 = T_Canil1999(input_sudholz_ext)

fig = plt.figure()
# y = np.arange(0,12,1)
y = np.arange(0,1500,10)
x = y
ax1 = plt.subplot(111)
ax1.plot(x,y,color = 'r',linewidth = 0.5)
ax1.plot(t_ext_sudholz+273,T2, 'o')
ax1.set_xlabel('T taken from Sudholz2021')
ax1.set_ylabel('T calculated with Thermobar')
plt.show()

fig_2 = plt.figure()
ax2 = plt.subplot(111)
ax2.plot(x,y,color = 'r',linewidth = 0.5)
ax2.plot(t_ext_canil+273,T3, 'o')
ax2.set_xlabel('T taken from Sudholz2021')
ax2.set_ylabel('T calculated with Thermobar')
plt.show()

"""
plot_CA_CR(gt_comps = my_input_gt, T_Ni = T, P_Cr = P, BDL_T = 1125,
 SHF_low = 35, SHF_high = 45, SHF_chosen = 37, max_depth = 300,
  temp_unit = 'Celsius', plot_type = 'show')

plot_garnet_composition_section(gt_comps = my_input_gt, depth_interval = 10,
min_section_depth = 50, max_section_depth = 300, T_Ni = T, P_Cr = P, BDL_T = 1125,
 SHF_low = 35, SHF_high = 45, SHF_chosen = 37, temp_unit = 'Celsius',
 plot_type = 'save', filename_save = 'CARP_plot_2')
"""
