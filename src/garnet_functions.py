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

#Now load the data from Ozaydin et al. (2021)
file = "../Examples/Garnet/Group1_Kimberley.xlsx" #Wherever that /Examples/Garnet is
data = import_excel(file, sheet_name = "Sheet1")
my_input_gt = data['my_input']

#Calculating T from Ryan_1996
T = calculate_gt_temp(gt_comps = my_input_gt, equationT = 'T_Ryan1996') #Calculating T_Ryan1996

#Creating fig object
fig = plt.figure(figsize = (12,6))
# y = np.arange(0,12,1)
y = np.arange(0,2500,10)
x = y
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
#1:1 line
ax1.plot(x,y,color = 'r',linewidth = 1)

#Loading external file including the solution from Ozaydin et al. (2021)
file_ext = "../Benchmarking/garnet/Group_1_PT_solution.xlsx"
data_ext = import_excel(file_ext, sheet_name = "Group_1_PT_solution")
input_gt_ext = data_ext['my_input']
p_ext = np.array(input_gt_ext['P_Kb'])
t_ext = np.array(input_gt_ext['T_C'] + 273.0)

#plotting calculated vs read T_Ryan1996
ax1.plot(t_ext,T, 'o')
#Creating histogram of differences
counts, edges, patches = ax2.hist(t_ext - T,
bins = np.linspace(np.amin(t_ext - T),np.amax(t_ext - T), 10),color = '#2d8e5f')
centers = 0.5*(edges[:-1] + edges[1:])
ax1.set_xlabel('T_Ryan1996 taken from Ozaydin2021')
ax1.set_ylabel('T_Ryan1996 calculated with Thermobar')
ax2.set_xlabel('Difference (T_Ryan1996_Ozaydin2021 - T_Thermobar)')
ax2.set_ylabel('Count')
plt.show()


#####
#Calculating P_Ryan_1996
P = calculate_gt_press(gt_comps = my_input_gt, equationP = 'P_Ryan1996', T = T)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.plot([0, 100], [0, 100],color = 'r',linewidth = 1)

#plotting calculated vs read T_Ryan1996
ax1.plot(p_ext,P, 'o')

ax2.plot(p_ext, p_ext-P, 'ok')
#ax2.set_yticks([-0.05, 0, 0.05])

ax1.set_ylabel('P Thermobar')
ax1.set_xlabel('P Published')


ax2.set_ylabel('P Published - P Thermobar')
ax2.set_xlabel('P Published')
fig.tight_layout()

plt.show()

#Loading data from Sudholz et al. (2021) for benchmarking tests of
#Sudholz2021 and Canil1999. This file consists both the solution and composition data
file_sudholz = "../Benchmarking/garnet/PT_Sudholz.xlsx"
data_sudholz = import_excel(file_sudholz, sheet_name = "PT_Sudholz")
input_sudholz_ext = data_sudholz['my_input']

fig = plt.figure(figsize = (12,6))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

T2 = calculate_gt_temp(gt_comps = input_sudholz_ext, equationT = 'T_Sudholz2021')
t_ext_sudholz = np.array(input_sudholz_ext['T_Sudholz'])

y = np.arange(0,1800,10)
x = y
ax1.plot(x,y,color = 'r',linewidth = 1)
ax1.plot(t_ext_sudholz+273,T2, 'o')
counts, edges, patches = ax2.hist((t_ext_sudholz + 273) - T2,
bins = np.linspace(np.amin((t_ext_sudholz + 273) - T2),np.amax((t_ext_sudholz + 273) - T2), 10),color = '#2d8e5f')
centers = 0.5*(edges[:-1] + edges[1:])
ax1.set_xlabel('T_Sudholz2021 taken from Sudholz2021')
ax1.set_ylabel('T_Sudholz2021 calculated with Thermobar')
ax2.set_xlabel('Difference (T_Sudholz2021 - T_Thermobar)')
ax2.set_ylabel('Count')
plt.show()

#T_Canil1999
t_ext_canil = np.array(input_sudholz_ext['T_Canil'])
T3 = calculate_gt_temp(gt_comps = input_sudholz_ext, equationT = 'T_Canil1999')

fig = plt.figure(figsize = (12,6))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(x,y,color = 'r',linewidth = 0.5)
ax1.plot(t_ext_canil+273,T3, 'o')
counts, edges, patches = ax2.hist((t_ext_canil + 273) - T3,
bins = np.linspace(np.amin((t_ext_canil + 273) - T3),np.amax((t_ext_canil + 273) - T3), 10),color = '#2d8e5f')
centers = 0.5*(edges[:-1] + edges[1:])
ax1.set_xlabel('T_Canil1999 taken from Sudholz2021')
ax1.set_ylabel('T_Canil1999 calculated with Thermobar')
ax2.set_xlabel('Difference (T_Canil1999 - T_Thermobar)')
ax2.set_ylabel('Count')
plt.show()

#These are not benchmarking tests but an example for plotting functions
plot_garnet_geotherm(gt_comps = my_input_gt, T_Ni = T, P_Cr = P, BDL_T = 1125,
 SHF_low = 35, SHF_high = 45, SHF_chosen = 37, max_depth = 300,
  temp_unit = 'Celsius', plot_type = 'show')

plot_garnet_composition_section(gt_comps = my_input_gt, depth_interval = 10,
min_section_depth = 50, max_section_depth = 300, T_Ni = T, P_Cr = P, BDL_T = 1125,
 SHF_low = 35, SHF_high = 45, SHF_chosen = 37, temp_unit = 'Celsius',
 plot_type = 'show', filename_save = 'CARP_plot_2')
