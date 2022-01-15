import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as PATH
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
import pandas as pd
import os,sys
import itertools
from Thermobar.core import *
from Thermobar.geotherm import *
from Thermobar.garnet import *
from Thermobar.garnet_class import *

def plot_CA_CR(garnet_comps, T_Ni, P_Cr, BDL_T, SHF_low, SHF_high, max_depth, SHF_chosen = None, temp_unit = 'Celsius', plot_type = 'show', plot_path = None):

	geotherms = []
	pressures = []

	for i in range(SHF_low, SHF_high):

		T, depth, p = calculate_hasterok2011_geotherm(SHF = i, BDL_T = BDL_T+273, T_0 = 0, max_depth = max_depth, moho = 38, kinked = True)
		if temp_unit == 'Celsius':
			geotherms.append(T-273.0)
		pressures.append(p)


	geotherms = np.array(geotherms)
	pressures = np.array(pressures)

	d_g_p = [35,37.5,40,42.5,45,47.5,50,52.5,55,57.5,60]

	if temp_unit == 'Celsius':
		T_Ni = T_Ni - 273.0
		d_g_t = [600,700,800,900,1000,1100,1200,1300,1400,1500,1600]
	else:
		d_g_t = np.array([600,700,800,900,1000,1100,1200,1300,1400,1500,1600]) + 273


	ca_cr_class = garnet_ca_cr_class_Griffin2002(garnet_comps)

	lherz_p = []
	lherz_t = []
	lowcaharz_p = []
	lowcaharz_t = []
	harz_p = []
	harz_t = []
	low_cr_gt_p = []
	low_cr_gt_t = []
	wehrlite_p = []
	wehrlite_t = []
	unclass_p = []
	unclass_t = []

	for i in range(0,len(T_Ni)):
		if ca_cr_class[i] == 'Lherzolite':
			lherz_p.append(float(P_Cr[i]))
			lherz_t.append(T_Ni[i])
		elif ca_cr_class[i] == 'Low-Ca Harzburgite':
			lowcaharz_p.append(float(P_Cr[i]))
			lowcaharz_t.append(T_Ni[i])
		elif ca_cr_class[i] == 'Ca-Harzburgite':
			harz_p.append(float(P_Cr[i]))
			harz_t.append(T_Ni[i])
		elif ca_cr_class[i] == 'Low-Cr Peridotite':
			low_cr_gt_p.append(float(P_Cr[i]))
			low_cr_gt_t.append(T_Ni[i])
		elif ca_cr_class[i] == 'Wehrlite':
			wehrlite_p.append(float(P_Cr[i]))
			wehrlite_t.append(T_Ni[i])
		else:
			unclass_p.append(float(P_Cr[i]))
			unclass_t.append(T_Ni[i])

	fig3 = plt.figure(figsize = (5,7))
	ax1 = plt.subplot(211)


	ax1.plot(d_g_t,np.array(d_g_p)/10.0,'-',color = '#a832a4')

	for i in range(0,len(geotherms)):
		ax1.plot(geotherms[i],pressures[i],'--', linewidth = 0.75)


	if len(lherz_t) != 0:
		ax1.plot(lherz_t,lherz_p,'*',color = '#339e51', label = 'Lherzolite',markersize = 1.25)
	if len(lowcaharz_t) != 0:
		ax1.plot(lowcaharz_t,lowcaharz_p,'s',color = '#8b1f8f', label = 'Low-Ca Harzburgite',markersize = 1.25)
	if len(harz_t) != 0:
		ax1.plot(harz_t,harz_p,'s',color = '#ef2fa1', label = 'Harzburgite',markersize = 1.25)
	if len(low_cr_gt_t) != 0:
		ax1.plot(low_cr_gt_t,low_cr_gt_p,'o',color = '#d9c1cb', label = 'Low-Cr Peridotite',markersize = 1.25)
	if len(wehrlite_t) != 0:
		ax1.plot(wehrlite_t,wehrlite_p,'^',color = '#b06f90', label = 'Wehrlite',markersize = 1.25)
	if len(unclass_t) != 0:
		ax1.plot(unclass_t,unclass_p,'x',color = 'k',alpha = 0.25, label = 'Unclassified',markersize = 1.25)

	ax1.axvline(BDL_T,linestyle = '--',color = 'k',linewidth = 2)

	if SHF_chosen != None:
		T_chosen, depth_chosen, p_chosen = calculate_hasterok2011_geotherm(SHF = SHF_chosen, BDL_T = BDL_T+273, T_0 = 0, max_depth = max_depth, moho = 38, kinked = True)
		ax1.plot(T_chosen-273.0,p_chosen,'--', color = 'k', linewidth = 2)

	ax1.set_xlim((600,1800))
	ax1.set_ylim((9,0))
	ax1.legend(fontsize = 7,framealpha = 1)
	ax1.set_ylabel('P (GPa)')

	ax2 = plt.subplot(212)

	lherz_t = []
	lherz_y = []
	lowcaharz_t = []
	lowcaharz_y = []
	harz_t = []
	harz_y = []
	low_cr_gt_t = []
	low_cr_gt_y = []
	wehrlite_t = []
	wehrlite_y = []
	unclass_t = []
	unclass_y = []

	for i in range(0,len(T_Ni)):
		if ca_cr_class[i] == 'Lherzolite':
			lherz_t.append(float(T_Ni[i]))
			lherz_y.append(garnet_comps['Y_Gt'][i])
		elif ca_cr_class[i] == 'Low-Ca Harzburgite':
			lowcaharz_t.append(float(T_Ni[i]))
			lowcaharz_y.append(garnet_comps['Y_Gt'][i])
		elif ca_cr_class[i] == 'Ca-Harzburgite':
			harz_t.append(float(T_Ni[i]))
			harz_y.append(garnet_comps['Y_Gt'][i])
		elif ca_cr_class[i] == 'Low-Cr Peridotite':
			low_cr_gt_t.append(float(T_Ni[i]))
			low_cr_gt_y.append(garnet_comps['Y_Gt'][i])
		elif ca_cr_class[i] == 'Wehrlite':
			wehrlite_t.append(float(T_Ni[i]))
			wehrlite_y.append(garnet_comps['Y_Gt'][i])
		else:
			unclass_t.append(float(T_Ni[i]))
			unclass_y.append(garnet_comps['Y_Gt'][i])

	if len(lherz_t) != 0:
		ax2.plot(lherz_t,lherz_y,'*',color = '#339e51', label = 'Lherzolite',markersize = 3)
	if len(lowcaharz_t) != 0:
		ax2.plot(lowcaharz_t,lowcaharz_y,'s',color = '#8b1f8f', label = 'Low-Ca Harzburgite',markersize = 3)
	if len(harz_t) != 0:
		ax2.plot(harz_t,harz_y,'s',color = '#ef2fa1', label = 'Harzburgite',markersize = 3)
	if len(low_cr_gt_t) != 0:
		ax2.plot(low_cr_gt_t,low_cr_gt_y,'s',color = '#d9c1cb', label = 'Low-Cr Peridotite',markersize = 3)
	if len(wehrlite_t) != 0:
		ax2.plot(wehrlite_t,wehrlite_y,'^',color = '#b06f90', label = 'Wehrlite',markersize = 3)
	if len(unclass_t) != 0:
		ax2.plot(unclass_t,unclass_y,'x',color = 'k',alpha = 0.25, label = 'Unclassified',markersize = 3)

	ax2.set_ylim(1e-1,1e2)
	ax2.set_xlim(600,1800)
	if temp_unit == 'Celsius':
		ax2.set_xlabel(r'$T_{Ni} \, C^{\circ}$')
	else:
		ax2.set_xlabel(r'$T_{Ni} \, K^{\circ}$')
	ax2.set_ylabel('Y (wt ppm)')
	ax2.set_yscale('log')
	ax2.axvline(BDL_T,linestyle = '--',color = 'k',linewidth = 2)

	if plot_type == 'show':
		plt.show()
	elif plot_type = 'save':
		if plot_path == None:
			save_str = os.path.join((os.getcwd(),'CA_CR_PLOT.png'))
			plt.savefig(save_str, dpi = 300.0)
		else:
			plt.savefig(plot_path, dpi = 300.0)

def plot_garnet_composition_section(garnet_comps, depth_interval, min_section_depth, max_section_depth, T_Ni, P_Cr, BDL_T, SHF_low, SHF_high, SHF_chosen = None, temp_unit = 'Celsius', plot_type = 'show', plot_path = None):

	geotherms = []
	pressures = []

	for i in range(SHF_low, SHF_high):

		T, depth, p = calculate_hasterok2011_geotherm(SHF = i, BDL_T = BDL_T+273, T_0 = 0, max_depth = max_depth, moho = 38, kinked = True)
		if temp_unit == 'Celsius':
			geotherms.append(T-273.0)
		pressures.append(p)


	geotherms = np.array(geotherms)
	pressures = np.array(pressures)

	d_g_p = [35,37.5,40,42.5,45,47.5,50,52.5,55,57.5,60]

	if temp_unit == 'Celsius':
		T_Ni = T_Ni - 273.0
		d_g_t = [600,700,800,900,1000,1100,1200,1300,1400,1500,1600]
	else:
		d_g_t = np.array([600,700,800,900,1000,1100,1200,1300,1400,1500,1600]) + 273

	depth_fields = np.arange(min_section_depth, max_section_depth + depth_interval, depth_interval) 

	carp_class = garnet_CARP_class_Griffin2002(garnet_comps)
