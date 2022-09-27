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

def plot_garnet_geotherm(gt_comps, T_Ni, P_Cr, BDL_T, SHF_low, SHF_high, max_depth, SHF_chosen = None, temp_unit = 'Celsius', plot_type = 'show', filename_save = None):

	'''
	A function to plot the CaO CR2O3 based garnet xenocryst classifications
	and to determine the base of the depleted lithosphere BDL_T.

	###Parameters###
	gt_comps: garnet composition dataframe imported from core function

	BDL_T: is Temperature at the base of the depleted lithosphere in Celsius

	T_Ni: Temperature output taken from one of the garnet thermometry methods in
	Kelvin.

	P_Cr: Cr2O3 based pressure estimate of the garnet xenocrysts in KBar

	SHF_low: Surface heat flow value for the lower end of the geotherms to be plotted in mW/m^2

	SHF_high: Surface heat flow value for the higher end of the geotherms to be plotted in mW/m^2

	max_depth: maximum depth for calculations and plotting

	SHF_chosen: Surface heat flow value for the chosen geotherm to be plotted in mW/m^2

	temp_unit: temperature unit 'Celsius' or 'Kelvin'

	plot_type: parameter to decide whether to 'show' the plot or 'save' it as a png file.

	filename_save: str parameter to determine the filename to be saved. If none entered it
	defaults to a certain name.

	'''

	geotherms = []
	pressures = []

	for i in range(SHF_low, SHF_high):

		T, depth, p, depth_intercepts = calculate_hasterok2011_geotherm(SHF = i, BDL_T = BDL_T + 273, T_0 = 0, max_depth = max_depth, moho = 38, kinked = True, adiabat = False)
		if temp_unit == 'Celsius':
			geotherms.append(T-273.0)
		pressures.append(p)


	geotherms = np.array(geotherms)
	pressures = np.array(pressures)

	P_Cr = P_Cr / 10.0

	d_g_p = [35,37.5,40,42.5,45,47.5,50,52.5,55,57.5,60]

	if temp_unit == 'Celsius':
		T_Ni = T_Ni - 273.0
		d_g_t = [600,700,800,900,1000,1100,1200,1300,1400,1500,1600]
	else:
		d_g_t = np.array([600,700,800,900,1000,1100,1200,1300,1400,1500,1600]) + 273

	#Checking if Y_Gt exist at all

	try:
		gt_comps['Y_Gt']
		trace_exist = True
	except KeyError:
		trace_exist = False

	if trace_exist == True:

		for i in range(0,len(gt_comps)):

			if pd.isna(gt_comps['Y_Gt'][i]) == True:

				print('WARNING')
				print('There is Y_Gt missing in one of the samples. idx = ' + str(i))

	ca_cr_class = garnet_ca_cr_class_Griffin2002(gt_comps)

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
	if trace_exist == True:
		ax1 = plt.subplot(211)
	elif trace_exist == False:
		ax1 = plt.subplot(111)

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
		T_chosen, depth_chosen, p_chosen, depth_intercept = calculate_hasterok2011_geotherm(SHF = SHF_chosen, BDL_T = BDL_T+273, T_0 = 0, max_depth = max_depth, moho = 38, kinked = True, adiabat = False)
		ax1.plot(T_chosen-273.0,p_chosen,'--', color = 'k', linewidth = 2)

	ax1.set_xlim((600,1800))
	ax1.set_ylim((9,0))
	ax1.legend(fontsize = 7,framealpha = 1)
	ax1.set_ylabel('P (GPa)')

	if trace_exist == True:
		ax2 = plt.subplot(212)

	if trace_exist == True:
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
				lherz_y.append(gt_comps['Y_Gt'][i])
			elif ca_cr_class[i] == 'Low-Ca Harzburgite':
				lowcaharz_t.append(float(T_Ni[i]))
				lowcaharz_y.append(gt_comps['Y_Gt'][i])
			elif ca_cr_class[i] == 'Ca-Harzburgite':
				harz_t.append(float(T_Ni[i]))
				harz_y.append(gt_comps['Y_Gt'][i])
			elif ca_cr_class[i] == 'Low-Cr Peridotite':
				low_cr_gt_t.append(float(T_Ni[i]))
				low_cr_gt_y.append(gt_comps['Y_Gt'][i])
			elif ca_cr_class[i] == 'Wehrlite':
				wehrlite_t.append(float(T_Ni[i]))
				wehrlite_y.append(gt_comps['Y_Gt'][i])
			else:
				unclass_t.append(float(T_Ni[i]))
				unclass_y.append(gt_comps['Y_Gt'][i])

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
	elif plot_type == 'save':
		if plot_path == None:
			save_str = os.path.join((os.getcwd(),'CA_CR_PLOT.png'))
			plt.savefig(save_str, dpi = 300.0)
		else:
			try:
				plt.savefig(filename_save, dpi = 300.0,bbox_inches = "tight")
			except ValueError:
				print('Not a proper format for the filename_save')

def plot_garnet_composition_section(gt_comps, depth_interval, min_section_depth, max_section_depth, T_Ni, P_Cr, BDL_T, SHF_low, SHF_high, SHF_chosen = None, temp_unit = 'Celsius', plot_type = 'show', filename_save = None):

	'''
	A function to plot a collection of the garnet xenocryst based classifications

	###Parameters###
	gt_comps: garnet composition dataframe imported from core function

	depth_interval: depth interval for the histogram sections in km

	min_section_depth: depth for the minimum depth to be plotted in km

	max_section_depth: depth for the maximum depth to be plotted in km

	T_Ni: Temperature output taken from one of the garnet thermometry methods in
	Kelvin.

	P_Cr: Cr2O3 based pressure estimate of the garnet xenocrysts in KBar

	BDL_T: is Temperature at the base of the depleted lithosphere in Celsius

	SHF_low: Surface heat flow value for the lower end of the geotherms to be plotted in mW/m^2

	SHF_high: Surface heat flow value for the higher end of the geotherms to be plotted in mW/m^2

	SHF_chosen: Surface heat flow value for the chosen geotherm to be plotted in mW/m^2

	temp_unit: temperature unit 'Celsius' or 'Kelvin'

	max_depth: maximum depth for geotherm calculations in km

	plot_type: parameter to decide whether to 'show' the plot or 'save' it as a png file.

	filename_save: str parameter to determine the filename to be saved. If none entered it
	defaults to a certain name.

	'''

	P_Cr = P_Cr / 10.0

	#Calculating range of geotherms to be plotted
	geotherms = []
	pressures = []
	depths = []

	for i in range(SHF_low, SHF_high):

		T, depth, p, depth_intercepts = calculate_hasterok2011_geotherm(SHF = i, BDL_T = BDL_T+273, T_0 = 0, max_depth = max_section_depth, moho = 38, kinked = True, adiabat = False)
		if temp_unit == 'Celsius':
			geotherms.append(T-273.0)
		pressures.append(p)
		depths.append(depth)

	geotherms = np.array(geotherms)
	pressures = np.array(pressures)
	depths = np.array(depths)

	d_g_p = [35,37.5,40,42.5,45,47.5,50,52.5,55,57.5,60] #diamond-graphite transition pressure in kBar

	if temp_unit == 'Celsius':
		T_Ni = T_Ni - 273.0
		d_g_t = [600,700,800,900,1000,1100,1200,1300,1400,1500,1600] #diamond graphite transition temp in Celsius
	else:
		d_g_t = np.array([600,700,800,900,1000,1100,1200,1300,1400,1500,1600]) + 273 #diamond graphite transition temp in Kelvin

	depth_fields = np.arange(min_section_depth, max_section_depth + depth_interval, depth_interval) * 1e3 #Creating depth ranges for the plot sections with the
	#interval inputted with min_section_depth, max_section_depth and depth_interval

	gt_calc = calculate_garnet_components(gt_comps = gt_comps)

	xMg = np.array(gt_calc['Mg_MgFeCa_Gt'])
	xCa = np.array(gt_calc['Ca_MgFeCa_Gt'])
	xFe = np.array(gt_calc['Fe_MgFeCa_Gt'])
	xAl = np.array(gt_calc['Al_AlCr_Gt'])
	xCr = np.array(gt_calc['Cr_AlCr_Gt'])

	carp_depleted_harz, carp_depleted_lherz, carp_depleted_metasomatised, carp_fertile_lherz, carp_melt_metas, carp_unclass, len_tot = garnet_CARP_class_Griffin2002(gt_comps = gt_comps) #determining CARP classes from garnet composition
	cacr_class = garnet_ca_cr_class_Griffin2002(gt_comps = gt_comps)
	g_class = garnet_class_Grutter2004(gt_comps = gt_comps)
	MG = calculate_ol_mg(gt_comps = gt_comps, T_Ni = T_Ni) #calculating olivine magnesium number from garnet composition
	AL = calculate_al2o3_whole_rock(gt_comps = gt_comps) #calculating Al2O3 whole rock from garnet compostion

	depth_normal = np.zeros(len(P_Cr))

	#Creating the chosen geotherm for the garnet
	if SHF_chosen != None:
		T_chosen, depth_chosen, p_chosen, depth_intercept = calculate_hasterok2011_geotherm(SHF = SHF_chosen, BDL_T = BDL_T + 273, T_0 = 0, max_depth = max_section_depth, moho = 38, kinked = True, adiabat = False)

		if temp_unit == 'Celsius':
			T_chosen = T_chosen - 273.0

		interp_T = np.arange(np.amin(T_chosen),np.amax(T_chosen),3)
		interp_depth = np.interp(interp_T, T_chosen, depth_chosen)

		BDL_depth = depth_chosen[depth_intercept]

		for i in range(0,len(P_Cr)):
			#for look that downprojects the P_Cr onto the selected
			#geotherö
			idx_fnd = (np.abs(interp_T-T_Ni[i])).argmin()
			depth_normal[i] = interp_depth[idx_fnd]

	def count_arrays(array,depth_array,depth_grid,index_depth):
		#local def to count arrays
		idx = 0

		for index in array:
			if (depth_array[index] < depth_grid[index_depth]) and (depth_array[index] >= depth_grid[index_depth-1]):
				idx = idx + 1

		return idx

	def pct_count(count,tot_num):
		#local def to convert
		try:
			pct = (float(count) / float(tot_num)) * 100.0
		except ZeroDivisionError:
			pct = 0
		return pct

	deplet_harz_pct_list = []
	deplet_lherz_pct_list = []
	deplet_metas_pct_list = []
	fertile_pct_list = []
	melt_pct_metas = []
	l10a_over_fert = []
	unclass_pct = []
	total_list = []
	total_classified = []

	for j in range(1,len(depth_fields)):

		tot_num_depth = 0
		carp_deplet_harz_count = count_arrays(carp_depleted_harz,depth_normal,depth_fields,j)
		carp_deplet_lherz_count = count_arrays(carp_depleted_lherz,depth_normal,depth_fields,j)
		carp_depleted_metasomatised_count = count_arrays(carp_depleted_metasomatised,depth_normal,depth_fields,j)
		carp_fertile_lherz_count = count_arrays(carp_fertile_lherz,depth_normal,depth_fields,j)
		carp_melt_metas_count = count_arrays(carp_melt_metas,depth_normal,depth_fields,j)
		carp_unclass_count = count_arrays(carp_unclass,depth_normal,depth_fields,j)

		for i in range(0,len_tot):
			if (depth_normal[i] < depth_fields[j]) and (depth_normal[i] >= depth_fields[j-1]):
				tot_num_depth = tot_num_depth + 1

		depth_count = carp_melt_metas_count + carp_fertile_lherz_count +\
		 carp_depleted_metasomatised_count + carp_deplet_harz_count + carp_deplet_lherz_count

		carp_deplet_harz_pct = pct_count(carp_deplet_harz_count,tot_num_depth)
		carp_deplet_lherz_pct = pct_count(carp_deplet_lherz_count,tot_num_depth)
		carp_depleted_metasomatised_pct = pct_count(carp_depleted_metasomatised_count,tot_num_depth)
		carp_fertile_lherz_pct = pct_count(carp_fertile_lherz_count,tot_num_depth)
		carp_melt_metas_pct = pct_count(carp_melt_metas_count,tot_num_depth)
		carp_unclass_pct = pct_count(carp_unclass_count,tot_num_depth)

		total = carp_melt_metas_pct + carp_fertile_lherz_pct + carp_depleted_metasomatised_pct +\
		carp_deplet_harz_pct + carp_deplet_lherz_pct

		deplet_harz_pct_list.append(carp_deplet_harz_pct)
		deplet_lherz_pct_list.append(carp_deplet_lherz_pct)
		deplet_metas_pct_list.append(carp_depleted_metasomatised_pct)
		fertile_pct_list.append(carp_fertile_lherz_pct)
		melt_pct_metas.append(carp_melt_metas_pct)
		unclass_pct.append(carp_unclass_pct)
		total_list.append(tot_num_depth)
		total_classified.append(depth_count)

	#making plot_ranges

	last_box = np.array(deplet_harz_pct_list) + np.array(deplet_lherz_pct_list) +\
	np.array(deplet_metas_pct_list) + np.array(fertile_pct_list) +\
	np.array(melt_pct_metas)
	fourth_box = np.array(deplet_harz_pct_list) + np.array(deplet_lherz_pct_list) +\
	np.array(deplet_metas_pct_list) + np.array(fertile_pct_list)
	third_box = np.array(deplet_harz_pct_list) + np.array(deplet_lherz_pct_list) + \
	np.array(deplet_metas_pct_list)
	second_box = np.array(deplet_harz_pct_list) + np.array(deplet_lherz_pct_list)
	first_box = np.array(deplet_harz_pct_list)

	deplet_harz_plot = []
	deplet_lherz_plot = []
	deplet_metas_plot = []
	fertile_plot = []
	melt_plot = []
	depth_plot = []

	for i in range(0,len(deplet_harz_pct_list)):
		for k in range(0,2):
			deplet_harz_plot.append(first_box[i])
			deplet_lherz_plot.append(second_box[i])
			deplet_metas_plot.append(third_box[i])
			fertile_plot.append(fourth_box[i])
			melt_plot.append(last_box[i])

			if i == 0:
				if k == 0:
					depth_plot.append(depth_fields[i])
			elif i == len(deplet_harz_pct_list)-1:
				depth_plot.append(depth_fields[i])
				if k == 1:
					depth_plot.append(depth_fields[i] + (depth_interval * 1e3))
			else:
				depth_plot.append(depth_fields[i])

	zeros = np.zeros(len(depth_plot))
	depth_plot = np.array(depth_plot)
	fig1 = plt.figure(figsize = (20,7))
	ax1 = plt.subplot2grid((16,16),(0,0), rowspan = 16,colspan = 2, fig = fig1)


	ax1.fill_betweenx(depth_plot / 1e3,melt_plot,zeros,color = '#ad0f1f',label = 'Melt Metasomatised')
	ax1.fill_betweenx(depth_plot / 1e3,fertile_plot,zeros,color = '#0b820f',label = 'Fertile')
	ax1.fill_betweenx(depth_plot / 1e3,deplet_metas_plot,zeros,color = '#0b3c82',label = 'Depleted Lherz. + Phlg. Metas.')
	ax1.fill_betweenx(depth_plot / 1e3,deplet_lherz_plot,zeros,color = '#dbc14e', label = 'Depleted Lherzolites')
	ax1.fill_betweenx(depth_plot / 1e3,deplet_harz_plot,zeros,color = '#ab963a',label = 'Depleted Harzburgite',hatch = 'xxx')
	for i in range(0,len(depth_fields)):
		ax1.axhline(depth_fields[i] / 1e3, linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')
	ax1.invert_yaxis()
	ax1.legend(fontsize = 6, bbox_to_anchor=(0.05, 0.925), borderaxespad=0.1,framealpha = 1.0)
	ax1.set_xlim(0,100)
	ax1.set_ylim((np.amax(depth_fields) / 1e3) + 10, (np.amin(depth_fields) / 1e3)-10)
	ax1.set_facecolor('#f7f7f2')
	#ax1.set_ylim(max(depth_plot) + 20,40)
	ax1.set_xlabel('Percent')
	ax1.set_ylabel('Depth (km)')
	ax1.axhline(BDL_depth / 1e3,linewidth = 2,color = 'k',linestyle = ':',label = 'Y-edge')

	depth_second = []
	for i in range(1,len(depth_fields)):
		depth_second.append((depth_fields[i] + depth_fields[i-1])/2e3)
	ax2 = plt.subplot2grid((16,16),(0,2), rowspan = 16,colspan = 1, fig = fig1)
	ax2.plot(total_classified,depth_second,label = 'Classified', color = '#b02f1e')
	ax2.plot(total_list,depth_second, linestyle = '--',dashes=(5, 12), label = 'Analysed', color = 'k')
	ax2.set_ylim((np.amax(depth_fields) / 1e3)+10,(np.amin(depth_fields) / 1e3)-10)
	ax2.set_facecolor('#f7f7f2')
	#ax2.set_ylim(max(depth_plot) + 20,40)
	for i in range(0,len(depth_fields)):
		ax2.axhline((depth_fields[i] / 1e3), linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')
	ax2.fill_betweenx(depth_second,total_list,total_classified,color = 'k',alpha = 0.1)
	ax2.legend(fontsize = 6)
	ax2.set_xlabel('No.Samples - Log')
	ax2.grid(alpha = 0.6)
	#ax2.set_title('CARP - analysed samples,\n total = ' + str(len(T_Ni)),fontsize = 4)
	ax2.set_yticklabels([])
	ax2.axhline(BDL_depth / 1e3,linewidth = 2,color = 'k',linestyle = ':')
	ax2.set_xscale('log')
	ax2.set_xlim((1,10000))
	ax2.set_xticks([1,10,100,1000,1e4])
	ax2.set_xticklabels([0,1,2,3,4])
	ax2.grid(which = 'minor')

	low_cr_pct_list = []
	lowca_pct_list = []
	ca_harz_pct_list = []
	lherz_pct_list = []
	wehr_pct_list = []
	unclas_pct_list = []

	for j in range(1,len(depth_fields)):
		lowcr_count = 0
		lowca_count = 0
		harz_count = 0
		lherz_count = 0
		wehr_count = 0
		unclass_count = 0
		for i in range(0,len(depth_normal)):
			if (depth_normal[i] < depth_fields[j]) and (depth_normal[i] >= depth_fields[j-1]):
				if cacr_class[i] == 'Low-Cr Peridotite':
					lowcr_count = lowcr_count + 1
				elif cacr_class[i] == 'Low-Ca Harzburgite':
					lowca_count = lowca_count + 1
				elif cacr_class[i] == 'Ca-Harzburgite':
					harz_count = harz_count + 1
				elif cacr_class[i] == 'Lherzolite':
					lherz_count = lherz_count + 1
				elif cacr_class[i] == 'Wehrlite':
					wehr_count = wehr_count + 1
				else:
					unclass_count = unclass_count + 1

		tot_count = lowcr_count + lowca_count + harz_count +\
		lherz_count + wehr_count + unclass_count

		if tot_count != 0:

			low_cr_pct_list.append((lowcr_count / tot_count) * 1e2)
			lowca_pct_list.append((lowca_count / tot_count) * 1e2)
			ca_harz_pct_list.append((harz_count / tot_count) * 1e2)
			lherz_pct_list.append((lherz_count / tot_count) * 1e2)
			wehr_pct_list.append((wehr_count / tot_count) * 1e2)
			unclas_pct_list.append((unclass_count / tot_count) * 1e2)
		else:
			low_cr_pct_list.append(0)
			lowca_pct_list.append(0)
			ca_harz_pct_list.append(0)
			lherz_pct_list.append(0)
			wehr_pct_list.append(0)
			unclas_pct_list.append(0)

	last_box = np.array(lowca_pct_list) + np.array(ca_harz_pct_list) +\
	np.array(lherz_pct_list) + np.array(low_cr_pct_list) +\
	np.array(wehr_pct_list)
	fourth_box = np.array(lowca_pct_list) + np.array(ca_harz_pct_list) +\
	np.array(lherz_pct_list) + np.array(low_cr_pct_list)
	third_box = np.array(lowca_pct_list) + np.array(ca_harz_pct_list) +\
	np.array(lherz_pct_list)
	second_box = np.array(lowca_pct_list) + np.array(ca_harz_pct_list)
	first_box = np.array(lowca_pct_list)


	lowca_plot = []
	ca_harz_plot = []
	lherz_plot = []
	lowcr_plot = []
	wehr_plot = []
	depth_plot = []

	for i in range(0,len(lowca_pct_list)):
		for k in range(0,2):
			lowca_plot.append(first_box[i])
			ca_harz_plot.append(second_box[i])
			lherz_plot.append(third_box[i])
			lowcr_plot.append(fourth_box[i])
			wehr_plot.append(last_box[i])
			if i == 0:
				if k == 0:
					depth_plot.append(depth_fields[i])
			elif i == len(deplet_harz_pct_list)-1:
				depth_plot.append(depth_fields[i])
				if k == 1:
					depth_plot.append(depth_fields[i] + depth_interval)
			else:
				depth_plot.append(depth_fields[i])

	depth_plot = np.array(depth_plot)

	zeros = np.zeros(len(depth_plot))
	ax3 = plt.subplot2grid((16,16),(0,3), rowspan = 16,colspan = 2, fig = fig1)
	ax3.fill_betweenx(depth_plot / 1e3, wehr_plot,zeros,color = '#b06f90',label = 'Wehrlite')
	ax3.fill_betweenx(depth_plot / 1e3, lowcr_plot,zeros,color = '#d9c1cb',label = 'Low-Cr')
	ax3.fill_betweenx(depth_plot / 1e3, lherz_plot,zeros,color = '#339e51',label = 'Lherzolite')
	ax3.fill_betweenx(depth_plot / 1e3, ca_harz_plot,zeros,color = '#ef2fa1', label = 'Ca Harzburgite')
	ax3.fill_betweenx(depth_plot / 1e3, lowca_plot,zeros,color = '#8b1f8f',label = 'Low-Ca Harzburgite',hatch = 'xxx')
	for i in range(0,len(depth_fields)):
		ax3.axhline(depth_fields[i] / 1e3, linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')
	ax3.invert_yaxis()
	ax3.set_facecolor('#f7f7f2')
	ax3.legend(fontsize = 8, bbox_to_anchor=(0.075, 0.925), borderaxespad=0.1,framealpha = 1.0)
	ax3.set_xlim(0,100)
	ax3.set_xlabel('Percent')
	ax3.set_ylim((np.amax(depth_fields) / 1e3)+10,(np.amin(depth_fields)/1e3)-10)
	#ax3.set_ylim(max(depth_plot) + 20,40)
	ax3.set_yticklabels([])
	ax3.axhline(BDL_depth / 1e3,linewidth = 2,color = 'k',linestyle = ':')

	g10_list = []
	g10d_list = []
	g9_list = []
	g12_list = []
	g1_list = []
	g11_list = []
	g5_list = []
	g5d_list = []
	g4_list = []
	g4d_list = []
	g3_list = []
	g3d_list = []
	g0_list = []

	for j in range(1,len(depth_fields)):

		g10_count = 0
		g10d_count = 0
		g9_count = 0
		g12_count = 0
		g1_count = 0
		g11_count = 0
		g5_count = 0
		g5d_count = 0
		g4_count = 0
		g4d_count = 0
		g3_count = 0
		g3d_count = 0
		g0_count = 0
		tot_num_depth = 0

		for i in range(0,len(depth_normal)):
			if (depth_normal[i] < depth_fields[j]) and (depth_normal[i] >= depth_fields[j-1]):
				if g_class[i] == 'G10':
					g10_count = g10_count + 1
				elif g_class[i] == 'G10D':
					g10d_count = g10d_count + 1
				elif g_class[i] == 'G9':
					g9_count = g9_count + 1
				elif g_class[i] == 'G12':
					g12_count = g12_count + 1
				elif g_class[i] == 'G1':
					g1_count = g1_count + 1
				elif g_class[i] == 'G11':
					g11_count = g11_count + 1
				elif g_class[i] == 'G5':
					g5_count = g5_count + 1
				elif g_class[i] == 'G5D':
					g5d_count = g5d_count + 1
				elif g_class[i] == 'G4':
					g4_count = g4_count + 1
				elif g_class[i] == 'G4D':
					g4d_count = g4d_count + 1
				elif g_class[i] == 'G3':
					g3_count = g3_count + 1
				elif g_class[i] == 'G3D':
					g3d_count = g3d_count + 1
				elif g_class[i] == 'G0':
					g0_count + g0_count + 1

		depth_count = g10_count + g10d_count + g9_count + g12_count +\
		g1_count + g11_count + g5_count + g5d_count + g4_count + g4d_count +\
		g3_count + g3d_count + g0_count

		g10_list.append(pct_count(g10_count,depth_count))
		g10d_list.append(pct_count(g10d_count,depth_count))
		g9_list.append(pct_count(g9_count,depth_count))
		g12_list.append(pct_count(g12_count,depth_count))
		g1_list.append(pct_count(g1_count,depth_count))
		g11_list.append(pct_count(g11_count,depth_count))
		g5_list.append(pct_count(g5_count,depth_count))
		g5d_list.append(pct_count(g5d_count,depth_count))
		g4_list.append(pct_count(g4_count,depth_count))
		g4d_list.append(pct_count(g4d_count,depth_count))
		g3_list.append(pct_count(g3_count,depth_count))
		g3d_list.append(pct_count(g3d_count,depth_count))
		g0_list.append(pct_count(g0_count,depth_count))

	ax4 = plt.subplot2grid((16,16),(0,5), rowspan = 16,colspan = 2, fig = fig1)

	list_lists = [np.array(g10_list),np.array(g10d_list),
	np.array(g9_list),np.array(g12_list),np.array(g1_list),
	np.array(g11_list),np.array(g5_list),np.array(g5d_list),
	np.array(g4_list),np.array(g4d_list),np.array(g3_list),
	np.array(g3d_list),np.array(g0_list)]
	labels_g = ['G10','G10D','G9','G12','G1','G11','G5','G5D','G4','G4D',
	'G3','G3D','G0']
	colors_g = ['#F18F01','#CB7A01', '#81A739','#79B791', '#048BA8', '#036277',
	'#FFDD33','#CCAA00','#D84727','#9B341C','#896a8a','#5C475C','#F7FFF7']
	boxes_g = []
	boxes_g_plot = []
	for i in range(0,len(list_lists)):
		boxes_g.append(sum(list_lists[:i+1]))
	for i in range(0,len(boxes_g)):
		boxes_g_plot_local = []
		for j in range(0,len(boxes_g[i])):
			for k in range(0,2):
				boxes_g_plot_local.append(boxes_g[i][j])
		boxes_g_plot.append(boxes_g_plot_local)

	boxes_g_plot = boxes_g_plot[::-1]
	labels_g = labels_g[::-1]
	colors_g = colors_g[::-1]
	for i in range(0,len(boxes_g_plot)):
		ax4.fill_betweenx(depth_plot / 1e3,boxes_g_plot[i],zeros,color = colors_g[i], label = labels_g[i])

	for i in range(0,len(depth_fields)):
		ax4.axhline(depth_fields[i] / 1e3, linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')

	ax4.invert_yaxis()
	ax4.set_facecolor('#f7f7f2')
	ax4.legend(fontsize = 6, bbox_to_anchor=(0.05, 0.925), borderaxespad=0.075,framealpha = 1.0,ncol = 3)
	ax4.set_xlim(0,100)
	ax4.set_xlabel('Percent')
	ax4.set_ylim((np.amax(depth_fields) / 1e3)+10,(np.amin(depth_fields)/1e3)-10)
	#ax4.set_ylim(max(depth_plot) + 20,40)
	ax4.set_yticklabels([])
	ax4.axhline(BDL_depth/1e3,linewidth = 2,color = 'k',linestyle = ':')


	ax5 = plt.subplot2grid((16,16),(0,7), rowspan = 16,colspan = 5, fig = fig1)

	for i in range(0,len(geotherms)):
		ax5.plot(geotherms[i],depths[i] / 1e3,'--', linewidth = 0.75)

	d_g_p = np.array(d_g_p)
	z_1 = [2.50051235e-11, -9.09533105e-09,  1.34139387e-06, -1.03026414e-04,  4.40294443e-03, -1.03459682e-01,  4.27676616e+00]
	p_cr_depth = ((z_1[0] * (P_Cr * 10)**6.0) + (z_1[1] * (P_Cr * 10)**5.0) + (z_1[2] * (P_Cr * 10)**4.0) + (z_1[3] * (P_Cr * 10)**3.0) + (z_1[4] * (P_Cr * 10)**2.0) + (z_1[5] * (P_Cr * 10)) + z_1[6]) * (P_Cr * 10)
	d_g_depth = ((z_1[0] * d_g_p**6.0) + (z_1[1] * d_g_p**5.0) + (z_1[2] * d_g_p**4.0) + (z_1[3] * d_g_p**3.0) + (z_1[4] * d_g_p**2.0) + (z_1[5] * d_g_p) + z_1[6]) * d_g_p

	ax5.plot(d_g_t,d_g_depth,'-',color = '#a832a4')
	ax5.plot(T_chosen,depth_chosen/1e3,'--', color = 'k', linewidth = 2)

	ax5.set_facecolor('#f7f7f2')

	color_list_carp = ['#ab963a','#dbc14e','#0b3c82','#0b820f','#ad0f1f','k']
	carp_depleted_p = []
	carp_depleted_t = []
	carp_depleted_y = []

	for i in carp_depleted_harz:
		carp_depleted_p.append(p_cr_depth[i])
		carp_depleted_t.append(T_Ni[i])
	ax5.plot(carp_depleted_t,carp_depleted_p,'d', color = color_list_carp[0],
	markeredgewidth = 0.3,markeredgecolor = 'k', label = 'Depleted Harzburgite')

	carp_depleted_lherz_p = []
	carp_depleted_lherz_t = []
	carp_depleted_lherz_y = []
	for i in carp_depleted_lherz:
		carp_depleted_lherz_p.append(p_cr_depth[i])
		carp_depleted_lherz_t.append(T_Ni[i])
	ax5.plot(carp_depleted_lherz_t,carp_depleted_lherz_p,'s', color = color_list_carp[1],
	markeredgewidth = 0.3,markeredgecolor = 'k',label = 'Depleted Lherzolite')

	carp_met_depleted_p = []
	carp_met_depleted_t = []
	carp_met_depleted_y = []
	for i in carp_depleted_metasomatised:
		carp_met_depleted_p.append(p_cr_depth[i])
		carp_met_depleted_t.append(T_Ni[i])
	ax5.plot(carp_met_depleted_t,carp_met_depleted_p,'^', color = color_list_carp[2],
	markeredgewidth = 0.3,markeredgecolor = 'k',label = 'Depleted Lherzolite + Phlg. Metas.')

	carp_fertile_lherz_p = []
	carp_fertile_lherz_t = []
	carp_fertile_lherz_y = []
	for i in carp_fertile_lherz:
		carp_fertile_lherz_p.append(p_cr_depth[i])
		carp_fertile_lherz_t.append(T_Ni[i])
	ax5.plot(carp_fertile_lherz_t,carp_fertile_lherz_p,'*', color = color_list_carp[3],
	markeredgewidth = 0.3,markeredgecolor = 'k',label = 'Fertile')

	carp_melt_metas_p = []
	carp_melt_metas_t = []
	carp_melt_metas_y = []
	for i in carp_melt_metas:
		carp_melt_metas_p.append(p_cr_depth[i])
		carp_melt_metas_t.append(T_Ni[i])
	ax5.plot(carp_melt_metas_t,carp_melt_metas_p,'h', color = color_list_carp[4],
	markeredgewidth = 0.3,markeredgecolor = 'k',label = 'Melt Metasomatised')

	carp_unclass_p = []
	carp_unclass_t = []
	carp_unclass_y = []
	for i in carp_unclass:
		carp_unclass_p.append(p_cr_depth[i])
		carp_unclass_t.append(T_Ni[i])
	ax5.plot(carp_unclass_t,carp_unclass_p,'x', color = color_list_carp[5],label = 'Unclassified')

	ax5.axhline(BDL_depth/1e3,linewidth = 2,color = 'k',linestyle = ':')
	ax5.set_xlim((500,1500))
	ax5.set_ylim((np.amax(depth_fields)/1e3)+10,(np.amin(depth_fields)/1e3)-10)
	#ax5.set_ylim(max(depth_plot) + 20,40)
	for i in range(0,len(depth_fields)):
		ax5.axhline(depth_fields[i]/1e3, linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')
	ax5.set_xlabel(r'$T_{Ni}({}^{\circ}C)$')
	ax5.set_yticklabels([])


	ax5.axvline(BDL_T,linestyle = '--',color = 'k',linewidth = 1)

	#ax5.legend(fontsize = 8, bbox_to_anchor=(0.25, 0.97), borderaxespad=0.1,framealpha = 1.0)
	ax5.legend(fontsize = 8,framealpha = 1.0)

	al_mean_list = []
	al_median_list = []

	for j in range(1,len(depth_fields)):
		al_local = []
		for i in range(0,len(depth_normal)):
			if (depth_normal[i] < depth_fields[j]) and (depth_normal[i] >= depth_fields[j-1]):
				al_local.append(AL[i])

		al_mean_list.append(np.mean(np.array(al_local)))
		al_median_list.append(np.median(np.array(al_local)))

	ax6 = plt.subplot2grid((16,16),(0,12), rowspan = 16,colspan = 1, fig = fig1)
	ax6.plot(al_mean_list,depth_second,label = 'Mean', color = '#b02f1e')
	ax6.plot(al_median_list,depth_second, linestyle = '--',dashes=(5, 12), label = 'Median', color = 'k')
	ax6.set_ylim((np.amax(depth_fields)/1e3)+10,(np.amin(depth_fields)/1e3)-10)
	ax6.set_facecolor('#f7f7f2')
	#ax6.set_ylim(max(depth_plot) + 20,40)
	for i in range(0,len(depth_fields)):
		ax6.axhline(depth_fields[i]/1e3, linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')
	ax6.legend(fontsize = 6,framealpha = 1.0)
	ax6.set_xlim((0,3))
	ax6.axhline(BDL_depth/1e3,linewidth = 2,color = 'k',linestyle = ':')
	ax6.set_xlabel(r'WR $Al_2O_3$')
	ax6.set_yticklabels([])

	mgno_mean_list = []
	mgno_median_list = []

	for j in range(1,len(depth_fields)):
		mgno_local = []
		for i in range(0,len(depth_normal)):
			if (depth_normal[i] < depth_fields[j]) and (depth_normal[i] >= depth_fields[j-1]):
				mgno_local.append(MG[i])

		mgno_mean_list.append(np.mean(np.array(mgno_local)))
		mgno_median_list.append(np.median(np.array(mgno_local)))

	ax7 = plt.subplot2grid((16,16),(0,13), rowspan = 16,colspan = 1, fig = fig1)
	for item in [0.88,0.9,0.92,0.94]:
		ax7.axvline(item,linewidth = 0.5,color = 'k', alpha = 0.3)
	ax7.plot(mgno_mean_list,depth_second,label = 'Mean', color = '#b02f1e')
	ax7.plot(mgno_median_list,depth_second, linestyle = '--',dashes=(5, 12), label = 'Median', color = 'k')
	ax7.set_ylim((np.amax(depth_fields)/1e3)+10,(np.amin(depth_fields)/1e3)-10)
	ax7.set_facecolor('#f7f7f2')
	#ax7.set_ylim(max(depth_plot) + 20, 40)
	for i in range(0,len(depth_fields)):
		ax7.axhline(depth_fields[i]/1e3, linewidth = 0.75, alpha = 0.5, color = 'k',linestyle = '-.')
	ax7.axhline(BDL_depth / 1e3,linewidth = 2,color = 'k',linestyle = ':')

	ax7.legend(fontsize = 6,framealpha = 1.0)
	ax7.set_xlabel(r'#${Mg}^{ol}$')
	ax7.set_xlim((0.87,0.96))
	ax7.set_yticklabels([])
	ax7.set_xticks([0.88,0.9,0.92,0.94,0.96])
	ax7.set_xticklabels([88,90,92,94,96],fontsize = 7)

	if plot_type == 'show':
		plt.show()
	elif plot_type == 'save':
		if filename_save == None:
			save_str = os.path.join((os.getcwd(),'CARP_PLOT.png'),bbox_inches = "tight")
			plt.savefig(save_str, dpi = 300.0)
		else:
			try:
				plt.savefig(filename_save, dpi = 300.0,bbox_inches = "tight")
			except ValueError:
				print('Not a proper format for the filename_save')
	return plt