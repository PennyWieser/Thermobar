import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as PATH
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
import pandas as pd
import os,sys
import itertools
from Thermobar.core import *

def garnet_CARP_class_Griffin2002(gt_comps):

    '''
    A function to classify Cr-pyrope garnet xenocrysts according to
    the study of Griffin et al. (2002)
    :cite: `griffin2002`

    ###Parameters###
    gt_comps: garnet composition dataframe imported from core function
    '''

    idx_critical_missing = []
    for i in range(0,len(gt_comps)):

        if (pd.isna(gt_comps['CaO_Gt'][i]) == True) or (pd.isna(gt_comps['Cr2O3_Gt'][i]) == True) or\
         (pd.isna(gt_comps['MgO_Gt'][i]) == True) or (pd.isna(gt_comps['FeOt_Gt'][i]) == True) or\
         (pd.isna(gt_comps['TiO2_Gt'][i]) == True) or  (pd.isna(gt_comps['Zr_Gt'][i]) == True) or\
         (pd.isna(gt_comps['Y_Gt'][i]) == True):

            idx_critical_missing.append(i)

    if len(idx_critical_missing) > 0:
        print('WARNING')
        print('There are critical components missing in some of the samples for CARP analysis and might lead to wrong analysis. idx list: ')
        print(idx_critical_missing)

    h1 = []
    h2 = []
    h3 = []
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    l9 = []
    l10a = []
    l10b = []
    l11 = []
    l12 = []
    l13 = []
    l14 = []
    l15 = []
    l16 = []
    l17 = []
    l18 = []
    l19 = []
    l20 = []
    l21 = []
    l22 = []
    l23 = []
    l24 = []
    l25 = []
    l26 = []
    l27 = []
    l28 = []
    carp_type_list = []

    #Defining CARP tree from Griffin et al. (2002, G-Cubed,doi:10.1029/2002gc000298)

    for i in range(0,len(np.array(gt_comps['CaO_Gt']))):

        if gt_comps['CaO_Gt'][i] < 4.0:
            if gt_comps['MgO_Gt'][i] < 21.05:
                h1.append(i)
                type = 'H1'
            else:
                if gt_comps['Y_Gt'][i] < 11.05:
                    h2.append(i)
                    type = 'H2'
                else:
                    h3.append(i)
                    type = 'H3'
        else:
            if gt_comps['MgO_Gt'][i] > 22.55:
                l28.append(i)
                type = 'L28'
            else:
                if gt_comps['Cr2O3_Gt'][i] < 0.73:
                    if gt_comps['FeOt_Gt'][i] < 9.1:
                        l1.append(i)
                        type = 'L1'
                    else:
                        l2.append(i)
                        type = 'L2'
                else:
                    if gt_comps['Zn_Gt'][i] > 22.0:
                        if gt_comps['TiO2_Gt'][i] < 0.47:
                            l25.append(i)
                            type = 'L25'
                        else:
                            if gt_comps['Zr_Gt'][i] < 35.0:
                                l26.append(i)
                                type = 'L26'
                            else:
                                l27.append(i)
                                type = 'L27'
                    else:
                        if gt_comps['Y_Gt'][i] < 5.0:
                            if gt_comps['TiO2_Gt'][i] > 0.135:
                                l7.append(i)
                                type = 'L7'
                            else:
                                if gt_comps['Zr_Gt'][i] < 14.5:
                                    l3.append(i)
                                    type = 'L3'
                                else:
                                    if gt_comps['Ga_Gt'][i] > 8.8:
                                        l6.append(i)
                                        type = 'L6'
                                    else:
                                        if gt_comps['Cr2O3_Gt'][i] < 3.8:
                                            l4.append(i)
                                            type = 'L4'
                                        else:
                                            l5.append(i)
                                            type = 'L5'
                        else:
                            if gt_comps['Cr2O3_Gt'][i] > 5.84:
                                if gt_comps['CaO_Gt'][i] < 5.4:
                                    l18.append(i)
                                    type = 'L18'
                                else:
                                    if gt_comps['FeOt_Gt'][i] > 8.3:
                                        l24.append(i)
                                        type = 'L24'
                                    else:
                                        if gt_comps['MgO_Gt'][i] > 20.45:
                                            if gt_comps['FeOt_Gt'][i] < 6.85:
                                                l22.append(i)
                                                type = 'L22'
                                            else:
                                                l23.append(i)
                                                type = 'L23'
                                        else:
                                            if gt_comps['Y_Gt'][i] < 15.4:
                                                l19.append(i)
                                                type = 'L19'
                                            else:
                                                if gt_comps['Zr_Gt'][i] < 53.7:
                                                    l20.append(i)
                                                    type = 'L20'
                                                else:
                                                    l21.append(i)
                                                    type = 'L21'
                            else:
                                if gt_comps['CaO_Gt'][i] > 6.09:
                                    l17.append(i)
                                    type = 'L17'
                                else:
                                    if gt_comps['Sr_Gt'][i] > 2.5:
                                        l16.append(i)
                                        type = 'L16'
                                    else:
                                        if gt_comps['MgO_Gt'][i] < 19.15:
                                            if gt_comps['FeOt_Gt'][i] < 8.1:
                                                l8.append(i)
                                                type = 'L8'
                                            else:
                                                l9.append(i)
                                                type = 'L9'
                                        else:
                                            if gt_comps['CaO_Gt'][i] > 5.54:
                                                if gt_comps['Cr2O3_Gt'][i] < 2.28:
                                                    l14.append(i)
                                                    type = 'L14'
                                                else:
                                                    l15.append(i)
                                                    type = 'L15'
                                            else:
                                                if gt_comps['Ga_Gt'][i] > 8.6:
                                                    if gt_comps['TiO2_Gt'][i] < 0.13:
                                                        l12.append(i)
                                                        type = 'L12'
                                                    else:
                                                        l13.append(i)
                                                        type = 'L13'
                                                else:
                                                    if gt_comps['TiO2_Gt'][i] > 0.37:
                                                        l11.append(i)
                                                        type = 'L11'
                                                    else:
                                                        if gt_comps['TiO2_Gt'][i] > 0.1:
                                                            l10b.append(i)
                                                            type = 'L10b'
                                                        else:
                                                            l10a.append(i)
                                                            type = 'L10a'
        carp_type_list.append(type)

    len_tot = len(h1) + len(h2) + len(h3) + len(l28) + len(l1) + len(l2) +\
    len(l25) + len(l26) + len(l27) + len(l7) + len(l3) + len(l6) +\
    len(l4) + len(l5) + len(l18) + len(l24) + len(l22) + len(l23) +\
    len(l19) + len(l20) + len(l21) + len(l17) + len(l16) + len(l8) +\
    len(l9) + len(l14) + len(l15) + len(l12) + len(l13) + len(l11) +\
    len(l10b) + len(l10a)
    carp_depleted_harz = h2
    carp_depleted_lherz = list(itertools.chain(l3,l5))
    carp_depleted_metasomatised = list(itertools.chain(h3,l15,
    l18,l19,l21))
    carp_fertile_lherz = list(itertools.chain(l9,l10a,l10b,l16))
    carp_melt_metas = list(itertools.chain(l13,l25,l27))
    carp_unclass = list(itertools.chain(h1,l4,l6,l7,l8,l22,
    l12,l14,l17,l20,l11,l23,l24,l26,l28))
    carp_depleted_harz.sort()
    carp_depleted_metasomatised.sort()
    carp_fertile_lherz.sort()
    carp_melt_metas.sort()
    carp_unclass.sort()

    return carp_depleted_harz, carp_depleted_lherz, carp_depleted_metasomatised, carp_fertile_lherz, carp_melt_metas, carp_unclass, len_tot

def garnet_class_Grutter2004(gt_comps):

    '''
    A function to classify Cr-pyrope garnet xenocrysts according to
    the study of Grutter et al. (2004)
    :cite: `grutter2004`

    ###Parameters###
    gt_comps: garnet composition dataframe imported from core function
    '''

    idx_critical_missing = []
    idx_na_missing = []
    for i in range(0,len(gt_comps)):
        if (pd.isna(gt_comps['CaO_Gt'][i]) == True) or (pd.isna(gt_comps['Cr2O3_Gt'][i]) == True) or\
         (pd.isna(gt_comps['MgO_Gt'][i]) == True) or (pd.isna(gt_comps['FeOt_Gt'][i]) == True) or\
         (pd.isna(gt_comps['MnO_Gt'][i]) == True) or  (pd.isna(gt_comps['TiO2_Gt'][i]) == True):

            idx_critical_missing.append(i)

        if (pd.isna(gt_comps['Na2O_Gt'][i]) == True):

            idx_na_missing.append(i)

    if len(idx_critical_missing) > 0:
        print('WARNING')
        print('There are critical components missing in some of the samples for Grutter2004 analysis. For these samples analyses might be wrong. idx list: ')
        print(idx_critical_missing)

    if len(idx_na_missing) > 0:
        print('WARNING')
        print('Na2O component is missing in some of the samples, diamondiferious classificatons might be wrong, defaulting to non-indicator classification. idx list: ')
        print(idx_na_missing)

    ca_int = []
    mg_num_g = []
    g10 = []
    g10d = []
    g9 = []
    g12 = []
    g1 = []
    g11 = []
    g5 = []
    g4 = []
    g4d = []
    g3 = []
    g3d = []
    g5d = []
    g_class = []
    g_all = []

    for i in range(0,len(np.array(gt_comps['CaO_Gt']))):
        if gt_comps['CaO_Gt'][i] < ((gt_comps['Cr2O3_Gt'][i]*0.25) + 3.375):
            ca_int.append((13.5*gt_comps['CaO_Gt'][i])/(gt_comps['Cr2O3_Gt'][i]+13.5))
        else:
            ca_int.append(gt_comps['CaO_Gt'][i] - (0.25*gt_comps['Cr2O3_Gt'][i]))

        mg_num_g.append((gt_comps['MgO_Gt'][i] / 40.3) / ((gt_comps['MgO_Gt'][i] / 40.3) + (gt_comps['FeOt_Gt'][i]/71.85)))

        stop_search = False
        one_before_g1 = False

        if (gt_comps['Cr2O3_Gt'][i] >= 1.0) and (gt_comps['Cr2O3_Gt'][i] < 22.0):
            if (ca_int[i] > 0.0) and (ca_int[i] < 3.375):
                if (mg_num_g[i] >= 0.75) and (mg_num_g[i] < 0.95):
                    if (gt_comps['Cr2O3_Gt'][i] >= ((0.94*gt_comps['CaO_Gt'][i]) + 5)) and (gt_comps['MnO_Gt'][i] < 0.36):
                        g10d.append(i)
                        g_class.append('G10D')
                        stop_search = True
                    else:
                        g10.append(i)
                        g_class.append('G10')
                        stop_search = True

        if stop_search == False:
            if (gt_comps['Cr2O3_Gt'][i] >= 1.0):
                if (ca_int[i] >= 3.375) and (ca_int[i] < 5.4):
                    if (mg_num_g[i] >= 0.70) and (mg_num_g[i] < 0.90):
                        g9.append(i)
                        g_class.append('G9')
                        stop_search = True
                    elif (mg_num_g[i] >= 0.3) and (mg_num_g[i] < 0.7):
                        if gt_comps['TiO2_Gt'][i] < (2.13 - (2.1 * mg_num_g[i])):
                            if (gt_comps['Na2O_Gt'][i] > 0.07):
                                g5d.append(i)
                                g_class.append('G5D')
                                stop_search = True
                            else:
                                g5.append(i)
                                g_class.append('G5')
                                stop_search = True


        if stop_search == False:
            if (gt_comps['TiO2_Gt'][i] >= (2.13 - 2.1*mg_num_g[i])) and (gt_comps['TiO2_Gt'][i] < 4.0):
                if (gt_comps['Cr2O3_Gt'][i] >= 0.0) and (gt_comps['Cr2O3_Gt'][i] < 4.0):
                    if (ca_int[i] >= 3.375) and (ca_int[i] < 6.0):
                        if (mg_num_g[i] >= 0.65) and (mg_num_g[i] < 0.85):
                            g1.append(i)
                            g_class.append('G1')
                            one_before_g1 = True


        if stop_search == False:
            if (gt_comps['TiO2_Gt'][i] >= (2.13 - 2.1*mg_num_g[i])) and (gt_comps['TiO2_Gt'][i] < 4.0):
                if (gt_comps['Cr2O3_Gt'][i] >= 1.0) and (gt_comps['Cr2O3_Gt'][i] < 20.0):
                    if (ca_int[i] >= 3.0):
                        if gt_comps['CaO_Gt'][i] < 28.0:
                            if (mg_num_g[i] >= 0.65) and (mg_num_g[i] < 0.90):
                                if one_before_g1 == False:
                                    g11.append(i)
                                    g_class.append('G11')
                                    stop_search = True
                                else:
                                    g11.append(i)
                                    del g1[-1]
                                    g_class[-1] == 'G11'
                                    stop_search = True

        if stop_search == False:
            if (gt_comps['Cr2O3_Gt'][i] >= 1.0) and (gt_comps['Cr2O3_Gt'][i] < 20.0):
                if (ca_int[i] >= 5.4):
                    if gt_comps['CaO_Gt'][i] < 28.0:
                        if gt_comps['MgO_Gt'][i] > 5.0:
                            if one_before_g1 == True:
                                pass
                            else:
                                g12.append(i)
                                g_class.append('G12')
                                stop_search = True

        if stop_search == False:
            if (gt_comps['Cr2O3_Gt'][i] < 1.0):
                if (gt_comps['CaO_Gt'][i] >= 2.0) and (gt_comps['CaO_Gt'][i] < 6.0):
                    if (mg_num_g[i] >= 0.3) and (mg_num_g[i] < 0.9):
                        if gt_comps['TiO2_Gt'][i] < (2.13 - (2.1 * mg_num_g[i])):
                            if (gt_comps['Na2O_Gt'][i] > 0.07):
                                g4d.append(i)
                                g_class.append('G4D')
                                stop_search = True
                            else:
                                g4.append(i)
                                g_class.append('G4')
                                stop_search = True

        if stop_search == False:
            if (gt_comps['Cr2O3_Gt'][i] >= 0.0) and (gt_comps['Cr2O3_Gt'][i] < 1.0):
                if (gt_comps['CaO_Gt'][i] >= 6.0) and (gt_comps['CaO_Gt'][i] < 32.0):
                    if (mg_num_g[i] >= 0.17) and (mg_num_g[i] < 0.86):
                        if gt_comps['TiO2_Gt'][i] < (2.13 - (2.1 * mg_num_g[i])):
                            if gt_comps['TiO2_Gt'][i] < 2.0:
                                if one_before_g1 == True:
                                    pass
                                else:
                                    if (gt_comps['Na2O_Gt'][i] > 0.07):
                                        g3d.append(i)
                                        g_class.append('G3D')
                                        stop_search = True
                                    else:
                                        g3.append(i)
                                        g_class.append('G3')
                                        stop_search = True

    total_G = list(itertools.chain(g10,g10d,g9,g12,
    g1,g11,g5,g5d,g4,g4d,g3,g3d))

    #Checking for duplicates
    total_G_duplicates = set([x for x in total_G if total_G.count(x) > 1])
    if len(total_G_duplicates) != 0:
        print('There are duplicates in the G-classification scheme, check whats wrong')
    else:
        len_tot_g = len(total_G)

    g0 = []
    for i in range(0,len(np.array(gt_comps['CaO_Gt']))):
        if (i in total_G == False):
            g0.append(i)

    return g_class

def garnet_ca_cr_class_Griffin2002(gt_comps):

    '''
    A function to make CaO-Cr2O3 classifications of Cr-pyrope garnet xenocrysts
    according to the study of Griffin et al. (2002)
    :cite: `griffin2002`

    ###Parameters###
    gt_comps: garnet composition dataframe imported from core function
    '''

    #Checking if any CaO or Cr2O3 is missing

    for i in range(0,len(gt_comps)):

        if (pd.isna(gt_comps['CaO_Gt'][i]) == True) or (pd.isna(gt_comps['Cr2O3_Gt'][i]) == True):

            print('WARNING')
            print('There is CaO or Cr2O3 missing in one of the samples. idx = ' + str(i))

    low_cr_ca = [0,9,9,0]
    low_cr_cr = [0,0,1.52,1.52]

    low_cr_polygon = []
    for i in range(0,len(low_cr_ca)):
        low_cr_polygon.append((low_cr_ca[i],low_cr_cr[i]))
    low_cr_polygon_shp = PATH(low_cr_polygon)

    low_ca_harz_ca = [2.17,5.21,0.06832,0.029933,0.00055432]
    low_ca_harz_cr = [1.5201,12.01,12.01,6.9085,1.5305]

    low_ca_harz_polygon = []
    for i in range(0,len(low_ca_harz_ca)):
        low_ca_harz_polygon.append((low_ca_harz_ca[i],low_ca_harz_cr[i]))
    low_ca_harz_polygon_shp = PATH(low_ca_harz_polygon)

    ca_harz_ca = [2.172,5.215,7.062,3.914]
    ca_harz_cr = [1.52,12.001,12.001,1.52]

    ca_harz_polygon = []
    for i in range(0,len(ca_harz_cr)):
        ca_harz_polygon.append((ca_harz_ca[i],ca_harz_cr[i]))
    ca_harz_polygon_shp = PATH(ca_harz_polygon)

    lherz_ca = [3.9141,7.06205,8.73,5.75]
    lherz_cr = [1.5201,12.001,12.001,1.5201]

    lherz_polygon = []
    for i in range(0,len(ca_harz_cr)):
        lherz_polygon.append((lherz_ca[i],lherz_cr[i]))
    lherz_polygon_shp = PATH(lherz_polygon)

    wehrl_ca = [5.752,8.74,9,9]
    wehrl_cr = [1.5201,12.01,12.01,1.5201]

    wehrl_polygon = []
    for i in range(0,len(wehrl_cr)):
        wehrl_polygon.append((wehrl_ca[i],wehrl_cr[i]))
    wehrl_polygon_shp = PATH(wehrl_polygon)

    cacr_class = []
    list_cacr_class_list = []
    for i in range(0,len(np.array(gt_comps['CaO_Gt']))):
        list_cacr_class_list.append((gt_comps['CaO_Gt'][i],gt_comps['Cr2O3_Gt'][i]))
    a = low_cr_polygon_shp.contains_points(list_cacr_class_list)
    b = low_ca_harz_polygon_shp.contains_points(list_cacr_class_list)
    c = ca_harz_polygon_shp.contains_points(list_cacr_class_list)
    d = lherz_polygon_shp.contains_points(list_cacr_class_list)
    e = wehrl_polygon_shp.contains_points(list_cacr_class_list)
    for i in range(0,len(list_cacr_class_list)):
        check = False
        if a[i] == True:
            cacr_class.append('Low-Cr Peridotite')
            check = True
        if b[i] == True:
            cacr_class.append('Low-Ca Harzburgite')
            check = True
        if c[i] == True:
            cacr_class.append('Ca-Harzburgite')
            check = True
        if d[i] == True:
            cacr_class.append('Lherzolite')
            check = True
        if e[i] == True:
            cacr_class.append('Wehrlite')
            check == True
        if check == False:
            cacr_class.append('Unclassified')

    return cacr_class

def y_zr_classification_Griffin2002(gt_comps):

    '''
    A function to make Y-Zr classifications of Cr-pyrope garnet xenocrysts
    according to the study of Griffin et al. (2002)
    :cite: `griffin2002`

    ###Parameters###
    gt_comps: garnet composition dataframe imported from core function
    '''

    Zr_depleted = [0.1003,3.634,7.168,10.14,13.49,15.91,17.96,19.81,
    21.67,23.34,24.45,25.38,26.49,27.41,28.15,28.88,29.25,29.98,29.98,
    30.16,30.34,30.33,30.51,0.5567]
    Y_depleted = [13.59,13.4,13.34,13.21,13.08,13.08,12.89,12.55,
    12.15,11.35,10.75,9.948,9.213,8.343,7.339,6.536,5.397,4.326,3.455,
    2.517,1.579,0.3723,-0.02949,-0.07968]
    depleted_polygon = []
    for i in range(0,len(Zr_depleted)):
        depleted_polygon.append((Zr_depleted[i],Y_depleted[i]))
    depleted_polygon_shp = PATH(depleted_polygon)


    Zr_melt = [16.656,18.149,20.389,22.813,24.492,27.102,31.394,35.497,39.787,
    44.079,49.299,55.824,61.603,67.381,73.719,80.055,86.576,92.914,99.06,105.58,113.22,
    119.36,126.99,134.63,141.33,141.1,135.88,131.77,127.29,123,117.58,111.99,105.84,
    100.43,94.093,87.568,82.164,76.015,69.121,62.975,55.526,47.893,41.19,33.745,27.787,
    26.492,23.714,21.302,18.886]
    Y_melt = [13.218,13.958,14.833,15.574,16.247,16.989,18.672,20.019,21.434,23.184,
    24.667,26.42,27.972,29.456,31.075,32.493,33.778,35.464,36.278,37.362,
    38.447,39.261,40.012,40.695,41.309,35.343,33.927,32.245,29.958,27.739,25.586,23.633,
    22.081,20.463,18.844,17.023,15.942,14.658,13.306,12.357,11.474,10.791,9.975,9.5604,
    9.0142,10.017,11.688,12.556,12.887]
    melt_polygon = []
    for i in range(0,len(Zr_melt)):
        melt_polygon.append((Zr_melt[i],Y_melt[i]))
    melt_polygon_shp = PATH(melt_polygon)

    Zr_fertile = [0.35165,5.561,11.145,17.107,21.953,27.175,31.468,
    35.203,38.382,40.819,42.887,44.21,45.163,45.373,45.207,45.807,
    -0.14714,-10.465,-9.482]
    Y_fertile = [22.373,22.382,22.726,23.876,25.023,26.775,28.658,
    30.541,32.691,35.175,38.128,40.878,43.896,47.113,49.861,55.425,
    55.348,43.937,25.976]
    fertile_polygon = []
    for i in range(0,len(Zr_fertile)):
        fertile_polygon.append((Zr_fertile[i],Y_fertile[i]))
    fertile_polygon_shp = PATH(fertile_polygon)

    Zr_phlg = [28.34,30.918,35.569,45.058,56.967,67.198,80.409,89.713,
    96.79,107.04,113.76,119.36,125.15,130.56,137.84,140.46,139.76,135.85,130.62,122.97,
    115.7,108.24,100.97,93.696,85.683,77.11,66.865,50.856,42.297,35.598]
    Y_phlg = [8.2779,4.7299,4.6707,4.7536,5.0416,4.8577,5.0139,5.3646,
    6.2478,8.8789,11.973,14.731,17.086,19.374,22.536,24.216,30.918,29.303,27.216,
    24.522,22.097,19.873,17.649,15.492,13.803,11.711,10.085,8.9188,8.7704,8.5581]
    phlg_polygon = []
    for i in range(0,len(Zr_phlg)):
        phlg_polygon.append((Zr_phlg[i],Y_phlg[i]))
    phlg_polygon_shp = PATH(phlg_polygon)

    #Classification for lists
    yzr_class = []
    list_yzr_class_list = []
    for i in range(0,len(np.array(gt_comps['Y_Gt']))):
        list_yzr_class_list.append((gt_comps['Zr_Gt'][i],gt_comps['Y_Gt'][i]))
    a = melt_polygon_shp.contains_points(list_yzr_class_list)
    b = phlg_polygon_shp.contains_points(list_yzr_class_list)
    c = fertile_polygon_shp.contains_points(list_yzr_class_list)
    d = depleted_polygon_shp.contains_points(list_yzr_class_list)
    for i in range(0,len(list_yzr_class_list)):
        check = False
        if a[i] == True:
            yzr_class.append('Melt Metasomatised')
            check = True
        if b[i] == True:
            yzr_class.append('Phlogopite Metasomatism')
            check = True
        if c[i] == True:
            yzr_class.append('Fertile')
            check = True
        if d[i] == True:
            yzr_class.append('Depleted')
            check = True
        if check == False:
            yzr_class.append('Unclassified')

    return yzr_class

def calculate_ol_mg(gt_comps, T_Ni):

    '''
    A function to derive olivine Mg# from coxisting Cr-pyrope garnet xenocrysts.
    :cite: `gaul2000`

    ###Parameters###
    gt_comps: garnet composition dataframe imported from core function
    T_Ni: Temperature output taken from one of the garnet thermometry methods in
    Kelvin.

    Returns
    --------------
    Numpy array
        Ol Mg#
    '''

    mg_ol = np.zeros(len(gt_comps['MgO_Gt']))

    gt_calc = calculate_garnet_components(gt_comps = gt_comps)

    xMg = np.array(gt_calc['Mg_MgFeCa_Gt'])
    xCa = np.array(gt_calc['Ca_MgFeCa_Gt'])
    xFe = np.array(gt_calc['Fe_MgFeCa_Gt'])

    T_Ni = np.array(T_Ni) - 273.0 #converting to celsius for calculations

    for i in range(0,len(np.array(gt_comps['MgO_Gt']))):

        p_calc = (0.00001786*(T_Ni[i]**2.0)) + (0.027419*T_Ni[i]) - 1.198
        p1 = p_calc - (0.000263*(p_calc**2.0)) - 29.76
        p2 = p_calc - (0.00039*(p_calc**2.0)) - 29.65
        p3 = p_calc - (0.000236*(p_calc**2.0)) - 29.79
        p4 = p_calc - (0.00045*(p_calc**2.0)) - 29.6
        tk = T_Ni[i] + 273.0
        dv1 = -462.5*(1.0191+((tk-1073)*0.0000287))*p1
        dv2 = -262.4*(1.0292+((tk-1073)*0.000045))*p2
        dv3 = 454.0*(1.02+((tk-1073)*0.0000284))*p3
        dv4 = 278.3*(1.0234+((tk-1073)*0.000023))*p4
        #(1347*V2+902+AJ2+(0.9-(1-0.9))*(498+1.51*(R2-30))-98*(T2-U2))/W2-0.357
        dv = dv1 + dv2 + dv3 + dv4
        kd = np.exp((((1347*xCa[i])+902+dv+(0.9-(1-0.9))*(498+1.51*(p_calc-30))-98*(xMg[i]-xFe[i]))/tk) - 0.357)
        xmgfe = kd * (xMg[i]/xFe[i])
        mg_ol[i] = xmgfe / (1+xmgfe)

    return mg_ol

def calculate_al2o3_whole_rock(gt_comps):

    '''
    A function to derive whole-rock Al2O3 from coxisting Cr-pyrope garnets xenocrysts.
    :cite: `oreilly2006`

    ###Parameters###
    gt_comps: garnet composition dataframe imported from core function

    Returns
    --------------
    Numpy array
        Whole rock Al2O3 values.

    '''

    wr_al_y = np.zeros(len(gt_comps['Y_Gt']))

    for i in range(0,len(gt_comps['Y_Gt'])):

        if (gt_comps['Y_Gt'][i] > 0.0):
            wr_al_y[i] = (10.0**(0.778-0.14*(-3.2253*np.log(gt_comps['Y_Gt'][i])+13.006)))
        else:
            wr_al_y[i] = None

    return wr_al_y
