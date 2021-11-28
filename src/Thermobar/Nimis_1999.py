import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from pickle import load
import pickle
from pathlib import Path
Thermobar_dir=Path(__file__).parent
import joblib

from Thermobar.core import *

def first_step_Nimis_reclassification(cpx_comps):
    '''
    This function takes cpx_comps, and converst them into cations following Nimis 1999
    We had to write a separate function for this, because the molar masses they use are
    different enough you get noticably different results using our cation routines.

    Returns necessary inputs for function later_step_Nimis_reclassification

    '''
    # We rewrote this function because the molar masses are so different, by the
    # time you iterate 7 times, you end up with big pressure discrepencies
    cat_6_ox=pd.DataFrame(data={'Si': cpx_comps['SiO2_Cpx']/30.045,
                                 'Ti': cpx_comps['TiO2_Cpx']/39.95,
                                 'Al': cpx_comps['Al2O3_Cpx']/33.98,
                                'Cr': cpx_comps['Cr2O3_Cpx']/50.673,
                                 'Fet': cpx_comps['FeOt_Cpx']/71.85,
                                 'Mn': cpx_comps['MnO_Cpx']/70.94,
                                 'Mg': cpx_comps['MgO_Cpx']/40.32,
                                 'Ca': cpx_comps['CaO_Cpx']/56.08,
                                 'Na': cpx_comps['Na2O_Cpx']/61.982,
                                 'K': cpx_comps['K2O_Cpx']/94.204,
                                })
    SumB=cat_6_ox.sum(axis=1)


    # Then, we multiply by Sum B
    ox6_Alpt=pd.DataFrame(data={'SumB': SumB,
                                'Si': 3*cat_6_ox['Si']/SumB,
                                 'Ti': 3*cat_6_ox['Ti']/SumB,
                                 'Al': 4*cat_6_ox['Al']/SumB,
                                'Cr': 4*cat_6_ox['Cr']/SumB,
                                 'Fet': 6*cat_6_ox['Fet']/SumB,
                                 'Mn': 6*cat_6_ox['Mn']/SumB,
                                 'Mg': 6*cat_6_ox['Mg']/SumB,
                                 'Ca': 6*cat_6_ox['Ca']/SumB,
                                 'Na': 12*cat_6_ox['Na']/SumB,
                                 'K': 12*cat_6_ox['K']/SumB,

                                })

        #  Now lets allocate Al
    # If Si cations >2, Al4 is 0. Set as default
    HighSi=ox6_Alpt['Si']>2
    ox6_Alpt['Al4']=0
    # Else
    # If 2-Si cat > Al cations, equatil to Al cations
    HighSivsAl=(2-ox6_Alpt['Si'])>ox6_Alpt['Al']
    ox6_Alpt.loc[((HighSivsAl)&(~HighSi)), 'Al4']=ox6_Alpt['Al']
    # Else, equal to 2-Si
    ox6_Alpt.loc[((~HighSivsAl)&(~HighSi)), 'Al4']=2-ox6_Alpt['Si']

    # Al6, if if Al> Al4, Else, zero.
    ox6_Alpt['Al6']=ox6_Alpt['Al']-ox6_Alpt['Al4']
    Al4Grater=ox6_Alpt['Al4']>ox6_Alpt['Al']
    ox6_Alpt.loc[Al4Grater, 'Al6']=0

    # Fe 3
    Sum_1=(ox6_Alpt['Al4']+ox6_Alpt['Na']-ox6_Alpt['Al6']-ox6_Alpt['Cr']
    -(2*ox6_Alpt['Ti']))
    ox6_Alpt['Sum_1']=Sum_1

    # If Sum_1 is greater than 0, if Sum 1 is greater than Fe total, equal to AE total. Else set to sum 1.
    # So set default as zero
    ox6_Alpt['Fe3']=0
    Sum_1_gr0=Sum_1>0
    Sum1_grFe=Sum_1>ox6_Alpt['Fet']
    ox6_Alpt.loc[((Sum_1_gr0)&(Sum1_grFe)), 'Fe3']=ox6_Alpt['Fet']
    ox6_Alpt.loc[((Sum_1_gr0)&(~Sum1_grFe)), 'Fe3']=Sum_1

    # Fe2+ - If Fe > Fe3, Fe3-Fe, else 0

    ox6_Alpt['Fe2']=ox6_Alpt['Fet']-ox6_Alpt['Fe3']
    Fe3gr0=ox6_Alpt['Fe3']>ox6_Alpt['Fet']
    ox6_Alpt.loc[(Fe3gr0), 'Fe2']=0

    # Then for some bizzare reason, they calculate Fe2O3 again
    SumB=(ox6_Alpt['Si']+ox6_Alpt['Ti']+ox6_Alpt['Al']+ox6_Alpt['Cr']
          +ox6_Alpt['Fet']+ox6_Alpt['Mn']+ox6_Alpt['Mg']+ox6_Alpt['Ca']
          +ox6_Alpt['Na']+ox6_Alpt['K'])
    ox6_Alpt['Fe2O3_Cpx']=ox6_Alpt['Fe3']*53.233*(ox6_Alpt['SumB']/6)*1.5
    ox6_Alpt['FeO_Cpx']=ox6_Alpt['Fe2']*71.85*(ox6_Alpt['SumB']/6)
    ox6_Alpt['test']=(SumB/6)*1.5

    return ox6_Alpt




def later_step_Nimis_reclassification(cpx_comps_withFeO):
    '''
    This function takes cpx_comps which have been converted by "first step..." into FeO and Fe2O3,
    and recalculates cations.

    '''
    cat_6_ox=pd.DataFrame(data={'Si': cpx_comps_withFeO['SiO2_Cpx']/30.045,
                                 'Ti': cpx_comps_withFeO['TiO2_Cpx']/39.95,
                                 'Al': cpx_comps_withFeO['Al2O3_Cpx']/33.98,
                                'Cr': cpx_comps_withFeO['Cr2O3_Cpx']/50.673,
                                'Fe3': cpx_comps_withFeO['Fe2O3_Cpx']/53.233,
                                 'Fe2': cpx_comps_withFeO['FeO_Cpx']/71.85,
                                 'Mn': cpx_comps_withFeO['MnO_Cpx']/70.94,
                                 'Mg': cpx_comps_withFeO['MgO_Cpx']/40.32,
                                 'Ca': cpx_comps_withFeO['CaO_Cpx']/56.08,
                                 'Na': cpx_comps_withFeO['Na2O_Cpx']/61.982,
                                 'K': cpx_comps_withFeO['K2O_Cpx']/94.204,
                                })
    SumB=cat_6_ox.sum(axis=1)


    # Then, we multiply by Sum B
    ox6_Alpt=pd.DataFrame(data={'SumB': SumB,
                                'Si': 3*cat_6_ox['Si']/SumB,
                                 'Ti': 3*cat_6_ox['Ti']/SumB,
                                 'Al': 4*cat_6_ox['Al']/SumB,
                                'Cr': 4*cat_6_ox['Cr']/SumB,
                                'Fe3': 4*cat_6_ox['Fe3']/SumB,
                                 'Fe2': 6*cat_6_ox['Fe2']/SumB,
                                 'Mn': 6*cat_6_ox['Mn']/SumB,
                                 'Mg': 6*cat_6_ox['Mg']/SumB,
                                 'Ca': 6*cat_6_ox['Ca']/SumB,
                                 'Na': 12*cat_6_ox['Na']/SumB,
                                 'K': 12*cat_6_ox['K']/SumB,

                                })

        #  Now lets allocate Al
    # If Si cations >2, Al4 is 0. Set as default
    HighSi=ox6_Alpt['Si']>2
    ox6_Alpt['Al4']=0
    # Else
    # If 2-Si cat > Al cations, equatil to Al cations
    HighSivsAl=(2-ox6_Alpt['Si'])>ox6_Alpt['Al']
    ox6_Alpt.loc[((HighSivsAl)&(~HighSi)), 'Al4']=ox6_Alpt['Al']
    # Else, equal to 2-Si
    ox6_Alpt.loc[((~HighSivsAl)&(~HighSi)), 'Al4']=2-ox6_Alpt['Si']

    # Al6, if if Al> Al4, Else, zero.
    ox6_Alpt['Al6']=ox6_Alpt['Al']-ox6_Alpt['Al4']
    Al4Grater=ox6_Alpt['Al4']>ox6_Alpt['Al']
    ox6_Alpt.loc[Al4Grater, 'Al6']=0

    # Fe 3
    Sum_1=(ox6_Alpt['Al4']+ox6_Alpt['Na']-ox6_Alpt['Al6']-ox6_Alpt['Cr']
    -(2*ox6_Alpt['Ti']))
    ox6_Alpt['Sum_1']=Sum_1

    # If Sum_1 is greater than 0, if Sum 1 is greater than Fe total, equal to AE total. Else set to sum 1.
    # So set default as zero
    ox6_Alpt['Fe3_recalc']=0
    Sum_1_gr0=Sum_1>0
    Sum1_grFe=Sum_1>(ox6_Alpt['Fe2']+ ox6_Alpt['Fe3'])
    ox6_Alpt.loc[((Sum_1_gr0)&(Sum1_grFe)), 'Fe3_recalc']=(ox6_Alpt['Fe2']+ ox6_Alpt['Fe3'])
    ox6_Alpt.loc[((Sum_1_gr0)&(~Sum1_grFe)), 'Fe3_recalc']=Sum_1

    # Fe2+ - If Fe > Fe3, Fe3-Fe, else 0

    ox6_Alpt['Fe2_recalc']=(ox6_Alpt['Fe2']+ox6_Alpt['Fe3'])-ox6_Alpt['Fe3_recalc']
    Fe3gr0=ox6_Alpt['Fe3_recalc']>(ox6_Alpt['Fe2']+ox6_Alpt['Fe3'])
    ox6_Alpt.loc[(Fe3gr0), 'Fe2_recalc']=0

    # Then for some bizzare reason, they calculate Fe2O3 again
    SumB=(ox6_Alpt['Si']+ox6_Alpt['Ti']+ox6_Alpt['Al']+ox6_Alpt['Cr']
          +ox6_Alpt['Fe2']+ox6_Alpt['Fe3']+ox6_Alpt['Mn']+ox6_Alpt['Mg']+ox6_Alpt['Ca']
          +ox6_Alpt['Na']+ox6_Alpt['K'])
    ox6_Alpt['Fe2O3_Cpx']=ox6_Alpt['Fe3_recalc']*53.233*(ox6_Alpt['SumB']/6)*1.5
    ox6_Alpt['FeO_Cpx']=ox6_Alpt['Fe2_recalc']*71.85*(ox6_Alpt['SumB']/6)

    return ox6_Alpt



def combine_Nimis_Class_Steps(cpx_comps, iterations=7):

    '''
    This function takes cpx_comps, and does the whole Nimis reclassification routine,
    looping through the Fe allocation for the number of iterations specified (7 in the spreadsheet)


    '''

    cpx_comps_c=cpx_comps.copy()
    # First step, calculate initial FeO and Fe2O3
    class1=first_step_Nimis_reclassification(cpx_comps)
    cpx_comps_c['Fe2O3_Cpx']=class1['Fe2O3_Cpx']
    cpx_comps_c['FeO_Cpx']=class1['FeO_Cpx']

    # Second step (say 1 iteration), feed into reclassification algorithm

    for i in range(0, iterations):

        cpx_comps_c2=later_step_Nimis_reclassification(cpx_comps_c) # Gets you up to Salmon in excel
        cpx_comps_c['Fe2O3_Cpx']=cpx_comps_c2['Fe2O3_Cpx']
        cpx_comps_c['FeO_Cpx']=cpx_comps_c2['FeO_Cpx']

    cpx_comps_c2['Final_sum']=(cpx_comps_c2['Si']+cpx_comps_c2['Ti']+cpx_comps_c2['Al']
    +cpx_comps_c2['Cr']+cpx_comps_c2['Fe3']+cpx_comps_c2['Fe2']+cpx_comps_c2['Mn']
    +cpx_comps_c2['Mg']+cpx_comps_c2['Ca']+cpx_comps_c2['Na']+cpx_comps_c2['K'])

    cpx_comps_c3=pd.DataFrame(data={'Si': cpx_comps_c2['Si']*4/cpx_comps_c2['Final_sum'],
                                    'Ti': cpx_comps_c2['Ti']*4/cpx_comps_c2['Final_sum'],
                                    'Al': cpx_comps_c2['Al']*4/cpx_comps_c2['Final_sum'],
                                    'Cr': cpx_comps_c2['Cr']*4/cpx_comps_c2['Final_sum'],
                                    'Fe3': cpx_comps_c2['Fe3']*4/cpx_comps_c2['Final_sum'],
                                    'Fe2': cpx_comps_c2['Fe2']*4/cpx_comps_c2['Final_sum'],
                                    'Mn': cpx_comps_c2['Mn']*4/cpx_comps_c2['Final_sum'],
                                    'Mg': cpx_comps_c2['Mg']*4/cpx_comps_c2['Final_sum'],
                                    'Ca': cpx_comps_c2['Ca']*4/cpx_comps_c2['Final_sum'],
                                    'Na': cpx_comps_c2['Na']*4/cpx_comps_c2['Final_sum'],
                                    'K': cpx_comps_c2['K']*4/cpx_comps_c2['Final_sum'],
                                   })
        # If Si cations >2, Al4 is 0. Set as default
    HighSi=cpx_comps_c3['Si']>2
    cpx_comps_c3['Al4']=0
    # Else
    # If 2-Si cat > Al cations, equatil to Al cations
    HighSivsAl=(2-cpx_comps_c3['Si'])>cpx_comps_c3['Al']
    cpx_comps_c3.loc[((HighSivsAl)&(~HighSi)), 'Al4']=cpx_comps_c3['Al']
    # Else, equal to 2-Si
    cpx_comps_c3.loc[((~HighSivsAl)&(~HighSi)), 'Al4']=2-cpx_comps_c3['Si']

    # Al6, if if Al> Al4, Else, zero.
    cpx_comps_c3['Al6']=cpx_comps_c3['Al']-cpx_comps_c3['Al4']
    Al4Grater=cpx_comps_c3['Al4']>cpx_comps_c3['Al']
    cpx_comps_c3.loc[Al4Grater, 'Al6']=0



    return cpx_comps_c2, cpx_comps_c3

def calculate_P_Nimmis_BA(cpx_comps, iterations=7):
    Iter_final_nodiv, Iter_final=combine_Nimis_Class_Steps(cpx_comps, iterations)

    # Equals 2 if Si>2
    Iter_final['Si_recalc']=2
    Si_gr2=Iter_final['Si']>2
    # If less than 2, if Si+ Al <2, Si*2/(Si_Al), otherwise Si
    SiAl_l2=(Iter_final['Si']+Iter_final['Al'])<2
    Iter_final.loc[((~Si_gr2)&(SiAl_l2)), 'Si_recalc']=Iter_final['Si']*2/(Iter_final['Si']+Iter_final['Al'])
    Iter_final.loc[((~Si_gr2)&(~SiAl_l2)), 'Si_recalc']=Iter_final['Si']

    # Equals zero if 2-First Si<0
    Iter_final['Al_IV']=0
    two_minsSi_l0=(2-Iter_final['Si'])<0
    Iter_final.loc[((~two_minsSi_l0)&(SiAl_l2)), 'Al_IV']=Iter_final['Al']*2/(Iter_final['Si']+Iter_final['Al'])
    Iter_final.loc[((~two_minsSi_l0)&(~SiAl_l2)), 'Al_IV']=2-Iter_final['Si']
    Iter_final['AlM1_prov']=Iter_final['Al']-Iter_final['Al_IV']

    # Sum T sites - column AS
    Sum_nodiv_Si_Al_l2=(Iter_final_nodiv['Si']+Iter_final_nodiv['Al'])<2
    Iter_final['Sum_T']=Iter_final_nodiv['Si']+Iter_final_nodiv['Al']
    Si_Above2=Iter_final_nodiv['Si']>2
    Iter_final.loc[((~Sum_nodiv_Si_Al_l2)&(Si_Above2)), 'Sum_T']=Iter_final_nodiv['Si']
    Iter_final.loc[((~Sum_nodiv_Si_Al_l2)&(~Si_Above2)), 'Sum_T']=2


    # Sum M -
    Iter_final['Sum_M']=(Iter_final_nodiv['Ti']+Iter_final_nodiv['Al6']+Iter_final_nodiv['Cr']
    +Iter_final_nodiv['Fe3']+Iter_final_nodiv['Fe2']+Iter_final_nodiv['Mn']+Iter_final_nodiv['Mg']
    +Iter_final_nodiv['Ca']+Iter_final_nodiv['Na']+Iter_final_nodiv['K'])
    Iter_final['Sum_M']

    Iter_final['Sum_T_Norm']=Iter_final['Si_recalc']+Iter_final['Al_IV']
    Iter_final['Sum_M_prov']=(Iter_final['Ti']+Iter_final['Cr']+Iter_final['Fe3']
    +Iter_final['Fe2']+Iter_final['Mn']+Iter_final['Mg']+Iter_final['Ca']
    +Iter_final['Na']+Iter_final['K']+Iter_final['AlM1_prov'])


    Iter_final['Ti_recalc_Acells']=Iter_final['Ti']*2/Iter_final['Sum_M_prov']
    Iter_final['AlM1_recalc_Acells']=Iter_final['AlM1_prov']*2/Iter_final['Sum_M_prov']
    Iter_final['Cr_recalc_Acells']=Iter_final['Cr']*2/Iter_final['Sum_M_prov']
    Iter_final['Fe3_recalc_Acells']=Iter_final['Fe3']*2/Iter_final['Sum_M_prov']
    Iter_final['Fe2_recalc_Acells']=Iter_final['Fe2']*2/Iter_final['Sum_M_prov']
    Iter_final['Mn_recalc_Acells']=Iter_final['Mn']*2/Iter_final['Sum_M_prov']
    Iter_final['Mg_recalc_Acells']=Iter_final['Mg']*2/Iter_final['Sum_M_prov']
    Iter_final['Ca_recalc_Acells']=Iter_final['Ca']*2/Iter_final['Sum_M_prov']
    Iter_final['Mg_recalc_Acells']=Iter_final['Mg']*2/Iter_final['Sum_M_prov']
    Iter_final['Na_recalc_Acells']=Iter_final['Na']*2/Iter_final['Sum_M_prov']
    Iter_final['K_recalc_Acells']=Iter_final['K']*2/Iter_final['Sum_M_prov']
    Iter_final['CNM']=(Iter_final['Mn_recalc_Acells']+Iter_final['Na_recalc_Acells']
                    +Iter_final['Ca_recalc_Acells']+Iter_final['K_recalc_Acells'])
    Iter_final['R3']=(Iter_final['Ti_recalc_Acells']+Iter_final['AlM1_recalc_Acells']
                    +Iter_final['Cr_recalc_Acells']+Iter_final['Fe3_recalc_Acells'])
    Iter_final['KD']=(2.718282**((0.238*Iter_final['R3'])+(0.289*Iter_final['CNM'])-2.315))
    Iter_final['a']=Iter_final['KD']-1
    Iter_final['b']=((Iter_final['KD']*Iter_final['Mg_recalc_Acells'])-
                    (Iter_final['KD']*(1-Iter_final['CNM']))+Iter_final['Fe2_recalc_Acells']
                    +(1-Iter_final['CNM']))
    Iter_final['c']=-Iter_final['Fe2_recalc_Acells']*(1-Iter_final['CNM'])
    Iter_final['cat1**']=(-Iter_final['b']+np.sqrt((Iter_final['b']*Iter_final['b'])
    -(4*Iter_final['a']*Iter_final['c'])))/(2*Iter_final['a'])

    Iter_final['cat1**']

    # Then need to allocate FeM2
    # If Fe2>0 and Mg>0, 1-CNM, (set as default)
    Fe2Mg_g0=(Iter_final['Fe2_recalc_Acells']>0)&(Iter_final['Mg_recalc_Acells']>0)&(1-Iter_final['CNM']>0)
    Iter_final['FeM2']=Iter_final['cat1**']
    # Else, if that isn't true, has funny statement if 1-VNM<0, and 1-CNM=0, set as zero (this statement
    #cant ever be true, so move on, if Fe2>0 and Mg=0, set as 1-CNM, else set as zero
    Fe2_g0_Mg_ez=(Iter_final['Fe2_recalc_Acells']>0)&(Iter_final['Mg_recalc_Acells']==0)
    Iter_final.loc[((~Fe2Mg_g0)&(Fe2_g0_Mg_ez)), 'FeM2']=1-Iter_final['CNM']
    Iter_final.loc[((~Fe2Mg_g0)&(~Fe2_g0_Mg_ez)), 'FeM2']=0


    # Then need to allocate Mg M2. If the Fe2Mg_g0 if statement is true:
    Iter_final['MgM2']=1-Iter_final['CNM']-Iter_final['FeM2']
    # Also then has impossible zero statement, so comes down to same if as above
    Iter_final.loc[((~Fe2Mg_g0)&(Fe2_g0_Mg_ez)), 'MgM2']=1-Iter_final['CNM']
    Iter_final.loc[((~Fe2Mg_g0)&(~Fe2_g0_Mg_ez)), 'MgM2']=0


    # Need to allocate Ca M2. If (1-CNM)>0, Ca content.
    Iter_final['CaM2']=Iter_final['Ca_recalc_Acells']
    OneMinusCNM_g0=(1-Iter_final['CNM'])>0
    Iter_final.loc[~OneMinusCNM_g0, 'CaM2']=(1-Iter_final['Na_recalc_Acells']
    -Iter_final['Ca_recalc_Acells']-Iter_final['K_recalc_Acells']-Iter_final['Mn_recalc_Acells'])
    Iter_final['CaM2']

    # Need to allocate Na M2. If 1-CNM>0, equal to Na + K
    Iter_final['NaM2']=(Iter_final['Na_recalc_Acells']+Iter_final['K_recalc_Acells'])
    # Second if statement.
    Ca_g0=Iter_final['Ca_recalc_Acells']>0
    Iter_final.loc[((OneMinusCNM_g0)&(~Ca_g0)), 'NaM2']=(1-Iter_final['Mn_recalc_Acells'])

    # Need to allocate FeM1 and MgM1
    Iter_final['FeM1']=Iter_final['Fe2_recalc_Acells']-Iter_final['FeM2']
    Iter_final['MgM1']=Iter_final['Mg_recalc_Acells']-Iter_final['MgM2']

    Iter_final['Vcell']=(
    (Iter_final['FeM2']**2*(-12.741))+(Iter_final['FeM2']*432.56)+(Iter_final['Mn_recalc_Acells']*428.03)
    +(Iter_final['MgM2']**2*(-28.652))+(Iter_final['MgM2']*431.72)
    +(Iter_final['CaM2']*439.97)+(Iter_final['NaM2']*419.68)+(Iter_final['Ti_recalc_Acells']*11.794)
    +(Iter_final['AlM1_recalc_Acells']*-18.375)+(Iter_final['Fe3_recalc_Acells']*9.107)+(Iter_final['FeM1']*11.864)+(Iter_final['Cr_recalc_Acells']*-1.4925)
    )








    Iter_final['VM1']=(
    (Iter_final['FeM2']**2*1.1661)+(Iter_final['FeM2']*11.885)
    +(Iter_final['Mn_recalc_Acells']*12.038)+(Iter_final['MgM2']**2*2.4335)
    +(Iter_final['MgM2']*11.432)+(Iter_final['CaM2']*11.931)
    +(Iter_final['NaM2']*11.288)+(Iter_final['Ti_recalc_Acells']*-1.0864)
    +(Iter_final['AlM1_recalc_Acells']*-2.029)+(Iter_final['Fe3_recalc_Acells']*-0.41726)
    +(Iter_final['FeM1']*0.813)+(Iter_final['Cr_recalc_Acells']*-0.8001)+(Iter_final['AlM1_prov']*-0.30853)
    )

    Iter_final['P_kbar_calc']=771.475-(1.323*Iter_final['Vcell'])-(16.064*Iter_final['VM1'])

    cols_to_move = ['P_kbar_calc', 'Vcell', 'VM1']
    Iter_final = Iter_final[cols_to_move + [
        col for col in Iter_final.columns if col not in cols_to_move]]

    return Iter_final



