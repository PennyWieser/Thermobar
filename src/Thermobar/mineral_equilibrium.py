import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
import ternary
from scipy import interpolate
from Thermobar.core import *

## Equilibrium things for Olivine
def calculate_eq_olivine(Kd, *, Liq_Mgno):
    '''calculates equilibrium forsterite contents based on inputtted liquid Mg# and Kd Fe-Mg
     '''
    return 1 / ((Kd / Liq_Mgno) + (1 - Kd))

def calculate_ol_fo(ol_comps):
    Fo=(ol_comps['MgO_Ol']/40.3044)/((ol_comps['MgO_Ol']/40.3044)+(ol_comps['FeOt_Ol']/71.844))
    return Fo

def calculate_cpx_mgno(cpx_comps):
    Mgno=(cpx_comps['MgO_Cpx']/40.3044)/((cpx_comps['MgO_Cpx']/40.3044)+(cpx_comps['FeOt_Cpx']/71.844))
    return Mgno

def calculate_opx_mgno(opx_comps):
    Mgno=(cpx_comps['MgO_Opx']/40.3044)/((cpx_comps['MgO_Opx']/40.3044)+(cpx_comps['FeOt_Opx']/71.844))
    return Mgno

def calculate_liq_mgno(liq_comps, Fe3Fet_Liq=None):
    liq_comps_c=liq_comps.copy()
    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq']=Fe3Fet_Liq
    Mgno=(liq_comps['MgO_Ol']/40.3044)/((liq_comps['MgO_Ol']/40.3044)+(liq_comps_c['Fe3Fet_Liq']*liq_comps['FeOt_Ol']/71.844))
    return Fo

def calculate_toplis2005_kd(X_fo, *, SiO2_mol, Na2O_mol, K2O_mol, P, H2O, T):
    '''
    calculates olivine-liq Kd Fe-Mg using the expression of Toplis, 2005.
    '''
    SiO2_mol = 100 * SiO2_mol
    Na2O_mol = 100 * Na2O_mol
    K2O_mol = 100 * K2O_mol
    P = P * 1000
    R = 8.3144626181
    PSI_SiO2_60plus = (11 - 5.5 * (100 / (100 - SiO2_mol))) * \
        np.exp(-0.13 * (K2O_mol + Na2O_mol))
    PSI_SiO2_60minus = (0.46 * (100 / (100 - SiO2_mol)) - 0.93) * \
        (K2O_mol + Na2O_mol) + (-5.33 * (100 / (100 - SiO2_mol)) + 9.69)
    Adjusted_Si_Ksparalis_60plus = SiO2_mol + \
        PSI_SiO2_60plus * (Na2O_mol + K2O_mol)
    Adjusted_Si_Ksparalis_60minus = SiO2_mol + \
        PSI_SiO2_60minus * (Na2O_mol + K2O_mol)
    Adjusted_Si_Ksparalis_H2O_60plus = Adjusted_Si_Ksparalis_60plus + 0.8 * H2O
    Adjusted_Si_Ksparalis_H2O_60minus = Adjusted_Si_Ksparalis_60minus + 0.8 * H2O


    # Issues with object
    X_fo = X_fo.astype(float) if isinstance(X_fo, pd.Series) else float(X_fo)
    R = R.astype(float) if isinstance(R, pd.Series) else float(R)
    T = T.astype(float) if isinstance(T, pd.Series) else float(T)
    P = P.astype(float) if isinstance(P, pd.Series) else float(P)

    Kd_Toplis_60plus = np.exp((-6766 / (R * T) - 7.34 / R) + np.log(0.036 * Adjusted_Si_Ksparalis_H2O_60plus - 0.22)
                              + (3000 * (1 - 2 * X_fo)) / (R * T) + (0.035 * (P - 1)) / (R * T))
    Kd_Toplis_60minus = np.exp((-6766 / (R * T) - 7.34 / R) + np.log(0.036 * Adjusted_Si_Ksparalis_H2O_60minus - 0.22)
                               + (3000 * (1 - 2 * X_fo)) / (R * T) + (0.035 * (P - 1)) / (R * T))


    if isinstance(SiO2_mol, int) or isinstance(SiO2_mol, float):
        if SiO2_mol > 60:
            Kd_Toplis = Kd_Toplis_60plus
        if SiO2_mol < 60:
            Kd_Toplis = Kd_Toplis_60minus
    else:
        Kd_Toplis = np.zeros(len(SiO2_mol), dtype=float)

        for i in range(0, len(SiO2_mol)):
            if SiO2_mol[i] > 60:
                Kd_Toplis[i] = Kd_Toplis_60plus[i]
            if SiO2_mol[i] < 60:
                Kd_Toplis[i] = Kd_Toplis_60minus[i]
    return Kd_Toplis


def calculate_eq_ol_content(liq_comps, Kd_model, ol_comps=None, T=None, P=None,
Fe3Fet_Liq=None, ol_fo=None, H2O_Liq=None, logfo2=None):
    '''calculates equilibrium forsterite contents based on inputtted liquid compositions.


   Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Ol, MgO_Ol etc.


    Kd_model: str, int or float
        Specify which Kd model you wish to use.

        int or float: e.g. Kd=0.35, will return that number.

        "Shea2022": Uses 0.335+-0.1 (not sensitive to P, T, or ol Fo content. For hawaiian tholeiites)

        "Roeder1970": uses Kd=0.3+0.03 (Not sensitive to P, T, or Ol Fo content)

        "Matzen2011": uses Kd=0.34+0.012 (Not sensitive to P, T, or Ol Fo content)

        "Toplis2005": calculates Kd based on melt SiO2, Na2O, K2O, P, T, H2O, Ol Fo content.
        Users can specify a ol_fo content, or the function iterates Kd and Fo and returns both.

        "Putirka2016":
        Uses equation 8a, 8b and 8c of Putirka (2016)
        These are recomended when the proportion of Fe2O3 is known.
        8a=0.33+-0.04 (constant)
        8b= function of P, Si, Na and K
        8c= function of Si, Na and K

        Also uses equation 9a and 9b of Putirka, which are
        designed for FeOt. 9a is 0.29+-0.051, 9b is a function of Si, P, Na and
        K, and fo2.


        "All": Returns outputs for all models

    Fe3FeT: optional, float or int.
        overwrites Fe3Fet_Liq in liq_comps DataFrame

    Additional required inputs for Toplis, 2005:
        P: Pressure in kbar
        T: Temperature in Kelvin
        H2O: melt H2O content
        Optional:
            ol_fo: If specify Fo content (decimal, 0-1), calculates Kd
            Else, will iterate to find equilibrium Ol content and Kd.

    Returns
    -------
    pandas DataFrame
        returns equilibrium olivine contents (+- sigma for Roeder and Matzen).
        For Toplis, returns Kd-Ol Fo pair if an olivine-forsterite content wasn't specified

    '''
    liq_comps_c=liq_comps.copy()
    if ol_comps is not None:
        ol_comps['Fo_meas']=calculate_ol_fo(ol_comps)
    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq
    if H2O_Liq is not None:
        liq_comps_c['H2O_Liq'] = H2O_Liq

    liq = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
    Mgno = liq['Mg_Number_Liq_Fe3']
    Mgno_noFe3 = liq['Mg_Number_Liq_NoFe3']

    if isinstance(Kd_model, int) or isinstance(Kd_model, float):
        Eq_ol=1 / ((Kd_model / Mgno) + (1 - Kd_model))

        Kd_out=pd.Series(Eq_ol)

        return Kd_out
    else:

        if Kd_model=="Shea2022" or Kd_model == "All":
            Eq_ol_0335 = 1 / ((0.335 / Mgno) + (1 - 0.335))
            Eq_ol_0325 = 1 / ((0.345 / Mgno) + (1 - 0.345))
            Eq_ol_0345 = 1 / ((0.325 / Mgno) + (1 - 0.325))
            Kd_out_shea = pd.DataFrame(data={'Eq Fo (Shea, Kd=0.335)': Eq_ol_0335,
                                    'Eq Fo (Shea, Kd=0.325)': Eq_ol_0325, 'Eq Fo (Shea, Kd=0.345)': Eq_ol_0345})

        if Kd_model == "Roeder1970" or Kd_model == "All":
            Eq_ol_03 = 1 / ((0.3 / Mgno) + (1 - 0.3))
            Eq_ol_027 = 1 / ((0.27 / Mgno) + (1 - 0.27))
            Eq_ol_033 = 1 / ((0.33 / Mgno) + (1 - 0.33))
            Kd_out_ro = pd.DataFrame(data={'Eq Fo (Roeder, Kd=0.3)': Eq_ol_03,
                                    'Eq Fo (Roeder, Kd=0.33)': Eq_ol_033, 'Eq Fo (Roeder, Kd=0.27)': Eq_ol_027})

        if Kd_model == "Matzen2011" or Kd_model == "All":
            Eq_ol_034 = 1 / ((0.34 / Mgno) + (1 - 0.34))
            Eq_ol_032 = 1 / ((0.328 / Mgno) + (1 - 0.328))
            Eq_ol_035 = 1 / ((0.352 / Mgno) + (1 - 0.352))
            Kd_out_mat = pd.DataFrame(data={'Eq Fo (Matzen, Kd=0.34)': Eq_ol_034,
                                    'Eq Fo (Matzen, Kd=0.352)': Eq_ol_035, 'Eq Fo (Matzen, Kd=0.328)': Eq_ol_032})


        if Kd_model =="Putirka2016" or Kd_model == "All":

            Kd_8a=0.33
            Eq_ol_8a=1 / ((Kd_8a / Mgno) + (1 - Kd_8a))

            Kd_8a_m_1sigma=0.33-0.044
            Eq_ol_8a_m_1sigma=1 / ((Kd_8a_m_1sigma / Mgno) + (1 - Kd_8a_m_1sigma))

            Kd_8a_p_1sigma=0.33+0.044
            Eq_ol_8a_p_1sigma=1 / ((Kd_8a_p_1sigma / Mgno) + (1 - Kd_8a_p_1sigma))



            Kd_8c=(0.25 + 0.0018*liq_comps_c['SiO2_Liq']
            -3.27*10**(-4)*(liq_comps_c['Na2O_Liq']+liq_comps_c['K2O_Liq'])**2)
            Eq_ol_8c=1 / ((Kd_8c / Mgno) + (1 - Kd_8c))

            Kd_9a=0.29
            Eq_ol_9a=1 / ((Kd_9a / Mgno_noFe3) + (1 - Kd_9a))

            Kd_9a_m_1sigma=0.29-0.051
            Eq_ol_9a_m_1sigma=1 / ((Kd_9a_m_1sigma / Mgno_noFe3 ) + (1 - Kd_9a_m_1sigma))

            Kd_9a_p_1sigma=0.29+0.051
            Eq_ol_9a_p_1sigma=1 / ((Kd_9a_p_1sigma / Mgno_noFe3 ) + (1 - Kd_9a_p_1sigma))



            Kd_out_Put=pd.DataFrame(
            data={
            'Eq Fo (Putirka 8a Fe2, Kd=0.33)': Eq_ol_8a,
            'Eq Fo (Putirka 8a Fe2, Kd=0.33-0.044)': Eq_ol_8a_m_1sigma,
            'Eq Fo (Putirka 8a Fe2, Kd=0.33+0.044)': Eq_ol_8a_p_1sigma,
            'Eq Fo (Putirka 8a Fe2, Kd=0.33+0.044)': Eq_ol_8a_p_1sigma,
            'Calc Kd (Putirka 8c, Fe2)': Kd_8c,
            'Eq Fo (Putirka 8c Fe2)': Eq_ol_8c,
            'Eq Fo (Putirka 9a Fet, Kd=0.29)': Eq_ol_9a,
            'Eq Fo (Putirka 9a Fet, Kd=0.29-0.051)': Eq_ol_9a_m_1sigma,
            'Eq Fo (Putirka 9a Fet, Kd=0.29+0.051)': Eq_ol_9a_m_1sigma})

            if P is None:
                w.warn(
                    'Putirka (2016) Kd models equation 8b and 9b are P-dependent you need to enter a P in kbar to get these outputs')



            if P is not None:
                Kd_8b=(0.21+0.008*(P/10) + 0.0025*liq_comps_c['SiO2_Liq']
                -3.63*10**(-4)*(liq_comps_c['Na2O_Liq']+liq_comps_c['K2O_Liq'])**2)
                Eq_ol_8b=1 / ((Kd_8b / Mgno) + (1 - Kd_8b))
                Kd_out_Put['Calc Kd (Putirka 8b, Fe2)']=Kd_8b
                Kd_out_Put['Eq Fo (Putirka 8b Fe2)']=Eq_ol_8b


                if logfo2 is not None:
                    Kd_9b=(0.0583+0.00252*liq_comps_c['SiO2_Liq']+ 0.028*(P/10)
                    -0.0091* (liq_comps_c['Na2O_Liq']+liq_comps_c['K2O_Liq'])
                    -0.013383*logfo2)

                    Eq_ol_9b=1 / ((Kd_9b / Mgno) + (1 - Kd_9b))
                    Kd_out_Put['Calc Kd (Putirka 9b, Fet)']=Kd_9b
                    Kd_out_Put['Eq Fo (Putirka 9b Fet)']=Eq_ol_9b


        if Kd_model == "Toplis2005" or Kd_model == "All":
            if P is None:
                raise Exception(
                    'The Toplis Kd model is P-dependent, please enter P in kbar into the function')
            if T is None:
                raise Exception(
                    'The Toplis Kd model is T-dependent, please enter T in Kelvin into the function')

            mol_perc = calculate_anhydrous_mol_fractions_liquid(liq_comps_c)
            SiO2_mol = mol_perc['SiO2_Liq_mol_frac']
            Na2O_mol = mol_perc['Na2O_Liq_mol_frac']
            K2O_mol = mol_perc['K2O_Liq_mol_frac']
            H2O_Liq = liq_comps_c['H2O_Liq']
            Kd_func = partial(calculate_toplis2005_kd, SiO2_mol=SiO2_mol,
                            Na2O_mol=Na2O_mol, K2O_mol=K2O_mol, P=P, H2O=H2O_Liq, T=T)
            if ol_fo is not None or ol_comps is not None:
                if ol_fo is not None and ol_comps is None:
                    Kd_calc = Kd_func(ol_fo)
                    Ol_calc = 1 / ((Kd_calc / Mgno) + (1 - Kd_calc))
                if ol_comps is not None:
                    Kd_calc = Kd_func(ol_comps['Fo_meas'])
                    Ol_calc = 1 / ((Kd_calc / Mgno) + (1 - Kd_calc))

                Kd_out_top = pd.DataFrame(
                        data={'Kd (Toplis, input Fo)': Kd_calc, 'Eq Fo (Toplis, input Fo)': Ol_calc})

            else:
                Eq_ol_func = partial(calculate_eq_olivine, Liq_Mgno=Mgno)
                iterations = 20
                Eq_ol_guess = 0.95
                for _ in range(iterations):
                    Kd_Guess = Kd_func(Eq_ol_guess)
                    Eq_ol_guess = Eq_ol_func(Kd_Guess)
                    Kd_out_top = pd.DataFrame(
                        data={'Kd (Toplis, Iter)': Kd_Guess, 'Eq Fo (Toplis, Iter)': Eq_ol_guess})

        if Kd_model == "All":
            Kd_out = pd.concat([Kd_out_shea, Kd_out_ro, Kd_out_mat, Kd_out_top, Kd_out_Put], axis=1)
        if Kd_model == "Roeder1970":
            Kd_out=Kd_out_ro
        if Kd_model == "Matzen2011":
            Kd_out=Kd_out_mat
        if Kd_model == "Toplis2005":
            Kd_out=Kd_out_top
        if Kd_model == "Putirka2016":
            Kd_out=Kd_out_Put
        if Kd_model=='Shea2022':
            Kd_out=Kd_out_shea

        if ol_comps is not None:
            Kd_out['Fo_meas']=ol_comps['Fo_meas']

        Kd_out.insert(0, 'Mg#_Liq_Fe2', Mgno)
        Kd_out.insert(1, 'Mg#_Liq_Fet', Mgno_noFe3)


        return Kd_out

## Same for Orhopyroxene
def calculate_eq_px_content(liq_comps,
Fe3Fet_Liq=None, px='Opx'):
    '''calculates equilibrium pyroxene contents based on inputtted liquid compositions.


   Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Ol, MgO_Ol etc.

    px: 'Opx' or 'Cpx'

    Fe3FeT: optional, float or int.
        overwrites Fe3Fet_Liq in liq_comps DataFrame



    Returns
    -------
    pandas DataFrame
        returns equilibrium Px Mg#s (+- sigma) from Putirka 2008

    '''
    liq_comps_c=liq_comps.copy()
    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq'] = Fe3Fet_Liq


    liq = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
    Mgno = liq['Mg_Number_Liq_Fe3']
    Mgno_noFe3 = liq['Mg_Number_Liq_NoFe3']




    if px == 'Opx':

        # These values are 0.29+0-.06, which is from Putirka, He also gives an expression in terms of the si content of the liquid.

        Eq_023 = 1 / ((0.23 / Mgno) + (1 - 0.23))
        Eq_029 = 1 / ((0.29 / Mgno) + (1 - 0.29))
        Eq_035 = 1 / ((0.35 / Mgno) + (1 - 0.35))


        # Or calculating as a function of the Si content.
        cat_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
        Si_mean_frac = np.nanmean(cat_frac['Si_Liq_cat_frac'])
        Kd = 0.4805 - 0.3733 * Si_mean_frac
        Eq_Opx = 1 / ((Kd / Mgno) + (1 - Kd))
        Kd_p_1_s = Kd + 0.06
        Kd_m_1_s = Kd - 0.06
        Eq_Opx_p1sigma = 1 / ((Kd_p_1_s / Mgno) + (1 - Kd_p_1_s))
        Eq_Opx_m1sigma = 1 / ((Kd_m_1_s / Mgno) + (1 - Kd_m_1_s))


        Kd_out= pd.DataFrame(data={'Kd_XSi_P2008': Kd,
        'Eq_Opx_Mg# (Kd_XSi_P2008)': Eq_Opx,
         'Eq_Opx_Mg# (Kd_XSi_P2008)+0.06': Eq_Opx_p1sigma,
        'Eq_Opx_Mg# (Kd_XSi_P2008)-0.06': Eq_Opx_m1sigma,
        'Eq_Opx_Mg# (Kd=0.23)': Eq_023,
        'Eq_Opx_Mg# (Kd=0.29)': Eq_029,
        'Eq_Opx_Mg# (Kd=0.35)': Eq_035

        })





        Kd_out.insert(0, 'Mg#_Liq_Fe2', Mgno)
        Kd_out.insert(1, 'Mg#_Liq_Fet', Mgno_noFe3)


        return Kd_out

    if px == 'Cpx':
        raise TypeError('Penny still needs to add this, if you need it , please message me!')



def calculate_ol_rhodes_diagram_lines(
        Min_Mgno, Max_Mgno, KdMin=None, KdMax=None):
    '''
    Input minimum and maximum liquid Mg#, calculates lines for equilibrium Fo content using Roeder and Emslie (1970) and Matzen (2011) Kd values.

   Parameters
    -------

       Min_Mgno: float or int
            Min liquid Mg# you want equilibrium lines for

        Max_Mgno: float or int
            Max liquid Mg# you want equilibrium lines for

        KdMin: float
            Optional. Also returns line for a user-specified Minimum Kd.
        KdMax: float
             Optional. Also returns line for a user-specified Maximum Kd.

    Returns
        Mg#_Liq (100 points between Min)
        Eq_OlRoeder (Kd=0.3): Line calculated for Kd=0.3 (Roeder and Emslie, 1970 preferred value)
        Eq_OlRoeder (Kd=0.33): Line calculated for Kd=0.33 (Roeder and Emslie, 1970 +1 sigma)
        Eq_OlRoeder (Kd=0.27): Line calculated for Kd=0.27 (Roeder and Emslie, 1970 -1 sigma)
        Eq_OlMatzen (Kd=0.34): Line calculated for Kd=0.34 (Matzen et al. 2011 preferred value)
        Eq_OlMatzen (Kd=0.328): Line calculated for Kd=0.328 (Matzen et al. 2011 - 1 sigma)
        Eq_OlMatzen (Kd=0.352): Line calculated for Kd=0.352 (Matzen et al. 2011 + 1 sigma)
    If user specifies KdMin and KdMax also returns:
        Eq_Ol_KdMax=KdMax, Eq_Ol_KdMin=KdMin

    '''
    Mgno = np.linspace(Min_Mgno, Max_Mgno, 100)

    Mgno = np.linspace(Min_Mgno, Max_Mgno, 100)
    Eq_Roeder_03 = 1 / ((0.3 / Mgno) + (1 - 0.3))
    Eq_Roeder_027 = 1 / ((0.27 / Mgno) + (1 - 0.27))
    Eq_Roeder_033 = 1 / ((0.33 / Mgno) + (1 - 0.33))
    Eq_Shea_0335 = 1 / ((0.335 / Mgno) + (1 - 0.335))
    Eq_Shea_0345 = 1 / ((0.345 / Mgno) + (1 - 0.345))
    Eq_Shea_0325 = 1 / ((0.325 / Mgno) + (1 - 0.325))


    Eq_ol_034 = 1 / ((0.34 / Mgno) + (1 - 0.34))
    Eq_ol_032 = 1 / ((0.328 / Mgno) + (1 - 0.328))
    Eq_ol_035 = 1 / ((0.352 / Mgno) + (1 - 0.352))
    Kd_out_mat = pd.DataFrame(data={'Mg#_Liq': Mgno, 'Eq_Ol_Fo_Roeder (Kd=0.3)': Eq_Roeder_03,
                                    'Eq_Ol_Fo_Roeder (Kd=0.27)': Eq_Roeder_027,
                                    'Eq_Ol_Fo_Roeder (Kd=0.33)': Eq_Roeder_033,
                                    'Eq_Ol_Fo_Matzen (Kd=0.34)': Eq_ol_034,
                                    'Eq_Ol_Fo_Matzen (Kd=0.328)': Eq_ol_032,
                                    'Eq_Ol_Fo_Matzen (Kd=0.352)': Eq_ol_035,
                                    'Eq_Ol_Fo_Shea (Kd=0.335)': Eq_Shea_0335,
                                    'Eq_Ol_Fo_Shea (Kd=0.345)': Eq_Shea_0345,
                                    'Eq_Ol_Fo_Shea (Kd=0.325)': Eq_Shea_0325,

                                    })

    if KdMin is not None and KdMax is not None:
        Eq_ol_KdMin = 1 / ((KdMin / Mgno) + (1 - KdMin))
        Eq_ol_KdMax = 1 / ((KdMax / Mgno) + (1 - KdMax))
        Kd_out_mat2 = pd.DataFrame(data={'Eq_Ol_Fo (KdMin=' + str(
            KdMin) + ')': Eq_ol_KdMin, 'Eq_Ol_Fo (KdMax=' + str(KdMax) + ')': Eq_ol_KdMax})
        Kd_out_mat = pd.concat([Kd_out_mat, Kd_out_mat2], axis=1)

    return Kd_out_mat

## Equilibrium things for Pyroxene

def calculate_opx_rhodes_diagram_lines(
        Min_Mgno, Max_Mgno, T=None, KdMin=None, KdMax=None, liq_comps=None):
    '''
    Input minimum and maximum liquid Mg#, calculates lines for equilibrium
    Opx Mg# content using a variety of choices for Kd Fe-Mg.

   Parameters
    -------
    Min_Mgno: float or int.
        Min liquid Mg# you want equilibrium lines for

    Max_Mgno: float or int.
        Max liquid Mg# you want equilibrium lines for


    By default, returns Mg#s for 0.29+-0.06 (Putirka). Can get other outputs as
    well using:

        KdMin: float. Optional.
            Also returns line for a user-specified Minimum Kd.

        KdMax: float. Optional.
            Also returns line for a user-specified Maximum Kd.

        liq_comps: pandas.DataFrame. Optional
            Uses average cation fraction of XSi in the liquid to
            calculate Kd Fe-Mg using the expression = 0.4805 - 0.3733 XSi (Putirka, 2008)

   Returns
    -------
        Mg#_Liq (100 points between Min) and equilibrium Opx compositions depending on inputs.
        Returns headings corresponding to options selected above.

    '''

    Mgno = np.linspace(Min_Mgno, Max_Mgno, 100)

    Mgno = np.linspace(Min_Mgno, Max_Mgno, 100)
    Eq_023 = 1 / ((0.23 / Mgno) + (1 - 0.23))
    Eq_029 = 1 / ((0.29 / Mgno) + (1 - 0.29))
    Eq_035 = 1 / ((0.35 / Mgno) + (1 - 0.35))
    Kd_out_mat_s = pd.DataFrame(data={'Eq_Opx_Mg# (Kd=0.23)': Eq_023,
                                    'Eq_Opx_Mg# (Kd=0.29)': Eq_029,
                                    'Eq_Opx_Mg# (Kd=0.35)': Eq_035})
    Kd_out_mat = Kd_out_mat_s

    if KdMin is not None and KdMax is not None:
        Eq_ol_KdMin = 1 / ((KdMin / Mgno) + (1 - KdMin))
        Eq_ol_KdMax = 1 / ((KdMax / Mgno) + (1 - KdMax))
        Kd_out_mat_MM = pd.DataFrame(data={'Eq_Opx_Mg# (KdMin=' + str(
            KdMin) + ')': Eq_ol_KdMin, 'Eq_Opx_Mg# (KdMax=' + str(KdMax) + ')': Eq_ol_KdMax})
        Kd_out_mat = pd.concat([Kd_out_mat, Kd_out_mat_MM], axis=1)

    if liq_comps is not None:
        liq_comps_c=liq_comps.copy()
        cat_frac = calculate_anhydrous_cat_fractions_liquid(liq_comps_c)
        Si_mean_frac = np.nanmean(cat_frac['Si_Liq_cat_frac'])
        Kd = 0.4805 - 0.3733 * Si_mean_frac
        Eq_Opx = 1 / ((Kd / Mgno) + (1 - Kd))
        Kd_p_1_s = Kd + 0.06
        Kd_m_1_s = Kd - 0.06
        Eq_Opx_p1sigma = 1 / ((Kd_p_1_s / Mgno) + (1 - Kd_p_1_s))
        Eq_Opx_m1sigma = 1 / ((Kd_m_1_s / Mgno) + (1 - Kd_m_1_s))
        Kd_out_mat_s = pd.DataFrame(data={'Kd_XSi_P2008': Kd, 'Eq_Opx_Mg# (Kd_XSi_P2008)':
        Eq_Opx, 'Eq_Opx_Mg# (Kd_XSi_P2008)+0.06': Eq_Opx_p1sigma,
        'Eq_Opx_Mg# (Kd_XSi_P2008)-0.06': Eq_Opx_m1sigma})

        Kd_out_mat = pd.concat([Kd_out_mat, Kd_out_mat_s], axis=1)



    Kd_out_mat.insert(0, "Mg#_Liq", Mgno)
    return Kd_out_mat


def calculate_cpx_rhodes_diagram_lines(
        Min_Mgno, Max_Mgno, T=None, KdMin=None, KdMax=None):
    '''
    Input minimum and maximum liquid Mg#, calculates lines for equilibrium Cpx Mg# contents based on user-specified Kd Fe-Mg options.

   Parameters
    -------


        Min_Mgno: float or int.
            Min liquid Mg# you want equilibrium lines for
        Max_Mgno: float or int.
            Max liquid Mg# you want equilibrium lines for

        By default, returns lines calculated using 0.28+-0.08 (Putirka, 2008).
        Can get other outputs as well using:

        T: float or int (optional)
            Temperature in Kelvin. returns lines calculated using Kd from T-sensitive eq 35 of Putirka (2008) (as well as +-0.08 error bounds)
        KdMin: float (optional)
            calculates equilibrium line for a user-specified Minimum Kd.
        KdMax: float (optional)
            calculates equilibrium line for a user-specified Minimum Kd
    Returns:
    -------
        Mg#_Liq (100 points between Min_Mgno and Max_Mgno), and a variety of equilibrium Cpx Mg#s


    '''

    Mgno = np.linspace(Min_Mgno, Max_Mgno, 100)
    Eq_02 = 1 / ((0.2 / Mgno) + (1 - 0.2))
    Eq_028 = 1 / ((0.28 / Mgno) + (1 - 0.28))
    Eq_036 = 1 / ((0.36 / Mgno) + (1 - 0.36))
    Kd_out_mat = pd.DataFrame(data={'Eq_Cpx_Mg# (Kd=0.28)': Eq_028,
                                        'Eq_Cpx_Mg# (Kd=0.2)': Eq_02, 'Eq_Cpx_Mg# (Kd=0.36)': Eq_036})


    if isinstance(T, int) or isinstance(T, float):
        Kd = np.exp(-0.107 - 1719 / T)
        Eq_Cpx = 1 / ((Kd / Mgno) + (1 - Kd))
        Kd_p_1_s = Kd + 0.08
        Kd_m_1_s = Kd - 0.08
        Eq_Cpx_p1sigma = 1 / ((Kd_p_1_s / Mgno) + (1 - Kd_p_1_s))
        Eq_Cpx_m1sigma = 1 / ((Kd_m_1_s / Mgno) + (1 - Kd_m_1_s))
        Kd_out_mat_s = pd.DataFrame(data={'Kd_Eq35_P2008': Kd, 'Eq_Cpx_Mg# (Kd from Eq 35 P2008)':
        Eq_Cpx, 'Eq_Cpx_Mg# (Eq 35 P2008)+0.08': Eq_Cpx_p1sigma,
        'Eq_Cpx_Mg# (Eq 35 P2008)-0.08': Eq_Cpx_m1sigma})
        Kd_out_mat=pd.concat([Kd_out_mat, Kd_out_mat_s], axis=1)


    if KdMin is not None and KdMax is not None:
        Eq_cpx_KdMin = 1 / ((KdMin / Mgno) + (1 - KdMin))
        Eq_cpx_KdMax = 1 / ((KdMax / Mgno) + (1 - KdMax))
        Kd_out_mat_MM = pd.DataFrame(data={'Eq_Cpx_Mg# (KdMin=' + str(
            KdMin) + ')': Eq_cpx_KdMin, 'Eq_Cpx_Mg# (KdMax=' + str(KdMax) + ')': Eq_cpx_KdMax})
        Kd_out_mat = pd.concat([Kd_out_mat, Kd_out_mat_MM], axis=1)

    Kd_out_mat.insert(0, "Mg#_Liq", Mgno)
    return Kd_out_mat


def calculate_eq_cpx_Mg_number(liq_comps, Fe3Fet_Liq=None, Kd_model=None, T=None):
    '''calculates equilibrium Cpx Mg#s  based on inputtted liquid compositions.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Ol, MgO_Ol etc.


    Kd_model: str, int or float
        Specify which Kd model you wish to use.
        "Putirka2008": uses eq 35 of Putirka (2008), function of T.
        "Masotta2013": Uses Masotta (2013), function of T, an Na and K in the melt.
        if int or float, uses that Kd.

    Returns
    -------
    DataFrame with equilibrium Mg#s, with column headings corresponding to model choosen.
  '''
    liq_comps_c=liq_comps.copy()

    if Fe3Fet_Liq is not None:
        liq_comps_c['Fe3Fet_Liq']=Fe3Fet_Liq

    cat_fracs=calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
    Mgno = cat_fracs['Mg_Number_Liq_Fe3']
    Mgno_Fet = cat_fracs['Mg_Number_Liq_NoFe3']

    if Kd_model == "Putirka2008":
        Kd=np.exp(-0.107 - 1719 / T)
        Eq_Mgno_Fe2 = 1 / (( Kd / Mgno) + (1 - Kd))
        Eq_Mgno_Fe2_m1sig = 1 / (( Kd / Mgno) + (1 - (Kd-0.08)))
        Eq_Mgno_Fe2_p1sig = 1 / (( Kd / Mgno) + (1 - (Kd+0.08)))
        Eq_Mgno_Fet = 1 / (( Kd / Mgno_Fet) + (1 - Kd))
        Eq_Mgno_Fet_m1sig = 1 / (( Kd /Mgno_Fet) + (1 - (Kd-0.08)))
        Eq_Mgno_Fet_p1sig = 1 / (( Kd / Mgno_Fet) + (1 - (Kd+0.08)))
        Kd_out = pd.DataFrame(data={'Eq Mg# (Putirka 2008, Fe2)': Eq_Mgno_Fe2,
                                       'Eq Mg# (Putirka 2008 -0.08, Fe2)': Eq_Mgno_Fe2_m1sig,
                                       'Eq Mg# (Putirka 2008 +0.08, Fe2)': Eq_Mgno_Fe2_p1sig,
                                       'Eq Mg# (Putirka 2008, Fet)': Eq_Mgno_Fet,
                                       'Eq Mg# (Putirka 2008 -0.08, Fet)': Eq_Mgno_Fet_m1sig,
                                       'Eq Mg# (Putirka 2008 +0.08, Fet)': Eq_Mgno_Fet_p1sig,

                             })
        return Kd_out

    if Kd_model == "Masotta2013":

        ratioMasotta = cat_fracs['Na_Liq_cat_frac'] / (
        cat_fracs['Na_Liq_cat_frac'] + cat_fracs['K_Liq_cat_frac'])
        Kd=np.exp(1.735 - 3056 / T - 1.668 * ratioMasotta)
        Eq_Mgno_Fet = 1 / (( Kd / Mgno_Fet) + (1 - Kd))
        Eq_Mgno_Fet_m1sig = 1 / (( Kd /Mgno_Fet) + (1 - (Kd-0.05)))
        Eq_Mgno_Fet_p1sig = 1 / (( Kd / Mgno_Fet) + (1 - (Kd+0.05)))


        Kd_out = pd.DataFrame(data={'Eq Mg# (Masotta 2013, Fet)': Eq_Mgno_Fet,
                                       'Eq Mg# (Masotta 2013 -0.05, Fet)': Eq_Mgno_Fet_m1sig,
                                       'Eq Mg# (Masotta 2013 +0.05, Fet)': Eq_Mgno_Fet_p1sig,

                             })
        return Kd_out


    if isinstance(Kd_model, float) or isinstance(Kd_model, int):
        Kd=Kd_model
        Eq_Mgno_Fe2 = 1 / (( Kd / Mgno) + (1 - Kd))
        Eq_Mgno_Fet = 1 / (( Kd / Mgno_Fet) + (1 - Kd))
        Kd_out = pd.DataFrame(data={'Eq Mg# (User-selected Kd, Fe2)': Eq_Mgno_Fe2,
            'Eq Mg# (User-selected Kd, Fet)': Eq_Mgno_Fet})
        return Kd_out

## Amphibole classification diagram

def add_Leake_Amp_Fields_Fig3bot(plot_axes, fontsize=8, color=[0.3, 0.3, 0.3],
linewidth=0.5, lower_text=0.3, upper_text=0.7, text_labels=True):
    """
    Code adapted from TAS plot
    (see https://bitbucket.org/jsteven5/tasplot/src/90ed07ec34fa13405e7d2d5c563341b3e5eef95f/tasplot.py?at=master)
    Following Putirka, all Fe is assumed to be Fet
    """
    # Check that plot_axis can plot
    if 'plot' not in dir(plot_axes):
        raise TypeError('plot_axes is not a matplotlib axes instance.')

# Si Boundaries
    Tremolite_Mgno_low=0.9
    Tremolite_Mgno_up=1
    Tremolite_Si_up=8
    Tremolite_Si_low=7.5

    Actinolite_Mgno_low=0.5
    Actinolite_Mgno_up=0.9
    Actinolite_Si_up=8
    Actinolite_Si_low=7.5

    Ferroactinolite_Mgno_low=0
    Ferroactinolite_Mgno_up=0.5
    Ferroactinolite_Si_up=8
    Ferroactinolite_Si_low=7.5

    Magnesiohornblende_Mgno_low=0.5
    Magnesiohornblende_Mgno_up=1
    Magnesiohornblende_Si_up=7.5
    Magnesiohornblende_Si_low=6.5

    Ferrohornblende_Mgno_low=0
    Ferrohornblende_Mgno_up=0.5
    Ferrohornblende_Si_up=7.5
    Ferrohornblende_Si_low=6.5


    Tschermakite_Mgno_low=0.5
    Tschermakite_Mgno_up=1
    Tschermakite_Si_up=6.5
    Tschermakite_Si_low=5.5

    Ferrotschermakite_Mgno_low=0
    Ferrotschermakite_Mgno_up=0.5
    Ferrotschermakite_Si_up=6.5
    Ferrotschermakite_Si_low=5.5


    from collections import namedtuple
    FieldLine = namedtuple('FieldLine', 'x1 y1 x2 y2')
    lines = (
             FieldLine(x1=Tremolite_Si_up, y1=Tremolite_Mgno_low,
                       x2=Tremolite_Si_up, y2=Tremolite_Mgno_up),
             FieldLine(x1=Tremolite_Si_low, y1=Tremolite_Mgno_low,
                       x2=Tremolite_Si_low, y2=Tremolite_Mgno_up),

             FieldLine(x1=Tremolite_Si_up, y1=Tremolite_Mgno_up,
                       x2=Tremolite_Si_low, y2=Tremolite_Mgno_up),
             FieldLine(x1=Tremolite_Si_up, y1=Tremolite_Mgno_low,
                       x2=Tremolite_Si_low, y2=Tremolite_Mgno_low),

             FieldLine(x1=Actinolite_Si_up, y1=Actinolite_Mgno_low,
                       x2=Actinolite_Si_up, y2=Actinolite_Mgno_up),
             FieldLine(x1=Actinolite_Si_low, y1=Actinolite_Mgno_low,
                       x2=Actinolite_Si_low, y2=Actinolite_Mgno_up),

             FieldLine(x1=Actinolite_Si_up, y1=Actinolite_Mgno_up,
                       x2=Actinolite_Si_low, y2=Actinolite_Mgno_up),
             FieldLine(x1=Actinolite_Si_up, y1=Actinolite_Mgno_low,
                       x2=Actinolite_Si_low, y2=Actinolite_Mgno_low),

             FieldLine(x1=Ferroactinolite_Si_up, y1=Ferroactinolite_Mgno_low,
                       x2=Ferroactinolite_Si_up, y2=Ferroactinolite_Mgno_up),
             FieldLine(x1=Ferroactinolite_Si_low, y1=Ferroactinolite_Mgno_low,
                       x2=Ferroactinolite_Si_low, y2=Ferroactinolite_Mgno_up),

             FieldLine(x1=Ferroactinolite_Si_up, y1=Ferroactinolite_Mgno_up,
                       x2=Ferroactinolite_Si_low, y2=Ferroactinolite_Mgno_up),
             FieldLine(x1=Ferroactinolite_Si_up, y1=Ferroactinolite_Mgno_low,
                       x2=Ferroactinolite_Si_low, y2=Ferroactinolite_Mgno_low),

             FieldLine(x1=Magnesiohornblende_Si_up, y1=Magnesiohornblende_Mgno_low,
                       x2=Magnesiohornblende_Si_up, y2=Magnesiohornblende_Mgno_up),
             FieldLine(x1=Magnesiohornblende_Si_low, y1=Magnesiohornblende_Mgno_low,
                       x2=Magnesiohornblende_Si_low, y2=Magnesiohornblende_Mgno_up),

             FieldLine(x1=Magnesiohornblende_Si_up, y1=Magnesiohornblende_Mgno_up,
                       x2=Magnesiohornblende_Si_low, y2=Magnesiohornblende_Mgno_up),
             FieldLine(x1=Magnesiohornblende_Si_up, y1=Magnesiohornblende_Mgno_low,
                       x2=Magnesiohornblende_Si_low, y2=Magnesiohornblende_Mgno_low),

             FieldLine(x1=Ferrohornblende_Si_up, y1=Ferrohornblende_Mgno_low,
                       x2=Ferrohornblende_Si_up, y2=Ferrohornblende_Mgno_up),
             FieldLine(x1=Ferrohornblende_Si_low, y1=Ferrohornblende_Mgno_low,
                       x2=Ferrohornblende_Si_low, y2=Ferrohornblende_Mgno_up),

             FieldLine(x1=Ferrohornblende_Si_up, y1=Ferrohornblende_Mgno_up,
                       x2=Ferrohornblende_Si_low, y2=Ferrohornblende_Mgno_up),
             FieldLine(x1=Ferrohornblende_Si_up, y1=Ferrohornblende_Mgno_low,
                       x2=Ferrohornblende_Si_low, y2=Ferrohornblende_Mgno_low),

             FieldLine(x1=Tschermakite_Si_up, y1=Tschermakite_Mgno_low,
                       x2=Tschermakite_Si_up, y2=Tschermakite_Mgno_up),
             FieldLine(x1=Tschermakite_Si_low, y1=Tschermakite_Mgno_low,
                       x2=Tschermakite_Si_low, y2=Tschermakite_Mgno_up),

             FieldLine(x1=Tschermakite_Si_up, y1=Tschermakite_Mgno_up,
                       x2=Tschermakite_Si_low, y2=Tschermakite_Mgno_up),
             FieldLine(x1=Tschermakite_Si_up, y1=Tschermakite_Mgno_low,
                       x2=Tschermakite_Si_low, y2=Tschermakite_Mgno_low),

             FieldLine(x1=Ferrotschermakite_Si_up, y1=Ferrotschermakite_Mgno_low,
                       x2=Ferrotschermakite_Si_up, y2=Ferrotschermakite_Mgno_up),
             FieldLine(x1=Ferrotschermakite_Si_low, y1=Ferrotschermakite_Mgno_low,
                       x2=Ferrotschermakite_Si_low, y2=Ferrotschermakite_Mgno_up),

             FieldLine(x1=Ferrotschermakite_Si_up, y1=Ferrotschermakite_Mgno_up,
                       x2=Ferrotschermakite_Si_low, y2=Ferrotschermakite_Mgno_up),
             FieldLine(x1=Ferrotschermakite_Si_up, y1=Ferrotschermakite_Mgno_low,
                       x2=Ferrotschermakite_Si_low, y2=Ferrotschermakite_Mgno_low),


            )
    FieldName = namedtuple('FieldName', 'name x y rotation')
    names = (FieldName('Tremolite', 7.75, 0.95, 0),
             FieldName('Actinolite', 7.75, upper_text, 0),
             FieldName('Ferroactinolite', 7.75, lower_text, 0),
             FieldName('Magnesio\nhornblende', 7, upper_text, 0),
             FieldName('Ferro\nhornblende', 7, lower_text, 0),
             FieldName('Tschermakite', 6, upper_text, 0),
              FieldName('Ferrotschermakite', 6, lower_text, 0),
            )
    for line in lines:
        plot_axes.plot([line.x1, line.x2], [line.y1, line.y2],
                       '-', color=color, zorder=0, lw=linewidth)
    if text_labels==True:

        for name in names:
            plot_axes.text(name.x, name.y, name.name, color=color, size=fontsize,
                    horizontalalignment='center', verticalalignment='top',
                    rotation=name.rotation, zorder=0)

def calculate_Leake_Diagram_Class(amp_comps):
    cat_23ox=calculate_23oxygens_amphibole(amp_comps)
    Leake_Sites=get_amp_sites_from_input(amp_comps)
    Leake_Sites['Classification']="Not a calcic or sodic-calcic amphibole"
    Leake_Sites['Diagram']="Not a calcic or sodic-calcic amphibole"


    # Calcic amphiboles - Fig 3 top - Ca >1.5, Na + K A >0.5, Ti_C<0.5 LHS
    High_Ca_B=Leake_Sites['Ca_B']>=1.5
    High_NaK_A=(Leake_Sites['Na_A']+Leake_Sites['K_A'])>=0.5
    low_TiC=(Leake_Sites['Ti_C'])<0.5
    low_CaA=(Leake_Sites['Ca_A'])<0.5
    Fig3bot_LHS=( (High_Ca_B) & (High_NaK_A) & (low_TiC))
    Fig3bot_RHS=( (High_Ca_B) & (High_NaK_A) & (~low_TiC))

    Leake_Sites.loc[(Fig3bot_LHS|Fig3bot_RHS), 'Classification']=="High Na-K calcic amphiboles"
    Leake_Sites.loc[Fig3bot_LHS, 'Diagram']="Fig. 3 - top - LHS"
    Leake_Sites.loc[Fig3bot_RHS, 'Diagram']="Fig. 3 - top - RHS"

    # Fig 3 - bottom -


    Fig3b_LHS=( (High_Ca_B) & (~High_NaK_A)  & (low_CaA) )
    Fig3b_RHS=( (High_Ca_B) & (~High_NaK_A) & (~low_CaA))
    Leake_Sites.loc[(Fig3b_LHS|Fig3b_RHS), 'Classification']=="Low Na-K calcic amphiboles"
    Leake_Sites.loc[Fig3b_LHS, 'Diagram']="Fig. 3 - bottom - LHS"
    Leake_Sites.loc[Fig3b_RHS, 'Diagram']="Fig. 3 - bottom - Cannilloite"

    # Figure 4 - Sodic-Calcic amphiboles
    High_Ca_NaB=(cat_23ox['Ca_Amp_cat_23ox']+ Leake_Sites['Na_B'])>=1
    Int_NaB=Leake_Sites['Na_B'].between(0.5, 1.5)
    Fig4_top=( (High_NaK_A) & (High_Ca_NaB) & (Int_NaB) )
    Fig4_bottom=( (~High_NaK_A) & (High_Ca_NaB) & (Int_NaB) )
    Leake_Sites.loc[(Fig4_top), 'Classification']="High Na-K sodic-calcic amphiboles"
    Leake_Sites.loc[(Fig4_bottom), 'Classification']="Low Na-K sodic-calcic amphiboles"

    Leake_Sites.loc[Fig4_top, 'Diagram']="Fig. 4 - top"
    Leake_Sites.loc[Fig4_bottom, 'Diagram']="Fig. 4 - bottom"

    cols_to_move = ['Diagram', 'Classification']
    Leake_Sites = Leake_Sites[cols_to_move + [
        col for col in Leake_Sites.columns if col not in cols_to_move]]

    return Leake_Sites

    # Figure 5 - sodic amphiboles




def plot_amp_class_Leake(amp_comps, fontsize=8, color=[0.3, 0.3, 0.3],
linewidth=0.5, lower_text=0.3, upper_text=0.7, text_labels=True, site_check=True,
plots="Ca_Amphiboles", marker='.k', dpi=300):


    cat_23ox=calculate_23oxygens_amphibole(amp_comps)
    Leake_Sites=get_amp_sites_from_input(amp_comps)
    low_Ca_B=Leake_Sites['Ca_B']<1.5
    High_Ca_A=Leake_Sites['Ca_A']>=0.5
    high_NaK_A=(Leake_Sites['Na_A']+Leake_Sites['K_A'])>0.5


    if site_check==False:

        if (any(Leake_Sites['Ca_A']>=0.5)):

            #print(str(sum(low_Ca_B))+ " amphiboles have Ca_B<1.5")
            w.warn(str(sum(high_Ca_A))+ " of your amphiboles have Ca_A>=0.5, so shouldnt be plotted on this diagram based on Leake. site_check=True filters these out, but youve choosen site_check is False")

        if (any(Leake_Sites['Ca_B']<1.5)):
            #print(str(sum(low_Ca_B))+ " amphiboles have Ca_B<1.5")
            w.warn(str(sum(low_Ca_B))+ " of your amphiboles have Ca_B<1.5, so shouldnt be plotted on this diagram based on Leake. site_check=True filters these out, but youve choosen site_check is False")
        if (any((Leake_Sites['Na_A']+Leake_Sites['K_A'])>0.5)):
            #print(str(sum(low_NaK_A=))+ " amphiboles have Na_A+K_A<1.5, so arent shown on this plot")
            w.warn(str(sum(low_NaK_A))+ " of your amphiboles have Na_A+K_A>0.5"
            " so shouldn\'t be plotted on this diagram based on Leake. site_check=True filters these out, but youve choosen site_check is False", stacklevel=2)

        fig, (ax1) = plt.subplots(1, 1, figsize = (7,5), dpi=dpi)


        ax1.plot(cat_23ox['Si_Amp_cat_23ox'], cat_23ox['Mgno_Amp'], 'ok')
        add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=fontsize, color=color,
        linewidth=linewidth, lower_text=lower_text, upper_text=upper_text,
        text_labels=text_labels)
        ax1.invert_xaxis()
        ax1.set_xlabel('Si (apfu)')
        ax1.set_ylabel('Mg# Amphibole')
        #     ax2.plot(Leake_Sites['Ca_B'], Leake_Sites['Na_A']+Leake_Sites['K_A'],
        #     'ok')
        #     ax2.plot([1.5, 1.5], [0, 0.5], '-r')
        #     ax2.plot([0, 1.5], [0.5, 0.5], '-r')
        # #out_range=
        #
        #     ax2.annotate("Out of range\n for this diagram", xy=(0.1, 0.25), xycoords="axes fraction", fontsize=8)
        #     ax2.set_xlabel('Ca_B site')
        #     ax2.set_ylabel('Na_A + K_A site')

    # if site_check==False:
    #     fig, (ax1) = plt.subplots(figsize = (8,5))
    #     ax1.plot(cat_23ox['Si_Amp_cat_23ox'], cat_23ox['Mgno_Amp'], 'ok')
    #     add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=fontsize, color=color,
    #     linewidth=linewidth, lower_text=lower_text, upper_text=upper_text,
    #     text_labels=text_labels)
    #     ax1.invert_xaxis()
    #     ax1.set_xlabel('Si (apfu)')
    #     ax1.set_ylabel('Mg# Amphibole')

    if site_check==True:
        if plots == "Ca_Amphiboles":
            print(str(sum(low_Ca_B))+ " amphiboles have Ca_B<1.5, so arent shown on this plot")
            print(str(sum(High_Ca_A))+ " amphiboles have Ca_A>=0.5, so arent shown on this plot")
            print(str(sum(high_NaK_A))+ " amphiboles have Na_A+K_A>0.5, so arent shown on this plot")
            fig, (ax1) = plt.subplots(1, 1, figsize = (7,5))

            ax1.plot(cat_23ox['Si_Amp_cat_23ox'].loc[(~high_NaK_A)&(~low_Ca_B)& (~High_Ca_A)],
            cat_23ox['Mgno_Amp'].loc[(~high_NaK_A)&(~low_Ca_B)& (~High_Ca_A)], marker)



            add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=fontsize, color=color,
        linewidth=linewidth, lower_text=lower_text, upper_text=upper_text,
        text_labels=text_labels)
            ax1.invert_xaxis()
            ax1.set_xlabel('Si (apfu)')
            ax1.set_ylabel('Mg# Amphibole')





## Equilirium things for Feldspar.
def calculate_eq_plag_An_number(liq_comps, T=None, P=None, An_model=None):
    '''calculates equilibrium Plag An#s  based on inputtted liquid compositions.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Ol, MgO_Ol etc.

    T: pandas.Series, float, int
        T in Kelvin, optional for 3 of the 4 outputs from Namur

    P: pandas.Series, float, int
        P in kbar. Optional for Namur


    An_model: str, int or float
        Specify which An model you wish to use.
        "Putirka2005": uses equilibrium An-Ab-Or from the Putirka (2005), as implemented
        in the spreadsheets of K. Putirka
        "Namur2011": Uses equilibrium An from Namur (2011)
        "All": returns values from Putirka2005 and Namur

    Returns
    -------
    DataFrame with equilibrium An values, with column headings corresponding to model choosen.
  '''
    cat_liqs = calculate_anhydrous_cat_fractions_liquid(liq_comps)

    if isinstance(P, int):
        P=float(P)
    if isinstance(T, int):
        T=float(T)

    if An_model == "Putirka2005" or An_model == "All":
        Pred_An_EqE = (np.exp(-3.485 + 22.93 * cat_liqs['Ca_Liq_cat_frac'].astype(float)
         + 0.0805 * cat_liqs['H2O_Liq'].astype(float)
        + 1.0925 * cat_liqs['Ca_Liq_cat_frac'].astype(float)
         / (cat_liqs['Ca_Liq_cat_frac'].astype(float) + cat_liqs['Na_Liq_cat_frac'].astype(float))
         +13.11 * cat_liqs['Al_Liq_cat_frac'].astype(float) / (
        cat_liqs['Al_Liq_cat_frac'].astype(float) + cat_liqs['Si_Liq_cat_frac'].astype(float))
        + 5.59258 *cat_liqs['Si_Liq_cat_frac'].astype(float)**3 -
        38.786 *P  / (T)- 125.04 *cat_liqs['Ca_Liq_cat_frac'].astype(float)
        *cat_liqs['Al_Liq_cat_frac'].astype(float)
        + 8.958 * cat_liqs['Si_Liq_cat_frac'].astype(float) * cat_liqs['K_Liq_cat_frac'].astype(float)
         - 2589.27 / (T)))

        Pred_Ab_EqF = (np.exp(-2.748 - 0.1553 * cat_liqs['H2O_Liq'].astype(float) + 1.017 * cat_liqs['Mg_Number_Liq_NoFe3'].astype(float) - 1.997 * cat_liqs['Si_Liq_cat_frac'].astype(float)**3
         + 54.556 *P  / T- 67.878 *cat_liqs['K_Liq_cat_frac'].astype(float) *
         cat_liqs['Al_Liq_cat_frac'].astype(float)
        - 99.03 * cat_liqs['Ca_Liq_cat_frac'].astype(float) * cat_liqs['Al_Liq_cat_frac'].astype(float) + 4175.307 / T))

        Pred_Or_EqG = (np.exp(19.42 - 12.5 * cat_liqs['Mg_Liq_cat_frac'].astype(float)
         - 161.4 * cat_liqs['Na_Liq_cat_frac'].astype(float)
         - 16.65 * cat_liqs['Ca_Liq_cat_frac'].astype(float) / (
        cat_liqs['Ca_Liq_cat_frac'].astype(float) + cat_liqs['Na_Liq_cat_frac'].astype(float))
         - 528.1 * cat_liqs['K_Liq_cat_frac'] * cat_liqs['Al_Liq_cat_frac'] -
        19.38 * cat_liqs['Si_Liq_cat_frac']**3
         + 168.2 *cat_liqs['Si_Liq_cat_frac'] *
         cat_liqs['Na_Liq_cat_frac']
         - 1951.2 * cat_liqs['Ca_Liq_cat_frac'] * cat_liqs['K_Liq_cat_frac'] - 10190 / T))

        df_out_Put=pd.DataFrame(data={'Pred_An_EqE_P2005': Pred_An_EqE, 'Pred_Ab_EqF_P2005': Pred_Ab_EqF,
                                 'Pred_Or_EqG_P2005': Pred_Or_EqG})
    if An_model == "Namur2011" or An_model == "All":
        df_out_Nam=calculate_An_Namur2011(liq_comps, T=T)
    if An_model == "Namur2011":
        return df_out_Nam
    if An_model == "Putirka2005":
        return df_out_Put
    if An_model == "All":
        df_out=pd.concat([df_out_Put, df_out_Nam], axis=1)
        return df_out


def calculate_An_Namur2011(liq_comps, T=None):
    '''calculates equilibrium An contents using the T and H2O-independent
    model of Namur et al. (2011). Returns An calculated using all 3 models,
    and selects the most suitable model based on the melt composition.

    Parameters
    -------

    liq_comps: pandas.DataFrame
        Liquid compositions with column headings SiO2_Ol, MgO_Ol etc.

    T: pandas.Series, int, float
        Temperature in Kelvin, needed for thermodynamic model.

    Returns
    -------
    DataFrame with An contents for all three equations, and an indication of the most
    suitable model using the major element filters of Namur et al. (2011) based on Si-Na-K.
  '''

    Liqs_c=liq_comps.copy()
    Liqs_c['MnO_Liq']=0
    Liqs_c['Cr2O3_Liq']=0
    mol_props=calculate_anhydrous_mol_fractions_liquid(liq_comps=Liqs_c)
    ox_cat_8_den=(2*mol_props['SiO2_Liq_mol_frac'] + mol_props['MgO_Liq_mol_frac']+
                 mol_props['FeOt_Liq_mol_frac']+mol_props['CaO_Liq_mol_frac']
                  +3*mol_props['Al2O3_Liq_mol_frac']+mol_props['Na2O_Liq_mol_frac']+
                 mol_props['K2O_Liq_mol_frac']+2*mol_props['TiO2_Liq_mol_frac']+
                 5*mol_props['P2O5_Liq_mol_frac'])
    ox_cat_8_den
    mol_prop_8_den=8*mol_props.divide(ox_cat_8_den, axis='rows')
    mol_prop_8_den_c=mol_prop_8_den.copy()
    mol_prop_8_den_c['Al2O3_Liq_mol_frac']=2*mol_prop_8_den_c['Al2O3_Liq_mol_frac']
    mol_prop_8_den_c['Na2O_Liq_mol_frac']=2*mol_prop_8_den_c['Na2O_Liq_mol_frac']
    mol_prop_8_den_c['K2O_Liq_mol_frac']=2*mol_prop_8_den_c['K2O_Liq_mol_frac']
    mol_prop_8_den_c['P2O5_Liq_mol_frac']=2*mol_prop_8_den_c['P2O5_Liq_mol_frac']

    mol_prop_8_den_c.columns = [str(col).replace('_mol_frac', '_8_ox')
                          for col in mol_prop_8_den_c.columns]
    Sum_8_ox=mol_prop_8_den_c.sum(axis=1)

    ox_8_cat_5=5*mol_prop_8_den_c.divide(Sum_8_ox, axis='rows')
    ox_8_cat_5.columns = [str(col).replace('_8_ox', '_8_ox_5_cat')
                          for col in ox_8_cat_5.columns]
    ox_8_cat_5['Ca_Ca_Na']=((ox_8_cat_5['CaO_Liq_8_ox_5_cat'])/
                            (ox_8_cat_5['CaO_Liq_8_ox_5_cat']+ox_8_cat_5['Na2O_Liq_8_ox_5_cat']))
    ox_8_cat_5['Al_Al_Si']=((ox_8_cat_5['Al2O3_Liq_8_ox_5_cat'])/
                            (ox_8_cat_5['Al2O3_Liq_8_ox_5_cat']+ox_8_cat_5['SiO2_Liq_8_ox_5_cat']))

    # For ultramafic-mafic, liquids with < 52 w% SiO2, and <5 wt% Na and K.
    Si_M_Um=0.55
    Al_M_Um=-0.04
    Fe_M_Um=0.35
    Mg_M_Um=0.19
    Ca_M_Um=0.19
    Na_M_Um=0.45
    Ca_Ca_Na_Um=1.09
    Al_Al_Si_Um=3.68

    An_Eq_Um=(-2.71 + (ox_8_cat_5['SiO2_Liq_8_ox_5_cat']*Si_M_Um)
              +(ox_8_cat_5['Al2O3_Liq_8_ox_5_cat']*Al_M_Um)
              +(ox_8_cat_5['FeOt_Liq_8_ox_5_cat']*Fe_M_Um)
              +(ox_8_cat_5['MgO_Liq_8_ox_5_cat']*Mg_M_Um)
              +(ox_8_cat_5['CaO_Liq_8_ox_5_cat']*Ca_M_Um)
              +(ox_8_cat_5['Na2O_Liq_8_ox_5_cat']*Na_M_Um)
              +(ox_8_cat_5['Ca_Ca_Na']*Ca_Ca_Na_Um)
              +(ox_8_cat_5['Al_Al_Si']*Al_Al_Si_Um))

    # For Alkaline melts, liquids with < 52 w% SiO2, and >5 wt% Na and K.

    Si_M_Alk=1.16
    Al_M_Alk=-3.14
    Fe_M_Alk=-0.34
    Mg_M_Alk=-0.16
    Ca_M_Alk=-0.82
    Na_M_Alk=0.58
    Ca_Ca_Na_Alk=2.17
    Al_Al_Si_Alk=16.48

    An_Eq_Alk=(-4.66 + (ox_8_cat_5['SiO2_Liq_8_ox_5_cat']*Si_M_Alk)
              +(ox_8_cat_5['Al2O3_Liq_8_ox_5_cat']*Al_M_Alk)
              +(ox_8_cat_5['FeOt_Liq_8_ox_5_cat']*Fe_M_Alk)
              +(ox_8_cat_5['MgO_Liq_8_ox_5_cat']*Mg_M_Alk)
              +(ox_8_cat_5['CaO_Liq_8_ox_5_cat']*Ca_M_Alk)
              +(ox_8_cat_5['Na2O_Liq_8_ox_5_cat']*Na_M_Alk)
              +(ox_8_cat_5['Ca_Ca_Na']*Ca_Ca_Na_Alk)
              +(ox_8_cat_5['Al_Al_Si']*Al_Al_Si_Alk))

    # Intermediate - Felsic, >52 wt% SiO2
    Si_M_IF=-0.22
    Al_M_IF=0.94
    Fe_M_IF=-0.49
    Mg_M_IF=0.07
    Ca_M_IF=0.41
    Na_M_IF=-0.04
    Ca_Ca_Na_IF=0.31
    Al_Al_Si_IF=-3.86

    An_Eq_IF=(1.17 + (ox_8_cat_5['SiO2_Liq_8_ox_5_cat']*Si_M_IF)
              +(ox_8_cat_5['Al2O3_Liq_8_ox_5_cat']*Al_M_IF)
              +(ox_8_cat_5['FeOt_Liq_8_ox_5_cat']*Fe_M_IF)
              +(ox_8_cat_5['MgO_Liq_8_ox_5_cat']*Mg_M_IF)
              +(ox_8_cat_5['CaO_Liq_8_ox_5_cat']*Ca_M_IF)
              +(ox_8_cat_5['Na2O_Liq_8_ox_5_cat']*Na_M_IF)
              +(ox_8_cat_5['Ca_Ca_Na']*Ca_Ca_Na_IF)
              +(ox_8_cat_5['Al_Al_Si']*Al_Al_Si_IF))

    # Thermodynamic model
    if T is None:
        print('Cant use thermodynamic model, as you havent entered a temperature')
    if T is not None:
        Si_M_Thermo=0.41
        Al_M_Thermo=-1.69
        Fe_M_Thermo=-0.34
        Mg_M_Thermo=-0.51
        Ca_M_Thermo=-0.5
        Na_M_Thermo=0.54
        Ca_Ca_Na_Thermo=1.99
        Al_Al_Si_Thermo=8.2

        ln_a_An_Liq=(5.72 - 15339/T + (ox_8_cat_5['SiO2_Liq_8_ox_5_cat']*Si_M_Thermo)
                +(ox_8_cat_5['Al2O3_Liq_8_ox_5_cat']*Al_M_Thermo)
                +(ox_8_cat_5['FeOt_Liq_8_ox_5_cat']*Fe_M_Thermo)
                +(ox_8_cat_5['MgO_Liq_8_ox_5_cat']*Mg_M_Thermo)
                +(ox_8_cat_5['CaO_Liq_8_ox_5_cat']*Ca_M_Thermo)
                +(ox_8_cat_5['Na2O_Liq_8_ox_5_cat']*Na_M_Thermo)
                +(ox_8_cat_5['Ca_Ca_Na']*Ca_Ca_Na_Thermo)
                +(ox_8_cat_5['Al_Al_Si']*Al_Al_Si_Thermo))

        ln_xAn_plag=46.58-0.0018*T-(5.77*np.log(T))+ln_a_An_Liq

        An_Eq_Thermo=np.exp(ln_xAn_plag)


    ox_8_cat_5['Choice']="Int_Fels"
    ox_8_cat_5['An_Eq_best_non_thermo_choice']=An_Eq_IF



    Low_Si=Liqs_c['SiO2_Liq']<52
    Low_Na_K=(Liqs_c['K2O_Liq']+Liqs_c['Na2O_Liq'])<5

    ox_8_cat_5.loc[(Low_Si&Low_Na_K), 'Choice']="Maf_Ultra"
    ox_8_cat_5.loc[(Low_Si&(~Low_Na_K)), 'Choice']="Alk_Maf_Ultra"

    ox_8_cat_5.loc[(Low_Si&Low_Na_K), 'An_Eq_best_choice']=An_Eq_Um
    ox_8_cat_5.loc[(Low_Si&(~Low_Na_K)), 'An_Eq_best_choice']=An_Eq_Alk
    An_out=pd.DataFrame(data={'An_Eq_best_non-thermo_choice_N2011':ox_8_cat_5['An_Eq_best_choice'],
                              'Selected non-thermo model_N2011': ox_8_cat_5['Choice'],
                                'An_Eq_Mafic_Ultramafic_eq33_N2011':An_Eq_Um,
                              'An_Eq_Alk_Mafic_Ultramafic_eq34_N2011':An_Eq_Alk,
                              'An_Eq_Int_Fels_eq35_N2011':An_Eq_IF

                             })
    if T is not None:
        An_out.insert(0, 'An_Eq_Thermo_eq31_N2011', An_Eq_Thermo)
    return An_out

## Feldspar Ternary Diagram things

# The function to create the classification diagram
def plot_fspar_classification(
    figsize=(6,6),
    major_grid=False,
    minor_grid=False,
    labels=False,
    ticks=True,
    major_grid_kwargs={"ls": ":", "lw": 0.5, "c": "k"},
    minor_grid_kwargs={"ls": "-", "lw": 0.25, "c": "lightgrey"},
    fontsize_component_labels=10,
    fontsize_axes_labels=14,
    Anorthite_label='An',
    Anorthoclase_label='AnC',
    Albite_label='Ab',
    Oligoclase_label='Ol',
    Andesine_label='Ad',
    Labradorite_label='La',
    Bytownite_label='By',
    Sanidine_label='San',

):
    """
    Plotting a feldspar ternary classification diagram according to Deer, Howie, and Zussman 1992 3rd edition.
    This function relies heavily on the python package python-ternary by Marc Harper et al. (2015).
    :cite:`harper2015`

    Inputs:
    figsize: tuple
    for figure size same as matplotlib

    major_grid: boolean,
    whether or not to show major grid lines shows lines every .2. Default = False

    minor_grid: boolean,
    whether or not to show minor grid lines...shows lines every .05. Default = False

    labels: boolean,
    whether or not to show abbreviated field labels for feldspar classification

    ticks: boolean.
        If True, adds ticks onto side of axes

    major_grid_kwargs: dict,
    inherited matplotlib kwargs for styling major grid

    minor_grid_kwargs: dict,
    inherited matplotlib kwargs for styling minor grid

    ...labels: str
    Can overwrite defauls for what the different regions are named. Defaults below.
    Anorthite_label='An',
    Anorthoclase_label='AnC',
    Albite_label='Ab',
    Oligoclase_label='Ol',
    Andesine_label='Ad',
    Labradorite_label='La',
    Bytownite_label='By',
    Sanidine_label='San',


    Returns:

    fig: matplotlib figure

    tax: ternary axis subplot from ternary package. To use matplotlib ax level styling
    and functions:
            # Example
            ax = tax.get_axes()
            ax.set_title('my title')


    """
    # plagioclase classification
    # anorthite      1.0 - 0.9
    # bytownite      0.9 - 0.7
    # labradorite    0.7 - 0.5
    # andesine       0.5 - 0.3
    # oligoclase     0.3 - 0.1
    # albite         0.1 - 0.0

    # figure and axis component
    figure, tax = ternary.figure()
    figure.set_size_inches(figsize)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=1.5,zorder = 0)  # outside triangle boundary width
    if major_grid is True:
        tax.gridlines(multiple=0.2, **major_grid_kwargs, zorder=0)
    if minor_grid is True:
        tax.gridlines(multiple=0.05, linewidth=0.5, **minor_grid_kwargs, zorder=0)
    # Set Axis labels and Title
    tax.right_corner_label("An", fontsize=fontsize_axes_labels)
    tax.top_corner_label("Or", fontsize=fontsize_axes_labels)
    tax.left_corner_label("Ab", fontsize=fontsize_axes_labels)

    # making the plag curve
    An = np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.20, 0.15])
    Or = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15])
    f_plag = interpolate.interp1d(An, Or, kind="linear")
    An_new = np.linspace(0.15, 1, 1000)
    Or_new = f_plag(An_new)
    Ab_new = 1 - An_new
    plag_curve = np.hstack([An_new[:, None], Or_new[:, None], Ab_new[:, None]])

    # making plag - kspar line

    Or_kp = np.array([0, 0.15])
    An_kp = np.array([0, 0.15])
    Ab_kp = np.array([1, 0.85])

    f_kp = interpolate.interp1d(An_kp, Ab_kp)
    An_kp_new = np.linspace(0, 0.15, 1000)
    Or_kp_new = An_kp_new
    Ab_kp_new = An_kp_new
    plag_kspar_line = np.hstack(
        [An_kp_new[:, None], Or_kp_new[:, None], Ab_kp_new[:, None]]
    )

    # making the kspar curve
    Or_k = np.array([1.0, 0.37, 0.15])
    An_k = np.array([0.05, 0.05, 0.15])
    f_kspar = interpolate.interp1d(Or_k, An_k)
    Or_k_new = np.linspace(0.15, 1, 1000)
    An_k_new = f_kspar(Or_k_new)
    Ab_k_new = 1 - Or_k_new
    kspar_curve = np.hstack([An_k_new[:, None], Or_k_new[:, None], Ab_k_new[:, None]])

    # anorthite - bytownite divider
    tax.line([0.9, 0, 0], plag_curve[plag_curve[:, 0] >= 0.9][0], color="k",  zorder=0)

    # bytownite - labradorite divider
    tax.line([0.7, 0, 0], plag_curve[plag_curve[:, 0] >= 0.7][0], color="k", zorder=0)

    # labradorite - andesine divider
    tax.line([0.5, 0, 0], plag_curve[plag_curve[:, 0] >= 0.5][0], color="k", zorder=0)

    # andesine - oligoclase divider
    tax.line([0.3, 0, 0], plag_curve[plag_curve[:, 0] >= 0.3][0], color="k", zorder=0)

    # oligoclase - albite divider
    tax.line([0.1, 0, 0], plag_kspar_line[plag_kspar_line[:, 0] >= 0.1][0], color="k", zorder=0)

    # sanidine - anorthoclase divider
    tax.line([0, 0.37, 0.63], kspar_curve[kspar_curve[:, 1] >= 0.37][0], color="k", zorder=0)

    # anorthoclase - albite divider
    tax.line([0, 0.1, 0.9], plag_kspar_line[plag_kspar_line[:, 0] >= 0.1][0], color="k", zorder=0)

    # making the plag - kspar divider
    tax.plot(plag_kspar_line[plag_kspar_line[:, 1] > 0.1], color="k", zorder=0)

    # plotting the curves
    tax.plot(plag_curve[:-60], color="k", zorder=0)
    tax.plot(kspar_curve[:-60], color="k", zorder=0)

    # Set ticks
    if ticks is True:
        tax.ticks(
            axis="lbr", linewidth=0.5, multiple=0.20, offset=0.02, tick_formats="%.1f"
        )

    # # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax._redraw_labels()

    if labels is True:
        # annotations
        ax = tax.get_axes()
        ax.text(0.3, 0.5, Sanidine_label, fontsize=fontsize_component_labels, rotation=60, zorder=0)
        ax.text(0.15, 0.2, Anorthoclase_label, fontsize=fontsize_component_labels, zorder=0)
        ax.text(0.05, 0.03, Albite_label, fontsize=fontsize_component_labels, zorder=0)
        ax.text(0.2, 0.03, Oligoclase_label, fontsize=fontsize_component_labels, zorder=0)
        ax.text(0.38, 0.01, Andesine_label, fontsize=fontsize_component_labels, zorder=0)
        ax.text(0.58, 0.01, Labradorite_label, fontsize=fontsize_component_labels, zorder=0)
        ax.text(0.78, 0.01, Bytownite_label, fontsize=fontsize_component_labels, zorder=0)
        ax.text(0.94, 0.01, Anorthite_label, fontsize=fontsize_component_labels, zorder=0)



    return figure, tax




def plot_px_classification(
    figsize=(7,5),
    cut_in_half=True,
    major_grid=False,
    minor_grid=False,
    labels=False,
    major_grid_kwargs={"ls": ":", "lw": 0.5, "c": "k"},
    minor_grid_kwargs={"ls": "-", "lw": 0.25, "c": "lightgrey"},
    fontsize_component_labels=10,
    fontsize_axes_labels=14,
    Enstatite_label='Enstatite',
    Ferrosilite_label='Ferrosilite',
    Pigeonite_label='Pigeonite',
    Augite_label='Augite',
    Diopside_label='Diopside',
    Hedenbergite_label='Hedenbergite'

):
    """
    Plotting a pyroxene ternary classification diagram according to Deer, Howie, and Zussman 1992 3rd edition.
    This function relies heavily on the python package python-ternary by Marc Harper et al. (2015).
    :cite:`harper2015`

    Inputs:
    figsize: tuple
        for figure size same as matplotlib. Default is 7-5, assuming you'll cut the top off.

    cut_in_half: boolean
        If True, cuts the top off to give the pyroxene quadrilateral.


    major_grid: boolean
        whether or not to show major grid lines shows lines every .2. Default = False

    minor_grid: boolean,
        whether or not to show minor grid lines...shows lines every .05. Default = False

    labels: boolean,
        whether or not to show abbreviated field labels for pyroxene classification

    major_grid_kwargs: dict,
        inherited matplotlib kwargs for styling major grid

    minor_grid_kwargs: dict,
        inherited matplotlib kwargs for styling minor grid

    ...labels: str
    Can overwrite defauls for what the different regions are named. Defaults below.
    Enstatite_label='Enstatite'
    Ferrosilite_label='Ferrosilite'
    Pigeonite_label='Pigeonite'
    Augite_label='Augite'
    Diopside_label='Diopside'
    Hedenbergite_label='Hedenbergite'


    Returns:

    fig: matplotlib figure

    tax: ternary axis subplot from ternary package. To use matplotlib ax level styling
    and functions:
            # Example
            ax = tax.get_axes()
            ax.set_title('my title')


    """

    # figure and axis component
    figure, tax = ternary.figure()
    figure.set_size_inches(figsize)



    # Draw Boundary and Gridlines
    tax.boundary(linewidth=1.5,zorder = 0)  # outside triangle boundary width
    if major_grid is True:
        tax.gridlines(multiple=0.2, **major_grid_kwargs, zorder=0)
    if minor_grid is True:
        tax.gridlines(multiple=0.05, linewidth=0.5, **minor_grid_kwargs, zorder=0)
    # Set Axis labels and Title

    if cut_in_half is True:
        axes=tax.get_axes()
        axes.set_ylim([-0.01, 0.434])


    tax.right_corner_label("", fontsize=fontsize_axes_labels)



    if cut_in_half is True:
        tax.top_corner_label("↑ Wo", fontsize=fontsize_axes_labels)
    else:
        tax.top_corner_label("Wo", fontsize=fontsize_axes_labels)
    tax.left_corner_label("En", fontsize=fontsize_axes_labels)
    tax.right_corner_label("Fs", fontsize=fontsize_axes_labels)

    # Adding Fields
    tax.line([0, 0.5, 0.5], [0.5, 0.5, 0], color="k", zorder=0)
    tax.line([0, 0.45, 0.55], [0.55, 0.45, 0], color="k", zorder=0)
    tax.line([0.25, 0.5, 0.25], [0.275, 0.45, 0.275], color="k", zorder=0)
    tax.line([0, 0.05, 0.95], [0.95, 0.05, 0], color="k", zorder=0)
    tax.line([0, 0.2, 0.8], [0.8, 0.2, 0], color="k", zorder=0)
    tax.line([0, 0.2, 0.8], [0.8, 0.2, 0], color="k", zorder=0)
    tax.line([0.5, 0, 0.5], [0.475, 0.05, 0.475], color="k", zorder=0)



    # # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax._redraw_labels()

    if labels is True:
        # annotations
        ax = tax.get_axes()
        ax.text(0.2, 0.01, Enstatite_label, fontsize=fontsize_component_labels)
        ax.text(0.7, 0.01, Ferrosilite_label, fontsize=fontsize_component_labels)
        ax.text(0.4, 0.1, Pigeonite_label, fontsize=fontsize_component_labels)
        ax.text(0.42, 0.25, Augite_label, fontsize=fontsize_component_labels)
        ax.text(0.28, 0.40, Diopside_label, fontsize=fontsize_component_labels)
        ax.text(0.52, 0.4, Hedenbergite_label, fontsize=fontsize_component_labels)


    return figure, tax


# function to get arrays in proper format for plotting on ternary
def tern_points(right, top, left):
    """Tern_points takes 3 equal size 1D arrays or pandas series and organizes them into points to be plotted on a ternary
         with the following arrangement:(lower right,top,lower left).
         This is a generic function to allow flexibiliy, see also tern_points_px to calculate the components
         for pyroxene, and tern_points_fspar to calculate components and coordinates for feldspar
             Inputs:
             x = 1D array like (lower right vertex)
             y = 1D array like (top vertex)
             z = 1D array like (lower left vertex)
    """
    if isinstance(right, pd.Series):
        right = right.to_numpy()
    if isinstance(top, pd.Series):
        top = top.to_numpy()
    if isinstance(left, pd.Series):
        left = left.to_numpy()

    points = np.hstack([right[:, None], top[:, None], left[:, None]])

    return points


def tern_points_px(px_comps=None):
    """Tern_points takes pyroxene compositions, and calculates Fs, En and Wo,
    and returns co-ordinates to plot on a ternary diagram as a np.array
    """
    # This just replaces columns, so if you load Opx, it treats as Cpx,
    # There are more elegant ways to handle this probably!
    px_comps_c=px_comps.copy()
    px_comps_c.columns = px_comps_c.columns.str.replace("_Opx", "_Cpx")
    cpx_comps=calculate_clinopyroxene_components(cpx_comps=px_comps_c)
    right=cpx_comps["Fs_Simple_MgFeCa_Cpx"]
    top=cpx_comps["Wo_Simple_MgFeCa_Cpx"]
    left=cpx_comps["En_Simple_MgFeCa_Cpx"]


    if isinstance(right, pd.Series):
        right = right.to_numpy()
    if isinstance(top, pd.Series):
        top = top.to_numpy()
    if isinstance(left, pd.Series):
        left = left.to_numpy()

    points = np.hstack([right[:, None], top[:, None], left[:, None]])

    return points


def tern_points_opx(px_comps=None):
    """Tern_points takes a dataframe, finds columns with _Opx in them, and calculates Fs, En and Wo,
    and returns co-ordinates to plot on a ternary diagram as a np.array
    """
    # This just replaces columns, so if you load Opx, it treats as Cpx,
    # There are more elegant ways to handle this probably!
    px_comps_c=px_comps.copy()
    px_comps_c = px_comps_c.filter(like='Opx')
    px_comps_c.columns = px_comps_c.columns.str.replace("_Opx", "_Cpx")
    cpx_comps=calculate_clinopyroxene_components(cpx_comps=px_comps_c)
    right=cpx_comps["Fs_Simple_MgFeCa_Cpx"]
    top=cpx_comps["Wo_Simple_MgFeCa_Cpx"]
    left=cpx_comps["En_Simple_MgFeCa_Cpx"]


    if isinstance(right, pd.Series):
        right = right.to_numpy()
    if isinstance(top, pd.Series):
        top = top.to_numpy()
    if isinstance(left, pd.Series):
        left = left.to_numpy()

    points = np.hstack([right[:, None], top[:, None], left[:, None]])

    return points

def tern_points_cpx(px_comps=None):
    """Tern_points takes a dataframe, finds columns with _Cpx in them, and calculates Fs, En and Wo,
    and returns co-ordinates to plot on a ternary diagram as a np.array
    """
    # This just replaces columns, so if you load Opx, it treats as Cpx,
    # There are more elegant ways to handle this probably!
    px_comps_c=px_comps.copy()
    px_comps_c = px_comps_c.filter(like='Cpx')
    cpx_comps=calculate_clinopyroxene_components(cpx_comps=px_comps_c)
    right=cpx_comps["Fs_Simple_MgFeCa_Cpx"]
    top=cpx_comps["Wo_Simple_MgFeCa_Cpx"]
    left=cpx_comps["En_Simple_MgFeCa_Cpx"]


    if isinstance(right, pd.Series):
        right = right.to_numpy()
    if isinstance(top, pd.Series):
        top = top.to_numpy()
    if isinstance(left, pd.Series):
        left = left.to_numpy()

    points = np.hstack([right[:, None], top[:, None], left[:, None]])

    return points


def tern_points_fspar(fspar_comps=None):
    """Tern_points takes feldspar compositions, and calculates An, Ab, Or,
    and returns co-ordinates to plot on a ternary diagram as a np.array
    You can input plag or kspar compositions as fspar_comps
    """
    # This just replaces columns, so if you load Opx, it treats as Cpx,
    # There are more elegant ways to handle this probably!
    fspar_comps_c=fspar_comps.copy()
    fspar_comps_c.columns = fspar_comps_c.columns.str.replace("_Kspar", "_Plag")
    fspar_comps=calculate_cat_fractions_plagioclase(plag_comps=fspar_comps_c)
    right=fspar_comps["An_Plag"]
    top=fspar_comps["Or_Plag"]
    left=fspar_comps["Ab_Plag"]


    if isinstance(right, pd.Series):
        right = right.to_numpy()
    if isinstance(top, pd.Series):
        top = top.to_numpy()
    if isinstance(left, pd.Series):
        left = left.to_numpy()

    points = np.hstack([right[:, None], top[:, None], left[:, None]])

    return points


    #return fig
