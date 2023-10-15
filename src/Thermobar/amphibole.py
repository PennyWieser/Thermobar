import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *


## Equations: Amphibole-only Barometers


def P_Kraw2012(T=None, *, Mgno_Amp, deltaNNO):
    '''
    Amphibole-only barometer (PH2O) from Krawczynski et al. (2012)
    :cite:`krawczynski2012amphibole`

    **Note - this is only the pressure for the first appearance of amphibole,
    so should only be applied to the highest Mg# amphiboles in each suite.
    it also only gives the partial pressure of H2O, if there is CO2 in the system,
    this will not be equal to the total pressure.**
    '''

    return 0.01*((Mgno_Amp/52.7 -0.014*deltaNNO)**15.12)

def P_Ridolfi2012_1a(T=None, *, Si_Amp_13_cat, Ti_Amp_13_cat, Fet_Amp_13_cat,
                     Mg_Amp_13_cat, Ca_Amp_13_cat, K_Amp_13_cat, Na_Amp_13_cat, Al_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1a of Ridolfi and Renzulli (2012).
    Calibrated between 1.3-22 kbars
    :cite:`ridolfi2012calcic`
    '''
    return 0.01 * (np.exp(125.9332115 - 9.587571403 * Si_Amp_13_cat - 10.11615567 * Ti_Amp_13_cat
    - 8.173455128 * Al_Amp_13_cat- 9.226076274 * Fet_Amp_13_cat - 8.793390507 * Mg_Amp_13_cat
    - 1.6658613 * Ca_Amp_13_cat + 2.48347198 * Na_Amp_13_cat + 2.519184959 * K_Amp_13_cat))


def P_Ridolfi2012_1b(T=None, *, Si_Amp_13_cat, Ti_Amp_13_cat, Fet_Amp_13_cat,
                     Mg_Amp_13_cat, Ca_Amp_13_cat, K_Amp_13_cat, Na_Amp_13_cat, Al_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1b of Ridolfi and Renzulli (2012).
    Calibrated between 1.3-5 kbars
    :cite:`ridolfi2012calcic`
    '''
    return (0.01 * (np.exp(38.722545085 - 2.695663047 * Si_Amp_13_cat - 2.35647038717941 * Ti_Amp_13_cat
            - 1.30063975020919 * Al_Amp_13_cat - 2.7779767369382 * Fet_Amp_13_cat
            - 2.48384821395444 * Mg_Amp_13_cat- 0.661386638563983 * Ca_Amp_13_cat
            - 0.270530207793162 * Na_Amp_13_cat + 0.111696322092308 * K_Amp_13_cat)))


def P_Ridolfi2012_1c(T=None, *, Si_Amp_13_cat, Ti_Amp_13_cat, Fet_Amp_13_cat, Mg_Amp_13_cat,
                     Ca_Amp_13_cat, K_Amp_13_cat, Na_Amp_13_cat, Al_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1c of Ridolfi and Renzulli (2012).
    Calibrated between 1.3-5 kbars
    :cite:`ridolfi2012calcic`
    '''
    return (0.01 * (24023.367332 - 1925.298250* Si_Amp_13_cat
    - 1720.63250944418 * Ti_Amp_13_cat - 1478.53847391822 * Al_Amp_13_cat
    - 1843.19249824537 * Fet_Amp_13_cat - 1746.94437497404 * Mg_Amp_13_cat
    - 158.279055907371 * Ca_Amp_13_cat - 40.4443246813322 * Na_Amp_13_cat
    + 253.51576430265 * K_Amp_13_cat))


def P_Ridolfi2012_1d(T=None, *, Si_Amp_13_cat, Ti_Amp_13_cat, Fet_Amp_13_cat, Mg_Amp_13_cat, Ca_Amp_13_cat,
                     K_Amp_13_cat, Na_Amp_13_cat, Al_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1d of Ridolfi and Renzulli (2012).
    Calibrated between 4-15 kbars
    :cite:`ridolfi2012calcic`
    '''
    return (0.01 * (26105.7092067 - 1991.93398583468 * Si_Amp_13_cat
    - 3034.9724955129 * Ti_Amp_13_cat - 1472.2242262718 * Al_Amp_13_cat - 2454.76485311127 * Fet_Amp_13_cat
    - 2125.79095875747 * Mg_Amp_13_cat - 830.644984403603 * Ca_Amp_13_cat
    + 2708.82902160291 * Na_Amp_13_cat + 2204.10480275638 * K_Amp_13_cat))


def P_Ridolfi2012_1e(T=None, *, Si_Amp_13_cat, Ti_Amp_13_cat, Fet_Amp_13_cat,
                     Mg_Amp_13_cat, Ca_Amp_13_cat, K_Amp_13_cat, Na_Amp_13_cat, Al_Amp_13_cat):
    '''
    Amphibole-only barometer: Equation 1e of Ridolfi and Renzulli (2012).
    Calibrated between 9.3-22 kbars
    :cite:`ridolfi2012calcic`
    '''
    return (0.01 * np.exp(26.5426319326957 - 1.20851740386237 * Si_Amp_13_cat
    - 3.85930939071001 * Ti_Amp_13_cat - 1.10536070667051 * Al_Amp_13_cat
    - 2.90677947035468 * Fet_Amp_13_cat - 2.64825741548332 *Mg_Amp_13_cat
    + 0.513357584438019 * Ca_Amp_13_cat
    + 2.9751971464851 * Na_Amp_13_cat + 1.81467032749331 * K_Amp_13_cat))


def P_Ridolfi2010(T=None, *, Al_Amp_cat_23ox, cation_sum_Si_Ca):
    '''
    Amphibole-only (Al) barometer: Ridolfi et al. (2010)
    :cite:`ridolfi2010stability`
    '''
    return (10 * (19.209 * np.exp(1.438 * Al_Amp_cat_23ox *
            13 / cation_sum_Si_Ca)) / 1000)


def P_Hammarstrom1986_eq1(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hammarstrom and Zen, 1986 eq.1
    :cite:`hammarstrom1986aluminum`
    '''
    return (-3.92 + 5.03 * Al_Amp_cat_23ox)


def P_Hammarstrom1986_eq2(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hammarstrom and Zen, 1986 eq.2
    :cite:`hammarstrom1986aluminum`
    '''
    return (1.27 * (Al_Amp_cat_23ox**2.01))


def P_Hammarstrom1986_eq3(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hammarstrom and Zen, 1986 eq.3
    :cite:`hammarstrom1986aluminum`
    '''
    return (0.26 * np.exp(1.48 * Al_Amp_cat_23ox))


def P_Hollister1987(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Hollister et al. 1987
    :cite:`hammarstrom1986aluminum`
    '''
    return (-4.76 + 5.64 * Al_Amp_cat_23ox)


def P_Johnson1989(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Johnson and Rutherford, 1989
    :cite:`hammarstrom1986aluminum`
    '''
    return (-3.46 + 4.23 * Al_Amp_cat_23ox)


def P_Anderson1995(T, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Anderson and Smith (1995)
    :cite:`hammarstrom1986aluminum`
    '''
    return (4.76 * Al_Amp_cat_23ox - 3.01 - (((T - 273.15 - 675) / 85)
            * (0.53 * Al_Amp_cat_23ox + 0.005294 * (T - 273.15 - 675))))


def P_Blundy1990(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Blundy et al. 1990
    :cite:`blundy1990calcic`
    '''
    return (5.03 * Al_Amp_cat_23ox - 3.53)


def P_Schmidt1992(T=None, *, Al_Amp_cat_23ox):
    '''
    Amphibole-only (Al) barometer: Schmidt 1992
    :cite:`schmidt1992amphibole`
    '''
    return (-3.01 + 4.76 * Al_Amp_cat_23ox)


def P_Medard2022_RidolfiSites(T=None,*,  amp_comps):
    """ Regression strategy of Medard 2022 linking AlVI to pressure,
    using site allocation strategy used by Ridolfi 2022 for consistency

    Statistics on calibration dataset:

    R2=0.93
    RMSE=93.48
    Int of reg=37.7
    Grad of reg=0.92


    """

    Sites_R=calculate_sites_ridolfi(amp_comps)
    P_Calc=817.22283194*Sites_R['Al_VI_C']+176.24032229
    return P_Calc/100



def P_Medard2022_LeakeSites(T=None, *,amp_comps):
    """ Regression strategy of Medard 2022 linking AlVI to pressure,
    using site allocation strategy used by Leake (implemented in Putirka) for consistency

    Statistics on calibration dataset:

    R2=0.94
    RMSE=87
    Int of reg=32
    Grad of reg=0.94


    """
    ox23=calculate_23oxygens_amphibole(amp_comps)
    Leake=get_amp_sites_leake(ox23)

    P_Calc=874.64558583*Leake['Al_C']+43.72682101
    return P_Calc/100


def P_Medard2022_MutchSites(T=None,*, amp_comps):
    """ Regression strategy of Medard 2022 linking AlVI to pressure,
    using site allocation strategy used by Mutch 2016 for consistency

    Statistics on calibration dataset:

    R2=0.93
    RMSE=93.9
    Int of reg=38.19
    Grad of reg=0.93


    """

    ox23=calculate_23oxygens_amphibole(amp_comps)
    Amp_sites_initial=get_amp_sites_mutch(ox23)
    norm_cat = amp_components_ferric_ferrous_mutch(Amp_sites_initial, ox23)
    Sites_M = get_amp_sites_ferric_ferrous_mutch(norm_cat)

    P_Calc=835.07125833*Sites_M['Al_C']+107.37542222
    return P_Calc/100


## Function: Amphibole-only barometry

Amp_only_P_funcs = { P_Ridolfi2012_1a, P_Ridolfi2012_1b, P_Ridolfi2012_1c, P_Ridolfi2012_1d,
P_Ridolfi2012_1e, P_Ridolfi2010, P_Hammarstrom1986_eq1, P_Hammarstrom1986_eq2, P_Hammarstrom1986_eq3, P_Hollister1987,
P_Johnson1989, P_Blundy1990, P_Schmidt1992, P_Anderson1995, P_Kraw2012, P_Medard2022_RidolfiSites,
P_Medard2022_LeakeSites, P_Medard2022_MutchSites} # put on outside

Amp_only_P_funcs_by_name= {p.__name__: p for p in Amp_only_P_funcs}

def calculate_amp_only_hygr(amp_comps=None, T=None):
    ''' Exists just to tell users to use a different function
    '''
    raise Exception('Please use calculate_amp_only_melt_comps, which will calculate H2O along with other melt composition parameters ')

def calculate_amp_only_melt_comps(amp_comps=None, T=None):
    '''
    Calculates melt compositions from Amphibole compositions using Zhang et al. (2017),
    Ridolfi et al. 2021 and Putirka (2016).

    Parameters
    -----------

    amp_comps: pandas.DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    T: optional, float, int
        Temperature in Kelvin, needed for equation3 and 5 of Zhang et al. (2017), and
        SiO2 from Putirka (2017)

    Returns
    -------
    DataFrame:
        Contains calculated melt compositions, and all amphibole components used
        in the calculation.

    '''



    # Calculations for Ridolfi
    amp_sites_R=calculate_sites_ridolfi(amp_comps=amp_comps)
    # Amp Sites for Zhang
    amp_sites=get_amp_sites_avferric_zhang(amp_comps=amp_comps)
    # For Putirka
    amp_23ox=calculate_23oxygens_amphibole(amp_comps=amp_comps)



    # Calculating Delta NNO from Ridolfi 2021

    deltaNNO_calc= (-10.3216023230583*amp_sites_R['Al_IV_T'] + 4.47045484316415*amp_sites_R['Al_VI_C']
    + 7.55122550171372*amp_sites_R['Ti_C'] + 5.46318534905121*amp_sites_R['Fe3_C'] -4.73884449358073*amp_sites_R['Mg_C']
        -7.20328571556139*amp_sites_R['Fe2_C']-17.5610110666215*amp_sites_R['Mn_C'] + 13.762022684517*amp_sites_R['Ca_B']
        + 13.7560270877436*amp_sites_R['Na_A']  + 27.5944871599305*amp_sites_R['K_A'])
    amp_sites.insert(0, "deltaNNO_Ridolfi21", deltaNNO_calc)

    # Calculating H2O form Ridofli 2021

    H2O_calc=(np.exp(-1.374845602*amp_sites_R['Al_IV_T'] + 1.7103210931239*amp_sites_R['Al_VI_C']
    + 0.85944576818503*amp_sites_R['Ti_C'] + 1.18881568772057*amp_sites_R['Fe3_C'] -0.675980097369545*amp_sites_R['Mg_C']
        -0.390086849565756*amp_sites_R['Fe2_C']-6.40208103925722*amp_sites_R['Mn_C'] + 2.54899046000297*amp_sites_R['Ca_B']
        + 1.37094801209146*amp_sites_R['Na_A']  + 1.25720999388625*amp_sites_R['K_A']))
    amp_sites.insert(0, "H2O_Ridolfi21", H2O_calc)


    if T is None:
        w.warn('You must enter a value for T in Kelvin to get results from equation3 and 5 from Zhang, and SiO2 from Putrka (2016)')

    amp_sites['SiO2_Eq1_Zhang17']=(-736.7170+288.733*np.log(amp_sites['Si_T_ideal'].astype(float))+56.536*amp_sites['Al_VI_C_ideal']
    +27.169*(amp_sites['Mg_C_ideal']+amp_sites['Mg_B_ideal'])
+ 62.665*amp_sites['Fe3_C_ideal']+34.814*(amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal'])
+83.989*(amp_sites['Ti_T_ideal']+amp_sites['Ti_C_ideal'])+44.225*amp_sites['Ca_B_ideal']+14.049*amp_sites['Na_A_ideal'])

    amp_sites['SiO2_Eq2_Zhang17']=(-399.9891 + 212.9463*np.log(amp_sites['Si_T_ideal'].astype(float)) + 11.7464*amp_sites['Al_VI_C_ideal'] +
    23.5653*amp_sites['Fe3_C_ideal'] + 6.8467*(amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal']) +
    24.7743*(amp_sites['Ti_T_ideal']+amp_sites['Ti_C_ideal']) + 24.4399 * amp_sites['Ca_B_ideal'])

    amp_sites['SiO2_Eq4_Zhang17']=(-222.614 + 167.517*np.log(amp_sites['Si_T_ideal'].astype(float)) -7.156*(amp_sites['Mg_C_ideal']
    +amp_sites['Mg_B_ideal']))

    amp_sites['TiO2_Eq6_Zhang17']=(np.exp(22.4650  -2.5975*amp_sites['Si_T_ideal']
        -1.15502*amp_sites['Al_VI_C_ideal'] -2.23287*amp_sites['Fe3_C_ideal'] -1.03193*(amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal'])
        -1.98253*amp_sites['Ca_B_ideal']-1.55912*amp_sites['Na_A_ideal']))

    amp_sites['FeO_Eq7_Zhang17']=(np.exp(24.4613  -2.72308*amp_sites['Si_T_ideal']
        -1.07345*amp_sites['Al_VI_C_ideal'] -1.0466*amp_sites['Fe3_C_ideal'] -0.25801*(amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal'])
        -1.93601*amp_sites['Ti_C_ideal']-2.52281*amp_sites['Ca_B_ideal']))

    amp_sites['FeO_Eq8_Zhang17']=(np.exp(15.6864  -2.09657*amp_sites['Si_T_ideal']
        +0.36457*amp_sites['Mg_C_ideal'] -1.33131*amp_sites['Ca_B_ideal']))

    amp_sites['MgO_Eq9_Zhang17']=(np.exp(12.6618  -2.63189*amp_sites['Si_T_ideal']
        +1.04995*amp_sites['Al_VI_C_ideal'] +1.26035*amp_sites['Mg_C_ideal']))

    amp_sites['CaO_Eq10_Zhang17']=(41.2784  -7.1955*amp_sites['Si_T_ideal']
        +3.6412*amp_sites['Mg_C_ideal'] -5.0437*amp_sites['Na_A_ideal'])

    amp_sites['CaO_Eq11_Zhang17']=np.exp((6.4192  -1.17372*amp_sites['Si_T_ideal']
        +1.31976*amp_sites['Al_VI_C_ideal'] +0.67733*amp_sites['Mg_C_ideal']))

    amp_sites['K2O_Eq12_Zhang17']=(100.5909  -4.3246*amp_sites['Si_T_ideal']
        -17.8256*amp_sites['Al_VI_C_ideal']-10.0901*amp_sites['Mg_C_ideal'] -15.683*amp_sites['Fe3_C_ideal']
        -8.8004*(amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal'])-19.7448*amp_sites['Ti_C_ideal']
        -6.3727*amp_sites['Ca_B_ideal']-5.8069*amp_sites['Na_A_ideal'])

    amp_sites['K2O_Eq13_Zhang17']=(-16.53  +1.6878*amp_sites['Si_T_ideal']
        +1.2354*(amp_sites['Fe3_C_ideal']+amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal'])
        +5.0404*amp_sites['Ti_C_ideal']+2.9703*amp_sites['Ca_B_ideal'])

    amp_sites['Al2O3_Eq14_Zhang17']=(4.573 + 6.9408*amp_sites['Al_VI_C_ideal']+1.0059*amp_sites['Mg_C_ideal']
        +4.5448*amp_sites['Fe3_C_ideal']+5.9679*amp_sites['Ti_C_ideal']
        +7.1501*amp_sites['Na_A_ideal'])

    cols_to_move = ['SiO2_Eq1_Zhang17', 'SiO2_Eq2_Zhang17', "SiO2_Eq4_Zhang17", "TiO2_Eq6_Zhang17",
                    'FeO_Eq7_Zhang17', 'MgO_Eq9_Zhang17', 'CaO_Eq10_Zhang17', 'CaO_Eq11_Zhang17', 'K2O_Eq12_Zhang17',
                        'K2O_Eq13_Zhang17', 'Al2O3_Eq14_Zhang17']
    amp_sites= amp_sites[cols_to_move +
                                    [col for col in amp_sites.columns if col not in cols_to_move]]


    if T is not None:
        SiO2_Eq3=(-228 + 0.01065*(T-273.15) + 165*np.log(amp_sites['Si_T_ideal'].astype(float))
        -7.219*(amp_sites['Mg_C_ideal']+amp_sites['Mg_B_ideal']))
        amp_sites.insert(4, "SiO2_Eq3_Zhang17", SiO2_Eq3)

        TiO2_Eq5=(np.exp( 23.4870 -0.0011*(T-273.15) +-2.5692*amp_sites['Si_T_ideal']
        -1.3919*amp_sites['Al_VI_C_ideal'] -2.1195361*amp_sites['Fe3_C_ideal'] -1.0510775*(amp_sites['Fe2_C_ideal']+amp_sites['Fe2_B_ideal'])
        -2.0634034*amp_sites['Ca_B_ideal']-1.5960633*amp_sites['Na_A_ideal']))
        amp_sites.insert(6, "TiO2_Eq5_Zhang17", TiO2_Eq5)

        # Putirka 2016 equation 10
        SiO2_Put2016=751.95-0.4*(T-273.15)-278000/(T-273.15)-9.184*amp_23ox['Al_Amp_cat_23ox']
        amp_sites.insert(0, 'SiO2_Eq10_Put2016', SiO2_Put2016)




    return amp_sites



def calculate_amp_only_press(amp_comps=None, equationP=None, T=None, deltaNNO=None,
classification=False, Ridolfi_Filter=True):

    """
    Amphibole-only barometry, returns pressure in kbar.

    Parameters
    -----------

    amp_comps: pandas.DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.


    EquationP: str
        | P_Mutch2016 (T-independent)
        | P_Ridolfi2012_1a (T-independent)
        | P_Ridolfi2012_1b (T-independent)
        | P_Ridolfi2012_1c (T-independent)
        | P_Ridolfi2012_1d (T-independent)
        | P_Ridolfi2012_1e (T-independent)
        | P_Ridolfi2021 - (T-independent)- Uses new algorithm in 2021 paper to
        select pressures from equations 1a-e.
        | P_Medard2022. Choose how you want the sites calculated:
            P_Medard2022_RidolfiSites, LeakeSites, MutchSites

        | P_Ridolfi2010  (T-independent)
        | P_Hammarstrom1986_eq1  (T-independent)
        | P_Hammarstrom1986_eq2 (T-independent)
        | P_Hammarstrom1986_eq3 (T-independent)
        | P_Hollister1987 (T-independent)
        | P_Johnson1989 (T-independent)
        | P_Blundy1990 (T-independent)
        | P_Schmidt1992 (T-independent)
        | P_Anderson1995 (*T-dependent*)

    T: float, int, pandas.Series, str  ("Solve")
        Temperature in Kelvin
        Only needed for T-sensitive barometers.
        If enter T="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas series
       Pressure in kbar
    """

    if equationP !="P_Ridolfi2021" and equationP !="P_Mutch2016":
        try:
            func = Amp_only_P_funcs_by_name[equationP]
        except KeyError:
            raise ValueError(f'{equationP} is not a valid equation') from None
        sig=inspect.signature(func)

        if sig.parameters['T'].default is not None:
            if T is None:
                raise ValueError(f'{equationP} requires you to enter T, or specify T="Solve"')
        else:
            if T is not None:
                print('Youve selected a T-independent function')

        if isinstance(T, pd.Series):
            if amp_comps is not None:
                if len(T) != len(amp_comps):
                    raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of amphibole compositions')


    if classification is True:
       name= pd.DataFrame(index=range(0,len(amp_comps)),columns=['classification_Ridolfi21', 'series'], dtype='str')
       name = calculate_sites_ridolfi(amp_comps).classification


    if equationP=="P_Medard2022_RidolfiSites":
        df_out=P_Medard2022_RidolfiSites(amp_comps=amp_comps)
        return df_out

    if equationP=="P_Medard2022_MutchSites":
        df_out=P_Medard2022_MutchSites(amp_comps=amp_comps)
        return df_out

    if equationP=="P_Medard2022_LeakeSites":
        df_out=P_Medard2022_LeakeSites(amp_comps=amp_comps)
        return df_out


    if equationP == "P_Kraw2012":
        w.warn('This barometer gives the PH2O for the first appearance of'
        ' amphibole. It should only be applied to the highest Mg# in each'
        ' sample suite. Note, if there is CO2 in the system P=/ PH2O')
        if deltaNNO is None:
            raise ValueError('P_Kraw2012 requires you to enter a deltaNNO value')
        Mgno_Amp=100*(amp_comps['MgO_Amp']/40.3044)/((amp_comps['MgO_Amp']/40.3044)+(amp_comps['FeOt_Amp']/71.844))
        P_kbar=P_Kraw2012(Mgno_Amp=Mgno_Amp,
        deltaNNO=deltaNNO)
        df_out=pd.DataFrame(data={'PH2O_kbar_calc': P_kbar,
        'Mg#_Amp': Mgno_Amp})
        return df_out


    if "Sample_ID_Amp" not in amp_comps:
        amp_comps['Sample_ID_Amp'] = amp_comps.index

    if equationP == "P_Mutch2016":
        # In spreadsheet provided by Mutch, doesnt use Cl and F for calcs.
        amp_comps_noHalogens=amp_comps.copy()
        amp_comps_noHalogens['Cl_Amp']=0
        amp_comps_noHalogens['F_Amp']=0
        ox23 = calculate_23oxygens_amphibole(amp_comps_noHalogens)
        Amp_sites_initial = get_amp_sites_mutch(ox23)
        norm_cat = amp_components_ferric_ferrous_mutch(Amp_sites_initial, ox23)
        final_cat = get_amp_sites_ferric_ferrous_mutch(norm_cat)
        final_cat['Al_tot'] = final_cat['Al_T'] + final_cat['Al_C']
        P_kbar = 0.5 + 0.331 * \
            final_cat['Al_tot'] + 0.995 * (final_cat['Al_tot'])**2
        final_cat.insert(0, "P_kbar_calc", P_kbar)
        if classification is True:
            final_cat.insert(1, "classification", name)
        return final_cat

    if 'Ridolfi2012' in equationP or equationP == "P_Ridolfi2021":

        cat13 = calculate_sites_ridolfi(amp_comps)
        Sum_input=cat13['Sum_input']


        kwargs_1a = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1a).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1b = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1b).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1c = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1c).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1d = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1d).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
        kwargs_1e = {name: cat13[name] for name, p in inspect.signature(
            P_Ridolfi2012_1e).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

        P_MPa_1a = 100 * partial(P_Ridolfi2012_1a, **kwargs_1a)(0)
        P_MPa_1b = 100 * partial(P_Ridolfi2012_1b, **kwargs_1b)(0)
        P_MPa_1c = 100 * partial(P_Ridolfi2012_1c, **kwargs_1c)(0)
        P_MPa_1d = 100 * partial(P_Ridolfi2012_1d, **kwargs_1d)(0)
        P_MPa_1e = 100 * partial(P_Ridolfi2012_1e, **kwargs_1e)(0)

        if equationP == "P_Ridolfi2021":


            XPae = (P_MPa_1a - P_MPa_1e) / P_MPa_1a
            deltaPdb = P_MPa_1d - P_MPa_1b

            P_MPa = np.empty(len(P_MPa_1a))
            name=np.empty(len(P_MPa_1a), dtype=np.dtype('U100'))
            for i in range(0, len(P_MPa_1a)):
                if P_MPa_1b[i] < 335:
                    P_MPa[i] = P_MPa_1b[i]
                    name[i]="1b"
                elif P_MPa_1b[i] < 399:
                    P_MPa[i] = (P_MPa_1b[i] + P_MPa_1c[i]) / 2
                    name[i]="(1b+1c)/2"
                elif P_MPa_1c[i] < 415:
                    P_MPa[i] = (P_MPa_1c[i])
                    name[i]="1c"
                elif P_MPa_1d[i] < 470:
                    P_MPa[i] = (P_MPa_1c[i])
                    name[i]="1c"
                elif XPae[i] > 0.22:
                    P_MPa[i] = (P_MPa_1c[i] + P_MPa_1d[i]) / 2
                    name[i]="1c+1d"
                elif deltaPdb[i] > 350:
                    P_MPa[i] = P_MPa_1e[i]
                    name[i]="1e"
                elif deltaPdb[i] > 210:
                    P_MPa[i] = P_MPa_1d[i]
                    name[i]="1d"
                elif deltaPdb[i] < 75:
                    P_MPa[i] = P_MPa_1c[i]
                    name[i]="1c"
                elif XPae[i] < -0.2:
                    P_MPa[i] = (P_MPa_1b[i] + P_MPa_1c[i]) / 2
                    name[i]="(1b+1c)/2"
                elif XPae[i] > 0.05:
                    P_MPa[i] = (P_MPa_1c[i] + P_MPa_1d[i]) / 2
                    name[i]="(1c+1d)/2"
                else:
                    P_MPa[i] = P_MPa_1a[i]
                    name[i]="1a"

                if Sum_input[i] < 90:
                    P_MPa[i] = np.nan


            Calcs_R=cat13.copy()
            Calcs_R['P_kbar_calc']=P_MPa / 100
            Calcs_R['equation']=name
            Calcs_R['Sum_input']=Sum_input
            Low_sum=Sum_input<90

            Calcs_R['APE']=np.abs(P_MPa_1a-P_MPa)/(P_MPa_1a+P_MPa)*200
            High_APE=Calcs_R['APE']>60
            Calcs_R.loc[(High_APE), 'Input_Check']=False
            Calcs_R.loc[(High_APE), 'Fail Msg']="APE >60"

            if Ridolfi_Filter is True:
                Failed_input=Calcs_R['Input_Check']==False
                Calcs_R.loc[Failed_input, 'P_kbar_calc']=np.nan

            cols_to_move = ['P_kbar_calc', 'Input_Check', "Fail Msg", "classification",
                            'equation', 'H2O_calc', 'Fe2O3_calc', 'FeO_calc', 'Total_recalc', 'Sum_input']
            Calcs_R= Calcs_R[cols_to_move +
                                            [col for col in Calcs_R.columns if col not in cols_to_move]]


            return Calcs_R # was P_kbar

        if equationP == "P_Ridolfi2012_1a":
            P_kbar = P_MPa_1a / 100


        if equationP == "P_Ridolfi2012_1b":
            P_kbar = P_MPa_1b / 100


        if equationP == "P_Ridolfi2012_1c":
            P_kbar = P_MPa_1c / 100


        if equationP == "P_Ridolfi2012_1d":
            P_kbar = P_MPa_1d / 100


        if equationP == "P_Ridolfi2012_1e":
            P_kbar = P_MPa_1e / 100

        if classification is False:
            return P_kbar

        if classification is True:
            p_name=pd.DataFrame(data={"P_kbar_calc": P_kbar, "classification":name})
            return p_name





    if equationP != "Mutch2016" and 'Ridolfi2012' not in equationP and  equationP != "P_Ridolfi2021":
        ox23_amp = calculate_23oxygens_amphibole(amp_comps=amp_comps)

    kwargs = {name: ox23_amp[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(T, str) or T is None:
        if T == "Solve":
            P_kbar = partial(func, **kwargs)
        if T is None:
            P_kbar=func(**kwargs)

    else:
        P_kbar=func(T, **kwargs)

    if classification is False:
        return P_kbar

    if classification is True:
        p_name=pd.DataFrame(data={"P_kbar_calc": P_kbar, "classification":name})
        return p_name

def calculate_amp_only_press_all_eqs(amp_comps, plot=False, H2O_Liq=None, Ridolfi_Filter=True):
    import warnings
    with w.catch_warnings():
        w.simplefilter('ignore')
        amp_comps_c=amp_comps.copy()
        #amp_comps_c=get_amp_sites_from_input(amp_comps=amp_comps)
        amp_calcs=calculate_amp_only_press(amp_comps=amp_comps_c,
        equationP="P_Ridolfi2021", Ridolfi_Filter=Ridolfi_Filter)
        T_calc=calculate_amp_only_press_temp(amp_comps=amp_comps_c,
        equationP="P_Ridolfi2021", equationT="T_Ridolfi2012", Ridolfi_Filter=Ridolfi_Filter).T_K_calc
        amp_calcs["P_Ridolfi21"]=amp_calcs['P_kbar_calc']
        amp_calcs["T_Ridolfi12"]=T_calc
        X_Ridolfi21_Sorted=np.sort(amp_calcs['P_Ridolfi21'])
        if plot==True:
            plt.step(np.concatenate([X_Ridolfi21_Sorted, X_Ridolfi21_Sorted[[-1]]]),
            np.arange(X_Ridolfi21_Sorted.size+1)/X_Ridolfi21_Sorted.size, color='blue', linewidth=1,
            label="Ridolfi21")

        out=pd.concat([amp_calcs, amp_comps_c], axis=1)


    return out

def calculate_amp_liq_all_equations(amp_comps, liq_comps, H2O_Liq=None):

    if len(amp_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as Amp comps. If you want to match up all possible pairs, use the _matching functions instead')

    # Amp-Liq temp, not P sensitive, but are H2O sensitive

    CalcT_4a=calculate_amp_liq_temp(amp_comps=amp_comps,
liq_comps=liq_comps, H2O_Liq=H2O_Liq, equationT="T_Put2016_eq4a_amp_sat")

    CalcT_4b=calculate_amp_liq_temp(amp_comps=amp_comps,
liq_comps=liq_comps, H2O_Liq=H2O_Liq, equationT="T_Put2016_eq4b")

    CalcT_9=calculate_amp_liq_temp(amp_comps=amp_comps,
liq_comps=liq_comps, H2O_Liq=H2O_Liq, equationT="T_Put2016_eq9")

    # Amp-Liq pressure - Eq 7a, 7b, and Eq7c, all H2O sensitive, but T not

    CalcP_eq7a=calculate_amp_liq_press(amp_comps=amp_comps,
liq_comps=liq_comps, H2O_Liq=H2O_Liq,  equationP="P_Put2016_eq7a")

    CalcP_eq7b=calculate_amp_liq_press(amp_comps=amp_comps,
liq_comps=liq_comps, H2O_Liq=H2O_Liq,  equationP="P_Put2016_eq7b")

    CalcP_eq7c=calculate_amp_liq_press(amp_comps=amp_comps,
liq_comps=liq_comps, H2O_Liq=H2O_Liq,  equationP="P_Put2016_eq7c")

    df_out=pd.DataFrame(data={'CalcT_4a_amp_sat (P-ind)': CalcT_4a,
                            'CalcT_4b (P-ind)': CalcT_4b,
                            'CalcT_9 (P-ind)': CalcT_9,
                            'CalcP_7a (T-ind)': CalcP_eq7a,
                            'CalcP_7b (T-ind)': CalcP_eq7b,
                            'CalcP_7c (T-ind)': CalcP_eq7c,
                                })

    # Eq4b for T,

    return df_out


## Amphibole-only thermometers


def T_Put2016_eq5(P=None, *, Si_Amp_cat_23ox,
                  Ti_Amp_cat_23ox, Fet_Amp_cat_23ox, Na_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Equation 5 of Putirka et al. (2016)
    :cite:`putirka2016amphibole`
    '''
    return (273.15 + 1781 - 132.74 * Si_Amp_cat_23ox + 116.6 *
            Ti_Amp_cat_23ox - 69.41 * Fet_Amp_cat_23ox + 101.62 * Na_Amp_cat_23ox)


def T_Put2016_eq6(P, *, Si_Amp_cat_23ox,
                  Ti_Amp_cat_23ox, Fet_Amp_cat_23ox, Na_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Equation 6 of Putirka et al. (2016)
    :cite:`putirka2016amphibole`
    '''
    return (273.15 + 1687 - 118.7 * Si_Amp_cat_23ox + 131.56 * Ti_Amp_cat_23ox -
            71.41 * Fet_Amp_cat_23ox + 86.13 * Na_Amp_cat_23ox + 22.44 * P / 10)


def T_Put2016_SiHbl(P=None, *, Si_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Si in Hbl, Putirka et al. (2016)
    :cite:`putirka2016amphibole`
    '''
    return (273.15 + 2061 - 178.4 * Si_Amp_cat_23ox)

def T_Ridolfi2012(P, *, Si_Amp_13_cat, Ti_Amp_13_cat, Fet_Amp_13_cat,
                  Mg_Amp_13_cat, Ca_Amp_13_cat, K_Amp_13_cat, Na_Amp_13_cat, Al_Amp_13_cat):
    '''
    Amphibole-only thermometer of Ridolfi and Renzuli, 2012
    :cite:`ridolfi2012calcic`

    SEE=22C
    '''
    return (273.15 + 8899.682 - 691.423 * Si_Amp_13_cat - 391.548 * Ti_Amp_13_cat - 666.149 * Al_Amp_13_cat
    - 636.484 * Fet_Amp_13_cat -584.021 * Mg_Amp_13_cat - 23.215 * Ca_Amp_13_cat
    + 79.971 * Na_Amp_13_cat - 104.134 * K_Amp_13_cat + 78.993 * np.log(P * 100))

def T_Put2016_eq8(P, *, Si_Amp_cat_23ox, Ti_Amp_cat_23ox,
                  Mg_Amp_cat_23ox, Na_Amp_cat_23ox):
    '''
    Amphibole-only thermometer: Eq8,  Putirka et al. (2016)
    :cite:`putirka2016amphibole`
    '''
    return (273.15+1201.4 - 97.93 * Si_Amp_cat_23ox + 201.82 * Ti_Amp_cat_23ox +
            72.85 * Mg_Amp_cat_23ox + 88.9 * Na_Amp_cat_23ox + 40.65 * P / 10)
## Equations: Amphibole-Liquid barometers

def P_Put2016_eq7a(T=None, *, Al_Amp_cat_23ox, Na_Amp_cat_23ox,
K_Amp_cat_23ox, Al2O3_Liq_mol_frac_hyd, Na2O_Liq_mol_frac_hyd,
H2O_Liq_mol_frac_hyd, P2O5_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid barometer: Equation 7a of Putirka et al. (2016)
    Preferred equation
    :cite:`putirka2016amphibole`

    '''
    # print('Note - Putirka 2016 spreadsheet calculates H2O using a H2O-solubility law of uncertian origin based on the pressure calculated for 7a, and iterates H"O and P. We dont do this, as we dont believe a pure h2o model is necessarily valid as you may be mixed fluid saturated or undersaturated. We recomend instead you choose a reasonable H2O content based on your system.')
    return (10 * (-3.093 - 4.274 * np.log(Al_Amp_cat_23ox.astype(float) / Al2O3_Liq_mol_frac_hyd.astype(float))
    - 4.216 * np.log(Al2O3_Liq_mol_frac_hyd.astype(float)) + 63.3 * P2O5_Liq_mol_frac_hyd +
    1.264 * H2O_Liq_mol_frac_hyd + 2.457 * Al_Amp_cat_23ox + 1.86 * K_Amp_cat_23ox
    + 0.4 * np.log(Na_Amp_cat_23ox.astype(float) / Na2O_Liq_mol_frac_hyd.astype(float))))


def P_Put2016_eq7b(T=None, *, Al2O3_Liq_mol_frac_hyd, P2O5_Liq_mol_frac_hyd, Al_Amp_cat_23ox,
    SiO2_Liq_mol_frac_hyd, Na2O_Liq_mol_frac_hyd, K2O_Liq_mol_frac_hyd, CaO_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid barometer: Equation 7b of Putirka et al. (2016)
    While 7a is preferred, Putirka (2008) say that 7b may be more precise at low T, and >10 kbar
    :cite:``

    '''
    return (-64.79 - 6.064 * np.log(Al_Amp_cat_23ox.astype(float) / Al2O3_Liq_mol_frac_hyd.astype(float))
    + 61.75 * SiO2_Liq_mol_frac_hyd + 682 * P2O5_Liq_mol_frac_hyd
    - 101.9 *CaO_Liq_mol_frac_hyd + 7.85 * Al_Amp_cat_23ox
    - 46.46 * np.log(SiO2_Liq_mol_frac_hyd.astype(float))
    - 4.81 * np.log(Na2O_Liq_mol_frac_hyd.astype(float) + K2O_Liq_mol_frac_hyd.astype(float)))


def P_Put2016_eq7c(T=None, *, Al_Amp_cat_23ox, K_Amp_cat_23ox,
                   P2O5_Liq_mol_frac, Al2O3_Liq_mol_frac, Na_Amp_cat_23ox, Na2O_Liq_mol_frac):
    '''
    Amphibole-Liquid barometer: Equation 7c of Putirka et al. (2016)
    :cite:`putirka2016amphibole`

    '''
    return (-45.55 + 26.65 * Al_Amp_cat_23ox + 22.52 * K_Amp_cat_23ox
    + 439 * P2O5_Liq_mol_frac - 51.1 * np.log(Al2O3_Liq_mol_frac.astype(float)) -
    46.3 * np.log(Al_Amp_cat_23ox.astype(float) / (Al2O3_Liq_mol_frac.astype(float)))
    + 5.231 * np.log(Na_Amp_cat_23ox.astype(float) / (Na2O_Liq_mol_frac.astype(float))))

## Equations: Amphibole-Liquid thermometers


def T_Put2016_eq4b(P=None, *, H2O_Liq_mol_frac_hyd, Fet_Amp_cat_23ox, FeOt_Liq_mol_frac_hyd, MgO_Liq_mol_frac_hyd,
                   MnO_Liq_mol_frac_hyd, Al2O3_Liq_mol_frac_hyd, Ti_Amp_cat_23ox, TiO2_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid thermometer: Eq4b,  Putirka et al. (2016)
    :cite:`putirka2016amphibole`

    '''
    return (273.15 + (8037.85 / (3.69 - 2.62 * H2O_Liq_mol_frac_hyd + 0.66 * Fet_Amp_cat_23ox
    - 0.416 * np.log(TiO2_Liq_mol_frac_hyd.astype(float)) + 0.37 * np.log(MgO_Liq_mol_frac_hyd.astype(float))
    -1.05 * np.log((FeOt_Liq_mol_frac_hyd.astype(float) + MgO_Liq_mol_frac_hyd.astype(float)
    + MnO_Liq_mol_frac_hyd.astype(float)) * Al2O3_Liq_mol_frac_hyd)
    - 0.462 * np.log(Ti_Amp_cat_23ox.astype(float) / TiO2_Liq_mol_frac_hyd.astype(float)))))


def T_Put2016_eq4a_amp_sat(P=None, *, FeOt_Liq_mol_frac_hyd, TiO2_Liq_mol_frac_hyd, Al2O3_Liq_mol_frac_hyd,
                           MnO_Liq_mol_frac_hyd, MgO_Liq_mol_frac_hyd, Na_Amp_cat_23ox, Na2O_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid thermometer Saturation surface of amphibole, Putirka et al. (2016)
    :cite:`putirka2016amphibole`

    '''
    return (273.15 + (6383.4 / (-12.07 + 45.4 * Al2O3_Liq_mol_frac_hyd + 12.21 * FeOt_Liq_mol_frac_hyd -
    0.415 * np.log(TiO2_Liq_mol_frac_hyd.astype(float)) - 3.555 * np.log(Al2O3_Liq_mol_frac_hyd.astype(float))
     - 0.832 * np.log(Na2O_Liq_mol_frac_hyd.astype(float)) -0.481 * np.log((FeOt_Liq_mol_frac_hyd.astype(float)
     + MgO_Liq_mol_frac_hyd.astype(float) + MnO_Liq_mol_frac_hyd.astype(float)) * Al2O3_Liq_mol_frac_hyd.astype(float))
     - 0.679 * np.log(Na_Amp_cat_23ox.astype(float) / Na2O_Liq_mol_frac_hyd.astype(float)))))


def T_Put2016_eq9(P=None, *, Si_Amp_cat_23ox, Ti_Amp_cat_23ox, Mg_Amp_cat_23ox,
Fet_Amp_cat_23ox, Na_Amp_cat_23ox,  FeOt_Liq_mol_frac_hyd, Al_Amp_cat_23ox, Al2O3_Liq_mol_frac_hyd,
K_Amp_cat_23ox, Ca_Amp_cat_23ox, Na2O_Liq_mol_frac_hyd, K2O_Liq_mol_frac_hyd):
    '''
    Amphibole-Liquid thermometer: Eq9,  Putirka et al. (2016)
    :cite:`putirka2016amphibole`


    '''
    NaM4_1=2-Fet_Amp_cat_23ox-Ca_Amp_cat_23ox
    NaM4=np.empty(len(NaM4_1))
    for i in range(0, len(NaM4)):
        if NaM4_1[i]<=0.1:
            NaM4[i]=0
        else:
            NaM4[i]=NaM4_1[i]

    HelzA=Na_Amp_cat_23ox-NaM4
    ln_KD_Na_K=np.log((K_Amp_cat_23ox.astype(float)/HelzA.astype(float))*(Na2O_Liq_mol_frac_hyd.astype(float)/K2O_Liq_mol_frac_hyd.astype(float)))

    return (273.15+(10073.5/(9.75+0.934*Si_Amp_cat_23ox-1.454*Ti_Amp_cat_23ox
    -0.882*Mg_Amp_cat_23ox-1.123*Na_Amp_cat_23ox-0.322*np.log(FeOt_Liq_mol_frac_hyd.astype(float))
    -0.7593*np.log(Al_Amp_cat_23ox.astype(float)/Al2O3_Liq_mol_frac_hyd.astype(float))-0.15*ln_KD_Na_K)))



## Function: Amphibole-only temperature

Amp_only_T_funcs = {T_Put2016_eq5, T_Put2016_eq6, T_Put2016_SiHbl, T_Put2016_eq8,
 T_Ridolfi2012, T_Put2016_eq4a_amp_sat, T_Put2016_eq8} # put on outside

Amp_only_T_funcs_by_name= {p.__name__: p for p in Amp_only_T_funcs}




def calculate_amp_only_temp(amp_comps, equationT, P=None):
    '''
    Amphibole-only thermometry, calculates temperature in Kelvin.

    Parameters
    -----------

    equationT: str
        choose from:

        |   T_Put2016_eq5 (P-independent)
        |   T_Put2016_eq6 (P-dependent)
        |   T_Put2016_SiHbl (P-independent)
        |   T_Ridolfi2012 (P-dependent)
        |   T_Put2016_eq8 (P-dependent)


    P: float, int, pandas.Series, str
        Pressure in kbar to perform calculations at,
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    Returns
    -------
    pandas.Series: Pressure in kbar (if eq_tests=False)

    '''
    try:
        func = Amp_only_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')
    else:
        if P is not None:
            print('Youve selected a P-independent function')

    if isinstance(P, pd.Series):
        if amp_comps is not None:
            if len(P) != len(amp_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of amphibole compositions')


    if equationT == "T_Ridolfi2012":
        if P is None:
            raise Exception(
                'You have selected a P-dependent thermometer, please enter an option for P')
        cat13 = calculate_13cations_amphibole_ridolfi(amp_comps)
        myAmps1_label = amp_comps.drop(['Sample_ID_Amp'], axis='columns')
        Sum_input = myAmps1_label.sum(axis='columns')

        kwargs = {name: cat13[name] for name, p in inspect.signature(
            T_Ridolfi2012).parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}

    else:
        amp_comps =calculate_23oxygens_amphibole(amp_comps=amp_comps)
        kwargs = {name: amp_comps[name] for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY}


    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:
        T_K=func(P, **kwargs)

    return T_K


## Function: PT Iterate Amphibole - only

def calculate_amp_only_press_temp(amp_comps, equationT, equationP, iterations=30,
T_K_guess=1300, Ridolfi_Filter=True, return_amps=True, deltaNNO=None):
    '''
    Solves simultaneous equations for temperature and pressure using
    amphibole only thermometers and barometers.


   Parameters
    -------

    amp_comps: pandas.DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    EquationP: str

        | P_Mutch2016 (T-independent)
        | P_Ridolfi2012_1a (T-independent)
        | P_Ridolfi2012_1b (T-independent)
        | P_Ridolfi2012_1c (T-independent)
        | P_Ridolfi2012_1d (T-independent)
        | P_Ridolfi2012_1e (T-independent)
        | P_Ridolfi2021 - (T-independent)- Uses new algorithm in 2021 paper to
        select pressures from equations 1a-e.

        | P_Ridolfi2010  (T-independent)
        | P_Hammarstrom1986_eq1  (T-independent)
        | P_Hammarstrom1986_eq2 (T-independent)
        | P_Hammarstrom1986_eq3 (T-independent)
        | P_Hollister1987 (T-independent)
        | P_Johnson1989 (T-independent)
        | P_Blundy1990 (T-independent)
        | P_Schmidt1992 (T-independent)
        | P_Anderson1995 (*T-dependent*)

    equationT: str
        |   T_Put2016_eq5 (P-independent)
        |   T_Put2016_eq6 (P-dependent)
        |   T_Put2016_SiHbl (P-independent)
        |   T_Ridolfi2012 (P-dependent)
        |   T_Put2016_eq8 (P-dependent)

    H2O_Liq: float, int, pandas.Series, optional
        Needed if you select P_Put2008_eq32b, which is H2O-dependent.

    Optional:

     iterations: int, default=30
         Number of iterations used to converge to solution.

     T_K_guess: int or float. Default is 1300 K
         Initial guess of temperature.


    Returns:
    -------
    pandas.DataFrame: Pressure in Kbar, Temperature in K
    '''
    T_func = calculate_amp_only_temp(amp_comps=amp_comps, equationT=equationT, P="Solve")
    if equationP !="P_Ridolfi2021" and equationP != "P_Mutch2016" and equationP!= "P_Kraw2012":
        P_func = calculate_amp_only_press(amp_comps=amp_comps, equationP=equationP, T="Solve")

    # If mutch, need to extract P from dataframe.
    if equationP == "P_Mutch2016":
        P_func = calculate_amp_only_press(amp_comps=amp_comps, equationP=equationP, T="Solve").P_kbar_calc

    if equationP == "P_Kraw2012":
        P_func = calculate_amp_only_press(amp_comps=amp_comps, equationP=equationP, T="Solve", deltaNNO=deltaNNO).PH2O_kbar_calc

    # If Ridolfi, need to extract Pkbar, as well as warning messages.
    if equationP == "P_Ridolfi2021":
        P_func_all = calculate_amp_only_press(amp_comps=amp_comps, equationP=equationP,
        T="Solve", Ridolfi_Filter=Ridolfi_Filter)
        P_func=P_func_all.P_kbar_calc



    if isinstance(T_func, pd.Series) and isinstance(P_func, pd.Series):
        P_guess = P_func
        T_K_guess = T_func

    if isinstance(T_func, pd.Series) and isinstance(P_func, partial):
        P_guess = P_func(T_func)
        T_K_guess = T_func

    if isinstance(P_func, pd.Series) and isinstance(T_func, partial):
        T_K_guess = T_func(P_func)
        P_guess = P_func

    if isinstance(P_func, partial) and isinstance(T_func, partial):

        count=0
        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)
            if count==iterations-2:
                # On the second last step, save the pressure
                P_out_loop=P_guess.values
                T_out_loop=T_K_guess.values
            count=count+1

        DeltaP=P_guess-P_out_loop
        DeltaT=T_K_guess-T_out_loop


    else:
        DeltaP=0
        DeltaT=0


    if equationP=="P_Ridolfi2021":
        PT_out=pd.DataFrame(data={'P_kbar_calc': P_guess,
         'T_K_calc': T_K_guess,
         'Input_Check': P_func_all.Input_Check,
         'Fail Msg': P_func_all['Fail Msg'],
         'classification': P_func_all['classification'],
         'equation': P_func_all['equation']

         })

    else:
        PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                    'T_K_calc': T_K_guess,
                                    'Delta_P_kbar_Iter': DeltaP,
                                    'Delta_T_K_Iter': DeltaT})
    if return_amps is True:
        PT_out2=pd.concat([PT_out, amp_comps], axis=1)
        return PT_out2
    else:
        return PT_out

## Function: Amphibole-Liquid barometer
Amp_Liq_P_funcs = {P_Put2016_eq7a, P_Put2016_eq7b, P_Put2016_eq7c}

Amp_Liq_P_funcs_by_name = {p.__name__: p for p in Amp_Liq_P_funcs}


def calculate_amp_liq_press(*, amp_comps=None, liq_comps=None,
                            meltmatch=None, equationP=None, T=None,
                             eq_tests=False, H2O_Liq=None):
    '''
    Amphibole-liquid barometer. Returns pressure in kbar

   Parameters
    -------

    amp_comps: pandas DataFrame
        amphibole compositions (SiO2_Amp, TiO2_Amp etc.)

    liq_comps: pandas DataFrame
        liquid compositions (SiO2_Liq, TiO2_Liq etc.)

    equationP: str
        | P_Put2016_eq7a (T-independent, H2O-dependent)
        | P_Put2016_eq7b (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7c (T-independent, H2O-dependent (as hyd frac))

    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series


    eq_tests: bool. Default False
        If True, also calcualtes Kd Fe-Mg, which Putirka (2016) suggest
        as an equilibrium test.


    Returns
    -------
    pandas.core.series.Series (for simple barometers)
        Pressure in kbar
    pandas DataFrame for barometers like P_Ridolfi_2021, P_Mutch2016

    '''

    if meltmatch is None and len(amp_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as Amp comps. If you want to match up all possible pairs, use the _matching functions instead')
    try:
        func = Amp_Liq_P_funcs_by_name[equationP]
    except KeyError:
        raise ValueError(f'{equationP} is not a valid equation') from None
    sig=inspect.signature(func)

    if equationP == "P_Put2016_eq7a" and meltmatch is None:
        w.warn('Note - Putirka 2016 spreadsheet calculates H2O using a H2O-solubility law of uncertian origin based on the pressure calculated for 7a, and iterates H2O and P. We dont do this, as we dont believe a pure h2o model is necessarily valid as you may be mixed fluid saturated or undersaturated. We recomend instead you choose a reasonable H2O content based on your system.')

    if sig.parameters['T'].default is not None:
        if T is None:
            raise ValueError(f'{equationP} requires you to enter T, or specify T="Solve"')
    else:
        if T is not None:
            print('Youve selected a T-independent function')

    if isinstance(T, pd.Series):
        if liq_comps is not None:
            if len(T) != len(liq_comps):
                raise ValueError('The panda series entered for Temperature isnt the same length as the dataframe of liquid compositions')


    if meltmatch is not None:
        Combo_liq_amps = meltmatch
    if meltmatch is None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

        amp_comps_23 = calculate_23oxygens_amphibole(amp_comps=amp_comps)
        liq_comps_hy = calculate_hydrous_cat_fractions_liquid(
            liq_comps=liq_comps_c)
        liq_comps_an = calculate_anhydrous_cat_fractions_liquid(
            liq_comps=liq_comps_c)
        Combo_liq_amps = pd.concat(
            [amp_comps_23, liq_comps_hy, liq_comps_an], axis=1)


    kwargs = {name: Combo_liq_amps[name] for name, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if isinstance(T, str) or T is None:
        if T == "Solve":
            P_kbar = partial(func, **kwargs)
        if T is None:
            P_kbar=func(**kwargs)

    else:
        P_kbar=func(T, **kwargs)

    if eq_tests is False:
        return P_kbar
    if eq_tests is True:
        MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
        Kd=(MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/(liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd'])

        b = np.empty(len(MolProp), dtype=str)
        for i in range(0, len(MolProp)):

            if Kd[i] >= 0.17 and Kd[i] <= 0.39:
                b[i] = str("Yes")
            else:
                b[i] = str("No")
        Out=pd.DataFrame(data={'P_kbar_calc': P_kbar, 'Kd-Fe-Mg': Kd, "Eq Putirka 2016?": b})
    return Out
## Function: Amp-Liq temp

Amp_Liq_T_funcs = {T_Put2016_eq4b,  T_Put2016_eq4a_amp_sat, T_Put2016_eq9}

Amp_Liq_T_funcs_by_name = {p.__name__: p for p in Amp_Liq_T_funcs}

def calculate_amp_liq_temp(*, amp_comps=None, liq_comps=None, meltmatch=None, equationT=None,
P=None, H2O_Liq=None, eq_tests=False):
    '''
    Amphibole-liquid thermometers. Returns temperature in Kelvin.

   Parameters
    -------

    amp_comps: pandas DataFrame
        amphibole compositions (SiO2_Amp, TiO2_Amp etc.)

    liq_comps: pandas DataFrame
        liquid compositions (SiO2_Liq, TiO2_Liq etc.)

    equationT: str
        T_Put2016_eq4a_amp_sat (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq4b (P-independent, H2O-dep)
        T_Put2016_eq9 (P-independent, H2O-dep through hydrous fractions)

    P: float, int, pandas.Series, str  ("Solve")
        Pressure in kbar
        Only needed for P-sensitive thermometers.
        If enter P="Solve", returns a partial function
        Else, enter an integer, float, or panda series

    eq_tests: bool. Default False
        If True, also calcualtes Kd Fe-Mg, which Putirka (2016) suggest
        as an equilibrium test.


    Returns
    -------
    pandas.core.series.Series
        Temperature in Kelvin
    '''

    if meltmatch is None and len(amp_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as Amp comps. If you want to match up all possible pairs, use the _matching functions instead')

    try:
        func = Amp_Liq_T_funcs_by_name[equationT]
    except KeyError:
        raise ValueError(f'{equationT} is not a valid equation') from None
    sig=inspect.signature(func)

    if sig.parameters['P'].default is not None:
        if P is None:
            raise ValueError(f'{equationT} requires you to enter P, or specify P="Solve"')
    else:
        if P is not None:
            print('Youve selected a P-independent function')

    if isinstance(P, pd.Series):
        if liq_comps is not None:
            if len(P) != len(liq_comps):
                raise ValueError('The panda series entered for Pressure isnt the same length as the dataframe of liquid compositions')

    if meltmatch is not None:
        Combo_liq_amps=meltmatch
    if meltmatch is None:
        liq_comps_c = liq_comps.copy()
        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq'] = H2O_Liq

        amp_comps_23 = calculate_23oxygens_amphibole(amp_comps=amp_comps)
        liq_comps_hy = calculate_hydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
        liq_comps_an = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
        Combo_liq_amps = pd.concat([amp_comps_23, liq_comps_hy, liq_comps_an], axis=1)


    kwargs = {name: Combo_liq_amps[name] for name, p in sig.parameters.items()
    if p.kind == inspect.Parameter.KEYWORD_ONLY}


    if isinstance(P, str) or P is None:
        if P == "Solve":
            T_K = partial(func, **kwargs)
        if P is None:
            T_K=func(**kwargs)

    else:
        T_K=func(P, **kwargs)


    if eq_tests is False:
        return T_K
    if eq_tests is True:
        MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
        Kd=((MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/
        (liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd']))

        b = np.empty(len(MolProp), dtype=str)
        for i in range(0, len(MolProp)):

            if Kd[i] >= 0.17 and Kd[i] <= 0.39:
                b[i] = str("Yes")
            else:
                b[i] = str("No")
        Out=pd.DataFrame(data={'T_K_calc': T_K, 'Kd-Fe-Mg': Kd, "Eq Putirka 2016?": b})
    return Out

## Function for amphibole-liquid PT iter (although technically not needed)


def calculate_amp_liq_press_temp(*, liq_comps=None, amp_comps=None, meltmatch=None, equationT, equationP, iterations=30,
T_K_guess=1300, H2O_Liq=None, eq_tests=False):
    '''
    Solves simultaneous equations for temperature and pressure using
    amphibole only thermometers and barometers.


   Parameters
    -------

    amp_comps: pandas.DataFrame
        Amphibole compositions with column headings SiO2_Amp, MgO_Amp etc.

    equationP: str
        | P_Put2016_eq7a (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7b (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7c (T-independent, H2O-dependent (as hyd frac))

    equationT: str
        T_Put2016_eq4a_amp_sat (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq4b (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq9 (P-independent, H2O-dep through hydrous fractions)


    H2O_Liq: float, int, pandas.Series, optional
        Needed if you select P_Put2008_eq32b, which is H2O-dependent.

    Optional:

    iterations: int, default=30
         Number of iterations used to converge to solution.

    T_K_guess: int or float. Default is 1300 K
         Initial guess of temperature.

    eq_tests: bool. Default False
        If True, also calcualtes Kd Fe-Mg, which Putirka (2016) suggest
        as an equilibrium test.



    Returns:
    -------
    pandas.DataFrame: Pressure in Kbar, Temperature in K, Kd-Fe-Mg if eq_tests=True
    '''

    if meltmatch is None and len(amp_comps)!=len(liq_comps):
        raise ValueError('Liq comps need to be same length as Amp comps. If you want to match up all possible pairs, use the _matching functions instead')

    if meltmatch is None:

        liq_comps_c=liq_comps.copy()

        if H2O_Liq is not None:
            liq_comps_c['H2O_Liq']=H2O_Liq

        T_func = calculate_amp_liq_temp(liq_comps=liq_comps_c,
        amp_comps=amp_comps, equationT=equationT, P="Solve")

        P_func = calculate_amp_liq_press(liq_comps=liq_comps_c,
        amp_comps=amp_comps, equationP=equationP, T="Solve")

    if meltmatch is not None:
        T_func = calculate_amp_liq_temp(meltmatch=meltmatch, equationT=equationT, P="Solve")

        P_func = calculate_amp_liq_press(meltmatch=meltmatch, equationP=equationP, T="Solve")

    if isinstance(T_func, pd.Series) and isinstance(P_func, pd.Series):
        P_guess = P_func
        T_K_guess = T_func

    if isinstance(T_func, pd.Series) and isinstance(P_func, partial):
        P_guess = P_func(T_func)
        T_K_guess = T_func

    if isinstance(P_func, pd.Series) and isinstance(T_func, partial):
        T_K_guess = T_func(P_func)
        P_guess = P_func

    if isinstance(P_func, partial) and isinstance(T_func, partial):

        count=0
        for _ in range(iterations):
            P_guess = P_func(T_K_guess)
            T_K_guess = T_func(P_guess)
            if count==iterations-2:
                # On the second last step, save the pressure
                P_out_loop=P_guess.values
                T_out_loop=T_K_guess.values
            count=count+1

        DeltaP=P_guess-P_out_loop
        DeltaT=T_K_guess-T_out_loop


    else:
        DeltaP=0
        DeltaT=0

    PT_out = pd.DataFrame(data={'P_kbar_calc': P_guess,
                                'T_K_calc': T_K_guess,
                                'Delta_P_kbar_Iter': DeltaP,
                                'Delta_T_K_Iter': DeltaT})

    if eq_tests is False:
        return PT_out
    if eq_tests is True:
        liq_comps_hy = calculate_hydrous_cat_fractions_liquid(
            liq_comps=liq_comps_c)
        MolProp=calculate_mol_proportions_amphibole(amp_comps=amp_comps)
        Kd=(MolProp['FeOt_Amp_mol_prop']/MolProp['MgO_Amp_mol_prop'])/(liq_comps_hy['FeOt_Liq_mol_frac_hyd']/liq_comps_hy['MgO_Liq_mol_frac_hyd'])
        PT_out['Kd-Fe-Mg']=Kd

        b = np.empty(len(MolProp), dtype=str)
        for i in range(0, len(MolProp)):

            if Kd[i] >= 0.17 and Kd[i] <= 0.39:
                b[i] = str("Yes")
            else:
                b[i] = str("No")

        PT_out["Eq Putirka 2016?"]=b

    return PT_out

## Assessing all possible matches


def calculate_amp_liq_press_temp_matching(*, liq_comps, amp_comps, equationT=None,
equationP=None, P=None, T=None,  H2O_Liq=None,
 Kd_Match=0.28, Kd_Err=0.11, return_all_pairs=False, iterations=30):

    '''
    Evaluates all possible Amp-Liq pairs from  N Liquids, M amp compositions
    returns P (kbar) and T (K) for those in equilibrium.

    Parameters
    -----------

    liq_comps: pandas.DataFrame
        Panda DataFrame of liquid compositions with column headings SiO2_Liq etc.

    amp_comps: pandas.DataFrame
        Panda DataFrame of amp compositions with column headings SiO2_Amp etc.


    equationP: str
        | P_Put2016_eq7a (T-independent, H2O-dependent)
        | P_Put2016_eq7b (T-independent, H2O-dependent (as hyd frac))
        | P_Put2016_eq7c (T-independent, H2O-dependent (as hyd frac))

    equationT: str
        T_Put2016_eq4a_amp_sat (P-independent, H2O-dep through hydrous fractions)
        T_Put2016_eq4b (P-independent, H2O-dep)
        T_Put2016_eq9 (P-independent, H2O-dep through hydrous fractions)

    Or:

    P: int, float
        Can also specify a pressure to run calculations at, rather than iterating
        using an equation for pressure. E.g., specify an equationT, but no equationP

    T: int, float
        Can also specify a temperature to run calculations at, rather than iterating
        using an equation for temperature.  E.g., specify an equationP, but no equationT

    Optional:

    Kd_Match: int of float, optional
        Allows users to ovewrite the default where Kd is assessed using the 0.28+-0.11 value of Putirka (2016)

    Kd_Err: int or float, optional
        Allows users to override the defualt 1 sigma on Kd matches of +-0.11



   Fe3Fet_Liq: int or float, optional
        Fe3FeT ratio used to assess Kd Fe-Mg equilibrium between amp and melt.
        If users don't specify, uses Fe3Fet_Liq from liq_comps.
        If specified, overwrites the Fe3Fet_Liq column in the liquid input.

    Returns: dict

        Av_PTs: Average P and T for each amp.
        E.g., if amp1 matches Liq1, Liq4, Liq6, Liq10, averages outputs for all 4 of those liquids.
        Returns mean and 1 sigma of these averaged parameters for each Amp.

        All_PTs: Returns output parameters for all matches (e.g, amp1-Liq1, amp1-Liq4) without any averaging.

    '''
    # This checks that inputs are consistent, and not contradictory
    if equationP is not None and P is not None:
        raise ValueError('You have entered an equation for P and specified a pressure. '
        ' The code doesnt know what you want it to do. Either enter an equation, or choose a pressure. ')

    if equationT is not None and T is not None:
        raise ValueError('You have entered an equation for T and specified a temperature. '
        'The code doesnt know what you want it to do. Either enter an equation, or choose a temperature.  ')
    if equationP == "P_Put2016_eq7a":
        print('Note - Putirka 2016 spreadsheet calculates H2O using a H2O-solubility law of uncertian origin based on the pressure calculated for 7a, and iterates H2O and P. We dont do this, as we dont believe a pure h2o model is necessarily valid as you may be mixed fluid saturated or undersaturated. We recomend instead you choose a reasonable H2O content based on your system.')

    # This over-writes inputted Fe3Fet_Liq and H2O_Liq inputs.
    liq_comps_c = liq_comps.copy()
    amp_comps_c=amp_comps.copy()
    if H2O_Liq is not None and not isinstance(H2O_Liq, str):
        liq_comps_c['H2O_Liq'] = H2O_Liq
    if "Fe3Fet_Liq" not in liq_comps:
        liq_comps_c['Fe3Fet_Liq'] = 0

    # Adding sample names if there aren't any
    if "Sample_ID_Liq" not in liq_comps:
        liq_comps_c['Sample_ID_Liq'] = liq_comps_c.index
    if "Sample_ID_Amp" not in amp_comps:
        amp_comps_c['Sample_ID_Amp'] = amp_comps.index

    amp_comps_c['ID_AMP'] = amp_comps_c.index
    liq_comps_c['ID_Liq'] = liq_comps_c.index


    amp_comps_23_sim = calculate_23oxygens_amphibole(amp_comps=amp_comps_c)
    amp_mols=calculate_mol_proportions_amphibole(amp_comps=amp_comps_c)
    amp_comps_23=pd.concat([amp_comps_23_sim, amp_mols, amp_comps_c], axis=1)
    liq_comps_hy = calculate_hydrous_cat_fractions_liquid(liq_comps=liq_comps_c)
    liq_comps_an = calculate_anhydrous_cat_fractions_liquid(liq_comps=liq_comps_c)

    if 'Sample_ID_Liq' in liq_comps_hy.columns:
        liq_comps_hy.drop(["Sample_ID_Liq"], axis=1, inplace=True)

    Combo_liqs_hyd_anhyd = pd.concat([liq_comps_hy, liq_comps_an], axis=1)





    # This duplicates AMPs, repeats amp1-amp1*N, amp2-amp2*N etc.
    DupAMPs = pd.DataFrame(
        np.repeat(amp_comps_23.values, np.shape(Combo_liqs_hyd_anhyd)[0], axis=0))
    DupAMPs.columns = amp_comps_23.columns



    # This duplicates liquids like liq1-liq2-liq3 for amp1, liq1-liq2-liq3 for
    # amp2 etc.
    DupLiqs = pd.concat([Combo_liqs_hyd_anhyd] *
                        np.shape(amp_comps_23)[0]).reset_index(drop=True)


    # Combines these merged liquids and amp dataframes
    Combo_liq_amps = pd.concat([DupLiqs, DupAMPs], axis=1)



    LenAmp=len(amp_comps)
    LenLiqs=len(liq_comps)
    print("Considering N=" + str(LenAmp) + " Amp & N=" + str(LenLiqs) +" Liqs, which is a total of N="+ str(len(Combo_liq_amps)) +
          " Amp-Liq pairs, be patient if this is >>1 million!")



    # calculate Kd for this merged dataframe
    Combo_liq_amps['Kd']=((Combo_liq_amps['FeOt_Amp_mol_prop']/Combo_liq_amps['MgO_Amp_mol_prop'])/
    (Combo_liq_amps['FeOt_Liq_mol_frac_hyd']/Combo_liq_amps['MgO_Liq_mol_frac_hyd']))



    if return_all_pairs is True:
        Combo_liq_amp_fur_filt=Combo_liq_amps.copy()

    if return_all_pairs is False:
        Filt=np.abs(Combo_liq_amps['Kd']-Kd_Match)<=Kd_Err
        Combo_liq_amp_fur_filt=Combo_liq_amps.loc[Filt].reset_index(drop=True)






#         # If users want to melt match specifying an equation for both T and P
    if equationP is not None and equationT is not None:

        PT_out = calculate_amp_liq_press_temp(meltmatch=Combo_liq_amp_fur_filt,
         equationP=equationP, equationT=equationT, iterations=iterations)
        P_guess = PT_out['P_kbar_calc'].astype('float64')
        T_K_guess = PT_out['T_K_calc'].astype('float64')
        Delta_T_K_Iter=PT_out['Delta_T_K_Iter'].astype(float)
        Delta_P_kbar_Iter=PT_out['Delta_P_kbar_Iter'].astype(float)


    if equationP is not None and equationT is None:
        P_guess = calculate_amp_liq_press_temp(meltmatch=Combo_liq_amp_fur_filt,
        equationP=equationP, T=T)
        T_K_guess = T
        Delta_T_K_Iter=0
        Delta_P_kbar_Iter=0
    # Same if user doesnt specify an equation for P, but a real P
    if equationT is not None and equationP is None:
        T_guess = calculate_amp_liq_press_temp(meltmatch=Combo_liq_amp_fur_filt,
        equationT=equationT, P=P)
        P_guess = P
        Delta_T_K_Iter=0
        Delta_P_kbar_Iter=0

    Combo_liq_amp_fur_filt.insert(0, "P_kbar_calc", P_guess.astype(float))
    Combo_liq_amp_fur_filt.insert(1, "T_K_calc", T_K_guess.astype(float))
    Combo_liq_amp_fur_filt.insert(2, "Delta_P_kbar_Iter", Delta_P_kbar_Iter)
    Combo_liq_amp_fur_filt.insert(3, "Delta_T_K_Iter", Delta_T_K_Iter)

    Combo_liq_amp_fur_filt.insert(4, 'Delta_Kd', Kd_Match-Combo_liq_amps['Kd'])








    # Final step, calcuate a 3rd output which is the average and standard
    # deviation for each Amp (e.g., Amp1-Melt1, Amp1-melt3 etc. )



    Liq_sample_ID=Combo_liq_amp_fur_filt["Sample_ID_Liq"]
    Combo_liq_amp_fur_filt.drop(["Sample_ID_Liq"], axis=1, inplace=True)
    print('Liq ID you are giving')
    print(Liq_sample_ID)



    AmpNumbers = Combo_liq_amp_fur_filt['ID_AMP'].unique()

    if len(AmpNumbers) > 0:
        df1_Mean_nopref=Combo_liq_amp_fur_filt.groupby(['ID_AMP', 'Sample_ID_Amp'], as_index=False).mean()
        df1_Std_nopref=Combo_liq_amp_fur_filt.groupby(['ID_AMP', 'Sample_ID_Amp'], as_index=False).std()
        count=Combo_liq_amp_fur_filt.groupby('ID_AMP',as_index=False).count().iloc[:, 1]
        df1_Mean_nopref['# of Liqs Averaged']=count
        Sample_ID_Amp_Mean=df1_Mean_nopref['Sample_ID_Amp']
        Sample_ID_Amp_Std=df1_Std_nopref['Sample_ID_Amp']
        df1_Mean=df1_Mean_nopref.add_prefix('Mean_')
        df1_Std=df1_Std_nopref.add_prefix('Std_')

        df1_Mean.rename(columns={"Mean_ID_AMP": "ID_AMP"}, inplace=True)
        df1_Mean.rename(columns={"Mean_# of Liqs Averaged": "# of Liqs Averaged"}, inplace=True)
        df1_Std.rename(columns={"Std_ID_AMP": "ID_AMP"}, inplace=True)



        df1_M=pd.merge(df1_Mean, df1_Std, on=['ID_AMP'])
        df1_M['Sample_ID_Amp']=Sample_ID_Amp_Mean

        if equationT is not None and equationP is not None:
            cols_to_move = ['Sample_ID_Amp', '# of Liqs Averaged',
                        'Mean_T_K_calc', 'Std_T_K_calc', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']

        if equationT is not None and equationP is None:
            cols_to_move = ['Sample_ID_Amp',
                        'Mean_P_kbar_input',
                        'Std_P_kbar_input', 'Mean_T_K_calc', 'Std_T_K_calc']

        if equationT is None and equationP is not None:
            cols_to_move = ['Sample_ID_Amp',
                        'Mean_T_K_input', 'Std_T_K_input', 'Mean_P_kbar_calc',
                        'Std_P_kbar_calc']

        df1_M = df1_M[cols_to_move +
                    [col for col in df1_M.columns if col not in cols_to_move]]


    else:
        raise Exception(
            'No Matches - to set less strict filters, change our Kd filter')



    print('Done!!! I found a total of N='+str(len(Combo_liq_amp_fur_filt)) + ' Amp-Liq matches using the specified filter. N=' + str(len(df1_M)) + ' Amp out of the N='+str(LenAmp)+' Amp that you input matched to 1 or more liquids')



    Combo_liq_amp_fur_filt['Sample_ID_Liq']=Liq_sample_ID
    return {'Av_PTs': df1_M, 'All_PTs': Combo_liq_amp_fur_filt}



## Amphibole-Plag temperatures, Holland and Blundy 1994


def calculate_amp_plag_temp(*, amp_comps=None, plag_comps=None, XAn=None, XAb=None, equationT=None, P=None, meltmatch=None):
    """ This function calculates Plag and Amp temperatures

    Parameters
    ------------------



    amp_comps: pandas.DataFrame
        Panda DataFrame of amp compositions with column headings SiO2_Amp etc.

    Either

    plag_comps: pandas.DataFrame
        Panda DataFrame of plag compositions with column headings SiO2_Plag etc.

    OR

    XAn, XAb: int, float, pd.Series
        An and Ab content of plag, all the function uses

    equationT: str
        Choose from T_HB1994_A and T_HB1994__B (Holland and Blundy, 1994).

    P: str, int, pd.Series:
        Pressure in kbar.

    Returns
    -----------------
    pd.Series of temp


    """

    if meltmatch is not None and plag_comps is not None and len(plag_comps)!=len(amp_comps):
        raise ValueError('Amp comps need to be same length as Plag comps. If you want to match up all possible pairs, use the _matching functions instead')

    if equationT != "T_HB1994_A" and equationT != "T_HB1994_B":
        raise Exception('At the moment, the only options are T_HB1994_A and _B')
    if P is None:
        raise Exception('Please select a P in kbar')

    if meltmatch is None:
        # Dealing with individual ones
        amp_comps_c=amp_comps.copy()



        if plag_comps is not None:
            plag_comps_c=plag_comps.copy()
            plag_components=calculate_cat_fractions_plagioclase(plag_comps=plag_comps_c)
            XAb=plag_components['Ab_Plag']
            XAn=plag_components['An_Plag']
        if plag_comps is None:
            if isinstance(XAn, int) or isinstance(XAn, float):
                XAn = pd.Series([XAn] * len(amp_comps))
                XAb = pd.Series([XAb] * len(amp_comps))

        amp_apfu_df=calculate_23oxygens_amphibole(amp_comps=amp_comps_c)

    if meltmatch is not None:
        amp_apfu_df=meltmatch
        XAb=meltmatch['Ab_Plag']
        XAn=meltmatch['An_Plag']


    f1=16/(amp_apfu_df['cation_sum_All'])
    f2=8/(amp_apfu_df['Si_Amp_cat_23ox'])
    f3=15/(amp_apfu_df['cation_sum_All']-amp_apfu_df['Na_Amp_cat_23ox']-amp_apfu_df['K_Amp_cat_23ox'])
    f4=2/amp_apfu_df['Ca_Amp_cat_23ox']
    f5=1
    f6=8/(amp_apfu_df['Si_Amp_cat_23ox']+amp_apfu_df['Al_Amp_cat_23ox'])
    f7=15/(amp_apfu_df['cation_sum_All']-amp_apfu_df['K_Amp_cat_23ox'])
    f8=12.9/(amp_apfu_df['cation_sum_All']-amp_apfu_df['Ca_Amp_cat_23ox']-amp_apfu_df['Na_Amp_cat_23ox']
            -amp_apfu_df['K_Amp_cat_23ox'])
    f9=36/(46-amp_apfu_df['Si_Amp_cat_23ox']-amp_apfu_df['Al_Amp_cat_23ox']-amp_apfu_df['Ti_Amp_cat_23ox'])
    f10=46/(amp_apfu_df['Fet_Amp_cat_23ox']+46)
    fa=pd.DataFrame(data={'f1': f1, 'f2': f2, 'f3':f3, 'f4': f4, 'f5': f5})
    fb=pd.DataFrame(data={'f6': f6, 'f7': f7, 'f8':f8, 'f9': f9, 'f10': f10})
    fa_min=fa.min(axis="columns")
    fb_max=fb.max(axis="columns")
    f=(fa_min+fb_max)/2
    fmin_greater1=fa_min>1
    fmax_greater1=fb_max>1
    f.loc[fmin_greater1]=1
    f.loc[fmax_greater1]=1

    amp_apfu_df_recalc=amp_apfu_df.drop(columns=['Fet_Amp_cat_23ox', 'oxy_renorm_factor',
                            'cation_sum_Si_Mg', 'cation_sum_Si_Ca', 'cation_sum_All', 'Mgno_Amp'])
    amp_apfu_df_recalc=amp_apfu_df_recalc.multiply(f, axis='rows')
    amp_apfu_df_recalc['Fe3_Amp_cat_23ox']=46*(1-f)
    amp_apfu_df_recalc['Fe2_Amp_cat_23ox']=(amp_apfu_df.multiply(f, axis='rows').get('Fet_Amp_cat_23ox')
    - amp_apfu_df_recalc['Fe3_Amp_cat_23ox'])

    cm=((amp_apfu_df_recalc['Si_Amp_cat_23ox']+amp_apfu_df_recalc['Al_Amp_cat_23ox']
    +amp_apfu_df_recalc['Ti_Amp_cat_23ox']+amp_apfu_df_recalc['Fe3_Amp_cat_23ox']
    + amp_apfu_df_recalc['Fe2_Amp_cat_23ox']+amp_apfu_df_recalc['Mg_Amp_cat_23ox']
    +amp_apfu_df_recalc['Mn_Amp_cat_23ox'])-13)

    XSi_T1=(amp_apfu_df_recalc['Si_Amp_cat_23ox']-4)/4
    XAl_T1=(8-amp_apfu_df_recalc['Si_Amp_cat_23ox'])/4
    XAl_M2=(amp_apfu_df_recalc['Si_Amp_cat_23ox']+amp_apfu_df_recalc['Al_Amp_cat_23ox']-8)/2
    XK_A=amp_apfu_df_recalc['K_Amp_cat_23ox']
    Xsq_A=(3-amp_apfu_df_recalc['Ca_Amp_cat_23ox']-amp_apfu_df_recalc['Na_Amp_cat_23ox']-amp_apfu_df_recalc['K_Amp_cat_23ox']
    -cm)
    XNa_A=amp_apfu_df_recalc['Ca_Amp_cat_23ox']+amp_apfu_df_recalc['Na_Amp_cat_23ox']+cm-2
    XNa_M4=(2-amp_apfu_df_recalc['Ca_Amp_cat_23ox']-cm)/2
    XCa_M4=amp_apfu_df_recalc['Ca_Amp_cat_23ox']/2
    Ked_trA=(27/256)*(Xsq_A*XSi_T1*XAb)/(XNa_A*XAl_T1)
    Ked_trB=(27/64)*(XNa_M4*XSi_T1*XAn)/(XCa_M4*XAl_T1*XAb)
    YAb=12*(1-XAb)**2-3
    HighXAb=XAb>0.5
    YAb[HighXAb]=0

    Ta=((-76.95+P*0.79+YAb+39.4*XNa_A+22.4*XK_A+(41.5-2.89*P)*XAl_M2)/
        (-0.065-0.0083144*np.log(Ked_trA.astype(float))))

    YAb_B=12*(2*XAb-1)+3
    YAb_B[HighXAb]=3
    Tb=((78.44 +YAb_B - 33.6*XNa_M4 - (66.8 -2.92*P)*XAl_M2 +78.5*XAl_T1 +9.4*XNa_A )/
        (0.0721-0.0083144*np.log(Ked_trB.astype(float))))

    if equationT=="T_HB1994_A":
        return Ta
    if equationT=="T_HB1994_B":
        return Tb

def calculate_amp_plag_temp_matching(amp_comps, plag_comps, equationT=None, P=None):
    """ This function pairs up all possible plag-amp pairs, and calculates temps for them.

    Parameters
    ------------------



    amp_comps: pandas.DataFrame
        Panda DataFrame of amp compositions with column headings SiO2_Amp etc.



    plag_comps: pandas.DataFrame
        Panda DataFrame of plag compositions with column headings SiO2_Plag etc.



    equationT: str
        Choose from T_HB1994_A and T_HB1994__B (Holland and Blundy, 1994).

    P: str, int, pd.Series:
        Pressure in kbar.

    Returns
    -----------------
    Dataframe of temps and matches and all intermediate calc steps


    """

# Adding an ID label to help with melt-Amp rematching later


    myAmps1_concat=calculate_23oxygens_amphibole(amp_comps=amp_comps)
    myPlag_concat=calculate_cat_fractions_plagioclase(plag_comps=plag_comps)

    myAmps1_concat['ID_Amp'] = myAmps1_concat.index
    if "Sample_ID_Amp" not in amp_comps:
        myAmps1_concat['Sample_ID_Amp'] = myAmps1_concat.index
    else:
        myAmps1_concat['Sample_ID_Amp']=amp_comps['Sample_ID_Amp']

    if "Sample_ID_Plag" not in plag_comps:
        myPlag_concat['Sample_ID_Plag'] = myPlag_concat.index
    else:
        myPlag_concat['Sample_ID_Plag']=plag_comps['Sample_ID_Plag']

    myAmps1_concat['ID_Amp']=myAmps1_concat.index
    myPlag_concat['ID_Plag'] = myPlag_concat.index


    # This duplicates Amps, repeats Amp1-Amp1*N, Amp2-Amp2*N etc.
    DupAmps = pd.DataFrame(np.repeat(myAmps1_concat.values,
    np.shape(myPlag_concat)[0], axis=0))
    DupAmps.columns = myAmps1_concat.columns

    # This duplicates Plaguids like Plag1-Plag2-Plag3 for Amp1, Plag1-Plag2-Plag3 for
    # Amp2 etc.
    DupPlags = pd.concat([myPlag_concat] *
                        np.shape(myAmps1_concat)[0]).reset_index(drop=True)

    # Combines these merged Plaguids and Amp dataframes
    Combo_Plag_Amps = pd.concat([DupPlags, DupAmps], axis=1)
    LenCombo = str(np.shape(Combo_Plag_Amps)[0])

    # Status update for user
    LenAmp=len(amp_comps)
    LenPlags=len(plag_comps)
    print("Considering N=" + str(LenAmp) + " Amp & N=" + str(LenPlags) +" Plags, which is a total of N="+ str(LenCombo) +
            " Plag-Amp pairs, be patient if this is >>1 million!")

    Temp=calculate_amp_plag_temp(meltmatch=Combo_Plag_Amps, equationT=equationT, P=P)

    Combo_Plag_Amps['T_K_calc']= Temp

    cols_to_move = ['T_K_calc', 'Sample_ID_Plag', 'Sample_ID_Amp']
    Combo_Plag_Amps_filt = Combo_Plag_Amps[cols_to_move + [
        col for col in Combo_Plag_Amps.columns if col not in cols_to_move]]


    return Combo_Plag_Amps_filt




