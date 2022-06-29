import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def rasmussen(P_kbar):
    """ Linear fit to the supporting information of Rasmussen et al. 2022,
    overall best fit density vs. depth"""
    P=P_kbar
    D=3.784444149179223*P+0.2857169093579655
    return D

def hill_zucca(P_kbar):
    """ Parameterization of Hill and Zucca (1987),
    as given in Putirka (2017) Down the Crater Elements supplement
    """
    P=P_kbar

    D=-2.77*10**(-5) * (P**4) - 2.0*10**(-3)*(P**3) - 4.88*10**(-2)*P**2 + 3.6*P - 6.34*(10**(-2))

    return D

def ryan_lerner(P_kbar):
    """ Parameterization of Ryan 1987, actual equation from Lerner et al. 2021
    After 16.88 km (455 MPa), assume density is 2.746, as density turns around again. Used for Hawaii
    """
    P=P_kbar*100
    if P<455.09090909:
        D=(4.578*10**(-8) *P**3) - (4.151*10**(-5) *P**2) + (4.652*10**(-2) *P)
    else:
        D=P/(9.8*2.749643038642074)

    return D

def mavko_debari(P_kbar):
    """ Parameterization of Mavko and Thompson (1983) and DeBari and Greene (2011)
    as given in Putirka (2017) Down the Crater Elements supplement, used for Cascades
    """
    P=P_kbar
    D=0.4853881 + 3.6006116*P - 0.0117368*(P-1.3822)**2


    return D


def prezzi(P_kbar):
    """ Parameterization of Prezzi et al. (2009),
    as given in Putirka (2017) Down the Crater Elements supplement.
    Used for Andes.
    """
    P=P_kbar
    D=4.88 + 3.30*P - 0.0137*(P - 18.01)**2

    return D


Profile_funcs={ryan_lerner, mavko_debari, hill_zucca, prezzi, rasmussen}
Profile_funcs_by_name= {p.__name__: p for p in Profile_funcs}

def convert_pressure_to_depth(P_kbar=None, model=None, crust_dens_kgm3=None):
    """ Converts pressure in kbar to depth in km using a variety of crustal density profiles


    Parameters
    -----------

    P_kbar: int, float, pd.Series, np.ndarray
        Pressure in kbar

    model: str or float:
    If you want an existing model, choose from
        ryan_lerner: Parameterization of Ryan 1987, actual equation from Lerner et al. 2021
    After 16.88 km (455 MPa), assume density is 2.746, as density turns around again. This profile is tweaked for Hawaii

        mavko_debari: Parameterization of Mavko and Thompson (1983) and DeBari and Greene (2011)
    as given in Putirka (2017) Down the Crater Elements supplement.
    **Currently has a typo, have emailed Keith Putirka!***

        hill_zucca: Parameterization of Hill and Zucca (1987),
    as given in Putirka (2017) Down the Crater Elements supplement

        prezzi: Parameterization of Prezzi et al. (2009),
    as given in Putirka (2017) Down the Crater Elements supplement. Tweaked for Andes.

        rasmussen: Linear fit to the supporting information of Rasmussen et al. 2022,
    overall best fit density vs. depth

    OR

    crust_dens_kgm3: float
        Crustal density in kg/m3

    Else, just enter a crustal density in kg/m3, e.g., model=2700



    Returns
    -----------

    Depth in km as a panda series

    """
    if model is not None:

        try:
            func = Profile_funcs_by_name[model]
        except KeyError:
            raise ValueError(f'{model} is not a valid model') from None
        #sig=inspect.signature(func)
        if isinstance(P_kbar, float) or isinstance(P_kbar, int):
            D=func(P_kbar)

        if isinstance(P_kbar, pd.Series):
            D=np.empty(len(P_kbar), float)
            for i in range(0, len(P_kbar)):
                D[i]=func(P_kbar.iloc[i])

        if isinstance(P_kbar, np.ndarray):
            D=np.empty(len(P_kbar), float)
            for i in range(0, len(P_kbar)):
                D[i]=func(P_kbar[i])



    if crust_dens_kgm3 is not None:
        D=10**5*P_kbar/(9.8*crust_dens_kgm3)

    D_series=pd.Series(D)
    return D_series
