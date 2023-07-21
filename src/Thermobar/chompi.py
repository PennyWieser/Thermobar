import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd

from Thermobar.core import *

def calculate_CHOMPI_P_T_XH2O(liq_norm):

        """
        This function calculates T_K, P_kbar and XCO2H2O based on a normalized liquid composition.
        It returns the liquid, T_K, P_kbar, XCO2_mol, XH2O
        This is used as an intermediate calculate step for the calculate_CHOMPI function


        """


        T_K=(0+11.0697*liq_norm['SiO2_Liq']+127.175*liq_norm['TiO2_Liq']
        +17.968*liq_norm['Al2O3_Liq']+0*liq_norm['FeOt_Liq']+54.428*liq_norm['MgO_Liq']
        +0*liq_norm['CaO_Liq']+0*liq_norm['Na2O_Liq'])


        P_kbar=(0+0*liq_norm['SiO2_Liq']+0*liq_norm['TiO2_Liq']+0.7995*liq_norm['Al2O3_Liq']
                +0*liq_norm['FeOt_Liq']+0*liq_norm['MgO_Liq']+0*liq_norm['CaO_Liq']-2.1199*liq_norm['Na2O_Liq'])

        XCO2_mol=(21.0129+-0.23383*liq_norm['SiO2_Liq']+0*liq_norm['TiO2_Liq']
                +0*liq_norm['Al2O3_Liq']+0*liq_norm['FeOt_Liq']+0*liq_norm['MgO_Liq']
                -0.4354*liq_norm['CaO_Liq']-0.6723*liq_norm['Na2O_Liq'])
        XH2O=(1/(1+XCO2_mol))

        return liq_norm, T_K, P_kbar, XCO2_mol, XH2O


def calculate_CHOMPI_residuals(liq_norm, P_kbar, T_K, XCO2_mol, XH2O, deltaNNO, T_K_guess, df_out=True):

    """
    This function calculates the residuals for the CHOMPI function based on measured vs. calculated liquid
    compositions. It can return a df of all the calculated liquids, or just the residual (used as an intermediate function when duplicating rows in the calculate_CHOMPI function).

    """

    T_K_residual=T_K-T_K_guess

    SiO2_Calc=(-4741.99+0.688*P_kbar-0.74491*T_K-1.978*XCO2_mol+0*deltaNNO
            +0*P_kbar/T_K+0.847*liq_norm['K2O_Liq']+803.963*np.log(T_K.astype(float)))

    TiO2_Calc=(279.62+0*P_kbar+0.04336*T_K+0*XCO2_mol-0.024*deltaNNO
            +0*P_kbar/T_K+0*liq_norm['K2O_Liq']-46.702*np.log(T_K.astype(float)))

    Al2O3_Calc=(1311.64+0*P_kbar+0.19754*T_K+0*XCO2_mol+0*deltaNNO
                +0*P_kbar/T_K-0.69*liq_norm['K2O_Liq']-215.86*np.log(T_K.astype(float)))

    FeOt_Calc=(0+2.925*P_kbar+0.00323*T_K+0.961*XCO2_mol+0*deltaNNO
            -3668*P_kbar/T_K-0.271*liq_norm['K2O_Liq']+0*np.log(T_K.astype(float)))

    MgO_Calc=(1109.3-1.078*P_kbar+0.17246*T_K+0.194*XCO2_mol+0.044*deltaNNO
            +1199*P_kbar/T_K-0.182*liq_norm['K2O_Liq']-185.374*np.log(T_K.astype(float)))

    CaO_Calc=(-10.06+0*P_kbar+0.01282*T_K+0.416*XCO2_mol+0*deltaNNO
            +0*P_kbar/T_K-0.666*liq_norm['K2O_Liq']+0*np.log(T_K.astype(float)))

    Na2O_Calc=(1222.98+2.212*P_kbar+0.16236*T_K+0*XCO2_mol+0*deltaNNO
            -2929*P_kbar/T_K-0*liq_norm['K2O_Liq']+-199.31*np.log(T_K.astype(float)))


    Compositional_residual=(
        (SiO2_Calc-liq_norm['SiO2_Liq'])**2
        +(TiO2_Calc-liq_norm['TiO2_Liq'])**2
        +(Al2O3_Calc-liq_norm['Al2O3_Liq'])**2
        +(FeOt_Calc-liq_norm['FeOt_Liq'])**2
        +(MgO_Calc-liq_norm['MgO_Liq'])**2
        +(CaO_Calc-liq_norm['CaO_Liq'])**2
        +(Na2O_Calc-liq_norm['Na2O_Liq'])**2
    )

    Fail_Cali_test_SiO2= (liq_norm['SiO2_Liq']<53) | (liq_norm['SiO2_Liq']>78)
    Fail_Cali_test_Al2O3= (liq_norm['Al2O3_Liq']<12) | (liq_norm['Al2O3_Liq']>21)
    Fail_Cali_test_MgO= (liq_norm['MgO_Liq']>4)
    Fail_Cali_test_Na2O= (liq_norm['Na2O_Liq']>7)
    Fail_Cali_test_K2O= (liq_norm['K2O_Liq']>6)
    Fail_overall=((Fail_Cali_test_SiO2) | (Fail_Cali_test_Al2O3) | (Fail_Cali_test_MgO) |(Fail_Cali_test_Na2O) |(Fail_Cali_test_K2O))

    if df_out==True:
        df_out=pd.DataFrame(data={
                              'T_K_calc': T_K,
                            'P_kbar_calc': P_kbar,
                            'XCO2XH2O': XCO2_mol,
                            'XH2O_mol': XH2O,
                            'SiO2_calc': SiO2_Calc,
                            'TiO2_calc': TiO2_Calc,
                            'Al2O3_calc': Al2O3_Calc,
                            'FeOt_calc': FeOt_Calc,
                            'MgO_calc': MgO_Calc,
                            'CaO_calc': CaO_Calc,
                            'Na2O_calc': Na2O_Calc,
                            'K2O_calc': 100-(SiO2_Calc+TiO2_Calc+Al2O3_Calc+FeOt_Calc+MgO_Calc+CaO_Calc+Na2O_Calc),
                            'comp_residual': Compositional_residual,
                            'T_K_residual': T_K_residual,
                            'CHOMPI (SSI<6?)': 'True',
                            'In Cali Range?': ~Fail_overall,
                            'Pass_Cali_test_SiO2?': ~Fail_Cali_test_SiO2,
                            'Pass_Cali_test_Al2O3?': ~Fail_Cali_test_Al2O3,
                            'Pass_Cali_test_MgO?': ~Fail_Cali_test_MgO,
                            'Pass_Cali_test_Na2O?': ~Fail_Cali_test_Na2O,
                            'Pass_Cali_test_K2O?': ~Fail_Cali_test_K2O,

                            })

        concat=pd.concat([df_out, liq_norm], axis=1)


        return concat
    else:
        return Compositional_residual

def duplicate_rows(liq_norm_to_dup, N_rep):
    """ used to duplicate rows
    """
    duplicated_rows = np.repeat(liq_norm_to_dup.values, N_rep, axis=0)

    #Create a new DataFrame with duplicated rows
    duplicated_df = pd.DataFrame(duplicated_rows, columns=liq_norm_to_dup.columns)

    # Reset the index
    duplicated_df.reset_index(drop=True, inplace=True)
    return duplicated_df

def calculate_CHOMPI(liq_comps,  deltaNNO, T_K_guess, N_rep=100):
    """ This function follows the CHOMPI algorithmic steps of Blundy et al. (2022)
https://doi.org/10.1093/petrology/egac054

    Parameters
    --------------
    liq_comps: pd.DataFrame
        dataframe of liquid compositions from Thermobar input function, format SiO2_Liq, CaO_Liq etc

    deltaNNO: pd.Series
        Estimate of delta NNO of melt (relative to NNO buffer)

    T_K_guess:
        estimate of eruption temp, Blundy (2022) say should be within 40C

    N_rep: float, int (default = 100)
        How many rows to make for each sample to pertub P, T and XCO2 within limits.
        100 is 100 equally spaced temps between +16.2 and -16.2

    Returns
    ---------------
    pd.DataFrame
        Calculated P, T, certainty calculation, calculated liquid compositions, indication of whether you are in calibration range.



    """
    # First up, normalize your liquid
    liq_norm=normalize_liquid_100_anhydrous_chompi(liq_comps=liq_comps)


    # now lets perform the first round of CHOMPI calculations
    liq_norm, T_K, P_kbar, XCO2_mol, XH2O=calculate_CHOMPI_P_T_XH2O(liq_norm)
    df_out=calculate_CHOMPI_residuals(liq_norm, P_kbar, T_K, XCO2_mol, XH2O, deltaNNO, T_K_guess, df_out=True)

    #Now lets make 100 duplicates of these rows.
    N_rep=100
    liq_norm_to_dup=liq_norm.copy()
    liq_norm_to_dup['T_K']=T_K
    liq_norm_to_dup['P_kbar']=P_kbar
    liq_norm_to_dup['XCO2XH2O']=XCO2_mol
    liq_norm_to_dup['deltaNNO']=deltaNNO
    liq_norm_to_dup['T_K_guess']=T_K_guess

    # Pertubations for P-T-XCO2H2O
    T_perturb = np.linspace(-16.2, 16.2, N_rep)  # Perturbation values for 'T_K'
    P_perturb = np.linspace(-1.2, 1.2, N_rep)  # Perturbation values for 'P_kbar'
    XCO2_perturb = np.linspace(-0.427, 0.427, N_rep)  # Perturbation values for 'XCO2'

    duplicated_df_T_perturb = duplicate_rows(liq_norm_to_dup, N_rep)
    duplicated_df_P_perturb = duplicate_rows(liq_norm_to_dup, N_rep)
    duplicated_df_XCO2_perturb = duplicate_rows(liq_norm_to_dup, N_rep)
    N_sample=len(liq_norm_to_dup)
    duplicated_T_values = np.tile(T_perturb, N_sample)
    duplicated_P_values = np.tile(P_perturb, N_sample)
    duplicated_XCO2_values = np.tile(XCO2_perturb, N_sample)


    # Now add on these columns to each
    duplicated_df_T_perturb['T_K']=duplicated_df_T_perturb['T_K']+duplicated_T_values
    duplicated_df_P_perturb['P_kbar']=duplicated_df_P_perturb['P_kbar']+duplicated_P_values
    duplicated_df_P_perturb['XCO2XH2O']=duplicated_df_P_perturb['XCO2XH2O']+duplicated_XCO2_values

    appended_df = pd.concat([duplicated_df_T_perturb, duplicated_df_P_perturb, duplicated_df_P_perturb], ignore_index=True)

    # Now do the calculations again for this dataframe.
    XH2O=(1/(1+appended_df['XCO2XH2O']))


    comp_residual=calculate_CHOMPI_residuals(appended_df, P_kbar=appended_df['P_kbar'], T_K=appended_df['T_K'], XCO2_mol=appended_df['XCO2XH2O'],
                                        XH2O=XH2O, deltaNNO=appended_df['deltaNNO'], T_K_guess=appended_df['T_K_guess'], df_out=False)


    Strong_Positive = np.array([''] * N_sample, dtype='U1')
    Positive=np.array([''] * N_sample, dtype='U1')
    Permissive_positive=np.array([''] * N_sample, dtype='U1')
    Strong_negative=np.array([''] * N_sample, dtype='U1')
    i=0
    for sam in appended_df['Sample_ID_Liq'].unique():
        sam_ind=appended_df['Sample_ID_Liq']==sam
        comp_residual_sam=comp_residual.loc[sam_ind]

        # Strong positive - Initial value less than 6, no pertubation takes >6
        Strong_Positive[i]= ( ~(comp_residual_sam > 6).any()) & (df_out['comp_residual'].iloc[i]<6)
        # Positive, initial value <6, but some pertubations take away from 6
        Positive[i]=((comp_residual_sam > 6).any()) & (df_out['comp_residual'].iloc[i]<6)
        # permissive positive, initial value more than 6, but some pertubatoin takes less
        Permissive_positive[i]=(df_out['comp_residual'].iloc[i]>6) & ((comp_residual_sam < 6).any())
        # strong negative - initial value more than 6, no pertubation takes away
        Strong_negative[i]=(df_out['comp_residual'].iloc[i]>6) & (~(comp_residual_sam < 6).any())
        i=i+1

    df_assessment=pd.DataFrame(data={'Strong_positive': Strong_Positive,
                            'Positive': Positive,
                            'Permissive_positive': Permissive_positive,
                            'Strong_negative': Strong_negative}
                            )

    first_true_column = df_assessment.eq('T').values.argmax(axis=1)

    # Get the corresponding column names for the first 'T' occurrence
    column_names = np.array(df_assessment.columns)
    final_assessment = np.where(first_true_column >= 0, column_names[first_true_column], 'No True Found')

    # Assign the final assessment to the 'final_assessment' column
    df_out['final_assessment'] = final_assessment
    df_out.rename(columns={'XH2O_mol': 'XH2O_mol_calc'},  inplace=True)
    df_out.rename(columns={'comp_residual': 'SSR'},  inplace=True)
    cols_to_move = ['final_assessment', 'SSR','In Cali Range?', 'T_K_calc', 'T_K_residual', 'P_kbar_calc', 'XH2O_mol_calc', ]
    df_out = df_out[cols_to_move + [
        col for col in df_out.columns if col not in cols_to_move]]



    return df_out




