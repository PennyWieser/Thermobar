import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from Thermobar.core import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# This function is from matplotlib - https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def matplotlib_confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def av_noise_samples_series(calc, sampleID):
    '''
    This function calculates the mean, median, standard devation, maximum and
    minimum value of rows specified by "calc" based on values in "Sample ID" where both inputs are panda series.

    Parameters
    -------
    calc: Series
        Panda series of inputs you want to average.
    SampleID: str
        column heading for the thing you want to average by (e.g., Sample_ID_Cpx)

    Returns
    -------

    Dataframe with headings "Sample", "Mean_calc", "Median_calc",
    "St_dev_calc", "Max_calc", "Min_calc"

    '''


    if isinstance(calc, pd.Series):
        N = sampleID.unique()
        Av_mean = np.zeros(len(N), dtype=float)
        Av_median = np.zeros(len(N), dtype=float)
        Max = np.zeros(len(N), dtype=float)
        Min = np.zeros(len(N), dtype=float)
        Std = np.zeros(len(N), dtype=float)
        IQR_Std=np.zeros(len(N), dtype=float)
        i=0
        for ID in sampleID.unique():
            sam=ID
            # print(sam)
            # print(i)
            # print(np.nanmean(calc[sampleID == sam]))



            Av_mean[i] = np.nanmean(calc[sampleID == sam])
            Av_median[i] = np.nanmedian(calc[sampleID == sam])
            Std[i] = np.nanstd(calc[sampleID == sam])
            Min[i] = np.nanmin(calc[sampleID == sam])
            Max[i] = np.nanmax(calc[sampleID == sam])
            var=calc[sampleID == sam]
            IQR_Std[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))

            i=i+1
    len1=len(calc[sampleID == sam])
    Err_out = pd.DataFrame(data={'Sample': N, '# averaged': len1, 'Mean_calc': Av_mean,
    'Median_calc': Av_median, 'St_dev_calc': Std, 'St_dev_calc_from_percentiles': IQR_Std,
    'Max_calc': Max, 'Min_calc': Min})

    return Err_out


def av_noise_samples_df(dataframe, calc_heading, ID_heading):
    '''
    This function calculates the mean, median, standard devation, maximum and
    minimum value of rows in a datarame with column heading "calc_heading"
    grouping by values in "ID_heading".
    Parameters
    -------
    dataframe: pandas.DataFrame
        Panda datframe of inputs you want to average.
        Must contain column headings "calc_heading" and "ID_heading".
    calc_heading: str
        column heading for the thing you want to average (e.g, P_kbar_calc)
    ID_heading: str
        column heading for the thing you want to average by (e.g., Sample_ID)

    Returns
    -------

    Dataframe with headings "Sample", "Mean_calc", "Median_calc",
    "St_dev_calc", "Max_calc", "Min_calc"

    '''
    calc=dataframe[calc_heading]
    sampleID=dataframe[ID_heading]
    if isinstance(calc, pd.Series):
        N = sampleID.unique()
        Av_mean = np.zeros(len(N), dtype=float)
        Av_median = np.zeros(len(N), dtype=float)
        Max = np.zeros(len(N), dtype=float)
        Min = np.zeros(len(N), dtype=float)
        Std = np.zeros(len(N), dtype=float)
        IQR_Std=np.zeros(len(N), dtype=float)
        for i in range(0, len(N)):
            Av_mean[i] = np.nanmean(calc[sampleID == i])
            Av_median[i] = np.nanmedian(calc[sampleID == i])
            Std[i] = np.nanstd(calc[sampleID == i])
            Min[i] = np.nanmin(calc[sampleID == i])
            Max[i] = np.nanmax(calc[sampleID == i])
            var=calc[sampleID == sam]
            IQR_Std[i]=0.5*np.abs((np.percentile(var, 84) -np.percentile(var, 16)))


    Err_out = pd.DataFrame(data={'Sample': N, 'Mean_calc': Av_mean,
    'Median_calc': Av_median, 'St_dev_calc': Std,'St_dev_calc_from_percentiles': IQR_Std,
    'Max_calc': Max, 'Min_calc': Min})

    return Err_out

def turn_series_into_error(*, elx='Cpx', variable, variable_err):
# Define variables
    n_samples = len(variable_err)
    var = variable

# Define the column names
    cols = [
        'SiO2_{}_Err'.format(elx),
            'TiO2_{}_Err'.format(elx),
            'Al2O3_{}_Err'.format(elx),
            'FeOt_{}_Err'.format(elx),
            'MnO_{}_Err'.format(elx),
            'MgO_{}_Err'.format(elx),
            'CaO_{}_Err'.format(elx),
            'Na2O_{}_Err'.format(elx),
            'K2O_{}_Err'.format(elx),
            'Cr2O3_{}_Err'.format(elx)]

    # Create the empty DataFrame
    Error = pd.DataFrame(data=0, columns=cols, index=range(n_samples))

    # Fill in the appropriate column with the variable value
    var2=var + '_' + elx + '_Err'
    if var2 in cols:
        print(var2)
        Error[var2]=variable_err

    return Error


def add_noise_sample_1phase(phase_comp, phase_err=None,
phase_err_type="Abs",
variable=None, variable_err=None, variable_err_type=None, duplicates=10,
noise_percent=None, err_dist="normal", positive=True,
filter_q=None, append=False):
    '''
    This function generates N duplicates containing random noise from the
    compositions in the dataframe specified by phase_comp.


    Parameters
    -------

    Phase Comps: pandas dataframe
        Pandas dataframe of phase compositions. This can be generated
        from the import_excel function, or any dataframe with the
        headings _Liq for liquids, _Cpx for clinopyroxenes etc.

    Options for adding different types of error:

    1) If you want to specifying an error for >1 variable:

        phase_err: pandas dataframe
            Pandas dataframe with headings for the error of the oxide in each
            phase (e.g., SiO2_Liq_Err, or SiO2_Cpx_Err).
            This dataframe can be generated from a user-inputted spreadsheet
            with these column headings using the function import_excel_errors.
            Errors can be absolute, or percentage errors.
            the default is absolute errors (in wt%), but users can overwrite
            this using phase_err_Type="Perc".


        phase_err_type: "Abs" (default) or "Perc"
            Determins if specified errors are absolute (Abs) or percentage errors.

    2) If you want to specify error for a single variable:

        variable: str
            Name of column you wish to add error to (e.g. "Na2O" for Na2O in Liq)

        variable_err: flt, int
            Specifies how much error to add

    3) If you want to add a fixed percent of noise to all variables.

        noise_percent: flt, int
            Adds a fixed noise percent to all input variables.




    duplicates: flt, int (Default: 10)
        Number of new synthetic samples generated per sample in the original
        dataframe. E.g., if the user enteres 7 samples, and duplicates=1000,
        the function returns 7000 compositions by default.
        If append=True, the original dataframe is appended onto the end of the
        returned dataframe

    err_dist: "normal" (default) or "uniform"
        determins whether added error is normally distributed with
        1 sigma = entered value.
        Or uniformly distributed between +noise value and - noise value.

    positive: True (default) or False
        If True, doesn't allow negative values of oxide species,
        temperature or pressure. Can result in a non-normally distributed
        error distribution. If False, negative values are allowed.

    filter_q: str
        Filter criteria, e.g. if SiO2_Liq>60,
        only returns samples with SiO2_Liq>60

    append: False (default) or True
        If True, appends user-entered dataframe onto the synthetic dataframe
        once noise has been added.



    Returns
    -------

    Panda dataframe containing user-inputted samples with noise added.
    The output is sorted such that the first row in the input * the number
    of noise samples requested are the first N rows, then the new synthetic
    compositions for the second row in the input database are next.
    A heading called "Sample_ID_Liq_Num" is added, with all synthetic samples
    from the first row in the input dataframe have an index 0, the 2nd row have
    an index 1. etc.

    '''

    if phase_err_type not in ['Abs', 'Perc']:
            raise ValueError("Invalid value for phase_err_type. Please choose 'Abs' or 'Perc'.")
    if err_dist not in ['normal', 'uniform']:
            raise ValueError("Invalid value for phase_err_type. Please choose 'normal' or 'uniform'.'")


    # if variable_err is not None:
    #     if (type(variable_err) is not float) and (type(variable_err) is not int) and (type(variable_err) is not np.ndarray):
    #         raise Exception('variable error must be a float, integer, or np.ndarray. If youve entered a pandas series, do series.values')
    #
    if variable is not None and noise_percent is not None:
        raise Exception('noise_percent is an arguement on its own '
        'it adds noise to all variables. Either specify variable or '
        'noise_percent not both')
    if variable_err is not None and noise_percent is not None:
        raise Exception('noise_percent adds noise to all variables' \
        'while variable_err adds noise to a single variable'\
        'specify only one of these arguements')
    if filter_q is not None:
        Sample_c = phase_comp.query(filter_q).copy()

    else:
        Sample_c = phase_comp.copy()
    if phase_err is not None and noise_percent is not None:
        raise Exception('You have entered both a dataframe of noise and '\
        'specified a percent noise. Select only 1 of these options')

    # This works out what phase you have entered data for
    Phase_Options = ["Cpx", "Plag", "Opx", "Sp", "Kspar", "Amp", "Liq", "Ol"]
    for Option in Phase_Options:
        if any(Sample_c.columns.str.contains(f"_{Option}")):
            elx = Option

    if any(Sample_c.columns.str.contains('Sample_ID_{}'.format(elx))):
        name=True
    else:
        Sample_c['Sample_ID_{}'.format(elx)]='No Name Entered'

    if len(Sample_c['Sample_ID_{}'.format(elx)].unique() ) !=  len(Sample_c):
        w.warn('Non unique sample names. We have appended the index onto all sample names to save issues with averaging later')
        TEST=Sample_c.index.values
        for i in range(0, len(Sample_c)):
            Sample_c.loc[i, 'Sample_ID_{}'.format(elx)]=Sample_c['Sample_ID_{}'.format(elx)].iloc[i] + '_'+str(TEST[i])


    if phase_err is None or (phase_err is not None and err_dist == "uniform"):

        Sample_c['Sample_ID_{}_Num'.format(elx)] = Sample_c.index

        # This duplicates your entered composition the number of times
        # specified by noise samples (Cpx1-Cpx1-Cpx1, Cpx2, Cpx2,...)
        Dup_Sample = pd.DataFrame(
            np.repeat(Sample_c.values, duplicates, axis=0))
        Dup_Sample.columns = Sample_c.columns

        # Dropping sample name so it doesnt get averaged.
        Sample_name_num = Dup_Sample['Sample_ID_{}_Num'.format(elx)]
        Sample_name_str=Dup_Sample['Sample_ID_{}'.format(elx)]
        Dup_Sample.drop('Sample_ID_{}_Num'.format(elx), axis=1, inplace=True)
        Dup_Sample.drop('Sample_ID_{}'.format(elx), axis=1, inplace=True)

        if variable is not None and not isinstance(variable, pd.Series) and not isinstance(variable, np.ndarray):

            ely = variable
            if variable == "P_kbar" or variable == "T_K":
                if variable_err_type == "Abs":
                    if err_dist == "normal":
                        Noise = np.random.normal(0, variable_err,
                        Dup_Sample.shape[0])
                    if err_dist == "uniform":
                        Noise = np.random.uniform(- variable_err, +
                        variable_err, Dup_Sample.shape[0])
                if variable_err_type == "Perc":
                    variable_err_abs = Dup_Sample['{}'.format(
                        ely)] * (variable_err / 100)
                    if err_dist == "normal":
                        Noise = np.random.normal(
                        0, variable_err_abs, Dup_Sample.shape[0])
                    if err_dist == "uniform":
                        Noise = np.random.uniform(- variable_err_abs, +
                        variable_err_abs, Dup_Sample.shape[0])

                mynoisedDataframe = Dup_Sample.copy()

                mynoisedDataframe['{}'.format(
                    ely)] = mynoisedDataframe['{}'.format(ely)] + Noise

            else:
                if variable_err_type == "Abs":
                    if err_dist == "normal":
                        Noise = np.random.normal(
                            0, variable_err, Dup_Sample.shape[0])
                    if err_dist == "uniform":
                        Noise = np.random.uniform(- variable_err, +
                                                  variable_err, Dup_Sample.shape[0])
                if variable_err_type == "Perc":
                    variable_err_abs = Dup_Sample['{}_{}'.format(
                        ely, elx)] * (variable_err / 100)
                    if err_dist == "normal":
                        Noise = np.random.normal(
                            0, variable_err_abs, Dup_Sample.shape[0])
                    if err_dist == "uniform":
                        Noise = np.random.uniform(- variable_err_abs, +
                                                  variable_err_abs, Dup_Sample.shape[0])

                mynoisedDataframe = Dup_Sample.copy()
                mynoisedDataframe['{}_{}'.format(
                    ely, elx)] = mynoisedDataframe['{}_{}'.format(ely, elx)] + Noise

        if noise_percent is not None and err_dist == "uniform":
            noise = np.random.uniform(- noise_percent /
                                      100, + noise_percent / 100, Dup_Sample.shape)
            mynoisedDataframe = Dup_Sample + Dup_Sample * noise
        if noise_percent is not None and err_dist == "normal":
            noise = np.random.normal(0, noise_percent / 100, Dup_Sample.shape)
            mynoisedDataframe = Dup_Sample + Dup_Sample * noise



        if phase_err is not None and err_dist == "uniform":
            Sample_Err = phase_err.copy()
            Dup_Noise = pd.DataFrame(
                np.repeat(Sample_Err.values, duplicates, axis=0))
            Dup_Noise.columns = Sample_Err.columns
            noise = np.random.uniform(1, -1, Dup_Noise.shape)
            mynoisedDataframe = (Dup_Noise * noise).to_numpy() + Dup_Sample

        if variable is not None and (isinstance(variable_err, pd.Series) or isinstance(variable_err, np.ndarray)):
            print('got to here')
            phase_err=turn_series_into_error(elx=elx,
variable=variable,
variable_err=variable_err)
            phase_err_type=variable_err_type




    if phase_err is not None and err_dist == "normal":

        # This is for when users enter 2 dataframes, 1 of measurements, 1 of 1
        # sigma errors
        Data = Sample_c
        if 'Sample_ID_{}'.format(elx) in Data:

            Data=Data.drop('Sample_ID_{}'.format(elx), axis=1)



        # Set up empty things to fill in
        SiO2_Err = np.zeros((duplicates * len(Data)), dtype=float)
        TiO2_Err = np.zeros((duplicates * len(Data)), dtype=float)
        Al2O3_Err = np.zeros((duplicates * len(Data)), dtype=float)
        FeOt_Err = np.zeros((duplicates * len(Data)), dtype=float)
        MnO_Err = np.zeros((duplicates * len(Data)), dtype=float)
        MgO_Err = np.zeros((duplicates * len(Data)), dtype=float)
        CaO_Err = np.zeros((duplicates * len(Data)), dtype=float)
        Na2O_Err = np.zeros((duplicates * len(Data)), dtype=float)
        K2O_Err = np.zeros((duplicates * len(Data)), dtype=float)
        Cr2O3_Err = np.zeros((duplicates * len(Data)), dtype=float)
        NiO_Err = np.zeros((duplicates * len(Data)), dtype=float)
        P2O5_Err = np.zeros((duplicates * len(Data)), dtype=float)
        H2O_Err = np.zeros((duplicates * len(Data)), dtype=float)
        P_kbar_Err = np.zeros((duplicates * len(Data)), dtype=float)
        T_K_Err = np.zeros((duplicates * len(Data)), dtype=float)
        F_Err = np.zeros((duplicates * len(Data)), dtype=float)
        Cl_Err = np.zeros((duplicates * len(Data)), dtype=float)
        Sample_name_num = np.zeros((duplicates * len(Data)), dtype=float)
        Sample_name_str = np.zeros((duplicates * len(Data)), dtype=object)

        if phase_err_type == "Abs":
            Err = phase_err
        if phase_err_type == "Perc":
            Err_perc = phase_err.copy()
            # removing headings so can multiply 2 pandas


            Err_perc.columns = Err_perc.columns.str.replace('_Err', '')

            if 'Sample_ID_Cpx' in Err_perc.columns:
                Err_perc = Err_perc.drop('Sample_ID_Cpx', axis=1)



            Err = Data * (Err_perc / 100)
            # adding Err back in
            Err.columns = [str(col) + '_Err' for col in Err.columns]

        for i in range(0, len(Data)):

            if len(Err) != len(Data):
                raise Exception('Your data and error input data frames arent the same length')
            Sample_name_num[i * duplicates:(i * duplicates + duplicates)] = i
            Sample_name_str[i * duplicates:(i * duplicates + duplicates)] = Sample_c['Sample_ID_{}'.format(elx)].iloc[i]


            SiO2_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['SiO2_{}'.format(
                elx)].iloc[i], scale=Err['SiO2_{}_Err'.format(elx)].iloc[i], size=duplicates)

            TiO2_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['TiO2_{}'.format(
                elx)].iloc[i], scale=Err['TiO2_{}_Err'.format(elx)].iloc[i], size=duplicates)

            Al2O3_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['Al2O3_{}'.format(
                elx)].iloc[i], scale=Err['Al2O3_{}_Err'.format(elx)].iloc[i], size=duplicates)

            FeOt_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['FeOt_{}'.format(
                elx)].iloc[i], scale=Err['FeOt_{}_Err'.format(elx)].iloc[i], size=duplicates)

            MnO_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['MnO_{}'.format(
                elx)].iloc[i], scale=Err['MnO_{}_Err'.format(elx)].iloc[i], size=duplicates)

            MgO_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['MgO_{}'.format(
                elx)].iloc[i], scale=Err['MgO_{}_Err'.format(elx)].iloc[i], size=duplicates)

            CaO_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['CaO_{}'.format(
                elx)].iloc[i], scale=Err['CaO_{}_Err'.format(elx)].iloc[i], size=duplicates)

            Na2O_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['Na2O_{}'.format(
                elx)].iloc[i], scale=Err['Na2O_{}_Err'.format(elx)].iloc[i], size=duplicates)

            K2O_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['K2O_{}'.format(
                elx)].iloc[i], scale=Err['K2O_{}_Err'.format(elx)].iloc[i], size=duplicates)

            Cr2O3_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['Cr2O3_{}'.format(
                elx)].iloc[i], scale=Err['Cr2O3_{}_Err'.format(elx)].iloc[i], size=duplicates)

            if variable == "P_kbar":


                P_kbar_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['P_kbar'.format(
                    elx)].iloc[i], scale=Err['P_kbar_Err'.format(elx)].iloc[i], size=duplicates)

            if variable == "T_K":
                T_K_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['T_K'.format(
                    elx)].iloc[i], scale=Err['T_K_Err'.format(elx)].iloc[i], size=duplicates)

            if any(Data.columns.str.contains("NiO")):
                NiO_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['NiO_{}'.format(
                    elx)].iloc[i], scale=Err['NiO_{}_Err'.format(elx)].iloc[i], size=duplicates)
            else:
                NiO_Err = 0 * Data['SiO2_{}'.format(elx)]

            if any(Data.columns.str.contains("F_")):
                F_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['F_{}'.format(
                    elx)].iloc[i], scale=Err['F_{}_Err'.format(elx)].iloc[i], size=duplicates)
            else:
                F_Err = 0 * Data['SiO2_{}'.format(elx)]

            if any(Data.columns.str.contains("Cl_")):
                Cl_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['Cl_{}'.format(
                    elx)].iloc[i], scale=Err['Cl_{}_Err'.format(elx)].iloc[i], size=duplicates)
            else:
                Cl_Err = 0 * Data['SiO2_{}'.format(elx)]

            if any(Data.columns.str.contains("P2O5")):
                P2O5_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['P2O5_{}'.format(
                    elx)].iloc[i], scale=Err['P2O5_{}_Err'.format(elx)].iloc[i], size=duplicates)
            else:
                P2O5_Err = 0 * Data['SiO2_{}'.format(elx)]

            if any(Data.columns.str.contains("H2O")):
                H2O_Err[i * duplicates:(i * duplicates + duplicates)] = np.random.normal(loc=Data['H2O_{}'.format(
                    elx)].iloc[i], scale=Err['H2O_{}_Err'.format(elx)].iloc[i], size=duplicates)
            else:
                H2O_Err = 0 * Data['SiO2_{}'.format(elx)]

            if elx == 'Cpx' or elx == "Opx" or elx == "Plag" or elx == "Kspar":
                mynoisedDataframe = pd.DataFrame(data={'SiO2_{}'.format(elx): SiO2_Err, 'TiO2_{}'.format(elx): TiO2_Err, 'Al2O3_{}'.format(elx): Al2O3_Err, 'FeOt_{}'.format(elx): FeOt_Err, 'MnO_{}'.format(
                    elx): MnO_Err, 'MgO_{}'.format(elx): MgO_Err, 'CaO_{}'.format(elx): CaO_Err, 'Na2O_{}'.format(elx): Na2O_Err, 'K2O_{}'.format(elx): K2O_Err, 'Cr2O3_{}'.format(elx): Cr2O3_Err})

            if elx == 'Ol' or elx == "Sp":
                mynoisedDataframe = pd.DataFrame(data={'SiO2_{}'.format(elx): SiO2_Err, 'TiO2_{}'.format(elx): TiO2_Err, 'Al2O3_{}'.format(elx): Al2O3_Err, 'FeOt_{}'.format(elx): FeOt_Err, 'MnO_{}'.format(elx): MnO_Err, 'MgO_{}'.format(elx): MgO_Err, 'CaO_{}'.format(elx): CaO_Err, 'Na2O_{}'.format(elx): Na2O_Err, 'K2O_{}'.format(elx): K2O_Err, 'Cr2O3_{}'.format(elx): Cr2O3_Err,
                                                       'NiO_{}'.format(elx): NiO_Err})
            if elx == "Amp":
                mynoisedDataframe = pd.DataFrame(data={'SiO2_{}'.format(elx): SiO2_Err, 'TiO2_{}'.format(elx): TiO2_Err, 'Al2O3_{}'.format(elx): Al2O3_Err, 'FeOt_{}'.format(elx): FeOt_Err, 'MnO_{}'.format(elx): MnO_Err, 'MgO_{}'.format(elx): MgO_Err, 'CaO_{}'.format(elx): CaO_Err, 'Na2O_{}'.format(elx): Na2O_Err, 'K2O_{}'.format(elx): K2O_Err, 'Cr2O3_{}'.format(elx): Cr2O3_Err,
                                                       'F_{}'.format(elx): F_Err, 'Cl_{}'.format(elx): Cl_Err})
            if elx == "Liq":
                mynoisedDataframe = pd.DataFrame(data={'SiO2_{}'.format(elx): SiO2_Err, 'TiO2_{}'.format(elx): TiO2_Err, 'Al2O3_{}'.format(elx): Al2O3_Err, 'FeOt_{}'.format(elx): FeOt_Err, 'MnO_{}'.format(elx): MnO_Err, 'MgO_{}'.format(elx): MgO_Err, 'CaO_{}'.format(elx): CaO_Err, 'Na2O_{}'.format(elx): Na2O_Err, 'K2O_{}'.format(elx): K2O_Err, 'Cr2O3_{}'.format(elx): Cr2O3_Err,
                                                       'P2O5_{}'.format(elx): P2O5_Err, 'H2O_{}'.format(elx): H2O_Err})
                mynoisedDataframe = mynoisedDataframe.reindex(
                    df_ideal_liq.columns, axis=1).fillna(0)
                mynoisedDataframe = mynoisedDataframe.apply(
                    pd.to_numeric, errors='coerce').fillna(0)

        if variable == "T_K":
            mynoisedDataframe['P_kbar'] = P_kbar_Err
        if variable == "P_kbar":
            mynoisedDataframe['T_K'] = T_K_Err

    mynoisedDataframe['Sample_ID_{}_Num'.format(elx)] = Sample_name_num

    mynoisedDataframe['Sample_ID_{}'.format(elx)] = Sample_name_str

    if positive is True:
        num = mynoisedDataframe._get_numeric_data()
        num[num < 0] = 0
        print('All negative numbers replaced with zeros. '\
        'If you wish to keep these, set positive=False')

    mynoisedDataframe=mynoisedDataframe.fillna(0)
    if append is True:
        mynoisedDataframe2 = pd.concat([Sample_c, mynoisedDataframe], axis=0)
        return mynoisedDataframe2
    else:
        return mynoisedDataframe


def calculate_bootstrap_mixes(
        endmember1, endmember2, num_samples, self_mixing=False):
    '''Specify 2 end-members, generates synthetic liquids from mixing between these end-members

   Parameters
    -------

    endmember1: pandas.DataFrame
        Panda DataFrame of liquid compositions for end-member 1, with column headings SiO2_Liq etc.

    endmember2: pandas.DataFrame
        Panda DataFrame of liquid compositions for end-member 2, with column headings SiO2_Liq etc.

    num_samples: float or int
        If num_samples is less than the length of the end members, will randomly resample liquids entered to get to sufficient N.
        If num_samples greater than length of end members, will randomly downsample liquids to N=num_samples.

    self_mixing: None, False, True, "Partial"
        If None or False, will mix 2 end members in various proportions, but no mixing between end members
        If True, will mix between samples from a given end member as well as between the 2 end members.
        If Partial, half of outputted liquids will be generated by mixing within and between end members, and the other half from mixing between end members.

    Returns:
    -------
    pandas DataFrame
        synthetic liquids generated by mixing between end-members with column headings "SiO2_Liq" etc.
    '''
    Elements = ['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq', 'FeOt_Liq',
                'MnO_Liq', 'MgO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq', 'Cr2O3_Liq',
                'P2O5_Liq', 'H2O_Liq']
    f = np.repeat(np.random.uniform(0, 1, (num_samples, 1)),
                  (len(Elements)), axis=1)

    # Takes half mixes from self mixing, half from mixing between defined
    # end-members
    if self_mixing == "Partial":
        # self mixing part
        my_dataset3_self = pd.concat([endmember1, endmember2], ignore_index=True)
        endmember1_self = my_dataset3_self[Elements].sample(
            n=num_samples, replace=True).to_numpy()
        endmember2_self = my_dataset3_self[Elements].sample(
            n=num_samples, replace=True).to_numpy()
        combined_model_self = endmember1_self * f + endmember2_self * (1 - f)
        # normal mixing part
        endmember1 = endmember1[Elements].sample(
            n=num_samples, replace=True).to_numpy()
        endmember2 = endmember2[Elements].sample(
            n=num_samples, replace=True).to_numpy()
        combined_model = endmember1 * f + endmember2 * (1 - f)
        myDataframe_self = pd.DataFrame()
        for ix, my_el in enumerate(Elements):
            myDataframe_self[my_el] = combined_model_self[:, ix]
        myDataframe_mix = pd.DataFrame()
        for ix, my_el in enumerate(Elements):
            myDataframe_mix[my_el] = combined_model[:, ix]

        myDataframe = pd.concat([myDataframe_mix, myDataframe_self], )
        myDataframe = myDataframe.sample(n=num_samples, replace=True)

    if self_mixing is True:
        my_dataset3 = pd.concat([endmember1, endmember2], ignore_index=True)

        endmember1 = my_dataset3[Elements].sample(
            n=num_samples, replace=True).to_numpy()
        endmember2 = my_dataset3[Elements].sample(
            n=num_samples, replace=True).to_numpy()

    if self_mixing is False or self_mixing is None:
        endmember1 = endmember1[Elements].sample(
            n=num_samples, replace=True).to_numpy()
        endmember2 = endmember2[Elements].sample(
            n=num_samples, replace=True).to_numpy()

    if self_mixing is False or self_mixing is None or self_mixing is True:
        combined_model = endmember1 * f + endmember2 * (1 - f)

        myDataframe = pd.DataFrame()
        for ix, my_el in enumerate(Elements):
            myDataframe[my_el] = combined_model[:, ix]
    myDataframe = myDataframe.fillna(0)
    return myDataframe  # , f, endmember1, endmember2


