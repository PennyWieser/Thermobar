import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import inspect
import warnings as w
import numbers
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

# Redit says equation for RMSE ignores DoF. so /n instead of /n-k-1
def RMSE_func(x, x2):
    N=len(x)
    sum_squares=np.sum((x-x2)**2)
    RMSE=np.sqrt((1/N)*sum_squares)
    return RMSE


def Tukey_calc(x,y): #, name):
    x=pd.Series(x)
    y=pd.Series(y)
    vals=np.array([3.1, 6.2, 9.3, 13])
    X_Av=np.empty(len(vals), dtype=float)
    Y_Av=np.empty(len(vals), dtype=float)
    X_MedAv=np.empty(len(vals), dtype=float)
    Y_MedAv=np.empty(len(vals), dtype=float)

    Y_std=np.empty(len(vals), dtype=float)
    X_std=np.empty(len(vals), dtype=float)
    mask_iter=pd.DataFrame(index=x.index)
    mask_iter=pd.DataFrame(index=x.index)
    from scipy.stats import f_oneway
    for i in range(0, len(vals)):
        if i==0:
            elx=i
            mask_iter['mask_{}'.format(elx)]=(x<=vals[i])
            X_Av[i]=np.nanmean(x.loc[mask_iter['mask_{}'.format(elx)]])
            Y_Av[i]=np.nanmean(y.loc[mask_iter['mask_{}'.format(elx)]])
            X_MedAv[i]=np.nanmedian(x.loc[mask_iter['mask_{}'.format(elx)]])
            Y_MedAv[i]=np.nanmedian(y.loc[mask_iter['mask_{}'.format(elx)]])
            Y_std[i]=np.nanstd(y.loc[mask_iter['mask_{}'.format(elx)]])
            X_std[i]=np.nanstd(x.loc[mask_iter['mask_{}'.format(elx)]])

        else:
            elx=i
            mask_iter['mask_{}'.format(elx)]=(x>vals[i-1])&(x<=vals[i])
        #
            X_Av[i]=np.nanmean(x.loc[mask_iter['mask_{}'.format(elx)]])
            Y_Av[i]=np.nanmean(y.loc[mask_iter['mask_{}'.format(elx)]])
            X_MedAv[i]=np.nanmedian(x.loc[mask_iter['mask_{}'.format(elx)]])
            Y_MedAv[i]=np.nanmedian(y.loc[mask_iter['mask_{}'.format(elx)]])

            Y_std[i]=np.nanstd(y.loc[mask_iter['mask_{}'.format(elx)]])
            X_std[i]=np.nanstd(x.loc[mask_iter['mask_{}'.format(elx)]])

    a=np.array(y.loc[mask_iter['mask_0']].dropna())
    b=np.array(y.loc[mask_iter['mask_1']].dropna())
    c=np.array(y.loc[mask_iter['mask_2']].dropna())
    d=np.array(y.loc[mask_iter['mask_3']].dropna())
    data=[a, b, c, d]
    dfs=[pd.DataFrame({'score':a, 'group':i}) for a,i in zip(data, range(len(data)))]
    df=pd.concat(dfs, axis='rows')
    df['group'].replace({0: "Upper", 1: "Mid", 2: "Lower", 3: "Moho"}, inplace=True)

    tukey=pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05);

    plt.annotate(print(tukey), xy=(1, 1.5), xycoords="axes fraction", fontsize=9)


def calculate_R2(x, y, xy=True, df=False, round=5):
    """ Calculates statistics
    if xy= False doesnt return y and x pred
    """
    masknan=(~np.isnan(x) & ~np.isnan(y))
    regx=x[masknan].values.reshape(-1, 1)
    regy=y[masknan].values.reshape(-1, 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(regx[:,0],regy[:,0])
    p_value=np.round(p_value, round)
    lr=LinearRegression()
    lr.fit(regx,regy)
    Y_pred=lr.predict(regx)
    Int=lr.intercept_
    Grad=lr.coef_
    #R="R\N{SUPERSCRIPT TWO} = " +  str(np.round(r2_score(regy, Y_pred), 2))
    R=(np.round(r2_score(regy, Y_pred), round))
    RMSE=RMSE_func(regx, regy)

    RMSEp=np.round(RMSE, round)
    Median=np.nanmedian(regy-regx)#
    Medianp=np.round(Median, round)
    Mean=np.nanmean(regy-regx)
    Meanp=np.round(Mean, round)
    if round is False:
        return {'R2': '{0:.5f}'.format(R), 'RMSE':'{0:.5f}'.format(RMSEp), 'RMSE_num':RMSEp,
        'P_val':'{0:.5f}'.format(p_value), 'Median':'{0:.5f}'.format(Medianp), 'Mean':'{0:.5f}'.format(Meanp),
        'Int': Int, 'Grad':Grad[0]}

    if xy is True:

        return {'R2': '{0:.2f}'.format(R), 'RMSE':'{0:.2f}'.format(RMSEp), 'RMSE_num':RMSEp,
        'P_val':'{0:.3f}'.format(p_value), 'Median':'{0:.2f}'.format(Medianp), 'Mean':'{0:.2f}'.format(Meanp),
        'Int': Int, 'Grad':Grad[0],
    'x_pred': regx, 'y_pred': Y_pred}

    if xy is False and df is False:
        return {'R2': '{0:.2f}'.format(R), 'RMSE':'{0:.2f}'.format(RMSEp), 'RMSE_num':RMSEp,
        'P_val':'{0:.3f}'.format(p_value), 'Median':'{0:.2f}'.format(Medianp), 'Mean':'{0:.2f}'.format(Meanp),
        'Int': Int, 'Grad':Grad[0]}

    if xy is False and df is True:
        df=pd.DataFrame(data={'R2': R,
                                'RMSE': RMSEp,
                                'P_val': p_value,
                                'Median': Medianp,
                                'Mean': Meanp,
                                'Int': Int,
                                'Grad': Grad[0]
        })
        df=df.round(decimals=2)
        return df

def calculate_R2_devitre(x, y, xy=True, df=False, round=5,pval_format='decimal'):
    """ Calculates statistics
    if xy= False doesn't return y and x pred
    """
    masknan = (~np.isnan(x) & ~np.isnan(y))
    regx = x[masknan].values.reshape(-1, 1)
    regy = y[masknan].values.reshape(-1, 1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(regx[:, 0], regy[:, 0])

    lr = LinearRegression()
    lr.fit(regx, regy)
    Y_pred = lr.predict(regx)
    Int = lr.intercept_[0]
    Grad = lr.coef_[0][0]

    R = r2_score(regy, Y_pred)
    RMSE = np.sqrt(np.mean((regy - Y_pred) ** 2))
    Median = np.nanmedian(regy - regx)
    Mean = np.nanmean(regy - regx)

    if pval_format=='decimal':
        p_value=format(p_value, '.'+str(round)+'f')

    Rp = np.round(R, round)
    RMSEp = np.round(RMSE, round)
    Medianp = np.round(Median, round)
    Meanp = np.round(Mean, round)
    Intp=np.round(Int,round)
    Gradp=np.round(Grad,round)

    output = {
        "R\u00B2": str(Rp),
        'RMSE': str(RMSEp),
        'P_val': str(p_value),
        'Median': str(Medianp),
        'Mean': str(Meanp),
        'Int': str(Intp),
        'Grad': str(Gradp)
    }

    if xy:
        output['x_pred'] = Y_pred.tolist()
        output['y_pred'] = regy.tolist()

    return output


def calculate_R2_Tukey(x, y):
    masknan=(~np.isnan(x) & ~np.isnan(y))
    regx=x[masknan].values.reshape(-1, 1)
    regy=y[masknan].values.reshape(-1, 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(regx[:,0],regy[:,0])
    p_value=np.round(p_value, 3)
    lr=LinearRegression()
    lr.fit(regx,regy)
    Y_pred=lr.predict(regx)
    Int=lr.intercept_
    Int=np.round(Int[0], 2)
    Grad=lr.coef_
    Grad=np.round(Grad[0][0], 2)
    #R="R\N{SUPERSCRIPT TWO} = " +  str(np.round(r2_score(regy, Y_pred), 2))
    R=(np.round(r2_score(regy, Y_pred), 2))
    RMSE=RMSE_func(regx, regy)
    RMSEp=np.round(RMSE, 1)
    Median=np.nanmedian(regy-regx)#
    Medianp=np.round(Median, 2)
    Mean=np.nanmean(regy-regx)
    Meanp=np.round(Mean, 2)


    return {'R$^{2}$': '{0:.2f}'.format(R), 'RMSE':'{0:.2f}'.format(RMSEp), 'RMSE_num':RMSEp, 'Grad':Grad, 'Int': Int,
    'Median Error':'{0:.2f}'.format(Medianp), 'Mean Error':'{0:.2f}'.format(Meanp), 'p value':'{0:.3f}'.format(p_value),
     }

def calculate_R2_np(x, y, xy=True):
    if len(x)!=len(y):
        raise TypeError('X and y not same length')

    masknan=(~np.isnan(x) & ~np.isnan(y))

    regx=x[masknan].reshape(-1, 1)
    regy=y[masknan].reshape(-1, 1)
    lr=LinearRegression()
    lr.fit(regx,regy)
    Y_pred=lr.predict(regx)
    R='R2' +  str(np.round(r2_score(regy, Y_pred), 2))
    RMSE=RMSE_func(regx, regy)
    RMSEp='RMSE= ' +  str(np.round(RMSE, 5))
    Median=np.nanmedian(regy-regx)#
    Medianp='Median Off= ' +  str(np.round(Median, 5))
    Mean=np.nanmean(regy-regx)
    Meanp='Mean Off= ' +  str(np.round(Mean, 5))
    Intercept=lr.intercept_
    grad=lr.coef_

    if xy is True:
        return {'R2': R, 'RMSE':RMSEp, 'Median':Medianp, 'Mean':Meanp, 'Grad': grad, 'Int': Intercept, 'x_pred': regx, 'y_pred': Y_pred}
    else:
        return {'R2': R, 'RMSE':RMSEp, 'Median':Medianp, 'Mean':Meanp, 'Grad': grad, 'Int': Intercept}


def Tukey_Plot_np(x,y, name, xlower=-1, xupper=13, yupper=17, ylower=-3):
    x=pd.Series(x)
    y=pd.Series(y)
    with w.catch_warnings():
        w.simplefilter('ignore')
        fig, (ax1) = plt.subplots(1,1, figsize = (6,5)) # adjust dimensions of figure here
        ax1.set_ylabel('calculated Press')
        ax1.set_xlabel('Experimental P')
        vals=np.array([3.1, 6.2, 9.3, 13])
        X_Av=np.empty(len(vals), dtype=float)
        Y_Av=np.empty(len(vals), dtype=float)
        X_MedAv=np.empty(len(vals), dtype=float)
        Y_MedAv=np.empty(len(vals), dtype=float)

        Y_std=np.empty(len(vals), dtype=float)
        X_std=np.empty(len(vals), dtype=float)
        mask_iter=pd.DataFrame(index=x.index)
        from scipy.stats import f_oneway
        for i in range(0, len(vals)):
            if i==0:
                elx=i
                mask_iter['mask_{}'.format(elx)]=(x<=vals[i])
                X_Av[i]=np.nanmean(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_Av[i]=np.nanmean(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_MedAv[i]=np.nanmedian(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_MedAv[i]=np.nanmedian(y.loc[mask_iter['mask_{}'.format(elx)]])
                Y_std[i]=np.nanstd(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_std[i]=np.nanstd(x.loc[mask_iter['mask_{}'.format(elx)]])

            else:
                elx=i
                mask_iter['mask_{}'.format(elx)]=(x>vals[i-1])&(x<=vals[i])
        #
                X_Av[i]=np.nanmean(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_Av[i]=np.nanmean(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_MedAv[i]=np.nanmedian(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_MedAv[i]=np.nanmedian(y.loc[mask_iter['mask_{}'.format(elx)]])

                Y_std[i]=np.nanstd(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_std[i]=np.nanstd(x.loc[mask_iter['mask_{}'.format(elx)]])

        ax1.plot(x, y, 'ok', alpha=0.05, zorder=0)
        ax1.plot([0, xupper], [0, xupper], '-r')
        ax1.errorbar(X_Av, Y_Av, xerr=X_std, yerr=Y_std,
                     fmt='d', ecolor='k', elinewidth=0.8, mfc='cyan', ms=10, mec='k')
        #plt.plot(X_Av, Y_MedAv, 'sk')
        ax1.set_ylim([ylower, yupper])
        ax1.set_xlim([xlower, xupper])



        import matplotlib.patches as patches
        rectU = patches.Rectangle((0,0),vals[0],vals[0],linewidth=1,edgecolor='k',facecolor='yellow', alpha=0.05)
        rectM = patches.Rectangle((vals[0],vals[0]), vals[1]-vals[0], vals[1]-vals[0],linewidth=1,edgecolor='k',facecolor='red', alpha=0.05)
        rectL = patches.Rectangle((vals[1],vals[1]), vals[2]-vals[1], vals[2]-vals[1],linewidth=1,edgecolor='k',facecolor='blue', alpha=0.05)
        rectMo = patches.Rectangle((vals[2],vals[2]), vals[3]-vals[2], vals[3]-vals[2],linewidth=1,edgecolor='k',facecolor='grey', alpha=0.05)

        # Add the patch to the Axes
        ax1.add_patch(rectU)
        ax1.add_patch(rectM)
        ax1.add_patch(rectL)
        ax1.add_patch(rectMo)
        vals=np.array([3.1, 6.2, 9.3, 13])
        ax1.annotate("Upper", xy=(1, 16), xycoords="data", fontsize=12)
        ax1.annotate("Mid", xy=(4, 16), xycoords="data", fontsize=12)
        ax1.annotate("Lower", xy=(7, 16), xycoords="data", fontsize=12)
        ax1.annotate("Moho", xy=(11, 16), xycoords="data", fontsize=12)

        a=np.array(y.loc[mask_iter['mask_0']].dropna())
        b=np.array(y.loc[mask_iter['mask_1']].dropna())
        c=np.array(y.loc[mask_iter['mask_2']].dropna())
        d=np.array(y.loc[mask_iter['mask_3']].dropna())
        data=[a, b, c, d]
        dfs=[pd.DataFrame({'score':a, 'group':i}) for a,i in zip(data, range(len(data)))]
        df=pd.concat(dfs, axis='rows')
        df['group'].replace({0: "Upper", 1: "Mid", 2: "Lower", 3: "Moho"}, inplace=True)

        tukey=pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05);

        ax1.annotate(print(tukey), xy=(1, 1.5), xycoords="axes fraction", fontsize=9)
        Stats=calculate_R2(x, y)
        ax1.annotate(Stats['R2'], xy=(0.1, 0.88), xycoords="axes fraction", fontsize=9)
        ax1.annotate(Stats['RMSE'], xy=(0.1, 0.72), xycoords="axes fraction", fontsize=9)
        ax1.annotate(Stats['Mean'], xy=(0.1, 0.83), xycoords="axes fraction", fontsize=9)
        ax1.annotate(Stats['Median'], xy=(0.1, 0.78), xycoords="axes fraction", fontsize=9)
        strname=str(name)
        ax1.set_title(strname)

def Experimental_av_values(LEPRin, calc, name):

    ExperimentNumbers=LEPRin[name].unique()

    for exp in ExperimentNumbers:
        dff_M=pd.DataFrame(calc.loc[LEPRin[name]==exp].mean(axis=0)).T
        dff_Med=pd.DataFrame(calc.loc[LEPRin[name]==exp].median(axis=0)).T
        dff_M['Median_P_kbar_calc']=dff_Med['P_kbar_calc']
        if "T_K_calc" in dff_M:
            dff_M['Median_T_K_calc']=dff_Med['T_K_calc']

        dff_S=pd.DataFrame(calc.loc[LEPRin[name]==exp].std(axis=0)).T

        dff_M['Sample_ID']=LEPRin.loc[LEPRin[name]==exp, "Experiment_y"].iloc[0]
        dff_M['Pressure_Exp']=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'P_kbar_x'])
        dff_M['Temp_Exp']=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'T_K_x'])
        dff_M['Exp']=exp

        if exp==ExperimentNumbers[0]:
            df1_M=dff_M
        else:
            df1_M=pd.concat([df1_M, dff_M],  sort=False)

        if np.shape(LEPRin.loc[LEPRin[name]==exp])[0]==1: # This tells us if there is only 1, in which case std will return Nan
                    dff_S= dff_S.fillna(0)
                    dff_S['N']=1
        else:
            dff_S=dff_S
            dff_S['N']=np.shape(calc.loc[LEPRin[name]==exp])[0]
        if exp==ExperimentNumbers[0]:
            df1_S=dff_S
        else:
            df1_S=pd.concat([df1_S, dff_S])

    df1_M= df1_M.add_prefix('Mean_')
    df1_S=df1_S.add_prefix('st_dev_')
    df1_M.insert(0, "No. of Exp. averaged",  df1_S['st_dev_N'])
    df1_M.insert(1, "std_P_kbar_calc",  df1_S['st_dev_P_kbar_calc'])
    return df1_M

def calculate_average_values2(LEPRin, calc, name):
    ExperimentNumbers=LEPRin[name].unique()

    for exp in ExperimentNumbers:
        dff_M=calc.loc[LEPRin[name]==exp].mean(axis=0)
        dff_Med=calc.loc[LEPRin[name]==exp].median(axis=0)
        dff_SD=calc.loc[LEPRin[name]==exp].std(axis=0)
        mean_H2O=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'H2O_Liq'])
        #Sample_ID=LEPRin.loc[LEPRin[name]==exp, ''].iloc[0]
        P_Exp=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'P_kbar_x'])
        T_Exp=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'T_K_x'])
        Exp=exp
        NExp=len(LEPRin.loc[LEPRin[name]==exp, 'P_kbar_x'])
        df=pd.DataFrame(data={'Mean': dff_M, 'Median': dff_Med, 'Std': dff_SD,
                               'P_Exp': P_Exp, 'Av_H2O_Exp': mean_H2O, 'T_Exp': T_Exp, 'Exp': exp, 'NExp': NExp}, index=[0])

        if exp==ExperimentNumbers[0]:
            df1_M=df
        else:
            df1_M=pd.concat([df1_M, df],  sort=False)
    return df1_M

def Experimental_av_plot(LEPRin, calc, name="name"):
 # before, was Experiment_P_Name_y rather than name
    ExperimentNumbers=LEPRin[name].unique()

    for exp in ExperimentNumbers:
        dff_M=pd.DataFrame(calc.loc[LEPRin[name]==exp].mean(axis=0)).T
        dff_Med=pd.DataFrame(calc.loc[LEPRin[name]==exp].median(axis=0)).T
        dff_M['Median_P_kbar_calc']=dff_Med['P_kbar_calc']
        dff_M['Median_T_K_calc']=dff_Med['T_K_calc']

        dff_S=pd.DataFrame(calc.loc[LEPRin[name]==exp].std(axis=0)).T

        dff_M['Sample_ID']=LEPRin.loc[LEPRin[name]==exp, "Experiment_y"].iloc[0]
        dff_M['Pressure_Exp']=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'P_kbar_x'])
        dff_M['Temp_Exp']=np.nanmean(LEPRin.loc[LEPRin[name]==exp, 'T_K_x'])

        if exp==ExperimentNumbers[0]:
            df1_M=dff_M
        else:
            df1_M=pd.concat([df1_M, dff_M],  sort=False)

        if np.shape(LEPRin.loc[LEPRin[name]==exp])[0]==1: # This tells us if there is only 1, in which case std will return Nan
                    dff_S= dff_S.fillna(0)
                    dff_S['N']=1
        else:
            dff_S=dff_S
            dff_S['N']=np.shape(calc.loc[LEPRin[name]==exp])[0]
        if exp==ExperimentNumbers[0]:
            df1_S=dff_S
        else:
            df1_S=pd.concat([df1_S, dff_S])

    df1_M= df1_M.add_prefix('Mean_')
    df1_S=df1_S.add_prefix('st_dev_')
    df1_M.insert(0, "No. of Exp. averaged",  df1_S['st_dev_N'])
    df1_M.insert(1, "std_P_kbar_calc",  df1_S['st_dev_P_kbar_calc'])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5)) # adjust dimensions of figure here
    ax1.plot(LEPRin['P_kbar_x'], calc['P_kbar_calc'], 'ok', alpha=0.05, zorder=0, markersize=3)
    ax1.set_ylabel('calculated Press')
    ax1.set_xlabel('Experimental P')
    #ax1.plot(df1_M['Mean_Pressure_Exp'], df1_M['Mean_P_kbar_calc'], 'ok')
    Stats=calculate_R2(df1_M['Mean_Pressure_Exp'], df1_M['Mean_Median_P_kbar_calc'])
    ax1.annotate(Stats['R2'], xy=(0.1, 0.88), xycoords="axes fraction", fontsize=9)
    ax1.annotate(Stats['RMSE'], xy=(0.1, 0.72), xycoords="axes fraction", fontsize=9)
    ax1.annotate(Stats['Mean'], xy=(0.1, 0.83), xycoords="axes fraction", fontsize=9)
    ax1.annotate(Stats['Median'], xy=(0.1, 0.78), xycoords="axes fraction", fontsize=9)
    ax1.plot([min(df1_M['Mean_Median_P_kbar_calc']), max(df1_M['Mean_Median_P_kbar_calc'])],

            [min(df1_M['Mean_Median_P_kbar_calc']), max(df1_M['Mean_Median_P_kbar_calc'])], '-k')
    ax1.plot(Stats['x_pred'], Stats['y_pred'], '--r', lw=0.1)
    ax1.errorbar(df1_M['Mean_Pressure_Exp'], df1_M['Mean_Median_P_kbar_calc'],  yerr= df1_M['std_P_kbar_calc'],
                     fmt='d', ecolor='k', elinewidth=0.1, mfc='red', markeredgewidth=0.1, ms=5, mec='k')


    ax1.plot(df1_M['Mean_Pressure_Exp'], df1_M['Mean_Median_P_kbar_calc'], 'ok', mfc='r')

    ax2.plot(df1_M['No. of Exp. averaged'], abs(df1_M['Mean_Pressure_Exp']-df1_M['Mean_P_kbar_calc']), 'ok', mfc='r')
    ax2.set_xlabel('# of experiments averaged')
    ax2.set_ylabel('ABS (Mean Exp Pressure - Mean calc Pressure)')
    #ax2.set_xscale('log',base=2)
    return df1_M

def Tukey_Plot_np_values(x,y, name, xlower=-1, xupper=13, yupper=17, ylower=-3):
    x=pd.Series(x)
    y=pd.Series(y)
    with w.catch_warnings():
        w.simplefilter('ignore')
        vals=np.array([3.1, 6.2, 9.3, 13])
        X_Av=np.empty(len(vals), dtype=float)
        Y_Av=np.empty(len(vals), dtype=float)
        X_MedAv=np.empty(len(vals), dtype=float)
        Y_MedAv=np.empty(len(vals), dtype=float)

        Y_std=np.empty(len(vals), dtype=float)
        X_std=np.empty(len(vals), dtype=float)
        mask_iter=pd.DataFrame(index=x.index)
        from scipy.stats import f_oneway
        for i in range(0, len(vals)):
            if i==0:
                elx=i
                mask_iter['mask_{}'.format(elx)]=(x<=vals[i])
                X_Av[i]=np.nanmean(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_Av[i]=np.nanmean(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_MedAv[i]=np.nanmedian(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_MedAv[i]=np.nanmedian(y.loc[mask_iter['mask_{}'.format(elx)]])
                Y_std[i]=np.nanstd(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_std[i]=np.nanstd(x.loc[mask_iter['mask_{}'.format(elx)]])

            else:
                elx=i
                mask_iter['mask_{}'.format(elx)]=(x>vals[i-1])&(x<=vals[i])
        #
                X_Av[i]=np.nanmean(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_Av[i]=np.nanmean(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_MedAv[i]=np.nanmedian(x.loc[mask_iter['mask_{}'.format(elx)]])
                Y_MedAv[i]=np.nanmedian(y.loc[mask_iter['mask_{}'.format(elx)]])

                Y_std[i]=np.nanstd(y.loc[mask_iter['mask_{}'.format(elx)]])
                X_std[i]=np.nanstd(x.loc[mask_iter['mask_{}'.format(elx)]])


    return pd.DataFrame(data={'X_Av':X_Av, 'Y_Av':Y_Av, 'X_std':X_std, 'Y_std':Y_std})

def mantle_geotherm_plot(T, P, Depth, plot_style, Temp_unit, T_Sample, P_Sample, T_std, P_std, max_depth, plot_type, **kwargs):

    '''
    A function to plot calculate geotherm alongside the thermobarometric
    calculations.

    ###Parameters###
    T: Temperature array of the geotherm.

    P: Pressure array of the geotherm.

    Depth: Depth array of the geotherm in meters.

    plot_style: String parameter for the y-axis of the geotherm plot 'Pressure' or 'Depth'.

    Temp_unit: String parameter for the temperature unit, 'Celsius' or 'Kelvin'.

    T_Sample: Array of temperature of the thermobarometric solutions.

    P_Sample: Array of pressure of the thermobarometric solutions in GPa.

    T_std: Standart deviation of thermobarometric temperature estimation. Could be array or a single value.

    P_std: Standart deviation of thermobarometric pressure estimation. Could be array or a single value.

    max_depth: Maximum depth to show the plot.

    leg: Boolean parameter to set existence of a legend.

    plot_type: 'show' or 'save' the figure.

    moho: moho depth in km.

    lab: lab depth in km.

    Depth_Sample: Array of depths of the thermobarometric solutions.

    filename_save: string parameter for filename to save the figures.
    '''

    moho = kwargs.pop('moho', None)
    lab = kwargs.pop('lab', None)
    Depth_Sample = kwargs.pop('Depth_Sample', None)
    filename_save = kwargs.pop('filename_save', 'Geotherm_Plot.png')
    leg = kwargs.pop('leg', True)

    Depth = Depth/1e3

    fig = plt.figure(figsize = (4,10))
    ax1 = plt.subplot(111)
    if Temp_unit == 'Celsius':
        T = np.array(T) - 273.15
        ax1.set_xlabel('Temperature [$^{\circ} C$]')
    else:
        ax1.set_xlabel('Temperature [$^{\circ} K$]')

    if P_std is not list:
        P_std = np.ones(len(P_Sample)) * P_std
    if T_std is not list:
        T_std = np.ones(len(T_Sample)) * T_std

    if plot_style == 'Pressure':
        ax1.plot(T,P,'k',lw = 1.5)
        ax1.set_ylabel('Pressure [GPa]')
        if (T_Sample is not None) and (P_Sample is not None):

            if (Temp_unit == 'Celsius'):
                T_Sample = np.array(T_Sample) - 273.15

            ax1.errorbar(T_Sample, P_Sample, yerr = [P_std,P_std],xerr = [T_std,T_std],
            fmt = 'o',color = '#bd3b24',markersize = 5,label = 'Xenolith Data',
            ecolor = 'k',elinewidth = 0.5,alpha = 0.6, markeredgecolor = 'k')

        ax1.set_ylim(np.amax(P),0)

        ax1.set_xlim(0, np.amax(np.concatenate((T,T_Sample))) + 100.0)

    elif plot_style == 'Depth':
        ax1.plot(T,Depth,'k',lw = 1.5)
        ax1.set_ylabel('Depth [km]')

        if (T_Sample is not None):
            if (Temp_unit == 'Celsius'):
                T_Sample = np.array(T_Sample) - 273.15
            positive_pressure_diff_list = []
            negative_pressure_diff_list = []
            if Depth_Sample is None:
                Depth_Sample = np.zeros(len(P_Sample))
                run_dep = True
            for i in range(0,len(P_std)):

                if run_dep == True:
                    Depth_Sample[i] = Depth[(np.abs(P-P_Sample[i])).argmin()]
                positive_pressure = P_Sample[i] + P_std[i]
                positive_pressure_depth = Depth[(np.abs(P-positive_pressure)).argmin()]
                positive_pressure_diff_list.append((positive_pressure_depth - Depth_Sample[i]))
                negative_pressure = P_Sample[i] - P_std[i]
                negative_pressure_depth = Depth[(np.abs(P-negative_pressure)).argmin()]
                negative_pressure_diff_list.append((Depth_Sample[i] - positive_pressure_depth))

            ax1.errorbar(T_Sample, Depth_Sample, yerr = [negative_pressure_diff_list,positive_pressure_diff_list],xerr = [T_std,T_std],
            fmt = 'o',color = '#bd3b24',markersize = 5,label = 'Xenolith Data',
            ecolor = 'k',elinewidth = 0.5,alpha = 0.6, markeredgecolor = 'k')

        ax1.set_ylim(np.amax(Depth),0)

        ax1.set_xlim(0, np.amax(np.concatenate((T,T_Sample))) + 100.0)

    if (moho != None) or (lab != None):

        import matplotlib.patches as patches

    if moho != None:

        if plot_style == 'Depth':
            ax1.axhline(moho, linestyle = '--', color = 'k', label = 'MOHO')
            crust_obj = patches.Rectangle((0,0) ,100000.0, moho, color = '#6cc1c7', alpha = 0.6)
            ax1.add_patch(crust_obj)

        elif plot_style == 'Pressure':
            pressure_equivalent_moho = P[(np.abs((Depth / 1e3)-moho)).argmin()]
            ax1.axhline(pressure_equivalent_moho, linestyle = '--', color = 'k', label = 'MOHO')
            crust_obj = patches.Rectangle((0,0) ,100000.0, pressure_equivalent_moho, color = '#6cc1c7', alpha = 0.6)
            ax1.add_patch(crust_obj)

    if lab != None:

        if plot_style == 'Depth':
            ax1.axhline(lab, linestyle = '--', color = '#9d1414', label = 'LAB')
            mantle_obj = patches.Rectangle((0,lab) ,100000.0, 1e5, color = '#c76c75', alpha = 0.6)
            ax1.add_patch(mantle_obj)

        elif plot_style == 'Pressure':
            pressure_equivalent_lab = P[(np.abs((Depth / 1e3)-lab)).argmin()]
            ax1.axhline(pressure_equivalent_lab, linestyle = '--', color = '#9d1414', label = 'LAB')
            mantle_obj = patches.Rectangle((0,pressure_equivalent_lab) ,100000.0, 1e5, color = '#c76c75', alpha = 0.6)
            ax1.add_patch(mantle_obj)

    ax1.grid()

    if leg == True:
        ax1.legend()

    if plot_type == 'show':
        plt.show()
    elif plot_type == 'save':
        plt.savefig(filename_save, dpi = 300)
