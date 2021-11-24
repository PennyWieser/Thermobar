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

def std_dev(x, x2):
    N=len(x)
    sum_squares=np.sum((x-x2)**2)
    RMSE=np.sqrt((1/N)*sum_squares)
    return RMSE

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

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


def calculate_R2(x, y):
    masknan=(~np.isnan(x) & ~np.isnan(y))
    regx=x[masknan].values.reshape(-1, 1)
    regy=y[masknan].values.reshape(-1, 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(regx[:,0],regy[:,0])
    p_value=np.round(p_value, 3)
    lr=LinearRegression()
    lr.fit(regx,regy)
    Y_pred=lr.predict(regx)
    Int=lr.intercept_
    Grad=lr.coef_
    #R="R\N{SUPERSCRIPT TWO} = " +  str(np.round(r2_score(regy, Y_pred), 2))
    R=(np.round(r2_score(regy, Y_pred), 2))
    RMSE=std_dev(regx, regy)
    RMSEp=np.round(RMSE, 2)
    Median=np.nanmedian(regy-regx)#
    Medianp=np.round(Median, 2)
    Mean=np.nanmean(regy-regx)
    Meanp=np.round(Mean, 2)


    return {'R2': '{0:.2f}'.format(R), 'RMSE':'{0:.2f}'.format(RMSEp), 'RMSE_num':RMSEp,
    'P_val':'{0:.3f}'.format(p_value), 'Median':'{0:.2f}'.format(Medianp), 'Mean':'{0:.2f}'.format(Meanp),
'x_pred': regx, 'y_pred': Y_pred, 'Int': Int, 'Grad':Grad[0]}

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
    RMSE=std_dev(regx, regy)
    RMSEp=np.round(RMSE, 2)
    Median=np.nanmedian(regy-regx)#
    Medianp=np.round(Median, 2)
    Mean=np.nanmean(regy-regx)
    Meanp=np.round(Mean, 2)


    return {'R$^{2}$': '{0:.2f}'.format(R), 'RMSE':'{0:.2f}'.format(RMSEp), 'RMSE_num':RMSEp, 'Grad':Grad, 'Int': Int,
    'Median Error':'{0:.2f}'.format(Medianp), 'Mean Error':'{0:.2f}'.format(Meanp), 'p value':'{0:.3f}'.format(p_value),
     }

def calculate_R2_np(x, y):
    if len(x)!=len(y):
        raise TypeError('X and y not same length')

    masknan=(~np.isnan(x) & ~np.isnan(y))

    regx=x[masknan].reshape(-1, 1)
    regy=y[masknan].reshape(-1, 1)
    lr=LinearRegression()
    lr.fit(regx,regy)
    Y_pred=lr.predict(regx)
    R='R2' +  str(np.round(r2_score(regy, Y_pred), 2))
    RMSE=std_dev(regx, regy)
    RMSEp='RMSE= ' +  str(np.round(RMSE, 5))
    Median=np.nanmedian(regy-regx)#
    Medianp='Median Off= ' +  str(np.round(Median, 5))
    Mean=np.nanmean(regy-regx)
    Meanp='Mean Off= ' +  str(np.round(Mean, 5))
    Intercept=lr.intercept_
    grad=lr.coef_

    return {'R2': R, 'RMSE':RMSEp, 'Median':Medianp, 'Mean':Meanp, 'grad': grad, 'int': Intercept, 'x_pred': regx, 'y_pred': Y_pred}


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



