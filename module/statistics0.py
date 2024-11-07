

# region fdr_control_bh

def fdr_control_bh(
    pvalue,
    alpha=0.05,
    method='i',
    ):
    '''
    #-------- Input
    pvalue:     np.ndarray, 2-D or more dimension
    alpha:      Family-wise error rate
    
    #-------- Output
    
    '''
    
    from statsmodels.stats import multitest
    
    fdr_bh = multitest.fdrcorrection(
        pvalue.reshape(-1),
        alpha=alpha,
        method=method,
    )
    
    bh_test_res = fdr_bh[0].reshape(pvalue.shape)
    return(bh_test_res)


'''
# Benjamini/Hochberg procedures

#-------------------------------- check

from statsmodels.stats import multitest
import numpy as np

pvalue = (np.random.uniform(size=(960, 1080))) ** 2


#-------- method 4

bh_test4 = fdr_control_bh(pvalue)


#-------- method 1

fdr_bh = multitest.fdrcorrection(
    pvalue.reshape(-1),
    alpha=0.05,
    method='i',
)
bh_test1 = fdr_bh[0].reshape(pvalue.shape)
(bh_test1 == bh_test4).all()

bh_test1.sum()
(pvalue < 0.05).sum()

#-------- method 2

bh_fdr = 0.05

sortind = np.argsort(pvalue.reshape(-1))
pvals_sorted = np.take(pvalue.reshape(-1), sortind)
rank = np.arange(1, len(pvals_sorted)+1)
bh_critic = rank / len(pvals_sorted) * bh_fdr

where_smaller = np.where(pvals_sorted < bh_critic)

bh_test2 = pvalue <= pvals_sorted[where_smaller[0][-1]]

(bh_test1 == bh_test2).all()


#-------- method 3
import mne
fdr_bh3 = mne.stats.fdr_correction(
    pvalue.reshape(-1), alpha=0.05, method='indep')
bh_test3 = fdr_bh3[0].reshape(pvalue.shape)
(bh_test1 == bh_test3).all()

# others
fdr_bh1 = multitest.multipletests(
    ttest_djf_jja.flatten(),
    method='fdr_bh',)

'''
# endregion


# region check_normality_3d

def check_normality_3d(array):
    '''
    #---- Input
    array: 3-D numpy.ndarray. Normality is checked along the 1st dimension
    
    #---- Output
    
    '''
    
    import numpy as np
    from scipy import stats
    
    whether_normal = np.full(array.shape[1:], True)
    
    for ilat in range(whether_normal.shape[0]):
        for ilon in range(whether_normal.shape[1]):
            # ilat = 48; ilon = 96
            test_data = array[:, ilat, ilon][np.isfinite(array[:, ilat, ilon])]

            if (len(test_data) < 3):
                whether_normal[ilat, ilon] = False
            else:
                whether_normal[ilat, ilon] = stats.shapiro(test_data).pvalue > 0.05
    
    normal_frc = whether_normal.sum() / len(whether_normal.flatten())
    
    return(normal_frc)


'''
#-------------------------------- check
import numpy as np

array1 = np.random.normal(0, 1, size = (60, 96, 108))
check_normality_3d(array1)

array2 = np.random.uniform(0, 1, size = (60, 96, 108))
check_normality_3d(array2)

'''
# endregion


# region check_equal_variance_3d

def check_equal_variance_3d(array1, array2):
    '''
    #-------- Input
    array1/array2: 3-D numpy.ndarray. variance is checked along the 1st dimension
    
    #-------- Output
    
    '''
    
    import numpy as np
    from scipy import stats
    
    variance_equal = np.full(array1.shape[1:], True)
    
    for ilat in range(variance_equal.shape[0]):
        for ilon in range(variance_equal.shape[1]):
            # ilat = 48; ilon = 96
            test_data1 = array1[:, ilat, ilon][np.isfinite(array1[:, ilat, ilon])]
            test_data2 = array2[:, ilat, ilon][np.isfinite(array2[:, ilat, ilon])]
            
            variance_equal[ilat, ilon] = stats.fligner(test_data1, test_data2).pvalue > 0.05

    equal_frc = variance_equal.sum() / len(variance_equal.flatten())
    
    return(equal_frc)


'''
#-------------------------------- check
import numpy as np

array1 = np.random.normal(0, 1, size = (60, 96, 108))
array2 = np.random.normal(0, 2, size = (60, 96, 108))
check_equal_variance_3d(array1, array2)

array1 = np.random.normal(0, 1, size = (60, 96, 108))
array2 = np.random.normal(0, 1, size = (60, 96, 108))
check_equal_variance_3d(array1, array2)
'''
# endregion


# region ttest_fdr_control


def ttest_fdr_control(
    array1, array2,
    axis=0, equal_var=True, nan_policy='omit',alternative='two-sided',
    convert_nan=True,
    fdr_control=True,
    alpha=0.05, method='i',
    ):
    '''
    #-------- Input
    array1/array2: 3-D numpy.ndarray. test is done along the 1st dimension
    
    #-------- Output
    fdr_control=True: bhtest_res
    fdr_control=False: ttest_res
    '''
    
    import numpy as np
    from scipy import stats
    
    ttest_res = stats.ttest_ind(
        array1, array2,
        axis=axis, equal_var=equal_var, nan_policy=nan_policy,
        alternative=alternative,
    ).pvalue
    
    # at any grid point, if all data is np.nan, ttest_res is set to be 1
    ttest_res[np.isnan(array1).all(axis=0)] = 1
    ttest_res[np.isnan(array2).all(axis=0)] = 1
    
    if convert_nan:
        ttest_res[np.isnan(ttest_res)] = 1
    
    if fdr_control:
        bhtest_res = fdr_control_bh(ttest_res, alpha=alpha, method=method)
        return(bhtest_res)
    else:
        return(ttest_res)


'''
#-------------------------------- check
import numpy as np
# axis=0; equal_var=True; nan_policy='omit'; alternative='two-sided';
# convert_nan=True;
array1 = np.random.normal(0, 1, size = (60, 96, 108))
array2 = np.random.normal(0, 1, size = (60, 96, 108))
bhtest_res = ttest_fdr_control(array1, array2,)
bhtest_res.sum() / len(bhtest_res.flatten())

array1 = np.random.normal(0, 1, size = (60, 96, 108))
array2 = np.random.normal(0.5, 1, size = (60, 96, 108))
bhtest_res = ttest_fdr_control(array1, array2,)
bhtest_res.sum() / len(bhtest_res.flatten())

array1 = np.random.normal(0, 1, size = (60, 96, 108))
array2 = np.random.normal(1, 1, size = (60, 96, 108))
bhtest_res = ttest_fdr_control(array1, array2,)
bhtest_res.sum() / len(bhtest_res.flatten())

'''
# endregion


# region cplot_ttest

def cplot_ttest(
    array1, array2, ax, lon_2d, lat_2d
    ):
    '''
    #-------- Input
    array1/array2:  3-D numpy.ndarray. test is done along the 1st dimension
    ax:             ax to plot the data
    lon_2d, lat_2d: 2-D numpy.ndarray, longitude and latitude
    
    #-------- Output
    
    '''
    
    import cartopy.crs as ccrs
    
    ttest_fdr_res = ttest_fdr_control(array1, array2,)
    ax.scatter(
        x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
        s=0.5, c='k', marker='.', edgecolors='none',
        transform=ccrs.PlateCarree(),)

# endregion


# region xr_par_cor

def xr_par_cor(x, y, covar, output = 'r'):
    '''
    output: 'r' correlation coefficient; 'p' p-values.
    '''
    
    import pandas as pd
    import pingouin as pg
    import numpy as np
    
    if ((np.isfinite(x).sum() < 4) | \
        (np.isfinite(y).sum() < 4) | \
            (np.isfinite(covar).sum() < 4)):
        return(np.nan)
    
    dataframe = pd.DataFrame(data={
        'x': x,
        'y': y,
        'covar': covar,
    })
    
    par_corr = pg.partial_corr(data=dataframe, x='x', y='y', covar='covar')
    
    if (output == 'r'):
        return(par_corr.r.values[0])
    elif (output == 'p'):
        return(par_corr["p-val"].values[0])


def xr_par_cor_subset(x, y, covar, subset, output = 'r'):
    '''
    output: 'r' correlation coefficient; 'p' p-values.
    '''
    
    import pandas as pd
    import pingouin as pg
    import numpy as np
    
    if ((np.isfinite(x).sum() < 4) | \
        (np.isfinite(y).sum() < 4) | \
            (np.isfinite(covar).sum() < 4) | \
                (subset.sum() < 4)):
        return(np.nan)
    
    dataframe = pd.DataFrame(data={
        'x': x[subset],
        'y': y[subset],
        'covar': covar[subset],
    })
    
    par_corr = pg.partial_corr(data=dataframe, x='x', y='y', covar='covar')
    
    if (output == 'r'):
        return(par_corr.r.values[0])
    elif (output == 'p'):
        return(par_corr["p-val"].values[0])


'''
import pingouin as pg
ilat = 48
ilon = 96

x = d_ln_alltime[expid[i]]['ann'][:, ilat, ilon]
y = pre_weighted_var[expid[i]][ivar]['ann'][:, ilat, ilon]
covar = pre_weighted_var[expid[i]][control_var]['ann'][:, ilat, ilon]
xr_par_cor(x, y, covar, output = 'p')

dataframe = pd.DataFrame(data={
        'ivar1': x,
        'ivar2': y,
        'ivar3': covar,
    })
pg.partial_corr(
        data=dataframe, x='ivar1', y='ivar2', covar='ivar3')
'''
# endregion


# region xr_regression_y_x1

def xr_regression_y_x1(y, x, output = 'RMSE'):
    '''
    y:      dependent variable
    x:      independent variable
    output: 'RMSE', 'rsquared', 'slope', 'intercept'
    '''
    
    import numpy as np
    import statsmodels.api as sm
    
    subset = np.isfinite(x) & np.isfinite(y)
    x_sub = x[subset]
    y_sub = y[subset]
    
    if (len(x_sub) < 12):
        return(np.nan)
    else:
        ols_fit = sm.OLS(y_sub, sm.add_constant(x_sub), ).fit()

        intercept   = ols_fit.params[0]
        slope       = ols_fit.params[1]
        rsquared    = ols_fit.rsquared

        predicted_y = intercept + slope * x

        RMSE = np.sqrt(np.average(np.square(predicted_y[subset] - y_sub)))

        if (output == 'RMSE'):
            return(RMSE)
        elif (output == 'rsquared'):
            return(rsquared)
        elif (output == 'slope'):
            return(slope)
        elif (output == 'intercept'):
            return(intercept)

# endregion




