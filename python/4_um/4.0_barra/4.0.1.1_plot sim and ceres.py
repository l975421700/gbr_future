

# qsub -I -q normal -P nf33 -l walltime=5:00:00,ncpus=1,mem=60GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import pickle
from xmip.preprocessing import replace_x_y_nominal_lat_lon
import xesmf as xe

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# management
import os
import sys
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import string
import warnings
warnings.filterwarnings('ignore')
import re

# self defined
from mapplot import (
    globe_plot,
    regional_plot,
    remove_trailing_zero_pos)

from namelist import (
    month_jan,
    era5_varlabels,
    cmip6_era5_var)

from component_plot import (
    plt_mesh_pars)

from calculations import (
    regrid,
    time_weighted_mean)

from statistics0 import (
    ttest_fdr_control)

# endregion


# region import and plot CERES data


# TOA
# ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
# ceres_ebaf = ceres_ebaf.rename({
#     'toa_sw_all_mon': 'mtuwswrf',
#     'toa_lw_all_mon': 'mtnlwrf',
#     'solar_mon': 'mtdwswrf'})
# ceres_ebaf['mtuwswrf'] *= (-1)
# ceres_ebaf['mtnlwrf'] *= (-1)

# Surface
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc').sel(time=slice('2001', '2014'))
ceres_ebaf = ceres_ebaf.rename({
    'sfc_sw_down_all_mon': 'msdwswrf',
    'sfc_sw_up_all_mon': 'msuwswrf',
    'sfc_lw_down_all_mon': 'msdwlwrf',
    'sfc_lw_up_all_mon': 'msuwlwrf',
})
ceres_ebaf['msuwswrf'] *= (-1)
ceres_ebaf['msuwlwrf'] *= (-1)

for var in ['msdwswrf', 'msuwswrf', 'msdwlwrf', 'msuwlwrf']:
    # var='mtuwswrf'
    # ['mtuwswrf', 'mtnlwrf', 'mtdwswrf']
    print(f'#-------------------------------- {var}')
    
    plt_data = ceres_ebaf[var].pipe(time_weighted_mean)
    plt_data_gm = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
    
    cbar_label = 'CERES annual mean (2001-2014) ' + era5_varlabels[var] + '\nglobal mean: ' + str(np.round(plt_data_gm, 2))
    
    if var in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var=='mtdwswrf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msdwswrf', 'msdwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=330, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msuwswrf', 'msuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-220, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-540, cm_max=-120, cm_interval1=10, cm_interval2=40, cmap='viridis')
    
    fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
    
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks, extend='both',
        pad=0.02, fraction=0.13,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
    
    fig.savefig(f'figures/5_era5/5.0_global/5.0.0 global ceres annual mean {var}.png')

# endregion


# region plot CERES, ERA5, BARRA-R2, BARRA-C2, historical, amip, am sm

# import data
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

# settings
plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['CERES', 'ERA5 - CERES', 'BARRA-R2 - CERES', 'BARRA-C2 - CERES', r'$historical$ - CERES', r'$amip$ - CERES']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['rsut', 'rlut', 'rsdt']:
    # var2='rsdt'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/historical_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        historical_regridded_alltime_ens = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/amip_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        amip_regridded_alltime_ens = pickle.load(f)
    
    plt_data = {}
    for irow in plt_rownames: plt_data[irow] = {}
    plt_mean = {}
    plt_rmse = {}
    for irow in plt_rownames[1:]: plt_rmse[irow] = {}
    
    plt_data['CERES']['Annual mean'] = ceres_ebaf[var1].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_mean['Annual mean'] = plt_data['CERES']['Annual mean'].weighted(np.cos(np.deg2rad(plt_data['CERES']['Annual mean'].lat))).mean().values
    ceres_ann = ceres_ebaf[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
    
    plt_data['ERA5 - CERES']['Annual mean'] = (era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES']['Annual mean']).compute()
    plt_rmse['ERA5 - CERES']['Annual mean'] = np.sqrt(np.square(plt_data['ERA5 - CERES']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - CERES']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data['ERA5 - CERES']['Annual mean'] = plt_data['ERA5 - CERES']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-R2 - CERES']['Annual mean'] = (regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES']['Annual mean']) - plt_data['CERES']['Annual mean']).compute()
    plt_rmse['BARRA-R2 - CERES']['Annual mean'] = np.sqrt(np.square(plt_data['BARRA-R2 - CERES']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - CERES']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean), ds_out=plt_data['CERES']['Annual mean'])
        )
    plt_data['BARRA-R2 - CERES']['Annual mean'] = plt_data['BARRA-R2 - CERES']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-C2 - CERES']['Annual mean'] = (regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES']['Annual mean']) - plt_data['CERES']['Annual mean']).compute()
    plt_rmse['BARRA-C2 - CERES']['Annual mean'] = np.sqrt(np.square(plt_data['BARRA-C2 - CERES']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - CERES']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean), ds_out=plt_data['CERES']['Annual mean'])
        )
    plt_data['BARRA-C2 - CERES']['Annual mean'] = plt_data['BARRA-C2 - CERES']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$historical$ - CERES']['Annual mean'] = (historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES']['Annual mean']).compute()
    plt_rmse[r'$historical$ - CERES']['Annual mean'] = np.sqrt(np.square(plt_data[r'$historical$ - CERES']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data[r'$historical$ - CERES']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': '1YE'}).map(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$historical$ - CERES']['Annual mean'] = plt_data[r'$historical$ - CERES']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$amip$ - CERES']['Annual mean'] = (amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES']['Annual mean']).compute()
    plt_rmse[r'$amip$ - CERES']['Annual mean'] = np.sqrt(np.square(plt_data[r'$amip$ - CERES']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data[r'$amip$ - CERES']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': '1YE'}).map(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$amip$ - CERES']['Annual mean'] = plt_data[r'$amip$ - CERES']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    for jcolnames in plt_colnames[1:]:
        # jcolnames='DJF'
        print(f'#---------------- {jcolnames}')
        
        plt_data['CERES'][jcolnames] = ceres_ebaf[var1][ceres_ebaf[var1].time.dt.season==jcolnames].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
        plt_mean[jcolnames] = plt_data['CERES'][jcolnames].weighted(np.cos(np.deg2rad(plt_data['CERES'][jcolnames].lat))).mean().values
        ceres_sea = ceres_ebaf[var1].resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][ceres_ebaf[var1].resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
        
        plt_data['ERA5 - CERES'][jcolnames] = (era5_sl_mon_alltime['mon'][era5_sl_mon_alltime['mon'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES'][jcolnames]).compute()
        plt_rmse['ERA5 - CERES'][jcolnames] = np.sqrt(np.square(plt_data['ERA5 - CERES'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['ERA5 - CERES'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(
            ceres_sea,
            era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
            )
        plt_data['ERA5 - CERES'][jcolnames] = plt_data['ERA5 - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data['BARRA-R2 - CERES'][jcolnames] = (regrid(barra_r2_mon_alltime['mon'][barra_r2_mon_alltime['mon'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES'][jcolnames]) - plt_data['CERES'][jcolnames]).compute()
        plt_rmse['BARRA-R2 - CERES'][jcolnames] = np.sqrt(np.square(plt_data['BARRA-R2 - CERES'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - CERES'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(
            ceres_sea,
            regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames], ds_out=plt_data['CERES'][jcolnames])
            )
        plt_data['BARRA-R2 - CERES'][jcolnames] = plt_data['BARRA-R2 - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data['BARRA-C2 - CERES'][jcolnames] = (regrid(barra_c2_mon_alltime['mon'][barra_c2_mon_alltime['mon'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES'][jcolnames]) - plt_data['CERES'][jcolnames]).compute()
        plt_rmse['BARRA-C2 - CERES'][jcolnames] = np.sqrt(np.square(plt_data['BARRA-C2 - CERES'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - CERES'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(
            ceres_sea,
            regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames], ds_out=plt_data['CERES'][jcolnames])
            )
        plt_data['BARRA-C2 - CERES'][jcolnames] = plt_data['BARRA-C2 - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data[r'$historical$ - CERES'][jcolnames] = (historical_regridded_alltime_ens['mon'][:, historical_regridded_alltime_ens['mon'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES'][jcolnames]).compute()
        plt_rmse[r'$historical$ - CERES'][jcolnames] = np.sqrt(np.square(plt_data[r'$historical$ - CERES'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data[r'$historical$ - CERES'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(
            ceres_sea,
            historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
            )
        plt_data[r'$historical$ - CERES'][jcolnames] = plt_data[r'$historical$ - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data[r'$amip$ - CERES'][jcolnames] = (amip_regridded_alltime_ens['mon'][:, amip_regridded_alltime_ens['mon'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES'][jcolnames]).compute()
        plt_rmse[r'$amip$ - CERES'][jcolnames] = np.sqrt(np.square(plt_data[r'$amip$ - CERES'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data[r'$amip$ - CERES'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(
            ceres_sea,
            amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
            )
        plt_data[r'$amip$ - CERES'][jcolnames] = plt_data[r'$amip$ - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
    
    cbar_label1 = '2001-2014 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2001-2014 ' + era5_varlabels[var1]
    extend1 = 'both'
    extend2 = 'both'
    
    # print(stats.describe(np.concatenate([plt_data['CERES'][colname].values for colname in plt_colnames]), axis=None, nan_policy='omit'))
    # print(stats.describe(np.concatenate([plt_data[rowname][colname].values for rowname in plt_rownames[1:] for colname in plt_colnames]), axis=None, nan_policy='omit'))
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-160, cm_max=-40, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=2, cm_interval2=8, cmap='BrBG')
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-300, cm_max=-200, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=20, cm_interval1=2, cm_interval2=6, cmap='BrBG', asymmetric=True)
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
    
    nrow=len(plt_rownames)
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        axs[irow, 0].text(-0.05, 0.5, plt_rownames[irow], ha='right', va='center', rotation='vertical', transform=axs[irow, 0].transAxes)
        for jcol in range(ncol):
            axs[irow, jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[irow, jcol])
            axs[irow, jcol].text(0, 1.02, f'({string.ascii_lowercase[irow]}{jcol+1})', ha='left', va='bottom', transform=axs[irow, jcol].transAxes,)
            if irow==0:
                axs[0, jcol].text(0.5, 1.14, plt_colnames[jcol], ha='center', va='bottom', transform=axs[0, jcol].transAxes)
    
    for jcol in range(ncol):
        plt_mesh1 = axs[0, jcol].pcolormesh(
            plt_data[plt_rownames[0]][plt_colnames[jcol]].lon,
            plt_data[plt_rownames[0]][plt_colnames[jcol]].lat,
            plt_data[plt_rownames[0]][plt_colnames[jcol]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
        if jcol==0:
            plt_text='Mean: '+str(np.round(plt_mean[plt_colnames[jcol]],1))
        else:
            plt_text = np.round(plt_mean[plt_colnames[jcol]], 1)
        axs[0, jcol].text(
            0.5, 1.02, plt_text,
            ha='center', va='bottom', transform=axs[0, jcol].transAxes)
    
    for irow in range(nrow-1):
        for jcol in range(ncol):
            plt_mesh2 = axs[irow+1, jcol].pcolormesh(
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].lon,
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].lat,
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].values,
                norm=pltnorm2, cmap=pltcmp2,
                transform=ccrs.PlateCarree(),zorder=1)
            if (irow==0)&(jcol==0):
                plt_text='RMSE: '+str(np.round(plt_rmse[plt_rownames[irow+1]][plt_colnames[jcol]], 1))
            else:
                plt_text = np.round(plt_rmse[plt_rownames[irow+1]][plt_colnames[jcol]], 1)
            axs[irow+1, jcol].text(
                0.5, 1.02, plt_text,
                ha='center', va='bottom', transform=axs[irow+1, jcol].transAxes)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.01, 0.4, 0.015]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.01, 0.4, 0.015]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.98)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2, era5, and cmip6 am sm {var1}.png')
    
    del era5_sl_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens





# endregion


# region plot CERES, ERA5, BARRA-R2, BARRA-C2, historical, amip, am


# TOA
# ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
# ceres_ebaf = ceres_ebaf.rename({
#     'toa_sw_all_mon': 'mtuwswrf',
#     'toa_lw_all_mon': 'mtnlwrf',
#     'solar_mon': 'mtdwswrf'})
# ceres_ebaf['mtuwswrf'] *= (-1)
# ceres_ebaf['mtnlwrf'] *= (-1)

# Surface
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc').sel(time=slice('2001', '2014'))
ceres_ebaf = ceres_ebaf.rename({
    'sfc_sw_down_all_mon': 'msdwswrf',
    'sfc_sw_up_all_mon': 'msuwswrf',
    'sfc_lw_down_all_mon': 'msdwlwrf',
    'sfc_lw_up_all_mon': 'msuwlwrf',
})
ceres_ebaf['msuwswrf'] *= (-1)
ceres_ebaf['msuwlwrf'] *= (-1)


# settings
mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['CERES', 'ERA5 - CERES', 'BARRA-R2 - CERES', 'BARRA-C2 - CERES', r'$historical$ - CERES', r'$amip$ - CERES']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['rsus', 'rlus', 'rsds', 'rlds']:
    # var2='rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/historical_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        historical_regridded_alltime_ens = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/amip_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        amip_regridded_alltime_ens = pickle.load(f)
    
    plt_data = {}
    plt_rmse = {}
    
    plt_data['CERES'] = ceres_ebaf[var1].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_mean = plt_data['CERES'].weighted(np.cos(np.deg2rad(plt_data['CERES'].lat))).mean().values
    ceres_ann = ceres_ebaf[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    
    plt_data['ERA5 - CERES'] = (era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES']).compute()
    plt_rmse['ERA5 - CERES'] = np.sqrt(np.square(plt_data['ERA5 - CERES']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data['ERA5 - CERES'] = plt_data['ERA5 - CERES'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-R2 - CERES'] = (regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES']) - plt_data['CERES']).compute()
    plt_rmse['BARRA-R2 - CERES'] = np.sqrt(np.square(plt_data['BARRA-R2 - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean), ds_out=plt_data['CERES'])
        )
    plt_data['BARRA-R2 - CERES'] = plt_data['BARRA-R2 - CERES'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-C2 - CERES'] = (regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES']) - plt_data['CERES']).compute()
    plt_rmse['BARRA-C2 - CERES'] = np.sqrt(np.square(plt_data['BARRA-C2 - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean), ds_out=plt_data['CERES'])
        )
    plt_data['BARRA-C2 - CERES'] = plt_data['BARRA-C2 - CERES'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$historical$ - CERES'] = (historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES']).compute()
    plt_rmse[r'$historical$ - CERES'] = np.sqrt(np.square(plt_data[r'$historical$ - CERES']).weighted(np.cos(np.deg2rad(plt_data[r'$historical$ - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': '1YE'}).map(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$historical$ - CERES'] = plt_data[r'$historical$ - CERES'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$amip$ - CERES'] = (amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - plt_data['CERES']).compute()
    plt_rmse[r'$amip$ - CERES'] = np.sqrt(np.square(plt_data[r'$amip$ - CERES']).weighted(np.cos(np.deg2rad(plt_data[r'$amip$ - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': '1YE'}).map(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$amip$ - CERES'] = plt_data[r'$amip$ - CERES'].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(plt_data['CERES'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = '2001-2014 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2001-2014 ' + era5_varlabels[var1]
    extend1 = 'both'
    extend2 = 'both'
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-36, cm_max=12, cm_interval1=3, cm_interval2=6, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-290, cm_max=-210, cm_interval1=5, cm_interval2=10, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG', asymmetric=True)
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=310, cm_max=410, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG')
    elif var1 in ['msdwswrf', 'msdwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=140, cm_max=270, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msuwswrf', 'msuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-90, cm_max=0, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-480, cm_max=-350, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=300, cm_max=430, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    
    nrow=1
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[jcol])
        if jcol==0:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, Mean: {np.round(plt_mean, 1)}'
        elif jcol==1:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, RMSE: {np.round(plt_rmse[plt_colnames[jcol]], 1)}'
        else:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, {np.round(plt_rmse[plt_colnames[jcol]], 1)}'
        axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=9)
    
    plt_mesh1 = axs[0].pcolormesh(
            plt_data[plt_colnames[0]].lon,
            plt_data[plt_colnames[0]].lat,
            plt_data[plt_colnames[0]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    for jcol in range(ncol-1):
        plt_mesh2 = axs[jcol+1].pcolormesh(
            plt_data[plt_colnames[jcol+1]].lon,
            plt_data[plt_colnames[jcol+1]].lat,
            plt_data[plt_colnames[jcol+1]].values,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),zorder=1)
    # plt_mesh1 = axs[0].contourf(
    #         plt_data[plt_colnames[0]].lon,
    #         plt_data[plt_colnames[0]].lat,
    #         plt_data[plt_colnames[0]].values,
    #         norm=pltnorm1, cmap=pltcmp1, levels=pltlevel1, extend=extend1,
    #         transform=ccrs.PlateCarree(),zorder=1)
    # for jcol in range(ncol-1):
    #     plt_mesh2 = axs[jcol+1].contourf(
    #         plt_data[plt_colnames[jcol+1]].lon,
    #         plt_data[plt_colnames[jcol+1]].lat,
    #         plt_data[plt_colnames[jcol+1]].values,
    #         norm=pltnorm2, cmap=pltcmp2, levels=pltlevel2, extend=extend2,
    #         transform=ccrs.PlateCarree(),zorder=1)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.05]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.05]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2, era5, and cmip6 am {var1}.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens


'''
stats.describe(plt_data[plt_colnames[0]].values, axis=None, nan_policy='omit')
stats.describe(plt_data[plt_colnames[3]].values, axis=None, nan_policy='omit')
'''
# endregion


# region plot CERES, ERA5, BARRA-R2, BARRA-C2, historical, amip, am pct

# import data
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

# settings
mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['CERES', 'ERA5/CERES - 1', 'BARRA-R2/CERES - 1', 'BARRA-C2/CERES - 1', r'$historical$/CERES - 1', r'$amip$/CERES - 1']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['rsut', 'rlut', 'rsdt']:
    # var2='rsdt'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/historical_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        historical_regridded_alltime_ens = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/amip_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        amip_regridded_alltime_ens = pickle.load(f)
    
    plt_data = {}
    plt_rmse = {}
    
    plt_data['CERES'] = ceres_ebaf[var1].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_mean = plt_data['CERES'].weighted(np.cos(np.deg2rad(plt_data['CERES'].lat))).mean().values
    ceres_ann = ceres_ebaf[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    
    plt_data['ERA5/CERES - 1'] = ((era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / plt_data['CERES'] - 1) * 100).compute()
    plt_rmse['ERA5/CERES - 1'] = np.sqrt(np.square(plt_data['ERA5/CERES - 1']).weighted(np.cos(np.deg2rad(plt_data['ERA5/CERES - 1'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data['ERA5/CERES - 1'] = plt_data['ERA5/CERES - 1'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-R2/CERES - 1'] = ((regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES']) / plt_data['CERES'] - 1) * 100).compute()
    plt_rmse['BARRA-R2/CERES - 1'] = np.sqrt(np.square(plt_data['BARRA-R2/CERES - 1']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2/CERES - 1'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean), ds_out=plt_data['CERES'])
        )
    plt_data['BARRA-R2/CERES - 1'] = plt_data['BARRA-R2/CERES - 1'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-C2/CERES - 1'] = ((regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).pipe(time_weighted_mean), ds_out=plt_data['CERES']) / plt_data['CERES'] - 1) * 100).compute()
    plt_rmse['BARRA-C2/CERES - 1'] = np.sqrt(np.square(plt_data['BARRA-C2/CERES - 1']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2/CERES - 1'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': '1YE'}).map(time_weighted_mean), ds_out=plt_data['CERES'])
        )
    plt_data['BARRA-C2/CERES - 1'] = plt_data['BARRA-C2/CERES - 1'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$historical$/CERES - 1'] = ((historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / plt_data['CERES'] - 1) * 100).compute()
    plt_rmse[r'$historical$/CERES - 1'] = np.sqrt(np.square(plt_data[r'$historical$/CERES - 1']).weighted(np.cos(np.deg2rad(plt_data[r'$historical$/CERES - 1'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': '1YE'}).map(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$historical$/CERES - 1'] = plt_data[r'$historical$/CERES - 1'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$amip$/CERES - 1'] = ((amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').pipe(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / plt_data['CERES'] - 1) * 100).compute()
    plt_rmse[r'$amip$/CERES - 1'] = np.sqrt(np.square(plt_data[r'$amip$/CERES - 1']).weighted(np.cos(np.deg2rad(plt_data[r'$amip$/CERES - 1'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(
        ceres_ann,
        amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': '1YE'}).map(time_weighted_mean).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$amip$/CERES - 1'] = plt_data[r'$amip$/CERES - 1'].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(plt_data['CERES'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = '2001-2014 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2001-2014 ' + re.sub(r'\[.*?\]', r'[$\%$]', era5_varlabels[var1])
    extend1 = 'both'
    extend2 = 'both'
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-15, cm_max=45, cm_interval1=5, cm_interval2=5, cmap='BrBG_r', asymmetric=True)
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-290, cm_max=-210, cm_interval1=5, cm_interval2=10, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG_r')
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=310, cm_max=410, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.3, cm_max=0.3, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG')
    
    nrow=1
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[jcol])
        if jcol==0:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, Mean: {np.round(plt_mean, 1)}'
        elif jcol==1:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, RMSE: {np.round(plt_rmse[plt_colnames[jcol]], 1)}'
        else:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, {np.round(plt_rmse[plt_colnames[jcol]], 1)}'
        axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=9)
    
    plt_mesh1 = axs[0].pcolormesh(
            plt_data[plt_colnames[0]].lon,
            plt_data[plt_colnames[0]].lat,
            plt_data[plt_colnames[0]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    for jcol in range(ncol-1):
        plt_mesh2 = axs[jcol+1].pcolormesh(
            plt_data[plt_colnames[jcol+1]].lon,
            plt_data[plt_colnames[jcol+1]].lat,
            plt_data[plt_colnames[jcol+1]].values,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),zorder=1)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.05]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.05]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2, era5, and cmip6 am {var1} pct.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens


'''
stats.describe(plt_data[plt_colnames[0]].values, axis=None, nan_policy='omit')
stats.describe(plt_data[plt_colnames[3]].values, axis=None, nan_policy='omit')
'''
# endregion


# region plot CERES and BARRA-C2 ann

# import data
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc')
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

mpl.rc('font', family='Times New Roman', size=10)
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4

nrow = 4
ncol = 6
fm_bottom = 1.4 / (panelh*nrow + 1.4)

for var2 in ['rsut']:
    # ['rsut', 'rlut', 'rsdt']
    # var2='rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    
    opng=f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2 ann {var1}.png'
    cbar_label = f'BARRA-C2 - CERES {era5_varlabels[var1]}'
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=4, cm_interval2=8, cmap='BrBG')
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow + 1.4]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow=1; jcol=2
            year = 2001+irow*ncol+jcol
            if year>=2024: continue
            print(f'#---------------- {irow} {jcol} {year}')
            axs[irow, jcol] = regional_plot(
                extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
            
            ceres_ann = ceres_ebaf[var1].sel(time=slice(str(year), str(year))).pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
            barra_c2_ann = barra_c2_mon_alltime['ann'].sel(time=slice(str(year), str(year))).squeeze()
            if ((irow==0) & (jcol==0)):
                regridder = xe.Regridder(barra_c2_ann, ceres_ann, method='bilinear')
            barra_c2_ann = regridder(barra_c2_ann)
            
            plt_data = (barra_c2_ann - ceres_ann).compute()
            plt_rmse = np.sqrt(np.square(plt_data).weighted(np.cos(np.deg2rad(plt_data.lat))).mean()).values
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            
            if ((irow==0) & (jcol==0)):
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {year} RMSE: {np.round(plt_rmse, 1)}'
            else:
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {year} {np.round(plt_rmse, 1)}'
            
            axs[irow, jcol].text(
                0, 1.02, plt_text,
                transform=axs[irow, jcol].transAxes, fontsize=10,
                ha='left', va='bottom', rotation='horizontal')
    
    axs[nrow-1, ncol-1].set_visible(False)
    
    cbar = fig.colorbar(
        plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks, extend='both',
        cax=fig.add_axes([0.25, fm_bottom-0.01, 0.5, 0.02]))
    cbar.ax.set_xlabel(cbar_label)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=fm_bottom, top=0.98)
    fig.savefig(opng)



# endregion


# region plot CERES and BARRA-C2 mon

year = 2020
var2 = 'rsut'
var1 = cmip6_era5_var[var2]

# import data
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice(str(year), str(year)))
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
    barra_c2_mon_alltime = pickle.load(f)

if var1 in ['mtuwswrf', 'mtuwswrfcs']:
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-40, cm_max=40, cm_interval1=2, cm_interval2=8, cmap='BrBG')

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
cbar_label = f'{year} BARRA-C2 - CERES {era5_varlabels[var1]}'
nrow=3
ncol=4
fm_bottom=1.5/(4*nrow+1.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[irow, jcol])
        
        ceres_mon = ceres_ebaf[var1][irow*4+jcol].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
        barra_c2_mon = regrid(barra_c2_mon_alltime['mon'].sel(time=slice(str(year), str(year)))[irow*4+jcol], ds_out=ceres_mon)
        plt_data = (barra_c2_mon - ceres_mon).compute()
        plt_rmse = np.sqrt(np.square(plt_data).weighted(np.cos(np.deg2rad(plt_data.lat))).mean()).values
        
        if ((irow==0) & (jcol==0)):
            plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} RMSE: {np.round(plt_rmse, 1)}'
        else:
            plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} {np.round(plt_rmse, 1)}'
        
        axs[irow, jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[irow, jcol].transAxes)
        plt_mesh = axs[irow, jcol].pcolormesh(
            plt_data.lon, plt_data.lat, plt_data.values,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),zorder=1)

cbar = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
    format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='both',
    cax=fig.add_axes([0.25, fm_bottom-0.01, 0.5, 0.02]))
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right=0.99, bottom=fm_bottom, top=0.98)
fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2 {year} mon.png')


# endregion


# region plot CERES and BARRA-C2 mm

# import data
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc')
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

mpl.rc('font', family='Times New Roman', size=10)
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4

nrow = 3
ncol = 4
fm_bottom = 1.4 / (panelh*nrow + 1.4)

for var2 in ['rsut']:
    # ['rsut', 'rlut', 'rsdt']
    # var2='rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    
    opng=f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2 mm {var1}.png'
    cbar_label = f'2016-2023 BARRA-C2 - CERES {era5_varlabels[var1]}'
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=4, cm_interval2=8, cmap='BrBG')
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow + 1.4]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow=1; jcol=2
            print(f'#---------------- {irow} {jcol} {month_jan[irow*4+jcol]}')
            axs[irow, jcol] = regional_plot(
                extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
            
            ceres_mon = ceres_ebaf[var1][ceres_ebaf[var1].time.dt.month==(irow*4+jcol+1)].sel(time=slice('2016', '2023')).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
            ceres_mm = ceres_mon.mean(dim='time')
            barra_c2_mon = barra_c2_mon_alltime['mon'][barra_c2_mon_alltime['mon'].time.dt.month==(irow*4+jcol+1)].sel(time=slice('2016', '2023')).compute()
            if ((irow==0) & (jcol==0)):
                regridder = xe.Regridder(barra_c2_mon, ceres_mm, method='bilinear')
            barra_c2_mon = regridder(barra_c2_mon)
            barra_c2_mm = barra_c2_mon.mean(dim='time')
            
            plt_data = (barra_c2_mm - ceres_mm).compute()
            plt_rmse = np.sqrt(np.square(plt_data).weighted(np.cos(np.deg2rad(plt_data.lat))).mean()).values
            
            ttest_fdr_res = ttest_fdr_control(barra_c2_mon, ceres_mon)
            plt_data = plt_data.where(ttest_fdr_res, np.nan)
            plt_mesh = axs[irow, jcol].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            
            if ((irow==0) & (jcol==0)):
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} RMSE: {np.round(plt_rmse, 1)}'
            else:
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} {np.round(plt_rmse, 1)}'
            
            plt.text(
                0, 1.02, plt_text,
                transform=axs[irow, jcol].transAxes, fontsize=10,
                ha='left', va='bottom', rotation='horizontal')
    
    cbar = fig.colorbar(
        plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks, extend='both',
        cax=fig.add_axes([0.25, fm_bottom-0.01, 0.5, 0.02]))
    cbar.ax.set_xlabel(cbar_label)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=fm_bottom, top=0.98)
    fig.savefig(opng)
    
    del barra_c2_mon_alltime


# endregion


# region plot CERES, ERA5, BARRA-R2, BARRA-C2, BARPA-C, am

years = '2016'
yeare = '2021'

# TOA
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice(years, yeare))
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

# # Surface
# ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc').sel(time=slice(years, yeare))
# ceres_ebaf = ceres_ebaf.rename({
#     'sfc_sw_down_all_mon': 'msdwswrf',
#     'sfc_sw_up_all_mon': 'msuwswrf',
#     'sfc_lw_down_all_mon': 'msdwlwrf',
#     'sfc_lw_up_all_mon': 'msuwlwrf',
# })
# ceres_ebaf['msuwswrf'] *= (-1)
# ceres_ebaf['msuwlwrf'] *= (-1)


# settings
mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['CERES', 'ERA5 - CERES', 'BARRA-R2 - CERES', 'BARRA-C2 - CERES', 'BARPA-C - CERES']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['rsut']:
    # var2='rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var2}.pkl','rb') as f:
        barpa_c_mon_alltime = pickle.load(f)
    
    plt_data = {}
    plt_rmse = {}
    
    ceres_ann = ceres_ebaf[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_data['CERES'] = ceres_ann.mean(dim='time')
    plt_mean = plt_data['CERES'].weighted(np.cos(np.deg2rad(plt_data['CERES'].lat))).mean().values
    
    era5_ann = regrid(era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['CERES'])
    plt_data['ERA5 - CERES'] = (era5_ann.mean(dim='time') - plt_data['CERES']).compute()
    plt_rmse['ERA5 - CERES'] = np.sqrt(np.square(plt_data['ERA5 - CERES']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(ceres_ann, era5_ann)
    plt_data['ERA5 - CERES'] = plt_data['ERA5 - CERES'].where(ttest_fdr_res, np.nan)
    
    barra_r2_ann = regrid(barra_r2_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['CERES'])
    plt_data['BARRA-R2 - CERES'] = (barra_r2_ann.mean(dim='time') - plt_data['CERES']).compute()
    plt_rmse['BARRA-R2 - CERES'] = np.sqrt(np.square(plt_data['BARRA-R2 - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(ceres_ann, barra_r2_ann)
    plt_data['BARRA-R2 - CERES'] = plt_data['BARRA-R2 - CERES'].where(ttest_fdr_res, np.nan)
    
    barra_c2_ann = regrid(barra_c2_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['CERES'])
    plt_data['BARRA-C2 - CERES'] = (barra_c2_ann.mean(dim='time') - plt_data['CERES']).compute()
    plt_rmse['BARRA-C2 - CERES'] = np.sqrt(np.square(plt_data['BARRA-C2 - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(ceres_ann, barra_c2_ann)
    plt_data['BARRA-C2 - CERES'] = plt_data['BARRA-C2 - CERES'].where(ttest_fdr_res, np.nan)
    
    barpa_c_ann = regrid(barpa_c_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['CERES'])
    plt_data['BARPA-C - CERES'] = (barpa_c_ann.mean(dim='time') - plt_data['CERES']).compute()
    plt_rmse['BARPA-C - CERES'] = np.sqrt(np.square(plt_data['BARPA-C - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARPA-C - CERES'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(ceres_ann, barpa_c_ann)
    plt_data['BARPA-C - CERES'] = plt_data['BARPA-C - CERES'].where(ttest_fdr_res, np.nan)
    
    
    print(stats.describe(plt_data['CERES'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = f'{years}-{yeare} ' + era5_varlabels[var1]
    cbar_label2 = f'Difference in {years}-{yeare} ' + era5_varlabels[var1]
    extend1 = 'both'
    extend2 = 'both'
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-36, cm_max=12, cm_interval1=3, cm_interval2=6, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-290, cm_max=-210, cm_interval1=5, cm_interval2=10, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG', asymmetric=True)
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=310, cm_max=410, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG')
    elif var1 in ['msdwswrf', 'msdwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=140, cm_max=270, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msuwswrf', 'msuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-90, cm_max=0, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-480, cm_max=-350, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=300, cm_max=430, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG')
    
    nrow=1
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[jcol])
        if jcol==0:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, Mean: {np.round(plt_mean, 1)}'
        elif jcol==1:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, RMSE: {np.round(plt_rmse[plt_colnames[jcol]], 1)}'
        else:
            plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}, {np.round(plt_rmse[plt_colnames[jcol]], 1)}'
        axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=9)
    
    plt_mesh1 = axs[0].pcolormesh(
            plt_data[plt_colnames[0]].lon,
            plt_data[plt_colnames[0]].lat,
            plt_data[plt_colnames[0]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    for jcol in range(ncol-1):
        plt_mesh2 = axs[jcol+1].pcolormesh(
            plt_data[plt_colnames[jcol+1]].lon,
            plt_data[plt_colnames[jcol+1]].lat,
            plt_data[plt_colnames[jcol+1]].values,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),zorder=1)
    # plt_mesh1 = axs[0].contourf(
    #         plt_data[plt_colnames[0]].lon,
    #         plt_data[plt_colnames[0]].lat,
    #         plt_data[plt_colnames[0]].values,
    #         norm=pltnorm1, cmap=pltcmp1, levels=pltlevel1, extend=extend1,
    #         transform=ccrs.PlateCarree(),zorder=1)
    # for jcol in range(ncol-1):
    #     plt_mesh2 = axs[jcol+1].contourf(
    #         plt_data[plt_colnames[jcol+1]].lon,
    #         plt_data[plt_colnames[jcol+1]].lat,
    #         plt_data[plt_colnames[jcol+1]].values,
    #         norm=pltnorm2, cmap=pltcmp2, levels=pltlevel2, extend=extend2,
    #         transform=ccrs.PlateCarree(),zorder=1)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.05]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.05]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. era5, barra_r2c2, and barpa_c am {var1}.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime, barpa_c_mon_alltime


'''
stats.describe(plt_data[plt_colnames[0]].values, axis=None, nan_policy='omit')
stats.describe(plt_data[plt_colnames[3]].values, axis=None, nan_policy='omit')
'''
# endregion


# region plot global CERES vs. ERA5

years = '2001'
yeare = '2023'

for var in ['mtuwswrf', 'mtnlwrf', 'mtdwswrf', 'msdwswrf', 'msuwswrf', 'msdwlwrf', 'msuwlwrf', 'msnswrf', 'msnlwrf', 'toa_albedo']:
    # var = 'mtuwswrf'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/obs/CERES/ceres_mon_alltime_{var}.pkl', 'rb') as f:
        ceres_mon_alltime = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    # print(ceres_mon_alltime['ann'].shape)
    # print(era5_sl_mon_alltime['ann'].shape)
    
    if var in ['mtuwswrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['mtnlwrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['mtdwswrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msdwswrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=330, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msuwswrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-220, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msdwlwrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msuwlwrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-540, cm_max=-120, cm_interval1=10, cm_interval2=40, cmap='viridis')
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msnswrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msnlwrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-170, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['toa_albedo']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis')
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG')
    
    opng = f'figures/5_era5/5.1_era5_obs/5.1.1_ceres vs. era5 {var} {years}_{yeare}.png'
    cbar_label1 = f'{years}-{yeare} {era5_varlabels[var]}'
    cbar_label2 = f'Difference in {years}-{yeare} {era5_varlabels[var]}'
    plt_colnames = ['CERES', 'ERA5', 'ERA5 - CERES']
    
    nrow = 1
    ncol = 3
    fm_bottom = 1.5 / (4.4*nrow + 2)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
        subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = globe_plot(ax_org=axs[jcol])
        axs[jcol].text(
            0.5, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
            ha='center', va='bottom', transform=axs[jcol].transAxes)
    
    plt_mesh = axs[0].pcolormesh(
        ceres_mon_alltime['ann'].lon, ceres_mon_alltime['ann'].lat,
        ceres_mon_alltime['ann'].mean(dim='time'),
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    plt_mesh = axs[1].pcolormesh(
        era5_sl_mon_alltime['ann'].lon, era5_sl_mon_alltime['ann'].lat,
        era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)).mean(dim='time'),
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    
    if not 'regridder' in globals():
        regridder = xe.Regridder(era5_sl_mon_alltime['am'], ceres_mon_alltime['am'], 'bilinear')
    era5_ann_regrid = regridder(era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)))
    plt_data = era5_ann_regrid.mean(dim='time') - ceres_mon_alltime['ann'].mean(dim='time')
    ttest_fdr_res = ttest_fdr_control(era5_ann_regrid, ceres_mon_alltime['ann'])
    plt_data = plt_data.where(ttest_fdr_res, np.nan)
    plt_mesh2 = axs[2].pcolormesh(
        plt_data.lon, plt_data.lat, plt_data,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(
        plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
        ax=axs, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks, extend='both',
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
    cbar.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        ax=axs, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend='both',
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.01, right=0.99, bottom=fm_bottom, top=0.92)
    fig.savefig(opng)
    del ceres_mon_alltime, era5_sl_mon_alltime




# endregion



