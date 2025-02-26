

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53


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
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import pickle
from xmip.preprocessing import rename_cmip6, broadcast_lonlat, correct_lon, promote_empty_dims, replace_x_y_nominal_lat_lon, correct_units, correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds, combined_preprocessing

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string
import warnings
warnings.filterwarnings('ignore')
import re

# self defined
from mapplot import (
    globe_plot,
    regional_plot,
    ticks_labels,
    scale_bar,
    plot_maxmin_points,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    )

from namelist import (
    month,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    era5_varlabels,
    cmip6_era5_var,
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
)

from calculations import (
    mon_sea_ann,
    regrid,
    cdo_regrid,
    time_weighted_mean)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region import and plot CERES data

ceres_ebaf_toa = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
ceres_ebaf_toa = ceres_ebaf_toa.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf_toa['mtuwswrf'] *= (-1)
ceres_ebaf_toa['mtnlwrf'] *= (-1)

for var in ['mtuwswrf', 'mtnlwrf', 'mtdwswrf']:
    # var='mtuwswrf'
    print(f'#-------------------------------- {var}')
    
    plt_data = ceres_ebaf_toa[var].pipe(time_weighted_mean)
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
ceres_ebaf_toa = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
ceres_ebaf_toa = ceres_ebaf_toa.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf_toa['mtuwswrf'] *= (-1)
ceres_ebaf_toa['mtnlwrf'] *= (-1)

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
    
    plt_data['CERES']['Annual mean'] = ceres_ebaf_toa[var1].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_mean['Annual mean'] = plt_data['CERES']['Annual mean'].weighted(np.cos(np.deg2rad(plt_data['CERES']['Annual mean'].lat))).mean().values
    ceres_ann = ceres_ebaf_toa[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
    
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
        
        plt_data['CERES'][jcolnames] = ceres_ebaf_toa[var1][ceres_ebaf_toa[var1].time.dt.season==jcolnames].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
        plt_mean[jcolnames] = plt_data['CERES'][jcolnames].weighted(np.cos(np.deg2rad(plt_data['CERES'][jcolnames].lat))).mean().values
        ceres_sea = ceres_ebaf_toa[var1].resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][ceres_ebaf_toa[var1].resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
        
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
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. barra_c2, era5, and cmip6 am sm {var1} new.png')
    
    del era5_sl_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens





# endregion


# region plot CERES, ERA5, BARRA-R2, BARRA-C2, historical, amip, am

# import data
ceres_ebaf_toa = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
ceres_ebaf_toa = ceres_ebaf_toa.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf_toa['mtuwswrf'] *= (-1)
ceres_ebaf_toa['mtnlwrf'] *= (-1)

# settings
mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['CERES', 'ERA5 - CERES', 'BARRA-R2 - CERES', 'BARRA-C2 - CERES', r'$historical$ - CERES', r'$amip$ - CERES']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['rsut', 'rlut', 'rsdt']:
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
    
    plt_data['CERES'] = ceres_ebaf_toa[var1].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_mean = plt_data['CERES'].weighted(np.cos(np.deg2rad(plt_data['CERES'].lat))).mean().values
    ceres_ann = ceres_ebaf_toa[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    
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
    
    # plt_mesh1 = axs[0].pcolormesh(
    #         plt_data[plt_colnames[0]].lon,
    #         plt_data[plt_colnames[0]].lat,
    #         plt_data[plt_colnames[0]].values,
    #         norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    # for jcol in range(ncol-1):
    #     plt_mesh2 = axs[jcol+1].pcolormesh(
    #         plt_data[plt_colnames[jcol+1]].lon,
    #         plt_data[plt_colnames[jcol+1]].lat,
    #         plt_data[plt_colnames[jcol+1]].values,
    #         norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),zorder=1)
    plt_mesh1 = axs[0].contourf(
            plt_data[plt_colnames[0]].lon,
            plt_data[plt_colnames[0]].lat,
            plt_data[plt_colnames[0]].values,
            norm=pltnorm1, cmap=pltcmp1, levels=pltlevel1, extend=extend1,
            transform=ccrs.PlateCarree(),zorder=1)
    for jcol in range(ncol-1):
        plt_mesh2 = axs[jcol+1].contourf(
            plt_data[plt_colnames[jcol+1]].lon,
            plt_data[plt_colnames[jcol+1]].lat,
            plt_data[plt_colnames[jcol+1]].values,
            norm=pltnorm2, cmap=pltcmp2, levels=pltlevel2, extend=extend2,
            transform=ccrs.PlateCarree(),zorder=1)
    
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
ceres_ebaf_toa = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2001', '2014'))
ceres_ebaf_toa = ceres_ebaf_toa.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf_toa['mtuwswrf'] *= (-1)
ceres_ebaf_toa['mtnlwrf'] *= (-1)

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
    
    plt_data['CERES'] = ceres_ebaf_toa[var1].pipe(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    plt_mean = plt_data['CERES'].weighted(np.cos(np.deg2rad(plt_data['CERES'].lat))).mean().values
    ceres_ann = ceres_ebaf_toa[var1].resample({'time': '1YE'}).map(time_weighted_mean).pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)).compute()
    
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


