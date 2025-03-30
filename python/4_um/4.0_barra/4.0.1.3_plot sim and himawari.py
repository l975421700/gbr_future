

# qsub -I -q express -l walltime=5:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46


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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import pickle
from xmip.preprocessing import rename_cmip6, broadcast_lonlat, correct_lon, promote_empty_dims, replace_x_y_nominal_lat_lon, correct_units, correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds, combined_preprocessing
import xesmf as xe

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
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
import glob

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
    month_jan,
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


# region plot Himawari, ERA5, BARRA-R2, BARRA-C2, am

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['Himawari', 'ERA5 - Himawari', 'BARRA-R2 - Himawari', 'BARRA-C2 - Himawari']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['cll', 'clm', 'clh', 'clt']:
    # var2='cll'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    
    plt_data = {}
    plt_rmse = {}
    
    if var2=='cll':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus'])
    elif var2=='clm':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Altocumulus', 'Altostratus', 'Nimbostratus'])
    elif var2=='clh':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cirrus', 'Cirrostratus', 'Deep convection'])
    elif var2=='clt':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cirrus', 'Cirrostratus', 'Deep convection'])
    himawari_ann = himawari_ann.sum(dim='types').sel(time=slice('2016', '2023'), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    plt_data['Himawari'] = himawari_ann.mean(dim='time')
    plt_mean = plt_data['Himawari'].weighted(np.cos(np.deg2rad(plt_data['Himawari'].lat))).mean().values
    
    era5_ann = regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari'])
    plt_data['ERA5 - Himawari'] = (era5_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['ERA5 - Himawari'] = np.sqrt(np.square(plt_data['ERA5 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, era5_ann)
    plt_data['ERA5 - Himawari'] = plt_data['ERA5 - Himawari'].where(ttest_fdr_res, np.nan)
    
    barra_r2_ann = regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari'])
    plt_data['BARRA-R2 - Himawari'] = (barra_r2_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['BARRA-R2 - Himawari'] = np.sqrt(np.square(plt_data['BARRA-R2 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barra_r2_ann)
    plt_data['BARRA-R2 - Himawari'] = plt_data['BARRA-R2 - Himawari'].where(ttest_fdr_res, np.nan)
    
    barra_c2_ann = regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari'])
    plt_data['BARRA-C2 - Himawari'] = (barra_c2_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['BARRA-C2 - Himawari'] = np.sqrt(np.square(plt_data['BARRA-C2 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barra_c2_ann)
    plt_data['BARRA-C2 - Himawari'] = plt_data['BARRA-C2 - Himawari'].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(plt_data['Himawari'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = '2016-2023 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2016-2023 ' + era5_varlabels[var1]
    extend1 = 'neither'
    extend2 = 'both'
    
    if var2 in ['clh', 'clm', 'cll', 'clt']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
    
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
        axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=8)
    
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
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 himawari vs. barra_c2, and era5 am {var1}.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime




'''
'''
# endregion


# region plot Himawari, ERA5, BARRA-R2, BARRA-C2, am sm

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

# settings
mpl.rc('font', family='Times New Roman', size=10)
plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['Himawari', 'ERA5 - Himawari', 'BARRA-R2 - Himawari', 'BARRA-C2 - Himawari']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['cll', 'clm', 'clh', 'clt']:
    # var2='cll'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    
    plt_data = {}
    for irow in plt_rownames: plt_data[irow] = {}
    plt_mean = {}
    plt_rmse = {}
    for irow in plt_rownames[1:]: plt_rmse[irow] = {}
    
    if var2=='cll':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus'])
    elif var2=='clm':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Altocumulus', 'Altostratus', 'Nimbostratus'])
    elif var2=='clh':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cirrus', 'Cirrostratus', 'Deep convection'])
    elif var2=='clt':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cirrus', 'Cirrostratus', 'Deep convection'])
    himawari_ann = himawari_ann.sum(dim='types').sel(time=slice('2016', '2023'), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    plt_data['Himawari']['Annual mean'] = himawari_ann.mean(dim='time')
    plt_mean['Annual mean'] = plt_data['Himawari']['Annual mean'].weighted(np.cos(np.deg2rad(plt_data['Himawari']['Annual mean'].lat))).mean().values
    
    era5_ann = regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari']['Annual mean'])
    plt_data['ERA5 - Himawari']['Annual mean'] = (era5_ann.mean(dim='time') - plt_data['Himawari']['Annual mean']).compute()
    plt_rmse['ERA5 - Himawari']['Annual mean'] = np.sqrt(np.square(plt_data['ERA5 - Himawari']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - Himawari']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, era5_ann)
    plt_data['ERA5 - Himawari']['Annual mean'] = plt_data['ERA5 - Himawari']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    barra_r2_ann = regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari']['Annual mean'])
    plt_data['BARRA-R2 - Himawari']['Annual mean'] = (barra_r2_ann.mean(dim='time') - plt_data['Himawari']['Annual mean']).compute()
    plt_rmse['BARRA-R2 - Himawari']['Annual mean'] = np.sqrt(np.square(plt_data['BARRA-R2 - Himawari']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - Himawari']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barra_r2_ann)
    plt_data['BARRA-R2 - Himawari']['Annual mean'] = plt_data['BARRA-R2 - Himawari']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    barra_c2_ann = regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari']['Annual mean'])
    plt_data['BARRA-C2 - Himawari']['Annual mean'] = (barra_c2_ann.mean(dim='time') - plt_data['Himawari']['Annual mean']).compute()
    plt_rmse['BARRA-C2 - Himawari']['Annual mean'] = np.sqrt(np.square(plt_data['BARRA-C2 - Himawari']['Annual mean']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - Himawari']['Annual mean'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barra_c2_ann)
    plt_data['BARRA-C2 - Himawari']['Annual mean'] = plt_data['BARRA-C2 - Himawari']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    for jcolnames in plt_colnames[1:]:
        # jcolnames='DJF'
        print(f'#---------------- {jcolnames}')
        
        if var2=='cll':
            himawari_sea = cltype_frequency_alltime['sea'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus'])
        elif var2=='clm':
            himawari_sea = cltype_frequency_alltime['sea'].sel(types=['Altocumulus', 'Altostratus', 'Nimbostratus'])
        elif var2=='clh':
            himawari_sea = cltype_frequency_alltime['sea'].sel(types=['Cirrus', 'Cirrostratus', 'Deep convection'])
        elif var2=='clt':
            himawari_sea = cltype_frequency_alltime['sea'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cirrus', 'Cirrostratus', 'Deep convection'])
        himawari_sea = himawari_sea[himawari_sea.time.dt.season==jcolnames].sum(dim='types').sel(time=slice('2016', '2023'), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
        plt_data['Himawari'][jcolnames] = himawari_sea.mean(dim='time')
        plt_mean[jcolnames] = plt_data['Himawari'][jcolnames].weighted(np.cos(np.deg2rad(plt_data['Himawari'][jcolnames].lat))).mean().values
        
        era5_sea = regrid(era5_sl_mon_alltime['sea'][era5_sl_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari']['Annual mean'])
        plt_data['ERA5 - Himawari'][jcolnames] = (era5_sea.mean(dim='time') - plt_data['Himawari'][jcolnames]).compute()
        plt_rmse['ERA5 - Himawari'][jcolnames] = np.sqrt(np.square(plt_data['ERA5 - Himawari'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['ERA5 - Himawari'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(himawari_sea, era5_sea)
        plt_data['ERA5 - Himawari'][jcolnames] = plt_data['ERA5 - Himawari'][jcolnames].where(ttest_fdr_res, np.nan)
        
        barra_r2_sea = regrid(barra_r2_mon_alltime['sea'][barra_r2_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari']['Annual mean'])
        plt_data['BARRA-R2 - Himawari'][jcolnames] = (barra_r2_sea.mean(dim='time') - plt_data['Himawari'][jcolnames]).compute()
        plt_rmse['BARRA-R2 - Himawari'][jcolnames] = np.sqrt(np.square(plt_data['BARRA-R2 - Himawari'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - Himawari'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(himawari_sea, barra_r2_sea)
        plt_data['BARRA-R2 - Himawari'][jcolnames] = plt_data['BARRA-R2 - Himawari'][jcolnames].where(ttest_fdr_res, np.nan)
        
        barra_c2_sea = regrid(barra_c2_mon_alltime['sea'][barra_c2_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2016', '2023')), ds_out=plt_data['Himawari']['Annual mean'])
        plt_data['BARRA-C2 - Himawari'][jcolnames] = (barra_c2_sea.mean(dim='time') - plt_data['Himawari'][jcolnames]).compute()
        plt_rmse['BARRA-C2 - Himawari'][jcolnames] = np.sqrt(np.square(plt_data['BARRA-C2 - Himawari'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - Himawari'][jcolnames].lat))).mean()).values
        ttest_fdr_res = ttest_fdr_control(himawari_sea, barra_c2_sea)
        plt_data['BARRA-C2 - Himawari'][jcolnames] = plt_data['BARRA-C2 - Himawari'][jcolnames].where(ttest_fdr_res, np.nan)
    
    # print(stats.describe(np.concatenate([plt_data['Himawari'][colname].values for colname in plt_colnames]), axis=None, nan_policy='omit'))
    # print(stats.describe(np.concatenate([plt_data[rowname][colname].values for rowname in plt_rownames[1:] for colname in plt_colnames]), axis=None, nan_policy='omit'))
    
    cbar_label1 = '2016-2023 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2016-2023 ' + era5_varlabels[var1]
    extend1 = 'neither'
    extend2 = 'both'
    
    if var2 in ['clh', 'clm', 'cll', 'clt']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
    
    nrow=len(plt_rownames)
    ncol=len(plt_colnames)
    fm_bottom=1.4/(4*nrow+1.4)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.4]) / 2.54,
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
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.96)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 himawari vs. barra_c2, and era5 am sm {var1}.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime





# endregion


# region plot Himawari vs. ERA5/BARRA-R2/BARRA-C2 mm 2016-2023

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}

mpl.rc('font', family='Times New Roman', size=10)
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4
nrow = 3
ncol = 4
fm_bottom = 1.4 / (panelh*nrow + 1.4)

ids = 'ERA5' #'BARRA-R2' #'BARRA-C2' #

for icltype2 in ['cll', 'clm', 'clh', 'clt']:
    # ['cll', 'clm', 'clh', 'clt']
    # icltype2='cll'
    icltype = cmip6_era5_var[icltype2]
    print(f'#-------------------------------- {icltype} {icltype2}')
    print(cltypes[icltype])
    
    if ids == 'BARRA-C2':
        with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{icltype2}.pkl','rb') as f:
            ds_mon_alltime = pickle.load(f)
    elif ids == 'BARRA-R2':
        with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{icltype2}.pkl','rb') as f:
            ds_mon_alltime = pickle.load(f)
    elif ids == 'ERA5':
        with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{icltype}.pkl', 'rb') as f:
            ds_mon_alltime = pickle.load(f)
    
    opng = f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 himawari vs. {ids} mm {icltype}.png'
    cbar_label = f'2016-2023 {ids} - Himawari {era5_varlabels[icltype]}'
    
    if icltype in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow + 1.4]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow=0; jcol=0
            print(f'#---------------- {irow} {jcol} {month_jan[irow*4+jcol]}')
            axs[irow, jcol] = regional_plot(
                extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
            
            himawari_mon = cltype_frequency_alltime['mon'][cltype_frequency_alltime['mon'].time.dt.month == (irow*4+jcol+1)].sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat), types=cltypes[icltype]).sum(dim='types')
            himawari_mm = himawari_mon.mean(dim='time')
            
            ds_mon = ds_mon_alltime['mon'][ds_mon_alltime['mon'].time.dt.month == (irow*4+jcol+1)].sel(time=slice('2016', '2023')).compute()
            if ((irow==0) & (jcol==0)):
                regridder = xe.Regridder(ds_mon, himawari_mm, method='bilinear')
            ds_mon = regridder(ds_mon)
            ds_mm = ds_mon.mean(dim='time')
            
            plt_data = ds_mm - himawari_mm
            plt_rmse = np.sqrt(np.square(plt_data).weighted(np.cos(np.deg2rad(plt_data.lat))).mean()).values
            ttest_fdr_res = ttest_fdr_control(ds_mon, himawari_mon)
            plt_data = plt_data.where(ttest_fdr_res, np.nan)
            plt_mesh = axs[irow, jcol].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            
            if ((irow==0) & (jcol==0)):
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} RMSE: {np.round(plt_rmse, 1)}'
            else:
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} {np.round(plt_rmse, 1)}'
            
            # plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]}'
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
    fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.98)
    fig.savefig(opng)
    
    del ds_mon_alltime





# endregion


# region plot Himawari vs. BARRA-C2 ann

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}

mpl.rc('font', family='Times New Roman', size=10)
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4

nrow = 3
ncol = 3
fm_bottom = 1.4 / (panelh*nrow + 1.4)

for icltype2 in ['clm', 'clh', 'clt']:
    # ['cll', 'clm', 'clh', 'clt']
    # icltype2='cll'
    icltype = cmip6_era5_var[icltype2]
    print(f'#-------------------------------- {icltype} {icltype2}')
    print(cltypes[icltype])
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{icltype2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    
    opng = f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 himawari vs. barra_c2 ann {icltype}.png'
    cbar_label = f'BARRA-C2 - Himawari {era5_varlabels[icltype]}'
    
    if icltype in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow + 1.4]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow=0; jcol=0
            year = 2016+irow*ncol+jcol
            if year>=2024: continue
            print(f'#---------------- {irow} {jcol} {year}')
            axs[irow, jcol] = regional_plot(
                extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
            
            himawari_ann = cltype_frequency_alltime['ann'].sel(time=slice(str(year), str(year)), lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat), types=cltypes[icltype]).sum(dim='types').squeeze()
            barra_c2_ann = barra_c2_mon_alltime['ann'].sel(time=slice(str(year), str(year))).squeeze()
            if ((irow==0) & (jcol==0)):
                regridder = xe.Regridder(barra_c2_ann, himawari_ann, method='bilinear')
            barra_c2_ann = regridder(barra_c2_ann)
            
            plt_data = (barra_c2_ann - himawari_ann).compute()
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
    
    del barra_c2_mon_alltime



# endregion


# region plot overlapping of am clouds in ERA5, BARRA-R2, and BARRA-C2

mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['ERA5', 'BARRA-R2', 'BARRA-C2']
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent

# import data
era5_sl_mon_alltime = {}
barra_r2_mon_alltime = {}
barra_c2_mon_alltime = {}
for var2 in ['cll', 'clm', 'clh', 'clt']:
    # var2='cll'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)

plt_data = {}
plt_mean = {}
plt_data['ERA5'] = (era5_sl_mon_alltime['lcc']['ann'] + era5_sl_mon_alltime['mcc']['ann'] + era5_sl_mon_alltime['hcc']['ann'] - era5_sl_mon_alltime['tcc']['ann']).sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)).mean(dim='time')
plt_mean['ERA5'] = plt_data['ERA5'].weighted(np.cos(np.deg2rad(plt_data['ERA5'].lat))).mean().values

plt_data['BARRA-R2'] = (barra_r2_mon_alltime['cll']['ann'] + barra_r2_mon_alltime['clm']['ann'] + barra_r2_mon_alltime['clh']['ann'] - barra_r2_mon_alltime['clt']['ann']).sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).mean(dim='time')
plt_mean['BARRA-R2'] = plt_data['BARRA-R2'].weighted(np.cos(np.deg2rad(plt_data['BARRA-R2'].lat))).mean().values

plt_data['BARRA-C2'] = (barra_c2_mon_alltime['cll']['ann'] + barra_c2_mon_alltime['clm']['ann'] + barra_c2_mon_alltime['clh']['ann'] - barra_c2_mon_alltime['clt']['ann']).sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).mean(dim='time')
plt_mean['BARRA-C2'] = plt_data['BARRA-C2'].weighted(np.cos(np.deg2rad(plt_data['BARRA-C2'].lat))).mean().values


cbar_label = r'Difference in 2016-2023 (low+middle+high) and total cloud cover [$\%$]'
pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
    cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
nrow=1
ncol=3
fm_bottom=1.5/(4*nrow+1.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[jcol])
    if jcol==0:
        plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]} Mean: {np.round(plt_mean[plt_colnames[jcol]], 1)}'
    else:
        plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]} {np.round(plt_mean[plt_colnames[jcol]], 1)}'
    axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes)
    plt_mesh = axs[jcol].pcolormesh(
            plt_data[plt_colnames[jcol]].lon,
            plt_data[plt_colnames[jcol]].lat,
            plt_data[plt_colnames[jcol]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)

cbar1 = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
    format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks1, extend='max',
    cax=fig.add_axes([0.25, fm_bottom-0.05, 0.5, 0.05]))
cbar1.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 era5, barra_r2, and barra_c2 am overlap cc.png')



# endregion


# region plot overlapping of mm clouds in ERA5, BARRA-R2, and BARRA-C2


mpl.rc('font', family='Times New Roman', size=10)
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4
nrow = 3
ncol = 4
fm_bottom = 1.4 / (panelh*nrow + 1.4)
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)


for ids in ['ERA5', 'BARRA-R2', 'BARRA-C2']:
    # ids = 'BARRA-C2'
    print(f'#-------------------------------- {ids}')
    
    ds_mon_alltime = {}
    for var2 in ['cll', 'clm', 'clh', 'clt']:
        # var2='cll'
        var1 = cmip6_era5_var[var2]
        print(f'#---------------- {var1} and {var2}')
        
        if ids=='ERA5':
            with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
                ds_mon_alltime[var2] = pickle.load(f)
        elif ids=='BARRA-R2':
            with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
                ds_mon_alltime[var2] = pickle.load(f)
        elif ids=='BARRA-C2':
            with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
                ds_mon_alltime[var2] = pickle.load(f)
    
    opng = f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 {ids} mm overlap cc.png'
    cbar_label = f'Difference in 2016-2023 {ids} (low+middle+high) and total cloud cover ' + r'[$\%$]'
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow + 1.4]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow=0; jcol=0
            print(f'#---------------- {irow} {jcol} {month_jan[irow*4+jcol]}')
            axs[irow, jcol] = regional_plot(
                extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
            
            plt_data = (ds_mon_alltime['cll']['mon'][ds_mon_alltime['cll']['mon'].time.dt.month == (irow*4+jcol+1)].sel(time=slice('2016', '2023')) + ds_mon_alltime['clm']['mon'][ds_mon_alltime['clm']['mon'].time.dt.month == (irow*4+jcol+1)].sel(time=slice('2016', '2023')) + ds_mon_alltime['clh']['mon'][ds_mon_alltime['clh']['mon'].time.dt.month == (irow*4+jcol+1)].sel(time=slice('2016', '2023')) - ds_mon_alltime['clt']['mon'][ds_mon_alltime['clt']['mon'].time.dt.month == (irow*4+jcol+1)].sel(time=slice('2016', '2023'))).mean(dim='time')
            if ids=='ERA5':
                plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)).compute()
            elif ids in ['BARRA-R2', 'BARRA-C2']:
                plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).compute()
            plt_mean = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            
            if ((irow==0) & (jcol==0)):
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} Mean: {str(np.round(plt_mean, 1))}'
            else:
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} {str(np.round(plt_mean, 1))}'
            plt.text(
                0, 1.02, plt_text,
                transform=axs[irow, jcol].transAxes,
                ha='left', va='bottom', rotation='horizontal')
    
    cbar = fig.colorbar(
        plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks, extend='max',
        cax=fig.add_axes([0.25, fm_bottom-0.01, 0.5, 0.02]))
    cbar.ax.set_xlabel(cbar_label)
    fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.98)
    fig.savefig(opng)



# endregion


# region plot Himawari, ERA5, BARRA-R2, BARRA-C2, BARPA-C am

years='2016'
yeare='2021'

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['Himawari', 'ERA5 - Himawari', 'BARRA-R2 - Himawari', 'BARRA-C2 - Himawari', 'BARPA-C - Himawari']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['cll']:
    # var2='cll'
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
    
    if var2=='cll':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus'])
    elif var2=='clm':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Altocumulus', 'Altostratus', 'Nimbostratus'])
    elif var2=='clh':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cirrus', 'Cirrostratus', 'Deep convection'])
    elif var2=='clt':
        himawari_ann = cltype_frequency_alltime['ann'].sel(types=['Cumulus', 'Stratocumulus', 'Stratus', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cirrus', 'Cirrostratus', 'Deep convection'])
    himawari_ann = himawari_ann.sum(dim='types').sel(time=slice(years, yeare), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    plt_data['Himawari'] = himawari_ann.mean(dim='time')
    plt_mean = plt_data['Himawari'].weighted(np.cos(np.deg2rad(plt_data['Himawari'].lat))).mean().values
    
    era5_ann = regrid(era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['Himawari'])
    plt_data['ERA5 - Himawari'] = (era5_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['ERA5 - Himawari'] = np.sqrt(np.square(plt_data['ERA5 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, era5_ann)
    plt_data['ERA5 - Himawari'] = plt_data['ERA5 - Himawari'].where(ttest_fdr_res, np.nan)
    
    barra_r2_ann = regrid(barra_r2_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['Himawari'])
    plt_data['BARRA-R2 - Himawari'] = (barra_r2_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['BARRA-R2 - Himawari'] = np.sqrt(np.square(plt_data['BARRA-R2 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barra_r2_ann)
    plt_data['BARRA-R2 - Himawari'] = plt_data['BARRA-R2 - Himawari'].where(ttest_fdr_res, np.nan)
    
    barra_c2_ann = regrid(barra_c2_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['Himawari'])
    plt_data['BARRA-C2 - Himawari'] = (barra_c2_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['BARRA-C2 - Himawari'] = np.sqrt(np.square(plt_data['BARRA-C2 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barra_c2_ann)
    plt_data['BARRA-C2 - Himawari'] = plt_data['BARRA-C2 - Himawari'].where(ttest_fdr_res, np.nan)
    
    barpa_c_ann = regrid(barpa_c_mon_alltime['ann'].sel(time=slice(years, yeare)), ds_out=plt_data['Himawari'])
    plt_data['BARPA-C - Himawari'] = (barpa_c_ann.mean(dim='time') - plt_data['Himawari']).compute()
    plt_rmse['BARPA-C - Himawari'] = np.sqrt(np.square(plt_data['BARPA-C - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARPA-C - Himawari'].lat))).mean()).values
    ttest_fdr_res = ttest_fdr_control(himawari_ann, barpa_c_ann)
    plt_data['BARPA-C - Himawari'] = plt_data['BARPA-C - Himawari'].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(plt_data['Himawari'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = f'{years}-{yeare} ' + era5_varlabels[var1]
    cbar_label2 = f'Difference in {years}-{yeare} ' + era5_varlabels[var1]
    extend1 = 'neither'
    extend2 = 'both'
    
    if var2 in ['clh', 'clm', 'cll', 'clt']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
    
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
        axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=8)
    
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
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 himawari vs. era5, barra_r2_c2, and barpa_c am {var1}.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime




'''
'''
# endregion

