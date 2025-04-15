

# qsub -I -q express -l walltime=5:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18


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
import glob

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


# region plot CERES, ERA5, BARRA-R2, BARRA-C2, BARPA-R, BARPA-C, BARRA-C2-RAL3.3

year = 2020
season = 'JJA'


# TOA
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc')
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)

# # Surface
# ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc')
# ceres_ebaf = ceres_ebaf.rename({
#     'sfc_sw_down_all_mon': 'msdwswrf',
#     'sfc_sw_up_all_mon': 'msuwswrf',
#     'sfc_lw_down_all_mon': 'msdwlwrf',
#     'sfc_lw_up_all_mon': 'msuwlwrf',
# })
# ceres_ebaf['msuwswrf'] *= (-1)
# ceres_ebaf['msuwlwrf'] *= (-1)


# mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['CERES', 'ERA5 - CERES', 'BARRA-R2 - CERES', 'BARRA-C2 - CERES', 'BARPA-R - CERES', 'BARPA-C - CERES', 'C2-RAL3.3 - CERES']
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
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var2}.pkl','rb') as f:
        barpa_r_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var2}.pkl','rb') as f:
        barpa_c_mon_alltime = pickle.load(f)
    
    barra_c2_ral3p3 = xr.open_mfdataset(sorted(glob.glob(f'data/sim/um/BARRA-C2-RAL3.3/mon/{var2}/*/*')))[var2]
    
    plt_data = {}
    plt_rmse = {}
    
    if season == 'JJA':
        plt_data['CERES'] = ceres_ebaf[var1][(ceres_ebaf[var1].time.dt.year == year) & (ceres_ebaf[var1].time.dt.season == season)].pipe(time_weighted_mean).sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
    plt_mean = plt_data['CERES'].weighted(np.cos(np.deg2rad(plt_data['CERES'].lat))).mean().values
    
    plt_data['ERA5 - CERES'] = regrid(era5_sl_mon_alltime['sea'][(era5_sl_mon_alltime['sea'].time.dt.year == year) & (era5_sl_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['CERES']) - plt_data['CERES']
    plt_rmse['ERA5 - CERES'] = np.sqrt(np.square(plt_data['ERA5 - CERES']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - CERES'].lat))).mean()).values
    
    plt_data['BARRA-R2 - CERES'] = regrid(barra_r2_mon_alltime['sea'][(barra_r2_mon_alltime['sea'].time.dt.year == year) & (barra_r2_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['CERES']) - plt_data['CERES']
    plt_rmse['BARRA-R2 - CERES'] = np.sqrt(np.square(plt_data['BARRA-R2 - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - CERES'].lat))).mean()).values
    
    plt_data['BARRA-C2 - CERES'] = regrid(barra_c2_mon_alltime['sea'][(barra_c2_mon_alltime['sea'].time.dt.year == year) & (barra_c2_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['CERES']) - plt_data['CERES']
    plt_rmse['BARRA-C2 - CERES'] = np.sqrt(np.square(plt_data['BARRA-C2 - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - CERES'].lat))).mean()).values
    
    plt_data['BARPA-R - CERES'] = regrid(barpa_r_mon_alltime['sea'][(barpa_r_mon_alltime['sea'].time.dt.year == year) & (barpa_r_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['CERES']) - plt_data['CERES']
    plt_rmse['BARPA-R - CERES'] = np.sqrt(np.square(plt_data['BARPA-R - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARPA-R - CERES'].lat))).mean()).values
    
    plt_data['BARPA-C - CERES'] = regrid(barpa_c_mon_alltime['sea'][(barpa_c_mon_alltime['sea'].time.dt.year == year) & (barpa_c_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['CERES']) - plt_data['CERES']
    plt_rmse['BARPA-C - CERES'] = np.sqrt(np.square(plt_data['BARPA-C - CERES']).weighted(np.cos(np.deg2rad(plt_data['BARPA-C - CERES'].lat))).mean()).values
    
    plt_data['C2-RAL3.3 - CERES'] = regrid(barra_c2_ral3p3[barra_c2_ral3p3.time.dt.season == season].pipe(time_weighted_mean) * (-1), plt_data['CERES']) - plt_data['CERES']
    plt_rmse['C2-RAL3.3 - CERES'] = np.sqrt(np.square(plt_data['C2-RAL3.3 - CERES']).weighted(np.cos(np.deg2rad(plt_data['C2-RAL3.3 - CERES'].lat))).mean()).values
    
    # for ikey in plt_colnames[1:]:
    #     print(f'{ikey} : {str(np.round(plt_rmse[ikey], 1))}')
    
    print(stats.describe(plt_data['CERES'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = f'{season} {year} ' + era5_varlabels[var1]
    cbar_label2 = f'Difference in {season} {year} ' + era5_varlabels[var1]
    extend1 = 'both'
    extend2 = 'both'
    
    if var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-36, cm_max=36, cm_interval1=3, cm_interval2=6, cmap='BrBG')
    
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
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 ceres vs. era5, barra_r2c2, barpa_rc, barrac2ral3.3 {year} {season}.png')






# endregion


# region plot Himawari, ERA5, BARRA-R2, BARRA-C2, BARPA-R, BARPA-C, BARRA-C2-RAL3.3

year = 2020
season = 'JJA'


with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)
cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}


plt_colnames = ['Himawari', 'ERA5 - Himawari', 'BARRA-R2 - Himawari', 'BARRA-C2 - Himawari', 'BARPA-R - Himawari', 'BARPA-C - Himawari', 'C2-RAL3.3 - Himawari']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]


for var2 in ['cll', 'clm', 'clh', 'clt']:
    # var2='cll'
    # ['cll', 'clm', 'clh', 'clt']
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var2}.pkl','rb') as f:
        barpa_r_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var2}.pkl','rb') as f:
        barpa_c_mon_alltime = pickle.load(f)
    
    barra_c2_ral3p3 = xr.open_mfdataset(sorted(glob.glob(f'data/sim/um/BARRA-C2-RAL3.3/mon/{var2}/*/*')))[var2]
    
    plt_data = {}
    plt_rmse = {}
    
    plt_data['Himawari'] = cltype_frequency_alltime['sea'][(cltype_frequency_alltime['sea'].time.dt.year==year) & (cltype_frequency_alltime['sea'].time.dt.season==season)].squeeze().sel(types=cltypes[var1], lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)).sum(dim='types')
    plt_mean = plt_data['Himawari'].weighted(np.cos(np.deg2rad(plt_data['Himawari'].lat))).mean().values
    
    plt_data['ERA5 - Himawari'] = regrid(era5_sl_mon_alltime['sea'][(era5_sl_mon_alltime['sea'].time.dt.year == year) & (era5_sl_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['Himawari']) - plt_data['Himawari']
    plt_rmse['ERA5 - Himawari'] = np.sqrt(np.square(plt_data['ERA5 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['ERA5 - Himawari'].lat))).mean()).values
    
    plt_data['BARRA-R2 - Himawari'] = regrid(barra_r2_mon_alltime['sea'][(barra_r2_mon_alltime['sea'].time.dt.year == year) & (barra_r2_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['Himawari']) - plt_data['Himawari']
    plt_rmse['BARRA-R2 - Himawari'] = np.sqrt(np.square(plt_data['BARRA-R2 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - Himawari'].lat))).mean()).values
    
    plt_data['BARRA-C2 - Himawari'] = regrid(barra_c2_mon_alltime['sea'][(barra_c2_mon_alltime['sea'].time.dt.year == year) & (barra_c2_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['Himawari']) - plt_data['Himawari']
    plt_rmse['BARRA-C2 - Himawari'] = np.sqrt(np.square(plt_data['BARRA-C2 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - Himawari'].lat))).mean()).values
    
    plt_data['BARPA-R - Himawari'] = regrid(barpa_r_mon_alltime['sea'][(barpa_r_mon_alltime['sea'].time.dt.year == year) & (barpa_r_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['Himawari']) - plt_data['Himawari']
    plt_rmse['BARPA-R - Himawari'] = np.sqrt(np.square(plt_data['BARPA-R - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARPA-R - Himawari'].lat))).mean()).values
    
    plt_data['BARPA-C - Himawari'] = regrid(barpa_c_mon_alltime['sea'][(barpa_c_mon_alltime['sea'].time.dt.year == year) & (barpa_c_mon_alltime['sea'].time.dt.season == season)].squeeze(), plt_data['Himawari']) - plt_data['Himawari']
    plt_rmse['BARPA-C - Himawari'] = np.sqrt(np.square(plt_data['BARPA-C - Himawari']).weighted(np.cos(np.deg2rad(plt_data['BARPA-C - Himawari'].lat))).mean()).values
    
    plt_data['C2-RAL3.3 - Himawari'] = regrid(barra_c2_ral3p3[barra_c2_ral3p3.time.dt.season == season].pipe(time_weighted_mean), plt_data['Himawari']) - plt_data['Himawari']
    plt_rmse['C2-RAL3.3 - Himawari'] = np.sqrt(np.square(plt_data['C2-RAL3.3 - Himawari']).weighted(np.cos(np.deg2rad(plt_data['C2-RAL3.3 - Himawari'].lat))).mean()).values
    
    for ikey in plt_colnames[1:]:
        print(f'{ikey} : {str(np.round(plt_rmse[ikey], 1))}')
    print(stats.describe(plt_data['Himawari'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = f'{season} {year} ' + era5_varlabels[var1]
    cbar_label2 = f'Difference in {season} {year} ' + era5_varlabels[var1]
    extend2 = 'both'
    
    if var2 in ['clt']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'neither'
    elif var2 in ['clh', 'clm', 'cll']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'max'
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
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 himawari vs. era5, barra_r2c2, barpa_rc, barrac2ral3.3 {var1} {year} {season}.png')
    
    del era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime, barpa_r_mon_alltime, barpa_c_mon_alltime, barra_c2_ral3p3




# endregion


