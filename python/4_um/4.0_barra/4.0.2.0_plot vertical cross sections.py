

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60+gdata/py18


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
from datetime import datetime
from skimage.measure import block_reduce
from netCDF4 import Dataset
from haversine import haversine

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
from PIL import Image
from matplotlib.colors import ListedColormap

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
import glob
import argparse
import calendar
from pathlib import Path
import time

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
    month_num,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    ds_color,
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
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

from metplot import get_cross_section

# endregion


# region plot cross sections in ERA5, BARRA-R2/C2 through A1-A2 and A3-A4

# options
vars = ['hur'] # ['hur', 'hus', 'ta', 'ua', 'va', 'wap']
ds_names = ['BARRA-R2', 'BARRA-C2']
plt_modes = ['original', 'difference'] # ['original', 'difference']
years = '2016'; yeare = '2023'

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
cs_start = [min_lat, min_lon]; cs_end = [max_lat, max_lon]
cs_names = 'A1'; cs_namee = 'A2'
steps = 200
ptop = 600
nrow = 1
ncol = len(ds_names)
fm_bottom = 3 / (5*nrow+3)


for var2 in vars:
    # var2='wap' # ['hur', 'hus', 'ta', 'ua', 'va', 'wap']
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var2}')
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids == 'ERA5':
            # ids = 'ERA5'
            with open(f'data/sim/era5/mon/era5_pl_mon_alltime_{var1}.pkl', 'rb') as f:
                era5_pl_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = get_cross_section(era5_pl_mon_alltime['ann'], cs_start, cs_end, steps=steps)
        elif ids == 'BARRA-R2':
            # ids = 'BARRA-R2'
            with open(f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var2}.pkl','rb') as f:
                barra_r2_pl_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = get_cross_section(barra_r2_pl_mon_alltime['ann'], cs_start, cs_end, steps=steps)
        elif ids == 'BARRA-C2':
            # ids = 'BARRA-C2'
            with open(f'data/sim/um/barra_c2/barra_c2_pl_mon_alltime_{var2}.pkl','rb') as f:
                barra_c2_pl_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = get_cross_section(barra_c2_pl_mon_alltime['ann'], cs_start, cs_end, steps=steps)
        
        ds_data['ann'][ids] = ds_data['ann'][ids].sel(y=slice(ptop, 1000), time=slice(years, yeare)).transpose('time', 'y', 'x')
        ds_data['am'][ids] = ds_data['ann'][ids].mean(dim='time').compute()
    
    extend2 = 'both'
    if var1 == 'q':
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
        extend = 'max'
        pltlevel2 = np.array([-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5])
        pltticks2 = np.array([-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1 == 't':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-48, cm_max=32, cm_interval1=4, cm_interval2=8, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG', asymmetric=True)
    elif var1 == 'w':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-0.1, cm_max=0.1, cm_interval1=0.01, cm_interval2=0.02, cmap='PuOr')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.05, cm_max=0.05, cm_interval1=0.01, cm_interval2=0.01, cmap='BrBG')
    elif var1 == 'u':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-8, cm_max=32, cm_interval1=2, cm_interval2=4, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG')
    elif var1 == 'v':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=2, cmap='PuOr')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG')
    elif var1 == 'z':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=12000, cm_interval1=500, cm_interval2=2000, cmap='viridis_r')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1000, cm_max=1000, cm_interval1=200, cm_interval2=400, cmap='BrBG')
    elif var1 == 'r':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=2, cmap='BrBG_r')
    else:
        print('Warning: unspecified colorbar')
    
    for plt_mode in plt_modes:
        # plt_mode = 'difference' #'original' #
        print(f'#---------------- {plt_mode}')
        
        fig, axs = plt.subplots(nrow, ncol, figsize=np.array([6.6*ncol, 5*nrow+3]) / 2.54, sharey=True, gridspec_kw={'hspace': 0.01, 'wspace': 0.05})
        
        plt_mesh = axs[0].pcolormesh(
            ds_data['am'][ds_names[0]].x,
            ds_data['am'][ds_names[0]].y,
            ds_data['am'][ds_names[0]],
            norm=pltnorm, cmap=pltcmp)
        
        plt_colnames = [ds_names[0]]
        if plt_mode == 'original':
            plt_colnames += [f'{ids}' for ids in ds_names[1:]]
            for jcol in range(ncol-1):
                axs[jcol+1].pcolormesh(
                    ds_data['am'][ds_names[jcol+1]].x,
                    ds_data['am'][ds_names[jcol+1]].y,
                    ds_data['am'][ds_names[jcol+1]],
                    norm=pltnorm, cmap=pltcmp)
        elif plt_mode == 'difference':
            plt_colnames += [f'{ids} - {ds_names[0]}' for ids in ds_names[1:]]
            for jcol in range(ncol-1):
                common_y = np.intersect1d(ds_data['am'][ds_names[jcol+1]].y, ds_data['am'][ds_names[0]].y)
                plt_data = ds_data['am'][ds_names[jcol+1]].sel(y=common_y) - ds_data['am'][ds_names[0]].sel(y=common_y)
                if not var2 in ['hus', 'ta', 'wap']:
                    ttest_fdr_res = ttest_fdr_control(
                        ds_data['ann'][ds_names[jcol+1]].sel(y=common_y),
                        ds_data['ann'][ds_names[0]].sel(y=common_y))
                    plt_data = plt_data.where(ttest_fdr_res, np.nan)
                plt_mesh2 = axs[jcol+1].pcolormesh(
                    plt_data.x, plt_data.y, plt_data,
                    norm=pltnorm2, cmap=pltcmp2)
        
        for jcol in range(ncol):
            axs[jcol].invert_yaxis()
            axs[jcol].set_ylim(1000, ptop)
            # axs[jcol].set_yticks(np.arange(1000, ptop - 1e-4, -200))
            
            axs[jcol].set_xlim(0, ds_data['am'][ds_names[0]].x[-1])
            axs[jcol].set_xticks(
                np.arange(0, ds_data['am'][ds_names[0]].x[-1] + 1e-4, 1000),
                labels=(np.arange(0,ds_data['am'][ds_names[0]].x[-1]+1e-4,1000)/1000).astype(int))
            
            axs[jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
            axs[jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
            axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--')
            
            axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        axs[0].set_ylabel(r'Pressure [$hPa$]')
        axs[1].set_xlabel(f'Distance from {cs_names} to {cs_namee} ' + r'[$10^3 \; km$]')
        
        if plt_mode == 'original':
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.25, fm_bottom-0.2, 0.5, 0.04]))
            cbar.ax.set_xlabel(f'{years}-{yeare} {era5_varlabels[var1]}')
        elif plt_mode == 'difference':
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.2, 0.4, 0.04]))
            cbar.ax.set_xlabel(f'{years}-{yeare} {era5_varlabels[var1]}')
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.2, 0.4, 0.04]))
            cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1]}')
        
        fig.subplots_adjust(left=0.08, right=0.99, bottom=fm_bottom, top=0.92)
        fig.savefig(f'figures/4_um/4.0_barra/4.0.4_verticals/4.0.4.0 {var1} {', '.join(ds_names)} {cs_names}_{cs_namee} {plt_mode} {years}_{yeare} {ptop}hPa.png')
    
    # del era5_pl_mon_alltime, barra_c2_pl_mon_alltime, barra_r2_pl_mon_alltime




'''
# CS_A3 = [max_lon, min_lat]
# CS_A4 = [min_lon, max_lat]
# cs_names, cs_namee = 'A3', 'A4'
# if cs_names == 'A1':
#     cs_start = CS_A1
# elif cs_names == 'A3':
#     cs_start = CS_A3
# if cs_namee == 'A2':
#     cs_end = CS_A2
# elif cs_namee == 'A4':
#     cs_end = CS_A4



    data1 = get_cross_section(era5_pl_mon_alltime['am'], cs_start, cs_end)
    data2 = cross_section(era5_pl_mon_alltime['am'].squeeze().to_dataset().metpy.parse_cf()[var1], cs_start, cs_end, steps=len(data1.x))
    print((data1.squeeze().values[np.isfinite(data1.squeeze().values)] == data2.values[np.isfinite(data2.values)]).all())
    
    data3 = get_cross_section(barra_c2_pl_mon_alltime['am'], cs_start, cs_end)
    data4 = cross_section(barra_c2_pl_mon_alltime['am'].squeeze().to_dataset().metpy.parse_cf()[var2], cs_start, cs_end, steps=len(data3.x))
    print((data3.squeeze().values[np.isfinite(data3.squeeze().values)] == data4.values[np.isfinite(data4.values)]).all())
    
    data1 = get_cross_section(era5_pl_mon_alltime['ann'], cs_start, cs_end)
    data2 = cross_section(era5_pl_mon_alltime['ann'].squeeze().to_dataset().metpy.parse_cf()[var1], cs_start, cs_end, steps=len(data1.x))
    print((data1.squeeze().values[np.isfinite(data1.squeeze().values)] == data2.values[np.isfinite(data2.values)]).all())
    
    data3 = get_cross_section(barra_c2_pl_mon_alltime['ann'], cs_start, cs_end)
    data4 = cross_section(barra_c2_pl_mon_alltime['ann'].squeeze().to_dataset().metpy.parse_cf()[var2], cs_start, cs_end, steps=len(data3.x))
    print((data3.squeeze().values[np.isfinite(data3.squeeze().values)] == data4.values[np.isfinite(data4.values)]).all())


    # print(np.max(np.abs(era5_am - get_cross_section(era5_pl_mon_alltime['am'], cs_start, cs_end, steps=200).squeeze().sel(y=slice(200, 1000))) / era5_am).values)

    # print(np.max(np.abs(barra_r2_am - get_cross_section(barra_r2_pl_mon_alltime['am'], cs_start, cs_end, steps=200).squeeze().sel(y=slice(200, 1000))) / barra_r2_am).values)
    # print(np.max(np.abs(get_cross_section(barra_r2_pl_mon_alltime['ann'], cs_start, cs_end).sel(y=slice(200, 1000), time=slice(years, yeare)).mean(dim='time') - get_cross_section(barra_r2_pl_mon_alltime['am'], cs_start, cs_end).squeeze().sel(y=slice(200, 1000))) / get_cross_section(barra_r2_pl_mon_alltime['ann'], cs_start, cs_end).sel(y=slice(200, 1000), time=slice(years, yeare)).mean(dim='time')).values)

    # print(np.max(np.abs(barra_c2_am - get_cross_section(barra_c2_pl_mon_alltime['am'], cs_start, cs_end, steps=200).squeeze().sel(y=slice(200, 1000))) / barra_c2_am).values)
    # print(np.max(np.abs(get_cross_section(barra_c2_pl_mon_alltime['ann'], cs_start, cs_end).sel(y=slice(200, 1000), time=slice(years, yeare)).mean(dim='time') - get_cross_section(barra_c2_pl_mon_alltime['am'], cs_start, cs_end).squeeze().sel(y=slice(200, 1000))) / get_cross_section(barra_c2_pl_mon_alltime['ann'], cs_start, cs_end).sel(y=slice(200, 1000), time=slice(years, yeare)).mean(dim='time')).values)


'''
# endregion

