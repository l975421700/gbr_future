

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53


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
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot global era5 am data

era5_gridarea = xr.open_dataset('data/obs/era5/era5_gridarea.nc').cell_area

era5_sl_mon_alltime = {}
for var in ['e']:
    # var = 'tcw'
    # 'tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', 't2m', 'wind', 'mslhf', 'msnlwrf', 'msnswrf', 'msshf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'e', 'z', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', 'tciw', 'tclw'
    print(var)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    print(era5_sl_mon_alltime[var]['mon'].units)
    
    if var=='tp':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 1000
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean precipitation (1979-2023) [$mm \; day^{-1}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 3)) + r' $mm \; day^{-1}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        
        extend = 'max'
    elif var=='msl':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() / 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean sea level pressure (1979-2023) [$hPa$]' + '\nglobal mean: ' + str(int(plt_data_gm)) + r' $hPa$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=980, cm_max=1025, cm_interval1=2.5, cm_interval2=5, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='sst':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = r'ERA5 annual mean sea surface temperature (1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='hcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean high cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='mcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean medium cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='lcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean low cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='tcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean total cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='t2m':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = r'ERA5 annual mean 2 m temperature (1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-32, cm_max=32, cm_interval1=2, cm_interval2=8, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='wind':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        cbar_label = r'ERA5 annual mean 10 m wind speed (1979-2023) [$m\;s^{-1}$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=2, cm_max=12, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='mslhf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface latent heat flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-280, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='viridis',)
        
        extend = 'both'
    elif var=='msshf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface sensible heat flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
        
        extend = 'both'
    elif var=='msnlwrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface net longwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        
        extend = 'both'
    elif var=='msnswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface net shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=260, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='mtdwswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean TOA downward shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='mtnlwrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean TOA net longwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-120, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        
        extend = 'both'
    elif var=='mtnswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean TOA net shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=50, cm_max=350, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='msdwlwrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface downward longwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=90, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='msdwswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface downward shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='e':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 1000
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean evaporation (1979-2023) [$mm \; day^{-1}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 3)) + r' $mm \; day^{-1}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, -0.1, 0, 0.1])
        pltticks = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, -0.1, 0, 0.1])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis', len(pltlevel)-1)
        
        extend = 'both'
    elif var=='z':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() / 9.80665
        cbar_label = r'ERA5 orography [$m$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=6000, cm_interval1=250, cm_interval2=500, cmap='viridis_r',)
        
        extend = 'max'
    elif var=='tcw':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean total column water (1979-2023) [$kg \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $kg \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis_r',)
        
        extend = 'max'
    
    fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
    
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    
    if var in ['sst', 'wind']:
        ax.add_feature(cfeature.LAND,color='white',zorder=1,edgecolor=None,lw=0)
    if var in ['z']:
        ax.add_feature(cfeature.OCEAN,color='white',zorder=1,edgecolor=None,lw=0)
    
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks, extend=extend,
        pad=0.02, fraction=0.13,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
    
    fig.savefig(f'figures/5_era5/5.0_global/5.0.0 global era5 annual mean {var}.png')
    
    del era5_sl_mon_alltime[var]

'''
Mean surface latent heat flux: slhf
Mean surface sensible heat flux: sshf
Mean surface net long-wave radiation flux: msnlwrf
Mean surface net short-wave radiation flux: msnswrf
Mean surface downward long-wave radiation flux: msdwlwrf
Mean surface downward short-wave radiation flux: msdwswrf
Mean surface net long-wave radiation flux, clear sky: msnlwrfcs
Mean surface net short-wave radiation flux, clear sky: msnswrfcs
Mean surface downward long-wave radiation flux, clear sky: msdwlwrfcs
Mean surface downward short-wave radiation flux, clear sky: msdwswrfcs

Mean top downward short-wave radiation flux: mtdwswrf
Mean top net long-wave radiation flux: mtnlwrf
Mean top net short-wave radiation flux: mtnswrf
Mean top net long-wave radiation flux, clear sky: mtnlwrfcs
Mean top net short-wave radiation flux, clear sky: mtnswrfcs



'''
# endregion


# region plot barra-c2 am data

barra_c2_mon_alltime = {}
for var in ['pr']:
    # 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'tas', 'ts', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'psl',
    # var = 'rsut'
    print(var)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    print(barra_c2_mon_alltime[var]['mon'])
    print(barra_c2_mon_alltime[var]['mon'].units)
    
    if var=='pr':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() * seconds_per_d
        cbar_label = 'BARRA-C2 annual mean precipitation\n' + r'(1979-2023) [$mm \; day^{-1}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        
        extend = 'max'
    elif var=='clh':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean high cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='clm':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean medium cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='cll':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean low cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='clt':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean total cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='evspsbl':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() * seconds_per_d
        cbar_label = 'BARRA-C2 annual mean evaporation\n' + r'(1979-2023) [$mm \; day^{-1}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        
        extend = 'max'
    elif var=='hfls':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upward latent heat flux\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=280, cm_interval1=20, cm_interval2=40, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='hfss':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upward sensible heat flux\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=120, cm_interval1=10, cm_interval2=20, cmap='BrBG_r', asymmetric=True,)
        
        extend = 'both'
    elif var=='tas':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = 'BARRA-C2 annual mean near-surface air temperature\n' + r'(1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=28, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='ts':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = 'BARRA-C2 annual mean surface temperature\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=32, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rlds':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface downward longwave radiation\n' + r'(1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=280, cm_max=440, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rlus':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upward longwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=340, cm_max=490, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rlut':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean TOA outgoing longwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=280, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsds':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface downward shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=90, cm_max=270, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsdt':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean TOA incident shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=310, cm_max=410, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsus':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upwelling shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=110, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsut':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean TOA outgoing shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=50, cm_max=230, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    
    fig, ax = regional_plot(extent=[108, 160, -45.7, -5], central_longitude=180,)
    
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend=extend,
        pad=0.02, fraction=0.12,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.5, labelpad=4)
    fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.12, top = 0.99)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.0 whole region barra_c2 annual mean {var}.png')
    
    del barra_c2_mon_alltime[var]


'''
Precipitation: pr
High Level Cloud Fraction: clh
Mid Level Cloud Fraction: clm
Low Level Cloud Fraction: cll
Total Cloud Cover Percentage: clt
Evaporation Including Sublimation and Transpiration: evspsbl
sea level pressure: psl
near surface wind speed: sfcWind
Near surface air temperature: tas
Surface temperature: ts

Surface Upward Latent Heat Flux: hfls
Surface Upward Sensible Heat Flux: hfss
Surface downwelling LW radiation: rlds
Surface downwelling SW radiation: rsds
Surface Upwelling Longwave Radiation: rlus
Surface Upwelling Shortwave Radiation: rsus
Surface Downwelling Clear-Sky Longwave Radiation: rldscs
Surface Downwelling Clear-Sky Shortwave Radiation: rsdscs
Surface Upwelling Clear-Sky Longwave Radiation: rluscs
Surface Upwelling Clear-Sky Shortwave Radiation: rsuscs

TOA Incident Shortwave Radiation: rsdt
TOA Outgoing Longwave Radiation: rlut
TOA Outgoing Shortwave Radiation: rsut
TOA Outgoing Clear-Sky Longwave Radiation: rlutcs
TOA Outgoing Clear-Sky Shortwave Radiation: rsutcs




'''
# endregion


# region plot global era5 zm am data

era5_pl_mon_alltime = {}
for var in ['pv', 'q', 'r', 't', 'u', 'v', 'w', 'z']:
    # var = 'z'
    print(var)
    
    with open(f'data/obs/era5/mon/era5_pl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_pl_mon_alltime[var] = pickle.load(f)
    print(era5_pl_mon_alltime[var]['mon'])
    print(era5_pl_mon_alltime[var]['mon'].units)
    
    if var == 'pv':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon') / 1e-6
        cbar_label = r'ERA5 annual mean zonal mean potential vorticity (1979-2023) [$10^{-6} \; K \; m^{2} \; kg^{-1} \; s^{-1}$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-8, cm_max=8, cm_interval1=0.5, cm_interval2=2, cmap='BrBG',)
        extend = 'both'
    elif var == 'q':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon') * 1000
        cbar_label = r'ERA5 annual mean zonal mean specific humidity (1979-2023) [$g \; kg^{-1}$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 14])
        pltticks = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 14])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        extend = 'max'
    elif var == 'r':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon')
        cbar_label = r'ERA5 annual mean zonal mean relative humidity (1979-2023) [$\%$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='viridis_r',)
        extend = 'neither'
    elif var == 't':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon') - zerok
        cbar_label = r'ERA5 annual mean zonal mean temperature (1979-2023) [$°C$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=30, cm_interval1=2.5, cm_interval2=10, cmap='BrBG', asymmetric=True)
        extend = 'both'
    elif var == 'u':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon')
        cbar_label = r'ERA5 annual mean zonal mean u (1979-2023) [$m\;s^{-1}$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=30, cm_interval1=1, cm_interval2=5, cmap='BrBG', asymmetric=True)
        extend = 'both'
    elif var == 'v':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon')
        cbar_label = r'ERA5 annual mean zonal mean v (1979-2023) [$m\;s^{-1}$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-1.6, cm_max=2.2, cm_interval1=0.1, cm_interval2=0.4, cmap='BrBG', asymmetric=True)
        extend = 'both'
    elif var == 'w':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon')
        cbar_label = r'ERA5 annual mean zonal mean w (1979-2023) [$Pa\;s^{-1}$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-0.09, cm_max=0.24, cm_interval1=0.01, cm_interval2=0.03, cmap='BrBG', asymmetric=True)
        extend = 'both'
    elif var == 'z':
        plt_data = era5_pl_mon_alltime[var]['am'].sel(level=slice(200, 1000)).squeeze().mean(dim='lon') / 9.80665
        cbar_label = r'ERA5 annual mean zonal mean geopotential height (1979-2023) [$m$]'
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=12000, cm_interval1=1000, cm_interval2=2000, cmap='viridis_r')
        extend = 'both'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([13.2, 8.8]) / 2.54)
    
    plt_mesh = ax.pcolormesh(
        plt_data.lat, plt_data.level, plt_data.values,
        norm=pltnorm, cmap=pltcmp,)
    
    ax.set_xticks(np.arange(90, -90 - 1e-4, -30))
    ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
    ax.invert_yaxis()
    ax.set_ylim(1000, 200)
    ax.set_yticks(np.arange(1000, 200 - 1e-4, -200))
    ax.set_ylabel('Pressure [$hPa$]')
    ax.grid(True, lw=0.5, c='gray', alpha=0.5, linestyle='--',)
    
    cbar = fig.colorbar(
        plt_mesh, ax=ax, aspect=40, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend=extend,
        pad=0.1, fraction=0.04,)
    cbar.ax.set_xlabel(cbar_label)
    
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.98)
    fig.savefig(f'figures/5_era5/5.0_global/5.0.0 global era5 am zm {var}.png')
    
    del era5_pl_mon_alltime[var]




# endregion


# region plot ERA5 - BARRA-C2 am sm data

# settings
plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['ERA5', 'BARRA-C2', 'ERA5 - BARRA-C2']

era5_sl_mon_alltime = {}
barra_c2_mon_alltime = {}

for var1, var2 in zip(['e'], ['evspsbl']):
    # ['e'], ['evspsbl']
    # ['t2m'], ['tas']
    # ['msl'], ['psl']
    # ['msshf'], ['hfss']
    # ['mslhf'], ['hfls']
    # ['hcc', 'mcc', 'lcc', 'tcc'], ['clh', 'clm', 'cll', 'clt']
    # ['tp'], ['pr']
    print(f'{var1} in ERA5 vs. {var2} in BARRA-C2')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    print(era5_sl_mon_alltime[var1]['mon'])
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    print(barra_c2_mon_alltime[var2]['mon'])
    
    plt_data={}
    plt_data['ERA5']={}
    plt_data['BARRA-C2']={}
    plt_data['ERA5 - BARRA-C2']={}
    
    if var1=='tp':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() * 1000
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze() * seconds_per_d
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 1000,
                    regrid(barra_c2_mon_alltime[var2]['ann'] * seconds_per_d, ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) * 1000
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames) * seconds_per_d
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 1000,
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames] * seconds_per_d, ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'Precipitation (1979-2023) [$mm\;day^{-1}$]'
        cbar_label2=r'Difference in precipitation (1979-2023) [$mm\;day^{-1}$]'
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        extend1 = 'max'
        pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
        extend2 = 'both'
    elif var1=='hcc':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze()
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['ann'], ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'High cloud cover (1979-2023) [$\%$]'
        cbar_label2=r'Difference in high cloud cover (1979-2023) [$\%$]'
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
        extend2 = 'both'
    elif var1=='mcc':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze()
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['ann'], ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'Middle cloud cover (1979-2023) [$\%$]'
        cbar_label2=r'Difference in middle cloud cover (1979-2023) [$\%$]'
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
        extend2 = 'both'
    elif var1=='lcc':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze()
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['ann'], ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'Low cloud cover (1979-2023) [$\%$]'
        cbar_label2=r'Difference in low cloud cover (1979-2023) [$\%$]'
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
        extend2 = 'both'
    elif var1=='tcc':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze()
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['ann'], ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) * 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * 100,
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'Total cloud cover (1979-2023) [$\%$]'
        cbar_label2=r'Difference in total cloud cover (1979-2023) [$\%$]'
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
        extend2 = 'both'
    elif var1=='mslhf':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze()
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze() * (-1)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['ann'] * (-1), ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True)
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames) * (-1)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames] * (-1), ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
    elif var1=='msshf':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze()
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze() * (-1)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['ann'] * (-1), ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True)
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames) * (-1)
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames] * (-1), ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'Surface sensible heat flux (1979-2023) [$W \; m^{-2}$]'
        cbar_label2=r'Difference in surface sensible heat flux (1979-2023) [$W \; m^{-2}$]'
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='PRGn', asymmetric=True,)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
        extend2 = 'both'
    elif var1=='msl':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() / 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze() / 100
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['ann'], ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) / 100
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames) / 100
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
    elif var1=='t2m':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() - zerok
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze() - zerok
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['ann'], ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) - zerok
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames) - zerok
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)),
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'2 m temperature (1979-2023) [$°C$]'
        cbar_label2=r'Difference in 2 m temperature (1979-2023) [$°C$]'
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=2, cm_max=34, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=1, cmap='BrBG',)
        extend2 = 'both'
    elif var1=='e':
        for jcolnames in plt_colnames:
            print(jcolnames)
            if jcolnames=='Annual mean':
                # jcolnames='Annual mean'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)).squeeze() * (-1000)
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].squeeze() * seconds_per_d
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * (-1000),
                    regrid(barra_c2_mon_alltime[var2]['ann'] * seconds_per_d, ds_out=plt_data['ERA5'][jcolnames])
                )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            else:
                # jcolnames='MAM'
                plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(-5, -45.7), lon=slice(108, 160), drop=True) * (-1000)
                plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames) * seconds_per_d
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
                
                # statistical test
                ttest_fdr_res = ttest_fdr_control(
                    era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(-5, -45.7), lon=slice(108, 160)) * (-1000),
                    regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames] * seconds_per_d, ds_out=plt_data['ERA5'][jcolnames])
                    )
                plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
            
            print(stats.describe(plt_data['ERA5'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['BARRA-C2'][jcolnames].values, axis=None))
            print(stats.describe(plt_data['ERA5 - BARRA-C2'][jcolnames].values, axis=None, nan_policy='omit'))
        
        cbar_label1=r'Evaporation (1979-2023) [$mm\;day^{-1}$]'
        cbar_label2=r'Difference in evaporation (1979-2023) [$mm\;day^{-1}$]'
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        extend1 = 'both'
        pltlevel2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltticks2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
        extend2 = 'both'
    
    nrow=3
    ncol=5
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        axs[irow, 0].text(-0.05, 0.5, plt_rownames[irow], ha='right', va='center', rotation='vertical', transform=axs[irow, 0].transAxes,weight='bold')
        for jcol in range(ncol):
            axs[irow, jcol] = regional_plot(extent=[108, 160, -45.7, -5], central_longitude=180, ax_org=axs[irow, jcol])
            axs[irow, jcol].text(0, 1.02, f'({string.ascii_lowercase[irow]}{jcol+1})', ha='left', va='bottom', transform=axs[irow, jcol].transAxes,)
            if irow==0:
                axs[0, jcol].text(0.5, 1.02, plt_colnames[jcol], ha='center', va='bottom', transform=axs[0, jcol].transAxes,weight='bold')
    
    for irow in range(nrow-1):
        for jcol in range(ncol):
            plt_mesh1 = axs[irow, jcol].pcolormesh(
                plt_data[plt_rownames[irow]][plt_colnames[jcol]].lon,
                plt_data[plt_rownames[irow]][plt_colnames[jcol]].lat,
                plt_data[plt_rownames[irow]][plt_colnames[jcol]].values,
                norm=pltnorm1, cmap=pltcmp1,
                transform=ccrs.PlateCarree(),zorder=1,)
    
    for jcol in range(ncol):
        plt_mesh2 = axs[2, jcol].pcolormesh(
            plt_data[plt_rownames[2]][plt_colnames[jcol]].lon,
            plt_data[plt_rownames[2]][plt_colnames[jcol]].lat,
            plt_data[plt_rownames[2]][plt_colnames[jcol]].values,
            norm=pltnorm2, cmap=pltcmp2,
            transform=ccrs.PlateCarree(),zorder=1,)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.02, 0.4, 0.02]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.02, 0.4, 0.02]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.98)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 barra_c2 vs. era5 am sm {var1}.png')
    
    del era5_sl_mon_alltime[var1], barra_c2_mon_alltime[var2]


# endregion




