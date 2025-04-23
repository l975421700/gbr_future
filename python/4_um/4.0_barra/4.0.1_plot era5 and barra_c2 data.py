

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=60GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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
import xesmf as xe

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
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot global era5 am data

era5_gridarea = xr.open_dataset('data/obs/era5/era5_gridarea.nc').cell_area

era5_sl_mon_alltime = {}
for var in ['toa_albedo', 'toa_albedocs', 'toa_albedocl']:
    # var = 'sst'
    # 'msnrf', 'tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', 't2m', 'msnlwrf', 'msnswrf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'tciw', 'tclw', 'e', 'z', 'mslhf', 'msshf', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', 'si10', 'd2m', 'cp', 'lsp', 'deg0l', 'mper', 'pev', 'skt', 'u10', 'v10', 'u100', 'v100',    'msuwlwrf',  'msuwswrf',  'msuwlwrfcs',  'msuwswrfcs',  'msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl',  'mtuwswrf',  'mtuwswrfcs',  'mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl'
    print(f'#-------------------------------- {var}')
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    
    plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
    # print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
    # del era5_sl_mon_alltime[var]
    plt_data_gm = np.average(
        plt_data.values.ravel()[np.isfinite(plt_data.values.ravel())],
        weights=era5_gridarea.values.ravel()[np.isfinite(plt_data.values.ravel())])
    
    cbar_label = 'ERA5 annual mean (1979-2023) ' + era5_varlabels[var] + '\nglobal mean: ' + str(np.round(plt_data_gm, 2))
    
    if var in ['tp', 'tciw', 'tclw', 'z', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', 'cp', 'lsp']:
        extend = 'max'
    elif var in ['hcc', 'mcc', 'lcc', 'tcc', 'toa_albedo', 'toa_albedocs', 'toa_albedocl']:
        extend = 'neither'
    else:
        extend = 'both'
    
    if var in ['tp', 'cp', 'lsp']:
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
    elif var=='msl':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=980, cm_max=1025, cm_interval1=2.5, cm_interval2=5, cmap='viridis_r',)
    elif var=='sst':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
    elif var in ['t2m', 'skt']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-48, cm_max=32, cm_interval1=2, cm_interval2=8, cmap='BrBG', asymmetric=True,)
    elif var in ['si10']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=2, cm_max=12, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
    elif var in ['msnlwrf', 'msnlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-170, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['msnswrf', 'msnswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var=='mtdwswrf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['mtnswrf', 'mtnswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=50, cm_max=380, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msdwswrf', 'msdwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=330, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var=='cbh':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=7600, cm_interval1=200, cm_interval2=800, cmap='viridis_r',)
    elif var=='tciw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.14, cm_interval1=0.01, cm_interval2=0.02, cmap='viridis_r',)
    elif var=='tclw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.05, cmap='viridis_r',)
    elif var in ['e', 'mper', 'pev']:
        pltlevel = np.array([-0.1, 0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10])
        pltticks = np.array([-0.1, 0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis', len(pltlevel)-1)
    elif var=='z':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=6000, cm_interval1=250, cm_interval2=500, cmap='viridis_r',)
    elif var=='mslhf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-280, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='viridis',)
    elif var=='msshf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var in ['tcw', 'tcwv']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis_r',)
    elif var in ['tcsw']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.9, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis_r',)
    elif var=='tcrw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.05, cmap='viridis_r',)
    elif var=='tcslw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.2, cm_interval1=0.02, cm_interval2=0.04, cmap='viridis_r',)
    elif var=='d2m':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-56, cm_max=24, cm_interval1=2, cm_interval2=8, cmap='BrBG', asymmetric=True,)
    elif var=='deg0l':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=5200, cm_interval1=200, cm_interval2=400, cmap='BrBG', asymmetric=True,)
    elif var=='u10':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-12, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var=='v10':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-8, cm_max=14, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var=='u100':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-18, cm_max=12, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var=='v100':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-12, cm_max=20, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-540, cm_max=-120, cm_interval1=10, cm_interval2=40, cmap='viridis')
    elif var in ['msuwswrf', 'msuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-220, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['msnlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=80, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var in ['msnswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-140, cm_max=60, cm_interval1=10, cm_interval2=40, cmap='BrBG', asymmetric=True,)
    elif var in ['msdwlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=65, cm_interval1=5, cm_interval2=10, cmap='viridis_r')
    elif var in ['msdwswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['msuwlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-45, cm_max=45, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msuwswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-80, cm_max=80, cm_interval1=5, cm_interval2=20, cmap='BrBG')
    elif var in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['mtnlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=85, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True,)
    elif var in ['mtnswrfcl', 'mtuwswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-150, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var in ['toa_albedo', 'toa_albedocs', 'toa_albedocl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis')
    elif var in ['msnrf']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=160, cm_interval1=20, cm_interval2=40, cmap='BrBG')
    else:
        print(f'Warning unspecified colorbar for {var}')
    
    fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
    
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    
    if var in ['sst']:
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
'''
# endregion


# region plot global era5 derived am data



surface_radiation = {}
for var2 in ['rsus', 'rlus', 'rsds', 'rlds', 'hfls', 'hfss']:
    # var2='rsus'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        surface_radiation[var2] = pickle.load(f)['am'].squeeze()


plt_data = sum(surface_radiation[var] for var in surface_radiation.keys())
plt_mean = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
cbar_label = r'ERA5 annual mean (1979-2023) surface radiation excess [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_mean, 2))
extend = 'both'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-160, cm_max=160, cm_interval1=20, cm_interval2=40, cmap='BrBG',)
opng = f'figures/5_era5/5.0_global/5.0.0 global era5 annual mean surface radiation excess.png'


fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
plt_mesh = ax.pcolormesh(
    plt_data.lon, plt_data.lat, plt_data.values,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)
cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend=extend,
    pad=0.02, fraction=0.13)
cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
fig.savefig(opng)


# endregion


# region plot barra-c2 am data

barra_c2_mon_alltime = {}
for var in ['clwvi', 'clivi', 'prw']:
    # 'hurs', 'huss',
    # var = 'rsut'
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl']
    print(f'#-------------------------------- {var}')
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
    # print(era5_varlabels[cmip6_era5_var[var]])
    print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
    # del barra_c2_mon_alltime[var]
    
    cbar_label = 'BARRA-C2 annual mean\n' + era5_varlabels[cmip6_era5_var[var]]
    
    if var in ['pr', 'evspsbl', 'clwvi', 'clivi', 'prw']:
        extend = 'max'
    elif var in ['clh', 'clm', 'cll', 'clt']:
        extend = 'neither'
    else:
        extend = 'both'
    
    if var=='pr':
        pltlevel = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
    elif var in ['clh', 'clm', 'cll', 'clt']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
    elif var in ['evspsbl', 'evspsblpot']:
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
    elif var=='hfls':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-350, cm_max=0, cm_interval1=10, cm_interval2=50, cmap='viridis',)
    elif var=='hfss':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-130, cm_max=30, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var=='psl':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=1007, cm_max=1019, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
    elif var in ['rlds', 'rldscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=240, cm_max=440, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['rlus', 'rluscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-490, cm_max=-340, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['rlut', 'rlutcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-180, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['rsds', 'rsdscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=90, cm_max=360, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var=='rsdt':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=310, cm_max=410, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['rsus', 'rsuscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-120, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['rsut', 'rsutcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-230, cm_max=-40, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var=='sfcWind':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=1, cm_max=11, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
    elif var=='tas':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=29, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
    elif var=='ts':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=5, cm_max=32, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
    elif var=='uas':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-8, cm_max=8, cm_interval1=1, cm_interval2=2, cmap='BrBG',)
    elif var=='vas':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=6, cm_interval1=0.5, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var in ['rlns', 'rlnscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-125, cm_max=-15, cm_interval1=5, cm_interval2=10, cmap='viridis',)
    elif var in ['rsns', 'rsnscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['rlnscl', 'rldscl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
    elif var in ['rsnscl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-200, cm_max=0, cm_interval1=10, cm_interval2=40, cmap='viridis',)
    elif var in ['rsdscl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-240, cm_max=0, cm_interval1=10, cm_interval2=40, cmap='viridis',)
    elif var in ['rluscl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-4.5, cm_max=0, cm_interval1=0.5, cm_interval2=0.5, cmap='viridis',)
    elif var in ['rsuscl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=40, cm_interval1=5, cm_interval2=5, cmap='viridis_r',)
    elif var in ['rsnt', 'rsntcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=160, cm_max=380, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
    elif var in ['rlutcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
    elif var in ['rsntcl', 'rsutcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['prw']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=2.5, cm_interval2=10, cmap='viridis_r',)
    elif var=='clwvi':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.4, cm_interval1=0.05, cm_interval2=0.05, cmap='viridis_r',)
    elif var=='clivi':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1.2, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis_r',)
    else:
        print(f'Warning unspecified colorbar for {var}')
    
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
# ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl']


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
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
# min_lon, max_lon, min_lat, max_lat = [108, 160, -45.7, -5]

era5_sl_mon_alltime = {}
barra_c2_mon_alltime = {}

for var2 in ['clwvi', 'clivi', 'prw']:
    # var2='clh'
    # 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} in ERA5 vs. {var2} in BARRA-C2')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    
    plt_data={}
    plt_data['ERA5']={}
    plt_data['BARRA-C2']={}
    plt_data['ERA5 - BARRA-C2']={}
    
    for jcolnames in plt_colnames:
        print(f'#---------------- {jcolnames}')
        if jcolnames=='Annual mean':
            # jcolnames='Annual mean'
            plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)).squeeze()
            plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)).squeeze()
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
            
            ttest_fdr_res = ttest_fdr_control(
                era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)),
                regrid(barra_c2_mon_alltime[var2]['ann'].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)), ds_out=plt_data['ERA5'][jcolnames])
            )
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
        else:
            # jcolnames='MAM'
            plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon), drop=True)
            plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames, lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
            
            ttest_fdr_res = ttest_fdr_control(
                era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)),
                regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)), ds_out=plt_data['ERA5'][jcolnames])
                    )
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(np.concatenate([np.concatenate([plt_data['ERA5']['Annual mean'].values, plt_data['ERA5']['DJF'].values, plt_data['ERA5']['MAM'].values, plt_data['ERA5']['JJA'].values, plt_data['ERA5']['SON'].values]).ravel(), np.concatenate([plt_data['BARRA-C2']['Annual mean'].values, plt_data['BARRA-C2']['DJF'].values, plt_data['BARRA-C2']['MAM'].values, plt_data['BARRA-C2']['JJA'].values, plt_data['BARRA-C2']['SON'].values]).ravel()]), axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data['ERA5 - BARRA-C2']['Annual mean'].values, plt_data['ERA5 - BARRA-C2']['DJF'].values, plt_data['ERA5 - BARRA-C2']['MAM'].values, plt_data['ERA5 - BARRA-C2']['JJA'].values, plt_data['ERA5 - BARRA-C2']['SON'].values]).ravel(), axis=None, nan_policy='omit'))
    
    cbar_label1 = '1979-2023 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 1979-2023 ' + era5_varlabels[var1]
    
    if var1 in ['tp', 'tclw', 'tciw', 'tcwv']:
        extend1 = 'max'
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        extend1 = 'neither'
    else:
        extend1 = 'both'
    extend2 = 'both'
    
    if var1=='tp':
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['e', 'pev']:
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltticks2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1=='mslhf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-460, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-200, cm_max=200, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
    elif var1=='msshf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-200, cm_max=80, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True,)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-160, cm_max=160, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
    elif var1=='msl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.5, cm_max=4.5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG', asymmetric=True,)
    elif var1 in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=200, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=90, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var1 in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-540, cm_max=-300, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-60, cm_max=60, cm_interval1=5, cm_interval2=20, cmap='BrBG',)
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-310, cm_max=-180, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='BrBG',)
    elif var1 in ['msdwswrf', 'msdwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.3, cm_max=0.6, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG',)
    elif var1 in ['msuwswrf', 'msuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1 in ['si10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=1, cm_max=11, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
    elif var1=='t2m':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-2, cm_max=34, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4, cm_max=10, cm_interval1=1, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1=='skt':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-2, cm_max=40, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-7, cm_max=10, cm_interval1=1, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1 in ['u10', 'v10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-9, cm_max=9, cm_interval1=1, cm_interval2=2, cmap='PRGn',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=2, cmap='BrBG',)
    elif var1 in ['msnlwrf', 'msnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['msnswrf', 'msnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=30, cm_max=350, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1=='msnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-25, cm_max=70, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=5, cmap='BrBG',)
    elif var1=='msnswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-230, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='msdwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=70, cm_interval1=2.5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=10, cm_interval1=2.5, cm_interval2=5, cmap='BrBG', asymmetric=True)
    elif var1=='msdwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-270, cm_max=0, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='msuwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='PRGn',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1=='msuwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['mtnswrf', 'mtnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=70, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='mtnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-16, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtnswrfcl', 'mtuwswrfcl']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-180, cm_max=30, cm_interval1=10, cm_interval2=20, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1 in ['tcwv']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=2.5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
    elif var1 in ['tclw']:
        pltlevel1 = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
        pltticks1 = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis', len(pltlevel1)-1)
        pltlevel2 = np.array([-0.1, -0.06, -0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04, 0.06, 0.1])
        pltticks2 = np.array([-0.1, -0.06, -0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04, 0.06, 0.1])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG_r', len(pltlevel2)-1)
    elif var1 in ['tciw']:
        pltlevel1 = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltticks1 = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis', len(pltlevel1)-1)
        pltlevel2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05, -0.02, 0])
        pltticks2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05, -0.02, 0])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('Blues_r', len(pltlevel2)-1)
    else:
        print(f'Warning unspecified colorbar for {var1}')
    
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
            axs[irow, jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[irow, jcol])
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


# region plot barra-c2/barra-r2/era5 mm data

mpl.rc('font', family='Times New Roman', size=10)
extent=[110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4
nrow = 3
ncol = 4
fm_bottom = 1.4 / (panelh*nrow + 1.4)


ids = 'ERA5' #'BARRA-R2' #'BARRA-C2' #


ds_mon_alltime = {}
for var2 in ['cll', 'clm', 'clh', 'clt']:
    # var2='cll'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    if ids=='ERA5':
        with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
            ds_mon_alltime[var2] = pickle.load(f)
    elif ids=='BARRA-R2':
        with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
            ds_mon_alltime[var2] = pickle.load(f)
    elif ids=='BARRA-C2':
        with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
            ds_mon_alltime[var2] = pickle.load(f)
    
    opng=f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.0 mm {var2} {ids}.png'
    cbar_label = f'2016-2023 {ids} {era5_varlabels[var1]}'
    
    if var2 in ['cll', 'clm', 'clh']:
        extend='max'
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
    elif var2 in ['clt']:
        extend='neither'
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
    
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
            
            plt_data = ds_mon_alltime[var2]['mon'][ds_mon_alltime[var2]['mon'].time.dt.month==(irow*4+jcol+1)].sel(time=slice('2016', '2023')).mean(dim='time')
            if ids=='ERA5':
                plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)).compute()
            elif ids in ['BARRA-R2', 'BARRA-C2']:
                plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).compute()
            plt_mesh = axs[irow, jcol].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            
            plt_mean = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
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
        orientation="horizontal", ticks=pltticks, extend=extend,
        cax=fig.add_axes([0.25, fm_bottom-0.01, 0.5, 0.02]))
    cbar.ax.set_xlabel(cbar_label)
    fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.98)
    fig.savefig(opng)
    
    del ds_mon_alltime[var2]






'''
['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl']
'''
# endregion


# region plot ERA5, BARRA-R2, BARRA-C2 am sm data

years = '2016'
yeare = '2023'

# settings
plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['ERA5', 'BARRA-R2 - ERA5', 'BARRA-C2 - ERA5']
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent

era5_sl_mon_alltime = {}
barra_r2_mon_alltime = {}
barra_c2_mon_alltime = {}

for var2 in ['clwvi', 'clivi', 'prw']:
    # var2='clwvi'
    # 'clwvi', 'clivi', 'prw', 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} in ERA5 vs. {var2} in BARRA-R2/C2')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    
    plt_data={}
    for irow in plt_rownames: plt_data[irow] = {}
    plt_mean = {}
    plt_rmse = {}
    for irow in plt_rownames[1:]: plt_rmse[irow] = {}
    
    for jcolnames in plt_colnames:
        print(f'#---------------- {jcolnames}')
        if jcolnames=='Annual mean':
            # jcolnames='Annual mean'
            era5_ann = era5_sl_mon_alltime[var1]['ann'].sel(time=slice(years, yeare), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
            plt_data['ERA5'][jcolnames] = era5_ann.mean(dim='time')
            
            barra_r2_ann = barra_r2_mon_alltime[var2]['ann'].sel(time=slice(years, yeare))
            barra_r2_regridder = xe.Regridder(barra_r2_ann,era5_ann,'bilinear')
            barra_r2_ann = barra_r2_regridder(barra_r2_ann)
            plt_data['BARRA-R2 - ERA5'][jcolnames] = barra_r2_ann.mean(dim='time') - plt_data['ERA5'][jcolnames]
            ttest_fdr_res = ttest_fdr_control(era5_ann, barra_r2_ann)
            plt_data['BARRA-R2 - ERA5'][jcolnames] = plt_data['BARRA-R2 - ERA5'][jcolnames].where(ttest_fdr_res, np.nan)
            
            barra_c2_ann = barra_c2_mon_alltime[var2]['ann'].sel(time=slice(years, yeare))
            barra_c2_regridder = xe.Regridder(barra_c2_ann,era5_ann,'bilinear')
            barra_c2_ann = barra_c2_regridder(barra_c2_ann)
            plt_data['BARRA-C2 - ERA5'][jcolnames] = barra_c2_ann.mean(dim='time') - plt_data['ERA5'][jcolnames]
            ttest_fdr_res = ttest_fdr_control(era5_ann, barra_c2_ann)
            plt_data['BARRA-C2 - ERA5'][jcolnames] = plt_data['BARRA-C2 - ERA5'][jcolnames].where(ttest_fdr_res, np.nan)
        else:
            # jcolnames='DJF'
            era5_sea = era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(time=slice(years, yeare), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
            plt_data['ERA5'][jcolnames] = era5_sea.mean(dim='time')
            
            barra_r2_sea = barra_r2_regridder(barra_r2_mon_alltime[var2]['sea'][barra_r2_mon_alltime[var2]['sea'].time.dt.season==jcolnames].sel(time=slice(years, yeare)))
            plt_data['BARRA-R2 - ERA5'][jcolnames] = barra_r2_sea.mean(dim='time') - plt_data['ERA5'][jcolnames]
            ttest_fdr_res = ttest_fdr_control(era5_sea, barra_r2_sea)
            plt_data['BARRA-R2 - ERA5'][jcolnames] = plt_data['BARRA-R2 - ERA5'][jcolnames].where(ttest_fdr_res, np.nan)
            
            barra_c2_sea = barra_c2_regridder(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames].sel(time=slice(years, yeare)))
            plt_data['BARRA-C2 - ERA5'][jcolnames] = barra_c2_sea.mean(dim='time') - plt_data['ERA5'][jcolnames]
            ttest_fdr_res = ttest_fdr_control(era5_sea, barra_c2_sea)
            plt_data['BARRA-C2 - ERA5'][jcolnames] = plt_data['BARRA-C2 - ERA5'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_mean[jcolnames] = plt_data['ERA5'][jcolnames].weighted(np.cos(np.deg2rad(plt_data['ERA5'][jcolnames].lat))).mean().values
        plt_rmse['BARRA-R2 - ERA5'][jcolnames] = np.sqrt(np.square(plt_data['BARRA-R2 - ERA5'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['BARRA-R2 - ERA5'][jcolnames].lat))).mean()).values
        plt_rmse['BARRA-C2 - ERA5'][jcolnames] = np.sqrt(np.square(plt_data['BARRA-C2 - ERA5'][jcolnames]).weighted(np.cos(np.deg2rad(plt_data['BARRA-C2 - ERA5'][jcolnames].lat))).mean()).values
    
    print(stats.describe(np.concatenate([plt_data['ERA5'][colname].values for colname in plt_colnames]), axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[rowname][colname].values for rowname in plt_rownames[1:] for colname in plt_colnames]), axis=None, nan_policy='omit'))
    
    cbar_label1 = f'{years}-{yeare} ' + era5_varlabels[var1]
    cbar_label2 = f'Difference in {years}-{yeare} ' + era5_varlabels[var1]
    
    if var1 in ['tp', 'tclw', 'tciw', 'tcwv']:
        extend1 = 'max'
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        extend1 = 'neither'
    else:
        extend1 = 'both'
    extend2 = 'both'
    
    if var1=='tp':
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['e', 'pev']:
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltticks2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1=='mslhf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-460, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-200, cm_max=200, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
    elif var1=='msshf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-200, cm_max=80, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True,)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-160, cm_max=160, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
    elif var1=='msl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.5, cm_max=4.5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG', asymmetric=True,)
    elif var1 in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=200, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=90, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var1 in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-540, cm_max=-300, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-60, cm_max=60, cm_interval1=5, cm_interval2=20, cmap='BrBG',)
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-310, cm_max=-180, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='BrBG',)
    elif var1 in ['msdwswrf', 'msdwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.3, cm_max=0.6, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG',)
    elif var1 in ['msuwswrf', 'msuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1 in ['si10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=1, cm_max=11, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
    elif var1=='t2m':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-2, cm_max=34, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4, cm_max=10, cm_interval1=1, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1=='skt':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-2, cm_max=40, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-7, cm_max=10, cm_interval1=1, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1 in ['u10', 'v10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-9, cm_max=9, cm_interval1=1, cm_interval2=2, cmap='PRGn',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=2, cmap='BrBG',)
    elif var1 in ['msnlwrf', 'msnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['msnswrf', 'msnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=30, cm_max=350, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1=='msnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-25, cm_max=70, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=5, cmap='BrBG',)
    elif var1=='msnswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-230, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='msdwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=70, cm_interval1=2.5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=10, cm_interval1=2.5, cm_interval2=5, cmap='BrBG', asymmetric=True)
    elif var1=='msdwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-270, cm_max=0, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='msuwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='PRGn',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1=='msuwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['mtnswrf', 'mtnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=70, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='mtnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-16, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtnswrfcl', 'mtuwswrfcl']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-180, cm_max=30, cm_interval1=10, cm_interval2=20, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1 in ['tcwv']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=2.5, cm_interval2=10, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG_r')
    elif var1 in ['tclw']:
        pltlevel1 = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
        pltticks1 = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-0.1, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.1])
        pltticks2 = np.array([-0.1, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.1])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1 in ['tciw']:
        pltlevel1 = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltticks1 = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltticks2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    else:
        print(f'Warning unspecified colorbar for {var1}')
    
    nrow=3
    ncol=5
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        axs[irow, 0].text(-0.05, 0.5, plt_rownames[irow], ha='right', va='center', rotation='vertical', transform=axs[irow, 0].transAxes)
        for jcol in range(ncol):
            axs[irow, jcol] = regional_plot(extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
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
            plt_text = 'Mean: '+str(np.round(plt_mean[plt_colnames[jcol]], 2))
        else:
            plt_text = np.round(plt_mean[plt_colnames[jcol]], 2)
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
                plt_text = 'RMSE: '+str(np.round(plt_rmse[plt_rownames[irow+1]][plt_colnames[jcol]], 2))
            else:
                plt_text = np.round(plt_rmse[plt_rownames[irow+1]][plt_colnames[jcol]], 2)
            axs[irow+1, jcol].text(
                0.5, 1.02, plt_text,
                ha='center', va='bottom', transform=axs[irow+1, jcol].transAxes)
    
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
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.95)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 barra_r2_c2 vs. era5 am sm {var1}.png')
    
    del era5_sl_mon_alltime[var1], barra_c2_mon_alltime[var2], barra_r2_mon_alltime[var2]


# endregion

