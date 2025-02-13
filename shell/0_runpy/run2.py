

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


# region plot barra-c2 am data

barra_c2_mon_alltime = {}
for var in ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl']:
    # 'hurs', 'huss',
    # var = 'rsut'
    print(f'#-------------------------------- {var}')
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
    # print(era5_varlabels[cmip6_era5_var[var]])
    # print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
    # del barra_c2_mon_alltime[var]
    
    cbar_label = 'BARRA-C2 annual mean ' + era5_varlabels[cmip6_era5_var[var]]
    
    if var in ['pr', 'evspsbl']:
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
