

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=20GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import pandas as pd
import numpy.ma as ma
import glob
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
from pyhdf.error import HDF4Error
from satpy.scene import Scene
from skimage.measure import block_reduce
import xarray as xr

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)

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
    month_days,
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
    plot_loc,
    draw_polygon,
)

from calculations import (
    find_ilat_ilon,
    )

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs


# endregion


# region plot CM SAF

years = '2016'
yeare = '2023'
vars = {'clwvi': 'lwp_allsky', 'clivi': 'iwp_allsky',
        'rlut': 'LW_flux', 'rsut': 'SW_flux', 'CDNC': 'cdnc_liq'}


for ivar in ['clwvi', 'clivi']:
    # vars.keys()
    # ivar = 'clwvi'
    print(f'#---------------- {ivar}')
    
    fl = sorted(glob.glob(f'data/obs/CM_SAF/{ivar}/*.nc'))[444:540]
    # print(len(fl))
    cm_saf = xr.open_mfdataset(fl).sel(time=slice(years, yeare))
    plt_data = cm_saf[vars[ivar]].weighted(cm_saf[vars[ivar]].time.dt.days_in_month).mean(dim='time').compute()
    
    if ivar in ['rsut', 'rlut']:
        plt_data *= (-1)
    elif ivar in ['clwvi', 'clivi']:
        plt_data *= 1000
    elif ivar == 'CDNC':
        plt_data /= 1e6
    
    plt_data_gm = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
    
    cbar_label = f'CM SAF annual mean ({years}-{yeare}) {era5_varlabels[cmip6_era5_var[ivar]]}\nglobal mean: {str(np.round(plt_data_gm, 2))}'
    
    if ivar=='clivi':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        extend='max'
    elif ivar=='clwvi':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        extend='max'
    elif ivar=='CDNC':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=150, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        extend='max'
    elif ivar in ['prw']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis_r',)
        extend='max'
    elif ivar in ['rsut']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
        extend='both'
    elif ivar in ['rlut']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        extend='both'
    else:
        print(f'Warning unspecified colorbar for {vars[ivar]}')
    
    fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks, extend=extend,
        pad=0.02, fraction=0.13,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
    fig.savefig(f'figures/3_satellites/3.4_CM_SAF/3.4.0 CM SAF {ivar} global am.png')



'''


cm_saf['lwp'].weighted(cm_saf['lwp'].time.dt.days_in_month).mean(dim='time')
cm_saf['lwp_allsky'].weighted(cm_saf['lwp_allsky'].time.dt.days_in_month).mean(dim='time')
'''
# endregion

