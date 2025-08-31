

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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


# region plot ceres clouds

years = '2016'
yeare = '2023'
ceres_data = xr.open_dataset('data/obs/CERES/CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_201601-202412.nc').sel(time=slice(years, yeare))

vars = {'lwp_total_mon': 'clwvi', 'iwp_total_mon': 'clivi'}
# cldwatrad_total_mon, cldicerad_total_mon


for ivar in vars.keys():
    # ivar = vars[0]
    print(f'#---------------- {ivar} {vars[ivar]}')
    
    plt_data = ceres_data[ivar].weighted(ceres_data[ivar].time.dt.days_in_month).mean(dim='time', skipna=False)
    plt_data_gm = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
    
    cbar_label = f'CERES annual mean ({years}-{yeare}) {era5_varlabels[cmip6_era5_var[vars[ivar]]]}\nglobal mean: {str(np.round(plt_data_gm, 1))}'
    
    if vars[ivar]=='clivi':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif vars[ivar]=='clwvi':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif vars[ivar] in ['prw']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis',)
    else:
        print(f'Warning unspecified colorbar for {vars[ivar]}')
    
    fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks, extend='max',
        pad=0.02, fraction=0.13,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
    fig.savefig(f'figures/3_satellites/3.3_ceres/3.3.0 ceres {vars[ivar]} global am.png')









# endregion


