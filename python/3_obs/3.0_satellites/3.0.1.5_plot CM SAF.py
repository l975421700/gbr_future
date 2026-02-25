

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=10GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/gx60+gdata/py18


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
mpl.rc('font', family='Times New Roman', size=12)
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
    time_weighted_mean,
    coslat_weighted_mean,
    coslat_weighted_rmsd,
    land_ocean_mean,
    )

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs


# endregion


# region plot CM SAF

years = '2016'
yeare = '2023'
vars = {'clwvi': 'lwp_allsky', 'clivi': 'iwp_allsky',
        'rlut': 'LW_flux', 'rsut': 'SW_flux', 'CDNC': 'cdnc_liq'}
plt_regions = ['global'] # ['global', 'c2_domain', 'h9_domain']

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for ivar in ['CDNC']:
    # vars.keys()
    # ivar = 'clwvi'
    print(f'#---------------- {ivar}')
    
    fl = sorted(glob.glob(f'data/obs/CM_SAF/{ivar}/*.nc'))[444:540]
    cm_saf = xr.open_mfdataset(fl).sel(time=slice(years, yeare))
    # plt_data = cm_saf[vars[ivar]].weighted(cm_saf[vars[ivar]].time.dt.days_in_month).mean(dim='time').compute()
    plt_data = cm_saf[vars[ivar]].resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='time').compute()
    
    if ivar in ['rsut', 'rlut']:
        plt_data *= (-1)
    elif ivar in ['clwvi', 'clivi']:
        plt_data *= 1000
    elif ivar == 'CDNC':
        plt_data /= 1e6
    
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
            # cm_min=30, cm_max=130, cm_interval1=10, cm_interval2=10,
            cm_min=0, cm_max=400, cm_interval1=20, cm_interval2=40,
            cmap='pink',)
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
    
    for plt_region in plt_regions:
        print(f'plot {plt_region}')
        if plt_region == 'global':
            fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
        elif plt_region == 'c2_domain':
            plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
            fig, ax = regional_plot(
                figsize = np.array([8.8, 8]) / 2.54,
                extent=[min_lon, max_lon, min_lat, max_lat])
        
        plt_mean = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
        cbar_label = f'CM SAF {years}-{yeare} {era5_varlabels[cmip6_era5_var[ivar]]}'
        
        ax.text(0.01, 0.01,
                f'Mean: {str(np.round(plt_mean, 1))}', ha='left', va='bottom',
                transform=ax.transAxes)
        
        plt_mesh1 = ax.pcolormesh(
            plt_data.lon, plt_data.lat, plt_data,
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
        cbar = fig.colorbar(
            plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
            orientation="horizontal", shrink=0.8, ticks=pltticks, extend=extend,
            pad=0.02, fraction=0.13,)
        cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
        fig.savefig(f'figures/3_satellites/3.4_CM_SAF/3.4.0 CM SAF {ivar} {plt_region} am.png')


plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
print(plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values)
print(coslat_weighted_mean(plt_data))

land_ocean_mean(plt_data)


'''
#-------- check
import regionmask
land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
mask = land.mask(plt_data)

print(np.sum(plt_data))
print(np.sum(plt_data.where(~np.isnan(mask))) + np.sum(plt_data.where(np.isnan(mask))))

print(coslat_weighted_mean(plt_data.where(~np.isnan(mask))))
print(coslat_weighted_mean(plt_data.where(np.isnan(mask))))




cm_saf['lwp'].weighted(cm_saf['lwp'].time.dt.days_in_month).mean(dim='time')
cm_saf['lwp_allsky'].weighted(cm_saf['lwp_allsky'].time.dt.days_in_month).mean(dim='time')
'''
# endregion

