

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
from metpy.calc import pressure_to_height_std, geopotential_to_height, potential_temperature, equivalent_potential_temperature, specific_humidity_from_mixing_ratio, wind_components, relative_humidity_from_specific_humidity, dewpoint_from_specific_humidity
from metpy.units import units
import metpy.calc as mpcalc
import pickle
from datetime import datetime
from skimage.measure import block_reduce
from netCDF4 import Dataset
import xesmf as xe
import healpy as hp

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
import matplotlib.ticker as ticker

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
    panel_labels,
    era5_varlabels,
    era5_varlabels_sim,
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
    get_nn_lon_lat_index,
)

from calculations import (
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region animate vertical profiles
# Memory Used: 26.1GB; Walltime Used: 00:44:14

year, month = 2020, 6
colvars = ['hus', 'hur', 'ta', 'theta', 'theta_e', 'ua', 'va']
dss = ['ERA5', 'BARRA-R2', 'BARRA-C2', 'UM']
top_pressure = 700
station = 'Willis Island'
stationNumber = '94299'
slat = -16.288
slon = 149.965

rename = {'pressure_hPa': 'pressure',
          'geopotential height_m': 'height',
          'temperature_C': 'ta',
          'dew point temperature_C': 'dewpoint',
          'ice point temperature_C': 'icepoint',
          'relative humidity_%': 'hur',
          'mixing ratio_g/kg': 'mixr',
          'wind direction_degree': 'direction',
          'wind speed_m/s': 'speed'}

omp4=f'figures/4_um/4.0_barra/4.0.6_vertical_profiles/4.0.6.1 {year}-{month} {', '.join(sorted(colvars))} in {', '.join(dss)} at {station}.mp4'
radiosonde_fl = sorted(glob.glob(f'data/obs/radiosonde/Wyoming/{stationNumber}/{year}{month:02d}*-{stationNumber}.csv'))
minmaxlim = {'hus': [0, 20],    'hur':   [0, 100],
             'ta':  [5, 30],    'theta': [20, 50],  'theta_e': [30, 80],
             'ua':  [-20, 20],  'va':    [-20, 20]}
nrow = 1
ncol = len(colvars)
pwidth  = 3
pheight = 6.6
fm_left = 1.5/(pwidth*ncol+4)
fm_right = 1-2.5/(pwidth*ncol+4)
fm_bottom = 1.2/(pheight*nrow+2.4)
fm_top = 1 - 1.2/(pheight*nrow+2.4)


if 'UM' in dss:
    hk_um_z10_3H = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z10.zarr')

def std_func(ds_in, var):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    if var == 'hus':
        ds = ds * 1000
    elif var == 'ta':
        ds = ds - zerok
    return(ds)

ds = {}
for ids in dss: ds[ids] = {}
for var2 in ['hus', 'ta', 'ua', 'va', 'hur', 'theta', 'theta_e']:
    var1 = cmip6_era5_var[var2]
    # var2='hus'; var1='q'
    print(f'#---------------- {var2} vs. {var1} {era5_varlabels_sim[var1]}')
    
    if var2 in ['hus', 'ta', 'ua', 'va']:
        print('get ERA5')
        ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(level=slice(top_pressure, 1000)).sel(longitude=slon, latitude=slat, method='nearest').rename({'level': 'pressure'})
        if var2 in ['hus']:
            ds['ERA5'][var2] *= 1000
        elif var2 in ['ta']:
            ds['ERA5'][var2] -= zerok
        
        print('get BARRA-R2')
        ds['BARRA-R2'][var2] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}[0-9]*[!m]/latest/{var2}[0-9]*{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var=var2))[var2].sel(pressure=slice(top_pressure, 1000)).sel(lon=slon, lat=slat, method='nearest')
        
        print('get BARRA-C2')
        ds['BARRA-C2'][var2] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}[0-9]*[!m]/latest/{var2}[0-9]*{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var=var2))[var2].sel(pressure=slice(top_pressure, 1000)).sel(lon=slon, lat=slat, method='nearest')
        
        print('get UM')
        if 'UM' in dss:
            ds['UM'][var2] = hk_um_z10_3H[var2].sel(cell=hp.ang2pix(2**10, slon, slat, nest=True, lonlat=True), method='nearest').sel(pressure=slice(top_pressure, 1000), time=slice(f'{year}-{month}', f'{year}-{month}'))
            if var2 in ['hus']:
                ds['UM'][var2] *= 1000
            elif var2 in ['ta']:
                ds['UM'][var2] -= zerok
    elif var2=='hur':
        for ids in dss:
            print(ids)
            ds[ids]['hur'] = relative_humidity_from_specific_humidity(
                ds[ids]['ta'].pressure * units.hPa,
                ds[ids]['ta'] * units.degC,
                ds[ids]['hus'] /1000) * 100
    elif var2=='theta':
        for ids in dss:
            print(ids)
            ds[ids]['theta'] = potential_temperature(
                ds[ids]['ta'].pressure * units.hPa,
                ds[ids]['ta'] * units.degC).metpy.dequantify() - zerok
    elif var2=='theta_e':
        for ids in dss:
            print(ids)
            dewpoint = dewpoint_from_specific_humidity(
                ds[ids]['hus'].pressure * units.hPa,
                ds[ids]['hus'] * units('g/kg'))
            ds[ids]['theta_e'] = equivalent_potential_temperature(
                ds[ids]['ta'].pressure * units.hPa,
                ds[ids]['ta'] * units.degC,
                dewpoint).metpy.dequantify() - zerok

fig, axs = plt.subplots(
    nrow,ncol,sharey=True,
    figsize=np.array([pwidth*ncol+4,pheight*nrow+2.4])/2.54,
    gridspec_kw={'hspace': 0.01, 'wspace': 0.15})

axs[0].invert_yaxis()
axs[0].set_ylim(1030, top_pressure)
axs[0].set_ylabel(r'Pressure [$hPa$]')

for jcol in range(ncol):
    axs[jcol].set_xlabel(era5_varlabels_sim[cmip6_era5_var[colvars[jcol]]])
    if colvars[jcol]=='theta_e':
        axs[jcol].set_xticks(np.array([20, 40, 60, 80]))
    elif colvars[jcol] in ['ua', 'va']:
        axs[jcol].set_xticks(np.array([-10, 0, 10]))
    axs[jcol].set_xlim(minmaxlim[colvars[jcol]])
    axs[jcol].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axs[jcol].grid(which='both', lw=0.5, alpha=0.5, ls='--')
    axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]})',
                   ha='left', va='bottom', transform=axs[jcol].transAxes)

plt_objs = []
def update_frames(itime):
    # itime=0
    time1 = time.perf_counter()
    global plt_objs
    for plt_obj in plt_objs:
        try:
            plt_obj.remove()
        except ValueError:
            pass
    plt_objs = []
    
    ifile = radiosonde_fl[itime]
    day = int(ifile.split('/')[-1].split('-')[0][6:8])
    hour = int(ifile.split('/')[-1].split('-')[0][8:10])
    date = datetime(year, month, day, hour)
    
    df = pd.read_csv(ifile,skipfooter=1,engine='python').rename(columns=rename)
    df['theta'] = potential_temperature(
        df.pressure.values * units.hPa,
        df.ta.values * units.degC).to('degC').magnitude
    df['theta_e'] = equivalent_potential_temperature(
        df.pressure.values * units.hPa,
        df.ta.values * units.degC,
        df.dewpoint.values * units.degC).to('degC').magnitude
    df['hus'] = specific_humidity_from_mixing_ratio(
        df.mixr.values * units('g/kg')).magnitude
    ua, va = wind_components(
        df.speed.values * units('m/s'),
        df.direction.values * units.deg)
    df['ua'], df['va'] = ua.magnitude, va.magnitude
    df = df[df['pressure']>=top_pressure]
    
    plt_lines = []
    for jcol in range(ncol):
        plt_lines += axs[jcol].plot(
            df[colvars[jcol]], df['pressure'],
            '.-', lw=0.75, markersize=1,  c='k', label='Radiosonde')
        for ids in dss:
            try:
                plt_lines += axs[jcol].plot(
                    ds[ids][colvars[jcol]].sel(time=date, method='nearest'),
                    ds[ids][colvars[jcol]].pressure,
                    '.-', lw=0.75, markersize=4, c=ds_color[ids], label=ids)
            except RuntimeError:
                pass
    
    if itime==0:
        plt.legend(ncol=1, frameon=False, loc='center left', handlelength=1,
           bbox_to_anchor=(fm_right, 0.5), handletextpad=0.5)
    
    plt_text = fig.text(0.5, fm_top + 0.07, f'{str(date)[:13]}:00 UTC at {station}', ha='center', va='bottom')
    plt_objs += plt_lines + [plt_text]
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} s')
    return(plt_objs)

fig.subplots_adjust(fm_left, fm_bottom, fm_right, fm_top)
ani = animation.FuncAnimation(
    fig, update_frames, frames=len(radiosonde_fl), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4,progress_callback=lambda iframe,n:print(f'Frame {iframe}/{n-1}'))




# endregion

