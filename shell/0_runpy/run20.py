

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
mpl.rc('font', family='Times New Roman', size=12)
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
    get_nn_lon_lat_index,
)

from calculations import (
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

from um_postprocess import (
    preprocess_umoutput,
    stash2var, stash2var_gal, stash2var_ral,
    var2stash, var2stash_gal, var2stash_ral,
    suite_res, suite_label,
    interp_to_pressure_levels)

# endregion


# region plot mean vertical profiles

# options
colvars = ['hus', 'hur', 'ta', 'theta', 'theta_e', 'ua', 'va']
dss = [('Radiosonde',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)]
# [('Radiosonde',''), ('ERA5',''), ('BARRA-R2',''), ('BARRA-C2','')]
# [('Radiosonde',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)]
# [('Radiosonde',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)]

# settings
year, month = 2020, 6
starttime = datetime(year, month, 2)
endtime = datetime(year, month, 30, 23, 59)
ptop = 200
plevs_hpa = np.arange(1000, ptop-1e-4, -25)
target_pressures = np.arange(1000, ptop-1e-4, -1)
station = 'Willis Island'
stationNumber = '94299'
slat = -16.288
slon = 149.965

nrow = 1
ncol = len(colvars)
pwidth  = 4.4
pheight = 5.5
fm_left = 2.5/(pwidth*ncol)
fm_right = 0.99
fm_bottom = 4/(pheight*nrow+4)
fm_top = 0.96

if len(colvars) <= 3:
    mpl.rc('font', family='Times New Roman', size=10)
elif len(colvars) == 4:
    mpl.rc('font', family='Times New Roman', size=12)
elif len(colvars) == 5:
    mpl.rc('font', family='Times New Roman', size=14)
elif len(colvars) == 6:
    mpl.rc('font', family='Times New Roman', size=16)
elif len(colvars) >= 7:
    mpl.rc('font', family='Times New Roman', size=18)

def std_func(ds_in, var):
    ds = ds_in.drop_vars('crs', errors='ignore')
    if 'pressure' in ds.coords:
        ds = ds.expand_dims(dim='pressure', axis=1)
    elif 'pressure' in ds:
        ds = ds.expand_dims(dim={'pressure': [ds['pressure'].values]}, axis=1)
    varname=[varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    return(ds)

ofile_ds = f'data/sim/um/combined/{', '.join(x for x in colvars)} wi {', '.join(x[0] for x in dss)}.pkl'
if os.path.exists(ofile_ds):
  with open(ofile_ds, 'rb') as f:
    ds = pickle.load(f)
else:
  ds = {}
  for ids in dss:
    print(f'#---------------- Get {ids}')
    if ids[0] == 'Radiosonde':
        # ids = ('Radiosonde', '')
        
        fl = sorted(glob.glob(f'data/obs/radiosonde/Wyoming/{stationNumber}/{year}{month:02d}*.csv'))
        for idx, ifile in enumerate(fl):
            # idx=-1; ifile = fl[-1]
            # print(f'#---------------- {idx} {ifile}')
            
            df = pd.read_csv(ifile,skipfooter=1,engine='python').rename(columns={'pressure_hPa': 'pressure', 'geopotential height_m': 'height', 'temperature_C': 'ta', 'dew point temperature_C': 'dewpoint', 'ice point temperature_C': 'icepoint', 'relative humidity_%': 'hur', 'mixing ratio_g/kg': 'mixr', 'wind direction_degree': 'direction', 'wind speed_m/s': 'speed'})
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
            
            day = int(ifile.split('/')[-1].split('-')[0][6:8])
            hour = int(ifile.split('/')[-1].split('-')[0][8:10])
            date = datetime(year, month, day, hour)
            
            xdf = xr.Dataset(
                {var: (['time', 'pressure'], np.zeros((1, len(target_pressures))))
                 for var in colvars},
                coords={'time': [date], 'pressure': target_pressures})
            for var in colvars:
                xdf[var][:] = np.interp(np.log(target_pressures),
                                        np.log(df['pressure'].values[::-1]),
                                        df[var].values[::-1])
            
            if idx==0:
                ds[ids[0]] = xdf.copy()
            else:
                ds[ids[0]] = xr.concat([ds[ids[0]], xdf], dim='time')
    elif ids[0] == 'ERA5':
        # ids = ('ERA5', '')
        for var2 in ['hus', 'ta', 'hur', 'theta', 'theta_e', 'ua', 'va']:
            var1 = cmip6_era5_var[var2]
            # var2='hus'; var1='q'
            print(f'#-------- {var2}')
            
            ifile = f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc'
            if os.path.exists(ifile):
                era5_ds = xr.open_dataset(ifile)[var1].rename({'level': 'pressure'}).sel(pressure=slice(ptop, 1000), time=slice(starttime, endtime)).sel(longitude=slon, latitude=slat, method='nearest').compute()
            elif var2 == 'theta':
                era5_ds = potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC).metpy.dequantify().compute()
            elif var2 == 'theta_e':
                dewpoint = dewpoint_from_specific_humidity(
                    ds[ids[0]]['hus'].pressure * units.hPa,
                    ds[ids[0]]['hus'] * units('g/kg')).compute()
                era5_ds = equivalent_potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    dewpoint).metpy.dequantify().compute()
            else:
                print(f'Warning: {var2} not found')
            
            if var1 in ['t', 'theta', 'theta_e']:
                era5_ds -= zerok
            elif var1 in ['q', 'ciwc', 'clwc', 'crwc', 'cswc']:
                era5_ds *= 1000
            elif var1 in ['z', ]:
                era5_ds /= 9.80665
            
            if not ids[0] in ds.keys():
                ds[ids[0]] = era5_ds.rename(var2).compute().copy()
            else:
                ds[ids[0]] = xr.merge([ds[ids[0]], era5_ds.rename(var2)])
    elif ids[0] == 'BARRA-R2':
        # ids = ('BARRA-R2', '')
        for var2 in ['hus', 'ta', 'hur', 'theta', 'theta_e', 'ua', 'va']:
            # var2='hur'
            print(f'#-------- {var2}')
            
            fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}[0-9]*[!m]/latest/*{year}{month:02d}.nc'))
            if len(fl) > 0:
                r2_ds = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds: std_func(ds, var=var2))[var2].sel(pressure=slice(ptop, 1000), time=slice(starttime, endtime)).sel(lon=slon, lat=slat, method='nearest').compute()
            elif var2 == 'hur':
                r2_ds = (relative_humidity_from_specific_humidity(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    ds[ids[0]]['hus'] / 1000 * units('kg/kg')).metpy.dequantify() * 100).compute()
            elif var2 == 'theta':
                r2_ds = potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC).metpy.dequantify().compute()
            elif var2 == 'theta_e':
                dewpoint = dewpoint_from_specific_humidity(
                    ds[ids[0]]['hus'].pressure * units.hPa,
                    ds[ids[0]]['hus'] * units('g/kg')).compute()
                r2_ds = equivalent_potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    dewpoint).metpy.dequantify().compute()
            else:
                print(f'Warning: {var2} not found')
            
            if var2 in ['hus']:
                r2_ds *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                r2_ds -= zerok
            
            if not ids[0] in ds.keys():
                ds[ids[0]] = r2_ds.rename(var2).compute().copy()
            else:
                ds[ids[0]] = xr.merge([ds[ids[0]], r2_ds.rename(var2)])
    elif ids[0] == 'BARRA-C2':
        # ids = ('BARRA-C2', '')
        for var2 in ['hus', 'ta', 'hur', 'theta', 'theta_e', 'ua', 'va']:
            # var2='hus'
            print(f'#-------- {var2}')
            
            fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}[0-9]*[!m]/latest/*{year}{month:02d}.nc'))
            if len(fl) > 0:
                c2_ds = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds: std_func(ds, var=var2))[var2].sel(pressure=slice(ptop, 1000), time=slice(starttime, endtime)).sel(lon=slon, lat=slat, method='nearest').compute()
            elif var2 == 'hur':
                c2_ds = (relative_humidity_from_specific_humidity(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    ds[ids[0]]['hus'] / 1000 * units('kg/kg')).metpy.dequantify() * 100).compute()
            elif var2 == 'theta':
                c2_ds = potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC).metpy.dequantify().compute()
            elif var2 == 'theta_e':
                dewpoint = dewpoint_from_specific_humidity(
                    ds[ids[0]]['hus'].pressure * units.hPa,
                    ds[ids[0]]['hus'] * units('g/kg')).compute()
                c2_ds = equivalent_potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    dewpoint).metpy.dequantify().compute()
            else:
                print(f'Warning: {var2} not found')
            
            if var2 in ['hus']:
                c2_ds *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                c2_ds -= zerok
            
            if not ids[0] in ds.keys():
                ds[ids[0]] = c2_ds.rename(var2).compute().copy()
            else:
                ds[ids[0]] = xr.merge([ds[ids[0]], c2_ds.rename(var2)])
    elif ids[0] == 'BARPA-C':
        # ids = ('BARPA-C', '')
        for var2 in ['hus', 'ta', 'hur', 'theta', 'theta_e', 'ua', 'va']:
            # var2='hus'
            print(f'#-------- {var2}')
            
            fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/{var2}[0-9]*[!m]/latest/*{year}{month:02d}.nc'))
            if len(fl) > 0:
                pc_ds = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds: std_func(ds, var=var2))[var2].sel(pressure=slice(ptop, 1000), time=slice(starttime, endtime)).sel(lon=slon, lat=slat, method='nearest').compute()
            elif var2 == 'hur':
                pc_ds = (relative_humidity_from_specific_humidity(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    ds[ids[0]]['hus'] / 1000 * units('kg/kg')).metpy.dequantify() * 100).compute()
            elif var2 == 'theta':
                pc_ds = potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC).metpy.dequantify().compute()
            elif var2 == 'theta_e':
                dewpoint = dewpoint_from_specific_humidity(
                    ds[ids[0]]['hus'].pressure * units.hPa,
                    ds[ids[0]]['hus'] * units('g/kg')).compute()
                pc_ds = equivalent_potential_temperature(
                    ds[ids[0]]['ta'].pressure * units.hPa,
                    ds[ids[0]]['ta'] * units.degC,
                    dewpoint).metpy.dequantify().compute()
            else:
                print(f'Warning: {var2} not found')
            
            if var2 in ['hus']:
                pc_ds *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                pc_ds -= zerok
            
            if not ids[0] in ds.keys():
                ds[ids[0]] = pc_ds.rename(var2).compute().copy()
            else:
                ds[ids[0]] = xr.merge([ds[ids[0]], pc_ds.rename(var2)])
    elif ids[0] in suite_res.keys():
        # ids = ('u-ds714', 1)
        # ids = ('u-ds722',1)
        isuite = ids[0]
        ires = suite_res[isuite][ids[1]]
        ilabel = f'{suite_label[isuite]}'
        
        fl = sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year}{month:02d}??T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))
        ram3_ds = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput).sel(lon=slon, lat=slat, grid_longitude_cu=slon, grid_latitude_cv=slat, method='nearest')).sel(time=slice(starttime, endtime))
        ram3_pa = ram3_ds[var2stash_ral['pa']].compute()
        
        for var2 in ['hus', 'ta', 'hur', 'theta', 'theta_e', 'ua', 'va']:
            # var2='hus'
            print(f'#-------- {var2}')
            
            if var2 in var2stash_ral.keys():
                ram3_data = ram3_ds[var2stash_ral[var2]].compute()
                ram3_output = interp_to_pressure_levels(ram3_data, ram3_pa/100, plevs_hpa)
            elif var2 == 'hur':
                ram3_output = (relative_humidity_from_specific_humidity(
                    ds[ilabel]['ta'].pressure * units.hPa,
                    ds[ilabel]['ta'] * units.degC,
                    ds[ilabel]['hus'] / 1000 * units('kg/kg')).metpy.dequantify() * 100).compute()
            elif var2 == 'theta_e':
                dewpoint = dewpoint_from_specific_humidity(
                    ds[ilabel]['hus'].pressure * units.hPa,
                    ds[ilabel]['hus'] * units('g/kg')).compute()
                ram3_output = equivalent_potential_temperature(
                    ds[ilabel]['ta'].pressure * units.hPa,
                    ds[ilabel]['ta'] * units.degC,
                    dewpoint).metpy.dequantify().compute()
            else:
                print(f'Warning: {var2} not found')
        
            if var2 in ['hus', 'qcf', 'qcl', 'qr', 'qs', 'qc', 'qt', 'clslw', 'qg']:
                ram3_output *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                ram3_output -= zerok
            elif var2 in ['ACF', 'BCF', 'TCF']:
                ram3_output *= 100
            
            if not ilabel in ds.keys():
                ds[ilabel] = ram3_output.rename(var2).compute().copy()
            else:
                ds[ilabel] = xr.merge([ds[ilabel], ram3_output.rename(var2)])
  
  with open(ofile_ds, 'wb') as f:
    pickle.dump(ds, f)


opng=f"figures/4_um/4.0_barra/4.0.6_site_analysis/4.0.6.2 {', '.join(sorted(colvars))} in {', '.join(ids[0] for ids in dss)} at {station} {year}-{month:02d}.png"

fig, axs = plt.subplots(
    nrow, ncol, sharey=True,
    figsize=np.array([pwidth*ncol,pheight*nrow+4])/2.54,
    gridspec_kw={'hspace': 0.01, 'wspace': 0.15})

for jcol, var2 in enumerate(colvars):
    print(f'#-------- {jcol} {var2}')
    
    for ids in ds.keys():
        print(f'#---- {ids}')
        
        if ids == 'Radiosonde':
            axs[jcol].plot(
                ds[ids][var2].mean(dim='time'), ds[ids][var2].pressure,
                '.-', lw=0.75, markersize=1, c=ds_color[ids], label=ids)
        elif not 'Radiosonde' in ds.keys():
            axs[jcol].plot(
                ds[ids][var2].mean(dim='time'), ds[ids][var2].pressure,
                '.-', lw=0.75, markersize=4, c=ds_color[ids], label=ids)
        else:
            axs[jcol].plot(
                ds[ids][var2].sel(time=ds['Radiosonde'][var2].time, method='nearest').mean(dim='time'),
                ds[ids][var2].pressure,
                '.-', lw=0.75, markersize=4, c=ds_color[ids], label=ids)
    
    axs[jcol].invert_yaxis()
    axs[jcol].set_ylim(1000, ptop)
    axs[jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
    
    axs[jcol].text(
        0.5, -0.35,
        f'({string.ascii_lowercase[jcol]}) {era5_varlabels[cmip6_era5_var[var2]]}',
        ha='center', va='bottom', transform=axs[jcol].transAxes)
    axs[jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
    axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')

axs[0].set_ylabel(r'Pressure [$hPa$]')

plt.legend(ncol=ncol, frameon=False, loc='center', handlelength=0.8,
           handletextpad=0.5, columnspacing=1,
           bbox_to_anchor=(0.5, 0.04), bbox_transform=fig.transFigure)

fig.text(0.5, fm_bottom-0.28, f'{calendar.month_name[month]} {year} at {station}', va='center', ha='center')
fig.subplots_adjust(fm_left, fm_bottom, fm_right, fm_top)
fig.savefig(opng)



'''
if 'UM' in dss:
    izlev=9
    hk_um_z10_3H = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT3H.z{izlev}.zarr')
        
        print('get UM')
        if 'UM' in dss:
            ds['UM'][var2] = hk_um_z10_3H[var2].sel(cell=hp.ang2pix(2**izlev, slon, slat, nest=True, lonlat=True), method='nearest').sel(pressure=slice(ptop, 1000), time=slice(f'{year}-{month}', f'{year}-{month}')).compute()
            if var2 in ['hus']:
                ds['UM'][var2] *= 1000
            elif var2 in ['ta']:
                ds['UM'][var2] -= zerok


#-------------------------------- check
izlev = 9
print(hp.ang2pix(2**izlev, slon, slat, nest=True, lonlat=True))
theta = np.radians(90.0 - slat)
phi = np.radians(slon)
print(hp.ang2pix(2**izlev, theta, phi, nest=True))

print(get_nn_lon_lat_index(2**izlev, [slon], [slat]).values)


#-------------------------------- check

opng=f'figures/test.png'
minmaxlim = {'hus': [0, 20],    'hur':   [0, 100],
             'ta':  [5, 30],    'theta': [20, 60],  'theta_e': [30, 80],
             'ua':  [-20, 20],  'va':    [-20, 20]}

nrow = 1
ncol = len(colvars)
pwidth  = 3
pheight = 6.6
fm_left = 1.5/(pwidth*ncol+4)
fm_right = 1-2.5/(pwidth*ncol+4)
fm_bottom = 1.2/(pheight*nrow+2.4)
fm_top = 1 - 1.2/(pheight*nrow+2.4)

year, month, day, hour = 2020, 6, 2, 11
date = datetime(year, month, day, hour)

fig, axs = plt.subplots(
    nrow,ncol,sharey=True,
    figsize=np.array([pwidth*ncol+4,pheight*nrow+2.4])/2.54,
    gridspec_kw={'hspace': 0.01, 'wspace': 0.15})

axs[0].invert_yaxis()
axs[0].set_ylim(1030, ptop)
axs[0].set_ylabel(r'Pressure [$hPa$]')

for jcol in range(ncol):
    axs[jcol].plot(
        xdfs[colvars[jcol]].sel(time=date, method='nearest'), xdfs.pressure,
        '.-', lw=0.75, markersize=1,  c='k', label='Radiosonde')
    for ids in dss:
        # print(ids)
        axs[jcol].plot(
            ds[ids][colvars[jcol]].sel(time=date, method='nearest'),
            ds[ids][colvars[jcol]].pressure,
            '.-', lw=0.75, markersize=4, c=ds_color[ids], label=ids)
    
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

plt.legend(ncol=1, frameon=False, loc='center left', handlelength=1,
           bbox_to_anchor=(fm_right, 0.5), handletextpad=0.5)

fig.suptitle(f'{str(date)[:13]}:00 UTC at {station}')
fig.subplots_adjust(fm_left, fm_bottom, fm_right, fm_top)
fig.savefig(opng)
plt.close()



'''
# endregion

