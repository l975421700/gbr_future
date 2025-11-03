

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
import calendar
import glob
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity, relative_humidity_from_specific_humidity, dewpoint_from_specific_humidity, equivalent_potential_temperature, potential_temperature
from datetime import datetime, timedelta
from haversine import haversine

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
from matplotlib.patches import Rectangle

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
    time_weighted_mean,
    coslat_weighted_mean,
    coslat_weighted_rmsd,
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

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs, get_modis_latlonvar, get_modis_latlonvars, get_cross_section

# endregion


# region obs vs. sim monthly data Cross Sections
# Memory Used: 31.33GB; Walltime Used: 02:36:31

var2s = [
    'ua', 'va',
    # 'hus', 'ta', 'wap', 'zg', 'theta', 'theta_e', 'hur', 'ua', 'va',
    # 'qcf', 'qcl', 'qr', 'qs',
    # 'qc', 'qt', 'clslw', 'qg', 'ACF', 'BCF', 'TCF',
    ]
dsss = [
    # [('ERA5',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    # [('ERA5',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    [('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    ]
modes = ['original', 'difference'] # 'original', 'difference'

year, month = 2020, 6
starttime = datetime(year, month, 2)
endtime = datetime(year, month, 30, 23, 59)
ptop = 200
plevs_hpa = np.arange(1000, ptop-1e-4, -25)
wi_loc={'lat':-16.2876,'lon':149.962}
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
min_lons, max_lons, min_lats, max_lats = [143.0, 151.94, -20.0, -11.06]
pwidth = 4.4
pheight = 4.4
nrow = 1
fm_bottom = 4.2/(pheight*nrow+5)
fm_top = 1 - 0.8/(pheight*nrow+5)


def std_func(ds, var):
    ds = ds.drop_vars('crs', errors='ignore')
    if 'pressure' in ds.coords:
        ds = ds.expand_dims(dim='pressure', axis=1)
    elif 'pressure' in ds:
        ds = ds.expand_dims(dim={'pressure': [ds['pressure'].values]}, axis=1)
    varname=[varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    return(ds)

for dss in dsss:
  # dss = [('CERES',''),('ERA5',''),('BARRA-R2',1),('BARRA-C2','')],
  print(f'#-------------------------------- {dss}')
  if len(dss) <= 3:
      mpl.rc('font', family='Times New Roman', size=10)
  elif len(dss) == 4:
      mpl.rc('font', family='Times New Roman', size=12)
  elif len(dss) == 5:
      mpl.rc('font', family='Times New Roman', size=14)
  elif len(dss) == 6:
      mpl.rc('font', family='Times New Roman', size=16)
  elif len(dss) >= 7:
      mpl.rc('font', family='Times New Roman', size=18)
  ncol = len(dss)
  fm_left = 2.46 / (pwidth*ncol)
  
  for var2 in var2s:
    # var2 = 'theta_e'
    var1 = cmip6_era5_var[var2]
    print(f'#---------------- {var1} vs. {var2}')
    
    extend2 = 'both'
    if var2 in ['hus']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=16, cm_interval1=1, cm_interval2=2, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG_r')
        # pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
        # pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
        # pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        # pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
        # extend = 'max'
        # pltlevel2 = np.array([-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5])
        # pltticks2 = np.array([-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5])
        # pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        # pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var2 in ['qt']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=16, cm_interval1=1, cm_interval2=2, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG_r')
    elif var2 == 'ta':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=28, cm_interval1=1, cm_interval2=4, cmap='Oranges_r')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG')
    elif var2 == 'ua':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-8, cm_max=32, cm_interval1=2, cm_interval2=4, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG')
    elif var2 == 'va':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=2, cmap='PuOr')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5, cmap='BrBG')
    elif var2 == 'wap':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-0.2, cm_max=0.2, cm_interval1=0.02, cm_interval2=0.08, cmap='PuOr')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.2, cm_max=0.2, cm_interval1=0.02, cm_interval2=0.08, cmap='BrBG')
    elif var2 == 'zg':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=4000, cm_interval1=200, cm_interval2=800, cmap='viridis_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='BrBG')
    elif var2 == 'hur':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-25, cm_max=25, cm_interval1=5, cm_interval2=5, cmap='BrBG_r')
    elif var2 in ['ACF', 'BCF', 'TCF']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=40, cm_interval1=5, cm_interval2=5, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=5, cm_interval2=5, cmap='BrBG_r')
    elif var2 == 'theta':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=20, cm_max=60, cm_interval1=2, cm_interval2=4, cmap='Oranges_r')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG')
    elif var2 == 'theta_e':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=30, cm_max=70, cm_interval1=2, cm_interval2=4, cmap='Oranges_r')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-8, cm_max=8, cm_interval1=1, cm_interval2=2, cmap='BrBG')
    elif var2 in ['qr']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.04, cm_interval1=0.005, cm_interval2=0.01, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.04, cm_max=0.04, cm_interval1=0.005, cm_interval2=0.02, cmap='BrBG_r')
    elif var2 in ['qcl', 'qc']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.1, cm_interval1=0.01, cm_interval2=0.02, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.05, cm_max=0.05, cm_interval1=0.01, cm_interval2=0.02, cmap='BrBG_r')
    elif var2 in ['clslw', 'qcf', 'qs', 'qg']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.01, cm_interval1=0.001, cm_interval2=0.002, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.005, cm_max=0.005, cm_interval1=0.001, cm_interval2=0.002, cmap='BrBG_r')
    else:
        print('Warning: unspecified colorbar')
    
    ofile_ds = f'data/sim/um/combined/{var2} wi {', '.join(x[0] for x in dss)}.pkl'
    if os.path.exists(ofile_ds):
      with open(ofile_ds, 'rb') as f:
        ds = pickle.load(f)
    else:
      ds = {}
      for ids in dss:
        # ids = dss[0]
        print(f'Get {ids}')
        
        if ids[0] in suite_res.keys():
            # ids = ('u-ds722', 1)
            # ids = ('u-ds714', 1)
            isuite = ids[0]
            ires = suite_res[isuite][ids[1]]
            ilabel = f'{suite_label[isuite]}'
            
            fl = sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year}{month:02d}??T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))
            ds_pa = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['pa']].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral['pa']].sel(time=slice(starttime, endtime)).compute()
            if var2 == 'ua':
                ds_data = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral[var2]])[var2stash_ral[var2]].rename({'rho80': 'theta80', 'grid_longitude_cu': 'lon'}).sel(lon=wi_loc['lon'], method='nearest').sel(time=slice(starttime, endtime)).compute()
            elif var2 == 'va':
                ds_data = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral[var2]])[var2stash_ral[var2]].rename({'rho80': 'theta80', 'grid_latitude_cv': 'lat'}).sel(lon=wi_loc['lon'], method='nearest').sel(time=slice(starttime, endtime)).compute()
            elif var2 in var2stash_ral.keys():
                # var2='hus'
                ds_data = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral[var2]].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral[var2]].sel(time=slice(starttime, endtime)).compute()
            elif var2 == 'hur':
                # var2 = 'hur'
                ds_hus = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['hus']].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral['hus']].sel(time=slice(starttime, endtime)).compute()
                ds_ta = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['ta']].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral['ta']].sel(time=slice(starttime, endtime)).compute()
                ds_data = relative_humidity_from_specific_humidity(
                    ds_pa * units.Pa,
                    ds_ta * units.K,
                    ds_hus * units('kg/kg')
                    ).metpy.dequantify() * 100
            elif var2 == 'theta_e':
                # var2 = 'theta_e'
                ds_hus = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['hus']].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral['hus']].sel(time=slice(starttime, endtime)).compute()
                ds_ta = xr.open_mfdataset(fl, preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['ta']].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral['ta']].sel(time=slice(starttime, endtime)).compute()
                ds_dew = dewpoint_from_specific_humidity(
                    ds_pa * units.Pa,
                    ds_hus * units('kg/kg'))
                ds_data = equivalent_potential_temperature(
                    ds_pa * units.Pa,
                    ds_ta * units.K,
                    ds_dew).metpy.dequantify()
            else:
                print(f'Warning: {var2} not found')
            
            ds[ilabel] = interp_to_pressure_levels(ds_data, ds_pa/100, plevs_hpa)
            ds[ilabel] = ds[ilabel].mean(dim='time', skipna=True)
            
            if var2 in ['hus', 'qcf', 'qcl', 'qr', 'qs', 'qc', 'qt', 'clslw', 'qg']:
                ds[ilabel] *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                ds[ilabel] -= zerok
            elif var2 in ['ACF', 'BCF', 'TCF']:
                ds[ilabel] *= 100
            
        elif ids[0] == 'ERA5':
            # ids = ('ERA5', '')
            
            file = f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc'
            if os.path.exists(file):
                ds[ids[0]] = xr.open_dataset(file).rename({'longitude': 'lon', 'latitude':'lat', 'level':'pressure'}).sel(lon=wi_loc['lon'], method='nearest').sortby('lat').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))[var1].mean(dim='time').compute()
            elif var1 == 'theta':
                era5_t = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({'longitude': 'lon', 'latitude':'lat', 'level':'pressure'}).sel(lon=wi_loc['lon'], method='nearest').sortby('lat').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['t'].compute()
                ds[ids[0]] = potential_temperature(
                    era5_t.pressure * units.hPa,
                    era5_t * units.K).metpy.dequantify().mean(dim='time')
            elif var1 == 'theta_e':
                era5_q = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/q/{year}/q_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({'longitude': 'lon', 'latitude':'lat', 'level':'pressure'}).sel(lon=wi_loc['lon'], method='nearest').sortby('lat').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['q'].compute()
                era5_t = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({'longitude': 'lon', 'latitude':'lat', 'level':'pressure'}).sel(lon=wi_loc['lon'], method='nearest').sortby('lat').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['t'].compute()
                era5_dew = dewpoint_from_specific_humidity(
                    era5_q.pressure * units.hPa,
                    era5_q * units('kg/kg'))
                ds[ids[0]] = equivalent_potential_temperature(
                    era5_t.pressure * units.hPa,
                    era5_t * units.K,
                    era5_dew).metpy.dequantify().mean(dim='time')
            else:
                print(f'Warning: {var2} not found')
            
            if var1 in ['t', 'theta', 'theta_e']:
                ds[ids[0]] -= zerok
            elif var1 in ['q', 'ciwc', 'clwc', 'crwc', 'cswc']:
                ds[ids[0]] *= 1000
            elif var1 in ['z', ]:
                ds[ids[0]] /= 9.80665
            
        elif ids[0] == 'BARRA-C2':
            # ids = ('BARRA-C2', '')
            
            fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}[0-9]*[!m]/latest/*{year}{month:02d}.nc'))
            if len(fl) > 0:
                ds[ids[0]] = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds_in: std_func(ds_in, var=var2))
                ds[ids[0]] = ds[ids[0]].sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))[var2].mean(dim='time').compute()
            elif var2 == 'hur':
                ds_hus = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/hus[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='hus')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['hus'].compute()
                ds_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/ta[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='ta')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['ta'].compute()
                ds[ids[0]] = relative_humidity_from_specific_humidity(
                    ds_hus.pressure * units.hPa,
                    ds_ta * units.K,
                    ds_hus * units('kg/kg')
                    ).metpy.dequantify().mean(dim='time') * 100
            elif var2 == 'theta':
                ds_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/ta[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='ta')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['ta'].compute()
                ds[ids[0]] = potential_temperature(
                    ds_ta.pressure * units.hPa,
                    ds_ta * units.K).metpy.dequantify().mean(dim='time')
            elif var2 == 'theta_e':
                ds_hus = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/hus[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='hus')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['hus'].compute()
                ds_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/ta[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='ta')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['ta'].compute()
                ds_dew = dewpoint_from_specific_humidity(
                    ds_hus.pressure * units.hPa,
                    ds_hus * units('kg/kg'))
                ds[ids[0]] = equivalent_potential_temperature(
                    ds_ta.pressure * units.hPa,
                    ds_ta * units.K,
                    ds_dew).metpy.dequantify().mean(dim='time')
            else:
                print(f'Warning: {var2} not found')
            
            if var2 in ['hus']:
                ds[ids[0]] *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                ds[ids[0]] -= zerok
            
        elif ids[0] == 'BARPA-C':
            # ids = ('BARPA-C', '')
            
            fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/{var2}[0-9]*[!m]/latest/*{year}{month:02d}.nc'))
            if len(fl) > 0:
                ds[ids[0]] = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds_in: std_func(ds_in, var=var2))
                ds[ids[0]] = ds[ids[0]].sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))[var2].mean(dim='time').compute()
            elif var2 == 'hur':
                # var2 = 'hur'
                ds_hus = xr.open_mfdataset(sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/hus[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='hus')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['hus'].compute()
                ds_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/ta[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='ta')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['ta'].compute()
                ds[ids[0]] = relative_humidity_from_specific_humidity(
                    ds_hus.pressure * units.hPa,
                    ds_ta * units.K,
                    ds_hus * units('kg/kg')
                    ).metpy.dequantify().mean(dim='time') * 100
            elif var2 == 'theta':
                # var2 = 'theta'
                ds_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/ta[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='ta')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['ta'].compute()
                ds[ids[0]] = potential_temperature(
                    ds_ta.pressure * units.hPa,
                    ds_ta * units.K).metpy.dequantify().mean(dim='time')
            elif var2 == 'theta_e':
                # var2 = 'theta_e'
                ds_hus = xr.open_mfdataset(sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/hus[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='hus')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['hus'].compute()
                ds_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/ta[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds_in: std_func(ds_in, var='ta')).sel(lon=wi_loc['lon'], method='nearest').sel(lat=slice(min_lat, max_lat)).sel(time=slice(starttime, endtime)).sel(pressure=slice(ptop, 1000))['ta'].compute()
                ds_dew = dewpoint_from_specific_humidity(
                    ds_hus.pressure * units.hPa,
                    ds_hus * units('kg/kg'))
                ds[ids[0]] = equivalent_potential_temperature(
                    ds_ta.pressure * units.hPa,
                    ds_ta * units.K,
                    ds_dew).metpy.dequantify().mean(dim='time')
            else:
                print(f'Warning: {var2} not found')
            
            if var2 in ['hus']:
                ds[ids[0]] *= 1000
            elif var2 in ['ta', 'theta', 'theta_e']:
                ds[ids[0]] -= zerok
      
      with open(ofile_ds, 'wb') as f:
        pickle.dump(ds, f)
    
    for imode in modes:
        # imode = 'original'
        # imode='difference'
        print(f'#-------- {imode}')
        
        plt_colnames = list(ds.keys())
        opng = f"figures/4_um/4.1_access_ram3/4.1.1_sim_obs/4.1.1.4 {var2} {', '.join(x.replace('$', '') for x in ds.keys())} {imode} {str(np.round(min_lats, 2))}_{str(np.round(max_lats, 2))} {np.round(wi_loc['lon'], 2)} {year}-{month:02d}.png"
        cbar_label1 = f"{calendar.month_name[month]} {year} {era5_varlabels[var1]}"
        cbar_label2 = f"Difference in {era5_varlabels[var1]}"
        
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=np.array([pwidth*ncol, pheight*nrow+5])/2.54,
            sharey=True, gridspec_kw={'hspace': 0.01, 'wspace': 0.05},)
        
        if imode == 'original':
            for jcol, ids1 in enumerate(ds.keys()):
                print(f'#---- {jcol} {ids1}')
                plt_mesh = axs[jcol].pcolormesh(
                    ds[ids1].lat,
                    ds[ids1].pressure,
                    ds[ids1].transpose('pressure', 'lat'),
                    norm=pltnorm, cmap=pltcmp, zorder=1)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.26, fm_bottom-0.25, 0.48, fm_bottom/10]))
            cbar.ax.set_xlabel(cbar_label1)
        elif imode=='difference':
            # imode='difference'
            ids2 = list(ds.keys())[0]
            plt_mesh = axs[0].pcolormesh(
                ds[ids2].lat,
                ds[ids2].pressure,
                ds[ids2].transpose('pressure', 'lat'),
                norm=pltnorm, cmap=pltcmp, zorder=1)
            for jcol in range(ncol-1):
                ids1 = list(ds.keys())[jcol+1]
                print(f'#---- {jcol+1} {ids1} {ids2}')
                # common_p = np.intersect1d(ds[ids1].pressure, ds[ids2].pressure)
                plt_data = ds[ids1].transpose('pressure', 'lat') - ds[ids2].transpose('pressure', 'lat').interp(lat=ds[ids1].lat, pressure=ds[ids1].pressure)
                plt_mesh2 = axs[jcol+1].pcolormesh(
                    plt_data.lat,
                    plt_data.pressure,
                    plt_data,
                    norm=pltnorm2, cmap=pltcmp2, zorder=1)
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.03, fm_bottom-0.25, 0.44, fm_bottom/10]))
            cbar.ax.set_xlabel(cbar_label1)
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.53, fm_bottom-0.25, 0.44, fm_bottom/10]))
            cbar2.ax.set_xlabel(cbar_label2)
        
        for jcol in range(ncol):
            axs[jcol].text(
                0, 1.02,
                f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
            
            axs[jcol].invert_yaxis()
            axs[jcol].set_ylim(1000, ptop)
            axs[jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
            
            axs[jcol].set_xlim(min_lats, max_lats)
            axs[jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
            axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            
            axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
        
        axs[0].set_ylabel(r'Pressure [$hPa$]')
        # axs[3].set_xlabel(f'Meridional cross section along Willis Island', labelpad=8)
        fig.text(0.5, fm_bottom-0.14, f'Meridional cross section along Willis Island', va='center', ha='center')
        
        fig.subplots_adjust(left=fm_left, right=0.995, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)








'''
import time

start = time.time()
xr.open_dataset(fl[13])[var2stash_ral['pa']][:, :, :, 0].compute()
# xr.open_dataset(fl[13])[var2stash_ral['pa']].sel(grid_longitude_t=wi_loc['lon'], method='nearest').compute()
# xr.open_dataset(fl[13]).pipe(preprocess_umoutput)[var2stash_ral['pa']].sel(lon=wi_loc['lon'], method='nearest').compute()
end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")


xr.open_mfdataset(fl[8:10], preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['pa']].sel(lon=wi_loc['lon'], method='nearest'))[var2stash_ral['pa']].sel(time=slice(starttime, endtime)).compute()

xr.open_mfdataset(fl[8:10], preprocess=lambda ds_in: ds_in.pipe(preprocess_umoutput)[var2stash_ral['pa']].sel(lon=wi_loc['lon'], method='nearest'), combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override')[var2stash_ral['pa']].sel(time=slice(starttime, endtime)).compute()


#-------------------------------- check
itime = 32
ilat = 40

data1 = np.interp(plevs_hpa, ds_pa[itime, ::-1, ilat]/100, ds_data[itime, ::-1, ilat], left=np.nan, right=np.nan).astype(ds_data.dtype)
data2 = ds[ilabel][itime, ilat, :]
print((data1 == data2).all())

haversine(cs_start, cs_end, unit='km')
'''
# endregion

