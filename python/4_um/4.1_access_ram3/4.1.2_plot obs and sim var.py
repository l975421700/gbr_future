

# qsub -I -q normal -P v46 -l walltime=6:00:00,ncpus=1,mem=96GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60+gdata/py18+gdata/rv74


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


# region obs vs. sim timestamp

year, month, day, hour = 2020, 6, 2, 3
var2s = ['rsut']
# 'rsut', 'rlut', 'cll', 'clm', 'clh', 'clt', 'clivi', 'clwvi', 'blh'
modes = ['original'] # 'original', 'difference'
dsss = [
    # [('MODIS Aqua',''),('u-dq700',1),('u-dr041',1),('u-dq799',1),('u-dr145',1)],#res+CDNC
    # [('MODIS Aqua',''),('u-dq700',1),('u-dr041',1),('u-dr091',1),('u-dr147',1)],#Sres+CDNC
    # [('CERES',''),('u-dq700',1),('u-dr041',1),('u-dq799',1),('u-dr145',1)],#res+CDNC
    # [('CERES',''),('u-dq700',1),('u-dr041',1),('u-dr091',1),('u-dr147',1)],#Sres+CDNC
    
    # [('CERES',''),('BARRA-C2',''),('BARPA-C',''),('u-dq700',1)],#control
    # [('BARRA-C2',''),('u-dq700',1)],#control
    # [('CERES',''),('ERA5',''),('BARRA-R2',''),('BARRA-C2',''),('BARPA-R',''),('BARPA-C',''),('u-dq700',1)],#control
    
    # [('CERES',''),('u-dq700',1),('u-dr105',1),('u-dr107',1)],#levs
    # [('CERES',''),('u-dq700',1),('u-dq912',1)],#LD
    # [('CERES',''),('u-dq700',1),('u-ds922',1),('u-ds921',1),('u-dt038',1),('u-dt039',1),('u-dt040',1)],#spin-up
    # [('u-dq700',1),('u-dr107',1),('u-dq912',1),('u-dt039',1)],#setups
    
    # [('CERES',''),('u-dt042',1),('u-dq700',1),('u-dr040',1),('u-dr041',1),('u-dt020',1)],#CDNC
    # [('u-dq700',1),('u-dr040',1),('u-dr041',1)],#CDNC
    
    # [('CERES',''),('u-dq700',1),('u-dr095',1),('u-dr093',1),('u-dr091',1)],#Sres
    [('CERES',''),('u-dq700',1),('u-dq788',1),('u-dq911',1),('u-dq799',1)],#res
    # [('CERES',''),('u-dq700',1),('u-dr041',1),('u-dr091',1),('u-dr147',1)],#Sres+CDNC
    # [('CERES',''),('u-dq700',1),('u-dr041',1),('u-dq799',1),('u-dr145',1)],#res+CDNC
    
    # [('CERES',''),('u-dq700',1),('u-dr789',1),('u-dr922',1)],#clinho
    # [('CERES',''),('u-dq700',1),('u-dr108',1),('u-dr109',1)],#param
    # [('CERES',''),('u-dq700',1),('u-ds719',1)], #shortTS
    
    # Long run
    # [('u-ds714',1),('u-ds718',1),('u-ds717',1)],#CDNC
    # [('u-ds714',1),('u-ds722',1),('u-ds724',1),('u-ds726',1)],#res
    # [('u-ds714',1),('u-ds728',1),('u-ds730',1),('u-ds732',1)],#Sres
    # [('u-ds714',1),('u-dq788',1),('u-dq911',1),('u-dq799',1),('u-ds722',1),('u-ds724',1),('u-ds726',1)],#res
    # [('u-ds714',1),('u-dr095',1),('u-dr093',1),('u-dr091',1),('u-ds728',1),('u-ds730',1),('u-ds732',1)],#Sres
]

ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour
ptime = pd.Timestamp(year,month,day,hour) - pd.Timedelta('1h')
year0, month0, day0, hour0 = ptime.year, ptime.month, ptime.day, ptime.hour

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth = 4.4
nrow = 1
regridder = {}

for dss in dsss:
  # dss = [('CERES',''),('BARRA-C2',''),('u-dq700',1)]
  print(f'#-------------------------------- {dss}')
  if len(dss) <= 3:
      mpl.rc('font', family='Times New Roman', size=10)
  elif len(dss) == 4:
      mpl.rc('font', family='Times New Roman', size=12)
  elif len(dss) == 5:
      mpl.rc('font', family='Times New Roman', size=14)
  elif len(dss) >= 6:
      mpl.rc('font', family='Times New Roman', size=16)
  for var2 in var2s:
    # var2 = 'rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#---------------- {var1} vs. {var2}')
    
    if var2 == 'prw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['cll', 'clm', 'clh', 'clt']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='viridis')
        extend = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-100,cm_max=100,cm_interval1=10,cm_interval2=20,cmap='BrBG_r')
        extend2 = 'neither'
    elif var2 in ['clwvi']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=600, cm_interval1=50, cm_interval2=100, cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=100,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clivi']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=600, cm_interval1=50, cm_interval2=100, cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=100,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['cwp']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=800, cm_interval1=50, cm_interval2=100, cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=100,cmap='BrBG_r')
        extend2 = 'both'
    elif var2=='pr':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=8, cm_interval1=1, cm_interval2=1, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['evspsbl', 'evspsblpot']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['hfls']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-600, cm_max=300, cm_interval1=50, cm_interval2=100, cmap='PRGn', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['hfss']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-400, cm_max=400, cm_interval1=50, cm_interval2=100, cmap='PRGn')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['sfcWind']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['tas', 'ts']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['das']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['huss']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=0.5,cm_interval2=1,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['hurs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues_r')
        extend = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['rsut']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-800, cm_max=0, cm_interval1=50, cm_interval2=100, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-300,cm_max=300,cm_interval1=50,cm_interval2=100,cmap='BrBG')
        # pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        #     cm_min=-300, cm_max=-50, cm_interval1=25, cm_interval2=50, cmap='Greens')
        # extend = 'both'
        # pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        #     cm_min=-200,cm_max=200,cm_interval1=25,cm_interval2=50,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rsutcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-200, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlut', 'rlutcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-350, cm_max=-150, cm_interval1=10, cm_interval2=20, cmap='Greens')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40,cm_max=40,cm_interval1=10,cm_interval2=10,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlds', 'rldscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        extend2 = 'both'
    elif var2=='rsdt':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG',)
        extend2 = 'both'
    elif var2 in ['rsds', 'rsdscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlns', 'rlnscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['blh']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=400, cmap='viridis')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=200,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['orog']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=400, cmap='viridis')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-300,cm_max=300,cm_interval1=50,cm_interval2=100,cmap='BrBG')
        extend2 = 'both'
    elif var2=='psl':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
        extend2 = 'both'
    else:
        print('Warning: no colorbar specified')
    
    extents = []
    extentl = []
    ds = {}
    for ids in dss:
        # ids = dss[0]
        print(f'Get {ids}')
        
        if ids[0] == 'CERES':
            if var2 in ['rsdt', 'rsut', 'rsutcs', 'rlut', 'rlutcs', 'clwvi', 'clivi', 'cwp']:
                ds['CERES'] = xr.open_mfdataset(sorted(glob.glob(f'data/obs/CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}??-{year}{month:02d}??.nc'))[0])
                # ds['CERES'] = xr.open_dataset(f'data/obs/CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')
                ds['CERES'] = ds['CERES'].rename({
                    'toa_sw_clr_1h':    'rsutcs',
                    'toa_sw_all_1h':    'rsut',
                    'toa_lw_clr_1h':    'rlutcs',
                    'toa_lw_all_1h':    'rlut',
                    'toa_solar_all_1h': 'rsdt',
                    'lwp_total_1h':     'clwvi',
                    'iwp_total_1h':     'clivi',})
                ds['CERES']['cwp'] = (ds['CERES']['clwvi']+ds['CERES']['clivi'])
                ds['CERES'] = ds['CERES'][var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'), method='nearest').sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
                if var2 in ['rsut', 'rsutcs', 'rlut', 'rlutcs']:
                    ds['CERES'] *= (-1)
            else:
                print('Warning: no var in CERES')
        elif ids[0] == 'MODIS Aqua':
            doy = datetime(year, month, day).timetuple().tm_yday
            if var1 == 'cwp':
                fl = sorted(glob.glob(f'data/obs/MODIS/MYD06_L2/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))
                lats, lons, vardata = get_modis_latlonvars(fl, 'Cloud_Water_Path')
                ds['MODIS Aqua'] = xr.DataArray(
                    vardata, dims=('y', 'x'), name=var1,
                    coords={'lat': (('y', 'x'), lats),
                            'lon': (('y', 'x'), lons)})
        elif ids[0] == 'ERA5':
            if var1 in ['t2m', 'd2m', 'u10', 'v10', 'u100', 'v100']:
                if var1 == 't2m': vart = '2t'
                if var1 == 'd2m': vart = '2d'
                if var1 == 'u10': vart = '10u'
                if var1 == 'v10': vart = '10v'
                if var1 == 'u100': vart = '100u'
                if var1 == 'v100': vart = '100v'
                ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{vart}/{year}/{vart}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
            elif var1=='orog':
                ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/z/{year}/z_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['z'].sel(time=pd.Timestamp(year,month,day,hour))
            elif var1=='rh2m':
                era5_t2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'] = relative_humidity_from_dewpoint(era5_t2m * units.K, era5_d2m * units.K) * 100
                del era5_t2m, era5_d2m
            elif var1=='q2m':
                era5_sp = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'] = specific_humidity_from_dewpoint(era5_sp * units.Pa, era5_d2m * units.K) * 1000
                del era5_sp, era5_d2m
            elif var1=='mtuwswrf':
                era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year1}/mtnswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                ds['ERA5'] = era5_mtnswrf - era5_mtdwswrf
                del era5_mtnswrf, era5_mtdwswrf
            elif var1=='mtuwswrfcs':
                era5_mtnswrfcs = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrfcs/{year1}/mtnswrfcs_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrfcs'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                ds['ERA5'] = era5_mtnswrfcs - era5_mtdwswrf
                del era5_mtnswrfcs, era5_mtdwswrf
            elif var1=='si10':
                era5_u10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['u10'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_v10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['v10'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'] = (era5_u10**2 + era5_v10**2)**0.5
                del era5_u10, era5_v10
            elif var1 in ['tp', 'e', 'pev', 'mslhf', 'msshf', 'mtnlwrf', 'msdwlwrf', 'mtdwswrf', 'msdwswrfcs', 'msdwswrf', 'msnlwrf', 'mtnlwrfcs', 'msdwlwrfcs', ]:
                ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year1}/{var1}_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')[var1].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            elif var1 in ['skt', 'blh', 'msl', 'tcwv', 'hcc', 'mcc', 'lcc', 'tcc', 'tclw', 'tciw']:
                ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
            elif var1 == 'cwp':
                era5_tclw = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/tclw/{year}/tclw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['tclw'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_tciw = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/tciw/{year}/tciw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['tciw'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'] = era5_tclw + era5_tciw
            else:
                print('Warning: no var in ERA5')
            
            ds['ERA5']['longitude'] = ds['ERA5']['longitude'] % 360
            ds['ERA5'] = ds['ERA5'].sortby(['longitude', 'latitude']).rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
            
            if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
                ds['ERA5'] *= 24000 / 24
            elif var1 in ['msl']:
                ds['ERA5'] /= 100
            elif var1 in ['sst', 't2m', 'd2m', 'skt']:
                ds['ERA5'] -= zerok
            elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
                ds['ERA5'] *= 100
            elif var1 in ['z']:
                ds['ERA5'] /= 9.80665
            elif var1 in ['mper']:
                ds['ERA5'] *= seconds_per_d / 24
            elif var1 in ['tclw', 'tciw', 'cwp']:
                ds['ERA5'] *= 1000
            
            if var1 in ['e', 'pev', 'mper']:
                ds['ERA5'] *= (-1)
        elif ids[0] in suite_res.keys():
            isuite = ids[0]
            ires = suite_res[isuite][ids[1]]
            ilabel = f'{suite_label[isuite]}'
            
            if var2 in ['orog']:
                ds[ilabel] = xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash[var2]]
            elif var2 in ['ts', 'blh', 'tas', 'huss', 'hurs', 'das', 'clslw', 'psl', 'CAPE', 'prw']:
                ds[ilabel] = xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash[var2]].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['rlds', 'rlu_t_s', 'rsut', 'rsdt', 'rsutcs', 'rsdscs', 'rsds', 'rlns', 'rlut', 'rlutcs', 'rldscs', 'hfss', 'hfls', 'rain', 'snow', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi']:
                ds[ilabel] = xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year1}{month1:02d}{day1:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash[var2]].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            elif var2 in ['cwp']:
                ds[ilabel] = xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year1}{month1:02d}{day1:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).sel(time=pd.Timestamp(year1,month1,day1,hour1))
                ds[ilabel] = ds[ilabel][var2stash['clwvi']] + ds[ilabel][var2stash['clivi']]
            elif var2 in ['pr']:
                try:
                    ds[ilabel] = xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year1}{month1:02d}{day1:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash_gal[var2]].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                except KeyError:
                    ds[ilabel] = xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year1}{month1:02d}{day1:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash_ral['rain']].sel(time=pd.Timestamp(year1,month1,day1,hour1)) + xr.open_dataset(sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year1}{month1:02d}{day1:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash_ral['snow']].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            else:
                print(f'Warning: no var in {ilabel}')
            
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ilabel] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ilabel] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ilabel] *= (-1)
            elif var2 in ['psl']:
                ds[ilabel] /= 100
            elif var2 in ['huss', 'clwvi', 'clivi', 'cwp']:
                ds[ilabel] *= 1000
            elif var2 in ['cll', 'clm', 'clh', 'clt']:
                ds[ilabel] *= 100
            
            # update domain
            if len(extents)==0:
                extents = [ds[ilabel].lon[0].values, ds[ilabel].lon[-1].values,
                           ds[ilabel].lat[0].values, ds[ilabel].lat[-1].values]
                extentl = [ds[ilabel].lon[0].values, ds[ilabel].lon[-1].values,
                           ds[ilabel].lat[0].values, ds[ilabel].lat[-1].values]
            else:
                extents[0] = np.max((extents[0], ds[ilabel].lon[0].values))
                extents[1] = np.min((extents[1], ds[ilabel].lon[-1].values))
                extents[2] = np.max((extents[2], ds[ilabel].lat[0].values))
                extents[3] = np.min((extents[3], ds[ilabel].lat[-1].values))
                extentl[0] = np.min((extentl[0], ds[ilabel].lon[0].values))
                extentl[1] = np.max((extentl[1], ds[ilabel].lon[-1].values))
                extentl[2] = np.min((extentl[2], ds[ilabel].lat[0].values))
                extentl[3] = np.max((extentl[3], ds[ilabel].lat[-1].values))
        elif ids[0] == 'BARRA-R2':
            if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut', 'rsutcs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        elif ids[0] == 'BARRA-C2':
            if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut', 'rsutcs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        elif ids[0] == 'BARPA-R':
            if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/1hr/{var2}/latest/{var2}_AUS-15_ERA5_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_1hr_{year}01-{year}12.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut', 'rsutcs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/1hr/{var2}/latest/{var2}_AUS-15_ERA5_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_1hr_{year}01-{year}12.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        elif ids[0] == 'BARPA-C':
            if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut', 'rsutcs']:
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        
        if ids[0] in ['BARRA-R2', 'BARRA-C2', 'BARPA-R', 'BARPA-C']:
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ids[0]] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ids[0]] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ids[0]] *= (-1)
            elif var2 in ['psl']:
                ds[ids[0]] /= 100
            elif var2 in ['huss', 'clwvi', 'clivi', 'cwp']:
                ds[ids[0]] *= 1000
            
            if len(extents)==0:
                extents = [ds[ids[0]].lon[0].values, ds[ids[0]].lon[-1].values,
                           ds[ids[0]].lat[0].values, ds[ids[0]].lat[-1].values]
                extentl = [ds[ids[0]].lon[0].values, ds[ids[0]].lon[-1].values,
                           ds[ids[0]].lat[0].values, ds[ids[0]].lat[-1].values]
            else:
                extents[0] = np.max((extents[0], ds[ids[0]].lon[0].values))
                extents[1] = np.min((extents[1], ds[ids[0]].lon[-1].values))
                extents[2] = np.max((extents[2], ds[ids[0]].lat[0].values))
                extents[3] = np.min((extents[3], ds[ids[0]].lat[-1].values))
                extentl[0] = np.min((extentl[0], ds[ids[0]].lon[0].values))
                extentl[1] = np.max((extentl[1], ds[ids[0]].lon[-1].values))
                extentl[2] = np.min((extentl[2], ds[ids[0]].lat[0].values))
                extentl[3] = np.max((extentl[3], ds[ids[0]].lat[-1].values))
    
    min_lons, max_lons, min_lats, max_lats = extents
    min_lonl, max_lonl, min_latl, max_latl = extentl
    pheight = pwidth * (max_lats - min_lats) / (max_lons - min_lons)
    # pheight = pwidth * (max_latl - min_latl) / (max_lonl - min_lonl)
    fm_bottom = 2/(pheight*nrow+3)
    fm_top = 1 - 1/(pheight*nrow+3)
    
    for imode in modes:
        # imode = 'difference'
        print(f'#-------- {imode}')
        
        plt_colnames = list(ds.keys())
        if imode=='difference':
            plt_colnames = [plt_colnames[0]] + [f'{iname} - {plt_colnames[0]}' for iname in plt_colnames[1:]]
        
        opng = f"figures/4_um/4.1_access_ram3/4.1.1_sim_obs/4.1.1.2 {var2} {', '.join(x.replace('$', '').replace('\_', ' ') for x in ds.keys())} {imode} {str(np.round(min_lonl, 2))}_{str(np.round(max_lonl, 2))}_{str(np.round(min_latl, 2))}_{str(np.round(max_latl, 2))} {year}-{month:02d}-{day:02d} {hour:02d} UTC.png"
        ncol = len(plt_colnames)
        cbar_label1 = f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}'
        cbar_label2 = f"Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}"
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=np.array([pwidth*ncol, pheight*nrow+3])/2.54,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.05},)
        
        for jcol in range(ncol):
            axs[jcol] = regional_plot(extent=extents, central_longitude=180, ax_org=axs[jcol], lw=0.3)
            # axs[jcol] = regional_plot(extent=extentl, central_longitude=180, ax_org=axs[jcol], lw=0.1)
            # if extents != extentl:
            #     axs[jcol].add_patch(Rectangle(
            #         (min_lons, min_lats), max_lons-min_lons, max_lats-min_lats,
            #         ec='red', color='None', lw=0.5, linestyle=':',
            #         transform=ccrs.PlateCarree(), zorder=2))
        
        if imode == 'original':
            for jcol, ids1 in enumerate(ds.keys()):
                # jcol = 0; ids1 = list(ds.keys())[0]
                # print(f'#---- {jcol} {ids1}')
                plt_mesh = axs[jcol].pcolormesh(
                    ds[ids1].lon,
                    ds[ids1].lat,
                    ds[ids1],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                if ids1 in ['MODIS Aqua']:
                    mask = (ds[ids1].lon >= min_lons) & (ds[ids1].lon <= max_lons) & (ds[ids1].lat >= min_lats) & (ds[ids1].lat <= max_lats)
                    mean = ds[ids1].where(mask).weighted(np.cos(np.deg2rad(ds[ids1].lat.where(mask))).fillna(0)).mean().values
                else:
                    mean = ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).weighted(np.cos(np.deg2rad(ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).lat))).mean().values
                if jcol == 0:
                    if ids1 == 'MODIS Aqua':
                        plt_text = [f'']
                    else:
                        plt_text = [f'Mean: {str(np.round(mean, 1))}']
                elif (jcol==1) & (list(ds.keys())[0]=='MODIS Aqua'):
                    plt_text.append(f'Mean: {str(np.round(mean, 1))}')
                else:
                    plt_text.append(f'{str(np.round(mean, 1))}')
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.26, fm_bottom*0.75, 0.48, fm_bottom/8]))
                cbar.ax.set_xlabel(cbar_label1)
        elif imode=='difference':
            plt_mesh = axs[0].pcolormesh(
                ds[list(ds.keys())[0]].lon,
                ds[list(ds.keys())[0]].lat,
                ds[list(ds.keys())[0]],
                norm=pltnorm, cmap=pltcmp,
                transform=ccrs.PlateCarree(), zorder=1)
            mean = ds[list(ds.keys())[0]].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).weighted(np.cos(np.deg2rad(ds[list(ds.keys())[0]].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).lat))).mean().values
            plt_text = [f'Mean: {str(np.round(mean, 1))}']
            for jcol, ids1 in zip(range(1, ncol), list(ds.keys())[1:]):
                # jcol = 1; ids1 = list(ds.keys())[1]
                ids2 = list(ds.keys())[0]
                # print(f'#-------- {jcol} {ids1} {ids2}')
                if not f'{ids1} - {ids2}' in regridder.keys():
                    regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                        ds[ids1],
                        ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])),
                        method='bilinear')
                plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1]) - ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1]))
                rmsd = coslat_weighted_rmsd(plt_data.sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)))
                md = coslat_weighted_mean(plt_data.sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)))
                if jcol==1:
                    plt_text.append(f'RMSD: {str(np.round(rmsd, 1))}, MD: {str(np.round(md, 1))}')
                else:
                    plt_text.append(f'{str(np.round(rmsd, 1))}, {str(np.round(md, 1))}')
                plt_mesh2 = axs[jcol].pcolormesh(
                    plt_data.lon, plt_data.lat, plt_data,
                    norm=pltnorm2, cmap=pltcmp2,
                    transform=ccrs.PlateCarree(), zorder=1)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.01, fm_bottom*0.75, 0.48, fm_bottom/8]))
            cbar.ax.set_xlabel(cbar_label1)
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.51, fm_bottom*0.75, 0.48, fm_bottom/8]))
            cbar2.ax.set_xlabel(cbar_label2)
        
        for jcol in range(ncol):
            axs[jcol].text(
                0, 1.02,
                f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
            axs[jcol].text(
                0.01, 0.01, plt_text[jcol],
                ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        # fig.text(0.5, 0.01, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='bottom')
        fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)





'''
x = [500, 150, 100, 50, 5]
y = [-252.4, -230.7, -223.6, -208.5, -167.7]

# Create line plot
plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o', linestyle='-', color='tab:blue')

# Labels and title
plt.xlabel('Variable (e.g., value)')
plt.ylabel('Response (e.g., flux)')
plt.title('Line Plot of Two Arrays')

# Optional: reverse x-axis if values represent pressure levels
plt.gca().invert_xaxis()  # comment this out if not needed

plt.grid(True)
plt.tight_layout()
plt.savefig('figures/test.png')


ds1 = xr.open_mfdataset(sorted(glob.glob(f'data/obs/CERES/SYN1deg/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}??-{year}{month:02d}??.nc')))
ds2 = xr.open_dataset(f'data/obs/CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')

(ds1['toa_sw_all_1h'].values == ds2['toa_sw_all_1h'].values).all()

'''
# endregion


# region obs vs. sim monthly data

year, month = 2020, 6
starttime = datetime(year, month, 2)
endtime = datetime(year, month, 30, 23, 59)
var2s = ['sfcWind']
# 'rsds', 'rlds', 'rlus', 'rsus', 'sfcWind', 'ts', 'hurs', 'huss', 'prw', 'tas', 'psl', 'blh'
# 'rsut', 'clivi', 'clwvi', 'cll', 'clm', 'clh', 'clt', 'pr', 'hfls', 'hfss'
modes = ['original', 'difference'] # 'original', 'difference'
dsss = [
    # [('ERA5',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    # [('OAFlux',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    # [('IMERG',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    # [('Himawari',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    # [('CERES',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    
    # [('u-ds714',1),('u-ds718',1),('u-ds717',1)],#CDNC
    # [('OAFlux',''),('u-ds714',1),('u-ds718',1),('u-ds717',1)],#CDNC
    # [('IMERG',''),('u-ds714',1),('u-ds718',1),('u-ds717',1)],#CDNC
    # [('Himawari',''),('u-ds714',1),('u-ds718',1),('u-ds717',1)],#CDNC
    # [('CERES',''),('u-ds714',1),('u-ds718',1),('u-ds717',1)],#CDNC
    
    # [('ERA5',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds728',1),('u-ds732',1)],
    # [('OAFlux',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds728',1),('u-ds732',1)],
    # [('IMERG',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds728',1),('u-ds732',1)],
    # [('Himawari',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds728',1),('u-ds732',1)],
    # [('CERES',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds728',1),('u-ds732',1)],
    
    # [('IMERG',''),('ERA5',''),('BARRA-R2',''),('BARRA-C2',''),('BARPA-R',''),('BARPA-C',''),('u-ds714',1)],#control
    # [('CERES',''),('ERA5',''),('BARRA-R2',''),('BARRA-C2',''),('BARPA-R',''),('BARPA-C',''),('u-ds714',1)],#control
    
    [('u-ds722',1),('u-ds724',1),('u-ds726',1)],#res
    # [('CERES',''),('u-ds714',1),('u-ds722',1),('u-ds724',1),('u-ds726',1)],#res
    # [('CERES',''),('u-ds714',1),('u-ds728',1),('u-ds730',1),('u-ds732',1)],#Sres
]

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth = 4.4
nrow = 1
regridder = {}
cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}

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
  for var2 in var2s:
    # var2 = 'rsut'
    var1 = cmip6_era5_var[var2]
    print(f'#---------------- {var1} vs. {var2}')
    
    if var2 == 'prw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['cll']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0,cm_max=70,cm_interval1=5,cm_interval2=10,cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5,cm_interval2=10,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clm', 'clh']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=50, cm_interval1=5,cm_interval2=10,cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5,cm_interval2=10,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clt']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0,cm_max=100,cm_interval1=10,cm_interval2=10,cmap='Blues_r')
        extend = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5,cm_interval2=10,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clwvi']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0,cm_max=160,cm_interval1=10,cm_interval2=20,cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-120,cm_max=120,cm_interval1=10,cm_interval2=20,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clivi']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0,cm_max=160,cm_interval1=10,cm_interval2=20,cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-120,cm_max=120,cm_interval1=10,cm_interval2=20,cmap='BrBG_r')
        extend2 = 'both'
    elif var2=='pr':
        # pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        #     cm_min=0,cm_max=8,cm_interval1=0.5,cm_interval2=1,cmap='Blues_r')
        # extend = 'max'
        # pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        #     cm_min=-4,cm_max=4,cm_interval1=0.5,cm_interval2=1,cmap='BrBG_r')
        # extend2 = 'both'
        pltlevel = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
        extend = 'max'
        pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
        extend2 = 'both'
    elif var2 in ['evspsbl', 'evspsblpot']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['hfls']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300,cm_max=-100,cm_interval1=10,cm_interval2=20,cmap='Greens')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-80,cm_max=80,cm_interval1=10,cm_interval2=20,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['hfss']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-100, cm_max=0, cm_interval1=5, cm_interval2=10, cmap='Greens')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20,cm_max=20,cm_interval1=2,cm_interval2=4,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['sfcWind']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Purples_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['tas', 'ts']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['das']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['huss']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Blues_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4,cm_max=4,cm_interval1=0.5,cm_interval2=1,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['hurs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues_r')
        extend = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['rsut']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-130, cm_max=-50, cm_interval1=5, cm_interval2=10, cmap='Greens')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50,cm_max=50,cm_interval1=5,cm_interval2=10,cmap='BrBG')
        # pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        #     cm_min=-600, cm_max=0, cm_interval1=50, cm_interval2=100, cmap='Greens')
        # extend = 'min'
        # pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        #     cm_min=-300,cm_max=300,cm_interval1=50,cm_interval2=100,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rsutcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-800, cm_max=0, cm_interval1=50, cm_interval2=100, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-400,cm_max=400,cm_interval1=50,cm_interval2=100,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlut', 'rlutcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-360, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-180,cm_max=180,cm_interval1=20,cm_interval2=40,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlus', 'rluscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-360, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-180,cm_max=180,cm_interval1=20,cm_interval2=40,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rsus']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-50, cm_max=0, cm_interval1=5, cm_interval2=10, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlds', 'rldscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='Greens_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        extend2 = 'both'
    elif var2=='rsdt':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='Greens_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG',)
        extend2 = 'both'
    elif var2 in ['rsds', 'rsdscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='Greens_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['rlns', 'rlnscs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='Greens',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['blh']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=400, cmap='Greens_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=200,cmap='BrBG')
        extend2 = 'both'
    elif var2 in ['orog']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=400, cmap='Greens_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-300,cm_max=300,cm_interval1=50,cm_interval2=100,cmap='BrBG')
        extend2 = 'both'
    elif var2=='psl':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='Greens_r',)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
        extend2 = 'both'
    else:
        print('Warning: no colorbar specified')
    
    extents = []
    extentl = []
    ds = {}
    for ids in dss:
        # ids = dss[0]
        print(f'Get {ids}')
        
        if ids[0] == 'CERES':
            # ids = ('CERES', '')
            if var2 in ['rsdt', 'rsut', 'rsutcs', 'rlut', 'rlutcs']:
                ds['CERES'] = xr.open_dataset(f'data/obs/CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')
                ds['CERES'] = ds['CERES'].rename({
                    'toa_sw_clr_1h':    'rsutcs',
                    'toa_sw_all_1h':    'rsut',
                    'toa_lw_clr_1h':    'rlutcs',
                    'toa_lw_all_1h':    'rlut',
                    'toa_solar_all_1h': 'rsdt'})
                ds['CERES'] = ds['CERES'][var2].sel(time=slice(starttime, endtime), lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1)).mean(dim='time')
                if var2 in ['rsut', 'rsutcs', 'rlut', 'rlutcs']:
                    ds['CERES'] *= (-1)
            elif var2 in ['clwvi', 'clivi']:
                ds['CERES'] = xr.open_mfdataset(sorted(glob.glob(f'data/obs/CERES/CERES_SYN1deg-1H/{var1}/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}*.nc')))
                if var2=='clwvi':
                    ds['CERES'] = ds['CERES'].rename({'lwp_total_1h': var2})
                elif var2=='clivi':
                    ds['CERES'] = ds['CERES'].rename({'iwp_total_1h': var2})
                ds['CERES'] = ds['CERES'][var2].sel(time=slice(starttime, endtime), lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1)).mean(dim='time')
            else:
                print('Warning: no var in CERES')
        elif ids[0] == 'OAFlux':
            with open(f'data/obs/OAFlux/oaflux_mon_alltime_{var2}.pkl', 'rb') as f:
                oaflux_mon_alltime = pickle.load(f)
            ds['OAFlux'] = oaflux_mon_alltime['mon'].sel(time=f'{year}-{month:02d}').squeeze().sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
        elif ids[0] == 'IMERG':
            # ids = ('IMERG', '')
            with open(f'data/obs/IMERG/imerg_mon_alltime_pr.pkl', 'rb') as f:
                imerg_mon_alltime = pickle.load(f)
            ds['IMERG'] = imerg_mon_alltime['mon'].sel(time=f'{year}-{month:02d}').squeeze().sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
            # ds['IMERG'] = xr.open_mfdataset(sorted(glob.glob(f'data/obs/IMERG/3B-DAY.MS.MRG.3IMERG/3B-DAY.MS.MRG.3IMERG.{year}{month:02d}??-S000000-E235959.V07B.nc4.SUB.nc4')))
            # ds['IMERG'] = ds['IMERG']['precipitation'].sel(time=slice(starttime, endtime), lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1)).mean(dim='time').compute()
        elif ids[0] == 'Himawari':
            # ids = ('Himawari', '')
            cltype_count = xr.open_mfdataset(sorted(glob.glob(f'data/obs/jaxa/clp/{year}{month:02d}/??/cltype_count_{year}{month:02d}??.nc')))
            cltype_count = cltype_count['cltype_count'].sel(time=slice(starttime, endtime), lon=slice(min_lon1, max_lon1), lat=slice(max_lat1, min_lat1))
            ds['Himawari'] = cltype_count.sel(types=cltypes[var1]).sum(dim=['types', 'time']) / cltype_count.sel(types='finite').sum(dim='time') * 100
        elif ids[0] == 'ERA5':
            # ids = ('ERA5', '')
            if var1 in ['t2m', 'd2m', 'u10', 'v10', 'u100', 'v100']:
                # var1 = 't2m'
                if var1 == 't2m': vart = '2t'
                if var1 == 'd2m': vart = '2d'
                if var1 == 'u10': vart = '10u'
                if var1 == 'v10': vart = '10v'
                if var1 == 'u100': vart = '100u'
                if var1 == 'v100': vart = '100v'
                ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{vart}/{year}/{vart}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1)).mean(dim='time')
            elif var1=='rh2m':
                era5_t2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = relative_humidity_from_dewpoint(era5_t2m * units.K, era5_d2m * units.K).mean(dim='time') * 100
                del era5_t2m, era5_d2m
            elif var1=='q2m':
                era5_sp = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = specific_humidity_from_dewpoint(era5_sp * units.Pa, era5_d2m * units.K).mean(dim='time') * 1000
                del era5_sp, era5_d2m
            elif var1=='mtuwswrf':
                era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year}/mtnswrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['mtnswrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year}/mtdwswrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['mtdwswrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = (era5_mtnswrf - era5_mtdwswrf).mean(dim='time')
                del era5_mtnswrf, era5_mtdwswrf
            elif var1=='msuwlwrf':
                era5_msnlwrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/msnlwrf/{year}/msnlwrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['msnlwrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_msdwlwrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/msdwlwrf/{year}/msdwlwrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['msdwlwrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = (era5_msnlwrf - era5_msdwlwrf).mean(dim='time')
                del era5_msnlwrf, era5_msdwlwrf
            elif var1=='msuwswrf':
                era5_msnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/msnswrf/{year}/msnswrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['msnswrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_msdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/msdwswrf/{year}/msdwswrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['msdwswrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = (era5_msnswrf - era5_msdwswrf).mean(dim='time')
                del era5_msnswrf, era5_msdwswrf
            elif var1=='mtuwswrfcs':
                era5_mtnswrfcs = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrfcs/{year}/mtnswrfcs_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['mtnswrfcs'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year}/mtdwswrf_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['mtdwswrf'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = (era5_mtnswrfcs - era5_mtdwswrf).mean(dim='time')
                del era5_mtnswrfcs, era5_mtdwswrf
            elif var1=='si10':
                era5_u10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['u10'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_v10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['v10'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = ((era5_u10**2 + era5_v10**2)**0.5).mean(dim='time')
                del era5_u10, era5_v10
            elif var1 in ['tp', 'e', 'pev', 'mslhf', 'msshf', 'mtnlwrf', 'msdwlwrf', 'mtdwswrf', 'msdwswrfcs', 'msdwswrf', 'msnlwrf', 'mtnlwrfcs', 'msdwlwrfcs', 'skt', 'blh', 'msl', 'tcwv', 'hcc', 'mcc', 'lcc', 'tcc', 'tclw', 'tciw']:
                ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1)).mean(dim='time')
            elif var1 == 'cwp':
                era5_tclw = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/tclw/{year}/tclw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['tclw'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                era5_tciw = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/tciw/{year}/tciw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['tciw'].sel(time=slice(starttime, endtime), longitude=slice(min_lon1, max_lon1), latitude=slice(max_lat1, min_lat1))
                ds['ERA5'] = (era5_tclw + era5_tciw).mean(dim='time')
            else:
                print('Warning: no var in ERA5')
            
            ds['ERA5'] = ds['ERA5'].rename({'latitude':'lat','longitude':'lon'})
            
            if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
                ds['ERA5'] *= 24000
            elif var1 in ['msl']:
                ds['ERA5'] /= 100
            elif var1 in ['sst', 't2m', 'd2m', 'skt']:
                ds['ERA5'] -= zerok
            elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
                ds['ERA5'] *= 100
            elif var1 in ['z']:
                ds['ERA5'] /= 9.80665
            elif var1 in ['mper']:
                ds['ERA5'] *= seconds_per_d
            elif var1 in ['tclw', 'tciw', 'cwp']:
                ds['ERA5'] *= 1000
            
            if var1 in ['e', 'pev', 'mper']:
                ds['ERA5'] *= (-1)
        elif ids[0] == 'BARRA-R2':
            # ids = ('BARRA-R2', '')
            if var2 == 'blh':
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/zmla/latest/zmla_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')['zmla'].sel(time=slice(starttime, endtime)).mean(dim='time')
            else:
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=slice(starttime, endtime)).mean(dim='time')
        elif ids[0] == 'BARPA-R':
            # ids = ('BARPA-R', '')
            if var2 == 'blh':
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/1hr/zmla/latest/zmla_AUS-15_ERA5_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_1hr_{year}01-{year}12.nc')['zmla'].sel(time=slice(starttime, endtime)).mean(dim='time')
            else:
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/1hr/{var2}/latest/{var2}_AUS-15_ERA5_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_1hr_{year}01-{year}12.nc')[var2].sel(time=slice(starttime, endtime)).mean(dim='time')
        elif ids[0] == 'BARRA-C2':
            # ids = ('BARRA-C2', '')
            if var2 == 'blh':
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/zmla/latest/zmla_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')['zmla'].sel(time=slice(starttime, endtime)).mean(dim='time')
            else:
                ds[ids[0]] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=slice(starttime, endtime)).mean(dim='time')
        elif ids[0] == 'BARPA-C':
            # ids = ('BARPA-C', '')
            if var2 == 'blh':
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/zmla/latest/zmla_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')['zmla'].sel(time=slice(starttime, endtime)).mean(dim='time')
            else:
                ds[ids[0]] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=slice(starttime, endtime)).mean(dim='time')
        elif ids[0] in suite_res.keys():
            # ids = ('u-ds714', 1)
            isuite = ids[0]
            ires = suite_res[isuite][ids[1]]
            ilabel = f'{suite_label[isuite]}'
            
            fl = sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year}{month:02d}??T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))
            if var2 == 'pr':
                ds[ilabel] = xr.open_mfdataset(fl, preprocess=lambda ds: ds.pipe(preprocess_umoutput))
                if var2stash_gal[var2] in ds[ilabel].data_vars:
                    ds[ilabel] = ds[ilabel][var2stash_gal[var2]]
                elif (var2stash_ral['rain'] in ds[ilabel].data_vars) & (var2stash_ral['snow'] in ds[ilabel].data_vars):
                    ds[ilabel] = (ds[ilabel][var2stash_ral['rain']] + ds[ilabel][var2stash_ral['snow']])
                ds[ilabel] = ds[ilabel].sel(time=slice(starttime, endtime)).mean(dim='time').compute()
            else:
                ds[ilabel] = xr.open_mfdataset(fl, preprocess=lambda ds: ds.pipe(preprocess_umoutput)[var2stash[var2]])[var2stash[var2]].sel(time=slice(starttime, endtime)).mean(dim='time').compute()
            
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ilabel] *= seconds_per_d
            elif var2 in ['tas', 'ts']:
                ds[ilabel] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ilabel] *= (-1)
            elif var2 in ['psl']:
                ds[ilabel] /= 100
            elif var2 in ['huss', 'clwvi', 'clivi', 'cwp']:
                ds[ilabel] *= 1000
            elif var2 in ['cll', 'clm', 'clh', 'clt']:
                ds[ilabel] *= 100
            
            if len(extents)==0:
                extents = [ds[ilabel].lon[0].values, ds[ilabel].lon[-1].values,
                           ds[ilabel].lat[0].values, ds[ilabel].lat[-1].values]
                extentl = [ds[ilabel].lon[0].values, ds[ilabel].lon[-1].values,
                           ds[ilabel].lat[0].values, ds[ilabel].lat[-1].values]
            else:
                extents[0] = np.max((extents[0], ds[ilabel].lon[0].values))
                extents[1] = np.min((extents[1], ds[ilabel].lon[-1].values))
                extents[2] = np.max((extents[2], ds[ilabel].lat[0].values))
                extents[3] = np.min((extents[3], ds[ilabel].lat[-1].values))
                extentl[0] = np.min((extentl[0], ds[ilabel].lon[0].values))
                extentl[1] = np.max((extentl[1], ds[ilabel].lon[-1].values))
                extentl[2] = np.min((extentl[2], ds[ilabel].lat[0].values))
                extentl[3] = np.max((extentl[3], ds[ilabel].lat[-1].values))
        
        if ids[0] in suite_res.keys():
            ds[ilabel]['lon'] = ds[ilabel]['lon'] % 360
            ds[ilabel] = ds[ilabel].sortby(['lon', 'lat'])
        else:
            ds[ids[0]]['lon'] = ds[ids[0]]['lon'] % 360
            ds[ids[0]] = ds[ids[0]].sortby(['lon', 'lat'])
        
        if ids[0] in ['BARRA-R2', 'BARRA-C2', 'BARPA-R', 'BARPA-C']:
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ids[0]] *= seconds_per_d
            elif var2 in ['tas', 'ts']:
                ds[ids[0]] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ids[0]] *= (-1)
            elif var2 in ['psl']:
                ds[ids[0]] /= 100
            elif var2 in ['huss', 'clwvi', 'clivi', 'cwp']:
                ds[ids[0]] *= 1000
            
            if len(extents)==0:
                extents = [ds[ids[0]].lon[0].values, ds[ids[0]].lon[-1].values,
                           ds[ids[0]].lat[0].values, ds[ids[0]].lat[-1].values]
                extentl = [ds[ids[0]].lon[0].values, ds[ids[0]].lon[-1].values,
                           ds[ids[0]].lat[0].values, ds[ids[0]].lat[-1].values]
            else:
                extents[0] = np.max((extents[0], ds[ids[0]].lon[0].values))
                extents[1] = np.min((extents[1], ds[ids[0]].lon[-1].values))
                extents[2] = np.max((extents[2], ds[ids[0]].lat[0].values))
                extents[3] = np.min((extents[3], ds[ids[0]].lat[-1].values))
                extentl[0] = np.min((extentl[0], ds[ids[0]].lon[0].values))
                extentl[1] = np.max((extentl[1], ds[ids[0]].lon[-1].values))
                extentl[2] = np.min((extentl[2], ds[ids[0]].lat[0].values))
                extentl[3] = np.max((extentl[3], ds[ids[0]].lat[-1].values))
    
    min_lons, max_lons, min_lats, max_lats = extents
    min_lonl, max_lonl, min_latl, max_latl = extentl
    pheight = pwidth * (max_lats - min_lats) / (max_lons - min_lons)
    # pheight = pwidth * (max_latl - min_latl) / (max_lonl - min_lonl)
    fm_bottom = 2.2/(pheight*nrow+3)
    fm_top = 1 - 0.8/(pheight*nrow+3)
    
    for imode in modes:
        # imode = 'original'
        print(f'#-------- {imode}')
        
        plt_colnames = list(ds.keys())
        # if imode=='difference':
        #     plt_colnames = [plt_colnames[0]] + [f'{iname} - {plt_colnames[0]}' for iname in plt_colnames[1:]]
        
        opng = f"figures/4_um/4.1_access_ram3/4.1.1_sim_obs/4.1.1.3 {var2} {', '.join(x.replace('$', '').replace('\_', ' ') for x in ds.keys())} {imode} {str(np.round(min_lonl, 2))}_{str(np.round(max_lonl, 2))}_{str(np.round(min_latl, 2))}_{str(np.round(max_latl, 2))} {year}-{month:02d}.png"
        ncol = len(plt_colnames)
        cbar_label1 = f"{calendar.month_name[month]} {year} {era5_varlabels[var1]}"
        cbar_label2 = f"Difference in {era5_varlabels[var1]}"
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=np.array([pwidth*ncol, pheight*nrow+3])/2.54,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.05},)
        
        if var2 == 'pr':
            digit = 2
        else:
            digit = 1
        for jcol in range(ncol):
            axs[jcol] = regional_plot(extent=extents, central_longitude=180, ax_org=axs[jcol], lw=0.5)
        if imode == 'original':
            for jcol, ids1 in enumerate(ds.keys()):
                print(f'#---- {jcol} {ids1}')
                plt_mesh = axs[jcol].pcolormesh(
                    ds[ids1].lon,
                    ds[ids1].lat,
                    ds[ids1],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                mean = coslat_weighted_mean(ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)))
                if jcol==0:
                    plt_text = [f'Mean: {str(np.round(mean, digit))}']
                else:
                    plt_text.append(f'{str(np.round(mean, digit))}')
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.26, fm_bottom*0.75, 0.48, fm_bottom/8]))
                cbar.ax.set_xlabel(cbar_label1)
        elif imode=='difference':
            # imode='difference'
            plt_mesh = axs[0].pcolormesh(
                ds[list(ds.keys())[0]].lon,
                ds[list(ds.keys())[0]].lat,
                ds[list(ds.keys())[0]],
                norm=pltnorm, cmap=pltcmp,
                transform=ccrs.PlateCarree(), zorder=1)
            mean = coslat_weighted_mean(ds[list(ds.keys())[0]].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)))
            plt_text = [f'Mean: {str(np.round(mean, digit))}']
            for jcol, ids1 in zip(range(1, ncol), list(ds.keys())[1:]):
                # jcol = 1; ids1 = list(ds.keys())[1]
                ids2 = list(ds.keys())[0]
                # print(f'#-------- {jcol} {ids1} {ids2}')
                if not f'{ids1} - {ids2}' in regridder.keys():
                    regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                        ds[ids1],
                        ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])),
                        method='bilinear')
                plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1]) - ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).compute()
                rmsd = coslat_weighted_rmsd(plt_data.sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)))
                md = coslat_weighted_mean(plt_data.sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)))
                if jcol==1:
                    plt_text.append(f'{str(np.round(rmsd, digit))}, {str(np.round(md, digit))}')
                    # plt_text.append(f'RMSD: {str(np.round(rmsd, 1))}, MD: {str(np.round(md, 1))}')
                else:
                    plt_text.append(f'{str(np.round(rmsd, digit))}, {str(np.round(md, digit))}')
                plt_mesh2 = axs[jcol].pcolormesh(
                    plt_data.lon, plt_data.lat, plt_data,
                    norm=pltnorm2, cmap=pltcmp2,
                    transform=ccrs.PlateCarree(), zorder=1)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.03, fm_bottom*0.8, 0.44, fm_bottom/8]))
            cbar.ax.set_xlabel(cbar_label1)
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.53, fm_bottom*0.8, 0.44, fm_bottom/8]))
            cbar2.ax.set_xlabel(cbar_label2)
        
        if var2 == 'cll':
            offset = ncol
        elif var2 == 'clwvi':
            offset = ncol * 2
        else:
            offset = 0
        for jcol in range(ncol):
            axs[jcol].text(
                0, 1.02,
                f'({string.ascii_lowercase[jcol+offset]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
            axs[jcol].text(
                0.01, 0.01, plt_text[jcol],
                ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)





# endregion


# region obs vs. sim monthly data Cross Sections
# Memory Used: 31.33GB; Walltime Used: 02:36:31

var2s = [
    'qr',
    # 'hus', 'ta', 'wap', 'zg', 'theta', 'theta_e', 'hur', 'ua', 'va',
    # 'qcf', 'qcl', 'qr', 'qs',
    # 'qc', 'qt', 'clslw', 'qg', 'ACF', 'BCF', 'TCF',
    ]
dsss = [
    # [('ERA5',''),('BARRA-C2',''),('BARPA-C',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    [('ERA5',''),('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    # [('u-ds714',1),('u-ds717',1),('u-ds722',1),('u-ds726',1)],
    ]
modes = ['original', 'difference'] # 'original', 'difference'

year, month = 2020, 6
starttime = datetime(year, month, 2)
endtime = datetime(year, month, 30, 23, 59)
ptop = 600
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
            if var2 in var2stash_ral.keys():
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
            
            axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--')
        
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

