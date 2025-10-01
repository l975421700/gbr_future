

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=96GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60


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
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity

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
    suite_res, suite_label,)

# endregion


# region obs vs. sim timestamp

year, month, day, hour = 2020, 6, 2, 5
var2s = ['rsut']
# 'rsut', 'rlut', 'cll', 'clm', 'clh', 'clt', 'clivi', 'clwvi'
modes = ['difference'] # 'original', 'difference'
dsss = [
    # [('CERES',''),('ERA5',''),('BARRA-C2',''),('u-dq700',1)],#control
    # [('CERES',''),('ERA5',''),('u-dr095',1),('u-dr093',1),('u-dr091',1)],#Sres
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dr040',1),('u-dr041',1)],#CDNC
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dr091',1),('u-dr041',1),('u-dr147',1)],#Sres+CDNC
    # [('CERES',''),('ERA5',''),('u-dq788',1),('u-dq911',1),('u-dq799',1)],#res
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dq799',1),('u-dr041',1),('u-dr145',1)],#res+CDNC
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dr108',1),('u-dr109',1)],#param
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dq912',1)],#LD
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dr105',1),('u-dr107',1)],#levs
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-dr789',1),('u-dr922',1)],#clinho
    # [('CERES',''),('ERA5',''),('u-dq700',1),('u-ds719',1)], #shortTS
    # [('CERES',''),('ERA5',''),('u-ds728',1),('u-ds730',1),('u-ds732',1)],#SA1p1
    # [('CERES',''),('ERA5',''),('u-ds722',1),('u-ds724',1),('u-ds726',1)],#SA1p1
    [('CERES',''),('ERA5',''),('u-dq700',1),('u-ds921',1),('u-ds922',1)],#Spin
    
    # [('CERES',''),('u-dq700',1),('u-dr091',1)],
    # [('CERES',''),('u-dq700',1),('u-dr041',1)],
    # [('CERES',''),('u-dq700',1),('u-dr147',1)],
    
    # [('ERA5',''),('u-dq700',1),('u-dr040',1),('u-dr041',1)],#res+CDNC
    # [('ERA5',''),('u-dq788',1),('u-dq911',1),('u-dq799',1)],#res
    # [('ERA5',''),('u-dr095',1),('u-dr093',1),('u-dr091',1)],#Sres
    # [('ERA5',''),('u-dq700',1),('u-dq799',1),('u-dr041',1),('u-dr145',1)],#res+CDNC
    # [('ERA5',''),('u-dq700',1),('u-dr091',1),('u-dr041',1),('u-dr147',1)],#Sres+CDNC
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
            cm_min=0, cm_max=600, cm_interval1=50, cm_interval2=100, cmap='viridis')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=100,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clivi']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1000, cm_interval1=100, cm_interval2=200, cmap='viridis')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1000,cm_max=1000,cm_interval1=100,cm_interval2=200,cmap='BrBG_r')
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
        # pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        #     cm_min=-300, cm_max=-50, cm_interval1=25, cm_interval2=50, cmap='Greens')
        # extend = 'both'
        # pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        #     cm_min=-200,cm_max=200,cm_interval1=25,cm_interval2=50,cmap='BrBG')
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-600, cm_max=0, cm_interval1=50, cm_interval2=100, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-300,cm_max=300,cm_interval1=50,cm_interval2=100,cmap='BrBG')
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
            if var2 in ['rsdt', 'rsut', 'rsutcs', 'rlut', 'rlutcs', 'clwvi', 'clivi']:
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
                ds['CERES'] = ds['CERES'][var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'), method='nearest').sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
                if var2 in ['rsut', 'rsutcs', 'rlut', 'rlutcs']:
                    ds['CERES'] *= (-1)
            else:
                print('Warning: no var in CERES')
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
            elif var1 in ['tclw', 'tciw']:
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
            elif var2 in ['huss', 'clwvi', 'clivi']:
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
                ds['BARRA-R2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut']:
                ds['BARRA-R2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        elif ids[0] == 'BARRA-C2':
            if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
                ds['BARRA-C2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut']:
                ds['BARRA-C2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        
        if ids[0] in ['BARRA-R2', 'BARRA-C2']:
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ids[0]] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ids[0]] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ids[0]] *= (-1)
            elif var2 in ['psl']:
                ds[ids[0]] /= 100
            elif var2 in ['huss']:
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
    # pheight = pwidth * (max_latl - min_latl) / (max_lonl - min_lonl)
    pheight = pwidth * (max_lats - min_lats) / (max_lons - min_lons)
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
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=np.array([pwidth*ncol, pheight*nrow+3])/2.54,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
        
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
                # print(f'#---- {jcol} {ids1}')
                plt_mesh = axs[jcol].pcolormesh(
                    ds[ids1].lon,
                    ds[ids1].lat,
                    ds[ids1],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                mean = ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).weighted(np.cos(np.deg2rad(ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).lat))).mean().values
                if jcol==0:
                    plt_text = [f'Mean: {str(np.round(mean, 1))}']
                else:
                    plt_text.append(f'{str(np.round(mean, 1))}')
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([1/3, fm_bottom-0.1, 1/3, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=10, labelpad=1)
                cbar.ax.tick_params(labelsize=10, pad=1)
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
                cax=fig.add_axes([0.05, fm_bottom-0.1, 0.4, 0.03]))
            cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=10, labelpad=1)
            cbar.ax.tick_params(labelsize=10, pad=1)
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.1, 0.4, 0.03]))
            cbar2.ax.set_xlabel(f"Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}", fontsize=10, labelpad=1)
            cbar2.ax.tick_params(labelsize=10, pad=1)
        
        for jcol in range(ncol):
            axs[jcol].text(
                0, 1.02,
                f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
            axs[jcol].text(
                0.01, 0.01, plt_text[jcol],
                ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        fig.text(0.5, fm_bottom-0.01, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='top')
        fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)






'''
ds1 = xr.open_mfdataset(sorted(glob.glob(f'data/obs/CERES/SYN1deg/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}??-{year}{month:02d}??.nc')))
ds2 = xr.open_dataset(f'data/obs/CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')

(ds1['toa_sw_all_1h'].values == ds2['toa_sw_all_1h'].values).all()

'''
# endregion


# region obs vs. sim monthly data

year, month = 2020, 6
var2s = ['rsut']
# 'rsut', 'rlut', 'cll', 'clm', 'clh', 'clt', 'clivi', 'clwvi'
modes = ['difference'] # 'original', 'difference'
dsss = [
    # [('CERES',''),('ERA5',''),('BARRA-R2',1),('BARRA-C2','')],
    # [('CERES',''),('u-ds714',1),('u-ds718',1),('u-ds717',1),],
    [('CERES',''),('u-ds722',1),('u-ds724',1),('u-ds726',1),],
]

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth = 4.4
nrow = 1
regridder = {}

for dss in dsss:
  # dss = [('CERES',''),('ERA5',''),('BARRA-R2',1),('BARRA-C2','')],
  print(f'#-------------------------------- {dss}')
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
            cm_min=0, cm_max=600, cm_interval1=50, cm_interval2=100, cmap='viridis')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=100,cmap='BrBG_r')
        extend2 = 'both'
    elif var2 in ['clivi']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=1000, cm_interval1=100, cm_interval2=200, cmap='viridis')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1000,cm_max=1000,cm_interval1=100,cm_interval2=200,cmap='BrBG_r')
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
            cm_min=-100, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='Greens')
        extend = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40,cm_max=40,cm_interval1=5,cm_interval2=10,cmap='BrBG')
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
            if var1 in ['mtdwswrf', 'mtnlwrf', 'mtuwswrf']:
                ds['CERES'] = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').rename({
                    'toa_sw_all_mon': 'mtuwswrf',
                    'toa_lw_all_mon': 'mtnlwrf',
                    'solar_mon': 'mtdwswrf'})[var1].sel(time=f'{year}-{month:02d}').squeeze()
            # elif var1 in ['msdwswrf', 'msuwswrf', 'msdwlwrf', 'msuwlwrf']:
            # elif var1 in ['tclw', 'tciw']:
            
            if var1 in ['mtuwswrf', 'mtnlwrf', 'msuwswrf', 'msuwlwrf']:
                ds[ids[0]] *= (-1)
        elif ids[0] == 'ERA5':
            # ids[0] = 'ERA5'
            with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
                era5_sl_mon_alltime = pickle.load(f)
            ds[ids[0]] = era5_sl_mon_alltime['mon'].sel(time=f'{year}-{month:02d}').squeeze().copy()
            if var2 in ['clwvi', 'clivi']:
                ds[ids[0]] *= 1000
            del era5_sl_mon_alltime
        elif ids[0] == 'BARRA-R2':
            with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
                barra_r2_mon_alltime = pickle.load(f)
            ds[ids[0]] = barra_r2_mon_alltime['mon'].sel(time=f'{year}-{month:02d}').squeeze().copy()
            if var2 in ['clwvi', 'clivi']:
                ds[ids[0]] *= 1000
            del barra_r2_mon_alltime
        elif ids[0] == 'BARRA-C2':
            with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
                barra_c2_mon_alltime = pickle.load(f)
            ds[ids[0]] = barra_c2_mon_alltime['mon'].sel(time=f'{year}-{month:02d}').squeeze().copy()
            if var2 in ['clwvi', 'clivi']:
                ds[ids[0]] *= 1000
            del barra_c2_mon_alltime
        elif ids[0] == 'BARPA-C':
            with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var2}.pkl','rb') as f:
                barpa_c_mon_alltime = pickle.load(f)
            ds[ids[0]] = barpa_c_mon_alltime['mon'].sel(time=f'{year}-{month:02d}').squeeze().copy()
            if var2 in ['clwvi', 'clivi']:
                ds[ids[0]] *= 1000
            del barpa_c_mon_alltime
        elif ids[0] in suite_res.keys():
            isuite = ids[0]
            ires = suite_res[isuite][ids[1]]
            ilabel = f'{suite_label[isuite]}'
            
            fl = sorted(glob.glob(f'scratch/cylc-run/{isuite}/share/cycle/{year}{month:02d}??T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))
            ds[ilabel] = xr.open_mfdataset(fl, preprocess=lambda ds: ds.pipe(preprocess_umoutput)[var2stash[var2]])[var2stash[var2]].mean(dim='time')
            
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ilabel] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ilabel] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ilabel] *= (-1)
            elif var2 in ['psl']:
                ds[ilabel] /= 100
            elif var2 in ['huss', 'clwvi', 'clivi']:
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
        
        if ids[0] in ['BARRA-R2', 'BARRA-C2']:
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
    # pheight = pwidth * (max_latl - min_latl) / (max_lonl - min_lonl)
    pheight = pwidth * (max_lats - min_lats) / (max_lons - min_lons)
    fm_bottom = 2/(pheight*nrow+3)
    fm_top = 1 - 1/(pheight*nrow+3)
    
    for imode in modes:
        # imode = 'difference'
        print(f'#-------- {imode}')
        
        plt_colnames = list(ds.keys())
        if imode=='difference':
            plt_colnames = [plt_colnames[0]] + [f'{iname} - {plt_colnames[0]}' for iname in plt_colnames[1:]]
        
        opng = f"figures/4_um/4.1_access_ram3/4.1.1_sim_obs/4.1.1.3 {var2} {', '.join(x.replace('$', '').replace('\_', ' ') for x in ds.keys())} {imode} {str(np.round(min_lonl, 2))}_{str(np.round(max_lonl, 2))}_{str(np.round(min_latl, 2))}_{str(np.round(max_latl, 2))} {year}-{month:02d}.png"
        ncol = len(plt_colnames)
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=np.array([pwidth*ncol, pheight*nrow+3])/2.54,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
        
        for jcol in range(ncol):
            axs[jcol] = regional_plot(extent=extents, central_longitude=180, ax_org=axs[jcol], lw=0.3)
        
        if imode == 'original':
            for jcol, ids1 in enumerate(ds.keys()):
                # print(f'#---- {jcol} {ids1}')
                plt_mesh = axs[jcol].pcolormesh(
                    ds[ids1].lon,
                    ds[ids1].lat,
                    ds[ids1],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                mean = ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).weighted(np.cos(np.deg2rad(ds[ids1].sel(lon=slice(min_lons, max_lons), lat=slice(min_lats, max_lats)).lat))).mean().values
                if jcol==0:
                    plt_text = [f'Mean: {str(np.round(mean, 1))}']
                else:
                    plt_text.append(f'{str(np.round(mean, 1))}')
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([1/3, fm_bottom-0.1, 1/3, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=10, labelpad=1)
                cbar.ax.tick_params(labelsize=10, pad=1)
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
                plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1]) - ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).compute()
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
                cax=fig.add_axes([0.05, fm_bottom-0.1, 0.4, 0.03]))
            cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=10, labelpad=1)
            cbar.ax.tick_params(labelsize=10, pad=1)
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.1, 0.4, 0.03]))
            cbar2.ax.set_xlabel(f"Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}", fontsize=10, labelpad=1)
            cbar2.ax.tick_params(labelsize=10, pad=1)
        
        for jcol in range(ncol):
            axs[jcol].text(
                0, 1.02,
                f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                ha='left', va='bottom', transform=axs[jcol].transAxes)
            axs[jcol].text(
                0.01, 0.01, plt_text[jcol],
                ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        fig.text(0.5, fm_bottom-0.01, f'{year}-{month:02d}', ha='center', va='top')
        fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)





# endregion

