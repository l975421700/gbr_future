

# qsub -I -q normal -P gx60 -l walltime=3:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55


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


# region get and plot data

suite_res = {
    'u-dq700': ['d11km', 'd4p4km'],
    'u-dq788': ['d11km', 'd4p4kms'],
    'u-dq799': ['d11km', 'd1p1km'],
    'u-dq911': ['d11km', 'd2p2km'],
    'u-dq912': ['d11km', 'd4p4kml'],
    'u-dq987': ['d11km', 'd4p4km'],
    'u-dr040': ['d11km', 'd4p4km'],
    'u-dr041': ['d11km', 'd4p4km'],
    'u-dr091': ['d11km', 'd1p1kmsa'],
    'u-dr093': ['d11km', 'd2p2kmsa'],
    'u-dr095': ['d11km', 'd4p4kmsa'],
    'u-dr105': ['d11km', 'd4p4km'],
    'u-dr107': ['d11km', 'd4p4km'],
    'u-dr108': ['d11km', 'd4p4km'],
    'u-dr109': ['d11km', 'd4p4km'],
    'u-dr144': ['d11km', 'd4p4kms'],
    'u-dr145': ['d11km', 'd1p1km'],
    'u-dr146': ['d11km', 'd2p2km'],
    'u-dr147': ['d11km', 'd1p1kmsa'],
    'u-dr148': ['d11km', 'd2p2kmsa'],
    'u-dr149': ['d11km', 'd4p4kmsa'],
}
year, month, day, hour = 2020, 6, 2, 4
ntime = pd.Timestamp(year,month,day,hour)
# ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth  = 6.6
pheight = 6.6 * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)
nrow = 1
ncol = 3
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)

stash_var = {
    'cll':      'STASH_m01s09i203',
    'clm':      'STASH_m01s09i204',
    'clh':      'STASH_m01s09i205',
    'clt':      'STASH_m01s09i216',
    'prw':      'STASH_m01s30i461',
    'ts':      'STASH_m01s00i024',
    'tas':      'STASH_m01s03i236',
    'huss':     'STASH_m01s03i237',
    'hurs':     'STASH_m01s03i245',
    'hus':      'STASH_m01s00i010',
    'ta':       'STASH_m01s16i004',
    'ua':       'STASH_m01s00i002',
    'va':       'STASH_m01s00i003',
    'wa':       'STASH_m01s00i150',
    'wap':      'STASH_m01s30i008',
    'theta':    'STASH_m01s00i004',
    'qcf':      'STASH_m01s00i012',
    'qcl':      'STASH_m01s00i254',
    'qc':       'STASH_m01s16i206',
    'qs':       'STASH_m01s00i271',
    'qr':       'STASH_m01s00i272',
    'qg':       'STASH_m01s00i273',
    'qt':       'STASH_m01s16i207',
    'mv':       'STASH_m01s00i391',
    'mcl':      'STASH_m01s00i392',
    'mcf':      'STASH_m01s00i393',
    'mr':       'STASH_m01s00i394',
    'mg':       'STASH_m01s00i395',
    'mcf2':     'STASH_m01s00i396',
    'pa':       'STASH_m01s00i408',
    'hfls':     'STASH_m01s03i234',
    'hfss':     'STASH_m01s03i217',
    'rlut':     'STASH_m01s02i205',
    'rlds':     'STASH_m01s00i238',
    'rsut':     'STASH_m01s01i205',
    'rsdt':     'STASH_m01s01i207',
    'rsutcs':   'STASH_m01s01i209',
    'rsdscs':   'STASH_m01s01i210',
    'rsds':     'STASH_m01s01i235',
    'rlns':     'STASH_m01s02i201',
    'rlutcs':   'STASH_m01s02i206',
    'rldscs':   'STASH_m01s02i208',
    'psl':      'STASH_m01s16i222',
    'blh':      'STASH_m01s00i025',
    'iland':    'STASH_m01s00i030',
    'orog':     'STASH_m01s00i033',
    'ncloud':   'STASH_m01s00i075',
    'nrain':    'STASH_m01s00i076',
    'nice':     'STASH_m01s00i078',
    'nsnow':    'STASH_m01s00i079',
    'ngraupel': 'STASH_m01s00i081',
    'rlds2':    'STASH_m01s02i207',
    'rlu_t_s':  'STASH_m01s00i239',
    'das':      'STASH_m01s03i250',
    'blendingw':'STASH_m01s03i513',
    'radar_reflectivity':   'STASH_m01s04i118',
    'clslw':    'STASH_m01s04i224',
    'CAPE':     'STASH_m01s20i114',
    'clwvi':    'STASH_m01s30i405',
    'clivi':    'STASH_m01s30i406',
    # 'pr':   'STASH_m01s00i348',
}
nrad_time = {'d11km': 'T1HR_MN_rad',
             'd4p4km': 'T1HR_MN_rad_diag',
             'd4p4kms': 'T1HR_MN_rad_diag',
             'd4p4kml': 'T1HR_MN_rad_diag',
             'd1p1km': 'T1HR_MN_rad_diag',
             'd2p2km': 'T1HR_MN_rad_diag',
             'd1p1kmsa': 'T1HR_MN_rad_diag',
             'd2p2kmsa': 'T1HR_MN_rad_diag',
             'd4p4kmsa': 'T1HR_MN_rad_diag',
             }

regridder = {}
for isuite in ['u-dq700', 'u-dq788', 'u-dq799', 'u-dq911', 'u-dq912', 'u-dq987', 'u-dr040', 'u-dr041', 'u-dr091', 'u-dr093', 'u-dr095', 'u-dr105', 'u-dr107', 'u-dr108', 'u-dr109', 'u-dr144', 'u-dr145', 'u-dr146', 'u-dr147', 'u-dr148', 'u-dr149']:
    # isuite='u-dq700'
    # ['u-dq700', 'u-dq788', 'u-dq799', 'u-dq911', 'u-dq912', 'u-dq987', 'u-dr040', 'u-dr041', 'u-dr091', 'u-dr093', 'u-dr095']
    print(f'#-------------------------------- {isuite}')
    
    for var2 in ['clm', 'clh', 'clt']:
        # var2 = 'orog'
        # ['cll', 'clm', 'clh', 'clt', 'prw', 'ts', 'tas', 'huss', 'hurs', 'hus', 'ta', 'ua', 'va', 'wa', 'wap', 'theta', 'qcf', 'qcl', 'qc', 'qs', 'qr', 'qg', 'qt', 'mv', 'mcl', 'mcf', 'mr', 'mg', 'mcf2', 'pa', 'hfls', 'hfss', 'rlut', 'rlds', 'rsut', 'rsdt', 'rsutcs', 'rsdscs', 'rsds', 'rlns', 'rlutcs', 'rldscs', 'psl', 'blh', 'iland', 'orog', 'ncloud', 'nrain', 'nice', 'nsnow', 'ngraupel', 'rlds2', 'rlu_t_s', 'das', 'blendingw', 'radar_reflectivity', 'clslw', 'CAPE', 'clwvi', 'clivi']
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
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-100,cm_max=100,cm_interval1=10,cm_interval2=20,cmap='BrBG_r')
            extend2 = 'neither'
        elif var2 in ['clwvi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=0.6, cm_interval1=0.05, cm_interval2=0.1, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.6,cm_max=0.6,cm_interval1=0.1,cm_interval2=0.1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['clivi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
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
                cm_min=-400,cm_max=400,cm_interval1=50,cm_interval2=100,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rsutcs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-100, cm_max=0, cm_interval1=10, cm_interval2=10, cmap='Greens')
            extend = 'min'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
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
            pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG',)
        elif var2 in ['rsds', 'rsdscs']:
            pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        elif var2 in ['rlns', 'rlnscs']:
            pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
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
            pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
        else:
            print('Warning: no colorbar specified')
        
        ds = {}
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
        elif var1=='q2m':
            era5_sp = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'] = specific_humidity_from_dewpoint(era5_sp * units.Pa, era5_d2m * units.K) * 1000
        elif var1=='mtuwswrf':
            era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year1}/mtnswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            ds['ERA5'] = era5_mtnswrf - era5_mtdwswrf
        elif var1=='mtuwswrfcs':
            era5_mtnswrfcs = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrfcs/{year1}/mtnswrfcs_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrfcs'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            ds['ERA5'] = era5_mtnswrfcs - era5_mtdwswrf
        elif var1=='si10':
            era5_u10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['u10'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_v10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['v10'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'] = (era5_u10**2 + era5_v10**2)**0.5
        elif var1 in ['tp', 'e', 'pev', 'mslhf', 'msshf', 'mtnlwrf', 'msdwlwrf', 'mtdwswrf', 'msdwswrfcs', 'msdwswrf', 'msnlwrf', 'mtnlwrfcs', 'msdwlwrfcs', ]:
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year1}/{var1}_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')[var1].sel(time=pd.Timestamp(year1,month1,day1,hour1))
        elif var1 in ['skt', 'blh', 'msl', 'tcwv', 'hcc', 'mcc', 'lcc', 'tcc', 'tclw', 'tciw']:
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
        else:
            print('Warning: var file not found')
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
        
        if var1 in ['e', 'pev', 'mper']:
            ds['ERA5'] *= (-1)
        
        for ires in suite_res[isuite]:
            # ires = 'd11km'
            # ['d11km', 'd4p4km', 'd4p4kms', 'd1p1km']
            print(f'#-------- {ires}')
            
            ds[ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0])
            
            assert (ds[ires]['T1HR'].values == ds[ires][nrad_time[ires]].values).all()
            assert (ds[ires]['T1HR'].values == ds[ires]['T1HR_MN'].values).all()
            assert (ds[ires]['grid_latitude_t'].values == ds[ires]['grid_latitude_cu'].values).all()
            assert (ds[ires]['grid_longitude_t'].values == ds[ires]['grid_longitude_cv'].values).all()
            
            time = ds[ires]['T1HR'].values
            lat = ds[ires]['grid_latitude_t'].values
            lon = ds[ires]['grid_longitude_t'].values
            rho80 = ds[ires]['TH_1_80_eta_rho'].values
            theta80 = ds[ires]['TH_1_80_eta_theta'].values
            
            ds[ires] = ds[ires].rename_dims({
                'T1HR': 'time',
                nrad_time[ires]: 'time',
                'T1HR_MN': 'time',
                'grid_latitude_t': 'lat',
                'grid_longitude_t': 'lon',
                'grid_latitude_cu': 'lat',
                'grid_longitude_cv': 'lon',
                'TH_1_80_eta_rho': 'rho80',
                'TH_1_80_eta_theta': 'theta80',
                })
            
            ds[ires] = ds[ires].squeeze().reset_coords(drop=True)
            ds[ires] = ds[ires].drop_vars([
                'T1HR', nrad_time[ires], 'T1HR_MN',
                'grid_latitude_t', 'grid_longitude_t',
                'grid_latitude_cu', 'grid_longitude_cv',
                'TH_1_80_eta_rho', 'TH_1_80_eta_theta',
                'rotated_latitude_longitude',
                ] + [var for var in ds[ires].data_vars if var.endswith("bounds")])
            
            ds[ires] = ds[ires].assign_coords({
                'time': time,
                'lat': lat,
                'lon': lon,
                'rho80': rho80,
                'theta80': theta80,
                })
            
            if var2 == 'orog':
                ds[ires] = ds[ires][stash_var[var2]]
            else:
                ds[ires] = ds[ires][stash_var[var2]].sel(time=pd.Timestamp(year,month,day,hour))
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ires] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ires] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ires] *= (-1)
            elif var2 in ['psl']:
                ds[ires] /= 100
            elif var2 in ['huss']:
                ds[ires] *= 1000
            elif var2 in ['cll', 'clm', 'clh', 'clt']:
                ds[ires] *= 100
        
        for imode in ['org', 'diff']:
            # imode = 'diff'
            print(f'#-------- {imode}')
            
            if imode=='org':
                plt_colnames = list(ds.keys())
            elif imode=='diff':
                plt_colnames = ['ERA5'] + [f'{suite_res[isuite][0]} - ERA5'] + [f'{suite_res[isuite][1]} - {suite_res[isuite][0]}']
            
            opng = f"figures/4_um/4.1_access_ram3/4.1.0_case studies/4.1.0.0_{year}-{month}-{day}-{hour} {var2} in {isuite} {', '.join(suite_res[isuite])}, and ERA5, {imode} {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1}.png"
            
            fig, axs = plt.subplots(
                nrow, ncol,
                figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
            
            for jcol in range(ncol):
                axs[jcol] = regional_plot(extent=[min_lon1, max_lon1, min_lat1, max_lat1], central_longitude=180, ax_org=axs[jcol], lw=0.1)
                axs[jcol].add_patch(Rectangle(
                    (min_lon, min_lat), max_lon-min_lon, max_lat-min_lat,
                    ec='red', color='None', lw=0.5,
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (ds[suite_res[isuite][0]].lon[0], ds[suite_res[isuite][0]].lat[0]),
                    ds[suite_res[isuite][0]].lon[-1] - ds[suite_res[isuite][0]].lon[0],
                    ds[suite_res[isuite][0]].lat[-1] - ds[suite_res[isuite][0]].lat[0],
                    ec='red', color='None', lw=0.5, linestyle='--',
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (ds[suite_res[isuite][1]].lon[0], ds[suite_res[isuite][1]].lat[0]),
                    ds[suite_res[isuite][1]].lon[-1] - ds[suite_res[isuite][1]].lon[0],
                    ds[suite_res[isuite][1]].lat[-1] - ds[suite_res[isuite][1]].lat[0],
                    ec='red', color='None', lw=0.5, linestyle=':',
                    transform=ccrs.PlateCarree(), zorder=2))
            
            if imode=='org':
                for jcol, ids in enumerate(ds.keys()):
                    print(f'#---- {jcol} {ids}')
                    plt_mesh = axs[jcol].pcolormesh(
                        ds[ids].lon,
                        ds[ids].lat,
                        ds[ids],
                        norm=pltnorm, cmap=pltcmp,
                        transform=ccrs.PlateCarree(), zorder=1)
                    cbar = fig.colorbar(
                        plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks, extend=extend,
                        cax=fig.add_axes([1/3, fm_bottom-0.115, 1/3, 0.03]))
                    cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                    cbar.ax.tick_params(labelsize=9, pad=1)
            elif imode=='diff':
                plt_mesh = axs[0].pcolormesh(
                    ds['ERA5'].lon, ds['ERA5'].lat, ds['ERA5'],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                for jcol, ids1, ids2 in zip(range(1, ncol), list(ds.keys())[1:], list(ds.keys())[:-1]):
                    print(f'#-------- {jcol} {ids1} {ids2}')
                    if not f'{ids1} - {ids2}' in regridder.keys():
                        regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                            ds[ids1],
                            ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])),
                            method='bilinear')
                    plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1]) - ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1]))
                    rmse = np.sqrt(np.square(plt_data).weighted(np.cos(np.deg2rad(plt_data.lat))).mean()).values
                    plt_colnames[jcol] = f'{plt_colnames[jcol]}, RMSE: {np.round(rmse, 2)}'
                    plt_mesh2 = axs[jcol].pcolormesh(
                        plt_data.lon, plt_data.lat, plt_data,
                        norm=pltnorm2, cmap=pltcmp2,
                        transform=ccrs.PlateCarree(), zorder=1)
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.05, fm_bottom-0.115, 0.4, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
                cbar2 = fig.colorbar(
                    plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.55, fm_bottom-0.115, 0.4, 0.03]))
                cbar2.ax.set_xlabel(f"Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}",
                                    fontsize=9, labelpad=1)
                cbar2.ax.tick_params(labelsize=9, pad=1)
            
            for jcol in range(ncol):
                axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
            
            fig.text(0.5, fm_bottom-0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='top')
            fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng)
        
        del ds


'''
'''
# endregion


# region check data


def preprocess_umoutput(ds_in):
    if 'T1HR_MN_rad_diag' in list(ds_in.dims):
        ds_in = ds_in.rename({'T1HR_MN_rad_diag': 'T1HR_MN_rad'})
    
    assert (ds_in['T1HR'].values == ds_in['T1HR_MN_rad'].values).all()
    assert (ds_in['T1HR'].values == ds_in['T1HR_MN'].values).all()
    assert (ds_in['grid_latitude_t'].values == ds_in['grid_latitude_cu'].values).all()
    assert (ds_in['grid_longitude_t'].values == ds_in['grid_longitude_cv'].values).all()
    
    time = ds_in['T1HR'].values
    lat = ds_in['grid_latitude_t'].values
    lon = ds_in['grid_longitude_t'].values
    rho80 = ds_in['TH_1_80_eta_rho'].values
    theta80 = ds_in['TH_1_80_eta_theta'].values
    
    ds_in = ds_in.rename_dims({
        'T1HR': 'time',
        'T1HR_MN_rad': 'time',
        'T1HR_MN': 'time',
        'grid_latitude_t': 'lat',
        'grid_longitude_t': 'lon',
        'grid_latitude_cu': 'lat',
        'grid_longitude_cv': 'lon',
        'TH_1_80_eta_rho': 'rho80',
        'TH_1_80_eta_theta': 'theta80',
        })
    
    ds_in = ds_in.squeeze().reset_coords(drop=True)
    ds_in = ds_in.drop_vars([
        'T1HR', 'T1HR_MN_rad', 'T1HR_MN',
        'grid_latitude_t', 'grid_longitude_t',
        'grid_latitude_cu', 'grid_longitude_cv',
        'TH_1_80_eta_rho', 'TH_1_80_eta_theta',
        'rotated_latitude_longitude',
        ] + [var for var in ds_in.data_vars if var.endswith("bounds")])
    
    ds_in = ds_in.assign_coords({
        'time': time,
        'lat': lat,
        'lon': lon,
        'rho80': rho80,
        'theta80': theta80,
        })
    
    return(ds_in)


stash2var = {
    'STASH_m01s09i203': 'cll',
    'STASH_m01s09i204': 'clm',
    'STASH_m01s09i205': 'clh',
    'STASH_m01s09i216': 'clt',
    'STASH_m01s30i461': 'prw',
    'STASH_m01s00i024': 'ts',
    'STASH_m01s03i236': 'tas',
    'STASH_m01s03i237': 'huss',
    'STASH_m01s03i245': 'hurs',
    'STASH_m01s00i010': 'hus',
    'STASH_m01s16i004': 'ta',
    'STASH_m01s00i002': 'ua',
    'STASH_m01s00i003': 'va',
    'STASH_m01s00i150': 'wa',
    'STASH_m01s30i008': 'wap',
    'STASH_m01s00i004': 'theta',
    'STASH_m01s00i012': 'qcf',
    'STASH_m01s00i254': 'qcl',
    'STASH_m01s16i206': 'qc',
    'STASH_m01s00i272': 'qr',
    'STASH_m01s16i207': 'qt',
    'STASH_m01s00i391': 'mv',
    'STASH_m01s00i392': 'mcl',
    'STASH_m01s00i393': 'mcf',
    'STASH_m01s00i394': 'mr',
    'STASH_m01s00i395': 'mg',
    'STASH_m01s00i396': 'mcf2',
    'STASH_m01s00i408': 'pa',
    'STASH_m01s03i234': 'hfls',
    'STASH_m01s03i217': 'hfss',
    'STASH_m01s02i205': 'rlut',
    'STASH_m01s00i238': 'rlds',
    'STASH_m01s01i205': 'rsut',
    'STASH_m01s01i207': 'rsdt',
    'STASH_m01s01i209': 'rsutcs',
    'STASH_m01s01i210': 'rsdscs',
    'STASH_m01s01i235': 'rsds',
    'STASH_m01s02i201': 'rlns',
    'STASH_m01s02i206': 'rlutcs',
    'STASH_m01s02i208': 'rldscs',
    'STASH_m01s16i222': 'psl',
    'STASH_m01s00i025': 'blh',
    'STASH_m01s00i030': 'iland',
    'STASH_m01s00i033': 'orog',
    'STASH_m01s00i239': 'rlu_t_s',
    'STASH_m01s03i250': 'das',
    'STASH_m01s03i513': 'blendingw',
    'STASH_m01s04i118': 'radar_reflectivity',
    'STASH_m01s04i224': 'clslw',
    'STASH_m01s20i114': 'CAPE',
    'STASH_m01s30i405': 'clwvi',
    'STASH_m01s30i406': 'clivi',
    }

stash2var_gal = stash2var | {
    'STASH_m01s04i210': 'ncloud',
    'STASH_m01s05i216': 'pr',
    'STASH_m01s05i215': 'snow',
    'STASH_m01s05i214': 'rain',
    }

stash2var_ral = stash2var | {
    'STASH_m01s00i271': 'qs',
    'STASH_m01s00i273': 'qg',
    'STASH_m01s00i075': 'ncloud',
    'STASH_m01s00i076': 'nrain',
    'STASH_m01s00i078': 'nice',
    'STASH_m01s00i079': 'nsnow',
    'STASH_m01s00i081': 'ngraupel',
    'STASH_m01s04i203': 'rain',
    'STASH_m01s04i204': 'snow',
    'STASH_m01s04i212': 'graupel',
    'STASH_m01s04i304': 'snow_graupel',
    }

var2stash_gal = {stash2var_gal[ikey]: ikey  for ikey in stash2var_gal.keys()}
var2stash_ral = {stash2var_ral[ikey]: ikey  for ikey in stash2var_ral.keys()}




year, month, day, hour = 2020, 6, 2, 4
isuite = 'u-dq700'
ds = {}
ds['d11km'] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/d11km/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).rename(stash2var_gal)
ds['d4p4km'] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/d4p4km/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).rename(stash2var_ral)

for istash in [istash for istash in ds['d11km'].data_vars if istash.startswith('STASH')]:
    print(f'#-------------------------------- {istash}')
    print(ds['d11km'][istash])
for istash in [istash for istash in ds['d4p4km'].data_vars if istash.startswith('STASH')]:
    print(f'#-------------------------------- {istash}')
    print(ds['d4p4km'][istash])





'''
for istash in var_stash.keys():
    # istash = 'STASH_m01s00i271'
    print(f'#---------------- {istash}: {var_stash[istash]}')
    
    print(f'#-------- d11km')
    try:
        print(f'{ds['d11km'][istash].attrs['long_name']} [{ds['d11km'][istash].attrs['units']}]')
    except KeyError:
        try:
            print(f'{ds['d11km'][istash].attrs['long_name']}')
        except KeyError:
            print('Warning: no variable')
    
    print(f'#-------- d4p4km')
    try:
        print(f'{ds['d4p4km'][istash].attrs['long_name']} [{ds['d4p4km'][istash].attrs['units']}]')
    except KeyError:
        try:
            print(f'{ds['d4p4km'][istash].attrs['long_name']}')
        except KeyError:
            print('Warning: no variable')

# d11km does not have qs, qg, ncloud, nrain, nice, nsnow, ngraupel

stash_var = {
    'cll':      'STASH_m01s09i203',
    'clm':      'STASH_m01s09i204',
    'clh':      'STASH_m01s09i205',
    'clt':      'STASH_m01s09i216',
    'prw':      'STASH_m01s30i461',
    'ts':      'STASH_m01s00i024',
    'tas':      'STASH_m01s03i236',
    'huss':     'STASH_m01s03i237',
    'hurs':     'STASH_m01s03i245',
    'hus':      'STASH_m01s00i010',
    'ta':       'STASH_m01s16i004',
    'ua':       'STASH_m01s00i002',
    'va':       'STASH_m01s00i003',
    'wa':       'STASH_m01s00i150',
    'wap':      'STASH_m01s30i008',
    'theta':    'STASH_m01s00i004',
    'qcf':      'STASH_m01s00i012',
    'qcl':      'STASH_m01s00i254',
    'qc':       'STASH_m01s16i206',
    'qs':       'STASH_m01s00i271',
    'qr':       'STASH_m01s00i272',
    'qg':       'STASH_m01s00i273',
    'qt':       'STASH_m01s16i207',
    'mv':       'STASH_m01s00i391',
    'mcl':      'STASH_m01s00i392',
    'mcf':      'STASH_m01s00i393',
    'mr':       'STASH_m01s00i394',
    'mg':       'STASH_m01s00i395',
    'mcf2':     'STASH_m01s00i396',
    'pa':       'STASH_m01s00i408',
    'hfls':     'STASH_m01s03i234',
    'hfss':     'STASH_m01s03i217',
    'rlut':     'STASH_m01s02i205',
    'rlds':     'STASH_m01s00i238',
    'rsut':     'STASH_m01s01i205',
    'rsdt':     'STASH_m01s01i207',
    'rsutcs':   'STASH_m01s01i209',
    'rsdscs':   'STASH_m01s01i210',
    'rsds':     'STASH_m01s01i235',
    'rlns':     'STASH_m01s02i201',
    'rlutcs':   'STASH_m01s02i206',
    'rldscs':   'STASH_m01s02i208',
    'psl':      'STASH_m01s16i222',
    'blh':      'STASH_m01s00i025',
    'iland':    'STASH_m01s00i030',
    'orog':     'STASH_m01s00i033',
    'ncloud':   'STASH_m01s00i075',
    'nrain':    'STASH_m01s00i076',
    'nice':     'STASH_m01s00i078',
    'nsnow':    'STASH_m01s00i079',
    'ngraupel': 'STASH_m01s00i081',
    'rlu_t_s':  'STASH_m01s00i239',
    'das':      'STASH_m01s03i250',
    'blendingw':'STASH_m01s03i513',
    'radar_reflectivity':   'STASH_m01s04i118',
    'clslw':    'STASH_m01s04i224',
    'CAPE':     'STASH_m01s20i114',
    'clwvi':    'STASH_m01s30i405',
    'clivi':    'STASH_m01s30i406',
    # 'pr':   'STASH_m01s00i348',
}
var_stash = {stash_var[ikey]: ikey  for ikey in stash_var.keys()}

var_stash2 = var_stash.copy()
for istash in ['STASH_m01s00i271', 'STASH_m01s00i273', 'STASH_m01s00i075', 'STASH_m01s00i076', 'STASH_m01s00i078', 'STASH_m01s00i079', 'STASH_m01s00i081']:
    var_stash2.pop(istash)

for istash in ['STASH_m01s04i201', 'STASH_m01s04i202', 'STASH_m01s04i203', 'STASH_m01s04i204', 'STASH_m01s20i013']:
    print(np.nanmax(ds['d4p4km'][istash]))

['ncloud', 'nrain', 'nice', 'nsnow', 'ngraupel']
ds['d4p4km'][stash_var['ncloud']]
print(np.max(ds['d4p4km'][stash_var['ncloud']])) # 1.4994637e+08
print(np.unique(ds['d4p4km'][stash_var['ncloud']]))
# [0.0000000e+00 1.0485760e+06 2.0971520e+06 3.1457280e+06 4.1943040e+06
#  5.2428800e+06 6.2914560e+06 7.3400320e+06 8.3886080e+06 9.4371840e+06
#  1.0485760e+07 1.1534336e+07 1.2582912e+07 1.3631488e+07 1.4680064e+07
#  1.5728640e+07 1.6777216e+07 1.7825792e+07 1.8874368e+07 1.9922944e+07
#  2.0971520e+07 2.2020096e+07 2.3068672e+07 2.4117248e+07 2.5165824e+07
#  2.6214400e+07 2.7262976e+07 2.8311552e+07 2.9360128e+07 3.0408704e+07
#  3.1457280e+07 3.2505856e+07 3.3554432e+07 3.4603008e+07 3.5651584e+07
#  3.6700160e+07 3.7748736e+07 3.8797312e+07 3.9845888e+07 4.0894464e+07
#  4.1943040e+07 4.2991616e+07 4.4040192e+07 4.5088768e+07 4.6137344e+07
#  4.7185920e+07 4.8234496e+07 4.9283072e+07 5.0331648e+07 5.1380224e+07
#  5.2428800e+07 5.3477376e+07 5.4525952e+07 5.5574528e+07 5.6623104e+07
#  5.7671680e+07 5.8720256e+07 5.9768832e+07 6.0817408e+07 6.1865984e+07
#  6.2914560e+07 6.3963136e+07 6.5011712e+07 6.6060288e+07 6.7108864e+07
#  6.8157440e+07 6.9206016e+07 7.0254592e+07 7.1303168e+07 7.2351744e+07
#  7.3400320e+07 7.4448896e+07 7.5497472e+07 7.6546048e+07 7.7594624e+07
#  7.8643200e+07 7.9691776e+07 8.0740352e+07 8.1788928e+07 8.2837504e+07
#  8.3886080e+07 8.4934656e+07 8.5983232e+07 8.7031808e+07 8.8080384e+07
#  8.9128960e+07 9.0177536e+07 9.1226112e+07 9.2274688e+07 9.3323264e+07
#  9.4371840e+07 9.5420416e+07 9.6468992e+07 9.7517568e+07 9.8566144e+07
#  9.9614720e+07 1.0066330e+08 1.0171187e+08 1.0276045e+08 1.0380902e+08
#  1.0485760e+08 1.0590618e+08 1.0695475e+08 1.0800333e+08 1.0905190e+08
#  1.1010048e+08 1.1114906e+08 1.1219763e+08 1.1324621e+08 1.1429478e+08
#  1.1534336e+08 1.1639194e+08 1.1744051e+08 1.1848909e+08 1.1953766e+08
#  1.2058624e+08 1.2163482e+08 1.2268339e+08 1.2373197e+08 1.2478054e+08
#  1.2582912e+08 1.2687770e+08 1.2792627e+08 1.2897485e+08 1.3002342e+08
#  1.3107200e+08 1.3212058e+08 1.3316915e+08 1.3421773e+08 1.3526630e+08
#  1.3631488e+08 1.3736346e+08 1.3841203e+08 1.3946061e+08 1.4050918e+08
#  1.4155776e+08 1.4260634e+08 1.4365491e+08 1.4470349e+08 1.4575206e+08
#  1.4680064e+08 1.4784922e+08 1.4889779e+08 1.4994637e+08]
print(np.unique(ds['d4p4km'][stash_var['nrain']])) # [      0. 1048576. 2097152.]
print(np.unique(ds['d4p4km'][stash_var['nice']])) # [      0. 1048576. 2097152. 3145728. 4194304. 5242880.]
print(np.unique(ds['d4p4km'][stash_var['nsnow']])) # [      0. 1048576. 2097152. 3145728.]
print(np.unique(ds['d4p4km'][stash_var['ngraupel']])) # [      0. 1048576.]

np.unique(ds['d11km']['STASH_m01s04i210'].values)
# CDNC has only two values in d11km: array([      0., 5242880.], dtype=float32)

# rlds == rlds2
print(np.max(abs(ds['d11km'][stash_var['rlds2']].values - ds['d11km'][stash_var['rlds']].values) / ds['d11km'][stash_var['rlds']].values))
print(np.max(abs(ds['d4p4km'][stash_var['rlds2']].values - ds['d4p4km'][stash_var['rlds']].values) / ds['d4p4km'][stash_var['rlds']].values))

# qt = hus + ...
print(np.nanmean(abs(ds['d11km'][stash_var['hus']].values - ds['d11km'][stash_var['qt']].values) / ds['d11km'][stash_var['qt']].values))
print(np.nanmean(abs(ds['d4p4km'][stash_var['hus']].values - ds['d4p4km'][stash_var['qt']].values) / ds['d4p4km'][stash_var['qt']].values))
print(np.nanmean((ds['d11km'][stash_var['hus']].values - ds['d11km'][stash_var['qt']].values) / ds['d11km'][stash_var['qt']].values))
print(np.nanmean((ds['d4p4km'][stash_var['hus']].values - ds['d4p4km'][stash_var['qt']].values) / ds['d4p4km'][stash_var['qt']].values))

# d4p4km does not have pr, CDNC, rain, snow, pr2
# TOT PRECIP RATE AFTER TSTEP  KG/M2/S
# CLOUD DROP NUMBER CONC. /m3
# TOTAL RAINFALL RATE: LS+CONV KG/M2/S
# TOTAL SNOWFALL RATE: LS+CONV KG/M2/S
# TOTAL PRECIPITATION RATE     KG/M2/S

print(((ds['d11km']['STASH_m01s05i214'].values + ds['d11km']['STASH_m01s05i215'].values) == ds['d11km']['STASH_m01s05i216'].values).all())
print(np.max(abs((ds['d11km']['STASH_m01s05i214'].values + ds['d11km']['STASH_m01s05i215'].values) - ds['d11km']['STASH_m01s05i216'].values)))
print(np.mean(abs((ds['d11km']['STASH_m01s05i214'].values + ds['d11km']['STASH_m01s05i215'].values) - ds['d11km']['STASH_m01s05i216'].values)))
print(np.mean(ds['d11km']['STASH_m01s05i216'].values))


print((ds['d11km']['STASH_m01s00i348'].values == ds['d11km']['STASH_m01s05i216'].values).all())
print(np.mean(abs(ds['d11km']['STASH_m01s00i348'].values - ds['d11km']['STASH_m01s05i216'].values)) / np.mean(ds['d11km']['STASH_m01s05i216'].values))


ds = {}
ds['d12km'] = xr.open_dataset('cylc-run/u-dq126/share/cycle/20200601T0000Z/Flagship_ERA5to1km/12km/GAL9/um/umnsaa_pa000.nc')
ds['d5km'] = xr.open_dataset('cylc-run/u-dq126/share/cycle/20200601T0000Z/Flagship_ERA5to1km/5km/RAL3P2/um/umnsaa_pa000.nc')
# ds['d1km'] = xr.open_dataset('cylc-run/u-dq126/share/cycle/20200601T0000Z/Flagship_ERA5to1km/1km/RAL3P2/um/umnsaa_pa000.nc')

ds['d12km'] = preprocess_umoutput(ds['d12km']).rename(stash2var_gal)
ds['d5km'] = preprocess_umoutput(ds['d5km']).rename(stash2var_ral)
for istash in [istash for istash in ds['d12km'].data_vars if istash.startswith('STASH')]:
    print(f'#-------------------------------- {istash}')
    print(ds['d12km'][istash])
for istash in [istash for istash in ds['d5km'].data_vars if istash.startswith('STASH')]:
    print(f'#-------------------------------- {istash}')
    print(ds['d5km'][istash])

ds['d12km']['rlds'][0] == ds['d12km']['STASH_m01s02i207'][0].values
np.max(abs(ds['d12km']['rlds'][0] - ds['d12km']['STASH_m01s02i207'][0].values) / ds['d12km']['STASH_m01s02i207'][0].values)
np.max(abs(ds['d5km']['rlds'][0] - ds['d5km']['STASH_m01s02i207'][0].values) / ds['d5km']['STASH_m01s02i207'][0].values)

'''
# endregion



