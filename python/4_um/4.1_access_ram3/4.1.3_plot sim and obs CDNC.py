

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=20GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60


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
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
from metpy.interpolate import cross_section
from haversine import haversine, Unit

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

from um_postprocess import (
    preprocess_umoutput,
    stash2var, stash2var_gal, stash2var_ral,
    var2stash, var2stash_gal, var2stash_ral,
    suite_res,)

# endregion



# region get sim and obs CDNC

year, month, day, hour = 2020, 6, 2, 4
suites = ['u-dq987', 'u-dr040', 'u-dr041']
var2s = ['ncloud']

# min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
# up to Willis Island:
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -16.2876, -7.01
max_altitude = 12000

doy = datetime(year, month, day).timetuple().tm_yday
ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour
ptime = pd.Timestamp(year,month,day,hour) - pd.Timedelta('1h')
year0, month0, day0, hour0 = ptime.year, ptime.month, ptime.day, ptime.hour
doy0 = datetime(year0, month0, day0).timetuple().tm_yday

try:
    ifile = glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/2B-CWC-RO.P1_R05/{year}/{doy:03d}/{year}{doy}{hour:02d}*.hdf')[0]
except IndexError:
    ifile = glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/2B-CWC-RO.P1_R05/{year0}/{doy0:03d}/{year0}{doy0}{hour0:02d}*.hdf')[0]
hdf_vs = HDF(ifile).vstart()
lat = np.array(hdf_vs.attach('Latitude')[:]).squeeze()
lon = np.array(hdf_vs.attach('Longitude')[:]).squeeze()
mask = (lon >= min_lon) & (lon <= max_lon) & \
    (lat >= min_lat) & (lat <= max_lat)
startpoint = [lat[mask][0], lon[mask][0]]
endpoint = [lat[mask][-1], lon[mask][-1]]
steps = int(haversine(startpoint,endpoint,unit='km')/4.4)

um_output = {}
for isuite in suites:
    # isuite='u-dq987'
    # ['u-dq700', 'u-dq788', 'u-dq911', 'u-dq912', 'u-dq799', 'u-dq987', 'u-dr040', 'u-dr041']
    print(f'#-------------------------------- {isuite}')
    um_output[isuite] = {}
    # ires = suite_res[isuite][1]
    # um_output[isuite][ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).rename(stash2var_ral).sel(time=pd.Timestamp(year,month,day,hour)).metpy.parse_cf()
    ires = suite_res[isuite][0]
    um_output[isuite][ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).rename(stash2var_gal).sel(time=pd.Timestamp(year,month,day,hour)).metpy.parse_cf()
    
    for var2 in var2s:
        # var2 = 'ncloud'
        print(f'#---------------- {var2}')
        
        opng = f'figures/4_um/4.1_access_ram3/4.1.2_vertical profiles/4.1.2.0 {isuite} {ires} {var2} {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC {min_lon}_{max_lon}_{min_lat}_{max_lat} below 200 hPa.png'
        
        if var2 == 'ncloud':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20,
                cmap='viridis_r',)
            extend = 'max'
        else:
            continue
        
        um_data = cross_section(um_output[isuite][ires][var2], startpoint, endpoint, steps).transpose() / 10**6
        um_pressure = cross_section(um_output[isuite][ires]['pa'], startpoint, endpoint, steps).transpose() / 100
        um_lat_2d = np.tile(um_data.lat.values[:, np.newaxis],
                            (1, um_data.shape[1]))
        
        fm_bottom = 0.35
        fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
        plt_mesh = ax.pcolormesh(
            um_lat_2d, um_pressure, um_data,
            norm=pltnorm, cmap=pltcmp,)
        
        ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.set_ylabel(f'Pressure [$hPa$]', labelpad=-2)
        ax.invert_yaxis()
        ax.set_ylim(1000, 200)
        ax.set_yticks(np.arange(1000, 200 - 1e-4, -200))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--',)
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.05, fm_bottom-0.11, 0.9, 0.03]))
        cbar.ax.set_xlabel(f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC\n{era5_varlabels[var2]}\n{isuite} {ires}', linespacing=1.5)
        
        fig.subplots_adjust(left=0.14,right=0.99,bottom=fm_bottom,top=0.97)
        fig.savefig(opng)




'''

'''
# endregion

