

# qsub -I -q express -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22


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
import requests
import rioxarray
from datetime import datetime, timedelta
import glob
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
import joblib
import argparse

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
# import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

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

# endregion


# region plot tracks of CloudSat-CALIPSO for one day

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=86400-1, cm_interval1=3600, cm_interval2=3600, cmap='viridis',)

fm_bottom=1.5/(6+1.5)
fig, ax = globe_plot(figsize=np.array([12, 6+1.5]) / 2.54, fm_bottom=fm_bottom)

year=2017
doy = 119
date = datetime(year, 1, 1) + timedelta(days=doy - 1)
month = date.month
day = date.day


fl = sorted(glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/{doy}/*'))
for ifile in fl:
    # ifile=fl[0]
    hdf = HDF(ifile).vstart()
    lat = np.array(hdf.attach('Latitude')[:]).squeeze()
    lon = np.array(hdf.attach('Longitude')[:]).squeeze()
    seconds = hdf.attach('UTC_start')[:][0][0] + np.array(hdf.attach('Profile_time')[:]).squeeze()
    date_time = np.array([date + timedelta(seconds=s) for s in seconds])
    
    plt_scatter = ax.scatter(lon, lat, s=2, c=seconds, cmap=pltcmp, norm=pltnorm, lw=0, transform=ccrs.PlateCarree())

cbar = fig.colorbar(
    plt_scatter, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='max',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.9, 0.04]))
cbar.ax.set_xlabel(f'UTC on {str(date)[:10]} along the orbit of CloudSat/CALIPSO', labelpad=4)
cbar.ax.set_xticklabels(np.arange(0, 24, 1))

# iimage = f'data/others/Blue Marble Next Generation w: Topography and Bathymetry/world.topo.bathy.2004{month:02d}.3x5400x2700.jpg'
# img = Image.open(iimage)
# ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())

opng = f'figures/3_satellites/3.1_CloudSat_CALIPSO/3.1.0_Orbit of CloudSat and CALIPSO on {str(date)[:10]}.png'
fig.savefig(opng)



'''
hdf2 = SD(ifile, SDC.READ)
hdf2.datasets().keys()
CloudFraction = hdf2.select('CloudFraction').get()
CloudLayerType = hdf2.select('CloudLayerType').get()
CloudTypeQuality = hdf2.select('CloudTypeQuality').get()


stats.describe(seconds[1:] - seconds[:-1])
'''
# endregion


# region plot grid counts of 2B-CLDCLASS-LIDAR

cc_cldclass_count = xr.open_dataset(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/cc_cldclass_count.nc').cc_cldclass_count
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=60000+1e-4, cm_interval1=2000, cm_interval2=10000, cmap='viridis',)

fm_bottom=1.5/(6+1.5)
fig, ax = globe_plot(figsize=np.array([12, 6+1.5]) / 2.54, fm_bottom=fm_bottom)

plt_mesh = ax.pcolormesh(
    cc_cldclass_count.lon, cc_cldclass_count.lat, cc_cldclass_count,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='max',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.9, 0.04]))
cbar.ax.set_xlabel(f'Count of CloudSat/CALIPSO 2B-CLDCLASS-LIDAR observations', labelpad=4)
# cbar.ax.set_xticklabels(np.arange(0, 24, 1))

opng = f'figures/3_satellites/3.1_CloudSat_CALIPSO/3.1.0_CloudSat and CALIPSO 2B-CLDCLASS-LIDAR counts.png'
fig.savefig(opng)


'''
stats.describe(cc_cldclass_count.values, axis=None)
np.max(cc_cldclass_count)
(cc_cldclass_count == 0).sum() / 180 / 360 = 0.08888889
(cc_cldclass_count.sel(lat=slice(-80, 80)) == 0).sum()
'''
# endregion


# region check annual cycle of counts of 2B-CLDCLASS-LIDAR

geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')

monthly_counts = geolocation_all.date_time.dt.month.value_counts().sort_index()


'''
date_time
1     146096211
2     139433665
3     154127407
4     150753986
5     169138482
6     166030189
7     147512206
8     194844275
9     157537558
10    173294050
11    190798227
12    165380608
Name: count, dtype: int64
'''
# endregion

