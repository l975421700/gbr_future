

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+gdata/ra22


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
from xmip.preprocessing import rename_cmip6, broadcast_lonlat, correct_lon, promote_empty_dims, replace_x_y_nominal_lat_lon, correct_units, correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds, combined_preprocessing

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature

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
import re
import glob
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
    month,
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
    cdo_regrid,
    time_weighted_mean)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot himawari data

year=2020
month=7
day=1
hour=12
minute=0
ioption='true_color'
# ioption='night_microphysics'

dfolder = '/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest'
opng = f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari {ioption} {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png'

if ioption == 'true_color':
    bands = ['B01', 'B02', 'B03']
    plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\n True Color RGB Himawari 8/9'
elif ioption == 'night_microphysics':
    bands = ['B07', 'B13', 'B15']
    plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\n Night Microphysics RGB Himawari 8/9'

channels = {}
for iband in bands:
    # iband='B03'
    print(f'#-------------------------------- {iband}')
    
    ifile = sorted(glob.glob(f'{dfolder}/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/*OBS_{iband}*'))[-1]
    channels[iband] = xr.open_dataset(ifile)
    
    var_name = [var for var in channels[iband].data_vars if var.startswith('channel_00')][0]
    channels[iband] = channels[iband][var_name].squeeze()
    
    if iband == 'B03':
        channels[iband] = channels[iband].coarsen(y=2, x=2, boundary='trim').mean()

extent=[channels[iband].x.min().values, channels[iband].x.max().values, channels[iband].y.min().values, channels[iband].y.max().values]
transform=ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

if ioption == 'true_color':
    rgb = np.dstack([channels['B03'], channels['B02'], channels['B01']])

rgb = np.clip(rgb, 0, 1)
rgb = np.power(rgb, 1/2.2)

start_time = time.perf_counter()

fig, ax = regional_plot(extent=[80, 200, -60, 60], central_longitude=180, figsize = np.array([6.6, 6.6+0.9])/2.54, border_color='yellow', lw=0.1)

ax.imshow(rgb, extent=extent, transform=transform, interpolation='none')

plt.text(
        0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
        ha='center', va='top', rotation='horizontal', linespacing=1.5)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.9/(6.6+0.9), top=0.99)
fig.savefig(opng)

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")




import rioxarray as rxr
orig = rxr.open_rasterio(ifile, masked=True)


'''
'''
# endregion
