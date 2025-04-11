

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
from datetime import datetime

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
)

from calculations import (
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region animate cloud liquid/ice water

parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=2020; month=6

extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
ids = 'BARRA-C2'

if ids == 'BARRA-C2':
    clwvi = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/clwvi/latest/clwvi_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clwvi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    clivi = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/clivi/latest/clivi_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clivi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    prw = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/prw/latest/prw_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').prw.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

time_series = clwvi.time

max_value = 0.5
pltlevel = np.arange(0, max_value + 1e-4, 0.005)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
colors = np.ones((len(pltlevel)-1, 4))
colors[:, 3] = np.linspace(0, 1, len(pltlevel)-1)
pltcmp = ListedColormap(colors)


omp4 = f'figures/4_um/4.0_barra/4.0.2_cloud image/4.0.2.0 {ids} total column water, cloud water and ice {year}{month:02d}.mp4'

fig, ax = regional_plot(extent=extent, central_longitude=180,
                        figsize = np.array([8.8, 7.8]) / 2.54)
img = Image.open(f'data/others/Blue Marble Next Generation w: Topography and Bathymetry/world.topo.bathy.2004{month:02d}.3x5400x2700.jpg')
ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), zorder=0)

plt_objs = []
def update_frames(itime):
    # itime = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    day = time_series[itime].dt.day.values
    hour = time_series[itime].dt.hour.values
    minute = time_series[itime].dt.minute.values
    print(f'#-------------------------------- {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}')
    
    plt_mesh = ax.pcolormesh(
        clwvi.lon,
        clwvi.lat,
        clwvi.sel(time=datetime(year, month, day, hour, minute)) + clivi.sel(time=datetime(year, month, day, hour, minute)),
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    plt_text = ax.text(0.5, -0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nCloud liquid&ice water in {ids} (white: >{max_value} ' + r'[$kg \; m^{-2}$])',
        ha='center', va='top', transform=ax.transAxes, linespacing=1.3)
    plt_objs = [plt_mesh, plt_text]
    return(plt_objs)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.99)
ani = animation.FuncAnimation(
    fig, update_frames, frames=len(time_series), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4, progress_callback=lambda iframe, n: print(f'Frame {iframe}/{n}'))




# endregion
