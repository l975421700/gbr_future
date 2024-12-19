

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
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
    plot_loc,
)


# endregion


# region plot the globe

igra2_station = pd.read_fwf(
    'https://www1.ncdc.noaa.gov/pub/data/igra/igra2-station-list.txt',
    names=['id', 'lat', 'lon', 'altitude', 'name', 'starty', 'endy', 'count'])

subset = igra2_station.loc[[sid.startswith('AS') for sid in igra2_station['id']]]

fig, ax = globe_plot(figsize=np.array([17.6, 8.8]) / 2.54)

plot_loc(igra2_station['lon'], igra2_station['lat'], ax, s=6,lw=0.6)

fig.savefig('figures/test.png')


'''
ax.add_feature(
    cfeature.LAND, color='green', zorder=2, edgecolor=None,lw=0)
ax.add_feature(
    cfeature.OCEAN, color='blue', zorder=2, edgecolor=None,lw=0)
'''
# endregion


# region plot Australia

gbr_shp = gpd.read_file('data/others/Great_Barrier_Reef_Marine_Park_Boundary/Great_Barrier_Reef_Marine_Park_Boundary.shp')
WillisIsland_loc={'lat':-16.3,'lon':149.98}


lats = [-10.7, -24.5]
lons = [145, 154]

fig, ax = regional_plot(
    extent=[140, 155, -25, -10], figsize = np.array([6.6, 6.6]) / 2.54,
    ticks_and_labels = True, fontsize=10,)
gbr_shp.plot(ax=ax, edgecolor='tab:blue', facecolor='none', lw=0.8, zorder=2)
plot_loc(WillisIsland_loc['lon'], WillisIsland_loc['lat'], ax)

plot_loc(lons[0], lats[0], ax)
plot_loc(lons[1], lats[1], ax)

fig.savefig('figures/0_gbr/0.1_study region/0.0_gbr.png')




'''
ax.scatter(
    x = WillisIsland_loc['lon'], y = WillisIsland_loc['lat'],
    s=10, c='none', lw=0.8, marker='o', edgecolors='tab:blue', zorder=2,
    transform=ccrs.PlateCarree(),)

'''
# endregion

