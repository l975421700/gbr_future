

# check interpolation


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
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string

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
)


# endregion


# region plot data

with open('data/sim/cmip6/historical_Omon_tos.pkl', 'rb') as f:
    historical_Omon_tos = pickle.load(f)

models = list(historical_Omon_tos.keys())

output_png = 'figures/test.png'
cbar_label = r'CMIP6 $\mathit{historical}$' + ' monthly SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=28, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)

nrow = 10
ncol = 6
fm_bottom = 2 / (4.4*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(models)):
            panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
            model = models[jcol + ncol * irow]
            
            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
            
            axs[irow, jcol].text(
                -0.04, 1.04, model + '\n' + panel_label,
                ha='left', va='top', transform=axs[irow, jcol].transAxes,
                linespacing=1.6)
        else:
            axs[irow, jcol].set_visible(False)

# plot data
for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(models)):
            model = models[jcol + ncol * irow]
            print(model)
            
            plot_data = historical_Omon_tos[model]['ann'].sel(
                time=slice('1979', '2014')).mean(dim='time')
            
            plt_mesh = axs[irow, jcol].contourf(
                lon, lat, plot_data, levels=pltlevel, extend='both',
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, -0.84),)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.996, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)
plt.close()

# endregion

