

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
import glob

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

from calculations import (
    mon_sea_ann,
    cdo_regrid,)

# endregion


# region import data


CL_RelFreq = xr.open_dataset('scratch/data/obs/jaxa/clp/CL_RelFreq_2016.nc').CL_RelFreq

cloudtypes = [
    'Cirrus', 'Cirrostratus', 'Deep convection',
    'Altocumulus', 'Altostratus', 'Nimbostratus',
    'Cumulus', 'Stratocumulus', 'Stratus']


# endregion


# region plot the frequency

opng = 'figures/test.png'
nrow = 3
ncol = 3
fm_bottom = 1.5 / (6.6*nrow + 2)

cbar_label = r'Cloud occurrence frequency [$\%$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=30, cm_interval1=5, cm_interval2=5, cmap='Blues_r',)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([6.6*ncol, 6.6*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        print(cloudtypes[ipanel])
        axs[irow, jcol] = regional_plot(
            extent=[80, 200, -60, 60], central_longitude=180,
            ax_org=axs[irow, jcol],)
        
        plt.text(
            0, 1.02, f'{panel_labels[ipanel]} {cloudtypes[ipanel]}',
            transform=axs[irow, jcol].transAxes, fontsize=10,
            ha='left', va='bottom', rotation='horizontal')
        
        plt_data = CL_RelFreq.sel(types=cloudtypes[ipanel])
        plt_mesh = axs[irow, jcol].pcolormesh(
            plt_data.longitude, plt_data.latitude, plt_data,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        ipanel += 1

cbar = fig.colorbar(
    plt_mesh, # cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='max',
    anchor=(0.5, -0.56),)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(opng)


# endregion



