

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
from datetime import datetime

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=9)
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

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs, get_modis_latlonvar, get_modis_latlonvars

# endregion


# region

cwp = xr.open_dataset('/g/data/rv74/satellite-products/arc/der/himawari-ahi/cloud/cmic/latest/2020/06/02/S_NWC_CMIC_HIMA08_HIMA-N-NR_20200602T033000Z.nc')
ancil = xr.open_dataset('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')

trim = 1000
np.sum(np.isnan(ancil['lon'].squeeze().values[trim:(-1 * trim), trim:(-1 * trim)]))

# 110.58, 157.34, -43.69, -7.01
fig, ax = regional_plot(extent=[137.12, 157.3, -28.76, -7.05], figsize = np.array([4.4, 6.6])/2.54)
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=800, cm_interval1=50, cm_interval2=100,
    cmap='viridis_r') # 'Purples_r'

ax.pcolormesh(
    ancil['lon'].squeeze().values[trim:(-1 * trim), trim:(-1 * trim)],
    ancil['lat'].squeeze().values[trim:(-1 * trim), trim:(-1 * trim)],
    cwp['cmic_iwp'][trim:(-1 * trim), trim:(-1 * trim)] * 1000,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

fig.savefig('figures/test.png')




'''
cwp['cmic_iwp'].values
mask = np.isfinite(ancil['lon'].squeeze().values) & \
    np.isfinite(ancil['lat'].squeeze().values) & \
        (ancil['lon'].squeeze().values >= 110.58) & \
            (ancil['lon'].squeeze().values <= 157.34) & \
                (ancil['lat'].squeeze().values >= -43.69) & \
                    (ancil['lat'].squeeze().values <= -7.01)
'''
# endregion

