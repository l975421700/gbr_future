

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
from metpy.plots import Hodograph, SkewT
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
mpl.rcParams['lines.linewidth'] = 0.5
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
    remove_trailing_zero_pos_abs,
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


# region import data

wrf_output = xr.open_dataset('data/sim/wrf/test3/2021120803/wrfout_d01_2021-12-08_03:00:00')
# wrf_output = xr.open_dataset('data/sim/wrf/test4/2021120803/wrfout_d01_2021-12-08_04:00:00')

# endregion


# region check data

pressure = (wrf_output.P + wrf_output.PB) * 0.01
geop_height = (wrf_output.PH + wrf_output.PHB) / 9.81
pot_temp = wrf_output.T + 300
qcloud = wrf_output.QCLOUD
qrain = wrf_output.QRAIN
qvapor = wrf_output.QVAPOR

temp = mpcalc.temperature_from_potential_temperature(pressure * units('hPa'), pot_temp * units('K'))
dz = geop_height[:,1:,:,:] - geop_height[:,:-1,:,:]
rho = mpcalc.density(pressure * units('hPa'), temp, qvapor,)

lwp = ((qcloud+qrain) * rho.values * dz.values).sum(dim='bottom_top')

lwp.to_netcdf('data/others/test.nc')


lwp = xr.open_dataset('data/others/test.nc')['__xarray_dataarray_variable__']
# lwp1 = xr.open_dataset('data/others/test1.nc')['__xarray_dataarray_variable__']

np.max(lwp.values)
# np.max(lwp1.values)


'''
stats.describe(wrf_output.QCLOUD.values, nan_policy='omit', axis=None)
stats.describe(wrf_output.CLDFRA.values, nan_policy='omit', axis=None)
stats.describe(wrf_output.QCLOUD[-1, ].mean(dim='bottom_top').values, nan_policy='omit', axis=None)

(dz.values == np.diff(geop_height, axis=1)).all()
wrf_output.THM
wrf_output.P_HYD
'''
# endregion

