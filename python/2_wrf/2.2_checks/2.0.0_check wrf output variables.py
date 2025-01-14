

expid = '20160427_4nests'


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
import glob

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

wrf_output_fl = sorted(glob.glob(f'data/sim/wrf/{expid}/*/wrfout_d04_*'))[:56]
wrf_output_fl = [value for idx, value in enumerate(wrf_output_fl) if (idx % 7) != 0]
start_time  = wrf_output_fl[0][-19:].replace("_", " ")
end_time    = wrf_output_fl[-1][-19:].replace("_", " ")

wrf_output = xr.open_mfdataset(wrf_output_fl,combine='nested',concat_dim='Time')
wrf_output['Time'] = pd.date_range(start=start_time, end=end_time, freq="10min")


'''
wrf_output = xr.open_dataset('data/sim/wrf/20160428_test/2016042800/wrfout_d01_2016-04-28_00:00:00')

var = 'LANDMASK'
wrf_output[var].values
np.max(wrf_output[var].values)
np.sum(wrf_output[var].values)

'''
# endregion


# region animate LWP

pressure = (wrf_output.P + wrf_output.PB) * 0.01
geop_height = (wrf_output.PH + wrf_output.PHB) / 9.81
pot_temp = wrf_output.T + 300
qcloud = wrf_output.QCLOUD
qrain = wrf_output.QRAIN
qvapor = wrf_output.QVAPOR

temp = mpcalc.temperature_from_potential_temperature(pressure * units('hPa'), pot_temp * units('K'))
dz = geop_height[:,1:,:,:] - geop_height[:,:-1,:,:]
rho = mpcalc.density(pressure * units('hPa'), temp, qvapor,)

lwp = ((qcloud+qrain) * rho.values * dz.values).sum(dim='bottom_top').squeeze() * 1000 * units('g/m**2')


omp4 = f'figures/2_wrf/2.0_clouds/2.0.0_{expid} LWP {start_time} to {end_time}.mp4'
cbar_label = r'LWP [$g \; m^{-2}$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='Blues',)

fig, ax = plt.subplots(1, 1, figsize=np.array([5, 6.8]) / 2.54,)

ims = []
for idx, itime in enumerate(wrf_output['Time'].values):
    print(f'{idx} {itime}')
    
    plt_mesh = ax.pcolormesh(
        np.arange(0,4800,50), np.arange(0,4800,50), lwp[idx],
        norm=pltnorm, cmap=pltcmp)
    plt_text = ax.text(
        0.5, -0.43,
        str(lwp[idx]['Time'].values)[:16].replace("T", " ") + 'UTC',
        transform=ax.transAxes, ha='center', va='center')
    
    ims.append([plt_mesh,plt_text])

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    fraction=0.14, pad=0.05,)
cbar.ax.set_xlabel(cbar_label)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gca().set_aspect('equal')

fig.subplots_adjust(left=0.02, right = 0.98, bottom = 0.16, top = 0.98)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    omp4,
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)


# endregion


# region plot LWP

pressure = (wrf_output.P + wrf_output.PB) * 0.01
geop_height = (wrf_output.PH + wrf_output.PHB) / 9.81
pot_temp = wrf_output.T + 300
qcloud = wrf_output.QCLOUD
qrain = wrf_output.QRAIN
qvapor = wrf_output.QVAPOR

temp = mpcalc.temperature_from_potential_temperature(pressure * units('hPa'), pot_temp * units('K'))
dz = geop_height[:,1:,:,:] - geop_height[:,:-1,:,:]
rho = mpcalc.density(pressure * units('hPa'), temp, qvapor,)

lwp = ((qcloud+qrain) * rho.values * dz.values).sum(dim='bottom_top').squeeze() * 1000 * units('g/m**2')


# plot it

output_png = 'figures/test.png'
cbar_label = r'LWP [$g \; m^{-2}$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues',)

fig, ax = plt.subplots(1, 1, figsize=np.array([5, 6.8]) / 2.54,)

plt_mesh = ax.pcolormesh(np.arange(0,4800,50), np.arange(0,4800,50),
                         lwp[-1], norm=pltnorm, cmap=pltcmp)

ax.text(0.5, -0.43, str(lwp[-1]['Time'].values)[:16].replace("T", " ") + 'UTC',
        transform=ax.transAxes, ha='center', va='center')

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=20, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    fraction=0.14, pad=0.05,)
cbar.ax.set_xlabel(cbar_label)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gca().set_aspect('equal')
fig.subplots_adjust(left=0.02, right = 0.98, bottom = 0.16, top = 0.98)
fig.savefig(output_png)


'''
lwp.to_netcdf('data/others/test.nc')
lwp = xr.open_dataset('data/others/test.nc')['__xarray_dataarray_variable__']
np.max(lwp.values)


stats.describe(wrf_output.QCLOUD.values, nan_policy='omit', axis=None)
stats.describe(wrf_output.CLDFRA.values, nan_policy='omit', axis=None)
stats.describe(wrf_output.QCLOUD[-1, ].mean(dim='bottom_top').values, nan_policy='omit', axis=None)

(dz.values == np.diff(geop_height, axis=1)).all()
wrf_output.THM
wrf_output.P_HYD
'''
# endregion

