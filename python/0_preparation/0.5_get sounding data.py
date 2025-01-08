

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

from metplot import (
    plot_wyoming_sounding,
    output_wyoming_sounding,
    plot_wyoming_sounding_vertical,
    )

# endregion


# region get and plot Wyoming sounding skew-T

date = datetime(2016, 4, 29, 0)
station = '94299'
output_png=f'figures/0_gbr/0.0_sounding/0.0.0 sounding {str(date)[:13]} {station}.png'

df = WyomingUpperAir.request_data(date, station)
plot_wyoming_sounding(df, date, output_png=output_png,)




'''
df = get_wyoming_sounding(date, station)
print(df.columns)
'''
# endregion


# region get and output Wyoming sounding

date = datetime(2016, 4, 29, 0)
station = '94299'
outf = f'data/sim/wrf/input/sounding/willis_island_{str(date)[:13]}'

df = WyomingUpperAir.request_data(date, station)
output_wyoming_sounding(df, outf)


'''
'''
# endregion


# region get and plot Wyoming sounding (T, theta, RH, qv)

date = datetime(2016, 4, 29, 0)
station = '94299'
output_png=f'figures/0_gbr/0.0_sounding/0.0.0 vertical profiles {str(date)[:13]} {station}.png'

df = WyomingUpperAir.request_data(date, station)
df = df[['pressure', 'temperature', 'dewpoint']].dropna(subset=('temperature', 'dewpoint'), how='all').reset_index(drop=True)

p = df['pressure'].values * units.hPa
T = (df['temperature'].values * units.degC).to(units('K'))
Td = (df['dewpoint'].values * units.degC).to(units('K'))

thta = mpcalc.potential_temperature(p, T)
RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
mixr = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH).to('g/kg')

icol=4

fig, axs = plt.subplots(1,icol,sharey=True,figsize=np.array([icol*2+3, 6.4]) / 2.54)

axs[0].plot(T, p, c='tab:blue', marker='o', markersize=1)
axs[1].plot(thta, p, c='tab:blue', marker='o', markersize=1)
axs[2].plot(RH/100, p, c='tab:blue', marker='o', markersize=1)
axs[3].plot(mixr, p, c='tab:blue', marker='o', markersize=1)

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(270, 310)
axs[1].set_xlim(290, 330)
axs[2].set_xlim(0, 1)
axs[3].set_xlim(0, 20)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'T [$K$]')
axs[1].set_xlabel(r'$\theta$ [$K$]')
axs[2].set_xlabel(r'RH [-]')
axs[3].set_xlabel(r'$q_v$ [$g\;kg^{-1}$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')
axs[2].grid(lw=0.2, alpha=0.5, ls='--')
axs[3].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[3].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

plt.suptitle(str(date)[:13] + ' UTC', fontsize=10)
plt.subplots_adjust(1.5/(icol*2+3), 0.2, 1.02-1.5/(icol*2+3), 0.88)
plt.savefig(output_png)


'''
# mpcalc.mixing_ratio_from_relative_humidity(1000 * units.hPa, 300 * units.K, 100 * units('%')).to('g/kg')
'''
# endregion


