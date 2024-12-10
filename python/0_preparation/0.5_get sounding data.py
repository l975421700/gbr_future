

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


# region get and plot Wyoming sounding

date = datetime(2016, 1, 1, 0)
station = '94299'
df = WyomingUpperAir.request_data(date, station)

df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all').reset_index(drop=True).loc[df['pressure']>=200]

p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed.to(units('m/s')), wind_dir)

lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')


fig = plt.figure(figsize=np.array([8.8, 6.4]) / 2.54)
skew = SkewT(fig, rotation=30)

skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot(p, parcel_prof, 'k')
# skew.plot_barbs(p, u, v)
skew.plot(lcl_pressure, lcl_temperature, 'ko', markersize=2)
skew.shade_cin(p, T, parcel_prof, Td)
skew.shade_cape(p, T, parcel_prof)

ax_hod = inset_axes(skew.ax, '35%', '35%', bbox_transform=skew.ax.transAxes,
                    bbox_to_anchor=(0.05, 0, 1, 1))
h = Hodograph(ax_hod, component_range=14)
h.add_grid(increment=2, lw=0.2, alpha=0.5)
h.plot_colormapped(u, v, p, cmap='viridis_r', lw=0.5)
h.ax.get_yaxis().set_visible(False)
h.ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
h.ax.set_xlabel('Wind [$m \; s^{-1}$]')

skew.plot_dry_adiabats(lw=0.4, alpha=0.5)
skew.plot_moist_adiabats(lw=0.4, alpha=0.5)
# skew.plot_mixing_lines()
# skew.ax.axvline(0, color='c', ls='--')
skew.ax.grid(lw=0.2, alpha=0.5, ls='--')
skew.ax.set_xlabel('Temperature [$°C$]')
skew.ax.set_ylabel('Pressure [$hPa$]')
skew.ax.set_ylim(1030, 200)
skew.ax.set_xlim(-40, 40)
skew.ax.text(0.05, 0.05, str(date)[:13] + ' UTC', transform=skew.ax.transAxes,
             ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='white'))

fig.subplots_adjust(left=0.18, right = 0.96, bottom = 0.14, top = 0.98)
fig.savefig('figures/test.png')



'''
print(df.columns)
print(lcl_pressure, lcl_temperature)
'''
# endregion


# region get and output Wyoming sounding

date = datetime(2016, 1, 1, 0)
station = '94299'
df = WyomingUpperAir.request_data(date, station)

df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind']].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all').reset_index(drop=True).loc[df['pressure']>=700]


p = df['pressure'].values * units.hPa
T = (df['temperature'].values * units.degC).to(units('K'))
Td = (df['dewpoint'].values * units.degC).to(units('K'))
height = df['height'].values * units.m
wind_speed = (df['speed'].values * units.knots).to(units('m/s'))
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)


thta = mpcalc.potential_temperature(p, T)
RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
mixr = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH).to('g/kg')

outf = 'data/sim/wrf/input/sounding/willis_island_2016010100'

sp = p[0].magnitude
sthta = thta[0].magnitude
smixr = mixr[0].magnitude
nlevels = height.shape[0]

line1 = '{:9.2f} {:9.2f} {:10.2f}'.format(sp, sthta, smixr)
wrfformat = '{:9.2f} {:9.2f} {:10.2f} {:10.2f} {:10.2f}'

with open(outf, 'w') as f:
    f.write(line1+'\n')
    
    for ilev in range(nlevels):
        d = wrfformat.format(height[ilev].magnitude, thta[ilev].magnitude, mixr[ilev].magnitude, u[ilev].magnitude, v[ilev].magnitude)
        f.write(d+ '\n')







'''
print(df.columns)

'''
# endregion