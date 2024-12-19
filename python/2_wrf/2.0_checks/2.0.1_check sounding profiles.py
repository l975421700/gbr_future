

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
import time

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


# region animate sounding profiles skew-T

start_date = datetime(2021, 1, 1, 0)
end_date = datetime(2021, 12, 31, 23)
station = '94299'
output_mp4 = 'figures/test.mp4'

ilev = 1

fig = plt.figure(figsize=np.array([8.8, 6.4]) / 2.54)
skew = SkewT(fig, rotation=30)
ax_hod = inset_axes(
    skew.ax, '35%', '35%',
    bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=skew.ax.transAxes)
h = Hodograph(ax_hod, component_range=14)

ims = []
for date in pd.date_range(start_date, end_date, freq='12h'):
    try:
        df = WyomingUpperAir.request_data(date, station)
        df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all').reset_index(drop=True).loc[df['pressure']>=200]
        
        p = df['pressure'].values * units.hPa
        T = df['temperature'].values * units.degC
        Td = df['dewpoint'].values * units.degC
        wind_speed = (df['speed'].values * units.knots).to(units('m/s'))
        wind_dir = df['direction'].values * units.degrees
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
        
        lcl_p, lcl_t = mpcalc.lcl(p[ilev], T[ilev], Td[ilev])
        parcel_prof = mpcalc.parcel_profile(p[ilev:], T[ilev], Td[ilev]).to('degC')
        
        plt1 = skew.plot(p, T, 'r')
        plt2 = skew.plot(p, Td, 'g')
        plt3 = skew.plot(p[ilev:], parcel_prof, 'k')
        plt4 = skew.plot(lcl_p, lcl_t, 'ko', markersize=1.5)
        plt5 = skew.shade_cin(p[ilev:], T[ilev:], parcel_prof, Td[ilev:])
        plt6 = skew.shade_cape(p[ilev:], T[ilev:], parcel_prof)
        plt7 = skew.ax.text(
            0.05, 0.05, str(date)[:13] + ' UTC',
            transform=skew.ax.transAxes, ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='white'), zorder=2)
        
        plt8 = h.plot_colormapped(u, v, p, cmap='viridis_r', lw=0.5)
        
        ims.append(plt1+plt2+plt3+plt4 + [plt5,plt6,plt7,plt8])
        print(str(date)[:13])
    except:
        print('No data for ' + str(date)[:13])

h.add_grid(increment=2, lw=0.2, alpha=0.5)
h.ax.get_yaxis().set_visible(False)
h.ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
h.ax.set_xlabel(r'Wind [$m \; s^{-1}$]')

skew.ax.set_ylim(1030, 200)
skew.ax.set_xlim(-40, 40)
skew.plot_dry_adiabats(lw=0.4, alpha=0.5)
skew.plot_moist_adiabats(lw=0.4, alpha=0.5)
skew.ax.grid(lw=0.2, alpha=0.5, ls='--')
skew.ax.set_xlabel('Temperature [$°C$]')
skew.ax.set_ylabel('Pressure [$hPa$]')

fig.subplots_adjust(0.18, 0.14, 0.96, 0.98)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)


'''
date = datetime(2021, 12, 8, 0)
station = '94299'
df = WyomingUpperAir.request_data(date, station)
'''
# endregion


# region animate sounding profiles (T, theta, RH, qv)

start_date = datetime(2021, 1, 1, 0)
end_date = datetime(2021, 12, 31, 23)

# station = '94299'
# output_mp4 = 'figures/test2.mp4'
station = 'YBBN'
output_mp4 = f'figures/0_gbr/0.0_sounding/0.0.0 {station} {str(start_date)[:4]} sounding vertical profiles.mp4'


icol=4
fig, axs = plt.subplots(1,icol,sharey=True, figsize=np.array([icol*2+3, 6.4]) / 2.54)

retry_delay = 1
max_retries = 10

ims = []
for date in pd.date_range(start_date, end_date, freq='24h'):
    
    for attempt in range(max_retries):
        try:
            df = WyomingUpperAir.request_data(date, station)
            print(str(date)[:13])
            
            df = df[['pressure', 'temperature', 'dewpoint']].dropna(subset=('temperature', 'dewpoint'), how='all').reset_index(drop=True)
            
            p = df['pressure'].values * units.hPa
            T = (df['temperature'].values * units.degC).to(units('K'))
            Td = (df['dewpoint'].values * units.degC).to(units('K'))
            
            thta = mpcalc.potential_temperature(p, T)
            RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
            mixr = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH).to('g/kg')
            
            plt1 = axs[0].plot(T, p, c='tab:blue', marker='o', markersize=1)
            plt2 = axs[1].plot(thta, p, c='tab:blue', marker='o', markersize=1)
            plt3 = axs[2].plot(RH/100, p, c='tab:blue', marker='o', markersize=1)
            plt4 = axs[3].plot(mixr, p, c='tab:blue', marker='o', markersize=1)
            plt5 = plt.text(0.5, 0.95, str(date)[:13] + ' UTC', ha='center', fontsize=10, transform=fig.transFigure)
            
            ims.append(plt1+plt2+plt3+plt4 + [plt5])
            
            break
        except ValueError as e:
            print(f"No data for {date}: {e}")
            break
        except Exception as e:
            print(f"{e}")
            if attempt < max_retries-1:
                time.sleep(retry_delay)
                # retry_delay *= 2
            else:
                print("Max retries reached. Exiting.")

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

fig.subplots_adjust(1.5/(icol*2+3), 0.2, 1.02-1.5/(icol*2+3), 0.88)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)

# endregion


