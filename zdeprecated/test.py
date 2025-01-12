
# region download data

year_str        = '2022'
month_str_list  = np.arange(2, 3, 1).astype(str).tolist()
day_str_list    = np.arange(6, 10, 1).astype(str).tolist()
area            = '10/122/-45/175'

output_filename = 'data/sim/wrf/input/era5/era5_sl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area=area,
    surface_only=True,
    )

output_filename = 'data/sim/wrf/input/era5/era5_pl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area=area,
    surface_only=False,
    )


'''
area            = '10/122/-45/175'
'''
# endregion



# region get CL_Frequency

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

# loop through each day
daterange = pd.date_range(start='1/1/2016', end='12/31/2016')

for idate in daterange:
    # idate = daterange[0]
    print(idate)
    
    year=str(idate)[:4]
    month=str(idate)[5:7]
    day=str(idate)[8:10]
    
    clp_fl = sorted(glob.glob(f'scratch/data/obs/jaxa/clp/{year}{month}/{day}/*/CLP_{year}{month}{day}????.nc'))
    
    clp_ds = xr.open_mfdataset(clp_fl)
    
    CLTYPE_values = clp_ds.CLTYPE.values
    
    CL_Frequency = xr.DataArray(
        name='CL_Frequency',
        data=np.zeros((1, 12, clp_ds.CLTYPE.shape[1], clp_ds.CLTYPE.shape[2])),
        dims=['time', 'types', 'latitude', 'longitude',],
        coords={
            'time': [datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d')],
            'types': ['finite'] + list(ISCCP_types.keys()),
            'latitude': clp_ds.CLTYPE.latitude.values,
            'longitude': clp_ds.CLTYPE.longitude.values,
        }
    )
    
    CL_Frequency.loc[{'types': 'finite'}][0] = np.isfinite(CLTYPE_values).sum(axis=0)
    
    for itype in list(ISCCP_types.keys()):
        # print(itype)
        CL_Frequency.loc[{'types': itype}][0] = (CLTYPE_values == ISCCP_types[itype]).sum(axis=0)
    
    print((CL_Frequency[0, 0] == CL_Frequency[0, 1:].sum(axis=0)).all().values)
    print(CL_Frequency[0, 0].sum().values)
    
    ofile = f'scratch/data/obs/jaxa/clp/{year}{month}/{day}/CL_Frequency_{year}{month}{day}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    CL_Frequency.to_netcdf(ofile)
    
    print("Current time:", datetime.now())


aaa = xr.open_dataset('scratch/data/obs/jaxa/clp/201601/01/CL_Frequency_20160101.nc')


'''
himawari_fl = sorted(glob.glob('data/obs/jaxa/clp/*/*/*/NC_*'))
clp_fl = sorted(glob.glob('scratch/data/obs/jaxa/clp/*/*/*/CLP_*'))
print(len(himawari_fl))
print(len(clp_fl))
'''
# endregion


# region animate sounding profiles (theta, RH)

start_date = datetime(2021, 1, 1, 0)
end_date = datetime(2021, 12, 31, 23)
station = '94299'
output_mp4 = 'figures/test1.mp4'

fig, axs = plt.subplots(1, 2, sharey=True, figsize=np.array([8.8, 6.4]) / 2.54)

ims = []
for date in pd.date_range(start_date, end_date, freq='12h'):
    try:
        df = WyomingUpperAir.request_data(date, station)
        df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='any').reset_index(drop=True)
        
        p = df['pressure'].values * units.hPa
        T = (df['temperature'].values * units.degC).to(units('K'))
        Td = (df['dewpoint'].values * units.degC).to(units('K'))
        height = df['height'].values * units.m
        
        thta = mpcalc.potential_temperature(p, T)
        RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
        
        plt1 = axs[0].plot(thta, p, c='tab:blue')
        plt2 = axs[1].plot(RH, p, c='tab:blue')
        plt3 = plt.text(0.5, 0.95, str(date)[:13] + ' UTC', ha='center', fontsize=10, transform=fig.transFigure)
        
        ims.append(plt1+plt2 + [plt3])
        print(str(date)[:13])
    except:
        print('No data for ' + str(date)[:13])

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(290, 330)
axs[1].set_xlim(0, 100)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'$\theta$ [$K$]')
axs[1].set_xlabel(r'RH [$\%$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[1].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

fig.subplots_adjust(0.18, 0.18, 0.85, 0.88)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)

# endregion


# region get and plot Wyoming sounding (theta and RH)

date = datetime(2021, 12, 8, 0)
station = '94299'
output_png='figures/test.png'

df = WyomingUpperAir.request_data(date, station)
plot_wyoming_sounding_vertical(df, date, output_png=output_png)



'''
print(df.columns)
'''
# endregion


# region get and plot Wyoming sounding (T, theta, RH)

date = datetime(2021, 9, 16, 0)
station = '94299'
output_png='figures/test.png'

df = WyomingUpperAir.request_data(date, station)
df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all').reset_index(drop=True)

p = df['pressure'].values * units.hPa
T = (df['temperature'].values * units.degC).to(units('K'))
Td = (df['dewpoint'].values * units.degC).to(units('K'))
height = df['height'].values * units.m

thta = mpcalc.potential_temperature(p, T)
RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')

fig, axs = plt.subplots(1, 3, sharey=True, figsize=np.array([8.8, 6.4]) / 2.54)

axs[0].plot(T, p)
axs[1].plot(thta, p)
axs[2].plot(RH, p)

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(270, 310)
axs[1].set_xlim(290, 330)
axs[2].set_xlim(0, 100)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'T [$K$]')
axs[1].set_xlabel(r'$\theta$ [$K$]')
axs[2].set_xlabel(r'RH [$\%$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')
axs[2].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[2].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

plt.suptitle(str(date)[:13] + ' UTC', fontsize=10)
plt.subplots_adjust(0.18, 0.18, 0.85, 0.88)
plt.savefig(output_png)


# endregion


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

