

# qsub -I -q normal -P v46 -l walltime=2:00:00,ncpus=1,mem=48GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3+gdata/hr22+scratch/gx60+scratch/gb02+gdata/gb02


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
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity, relative_humidity_from_specific_humidity, dewpoint_from_specific_humidity, equivalent_potential_temperature, potential_temperature
from datetime import datetime, timedelta
from haversine import haversine

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
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
    suite_res, suite_label,
    interp_to_pressure_levels,
    amstash2var, amvar2stash, preprocess_amoutput)

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs, get_modis_latlonvar, get_modis_latlonvars, get_cross_section

# endregion


# region check ACCESS-AM3 output

iexp = 'am3-climaerosol'
streams = ['a', 'b', 'c']
stream_time = {'a': 'monthly', 'b': 'hourly', 'c': 'daily'}

for istream in ['a']: # streams
    # istream = 'a'
    print(f'#-------------------------------- {istream} {stream_time[istream]}')
    
    fl = sorted(glob.glob(f'scratch/cylc-run/{iexp}/share/data/History_Data/netCDF/*a.p{istream}*.nc'))
    
    # ds = xr.open_dataset(fl[0]).pipe(preprocess_amoutput)
    ds = xr.open_mfdataset(fl[12:24], preprocess=preprocess_amoutput, parallel=True)
    # print(ds)
    # print(ds.data_vars)
    
    for ivar in ds.data_vars:
        # ivar = list(ds.data_vars)[0]
        print(f'#---------------- {ivar}')
        
        dsm = ds[ivar].mean(dim='time').compute()
        print(f'Min: {np.nanmin(dsm)}')
        print(f'Max: {np.nanmax(dsm)}')
        
        # if ivar.startswith('fld_'):
        #     print(f"    '{ivar}': '', #'{ds[ivar].attrs['long_name']}',")
        if ivar in amstash2var.keys():
            print(ds[ivar].attrs)
            # print(f'#---------------- {ivar}:    {amstash2var[ivar]:<18}  {ds[ivar].attrs['long_name']}')
            # if f'STASH_m01{ivar[4:]}' in stash2var_gal:
            #     print(stash2var_gal[f'STASH_m01{ivar[4:]}'])
        elif ivar.startswith('fld_'):
            print(f'#---------------- {ivar}: {ds[ivar].attrs['long_name']}')
            print(ds[ivar])
        # else:
        #     print(f'#---------------- {ivar}')











# endregion


# region check ACCESS-AM3 CDNC

ofile = 'scratch/cylc-run/access3-configs/share/data/History_Data/netCDF/confia.pc19820101.nc'

ds = xr.open_dataset(ofile)['fld_s04i210']
ds = ds.where(ds != 0, np.nan)

ds = ds.mean(dim=['time_0', 'model_theta_level_number'], skipna=True) / 1e+6
# ds.to_netcdf('scratch/tmp/test.nc')

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=400, cm_interval1=20, cm_interval2=40, cmap='pink',)
extend='max'

fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
plt_mean = ds.weighted(np.cos(np.deg2rad(ds.lat))).mean().values
cbar_label = f'ACCESS-AM3 Jan 1982 {era5_varlabels[cmip6_era5_var['CDNC']]}'

ax.text(0.01, 0.01,
        f'Mean: {str(np.round(plt_mean, 1))}', ha='left', va='bottom',
        transform=ax.transAxes)

plt_mesh1 = ax.pcolormesh(
    ds.lon, ds.lat, ds,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend=extend,
    pad=0.02, fraction=0.13,)
cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
fig.savefig(f'figures/0_gbr/test.png')

# endregion

