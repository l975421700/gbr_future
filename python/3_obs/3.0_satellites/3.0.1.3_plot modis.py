

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import pickle
import glob
from datetime import datetime
from pyhdf.SD import SD, SDC

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
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
import matplotlib.ticker as mticker
import matplotlib.animation as animation

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
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    era5_varlabels,
    ds_color,
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
    mon_sea_ann,
    cdo_regrid,)

# endregion


# region plot modis terra and aqua

year, month, day, hour = 2020, 6, 2, 0
doy = datetime(year, month, day).timetuple().tm_yday

fl = {}
for iproduct in ['MOD02QKM', 'MYD02QKM', 'MOD02HKM', 'MYD02HKM', 'MOD021KM', 'MYD021KM']:
    print(f'#-------------------------------- {iproduct}')
    fl[iproduct] = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/{iproduct}.A{year}{doy:03d}.{hour:02d}??.061.{year}*.hdf'))









hdf = SD(fl['Terra'][0], SDC.READ)
# hdf.datasets()
# scn = Scene(filenames={'modis_l1b': [fl['Terra'][0]]})

hdf.select('Latitude')[:]
hdf.select('Longitude')[:]



# EV_250_Aggr1km_RefSB

'''
from satpy.scene import Scene
fl['Terra'] = sorted(glob.glob(f'scratch/data/obs/MODIS/MOD02QKM/{year}/{doy:03d}/MOD02QKM.A{year}{doy:03d}.{hour:02d}??.061.{year}*.hdf'))
fl['Aqua'] = sorted(glob.glob(f'scratch/data/obs/MODIS/MYD02QKM/{year}/{doy:03d}/MYD02QKM.A{year}{doy:03d}.{hour:02d}??.061.{year}*.hdf'))

datetime(2020, 6, 1).timetuple().tm_yday
datetime(2020, 6, 30).timetuple().tm_yday


https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/modis-L0L1/
'''
# endregion

