

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=60GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55


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

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
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

# endregion


# region check variables


year, month, day, hour = 2020, 6, 1, 0
isuite = 'u-ds714'
ds = {}
ds['d11km'] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/d11km/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).rename(stash2var_gal)
ds['d4p4km'] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/d4p4km/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput).rename(stash2var_ral)


for ivar in var2stash:
    # ivar = 'hus'
    print(f'#-------------------------------- {ivar}')
    print(ds['d11km'][ivar])
    print(ds['d4p4km'][ivar])

for ivar in ['ncloud', 'pr', 'snow', 'rain']:
    # ivar = 'mcl'
    print(f'#-------------------------------- {ivar}')
    print(ds['d11km'][ivar])

for ivar in ['qs', 'qg', 'ncloud', 'nrain', 'nice', 'nsnow', 'ngraupel', 'rain', 'snow', 'graupel', 'snow_graupel']:
    # ivar = 'mcl'
    print(f'#-------------------------------- {ivar}')
    print(ds['d4p4km'][ivar])




np.unique(ds['d11km']['ncloud']) # CLOUD DROP NUMBER CONC. /m3 [      0., 5242880.]
np.unique(ds['d4p4km']['ncloud']) # CLOUD NUMBER AFTER TIMESTEP [0.0000000e+00, 1.0485760e+06, ..., 1.4889779e+08, 1.4994637e+08]


# 10 METRE WIND U-COMP, STASH_m01s03i209, uas
# 10 METRE WIND V-COMP, STASH_m01s03i210, vas
# 10 METRE WIND SPEED ON C GRID, STASH_m01s03i230, sfcWind
# GEOPOTENTIAL HEIGHT ON THETA LEVELS, STASH_m01s16i201, zg
# SBCIN surface based CIN         J/kg, STASH_m01s20i115, CIN



'''
# unrecognized variables
for istash in [istash for istash in ds['d11km'].data_vars if istash.startswith('STASH')]:
    print(f'#-------------------------------- {istash}')
    print(ds['d11km'][istash])
for istash in [istash for istash in ds['d4p4km'].data_vars if istash.startswith('STASH')]:
    print(f'#-------------------------------- {istash}')
    print(ds['d4p4km'][istash])
'''
# endregion
