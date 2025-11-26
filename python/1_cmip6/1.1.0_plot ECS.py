

# qsub -I -q normal -P v46 -l walltime=4:00:00,ncpus=1,mem=20GB,jobfs=10GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3


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
    monthini,
    seasons,
    seconds_per_d,
    zerok,
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
    mon_sea_ann,
    cdo_regrid,)

# endregion


# region plot ECS

cmips = ['cmip6']

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    
    with open(f'data/sim/cmip/{icmip}/ecs.pkl', 'rb') as f: ecs = pickle.load(f)
    
    opng = f'figures/1_cmip/1.0_ecs/1.0.0 {icmip} ecs.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([15, 20]) / 2.54)
    
    sns.barplot(data=ecs, y='source_id', x='ecs', orient='h')
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(r'ECS [$K$]')
    ax.set_ylabel(f'Models')
    ax.grid(True, which='both', linewidth=0.6, color='gray',
            linestyle='--')
    fig.subplots_adjust(left=0.3, right=0.99, bottom=0.06, top=0.99)
    fig.savefig(opng)




# endregion

