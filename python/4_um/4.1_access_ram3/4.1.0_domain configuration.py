

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import xarray as xr
import pandas as pd
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity
from metpy.units import units
import calendar
import xesmf as xe
import pickle

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.use('Agg')
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string
import time
import glob

# self defined
from mapplot import (
    regional_plot,
    plot_maxmin_points,
    remove_trailing_zero_pos)

from namelist import (
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    month_jan)

from component_plot import (
    plt_mesh_pars)

# endregion


# region get domain


extent = {}
extent['era5']      = [88.49,   207.39,  13.01,  -57.99]
extent['d11km']     = [89.59,   206.29,  11.91,  -56.89]
extent['d4p4km']    = [108.02,  159.9,   -5.01,  -45.69]
extent['d4p4kms']   = [135.94,   159.9,   -5.01,  -30.01]
extent['d1p1km']    = [135.94,   159.9,   -5.01,  -30.01]

grid_spacings = {'era5': 0.1, 'd11km': 0.1, 'd4p4km': 0.04,
                 'd4p4kms': 0.04, 'd1p1km': 0.01}


for ires in ['era5', 'd11km', 'd4p4km', 'd4p4kms', 'd1p1km']:
    # ires = 'd11km'
    print(f'#-------------------------------- {ires}')
    
    print(f'#---------------- center longitude')
    print(np.round((extent[ires][0] + extent[ires][1])/2, 2))
    print(f'#---------------- center latitude')
    print(np.round((extent[ires][2] + extent[ires][3])/2, 2))
    
    print(f'#---------------- longitude ncells')
    print(np.round((extent[ires][1]-extent[ires][0])/grid_spacings[ires],1)+1)
    print(f'#---------------- latitude ncells')
    print(np.round((extent[ires][2]-extent[ires][3])/grid_spacings[ires],1)+1)



# Plotting land_binary_mask for d4p4km with shape (1018, 1298)
# Plotting land_binary_mask for d11km with shape (689, 1168)
# Plotting land_binary_mask for era5 with shape (711, 1190)
# d4p4kms has bounds -30.03 -5.03 135.92 159.88
# d4p4km has bounds -45.69 -5.01 108.02 159.90
# d11km has bounds -56.89 11.91 89.59 206.29
# era5 has bounds -57.99 13.01 88.49 207.39

'''

extent['d11km']     = [87.27,   208.6,  14.19,  -59.18]
extent['d4.4km']    = [107.02,  160.9,  -4.01,  -46.71]

'''

# endregion

