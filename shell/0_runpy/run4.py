

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
    interp_to_pressure_levels)

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs, get_modis_latlonvar, get_modis_latlonvars, get_cross_section

# endregion


# region get monthly and hourly Himawari cmic

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year = 2020; month = 6

# option
products = ['cloud']
categories = ['cmic']
vars = ['cmic_lwp']
# ['cmic_cot', 'cmic_iwp', 'cmic_lwp', 'cmic_phase', 'cmic_reff']
# categories = ['ctth']
# vars = ['ctth_alti']
# ['ctth_alti', 'ctth_effectiv', 'ctth_pres', 'ctth_tempe']

# settings
himawari_bom = '/g/data/rv74/satellite-products/arc/der/himawari-ahi'
ancillary = xr.open_dataset('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')
himawari_rename = {
    'cmic_cot': 'COT',
    'cmic_iwp': 'clivi',
    'cmic_lwp': 'clwvi',
    'cmic_phase': 'clphase',
    'cmic_reff': 'Reff',
    'ctth_alti': 'CTH',
    'ctth_effectiv': 'clt',
    'ctth_pres': 'CTP',
    'ctth_tempe': 'CTT'}

def preprocess_himawari(ds_in, ivar):
    # ds_in = xr.open_dataset(fl[5])
    ds_out = ds_in[ivar].rename(himawari_rename[ivar])
    ds_out = ds_out.expand_dims(time=[np.datetime64(ds_in.attrs['nominal_product_time'])])
    return(ds_out)

for iproduct in products: #os.listdir(himawari_bom): #
    print(f'#-------------------------------- {iproduct}')
    for icategory in categories: #os.listdir(f'{himawari_bom}/{iproduct}'): #
        print(f'#---------------- {icategory}')
        
        folder = f'{himawari_bom}/{iproduct}/{icategory}'
        if os.path.isdir(folder):
            fl = sorted(glob.glob(f'{folder}/latest/{year}/{month:02d}/*/*.nc'))
            print(f'Number of files: {len(fl)}')
            # print(os.path.getsize(fl[-1])/2**20)
            
            for ivar in vars:
                print(f'#-------- {ivar}')
                odir = f'data/obs/jaxa/{himawari_rename[ivar]}'
                os.makedirs(odir, exist_ok=True)
                
                ds = xr.open_mfdataset(fl, parallel=True, preprocess=lambda ds_in: preprocess_himawari(ds_in, ivar))[himawari_rename[ivar]]
                ds = ds.chunk({'time': -1, 'nx': 50, 'ny': 50})
                
                if ivar in ['cmic_iwp', 'cmic_lwp']:
                    # print('get mm')
                    # ds_mm = ds.resample({'time': '1M'}).mean().compute()
                    # ds_mm.to_netcdf(f'{odir}/{himawari_rename[ivar]}_{year}{month:02d}.nc')
                    print('get mhm')
                    ds_mhm = ds.resample(time='1M').map(lambda x: x.groupby('time.hour').mean()).compute()
                    ds_mhm.to_netcdf(f'{odir}/{himawari_rename[ivar]}_hourly_{year}{month:02d}.nc')




'''
#-------------------------------- check




#-------------------------------- others

products = ['cloud']

categories = ['cmic']
vars = ['cmic_cot', 'cmic_iwp', 'cmic_lwp', 'cmic_phase', 'cmic_reff'] # ['cmic_conditions', 'cmic_cot', 'cmic_iwp', 'cmic_lwp', 'cmic_phase', 'cmic_quality', 'cmic_reff', 'cmic_status_flag']

categories = ['ctth']
vars = ['ctth_alti', 'ctth_effectiv', 'ctth_pres', 'ctth_tempe'] #['ctth_alti', 'ctth_conditions', 'ctth_effectiv', 'ctth_method', 'ctth_pres', 'ctth_quality', 'ctth_status_flag', 'ctth_tempe']

# categories = ['ct']
# vars = ['ct', 'ct_conditions', 'ct_cumuliform', 'ct_multilayer', 'ct_quality', 'ct_status_flag']
# categories = ['cma']
# vars = ['cma', 'cma_cloudsnow', 'cma_conditions', 'cma_dust', 'cma_quality', 'cma_smoke', 'cma_status_flag', 'cma_testlist1', 'cma_testlist2', 'cma_volcanic']


products = ['precip']

# categories = ['crrph']
# vars = ['crrph_accum', 'crrph_conditions', 'crrph_intensity', 'crrph_quality', 'crrph_status_flag']

# categories = ['crr']
# vars = ['crr', 'crr_accum', 'crr_conditions', 'crr_intensity', 'crr_quality', 'crr_status_flag']


products = ['solar']

# categories = ['p1s']
# vars = ['surface_global_irradiance', 'direct_normal_irradiance', 'surface_diffuse_irradiance', 'quality_mask', 'cloud_type', 'cloud_optical_depth', 'solar_elevation', 'solar_azimuth', 'julian_date']

# categories = ['p1d']
# vars = ['daily_integral_of_surface_global_irradiance', 'daily_integral_of_direct_normal_irradiance', 'daily_integral_of_surface_diffuse_irradiance', 'number_of_observations', 'number_of_cloud_observations', 'quality_mask']

# categories = ['p1h']
# vars = ['hourly_integral_of_surface_global_irradiance', 'hourly_integral_of_direct_normal_irradiance', 'hourly_integral_of_surface_diffuse_irradiance', 'number_of_observations', 'number_of_cloud_observations', 'quality_mask']


'''
# endregion

