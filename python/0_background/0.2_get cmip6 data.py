

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
import xesmf as xe
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import cmip6_preprocessing.preprocessing as cpp
import fsspec
import intake

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
sys.path.append(os.getcwd() + '/code/RCM_GBR/module')

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

from calculations import (
    time_weighted_mean,
    mon_sea_ann,
    mon_sea_ann_average,
    regrid,
    inversion_top,
    find_ilat_ilon,
    find_gridvalue_at_site,
    find_gridvalue_at_site_time,
    )

from namelist import (
    month,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    )

from statistics0 import (
    fdr_control_bh,
    check_normality_3d,
    check_equal_variance_3d,
    ttest_fdr_control,
    cplot_ttest,
    xr_par_cor,
    xr_regression_y_x1,
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


# region define functions

def combined_preprocessing(ds_in):
    ds=ds_in.copy()
    ds=cpp.rename_cmip6(ds)
    ds=cpp.broadcast_lonlat(ds)
    return ds

def drop_all_bounds(ds):
    drop_vars = [vname for vname in ds.coords
                 if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop(drop_vars)

def open_dsets(df):
    '''Open datasets from cloud storage and return xarray dataset.'''
    dsets = [xr.open_zarr(fsspec.get_mapper(ds_url), consolidated=True, use_cftime=True) for ds_url in df.zstore]
    try:
        ds = xr.merge(dsets, join='exact')
        return ds
    except ValueError:
        return None

def open_delayed(df):
    '''A dask.delayed wrapper around `open_dsets`.
    Allows us to open many datasets in parallel.'''
    return dask.delayed(open_dsets)(df)

cmip_info = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
esm_datastore = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

# endregion


# region get data



query = dict(
    experiment_id=['historical'],
    table_id='Omon',
    variable_id=['tos'],
)
esm_data_subset = esm_datastore.search(**query)
print(esm_data_subset)
esm_data_subset.df.groupby("source_id").first()


query = dict(
    experiment_id=['ssp585'],
    table_id='Omon',
    variable_id=['tos'],
)
esm_data_subset = esm_datastore.search(**query)
print(esm_data_subset)
esm_data_subset.df.groupby("source_id").first()


query = dict(
    experiment_id=['ssp126'],
    table_id='Omon',
    variable_id=['tos'],
)
esm_data_subset = esm_datastore.search(**query)
print(esm_data_subset)
esm_data_subset.df.groupby("source_id").first()


query = dict(
    experiment_id=['amip'],
    table_id='Amon',
    variable_id=['tas'],
)
esm_data_subset = esm_datastore.search(**query)
print(esm_data_subset)
esm_data_subset.df.groupby("source_id").first()


'''
esm_data_subset = esm_datastore.search(**{
    'experiment_id': ['historical'], 'table_id': 'Omon', 'variable_id': ['tos']})

'''
# endregion