

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=96GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60+gdata/py18+gdata/rv74+gdata/xp65


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import glob
import rioxarray as rxr
import calendar
import pickle

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.ticker as mticker
import cartopy.feature as cfeature

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import warnings
warnings.filterwarnings('ignore')

from namelist import (
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    )

from mapplot import (
    regional_plot,
    remove_trailing_zero_pos,
    )

from component_plot import (
    plt_mesh_pars,)

from calculations import (
    mon_sea_ann,
    )

# endregion


# region get monthly and hourly Himawari cmic
# Memory Used: 300GB; Walltime Used: 04:00

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year; month=args.month
# year = 2015; month = 7

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
                
                ds = xr.open_mfdataset(fl, combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override', preprocess=lambda ds_in: preprocess_himawari(ds_in, ivar))[himawari_rename[ivar]]
                ds = ds.chunk({'time': -1, 'nx': 50, 'ny': 50})
                
                if ivar in ['cmic_iwp', 'cmic_lwp']:
                    print('get mm')
                    ofile1 = f'{odir}/{himawari_rename[ivar]}_{year}{month:02d}.nc'
                    if not os.path.exists(ofile1):
                        ds_mm = ds.resample({'time': '1M'}).mean().compute()
                        ds_mm.to_netcdf(ofile1)
                    
                    print('get mhm')
                    ofile2 = f'{odir}/{himawari_rename[ivar]}_hourly_{year}{month:02d}.nc'
                    if not os.path.exists(ofile2):
                        ds_mhm = ds.resample(time='1M').map(lambda x: x.groupby('time.hour').mean()).compute()
                        ds_mhm.to_netcdf(ofile2)




'''
#-------------------------------- check
year = 2025; month = 1
iproduct = 'cloud'
icategory = 'cmic'
ivar = 'cmic_lwp'

himawari_bom = '/g/data/rv74/satellite-products/arc/der/himawari-ahi'
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

folder = f'{himawari_bom}/{iproduct}/{icategory}'
fl = sorted(glob.glob(f'{folder}/latest/{year}/{month:02d}/*/*.nc'))
print(f'Number of files: {len(fl)}')
odir = f'data/obs/jaxa/{himawari_rename[ivar]}'

ds = xr.open_mfdataset(fl, combine='by_coords', parallel=True, data_vars='minimal', coords='minimal',compat='override', preprocess=lambda ds_in: preprocess_himawari(ds_in, ivar))[himawari_rename[ivar]]
# ds = ds.chunk({'time': -1, 'nx': 50, 'ny': 50})

ds_mm = xr.open_dataset(f'{odir}/{himawari_rename[ivar]}_{year}{month:02d}.nc')[himawari_rename[ivar]]
ds_mhm = xr.open_dataset(f'{odir}/{himawari_rename[ivar]}_hourly_{year}{month:02d}.nc')[himawari_rename[ivar]]

inx = 2000
iny = 2000

print(ds_mm[0, iny, inx].values)
print(np.mean(ds[:, iny, inx]).values)

print(ds_mhm[0, iny, inx, :].values)
print(ds[:, iny, inx].groupby('time.hour').mean().values)
print(ds[:, iny, inx].groupby('time.hour').mean(skipna=False).values)




#-------------------------------- coordinates

ancillary = xr.open_dataset('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')
subset = np.isfinite(ancillary['lat'].values[0]) & np.isfinite(ds_mm['lat'].values)

print((ds_mm['lat'].values[subset] == ancillary['lat'].values[0][subset]).all())
print((ds_mm['lon'].values[subset] == ancillary['lon'].values[0][subset]).all())


print(np.max(np.abs(ds_mm['lat'].values[subset] - ancillary['lat'].values[0][subset])))
print(np.max(np.abs(ds_mm['lon'].values[subset] - ancillary['lon'].values[0][subset])))

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


# region get alltime monthly and hourly Himawari cmic
# mm: Memory Used: 37.72GB, Walltime Used: 00:06:09
# mhm: Memory Used: 500.0GB, Walltime Used: 04:23:35

years = '2016'
yeare = '2024'
for ivar in ['clwvi', 'clivi']:
    # ivar = 'clivi'
    print(f'#-------------------------------- {ivar}')
    
    for iperiod in ['mm', 'mhm']:
        # iperiod = 'mhm'
        print(f'#---------------- {iperiod}')
        
        if iperiod == 'mm':
            fl = sorted(glob.glob(f'data/obs/jaxa/{ivar}/{ivar}_??????.nc'))
            ofile = f'data/obs/jaxa/{ivar}/{ivar}_alltime.pkl'
        elif iperiod == 'mhm':
            fl = sorted(glob.glob(f'data/obs/jaxa/{ivar}/{ivar}_hourly_*.nc'))
            ofile = f'data/obs/jaxa/{ivar}/{ivar}_hourly_alltime.pkl'
        
        himawari_mon = xr.open_mfdataset(fl, combine='by_coords', parallel=True, data_vars='minimal',compat='override', coords='minimal')[ivar].sel(time=slice(years, yeare))
        
        if ivar in ['clwvi', 'clivi']:
            himawari_mon *= 1000
        
        himawari_mon_alltime = mon_sea_ann(
            var_monthly=himawari_mon, lcopy=False, mm=True, sm=True, am=True)
        
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile,'wb') as f:
            pickle.dump(himawari_mon_alltime, f)




'''
#-------------------------------- check
year = 2020; month = 6
for ivar in ['clwvi', 'clivi']:
    # ivar = 'clwvi'
    print(f'#-------------------------------- {ivar}')
    for iperiod in ['mm']:
        # iperiod = 'mhm'
        # ['mm', 'mhm']
        print(f'#---------------- {iperiod}')
        
        if iperiod == 'mm':
            ifile = f'data/obs/jaxa/{ivar}/{ivar}_{year}{month:02d}.nc'
            ofile = f'data/obs/jaxa/{ivar}/{ivar}_alltime.pkl'
        elif iperiod == 'mhm':
            ifile = f'data/obs/jaxa/{ivar}/{ivar}_hourly_{year}{month:02d}.nc'
            ofile = f'data/obs/jaxa/{ivar}/{ivar}_hourly_alltime.pkl'
        
        data1 = xr.open_dataset(ifile)[ivar]
        if ivar in ['clwvi', 'clivi']:
            data1 *= 1000
        
        with open(ofile,'rb') as f:
            data2 = pickle.load(f)
        
        print((data1.values[np.isfinite(data1.values)] == data2['mon'].sel(time=f'{year}-{month:02d}').values[np.isfinite(data2['mon'].sel(time=f'{year}-{month:02d}').values)]).all())


'''
# endregion




# region check monthly data

# option
ivar = 'clwvi'
year = 2020; month = 6
file = f'data/obs/jaxa/{ivar}/{ivar}_{year}{month:02d}.nc'
cbar_label = f'{calendar.month_name[month]} {year} Himawari {era5_varlabels[cmip6_era5_var['clwvi']]}'

# setting
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20,
    cmap='Purples_r',)
extend = 'max'

fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})
ax.imshow(np.ones((100,100,3)),extent=extent,transform=transform,zorder=0)
fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='k',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='k', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='k', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))

# opng = f'figures/0_gbr/test.png'
# ds = xr.open_dataset(file).squeeze()
# ds.clwvi[:] *= 1000
# plt_mesh = ax.pcolormesh(
#     ds.nx, ds.ny, ds.clwvi,
#     norm=pltnorm, cmap=pltcmp, transform=transform)

# opng = f'figures/0_gbr/test2.png'
# ds = xr.open_dataset(file).squeeze()
# ds = ds.rio.write_crs(ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0), inplace=False)
# ds = ds.rio.reproject('epsg:4326')
# ds.clwvi[:] *= 1000
# plt_mesh = ax.pcolormesh(
#     ds.x, ds.y, ds.clwvi,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())

# cbar_label = f'2016-2024 Himawari {era5_varlabels[cmip6_era5_var['clwvi']]}'
# opng = f'figures/0_gbr/test3.png'
# with open(f'data/obs/jaxa/{ivar}/{ivar}_alltime.pkl','rb') as f:
#     himawari_mon_alltime = pickle.load(f)
# plt_mesh = ax.pcolormesh(
#     himawari_mon_alltime['am'].nx,
#     himawari_mon_alltime['am'].ny,
#     himawari_mon_alltime['am'].squeeze(),
#     norm=pltnorm, cmap=pltcmp, transform=transform)

cbar_label = f'2016-2024 Himawari {era5_varlabels[cmip6_era5_var['clwvi']]}'
opng = f'figures/0_gbr/test4.png'
with open(f'data/obs/jaxa/{ivar}/{ivar}_alltime.pkl','rb') as f:
    himawari_mon_alltime = pickle.load(f)
himawari_mon_alltime['am'] = himawari_mon_alltime['am'].rio.write_crs(ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0), inplace=False)
himawari_mon_alltime['am'] = himawari_mon_alltime['am'].rename({'nx':'x', 'ny':'y'}).rio.reproject('epsg:4326')
plt_mesh = ax.pcolormesh(
    himawari_mon_alltime['am'].x,
    himawari_mon_alltime['am'].y,
    himawari_mon_alltime['am'].squeeze(),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())

cbar = fig.colorbar(
    plt_mesh,
    format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend=extend,
    cax=fig.add_axes([0.04, 0.8/(7+1), 0.92, 0.02]))
cbar.ax.tick_params(labelsize=8, length=2, width=0.5, pad=1)
cbar.ax.tick_params(which='minor', length=1, width=0.5)
cbar.ax.set_xlabel(cbar_label, fontsize=8, labelpad=2)

fig.savefig(opng)





'''
# ds = xr.open_dataset(file, engine='rasterio').squeeze()
# ds = rxr.open_rasterio(file, masked=True).squeeze()

# plot directly
ds.clwvi.plot(ax=ax, norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())

'''
# endregion



