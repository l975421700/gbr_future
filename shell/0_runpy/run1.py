

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=20GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import pandas as pd
import numpy.ma as ma
import glob
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
from pyhdf.error import HDF4Error
from satpy.scene import Scene
from skimage.measure import block_reduce
import xarray as xr

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)

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
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
    plot_loc,
    draw_polygon,
)

from calculations import (
    find_ilat_ilon,
    )

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs


# endregion


# region get 'MOD08_M3', 'MYD08_M3': total column q, qcl, qcf

products = ['MOD08_M3']
vars = {'Atmospheric_Water_Vapor_Mean_Mean': 'prw',
        'Cloud_Water_Path_Ice_Mean_Mean': 'clivi',
        'Cloud_Water_Path_Liquid_Mean_Mean': 'clwvi'}

for iproduct in products:
    # iproduct = 'MOD08_M3'
    print(f'#---------------- {iproduct}')
    fl = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/????/???/*.hdf'))
    das = []
    lon = SD(fl[0], SDC.READ).select('XDim')[:]
    lat = SD(fl[0], SDC.READ).select('YDim')[:]
    
    for ifile in fl:
        # ifile = fl[0]
        print(f'#-------- {ifile}')
        
        hdf_sd = SD(ifile, SDC.READ)
        year = ifile.split('/')[-3]
        doy  = ifile.split('/')[-2]
        date = datetime.strptime(f'{year}{doy}', '%Y%j')
        
        for ivar in vars.keys():
            # ivar = list(vars.keys())[0]
            # print(f'#-------- {ivar}')
            
            ds = hdf_sd.select(ivar)[:].astype(float)
            ds_attr = hdf_sd.select(ivar).attributes()
            
            ds[(ds < ds_attr['valid_range'][0]) | (ds > ds_attr['valid_range'][1]) | (ds == ds_attr['_FillValue'])] = np.nan
            ds = ds_attr['scale_factor'] * (ds - ds_attr['add_offset'])
            if ivar == 'Atmospheric_Water_Vapor_Mean_Mean': ds *= 10
            # print(np.nanmean(ds))
            
            da = xr.DataArray(
                ds[None, ], dims=('time', 'lat', 'lon'),
                coords={'time': [date], 'lat': lat, 'lon': lon},
                name=vars[ivar])
            das.append(da)
        hdf_sd.end()
    
    dss = xr.merge(das)
    ofile = f'scratch/data/obs/MODIS/{iproduct}/{'_'.join(vars.values())}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    dss.to_netcdf(ofile)



