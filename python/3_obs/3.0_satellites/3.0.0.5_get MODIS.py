

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
# Memory Used: 128.2GB, Walltime Used: 00:32:29

products = ['MOD08_M3', 'MYD08_M3']
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




'''
#-------------------------------- check

iproduct = 'MOD08_M3'
vars = {'Atmospheric_Water_Vapor_Mean_Mean': 'prw',
        'Cloud_Water_Path_Ice_Mean_Mean': 'clivi',
        'Cloud_Water_Path_Liquid_Mean_Mean': 'clwvi'}

fl = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/????/???/*.hdf'))
itime = -10
ifile = fl[itime]
hdf_sd = SD(ifile, SDC.READ)
year = ifile.split('/')[-3]
doy  = ifile.split('/')[-2]
date = datetime.strptime(f'{year}{doy}', '%Y%j')
lon = hdf_sd.select('XDim')[:]
lat = hdf_sd.select('YDim')[:]

dss = xr.open_dataset(f'scratch/data/obs/MODIS/{iproduct}/{'_'.join(vars.values())}.nc')

for ivar in vars.keys():
    print(f'#-------- {ivar}')
    
    ds = hdf_sd.select(ivar)[:].astype(float)
    ds_attr = hdf_sd.select(ivar).attributes()
    
    ds[(ds < ds_attr['valid_range'][0]) | (ds > ds_attr['valid_range'][1]) | (ds == ds_attr['_FillValue'])] = np.nan
    ds = ds_attr['scale_factor'] * (ds - ds_attr['add_offset'])
    if ivar == 'Atmospheric_Water_Vapor_Mean_Mean': ds *= 10
    
    print((ds[np.isfinite(ds)] == dss[vars[ivar]][itime].values[np.isfinite(dss[vars[ivar]][itime].values)]).all())




hdf_sd.datasets().keys()



hdf_sd.select('Atmospheric_Water_Vapor_Mean_Mean')[:]
hdf_sd.select('Cloud_Water_Path_Ice_Mean_Mean')[:]
hdf_sd.select('Cloud_Water_Path_Liquid_Mean_Mean')[:]

hdf_sd.select('Atmospheric_Water_Vapor_Mean_Mean').attributes()
{'valid_range': [0, 20000],
 '_FillValue': -9999,
 'long_name': 'Precipitable Water Vapor (IR Retrieval) Total Column: Mean of Daily Mean',
 'units': 'cm',
 'scale_factor': 0.0010000000474974513,
 'add_offset': 0.0,
 'Level_2_Pixel_Values_Read_As': 'Real',
 'Included_Level_2_Nighttime_Data': 'True',
 'Quality_Assurance_Data_Set': 'Quality_Assurance',
 'Statistic_Type': 'Simple',
 'QA_Byte': 0,
 'QA_Useful_Flag_Bit': 4,
 'QA_Value_Start_Bit': 5,
 'QA_Value_Num_Bits': 3,
 'Aggregation_Data_Set': 'None',
 'Derived_From_Level_3_Daily_Data_Set': 'Atmospheric_Water_Vapor_Mean',
 'Weighting': 'Pixel_Weighted',
 'Weighted_Parameter_Data_Set': 'Atmospheric_Water_Vapor_Confidence_Histogram'}
hdf_sd.select('Cloud_Water_Path_Ice_Mean_Mean').attributes()
{'valid_range': [0, 6000],
 '_FillValue': -9999,
 'long_name': 'Ice Cloud Water Path: Mean of Daily Mean',
 'units': 'g/m^2',
 'scale_factor': 1.0,
 'add_offset': 0.0,
 'Level_2_Pixel_Values_Read_As': 'Real',
 'Included_Level_2_Nighttime_Data': 'False',
 'Quality_Assurance_Data_Set': 'Quality_Assurance_1km',
 'Statistic_Type': 'Simple',
 'QA_Byte': 2,
 'QA_Useful_Flag_Bit': 3,
 'QA_Value_Start_Bit': 0,
 'QA_Value_Num_Bits': 2,
 'Aggregation_Data_Set': 'Quality_Assurance_1km',
 'Aggregation_Byte': 2,
 'Aggregation_Value_Start_Bit': 0,
 'Aggregation_Value_Num_Bits': 3,
 'Aggregation_Category_Values': 3,
 'Aggregation_Valid_Category_Values': [1, 2, 3, 4],
 'Derived_From_Level_3_Daily_Data_Set': 'Cloud_Water_Path_Ice_Mean',
 'Weighting': 'Pixel_Weighted',
 'Weighted_Parameter_Data_Set': 'Cloud_Retrieval_Fraction_Ice_Pixel_Counts'}
hdf_sd.select('Cloud_Water_Path_Liquid_Mean_Mean').attributes()
{'valid_range': [0, 3000],
 '_FillValue': -9999,
 'long_name': 'Liquid Water Cloud Water Path: Mean of Daily Mean',
 'units': 'g/m^2',
 'scale_factor': 1.0,
 'add_offset': 0.0,
 'Level_2_Pixel_Values_Read_As': 'Real',
 'Included_Level_2_Nighttime_Data': 'False',
 'Quality_Assurance_Data_Set': 'Quality_Assurance_1km',
 'Statistic_Type': 'Simple',
 'QA_Byte': 2,
 'QA_Useful_Flag_Bit': 3,
 'QA_Value_Start_Bit': 0,
 'QA_Value_Num_Bits': 2,
 'Aggregation_Data_Set': 'Quality_Assurance_1km',
 'Aggregation_Byte': 2,
 'Aggregation_Value_Start_Bit': 0,
 'Aggregation_Value_Num_Bits': 3,
 'Aggregation_Category_Values': 2,
 'Aggregation_Valid_Category_Values': [1, 2, 3, 4],
 'Derived_From_Level_3_Daily_Data_Set': 'Cloud_Water_Path_Liquid_Mean',
 'Weighting': 'Pixel_Weighted',
 'Weighted_Parameter_Data_Set': 'Cloud_Retrieval_Fraction_Liquid_Pixel_Counts'}
'''
# endregion
