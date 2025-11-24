

# region import packages

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pandas as pd
import intake
from cdo import Cdo
cdo=Cdo()
import xarray as xr
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
import numpy as np
import os

# management
import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
import pickle
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import gc
import warnings
warnings.filterwarnings('ignore')
import json
import time

# self defined function
from calculations import (
    mon_sea_ann,
    cdo_regrid,
    )
from xmip.preprocessing import rename_cmip6, broadcast_lonlat, correct_lon, promote_empty_dims, replace_x_y_nominal_lat_lon, correct_units, correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds, combined_preprocessing

from namelist import cmip6_units, zerok, seconds_per_d


'''
cmip_info['experiment_id'].unique()
cmip_info['institution_id'].unique()
'''
# endregion


# region get global and zonal mean

# option
cmips = ['cmip6']
experiment_ids  = ['piControl']
table_ids       = ['Amon']
variable_ids    = ['rsut']

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile3 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens.pkl'
                with open(ofile3, 'rb') as f:
                    ds_regridded_alltime_ens = pickle.load(f)
                
                ds_regridded_alltime_ens_gzm = {}
                
                for ialltime in ds_regridded_alltime_ens.keys():
                    # ialltime = 'mon'
                    print(f'#-- {ialltime}')
                    ds_regridded_alltime_ens_gzm[ialltime] = {}
                    
                    ds_regridded_alltime_ens_gzm[ialltime]['zm'] = ds_regridded_alltime_ens[ialltime].mean(dim='x', skipna=True).compute()
                    ds_regridded_alltime_ens_gzm[ialltime]['gm'] = ds_regridded_alltime_ens[ialltime].weighted(np.cos(np.deg2rad(ds_regridded_alltime_ens[ialltime].lat))).mean(dim=['x', 'y'], skipna=True).compute()
                
                ofile4 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
                if os.path.exists(ofile4): os.remove(ofile4)
                with open(ofile4, 'wb') as f:
                    pickle.dump(ds_regridded_alltime_ens_gzm, f)
        
        del ds_regridded_alltime_ens, ds_regridded_alltime_ens_gzm




'''

#-------------------------------- check
cmip6_data_regridded_alltime_ens = {}
cmip6_data_regridded_alltime_ens_gzm = {}

ialltime = 'mon'
ith_source_id = -1
itime = -1

for experiment_id in ['piControl', 'historical', 'ssp585']:
    # experiment_id = 'historical'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    cmip6_data_regridded_alltime_ens_gzm[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon'], ['tas']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'/home/563/qg8515/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id] = pickle.load(f)
        
        data11 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime).mean(dim='x', skipna=True)
        data21 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime).weighted(np.cos(np.deg2rad(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime)['lat']))).mean(dim=['x', 'y'], skipna=True)
        data12 = cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['zm'].isel(source_id=ith_source_id, time=itime)
        data22 = cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['gm'].isel(source_id=ith_source_id, time=itime)
        print((data11.values[np.isfinite(data11.values)] == data12.values[np.isfinite(data12.values)]).all())
        print(np.max(np.abs(data21 - data22)).values < 1e-4)



'''
# endregion

