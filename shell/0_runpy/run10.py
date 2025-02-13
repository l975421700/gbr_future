

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

cmip6_data_regridded_alltime_ens = {}
cmip6_data_regridded_alltime_ens_gzm = {}

for experiment_id in ['amip']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    cmip6_data_regridded_alltime_ens_gzm[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon'], ['tas']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rlutcs', 'rsds', 'rsdscs', 'rsus', 'rsuscs', 'rsutcs']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        
        for ialltime in cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id].keys():
            # ialltime = 'ann'
            print(f'#-------- {ialltime}')
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime] = {}
            
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['zm'] = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].mean(dim='x', skipna=True).compute()
            
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['gm'] = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].weighted(np.cos(np.deg2rad(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].lat))).mean(dim=['x', 'y'], skipna=True).compute().astype(np.float32)
        
        ofile=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile, 'wb') as f:
            pickle.dump(cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id], f)
        
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]
        del cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id]




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
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id] = pickle.load(f)
        
        data11 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime).mean(dim='x', skipna=True)
        data21 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime).weighted(np.cos(np.deg2rad(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime)['lat']))).mean(dim=['x', 'y'], skipna=True)
        data12 = cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['zm'].isel(source_id=ith_source_id, time=itime)
        data22 = cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['gm'].isel(source_id=ith_source_id, time=itime)
        print((data11.values[np.isfinite(data11.values)] == data12.values[np.isfinite(data12.values)]).all())
        print(np.max(np.abs(data21 - data22)).values < 1e-4)



'''
# endregion
