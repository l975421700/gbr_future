

# region import packages

import xarray as xr
import glob
import os
import pickle

import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
from calculations import mon_sea_ann

# endregion


# region get alltime hourly count of each cloud type

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
fl = sorted(glob.glob(f'/scratch/v46/qg8515/data/obs/jaxa/clp/??????/cltype_hourly_count_??????.nc'))
cltype_hourly_count = xr.open_mfdataset(fl, parallel=True).cltype_hourly_count.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
cltype_hourly_count_alltime = mon_sea_ann(
    var_monthly=cltype_hourly_count, lcopy=False, mm=True, sm=True, am=True)

ofile='/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_hourly_count_alltime, f)




'''
#-------------------------------- check
with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
fl = sorted(glob.glob(f'/scratch/v46/qg8515/data/obs/jaxa/clp/??????/cltype_hourly_count_??????.nc'))
ifile = -1
ds = xr.open_dataset(fl[ifile]).cltype_hourly_count
print((cltype_hourly_count_alltime['mon'][ifile] == ds).all().values)


'''
# endregion
