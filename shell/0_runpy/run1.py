

# region import packages

# data analysis
import numpy as np
import xarray as xr
import joblib

# management
import os
import glob

# endregion


# region get BARRA-C2 monthly hourly data
# qsub -I -q normal -l walltime=4:00:00,ncpus=48,mem=192GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38


var = 'cll' # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/um/barra_c2/{var}'
os.makedirs(odir, exist_ok=True)

def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'))[0]
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile, chunks={})
    ds = ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds.to_netcdf(ofile)
    
    return f'Finished processing {ofile}'


joblib.Parallel(n_jobs=48)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(1979, 2024) for month in range(1, 13))





'''
# single job
process_year_month(2020, 1, var, odir)

for var in ['cll']:
    # var = 'cll'
    # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/*'))[:540]
    
    def preprocess(ds, var=var):
        ds_out = ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]})
        return(ds_out)
    
    %timeit xr.open_mfdataset(fl[:6], preprocess=preprocess, parallel=True).compute()
    
    ds = xr.open_dataset(fl[3], chunks={})
    ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]})
    preprocess(ds)


    ds = xr.open_dataset(fl[3], chunks={})
    ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]})
    
    ds = xr.open_mfdataset(fl[:12], chunks={})[var]
    ds.groupby(['time.month', 'time.hour']).mean()
    # (lat: 1018, lon: 1298, month: 12, hour: 24)
    
    
    ds = xr.open_mfdataset(fl[:24], chunks={})[var]
    ds.groupby(['time.year', 'time.month', 'time.hour']).mean()


for var in ['cll']:
    # var = 'cll'
    # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    odir = f'scratch/data/sim/um/barra_c2/{var}'
    os.makedirs(odir, exist_ok=True)
    
    for year in np.arange(1979, 2024, 1):
        print(f'#---------------- {year}')
        for month in np.arange(1, 13, 1):
            print(f'#-------- {month}')
            # year=2020; month=1
            ifile = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'))[0]
            ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
            
            ds = xr.open_dataset(ifile, chunks={})
            ds = ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
            
            if os.path.exists(ofile): os.remove(ofile)
            ds.to_netcdf(ofile)


# dask
client = Client(processes=True)

# 0
client.compute((dask.delayed(process_year_month)(2020, month, var, odir) for month in range(3, 5)))
# 1
tasks = [process_year_month(year, month, var, odir)
         for year in range(2024, 2025)
         for month in range(1, 7)]
# 2
tasks = []
for year in np.arange(2024, 2025, 1):
    for month in np.arange(1, 7, 1):
        print(f'#---------------- {year} {month:02d}')
        task = dask.delayed(process_year_month)(year, month, var, odir)
        tasks.append(task)

# 1
for task in tasks:
    future = client.compute(task)
    print(future.result())
# 2
futures = client.compute(tasks)
client.close()


'''
# endregion

