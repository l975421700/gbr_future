

# qsub -I -q normal -l walltime=00:30:00,ncpus=1,mem=30GB,storage=gdata/v46


# region import packages

# data analysis
import cdsapi
client = cdsapi.Client()
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import pandas as pd

# management
import os
import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
import pickle

# self defined function
from calculations import (
    mon_sea_ann,
    regrid,
    time_weighted_mean,
    )

# endregion


# era5 sl mm
# region sst


variable = "sea_surface_temperature"
var = 'sst'
folder = 'data/obs/era5/mon/'

output_file = folder + 'era5_mon_' + var + '.nc'
output_file_regrid = folder + 'era5_mon_' + var + '_regrid.nc'
output_file_regrid_alltime = folder + 'era5_mon_' + var + '_regrid_alltime.pkl'
output_file_alltime = folder + 'era5_mon_' + var + '_alltime.pkl'


dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": variable,
    "year": [
        "1940", "1941", "1942",
        "1943", "1944", "1945",
        "1946", "1947", "1948",
        "1949", "1950", "1951",
        "1952", "1953", "1954",
        "1955", "1956", "1957",
        "1958", "1959", "1960",
        "1961", "1962", "1963",
        "1964", "1965", "1966",
        "1967", "1968", "1969",
        "1970", "1971", "1972",
        "1973", "1974", "1975",
        "1976", "1977", "1978",
        "1979", "1980", "1981",
        "1982", "1983", "1984",
        "1985", "1986", "1987",
        "1988", "1989", "1990",
        "1991", "1992", "1993",
        "1994", "1995", "1996",
        "1997", "1998", "1999",
        "2000", "2001", "2002",
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023",
        "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client.retrieve(dataset, request, output_file)


dataset = xr.open_dataset(output_file)
dataset = dataset.rename({'date': 'time', 'latitude': 'lat', 'longitude': 'lon'})
dataset = dataset.assign_coords(time = pd.to_datetime(dataset.time, format='%Y%m%d'))

dataset_regrid = regrid(dataset)
dataset_regrid_alltime = mon_sea_ann(var_monthly=dataset_regrid[var])
dataset_alltime = mon_sea_ann(var_monthly=dataset[var])


dataset_regrid.to_netcdf(output_file_regrid)
with open(output_file_regrid_alltime, 'wb') as f:
    pickle.dump(dataset_regrid_alltime, f)
with open(output_file_alltime, 'wb') as f: pickle.dump(dataset_alltime, f)


'''
#-------------------------------- check
dataset = xr.open_dataset('data/obs/era5/mon/era5_mon_sst.nc')
dataset_regrid = xr.open_dataset('data/obs/era5/mon/era5_mon_sst_regrid.nc')
with open('data/obs/era5/mon/era5_mon_sst_regrid_alltime.pkl', 'rb') as f: dataset_regrid_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_sst_alltime.pkl', 'rb') as f: dataset_alltime = pickle.load(f)

print(dataset[var].shape)
print(dataset_regrid[var].shape)
print(dataset_regrid_alltime['mon'].shape)
print(dataset_regrid_alltime.keys())
print(dataset_alltime['mon'].shape)
print(dataset_alltime.keys())

del dataset, dataset_regrid, dataset_regrid_alltime, dataset_alltime
'''
# endregion


# region t2m

variable = "2m_temperature"
var = 't2m'
folder = 'data/obs/era5/mon/'

output_file = folder + 'era5_mon_' + var + '.nc'
output_file_regrid = folder + 'era5_mon_' + var + '_regrid.nc'
output_file_regrid_alltime = folder + 'era5_mon_' + var + '_regrid_alltime.pkl'
output_file_alltime = folder + 'era5_mon_' + var + '_alltime.pkl'


dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": variable,
    "year": [
        "1940", "1941", "1942",
        "1943", "1944", "1945",
        "1946", "1947", "1948",
        "1949", "1950", "1951",
        "1952", "1953", "1954",
        "1955", "1956", "1957",
        "1958", "1959", "1960",
        "1961", "1962", "1963",
        "1964", "1965", "1966",
        "1967", "1968", "1969",
        "1970", "1971", "1972",
        "1973", "1974", "1975",
        "1976", "1977", "1978",
        "1979", "1980", "1981",
        "1982", "1983", "1984",
        "1985", "1986", "1987",
        "1988", "1989", "1990",
        "1991", "1992", "1993",
        "1994", "1995", "1996",
        "1997", "1998", "1999",
        "2000", "2001", "2002",
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023",
        "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client.retrieve(dataset, request, output_file)


dataset = xr.open_dataset(output_file)
dataset = dataset.rename({'date': 'time', 'latitude': 'lat', 'longitude': 'lon'})
dataset = dataset.assign_coords(time = pd.to_datetime(dataset.time, format='%Y%m%d'))

dataset_regrid = regrid(dataset)
dataset_regrid_alltime = mon_sea_ann(var_monthly=dataset_regrid[var])
dataset_alltime = mon_sea_ann(var_monthly=dataset[var])


dataset_regrid.to_netcdf(output_file_regrid)
with open(output_file_regrid_alltime, 'wb') as f:
    pickle.dump(dataset_regrid_alltime, f)
with open(output_file_alltime, 'wb') as f: pickle.dump(dataset_alltime, f)


'''
#-------------------------------- check
dataset = xr.open_dataset('data/obs/era5/mon/era5_mon_t2m.nc')
dataset_regrid = xr.open_dataset('data/obs/era5/mon/era5_mon_t2m_regrid.nc')
with open('data/obs/era5/mon/era5_mon_t2m_regrid_alltime.pkl', 'rb') as f: dataset_regrid_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_t2m_alltime.pkl', 'rb') as f: dataset_alltime = pickle.load(f)

print(dataset[var].shape)
print(dataset_regrid[var].shape)
print(dataset_regrid_alltime['mon'].shape)
print(dataset_regrid_alltime.keys())
print(dataset_alltime['mon'].shape)
print(dataset_alltime.keys())

del dataset, dataset_regrid, dataset_regrid_alltime, dataset_alltime
'''
# endregion


# region tp


variable = "total_precipitation"
var = 'tp'
folder = 'data/obs/era5/mon/'

output_file = folder + 'era5_mon_' + var + '.nc'
output_file_regrid = folder + 'era5_mon_' + var + '_regrid.nc'
output_file_regrid_alltime = folder + 'era5_mon_' + var + '_regrid_alltime.pkl'
output_file_alltime = folder + 'era5_mon_' + var + '_alltime.pkl'


dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": variable,
    "year": [
        "1940", "1941", "1942",
        "1943", "1944", "1945",
        "1946", "1947", "1948",
        "1949", "1950", "1951",
        "1952", "1953", "1954",
        "1955", "1956", "1957",
        "1958", "1959", "1960",
        "1961", "1962", "1963",
        "1964", "1965", "1966",
        "1967", "1968", "1969",
        "1970", "1971", "1972",
        "1973", "1974", "1975",
        "1976", "1977", "1978",
        "1979", "1980", "1981",
        "1982", "1983", "1984",
        "1985", "1986", "1987",
        "1988", "1989", "1990",
        "1991", "1992", "1993",
        "1994", "1995", "1996",
        "1997", "1998", "1999",
        "2000", "2001", "2002",
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023",
        "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client.retrieve(dataset, request, output_file)


dataset = xr.open_dataset(output_file)
dataset = dataset.rename({'date': 'time', 'latitude': 'lat', 'longitude': 'lon'})
dataset = dataset.assign_coords(time = pd.to_datetime(dataset.time, format='%Y%m%d'))

dataset_regrid = regrid(dataset)
dataset_regrid_alltime = mon_sea_ann(var_monthly=dataset_regrid[var])
dataset_alltime = mon_sea_ann(var_monthly=dataset[var])


dataset_regrid.to_netcdf(output_file_regrid)
with open(output_file_regrid_alltime, 'wb') as f:
    pickle.dump(dataset_regrid_alltime, f)
with open(output_file_alltime, 'wb') as f: pickle.dump(dataset_alltime, f)


'''
#-------------------------------- check
dataset = xr.open_dataset('data/obs/era5/mon/era5_mon_tp.nc')
dataset_regrid = xr.open_dataset('data/obs/era5/mon/era5_mon_tp_regrid.nc')
with open('data/obs/era5/mon/era5_mon_tp_regrid_alltime.pkl', 'rb') as f: dataset_regrid_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_tp_alltime.pkl', 'rb') as f: dataset_alltime = pickle.load(f)

print(dataset[var].shape)
print(dataset_regrid[var].shape)
print(dataset_regrid_alltime['mon'].shape)
print(dataset_regrid_alltime.keys())
print(dataset_alltime['mon'].shape)
print(dataset_alltime.keys())

del dataset, dataset_regrid, dataset_regrid_alltime, dataset_alltime
'''
# endregion


# region msl


variable = "mean_sea_level_pressure"
var = 'msl'
folder = 'data/obs/era5/mon/'

output_file = folder + 'era5_mon_' + var + '.nc'
output_file_regrid = folder + 'era5_mon_' + var + '_regrid.nc'
output_file_regrid_alltime = folder + 'era5_mon_' + var + '_regrid_alltime.pkl'
output_file_alltime = folder + 'era5_mon_' + var + '_alltime.pkl'


dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": variable,
    "year": [
        "1940", "1941", "1942",
        "1943", "1944", "1945",
        "1946", "1947", "1948",
        "1949", "1950", "1951",
        "1952", "1953", "1954",
        "1955", "1956", "1957",
        "1958", "1959", "1960",
        "1961", "1962", "1963",
        "1964", "1965", "1966",
        "1967", "1968", "1969",
        "1970", "1971", "1972",
        "1973", "1974", "1975",
        "1976", "1977", "1978",
        "1979", "1980", "1981",
        "1982", "1983", "1984",
        "1985", "1986", "1987",
        "1988", "1989", "1990",
        "1991", "1992", "1993",
        "1994", "1995", "1996",
        "1997", "1998", "1999",
        "2000", "2001", "2002",
        "2003", "2004", "2005",
        "2006", "2007", "2008",
        "2009", "2010", "2011",
        "2012", "2013", "2014",
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023",
        "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client.retrieve(dataset, request, output_file)


dataset = xr.open_dataset(output_file)
dataset = dataset.rename({'date': 'time', 'latitude': 'lat', 'longitude': 'lon'})
dataset = dataset.assign_coords(time = pd.to_datetime(dataset.time, format='%Y%m%d'))

dataset_regrid = regrid(dataset)
dataset_regrid_alltime = mon_sea_ann(var_monthly=dataset_regrid[var])
dataset_alltime = mon_sea_ann(var_monthly=dataset[var])


dataset_regrid.to_netcdf(output_file_regrid)
with open(output_file_regrid_alltime, 'wb') as f:
    pickle.dump(dataset_regrid_alltime, f)
with open(output_file_alltime, 'wb') as f: pickle.dump(dataset_alltime, f)


'''
#-------------------------------- check
dataset = xr.open_dataset('data/obs/era5/mon/era5_mon_msl.nc')
dataset_regrid = xr.open_dataset('data/obs/era5/mon/era5_mon_msl_regrid.nc')
with open('data/obs/era5/mon/era5_mon_msl_regrid_alltime.pkl', 'rb') as f: dataset_regrid_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_msl_alltime.pkl', 'rb') as f: dataset_alltime = pickle.load(f)

print(dataset[var].shape)
print(dataset_regrid[var].shape)
print(dataset_regrid_alltime['mon'].shape)
print(dataset_regrid_alltime.keys())
print(dataset_alltime['mon'].shape)
print(dataset_alltime.keys())

del dataset, dataset_regrid, dataset_regrid_alltime, dataset_alltime
'''
# endregion



