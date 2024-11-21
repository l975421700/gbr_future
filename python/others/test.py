
# check SST at a point
import xarray as xr

era5_mon_sst = xr.open_dataset('data/obs/era5/mon/era5_mon_sst.nc')

(era5_mon_sst.sst.sel(longitude=148.5, latitude=-17.5, method='nearest').values).mean()


import metpy

metpy.calc.pressure_to_height_std(100 * metpy.units.units('Pa'))
# 20 km


import intake
cmip6 = intake.open_esm_datastore("/g/data/dk92/catalog/v2/esm/cmip6-oi10/catalog.json")

