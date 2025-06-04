
import xarray as xr
import numpy as np

ds = xr.open_dataset('/g/data/w40/clv563/BoM_data/AWS-data-QLD.nc')
ds = xr.open_dataset('/g/data/w40/clv563/BoM_data_202409/half_hourly_data_netcdf/AWS-data-QLD.nc')

ds = ds.sel(station=ds.station[ds.name == b'WILLIS ISLAND                                     ']).squeeze()

print(np.nanmean(ds.prec))
print(len(ds.prec))
print(np.isfinite(ds.prec).sum())
print(np.isnan(ds.prec).sum())
print((ds.prec==0).sum())


np.unique(ds.time[18::48].time.dt.hour)
np.nanmean(ds.prec[18::48])
np.nanmean(ds.prec[18::48]) / 24





