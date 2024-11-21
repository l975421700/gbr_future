

# region combined_preprocessing

def combined_preprocessing(ds_in):
    import cmip6_preprocessing.preprocessing as cpp
    
    ds=ds_in.copy()
    ds=cpp.rename_cmip6(ds)
    ds=cpp.broadcast_lonlat(ds)
    
    return ds

# endregion


# region drop_all_bounds

def drop_all_bounds(ds):
    drop_vars = [vname for vname in ds.coords
                 if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop_vars(drop_vars)

# endregion


# region open_dsets

def open_dsets(df):
    'Open datasets from cloud storage and return xarray dataset.'
    import fsspec
    import xarray as xr
    
    dsets = [xr.open_zarr(fsspec.get_mapper(ds_url), consolidated=False, use_cftime=True) for ds_url in df.zstore]
    
    try:
        ds = xr.merge(dsets, join='exact')
        return ds
    except ValueError:
        return None

# endregion


# region open_delayed

def open_delayed(df):
    import dask
    "A dask.delayed wrapper around 'open_dsets'. Allows us to open many datasets in parallel."
    return dask.delayed(open_dsets)(df)

# endregion



