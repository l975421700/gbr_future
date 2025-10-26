

# region land_ocean_mean

def land_ocean_mean(da):
    # da: xarray.DataArray
    
    import regionmask
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
    mask = land.mask(da)
    
    landmean = coslat_weighted_mean(da.where(~np.isnan(mask)))
    oceanmean = coslat_weighted_mean(da.where(np.isnan(mask)))
    
    return(mask, landmean, oceanmean)

'''
#-------- check
import regionmask
land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
mask = land.mask(plt_data)

print(np.sum(plt_data))
print(np.sum(plt_data.where(~np.isnan(mask))) + np.sum(plt_data.where(np.isnan(mask))))

print(coslat_weighted_mean(plt_data.where(~np.isnan(mask))))
print(coslat_weighted_mean(plt_data.where(np.isnan(mask))))

'''
# endregion


# region get_LTS

from metpy.calc import potential_temperature
from metpy.units import units
def get_LTS(tas, ps, ta700):
    # tas/ta700 in K, ps in Pa
    thetas = potential_temperature(ps * units.Pa, tas * units.K)
    theta700 = potential_temperature(700 * units.hPa, ta700 * units.K)
    LTS = theta700 - thetas
    return(LTS)

# endregion


# region get_inversion

import numpy as np
def get_inversion(temperature, altitude, topo = 0, oinversiont=False):
    '''
    Input --------
    temperature: 1D, bottom-up
    altitude: 1D in m
    
    Output --------
    inversionh, inversiont: scalar
    '''
    
    temperature = temperature[altitude > topo].copy()
    altitude = altitude[altitude > topo].copy()
    
    try:
        level = np.where(temperature[1:] - temperature[:-1] > 0)[0][0]
        inversionh = altitude[level]
        inversiont = temperature[level]
    except:
        inversionh = np.nan
        inversiont = np.nan
    
    if oinversiont:
        return(inversionh, inversiont)
    else:
        return(inversionh)


from numba import njit
@njit
def get_inversion_numba(temperature, altitude, topo=0.0):
    mask = altitude > topo
    t = temperature[mask]
    z = altitude[mask]
    
    n = t.size
    if n < 2:
        return np.nan
    
    for i in range(n-1):
        if t[i+1] - t[i] > 0:
            return z[i]
    
    return np.nan


'''
'''
# endregion


# region get_LCL

def get_LCL(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):
    # p in Pascals, T in Kelvins, rh dimensionless

   import math
   import scipy.special

   # Parameters
   Ttrip = 273.16     # K
   ptrip = 611.65     # Pa
   E0v   = 2.3740e6   # J/kg
   E0s   = 0.3337e6   # J/kg
   ggr   = 9.81       # m/s^2
   rgasa = 287.04     # J/kg/K 
   rgasv = 461        # J/kg/K 
   cva   = 719        # J/kg/K
   cvv   = 1418       # J/kg/K 
   cvl   = 4119       # J/kg/K 
   cvs   = 1861       # J/kg/K 
   cpa   = cva + rgasa
   cpv   = cvv + rgasv

   # The saturation vapor pressure over liquid water
   def pvstarl(T):
      return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
         math.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )
   
   # The saturation vapor pressure over solid ice
   def pvstars(T):
      return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
         math.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

   # Calculate pv from rh, rhl, or rhs
   rh_counter = 0
   if rh  is not None:
      rh_counter = rh_counter + 1
   if rhl is not None:
      rh_counter = rh_counter + 1
   if rhs is not None:
      rh_counter = rh_counter + 1
   if rh_counter != 1:
      print(rh_counter)
      exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
   if rh is not None:
      # The variable rh is assumed to be 
      # with respect to liquid if T > Ttrip and 
      # with respect to solid if T < Ttrip
      if T > Ttrip:
         pv = rh * pvstarl(T)
      else:
         pv = rh * pvstars(T)
      rhl = pv / pvstarl(T)
      rhs = pv / pvstars(T)
   elif rhl is not None:
      pv = rhl * pvstarl(T)
      rhs = pv / pvstars(T)
      if T > Ttrip:
         rh = rhl
      else:
         rh = rhs
   elif rhs is not None:
      pv = rhs * pvstars(T)
      rhl = pv / pvstarl(T)
      if T > Ttrip:
         rh = rhl
      else:
         rh = rhs
   if pv > p:
      return None

   # Calculate lcl_liquid and lcl_solid
   qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
   rgasm = (1-qv)*rgasa + qv*rgasv
   cpm = (1-qv)*cpa + qv*cpv
   if rh == 0:
      return cpm*T/ggr
   aL = -(cpv-cvl)/rgasv + cpm/rgasm
   bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
   cL = pv/pvstarl(T)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
   aS = -(cpv-cvs)/rgasv + cpm/rgasm
   bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
   cS = pv/pvstars(T)*math.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
   lcl = cpm*T/ggr*( 1 - \
      bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
   ldl = cpm*T/ggr*( 1 - \
      bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )

   # Return either lcl or ldl
   if return_ldl and return_min_lcl_ldl:
      exit('return_ldl and return_min_lcl_ldl cannot both be true')
   elif return_ldl:
      return ldl
   elif return_min_lcl_ldl:
      return min(lcl,ldl)
   else:
      return lcl




'''
# reference: https://romps.berkeley.edu/papers/pubs-2016-lcl.html
# Version 1.0 released by David Romps on September 12, 2017.

# (LCL) in meters.  The inputs are:
# - p in Pascals
# - T in Kelvins
# - Exactly one of rh, rhl, and rhs (dimensionless, from 0 to 1):
#    * The value of rh is interpreted to be the relative humidity with
#      respect to liquid water if T >= 273.15 K and with respect to ice if
#      T < 273.15 K. 
#    * The value of rhl is interpreted to be the relative humidity with
#      respect to liquid water
#    * The value of rhs is interpreted to be the relative humidity with
#      respect to ice
# - return_ldl is an optional logical flag.  If true, the lifting deposition
#   level (LDL) is returned instead of the LCL. 
# - return_min_lcl_ldl is an optional logical flag.  If true, the minimum of the
#   LCL and LDL is returned.

# test
# exec(open('lcl.py').read())

if abs(get_LCL(1e5,300,rhl=.5,return_ldl=False)/( 1433.844139279)-1) < 1e-10 and \
   abs(get_LCL(1e5,300,rhs=.5,return_ldl=False)/( 923.2222457185)-1) < 1e-10 and \
   abs(get_LCL(1e5,200,rhl=.5,return_ldl=False)/( 542.8017712435)-1) < 1e-10 and \
   abs(get_LCL(1e5,200,rhs=.5,return_ldl=False)/( 1061.585301941)-1) < 1e-10 and \
   abs(get_LCL(1e5,300,rhl=.5,return_ldl=True )/( 1639.249726127)-1) < 1e-10 and \
   abs(get_LCL(1e5,300,rhs=.5,return_ldl=True )/( 1217.336637217)-1) < 1e-10 and \
   abs(get_LCL(1e5,200,rhl=.5,return_ldl=True )/(-8.609834216556)-1) < 1e-10 and \
   abs(get_LCL(1e5,200,rhs=.5,return_ldl=True )/( 508.6366558898)-1) < 1e-10:
   print('Success')
else:
   print('Failure')


# check
def get_LCL2(pres, tem, rh):
    # ----Input
    # pres: in Pascals
    # tem: in Kelvins
    # rh: relative humidity with respect to liquid water if T >= 273.15 K
    #                       with respect to ice if T < 273.15 K.
    
    # ----output
    # lcl: the height of the lifting condensation level (LCL) in meters.
    import math
    import scipy.special
    import numpy as np
    
    # Parameters
    Ttrip = 273.16     # K
    ptrip = 611.65     # Pa
    E0v = 2.3740e6   # J/kg
    E0s = 0.3337e6   # J/kg
    ggr = 9.81       # m/s^2
    rgasa = 287.04     # J/kg/K
    rgasv = 461        # J/kg/K
    cva = 719        # J/kg/K
    cvv = 1418       # J/kg/K
    cvl = 4119       # J/kg/K
    cvs = 1861       # J/kg/K
    cpa = cva + rgasa
    cpv = cvv + rgasv
    
    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
        math.exp((E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T))
    
    # The saturation vapor pressure over solid ice
    def pvstars(T):
        return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
        math.exp((E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T))
    
    # Calculate pv from rh
    # The variable rh is assumed to be
    # with respect to liquid if T > Ttrip and
    # with respect to solid if T < Ttrip
    if tem > Ttrip:
        pv = rh * pvstarl(tem)
    else:
        pv = rh * pvstars(tem)
    
    rhl = pv / pvstarl(tem)
    rhs = pv / pvstars(tem)
    
    if pv > pres:
        return np.nan
    
    # Calculate lcl_liquid and lcl_solid
    qv = rgasa*pv / (rgasv*pres + (rgasa-rgasv)*pv)
    rgasm = (1-qv)*rgasa + qv*rgasv
    cpm = (1-qv)*cpa + qv*cpv
    if rh == 0:
        return cpm*tem/ggr
    
    aL = -(cpv-cvl)/rgasv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*tem)
    cL = pv/pvstarl(tem)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*tem))
    lcl = cpm*tem/ggr*(
        1 - bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL), -1).real))
    
    return lcl


from metpy.units import units
import metpy.calc as mpcalc
pres = 101320.75
tem = 296.5619
rh = 0.90995485

get_LCL(pres, tem, rh)
get_LCL2(pres, tem, rh)

'''
# endregion


# region get_EIS

from typhon.physics import moist_lapse_rate
def get_EIS(tas, ps, ta700, hurs, zg700, topo):
    # tas/ta700 in K, ps in Pa, hurs dimensionless, zg700 in m
    
    LCL = get_LCL(ps, tas, hurs)
    LTS = get_LTS(tas, ps, ta700)
    gamma = moist_lapse_rate(850 * 100, (tas+ta700)/2)
    EIS = LTS.m - gamma * (zg700 - topo - LCL)
    return(EIS)


def get_EIS_simplified(LCL, LTS, tas, ta700, zg700, topo):
    # tas/ta700 in K, ps in Pa, hurs dimensionless, zg700 in m
    
    # LCL = get_LCL(ps, tas, hurs)
    # LTS = get_LTS(tas, ps, ta700)
    gamma = moist_lapse_rate(850 * 100, (tas+ta700)/2)
    EIS = LTS - gamma * (zg700 - topo - LCL)
    return(EIS)


'''
https://arts2.mi.uni-hamburg.de/misc/typhon/doc/generated/typhon.physics.moist_lapse_rate.html?utm_source=chatgpt.com
'''
# endregion


# region coslat_weighted_mean, coslat_weighted_rmsd

def coslat_weighted_mean(dataarray):
    import numpy as np
    return(dataarray.weighted(np.cos(np.deg2rad(dataarray.lat))).mean().values)

def coslat_weighted_rmsd(da_diff):
    import numpy as np
    return(np.sqrt(np.square(da_diff).weighted(np.cos(np.deg2rad(da_diff.lat))).mean()).values)

# endregion


# region time_weighted_mean

def time_weighted_mean(ds):
    '''
    #---- Input
    ds: xarray.DataArray
    '''
    
    return ds.weighted(ds.time.dt.days_in_month).mean(dim='time', skipna=False)

# endregion


# region mon_sea_ann

def mon_sea_ann(
    var_daily = None, var_monthly = None, var_6hourly = None,
    lcopy = True, seasons = 'QE-FEB',
    mm=False, sm=False, am=False, mon_no_mm=False, ann_no_am=False,
    lsea_cpt=True, lann_cpt=True, lfloat32=True,
    ):
    '''
    #---- Input
    var_daily:   xarray.DataArray, daily variables, must have time dimension
    var_monthly: xarray.DataArray, monthly variables, must have time dimension
    lcopy:       whether to use copy of original var
    
    #---- Output
    var_alltime
    
    '''
    
    var_alltime = {}
    
    if not var_daily is None:
        if lcopy:
            var_alltime['daily'] = var_daily.copy()
        else:
            var_alltime['daily'] = var_daily
        
        var_alltime['mon'] = var_daily.resample({'time': '1M'}).mean(skipna=False).compute()
        
    elif not var_monthly is None:
        if lcopy:
            var_alltime['mon'] = var_monthly.copy().compute()
        else:
            var_alltime['mon'] = var_monthly
        
    elif not var_6hourly is None:
        if lcopy:
            var_alltime['6h'] = var_6hourly.copy()
        else:
            var_alltime['6h'] = var_6hourly
        
        var_alltime['daily'] = var_6hourly.resample({'time': '1d'}).mean(skipna=False).compute()
        
        var_alltime['mon'] = var_alltime['daily'].resample({'time': '1M'}).mean(skipna=False).compute()
    
    if (seasons == 'QE-FEB'):
        var_alltime['sea'] = var_alltime['mon'].resample({'time': seasons}).map(time_weighted_mean)[1:-1]
    elif (seasons == 'QE-MAR'):
        var_alltime['sea'] = var_alltime['mon'].resample({'time': seasons}).map(time_weighted_mean)
    if lsea_cpt: var_alltime['sea'] = var_alltime['sea'].compute()
    
    var_alltime['ann'] = var_alltime['mon'].resample({'time': '1YE'}).map(time_weighted_mean).compute()
    if lann_cpt: var_alltime['ann'] = var_alltime['ann'].compute()
    
    if mm:
        var_alltime['mm'] = var_alltime['mon'].groupby('time.month').mean(skipna=True).compute()
        var_alltime['mm'] = var_alltime['mm'].rename({'month': 'time'})
    
    if sm:
        var_alltime['sm'] = var_alltime['sea'].groupby('time.season').mean(skipna=True).compute()
        var_alltime['sm'] = var_alltime['sm'].rename({'season': 'time'})
    
    if am:
        var_alltime['am'] = var_alltime['ann'].mean(dim='time', skipna=True).compute()
        var_alltime['am'] = var_alltime['am'].expand_dims('time', axis=0)
    
    if mon_no_mm:
        var_alltime['mon no mm'] = (var_alltime['mon'].groupby('time.month') - var_alltime['mm']).compute()
    
    if ann_no_am:
        var_alltime['ann no am'] = (var_alltime['ann'] - var_alltime['am']).compute()
    
    if lfloat32:
        for key in var_alltime.keys():
            var_alltime[key] = var_alltime[key].astype('float32')
    
    return(var_alltime)




'''
#---- skipna
mon:    False
sea:    False
ann:    False
# otherwise, the results will be biased

mm:     True
sm:     True
am:     True
# it's okay to miss several month/season/year


#-------------------------------- check
import xarray as xr
import pandas as pd
import numpy as np

x = np.arange(0, 360, 1)
y = np.arange(-90, 90, 1)

#-------- check daily data

time = pd.date_range( "2001-01-01-00", "2009-12-31-00", freq="1D")

ds = xr.DataArray(
    data = np.random.rand(len(time),len(x), len(y)),
    coords={
            "time": time,
            "x": x,
            "y": y,
        }
)

ds_alltime = mon_sea_ann(ds)

# calculation in function and manually
(ds_alltime['mon'] == ds.resample({'time': '1M'}).mean()).all().values
(ds_alltime['sea'] == ds_alltime['mon'].resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].compute()).all().values
(ds_alltime['ann'] == ds_alltime['mon'].resample({'time': '1YE'}).map(time_weighted_mean).compute()).all().values
(ds_alltime['mm'] == ds_alltime['mon'].groupby('time.month').mean()).all().values
(ds_alltime['sm'] == ds_alltime['sea'].groupby('time.season').mean()).all().values
(ds_alltime['am'] == ds_alltime['ann'].mean(dim='time')).all().values
(ds_alltime['mon no mm'] == ds_alltime['mon'].groupby('time.month') - ds_alltime['mm']).all().values
(ds_alltime['ann no am'] == ds_alltime['ann'] - ds_alltime['am']).all().values

# mon
ilat=30
ilon=20
ds[-31:, ilat, ilon].values.mean()
ds_alltime['mon'][-1, ilat, ilon].values

# sea
ilat=30
ilon=40
ds[59:151, ilat, ilon].values.mean()
np.average(
    ds_alltime['mon'][2:5, ilat, ilon],
    weights=ds_alltime['mon'][2:5, ilat, ilon].time.dt.days_in_month,
)
ds_alltime['sea'][0, ilat, ilon].values

# ann
ilat=30
ilon=40
ds[:365, ilat, ilon].mean().values
np.average(
    ds_alltime['mon'][0:12, ilat, ilon],
    weights=ds_alltime['mon'][0:12, ilat, ilon].time.dt.days_in_month,
)
ds_alltime['ann'][0, ilat, ilon].values

# mm
ilat=30
ilon=60
ds_alltime['mon'].sel(time=(ds_alltime['mon'].time.dt.month==6).values)[:, ilat, ilon].mean().values
ds_alltime['mm'][5, ilat, ilon]

# sm
ds_alltime['sea'].sel(time=(ds_alltime['sea'].time.dt.month==8).values)[:, ilat, ilon].mean().values
ds_alltime['sm'].sel(season='JJA')[ilat, ilon].values

# am
ds_alltime['ann'][:, ilat, ilon].mean().values
ds_alltime['am'][ilat, ilon].values

#---- check 'QE-MAR'
ds_alltime = mon_sea_ann(ds, seasons = 'QE-MAR',)

# calculation in function and manually
(ds_alltime['mon'] == ds.resample({'time': '1M'}).mean()).all().values
(ds_alltime['sea'] == ds_alltime['mon'].resample({'time': 'QE-MAR'}).map(time_weighted_mean).compute()).all().values
(ds_alltime['ann'] == ds_alltime['mon'].resample({'time': '1YE'}).map(time_weighted_mean).compute()).all().values
(ds_alltime['mm'] == ds_alltime['mon'].groupby('time.month').mean()).all().values
(ds_alltime['sm'] == ds_alltime['sea'].groupby('time.month').mean()).all().values
(ds_alltime['am'] == ds_alltime['ann'].mean(dim='time')).all().values
(ds_alltime['mon no mm'] == ds_alltime['mon'].groupby('time.month') - ds_alltime['mm']).all().values
(ds_alltime['ann no am'] == ds_alltime['ann'] - ds_alltime['am']).all().values

#-------- check monthly data

time = pd.date_range( "2001-01-01-00", "2009-12-31-00", freq="1M")

ds = xr.DataArray(
    data = np.random.rand(len(time),len(x), len(y)),
    coords={
            "time": time,
            "x": x,
            "y": y,
        }
)

ds_alltime = mon_sea_ann(var_monthly = ds)

(ds_alltime['mon'] == ds).all().values
(ds_alltime['sea'] == ds.resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].compute()).all().values
(ds_alltime['ann'] == ds.resample({'time': '1YE'}).map(time_weighted_mean).compute()).all().values
(ds_alltime['mm'] == ds_alltime['mon'].groupby('time.month').mean()).all().values
(ds_alltime['sm'] == ds_alltime['sea'].groupby('time.season').mean()).all().values
(ds_alltime['am'] == ds_alltime['ann'].mean(dim='time')).all().values
(ds_alltime['mon no mm'] == ds_alltime['mon'].groupby('time.month') - ds_alltime['mm']).all().values
(ds_alltime['ann no am'] == ds_alltime['ann'] - ds_alltime['am']).all().values

# seasonal
i1 = 30
i3 = 40
i4 = 60
np.average(
    ds[(i1*3 + 2):(i1*3 + 5), i3, i4],
    weights = ds[(i1*3 + 2):(i1*3 + 5), i3, i4].time.dt.days_in_month)
ds_alltime['sea'][i1, i3, i4].values

# annual
i1 = 6
i3 = 30
i4 = 60
np.average(
    ds[(i1*12 + 0):(i1*12 + 12), i3, i4],
    weights = ds[(i1*12 + 0):(i1*12 + 12), i3, i4].time.dt.days_in_month)
ds_alltime['ann'][i1, i3, i4].values


test1 = ds.weighted(ds.time.dt.days_in_month).mean(dim='time', skipna=True).compute()
test2 = test1.values[np.isfinite(test1.values)] - ds_alltime['am'].values[np.isfinite(ds_alltime['am'].values)]
wheremax = np.where(abs(test2) == np.max(abs(test2)))
test2[wheremax]
np.max(abs(test2))
test1.values[np.isfinite(test1.values)][wheremax]
ds_alltime['am'].values[np.isfinite(ds_alltime['am'].values)][wheremax]

# (np.isfinite(test1.values) == np.isfinite(ds_alltime['am'].values)).all()


'''
# endregion


# region mon_sea_ann_average

def mon_sea_ann_average(ds, average, skipna = True):
    '''
    ds: xarray.DataArray, monthly mean values
    average: 'time.month', 'time.season', 'time.year'
    '''
    month_length = ds.time.dt.days_in_month
    
    weights = (
        month_length.groupby(average) /
        month_length.groupby(average).sum()
    )
    
    ds_weighted = (
        ds * weights).groupby(average).sum(dim="time", skipna=skipna)
    
    # Calculate the weighted average
    return ds_weighted

'''
# check the monthly average
month_length = era5_mon_sl_79_21_pre.time.dt.days_in_month

weights = (
    month_length.groupby('time.month') /
    month_length.groupby('time.month').sum()
)

pre_mon_average0 = (pre * weights).groupby('time.month').sum(dim="time")
pre_mon_average1 = mon_sea_ann_average(pre, 'time.month')
(pre_mon_average0 == pre_mon_average1).all()


# check the seasonal average
month_length = era5_mon_sl_79_21_pre.time.dt.days_in_month
weights = (
    month_length.groupby("time.season") /
    month_length.groupby("time.season").sum()
    )
pre_sea_average0 = (pre * weights).groupby("time.season").sum(dim="time")
pre_sea_average1 = mon_sea_ann_average(pre, 'time.season')
(pre_sea_average0 == pre_sea_average1).all()

# check the annual average
month_length = era5_mon_sl_79_21_pre.time.dt.days_in_month
weights = (
    month_length.groupby('time.year') /
    month_length.groupby('time.year').sum()
    )
pre_ann_average0 = (pre * weights).groupby("time.year").sum(dim="time")
pre_ann_average1 = mon_sea_ann_average(pre, 'time.year')
(pre_ann_average0 == pre_ann_average1).all()

'''
# endregion


# region regrid

def regrid(
    ds_in, ds_out=None, grid_spacing=1, method='bilinear',
    periodic=True, ignore_degenerate=True, unmapped_to_nan=True,
    extrap_method='nearest_s2d', extrap_num_src_pnts=8):
    '''
    ds_in: original xarray.DataArray
    ds_out: xarray.DataArray with target grid, default None
    '''
    
    import xesmf as xe
    
    if ds_out is None: ds_out = xe.util.grid_global(grid_spacing, grid_spacing)
    
    regridder = xe.Regridder(
        ds_in, ds_out, method, periodic=periodic,
        ignore_degenerate=ignore_degenerate, unmapped_to_nan=unmapped_to_nan,
        extrap_method=extrap_method, extrap_num_src_pnts=extrap_num_src_pnts)
    return regridder(ds_in)

'''
'''
# endregion


# region pop_fillvalue

def pop_fillvalue(ds):
    for coord in ds.coords:
        if ('_FillValue' in ds[coord].encoding and 'missing_value' in ds[coord].encoding):
            ds[coord].encoding["missing_value"] = None
    return ds

# endregion


# region cdo_regrid

def cdo_regrid(ds_in, target_grid='global_1'):
    import xarray as xr
    import tempfile
    from cdo import Cdo
    cdo=Cdo()
    
    with tempfile.NamedTemporaryFile(suffix='.nc') as temp_input:
        try:
            ds_in.to_netcdf(temp_input.name)
        except:
            print('Warning file output error')
            pop_fillvalue(regrid(ds_in)).to_netcdf(temp_input.name)
        with tempfile.NamedTemporaryFile(suffix='.nc') as temp_output:
            try:
                cdo.remapcon(target_grid, input=temp_input.name, output=temp_output.name)
            except:
                cdo.remapbil(target_grid, input=temp_input.name, output=temp_output.name)
            
            ds_out = xr.open_dataset(temp_output.name,use_cftime=True).compute()
    
    return(ds_out)


# endregion


# region inversion_top

# Function to find inversion top, defined as the first layer where temperature decreases with height
def inversion_top(temperature, height, height_unit = 'km'):
    '''
    Input --------
    temperature:
    height: decreasing, in km
    
    Output --------
    t_it:
    h_it:
    '''
    
    import numpy as np
    
    if (height_unit == 'm'):
        height = height.copy() / 1000
    
    if (height[0] > height[1]): # decreasing
        try:
            level = np.where(temperature[1:] - temperature[:-1] > 0)[0][-1]
            
            if (level == (len(temperature) - 2)):
                t_it = np.nan
                h_it = np.nan
            else:
                t_it = temperature[level + 1]
                h_it = height[level + 1]
        except:
            t_it = np.nan
            h_it = np.nan
    else:
        try:
            level = np.where(temperature[1:] - temperature[:-1] < 0)[0][0]
            if (level == 0):
                t_it = np.nan
                h_it = np.nan
            else:
                t_it = temperature[level]
                h_it = height[level]
        except:
            t_it = np.nan
            h_it = np.nan
    
    if (h_it > 5):
        t_it = np.nan
        h_it = np.nan
    
    return(t_it, h_it)






'''
# test

/albedo/work/user/qigao001/a_basic_analysis/c_codes/4_d_excess/4.4_climate/4.4.0_inversion.py

i = 0
imon = 0
isite = 'Rothera'
ilat = t63_sites_indices[isite]['ilat']
ilon = t63_sites_indices[isite]['ilon']
temperature = zh_st_ml[expid[i]]['st']['mon'][imon, :, ilat, ilon].values
height = zh_st_ml[expid[i]]['zh']['mon'][imon, :, ilat, ilon].values / 1000

t_it, h_it = inversion_top(temperature, height)


/albedo/work/user/qigao001/a_basic_analysis/c_codes/4_d_excess/4.4_climate/4.4.1_EDC_radiosonde.py

import pandas as pd
import numpy as np
EDC_df_drvd = pd.read_pickle('scratch/radiosonde/igra2/EDC_df_drvd.pkl')
date = np.unique(EDC_df_drvd.date)

altitude = EDC_df_drvd.iloc[
    np.where(EDC_df_drvd.date == date[i])[0]][
        'calculated_height'].values / 1000
temperature = EDC_df_drvd.iloc[
    np.where(EDC_df_drvd.date == date[i])[0]][
    'temperature'].values

'''
# endregion


# region find_ilat_ilon


#-------------------------------- find ilat/ilon for site lat/lon

def find_ilat_ilon(slat, slon, lat, lon):
    '''
    #-------- Input
    slat: latitude, scalar
    slon: longitude, scalar
    
    lat:  latitude, 1d array
    lon:  longitude, 1d array
    '''
    
    import numpy as np
    
    if (slon < 0): slon += 360
    lon[lon < 0] += 360
    
    ilon = np.argmin(np.abs(slon - lon))
    ilat = np.argmin(np.abs(slat - lat))
    
    return([ilat, ilon])


#-------------------------------- find ilat/ilon: general approach

def find_ilat_ilon_general(slat, slon, lat, lon):
    '''
    #-------- Input
    slat: site latitude, scalar
    slon: site longitude, scalar
    
    lat:  latitude, 1d or 2d array
    lon:  longitude, 1d or 2d array
    '''
    
    import numpy as np
    from haversine import haversine_vector
    
    if (lon.ndim == 2):
        lon1d = lon.reshape(-1, 1)
        lat1d = lat.reshape(-1, 1)
    elif (lon.ndim == 1):
        lon1d = lon
        lat1d = lat
    
    slocation_pair = [slat, slon]
    location_pairs = [[x, y] for x, y in zip(lat1d, lon1d)]
    
    distances1d = haversine_vector(
        slocation_pair, location_pairs, comb=True, normalize=True,
        )
    
    if (lon.ndim == 2):
        distances = distances1d.reshape(lon.shape)
    elif (lon.ndim == 1):
        distances = distances1d
    
    wheremin = np.where(distances == np.nanmin(distances))
    
    iind0 = wheremin[0][0]
    iind1 = wheremin[-1][0]
    
    return([iind0, iind1])


#-------------------------------- find grid value at site

def find_gridvalue_at_site(slat, slon, lat, lon, gridded_data):
    '''
    #-------- Input
    slat: site latitude, scalar
    slon: site longitude, scalar
    
    lat:  latitude, 1d or 2d array
    lon:  longitude, 1d or 2d array
    
    gridded_data: 2d array
    '''
    
    import numpy as np
    
    if (np.isnan(slat) | np.isnan(slon)):
        gridvalue = np.nan
    else:
        if (lon.ndim == 2):
            ilat, ilon = find_ilat_ilon_general(slat, slon, lat, lon)
        elif (lon.ndim == 1):
            ilat, ilon = find_ilat_ilon(slat, slon, lat, lon)
        
        gridvalue = gridded_data[ilat, ilon]
    
    return(gridvalue)


#-------------------------------- find a series of grid values at sites

def find_multi_gridvalue_at_site(latitudes, longitudes, lat, lon, gridded_data):
    '''
    #-------- Input
    latitudes: 1d array
    longitudes: 1d array
    
    lat: 1d or 2d array
    lon: 1d or 2d array
    
    gridded_data: 2d array
    '''
    
    import numpy as np
    
    gridvalues = np.zeros(len(latitudes))
    
    for i in range(len(latitudes)):
        gridvalues[i] = find_gridvalue_at_site(
            latitudes[i], longitudes[i], lat, lon, gridded_data)
    
    return(gridvalues)


#-------------------------------- find grid value at site: multiple methods

def find_gridvalue_at_site_interp(
    slat, slon, lat, lon, gridded_data, method='linear'):
    '''
    #-------- Input
    slat: site latitude, scalar
    slon: site longitude, scalar
    
    lat:  latitude, 1d array
    lon:  longitude, 1d array
    
    gridded_data: 2d array
    
    method: “linear”, “nearest”, “slinear”, “cubic”, “quintic”, “pchip”, and “splinef2d”
    '''
    
    import numpy as np
    from scipy.interpolate import interpn
    
    if (slon < 0): slon += 360
    lon[lon < 0] += 360
    
    if (np.isnan(slat) | np.isnan(slon)):
        gridvalue = np.nan
    else:
        points = (lat, lon)
        point  = np.array([slat, slon])
        gridvalue = interpn(
            points, gridded_data, point, method=method,
            bounds_error=False, fill_value=None)
    
    return(gridvalue)


#-------------------------------- find a series of grid values: multiple methods

def find_multi_gridvalue_at_site_interpn(
    latitudes, longitudes, lat, lon, gridded_data, method='linear'):
    '''
    #-------- Input
    latitudes: 1d array
    longitudes: 1d array
    
    lat: 1d array
    lon: 1d array
    
    gridded_data: 2d array
    
    method: “linear”, “nearest”, “slinear”, “cubic”, “quintic”, “pchip”, and “splinef2d”
    '''
    
    import numpy as np
    
    gridvalues = np.zeros(len(latitudes))
    
    for i in range(len(latitudes)):
        gridvalues[i] = find_gridvalue_at_site_interp(
            latitudes[i], longitudes[i], lat, lon, gridded_data, method=method)
    
    return(gridvalues)


#-------------------------------- find grid value at site time

def find_gridvalue_at_site_time(
    stime, slat, slon,
    time, lat, lon,
    gridded_data):
    '''
    #-------- Input
    stime: site time, Timestamp
    slat: site latitude, scalar
    slon: site longitude, scalar
    
    time: time, datetime64[ns]
    lat:  latitude, 1d or 2d array
    lon:  longitude, 1d or 2d array
    
    gridded_data: 3d array
    '''
    
    import numpy as np
    
    if (np.isnan(slat) | np.isnan(slon)):
        gridvalue = np.nan
    else:
        itime = np.argmin(abs(stime.asm8 - time))
        
        if (lon.ndim == 2):
            ilat, ilon = find_ilat_ilon_general(slat, slon, lat, lon)
        elif (lon.ndim == 1):
            ilat, ilon = find_ilat_ilon(slat, slon, lat, lon)
        
        gridvalue = gridded_data[itime, ilat, ilon]
    
    return(gridvalue)


def find_multi_gridvalue_at_site_time(
    times, latitudes, longitudes,
    time, lat, lon, gridded_data):
    '''
    #-------- Input
    latitudes: 1d array
    longitudes: 1d array
    
    lat: 1d or 2d array
    lon: 1d or 2d array
    
    gridded_data: 2d array
    '''
    
    import numpy as np
    
    gridvalues = np.zeros(len(latitudes))
    
    for i in range(len(latitudes)):
        gridvalues[i] = find_gridvalue_at_site_time(
            times[i], latitudes[i], longitudes[i],
            time, lat, lon,
            gridded_data)
    
    return(gridvalues)


'''
#-------------------------------- check find_ilat_ilon_general

from haversine import haversine

model = list(lig_sst.keys())[-1]
# model = 'AWI-ESM-1-1-LR'

slat = ec_sst_rec['original'].Latitude[2]
slon = ec_sst_rec['original'].Longitude[2]
lon = pi_sst[model].lon.values
lat = pi_sst[model].lat.values

iind0, iind1 = find_ilat_ilon_general(slat, slon, lat, lon)

if (lon.ndim == 2):
    print(lon[iind0, iind1])
    print(lat[iind0, iind1])
elif (lon.ndim == 1):
    print(lon[iind0])
    print(lat[iind0])

print(slon)
print(slat)

if (lon.ndim == 2):
    print(haversine([slat, slon], [lat[iind0, iind1], lon[iind0, iind1]]))
elif (lon.ndim == 1):
    print(haversine([slat, slon], [lat[iind0], lon[iind0]]))


#-------------------------------- check find_gridvalue_at_site
from scipy.interpolate import interpn
from a_basic_analysis.b_module.basic_calculations import find_gridvalue_at_site

irecord = 100

points = (lat.values, lon.values)
values = d_ln_alltime[expid[i]]['am'].values
point = np.array([Antarctic_snow_isotopes['lat'][irecord],
                  Antarctic_snow_isotopes['lon'][irecord]])
print(interpn(points, values, point, method='linear'))

print(find_gridvalue_at_site(
    Antarctic_snow_isotopes['lat'][irecord],
    Antarctic_snow_isotopes['lon'][irecord],
    lat.values,
    lon.values,
    d_ln_alltime[expid[i]]['am'].values))

print(find_gridvalue_at_site(
    Antarctic_snow_isotopes['lat'][irecord],
    Antarctic_snow_isotopes['lon'][irecord],
    lat_2d,
    lon_2d,
    d_ln_alltime[expid[i]]['am'].values))


#-------------------------------- check find_multi_gridvalue_at_site

import pandas as pd
import numpy as np
import pickle

Antarctic_snow_isotopes = pd.read_csv(
    'data_sources/ice_core_records/Antarctic_snow_isotopic_composition/Antarctic_snow_isotopic_composition_DB.tab',
    sep='\t', header=0, skiprows=97,)

Antarctic_snow_isotopes = Antarctic_snow_isotopes.rename(columns={
    'Latitude': 'lat',
    'Longitude': 'lon',
    'δD [‰ SMOW] (Calculated average/mean values)': 'dD',
    'δ18O H2O [‰ SMOW] (Calculated average/mean values)': 'dO18',
    'd xs [‰] (Calculated average/mean values)': 'd-excess',
})

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_600_5.0',]
i = 0
d_ln_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.d_ln_alltime.pkl', 'rb') as f:
    d_ln_alltime[expid[i]] = pickle.load(f)

lon = d_ln_alltime[expid[0]]['am'].lon
lat = d_ln_alltime[expid[0]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

latitudes = Antarctic_snow_isotopes['lat'].values
longitudes = Antarctic_snow_isotopes['lon'].values

result = find_multi_gridvalue_at_site(
    latitudes,
    longitudes,
    d_ln_alltime[expid[i]]['am'].lat.values,
    d_ln_alltime[expid[i]]['am'].lon.values,
    d_ln_alltime[expid[i]]['am'].values,
    )


for irecord in range(len(latitudes)):
    # irecord = 100
    
    slat = Antarctic_snow_isotopes['lat'].values[irecord]
    slon = Antarctic_snow_isotopes['lon'].values[irecord]
    
    if (np.isfinite(slat) & np.isfinite(slon)):
        ilat, ilon = find_ilat_ilon_general(slat, slon, lat_2d, lon_2d)
        site_value = d_ln_alltime[expid[i]]['am'].values[ilat, ilon]
    else:
        site_value = np.nan
    
    if (np.isfinite(site_value)):
        if (site_value != result[irecord]):
            print('mismatch: ' + str(irecord))

#-------------------------------- check find_multi_gridvalue_at_site_interpn

result1 = find_multi_gridvalue_at_site(
    Antarctic_snow_isotopes['lat'].values,
    Antarctic_snow_isotopes['lon'].values,
    d_ln_alltime[expid[i]]['am'].lat.values,
    d_ln_alltime[expid[i]]['am'].lon.values,
    d_ln_alltime[expid[i]]['am'].values,
    )

result2 = find_multi_gridvalue_at_site_interpn(
    Antarctic_snow_isotopes['lat'].values,
    Antarctic_snow_isotopes['lon'].values,
    d_ln_alltime[expid[i]]['am'].lat.values,
    d_ln_alltime[expid[i]]['am'].lon.values,
    d_ln_alltime[expid[i]]['am'].values,
    method='nearest'
    )

print((result1[np.isfinite(result1)] == result2[np.isfinite(result2)]).all())


# for irecord in range(len(Antarctic_snow_isotopes['lat'].values)):
#     # irecord = 16
#     print(irecord)
#     if (np.isfinite(Antarctic_snow_isotopes['lat'].values[irecord]) & np.isfinite(Antarctic_snow_isotopes['lon'].values[irecord])):
        
#         find_gridvalue_at_site_interp(
#             Antarctic_snow_isotopes['lat'].values[irecord],
#             Antarctic_snow_isotopes['lon'].values[irecord],
#             d_ln_alltime[expid[i]]['am'].lat.values,
#             d_ln_alltime[expid[i]]['am'].lon.values,
#             d_ln_alltime[expid[i]]['am'].values,
#             method='nearest'
#         )

#-------------------------------- check find_gridvalue_at_site_time

find_gridvalue_at_site_time(
    NK16_Australia_Syowa['1d']['time'][0],
    NK16_Australia_Syowa['1d']['lat'][0],
    NK16_Australia_Syowa['1d']['lon'][0],
    d_ln_q_sfc_alltime[expid[i]]['daily'].time.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lat.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lon.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].values
)

stime = NK16_Australia_Syowa['1d']['time'][0]
slat = NK16_Australia_Syowa['1d']['lat'][0]
slon = NK16_Australia_Syowa['1d']['lon'][0]

itime = np.argmin(abs(stime.asm8 - d_ln_q_sfc_alltime[expid[i]]['daily'].time).values)
ilat, ilon = find_ilat_ilon(
    slat, slon,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lat.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lon.values)

d_ln_q_sfc_alltime[expid[i]]['daily'][itime, ilat, ilon]


#-------------------------------- check find_multi_gridvalue_at_site_time

ires = 100
res1 = find_multi_gridvalue_at_site_time(
    NK16_Australia_Syowa['1d']['time'],
    NK16_Australia_Syowa['1d']['lat'],
    NK16_Australia_Syowa['1d']['lon'],
    d_ln_q_sfc_alltime[expid[i]]['daily']['time'].values,
    d_ln_q_sfc_alltime[expid[i]]['daily']['lat'].values,
    d_ln_q_sfc_alltime[expid[i]]['daily']['lon'].values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].values,
)[ires]


res2 = find_gridvalue_at_site_time(
    NK16_Australia_Syowa['1d']['time'][ires],
    NK16_Australia_Syowa['1d']['lat'][ires],
    NK16_Australia_Syowa['1d']['lon'][ires],
    d_ln_q_sfc_alltime[expid[i]]['daily']['time'].values,
    d_ln_q_sfc_alltime[expid[i]]['daily']['lat'].values,
    d_ln_q_sfc_alltime[expid[i]]['daily']['lon'].values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].values,
)

stime = NK16_Australia_Syowa['1d']['time'][ires]
slat = NK16_Australia_Syowa['1d']['lat'][ires]
slon = NK16_Australia_Syowa['1d']['lon'][ires]

itime = np.argmin(abs(stime.asm8 - d_ln_q_sfc_alltime[expid[i]]['daily'].time).values)
ilat, ilon = find_ilat_ilon(
    slat, slon,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lat.values,
    d_ln_q_sfc_alltime[expid[i]]['daily'].lon.values)

res3 = d_ln_q_sfc_alltime[expid[i]]['daily'][itime, ilat, ilon].values

print(res1); print(res2); print(res3)



'''
# endregion

