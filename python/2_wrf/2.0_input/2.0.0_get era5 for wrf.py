

# region import packages

import numpy as np
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

from wrf import get_era5_for_wrf

# endregion


# region download data

# setting
year_str    = '2016'
month_strs  = np.arange(4, 5, 1).astype(str).tolist()
day_strs    = np.arange(27, 29, 1).astype(str).tolist()
nlat        = -10
slat        = -25
wlon        = 140
elon        = 155
area        = f'{nlat}/{wlon}/{slat}/{elon}'

ofile1 = f'data/sim/wrf/input/era5/era5_sl_{year_str}{month_strs[0]}{day_strs[0]}_{year_str}{month_strs[-1]}{day_strs[-1]}_{nlat}.{slat}.{wlon}.{elon}.grib'
ofile2 = f'data/sim/wrf/input/era5/era5_pl_{year_str}{month_strs[0]}{day_strs[0]}_{year_str}{month_strs[-1]}{day_strs[-1]}_{nlat}.{slat}.{wlon}.{elon}.grib'

get_era5_for_wrf(
    year_str, month_strs, day_strs, ofile1, area=area, surface_only=True,)

get_era5_for_wrf(
    year_str, month_strs, day_strs, ofile2, area=area, surface_only=False,)




'''
output_filename = 'data/sim/wrf/input/era5/era5_sl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_strs, day_strs,
    output_filename, area=area,
    surface_only=True,
    )

output_filename = 'data/sim/wrf/input/era5/era5_pl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_strs, day_strs,
    output_filename, area=area,
    surface_only=False,
    )
area            = '10/122/-45/175'
'''
# endregion




