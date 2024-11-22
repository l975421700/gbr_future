

# region import packages

import numpy as np

import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

from wrf import (
    get_era5_for_wrf,
    )

# endregion


# region 2022 02 6-9

year_str        = '2022'
month_str_list  = np.arange(2, 3, 1).astype(str).tolist()
day_str_list    = np.arange(6, 10, 1).astype(str).tolist()
area            = '10/122/-45/175'

output_filename = 'data/sim/wrf/input/era5/era5_sl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area=area,
    surface_only=True,
    )

output_filename = 'data/sim/wrf/input/era5/era5_pl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area=area,
    surface_only=False,
    )


'''

'''
# endregion



