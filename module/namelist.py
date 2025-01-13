

import numpy as np
import cartopy as ctp


# region names

month = np.array(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
monthini = np.array(
    ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
month_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
month_dec = np.array(
    ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
     'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',])
month_dec_num = np.array([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
seasons_last_num = np.array([2, 5, 8, 11])

hours = ['00', '01', '02', '03', '04', '05',
         '06', '07', '08', '09', '10', '11',
         '12', '13', '14', '15', '16', '17',
         '18', '19', '20', '21', '22', '23',
         ]

months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

seconds_per_d = 86400

nci_ds = {
    'era5': '/g/data/rt52/era5/'
}

# endregion


# region physics

zerok = 273.15

# endregion


# region plot

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)',
                '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)',
                '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)',
                '(v)', '(w)', '(x)', '(y)', '(z)',
                ]


# endregion



