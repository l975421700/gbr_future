

# region import packages

# data analysis
import numpy as np
from metpy.calc import pressure_to_height_std
from metpy.units import units
import metpy.calc as mpcalc
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from siphon.simplewebservice.igra2 import IGRAUpperAir
import pandas as pd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
mpl.rcParams['lines.linewidth'] = 0.5
plt.rcParams.update({"mathtext.fontset": "stix"})

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')


from metplot import (
    plot_wyoming_sounding,
    output_wyoming_sounding,
    plot_wyoming_sounding_vertical,
    )

# endregion


# region plot Wyoming sounding

year, month, day, hour = 2020, 6, 2, 11
date = datetime(year, month, day, hour)
station = 'Willis Island'
stationNumber = '94299'
slat = -16.288
slon = 149.965

rename = {'pressure_hPa': 'pressure',
          'geopotential height_m': 'height',
          'temperature_C': 'temperature',
          'dew point temperature_C': 'dewpoint',
          'wind direction_degree': 'direction',
          'wind speed_m/s': 'speed'}

df = pd.read_csv(f'data/obs/radiosonde/Wyoming/{stationNumber}/{year}{month:02d}{day:02d}{hour:02d}-{stationNumber}.csv', skipfooter=1, engine='python').rename(columns=rename)

output_png=f'figures/0_gbr/0.0_sounding/0.0.0 sounding {str(date)[:13]} {stationNumber} WyomingWeb.png'
plot_wyoming_sounding(df, date, output_png=output_png,
                      subset=('temperature', 'dewpoint'), ihodograph=False)



'''
from haversine import haversine
distance_km = haversine((df.latitude.iloc[0], df.longitude.iloc[0]), (df.latitude.iloc[-1], df.longitude.iloc[-1]))
'''
# endregion








# region get and plot Wyoming sounding skew-T

date = datetime(2020, 6, 2, 12)
station = '94299'
output_png=f'figures/0_gbr/0.0_sounding/0.0.0 sounding {str(date)[:13]} {station}.png'

df = WyomingUpperAir.request_data(date, station)
plot_wyoming_sounding(df, date, output_png=output_png,
                      subset=('temperature', 'dewpoint'), ihodograph=False)



'''
df = get_wyoming_sounding(date, station)
print(df.columns)
'''
# endregion
# region get and plot IGRA2 sounding skew-T

date = datetime(2020, 6, 2, 12)
station = 'ASM00094299'
output_png=f'figures/0_gbr/0.0_sounding/0.0.0 sounding {str(date)[:13]} {station} IGRA2.png'

df, _ = IGRAUpperAir.request_data(date, station, derived=True)
plot_wyoming_sounding(df, date, output_png=output_png,
                      subset=('temperature', 'dewpoint'), ihodograph=False)

date = datetime(2020, 6, 2, 12)
station = 'ASM00094299'
df, header = IGRAUpperAir.request_data(date, station)


daterange = [datetime(2020, 6, 2, 0), datetime(2020, 6, 2, 23)]
station = 'ASM00094299'
df, _ = IGRAUpperAir.request_data(daterange, station)


# df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=('temperature', 'dewpoint'), how='all').reset_index(drop=True)
df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]]
df = df.loc[df['pressure']>=200]

# IGRAUpperAir.request_data(date, station)
# IGRAUpperAir.request_data(date, station)
# endregion
# region get and plot Wyoming sounding (T, theta, RH, qv)

date = datetime(2016, 4, 29, 0)
station = '94299'
output_png=f'figures/0_gbr/0.0_sounding/0.0.0 vertical profiles {str(date)[:13]} {station}.png'

df = WyomingUpperAir.request_data(date, station)
df = df[['pressure', 'temperature', 'dewpoint']].dropna(subset=('temperature', 'dewpoint'), how='all').reset_index(drop=True)

p = df['pressure'].values * units.hPa
T = (df['temperature'].values * units.degC).to(units('K'))
Td = (df['dewpoint'].values * units.degC).to(units('K'))

thta = mpcalc.potential_temperature(p, T)
RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
mixr = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH).to('g/kg')

icol=4

fig, axs = plt.subplots(1,icol,sharey=True,figsize=np.array([icol*2+3, 6.4]) / 2.54)

axs[0].plot(T, p, c='tab:blue', marker='o', markersize=1)
axs[1].plot(thta, p, c='tab:blue', marker='o', markersize=1)
axs[2].plot(RH/100, p, c='tab:blue', marker='o', markersize=1)
axs[3].plot(mixr, p, c='tab:blue', marker='o', markersize=1)

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(270, 310)
axs[1].set_xlim(290, 330)
axs[2].set_xlim(0, 1)
axs[3].set_xlim(0, 20)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'T [$K$]')
axs[1].set_xlabel(r'$\theta$ [$K$]')
axs[2].set_xlabel(r'RH [-]')
axs[3].set_xlabel(r'$q_v$ [$g\;kg^{-1}$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')
axs[2].grid(lw=0.2, alpha=0.5, ls='--')
axs[3].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[3].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

plt.suptitle(str(date)[:13] + ' UTC', fontsize=10)
plt.subplots_adjust(1.5/(icol*2+3), 0.2, 1.02-1.5/(icol*2+3), 0.88)
plt.savefig(output_png)


'''
# mpcalc.mixing_ratio_from_relative_humidity(1000 * units.hPa, 300 * units.K, 100 * units('%')).to('g/kg')
'''
# endregion
# region get IGRA2 data

daterange = [datetime(1960, 6, 5, 0), datetime(2025, 1, 1, 0)]
station = 'ASM00094299'
ofile1 = f'data/obs/radiosonde/IGRA2 {station} {str(daterange[0])[:13]} to {str(daterange[1])[:13]} df_drvd.pkl'
ofile2 = f'data/obs/radiosonde/IGRA2 {station} {str(daterange[0])[:13]} to {str(daterange[1])[:13]} header_drvd.pkl'

df_drvd, header_drvd = IGRAUpperAir.request_data(daterange, station)

df_drvd.to_pickle(ofile1)
header_drvd.to_pickle(ofile2)


'''
print(df_drvd.columns)
print(header_drvd.columns)
print(df_drvd.units)
print(header_drvd.units)
'''
# endregion
# region get and output Wyoming sounding

date = datetime(2016, 4, 29, 0)
station = '94299'
outf = f'data/sim/wrf/input/sounding/willis_island_{str(date)[:13]}'

df = WyomingUpperAir.request_data(date, station)
output_wyoming_sounding(df, outf)


'''
'''
# endregion


