

# region get CL_Frequency

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

# loop through each day
daterange = pd.date_range(start='1/1/2016', end='12/31/2016')

for idate in daterange:
    # idate = daterange[0]
    print(idate)
    
    year=str(idate)[:4]
    month=str(idate)[5:7]
    day=str(idate)[8:10]
    
    clp_fl = sorted(glob.glob(f'scratch/data/obs/jaxa/clp/{year}{month}/{day}/*/CLP_{year}{month}{day}????.nc'))
    
    clp_ds = xr.open_mfdataset(clp_fl)
    
    CLTYPE_values = clp_ds.CLTYPE.values
    
    CL_Frequency = xr.DataArray(
        name='CL_Frequency',
        data=np.zeros((1, 12, clp_ds.CLTYPE.shape[1], clp_ds.CLTYPE.shape[2])),
        dims=['time', 'types', 'latitude', 'longitude',],
        coords={
            'time': [datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d')],
            'types': ['finite'] + list(ISCCP_types.keys()),
            'latitude': clp_ds.CLTYPE.latitude.values,
            'longitude': clp_ds.CLTYPE.longitude.values,
        }
    )
    
    CL_Frequency.loc[{'types': 'finite'}][0] = np.isfinite(CLTYPE_values).sum(axis=0)
    
    for itype in list(ISCCP_types.keys()):
        # print(itype)
        CL_Frequency.loc[{'types': itype}][0] = (CLTYPE_values == ISCCP_types[itype]).sum(axis=0)
    
    print((CL_Frequency[0, 0] == CL_Frequency[0, 1:].sum(axis=0)).all().values)
    print(CL_Frequency[0, 0].sum().values)
    
    ofile = f'scratch/data/obs/jaxa/clp/{year}{month}/{day}/CL_Frequency_{year}{month}{day}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    CL_Frequency.to_netcdf(ofile)
    
    print("Current time:", datetime.now())


aaa = xr.open_dataset('scratch/data/obs/jaxa/clp/201601/01/CL_Frequency_20160101.nc')


'''
himawari_fl = sorted(glob.glob('data/obs/jaxa/clp/*/*/*/NC_*'))
clp_fl = sorted(glob.glob('scratch/data/obs/jaxa/clp/*/*/*/CLP_*'))
print(len(himawari_fl))
print(len(clp_fl))
'''
# endregion


# region animate sounding profiles (theta, RH)

start_date = datetime(2021, 1, 1, 0)
end_date = datetime(2021, 12, 31, 23)
station = '94299'
output_mp4 = 'figures/test1.mp4'

fig, axs = plt.subplots(1, 2, sharey=True, figsize=np.array([8.8, 6.4]) / 2.54)

ims = []
for date in pd.date_range(start_date, end_date, freq='12h'):
    try:
        df = WyomingUpperAir.request_data(date, station)
        df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='any').reset_index(drop=True)
        
        p = df['pressure'].values * units.hPa
        T = (df['temperature'].values * units.degC).to(units('K'))
        Td = (df['dewpoint'].values * units.degC).to(units('K'))
        height = df['height'].values * units.m
        
        thta = mpcalc.potential_temperature(p, T)
        RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
        
        plt1 = axs[0].plot(thta, p, c='tab:blue')
        plt2 = axs[1].plot(RH, p, c='tab:blue')
        plt3 = plt.text(0.5, 0.95, str(date)[:13] + ' UTC', ha='center', fontsize=10, transform=fig.transFigure)
        
        ims.append(plt1+plt2 + [plt3])
        print(str(date)[:13])
    except:
        print('No data for ' + str(date)[:13])

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(290, 330)
axs[1].set_xlim(0, 100)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'$\theta$ [$K$]')
axs[1].set_xlabel(r'RH [$\%$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[1].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

fig.subplots_adjust(0.18, 0.18, 0.85, 0.88)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)

# endregion


# region get and plot Wyoming sounding (theta and RH)

date = datetime(2021, 12, 8, 0)
station = '94299'
output_png='figures/test.png'

df = WyomingUpperAir.request_data(date, station)
plot_wyoming_sounding_vertical(df, date, output_png=output_png)



'''
print(df.columns)
'''
# endregion


# region get and plot Wyoming sounding (T, theta, RH)

date = datetime(2021, 9, 16, 0)
station = '94299'
output_png='figures/test.png'

df = WyomingUpperAir.request_data(date, station)
df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all').reset_index(drop=True)

p = df['pressure'].values * units.hPa
T = (df['temperature'].values * units.degC).to(units('K'))
Td = (df['dewpoint'].values * units.degC).to(units('K'))
height = df['height'].values * units.m

thta = mpcalc.potential_temperature(p, T)
RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')

fig, axs = plt.subplots(1, 3, sharey=True, figsize=np.array([8.8, 6.4]) / 2.54)

axs[0].plot(T, p)
axs[1].plot(thta, p)
axs[2].plot(RH, p)

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(270, 310)
axs[1].set_xlim(290, 330)
axs[2].set_xlim(0, 100)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'T [$K$]')
axs[1].set_xlabel(r'$\theta$ [$K$]')
axs[2].set_xlabel(r'RH [$\%$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')
axs[2].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[2].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

plt.suptitle(str(date)[:13] + ' UTC', fontsize=10)
plt.subplots_adjust(0.18, 0.18, 0.85, 0.88)
plt.savefig(output_png)


# endregion




