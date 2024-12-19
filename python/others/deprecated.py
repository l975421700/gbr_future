

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




