

# region get_wyoming_sounding

def get_wyoming_sounding(date, station):
    from siphon.simplewebservice.wyoming import WyomingUpperAir
    df = WyomingUpperAir.request_data(date, station)
    return(df)


# endregion


# region plot_wyoming_sounding

import numpy as np

def plot_wyoming_sounding(
    df, date, output_png=None,
    fig=None, figsize=np.array([8.8, 6.4]) / 2.54,
    ilev=1, plot_barbs=False, ihodograph=True, plot_ccl=False,
    subset=('temperature', 'dewpoint', 'direction', 'speed'),
    xmin=-40, xmax=40, ymin=1030, ymax=200,
    fm_left=0.18, fm_bottom=0.14, fm_right=0.96, fm_top=0.98,
    hodo_ratio='35%', hodo_bbox=(0.05, 0, 1, 1), hodorange=14,
    hodo_interval=2, hodo_label=r'Wind [$m \; s^{-1}$]',
    ):
    
    from metpy.units import units
    import metpy.calc as mpcalc
    from mapplot import remove_trailing_zero_pos_abs
    from metpy.plots import Hodograph, SkewT
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]]
    # df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=subset, how='all').reset_index(drop=True)
    df = df.loc[df['pressure']>=200]
    
    p = df['pressure'].values * units.hPa
    T = df['temperature'].values * units.degC
    Td = df['dewpoint'].values * units.degC
    wind_speed = (df['speed'].values * units.knots).to(units('m/s'))
    wind_dir = df['direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    
    lcl_p, lcl_t = mpcalc.lcl(p[ilev], T[ilev], Td[ilev])
    parcel_prof = mpcalc.parcel_profile(p[ilev:], T[ilev], Td[ilev]).to('degC')
    
    if (fig is None): fig = plt.figure(figsize=figsize)
    
    skew = SkewT(fig, rotation=30)
    
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    skew.plot(p[ilev:], parcel_prof, 'k')
    skew.plot(lcl_p, lcl_t, 'ko', markersize=1.5)
    skew.shade_cin(p[ilev:], T[ilev:], parcel_prof, Td[ilev:])
    skew.shade_cape(p[ilev:], T[ilev:], parcel_prof)
    if plot_barbs: skew.plot_barbs(p, u, v)
    if plot_ccl:
        ccl_p, ccl_t, t_c = mpcalc.ccl(p[ilev:], T[ilev:], Td[ilev:], which='bottom')
        skew.plot(ccl_p, ccl_t, 'bo', markersize=1.5)
    
    if ihodograph:
        ax_hod = inset_axes(
            skew.ax, hodo_ratio, hodo_ratio,
            bbox_to_anchor=hodo_bbox, bbox_transform=skew.ax.transAxes)
        h = Hodograph(ax_hod, component_range=hodorange)
        h.add_grid(increment=hodo_interval, lw=0.2, alpha=0.5)
        h.plot_colormapped(u, v, p, cmap='viridis_r', lw=0.5)
        h.ax.get_yaxis().set_visible(False)
        h.ax.xaxis.set_major_formatter(remove_trailing_zero_pos_abs)
        h.ax.set_xlabel(hodo_label)
    
    skew.plot_dry_adiabats(lw=0.4, alpha=0.5)
    skew.plot_moist_adiabats(lw=0.4, alpha=0.5)
    skew.ax.grid(lw=0.2, alpha=0.5, ls='--')
    skew.ax.set_xlabel(f'Temperature [$°C$] at {str(date)[:13]}:00 UTC')
    skew.ax.set_ylabel('Pressure [$hPa$]')
    skew.ax.set_ylim(ymin, ymax)
    skew.ax.set_xlim(xmin, xmax)
    # skew.ax.text(
    #     0.05, 0.05, str(date)[:13] + ' UTC',
    #     transform=skew.ax.transAxes, ha='left', va='bottom',
    #     bbox=dict(facecolor='white', edgecolor='white'))
    # skew.plot_mixing_lines()
    # skew.ax.axvline(0, color='c', ls='--')
    
    if not output_png is None:
        fig.subplots_adjust(fm_left, fm_bottom, fm_right, fm_top)
        fig.savefig(output_png)
    
    return(fig)


# endregion


# region output_wyoming_sounding

def output_wyoming_sounding(df, outf):
    from metpy.units import units
    import metpy.calc as mpcalc
    
    df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='any').reset_index(drop=True)
    
    p = df['pressure'].values * units.hPa
    T = (df['temperature'].values * units.degC).to(units('K'))
    Td = (df['dewpoint'].values * units.degC).to(units('K'))
    height = df['height'].values * units.m
    wind_speed = (df['speed'].values * units.knots).to(units('m/s'))
    wind_dir = df['direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    
    thta = mpcalc.potential_temperature(p, T)
    RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
    mixr = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH).to('g/kg')
    
    sp = p[0].magnitude
    sthta = thta[0].magnitude
    smixr = mixr[0].magnitude
    nlevels = height.shape[0]
    
    line1 = '{:9.2f} {:9.2f} {:10.2f}'.format(sp, sthta, smixr)
    wrfformat = '{:9.2f} {:9.2f} {:10.2f} {:10.2f} {:10.2f}'
    
    with open(outf, 'w') as f:
        f.write(line1+'\n')
        for ilev in range(nlevels):
            d = wrfformat.format(height[ilev].magnitude, thta[ilev].magnitude, mixr[ilev].magnitude, u[ilev].magnitude, v[ilev].magnitude)
            f.write(d+ '\n')


# endregion


# region plot_wyoming_sounding_vertical

def plot_wyoming_sounding_vertical(
    df, date, axs=None, output_png=None,
    figsize=np.array([8.8, 6.4]) / 2.54,
    xmin=290, xmax=330, ymin=1000, ymax=600,
    ):
    
    from metpy.units import units
    import metpy.calc as mpcalc
    import matplotlib.pyplot as plt
    from metpy.calc import pressure_to_height_std
    
    df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='any').reset_index(drop=True)
    
    p = df['pressure'].values * units.hPa
    T = (df['temperature'].values * units.degC).to(units('K'))
    Td = (df['dewpoint'].values * units.degC).to(units('K'))
    height = df['height'].values * units.m
    
    thta = mpcalc.potential_temperature(p, T)
    RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
    
    if axs is None: fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
    
    axs[0].plot(thta, p)
    axs[1].plot(RH, p)
    
    axs[0].invert_yaxis()
    axs[0].set_ylim(ymin, ymax)
    axs[0].set_xlim(xmin, xmax)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylabel('Pressure [$hPa$]')
    axs[0].set_xlabel(r'$\theta$ [$K$]')
    axs[1].set_xlabel(r'RH [$\%$]')
    axs[0].grid(lw=0.2, alpha=0.5, ls='--')
    axs[1].grid(lw=0.2, alpha=0.5, ls='--')
    
    # 2nd y-axis
    height = np.round(pressure_to_height_std(
        pressure=np.arange(ymin, ymax-1e-4, -100) * units('hPa')), 1,)
    ax2 = axs[1].twinx()
    ax2.invert_yaxis()
    ax2.set_ylim(ymin, ymax)
    ax2.set_yticks(np.arange(ymin, ymax-1e-4, -100))
    ax2.set_yticklabels(height.magnitude, c = 'gray')
    ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')
    
    plt.suptitle(str(date)[:13] + ' UTC', fontsize=10)
    
    if not output_png is None:
        plt.subplots_adjust(0.18, 0.18, 0.85, 0.88)
        plt.savefig(output_png)
    else:
        return(axs)

# endregion


# region get_cross_section

def get_cross_section(da, start, end, steps=None, interp_type='linear'):
    '''
    #---- Input
    da: 3D or 4D xarray.DataArray [(time,) level/pressure, lat, lon]
    start/end: [lat, lon]
    '''
    from haversine import haversine
    from metpy.interpolate import cross_section
    import numpy as np
    import xarray as xr
    
    cross_section_length = haversine(start, end, unit='km')
    if steps is None:
        midx_lat = int(np.ceil(len(da.lat)/2))
        midx_lon = int(np.ceil(len(da.lon)/2))
        mpoint = (da.lat.values[midx_lat], da.lon.values[midx_lon])
        mpoint_lon1 = (da.lat.values[midx_lat], da.lon.values[midx_lon + 1])
        mpoint_lat1 = (da.lat.values[midx_lat+1], da.lon.values[midx_lon])
        
        grid_spacing1 = haversine(mpoint, mpoint_lon1, unit='km')
        grid_spacing2 = haversine(mpoint, mpoint_lat1, unit='km')
        grid_spacing = (grid_spacing1 + grid_spacing2) / 2
        
        steps = int(np.ceil(cross_section_length / grid_spacing))
    
    da_parse = da.to_dataset().metpy.parse_cf()[da.name]
    da_cs = cross_section(da_parse, start, end, steps, interp_type)
    
    vertical_name = [var for var in list(da_cs.coords) if var not in ['metpy_crs', 'lon', 'lat', 'index', 'crs', 'time']][0]
    da_cs = da_cs.rename({vertical_name: 'y', 'index': 'x'})
    
    
    da_cs['x'] = np.linspace(0, cross_section_length, steps)
    da_cs['x'] = da_cs['x'].assign_attrs(units='km')
    da_cs['y'] = da_cs['y'].assign_attrs(units='hPa')
    
    return(da_cs)




'''
da = era5_pl_mon_alltime['am']
# da = barra_c2_pl_mon_alltime['am']
# da = era5_pl_mon_alltime['ann']
# da = barra_c2_pl_mon_alltime['ann']
start = start_position[::-1]
end = end_position[::-1]
interp_type = 'linear'

haversine((da_cs.lat.values[0], da_cs.lon.values[0]), (da_cs.lat.values[-1], da_cs.lon.values[-1]))



    haversine((da.lat.values[int(np.ceil(len(da.lat)/2))],
               da.lon.values[int(np.ceil(len(da.lon)/2))]),
              (da.lat.values[int(np.ceil(len(da.lat)/2))],
               da.lon.values[int(np.ceil(len(da.lon)/2)) + 1]), unit='km')
    
    haversine((da.lat.values[int(np.ceil(len(da.lat)/2))],
               da.lon.values[int(np.ceil(len(da.lon)/2))]),
              (da.lat.values[int(np.ceil(len(da.lat)/2))+1],
               da.lon.values[int(np.ceil(len(da.lon)/2))]), unit='km')
'''
# endregion
