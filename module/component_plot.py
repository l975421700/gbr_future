

# region rainbow_text: partial colouring of text

def rainbow_text(x,y,ls,lc,ax,**kw):
    '''
    x,y:    locations
    ls:     list of strings
    lc:     list of colors
    '''
    
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    
    # t = plt.gca().transData
    t = ax.transAxes
    fig = plt.gcf()
    # plt.show()

    #horizontal version
    for s,c in zip(ls,lc):
        text = plt.text(x,y,s,color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


'''
https://github.com/matplotlib/matplotlib/issues/697
'''
# endregion


# region change_snsbar_width: change bar width of sns.barplot

def change_snsbar_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

'''
https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
'''
# endregion


# region cplot_wind_vectors: plot wind vectors

import cartopy.crs as ccrs
def cplot_wind_vectors(
    x, y, u, v, ax,
    iarrow=3, color='blue', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),
    ):
    '''
    #-------- Input
    x, y: location, 1D
    u, v: wind conponents, 2D
    ax
    
    iarrow: intervals. plot every iarrow
    
    #-------- Output
    
    
    '''
    
    plt_quiver = ax.quiver(
        x[::iarrow], y[::iarrow],
        u[::iarrow, ::iarrow], v[::iarrow, ::iarrow],
        color=color, units=units, scale=scale,
        width=width, headwidth=headwidth, headlength=headlength, alpha=alpha,
        transform=transform,)
    
    return(plt_quiver)


# endregion


# region cplot_lon180: plot lon-plev mesh, from [0, 360] to [-180, 180]

def cplot_lon180(
    lon,
    y,
    ax, pltnorm, pltcmp,
    plt_data,
    ):
    '''
    #-------- Input
    lon: 1d, from 0-360
    y: 1d, y axis
    plt_data: 2d, with lon from 0-360
    pltnorm, pltcmp
    
    #-------- Output
    
    '''
    
    import numpy as np
    import xarray as xr
    
    lon_180 = np.concatenate([lon[int(len(lon) / 2):] - 360,
                              lon[:int(len(lon) / 2)], ])
    
    plt_data_180 = xr.concat([
        plt_data.sel(lon=slice(180, 360)),
        plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')
    
    plt_mesh = ax.pcolormesh(
        lon_180, y, plt_data_180, norm=pltnorm, cmap=pltcmp,)
    
    return(plt_mesh)

# endregion


# region cplot_lon180_ctr: plot lon-plev contour, from [0, 360] to [-180, 180]

import numpy as np
from mapplot import (
    remove_trailing_zero,
)

def cplot_lon180_ctr(
    lon, y, ax,
    plt_data,
    q_intervals = np.array([0.1, 0.5, 1, 2, 4, 6, 8, 10]),
    colors='b', linewidths=0.3, clip_on=True,
    inline_spacing=10, fontsize=6
    ):
    '''
    #-------- Input
    
    #-------- Output
    
    '''
    
    import xarray as xr
    
    lon_180 = np.concatenate([
        lon[int(len(lon) / 2):] - 360, lon[:int(len(lon) / 2)], ])
    
    plt_data_180 = xr.concat([
        plt_data.sel(lon=slice(180, 360)),
        plt_data.sel(lon=slice(0, 180 - 1e-4))], dim='lon')
    
    plt_ctr = ax.contour(
        lon_180, y, plt_data_180,
        colors=colors, levels=q_intervals, linewidths=linewidths,
        clip_on=clip_on)
    
    ax_clabel = ax.clabel(
        plt_ctr, inline=1, colors=colors, fmt=remove_trailing_zero,
        levels=q_intervals, inline_spacing=inline_spacing, fontsize=fontsize,)
    
    return([plt_ctr, ax_clabel])


# endregion


# region plt_mesh_pars: derive parameters for mesh plot


def plt_mesh_pars(
    cm_min, cm_max, cm_interval1, cm_interval2, cmap,
    clip=True, reversed=True, asymmetric=False,
    ):
    '''
    #-------- Input
    
    #-------- Output
    
    '''
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    
    pltlevel = np.arange(cm_min, cm_max + 1e-4, cm_interval1, dtype=np.float64)
    pltticks = np.arange(cm_min, cm_max + 1e-4, cm_interval2, dtype=np.float64)
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=clip)
    pltcmp = cm.get_cmap(cmap, len(pltlevel)-1)
    
    if(reversed):
        pltcmp = pltcmp.reversed()
    
    if (asymmetric):
        cm_range = np.max((abs(cm_min), abs(cm_max))) * 2
        pltcmp = cm.get_cmap(cmap, int(cm_range / cm_interval1))
        
        if(reversed):
            pltcmp = pltcmp.reversed()
        
        if (abs(cm_min) > abs(cm_max)):
            pltcmp = ListedColormap(
                [pltcmp(i) for i in range(pltcmp.N)][:(len(pltlevel)-1)])
        else:
            pltcmp = ListedColormap(
                [pltcmp(i) for i in range(pltcmp.N)][-(len(pltlevel)-1):])
    
    return([pltlevel, pltticks, pltnorm, pltcmp])

'''
# pltlevel = np.arange(-55, -20 + 1e-4, 2.5)
# pltticks = np.arange(-55, -20 + 1e-4, 5)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-55, cm_max=-20, cm_interval1=2.5, cm_interval2=5, cmap='PuOr',
)


#-------- derive diverging colormap
from matplotlib import colors

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.4, cm_max=0.1, cm_interval1=0.05, cm_interval2=0.1,
    cmap='viridis',)
pltcmp = cm.get_cmap(
    'PuOr', len(np.arange(-0.4,0.4+1e-4, 0.05))-1).reversed()
pltcmp = ListedColormap(
    [pltcmp(i) for i in range(pltcmp.N)][:(len(pltlevel)-1)])

pltlevel = np.arange(-0.4, 0.1 + 1e-4, 0.05, dtype=np.float64)
pltticks = np.arange(-0.4, 0.1 + 1e-4, 0.1, dtype=np.float64)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltnorm = colors.TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.1)
# pltcmp = 'PuOr'
pltcmp = cm.get_cmap(
    'PuOr',
    len(np.arange(-0.4, 0.4 + 1e-4, 0.05))-1).reversed()
pltcmp = colors.ListedColormap(
    [pltcmp(i) for i in range(pltcmp.N)][:(len(pltlevel)-1)]
)

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-0.4, cm_max=0.4, cm_interval1=0.05, cm_interval2=0.1,
    cmap='PuOr',)
# pltticks[-2] = 0
pltlevel = pltlevel[:11]
pltticks = pltticks[:6]
pltcmp = colors.ListedColormap(
    [pltcmp(i) for i in range(pltcmp.N)][:(len(pltlevel)-1)]
)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)

'''
# endregion



