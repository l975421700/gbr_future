

# region globe_plot


import numpy as np
import cartopy.crs as ccrs
def globe_plot(
    ax_org=None,
    figsize=np.array([8.8, 4.4]) / 2.54,
    projections = ccrs.Mollweide(central_longitude=180),
    add_atlas=True, atlas_color='black', lw=0.1,
    add_grid=True, grid_color='gray',
    add_grid_labels=False, ticklabel = None, labelsize=10,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.01, fm_top=0.99,
    plot_scalebar=False, sb_bars=2, sb_length=200, sb_location=(0.02, 0.015),
    sb_barheight=20, sb_linewidth=0.15, sb_col='black', sb_middle_label=False,
    ):
    '''
    ----Input
    
    ----output
    
    '''
    
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    
    if (ticklabel is None):
        ticklabel = ticks_labels(0, 360, -90, 90, 60, 30)
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': projections},)
    else:
        ax = ax_org
    
    if add_grid_labels:
        ax.set_xticks(ticklabel[0],)
        ax.set_xticklabels(ticklabel[1], fontsize=labelsize)
        ax.set_yticks(ticklabel[2],)
        ax.set_yticklabels(ticklabel[3], fontsize=labelsize)
        ax.tick_params(length=1, width=lw * 2)
    
    if add_atlas:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=atlas_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=atlas_color, facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if add_grid:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), linewidth=lw * 0.75, zorder=2,
            color=grid_color, linestyle='--')
        gl.xlocator = mticker.FixedLocator(ticklabel[0])
        gl.ylocator = mticker.FixedLocator(ticklabel[2])
    
    if plot_scalebar:
        scale_bar(ax, bars=sb_bars,
                  length=sb_length,
                  location=sb_location,
                  barheight=sb_barheight,
                  linewidth=sb_linewidth,
                  col=sb_col,
                  middle_label=sb_middle_label,)
    
    if (ax_org is None):
        fig.subplots_adjust(
            left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top)
    
    if (ax_org is None):
        return fig, ax
    else:
        return ax



'''
#-------------------------------- Test one plot
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=9)
mpl.rcParams['axes.linewidth'] = 0.2
import cartopy.crs as ccrs

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = globe_plot()

# plt_cmp = ax.pcolormesh(
#     x,
#     y,
#     z,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
    pad=0.1, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel('1st line\n2nd line', linespacing=2)
fig.savefig('figures/test/trial.png')


#-------------------------------- Test n*m plot
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=9)
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
mpl.rcParams['axes.linewidth'] = 0.2
from a_basic_analysis.b_module.namelist import (
    panel_labels,
)

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

output_png = 'figures/trial.png'
cbar_label = 'TEXT\nTEXT'

nrow = 2
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        if((irow!=0) | (jcol !=0)):
            axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol], add_grid_labels=False)
            plt.text(
                0, 1.05, panel_labels[ipanel],
                transform=axs[irow, jcol].transAxes,
                ha='left', va='center', rotation='horizontal')
            ipanel += 1
        else:
            axs[irow, jcol].axis('off')

# axs[0, 1].pcolormesh(
#     x,
#     y,
#     z,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
plt.text(
    0.5, 1.05, 'TEXT', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    -0.05, 0.5, 'TEXT', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs, aspect=40,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='max',
    anchor=(0.5, 0.4),
    )
cbar.ax.set_xlabel(cbar_label, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)

# cbar1 = fig.colorbar(
#     cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
#     orientation="horizontal",shrink=0.5,aspect=40,extend='max',
#     anchor=(-0.2, -0.3), ticks=pltticks)
# cbar1.ax.set_xlabel('TEXT\nTEXT', linespacing=2)

# cbar2 = fig.colorbar(
#     cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
#     orientation="horizontal",shrink=0.5,aspect=40,extend='max',
#     anchor=(1.1,-3.8),ticks=pltticks)
# cbar2.ax.set_xlabel('TEXT\nTEXT', linespacing=2)


'''
# endregion


# region regional_plot


def regional_plot(
    extent=None,
    figsize=None,
    central_longitude = 0,
    xmajortick_int = 10, ymajortick_int = 10,
    xminortick_int = 5, yminortick_int = 5,
    lw=0.25, country_boundaries=True, border_color = 'black',
    grid_color = 'gray',
    set_figure_margin = False, figure_margin=None,
    ticks_and_labels = False,
    ax_org=None, fontsize=10,
    ):
    '''
    ----Input
    ----output
    '''
    
    import numpy as np
    import cartopy.feature as cfeature
    import cartopy as ctp
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', family='Times New Roman', size=fontsize)
    
    ticklabel=ticks_labels(extent[0], extent[1], extent[2], extent[3],
                           xmajortick_int, ymajortick_int)
    xminorticks = np.arange(extent[0], extent[1] + 1e-4, xminortick_int)
    yminorticks = np.arange(extent[2], extent[3] + 1e-4, yminortick_int)
    transform = ctp.crs.PlateCarree(central_longitude=central_longitude)
    
    if (figsize is None):
        figsize = np.array([8.8, 8.8]) / 2.54
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': transform},)
    else:
        ax = ax_org
    
    ax.set_extent(extent, crs = ctp.crs.PlateCarree())
    
    if ticks_and_labels:
        ax.set_xticks(ticklabel[0], crs = ctp.crs.PlateCarree())
        ax.set_xticklabels(ticklabel[1])
        ax.set_yticks(ticklabel[2])
        ax.set_yticklabels(ticklabel[3])
        ax.tick_params(length=2)
    
    if country_boundaries:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if (central_longitude == 0):
        ax.gridlines(
            crs=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidth=lw, zorder=2,
            color=grid_color, alpha=0.5, linestyle='--',
            xlocs = xminorticks, ylocs=yminorticks,
            )
    else:
        ax.gridlines(
            crs=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidth=lw, zorder=2,
            color=grid_color, alpha=0.5, linestyle='--',
            )
    
    if set_figure_margin & (not(figure_margin is None)) & (ax_org is None):
        fig.subplots_adjust(
            left=figure_margin['left'], right=figure_margin['right'],
            bottom=figure_margin['bottom'], top=figure_margin['top'])
    elif (ax_org is None):
        fig.tight_layout()
    
    if (ax_org is None):
        return fig, ax
    else:
        return ax


'''
#---------------- check

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
import numpy as np

# plot southern atlantic ocean, 90deg
extent=[-70, 20, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([8.8, 5.2]) / 2.54)
fig.savefig('figures/test/trial1.png')


# plot southern indian ocean, 120deg
extent=[20, 140, -90, -38]
fig, ax = regional_plot(extent=extent, figsize = np.array([11.4, 5.2]) / 2.54)
fig.savefig('figures/test/trial2.png')


# plot southern pacific ocean, 150deg
extent=[140, 290, -90, -38]
fig, ax = regional_plot(
    extent=extent, figsize = np.array([13, 5.2]) / 2.54,
    central_longitude = 180)
fig.savefig('figures/test/trial3.png')


#---- debugs

extent=[140, 290, -90, -38]
figsize = np.array([8.8*11/9, 5.2]) / 2.54
central_longitude = 180
xmajortick_int = 20
ymajortick_int = 10
xminortick_int = 10
yminortick_int = 10
lw=0.25
country_boundaries=True
border_color = 'black'
gridlines=True
grid_color = 'gray'
set_figure_margin = False
figure_margin=None
ticks_and_labels = True
ax_org=None


'''
# endregion


# region ticks_labels

def ticks_labels(
    xmin, xmax, ymin, ymax, xspacing, yspacing
    ):
    '''
    # input ----
    xmin, xmax, ymin, ymax: range of labels
    xspacing: spacing of x ticks
    yspacing: spacing of y ticks
    
    # output ----
    xticks_pos, xticks_label, yticks_pos, yticks_label
    '''
    
    import numpy as np
    
    # get the x ticks
    xticks_pos = np.arange(xmin, xmax + 1e-4, xspacing)
    if not isinstance(xspacing, int):
        xticks_pos = np.around(xticks_pos, 1)
    else:
        xticks_pos = xticks_pos.astype('int')
    
    # Associate with '° W', '°', and '° E'
    xticks_label = [''] * len(xticks_pos)
    for i in np.arange(len(xticks_pos)):
        if (xticks_pos[i] > 180):
            xticks_pos[i] = xticks_pos[i] - 360
        
        if (abs(xticks_pos[i]) == 180) | (xticks_pos[i] == 0):
            xticks_label[i] = str(abs(xticks_pos[i])) + '°'
        elif xticks_pos[i] < 0:
            xticks_label[i] = str(abs(xticks_pos[i])) + '° W'
        elif xticks_pos[i] > 0:
            xticks_label[i] = str(xticks_pos[i]) + '° E'
    
    # get the y ticks
    yticks_pos = np.arange(ymin, ymax + 1e-4, yspacing)
    if not isinstance(yspacing, int):
        yticks_pos = np.around(yticks_pos, 1)
    else:
        yticks_pos = yticks_pos.astype('int')
    
    # Associate with '° N', '°', and '° S'
    yticks_label = [''] * len(yticks_pos)
    for i in np.arange(len(yticks_pos)):
        if yticks_pos[i] < 0:
            yticks_label[i] = str(abs(yticks_pos[i])) + '° S'
        if yticks_pos[i] == 0:
            yticks_label[i] = str(yticks_pos[i]) + '°'
        if yticks_pos[i] > 0:
            yticks_label[i] = str(yticks_pos[i]) + '° N'
    
    return xticks_pos, xticks_label, yticks_pos, yticks_label

'''
xmin = -180; xmax = 180; ymin = -90; ymax = 90; xspacing = 60; yspacing = 30
xticks_pos = np.arange(xmin, xmax + xspacing, xspacing)
ddd = ticks_labels(xmin, xmax, ymin, ymax, xspacing, yspacing)
'''
# endregion


# region scale_bar

def scale_bar(
    ax, bars = 2, length = None, location = (0.1, 0.05),
    barheight = 5, linewidth = 3, col = 'black',
    middle_label=True, fontcolor='black', vline = 1400):
    '''
    ax: the axes to draw the scalebar on.
    bars: the number of subdivisions
    length: in [km].
    location: left side of the scalebar in axis coordinates.
    (ie. 0 is the left side of the plot)
    barheight: in [km]
    linewidth: the thickness of the scalebar.
    col: the color of the scale bar
    middle_label: whether to plot the middle label
    '''
    
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle
    
    # Get the limits of the axis in lat lon
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    
    # Make tmc aligned to the left of the map, vertically at scale bar location
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly, approx = False)
    
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]
    
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000 / bars]
    # Plot the scalebar chunks
    barcol = 'white'
    
    for i in range(0, bars):
        # Generate the x coordinate for the left of ith bar
        barleft_x = sbx + i * length * 1000 / bars
        
        # plot the chunk
        ax.add_patch(
            Rectangle((barleft_x, sby),
                      length * 1000 / bars, barheight * 1000, ec = 'black',
                      color = barcol, lw = linewidth, transform = tmc))
        
        # ax.plot(bar_xs, [sby, sby], transform=tmc, color=barcol, linewidth=linewidth)
        
        # alternate the colour
        if barcol == 'white':
            barcol = col
        else:
            barcol = 'white'
        # Plot the scalebar label for that chunk
        if i == 0 or middle_label:
            ax.text(barleft_x, sby + barheight * 1200,
                    str(round(i * length / bars)), transform=tmc,
                    horizontalalignment='center', verticalalignment='bottom',
                    color=fontcolor)
    
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000 * 1.1
    # Plot the last scalebar label
    ax.text(bar_xt, sby + barheight * vline, str(round(length)) + ' km',
            transform=tmc, horizontalalignment='center',
            verticalalignment='bottom', color=fontcolor)

'''
# https://github.com/SciTools/cartopy/issues/490
# bars = 2; length = 1000; location=(0.1, 0.05); linewidth=3; col='black'

bars=scalebar_elements['bars'],
length=scalebar_elements['length'],
location=scalebar_elements['location'],
barheight=scalebar_elements['barheight'],
linewidth=scalebar_elements['linewidth'],
col=scalebar_elements['col'],
middle_label=scalebar_elements['middle_label']
'''
# endregion


# region plot_maxmin_points


def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    '''
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    '''
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    pretext = []
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            pretext_i = ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center',
                transform=transform)
            pretext.append(pretext_i)
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top',
                transform=transform)
    
    return(pretext)


# endregion


# region remove_trailing_zero

def remove_trailing_zero(x):
    return ('%f' % x).rstrip('0').rstrip('.')

def remove_trailing_zero_abs(x):
    return ('%f' % abs(x)).rstrip('0').rstrip('.')

from matplotlib.ticker import FuncFormatter
@FuncFormatter
def remove_trailing_zero_pos(x, pos):
    return ('%f' % x).rstrip('0').rstrip('.')

@FuncFormatter
def remove_trailing_zero_pos_abs(x, pos):
    return ('%f' % abs(x)).rstrip('0').rstrip('.')

'''
ax.clabel(fmt=remove_trailing_zero)
fig.colorbar(format=remove_trailing_zero_pos)
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)

https://stackoverflow.com/questions/2440692/formatting-floats-without-trailing-zeros

https://stackoverflow.com/questions/66817786/how-can-i-use-the-formatters-to-make-custom-ticks-in-matplotlib
'''
# endregion


