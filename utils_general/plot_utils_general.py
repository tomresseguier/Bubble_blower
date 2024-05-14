import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import *
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.table import Table
from tqdm import tqdm




def plot_image_RaDec(image_path, origin="lower", index=0, fig=None, pos=111, make_axes_labels=True, RGB=True, make_grid=True) :
    hdu_list = fits.open(image_path)
    if RGB :
        image_data = np.array([hdu_list[1].data, hdu_list[2].data, hdu_list[3].data])
        image_data = np.moveaxis(image_data, 0, 2)
        hdu = hdu_list[index]
    else :
        hdu = hdu_list[index]
        image_data = hdu.data
        zscale = ZScaleInterval()
        norm = ImageNormalize(image_data, interval=zscale)
    
    wcs = WCS(hdu.header)
    
    if fig is None :
        fig = plt.figure()
    
    ax = fig.add_subplot(pos, projection=wcs)
    
    if make_axes_labels :
        ax.coords[0].set_axislabel('RA')
        ax.coords[1].set_axislabel('DEC')
    else :
        ax.coords[0].set_axislabel(' ')
        ax.coords[1].set_axislabel(' ')
    
    if RGB :
        if make_grid :
            ax.coords.grid(True, color='white', ls='dashed', lw=0.3)
        ax.imshow(image_data, origin=origin)
    else :
        if make_grid :
            ax.coords.grid(True, color='black', ls='dotted')
        ax.imshow(image_data, cmap='gray', norm=norm, origin=origin)
    #ax.figure.tight_layout()
    return fig, ax


def plot_ellipses( cat, ax=None, image_path=None, color='blue', alpha=0.5, scale=1., facecolor=None, linewidth=1., make_grid=True ):
    if ax is None :
        fig, ax = plot_image_RaDec(image_path, make_grid=make_grid)
        to_return = fig, ax
    else :
        to_return = None
    
    for i in tqdm(range(len(cat))):
        if facecolor is None :
            ellipse = Ellipse( (cat['x'][i], cat['y'][i]), cat['a'][i], cat['b'][i], angle=cat['theta'][i], \
                               alpha=alpha, color=color, linewidth=linewidth )
        else :
            edgecolor = facecolor.copy()
            edgecolor[-1] = 1
            ellipse = Ellipse( (cat['x'][i], cat['y'][i]), cat['a'][i], cat['b'][i], angle=cat['theta'][i], \
                               facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth )
        ax.add_artist(ellipse)
    plt.show()
    return to_return







