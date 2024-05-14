import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize

import sys
import os
module_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(module_dir)
sys.path.append(main_dir)
sys.path.append(module_dir)
from utils_general.plot_utils_general import *
from utils_general.fits_image import fits_image



############### Initial manipulations to make images compatible with trilogy ###############
image_dir = '/Users/Margaux/Desktop/CLASSES/ISM/Final_project/ngc7496/'
image_path = image_dir + 'ngc7496_f1130w.fits'

"""
hdulist = fits.open(image_path)

idx = 1
wcs = WCS(hdulist[idx].header)
header = hdulist[idx].header
image_data = hdulist[idx].data
zscale = ZScaleInterval()
norm = ImageNormalize(image_data, interval=zscale)
fig, ax = plt.subplots()
ax.imshow(image_data, cmap='gray', norm=norm, origin='lower', extent=[0, image_data.shape[1], 0, image_data.shape[0]])




hdu = fits.PrimaryHDU(image_data)
wcs_header = wcs.to_header()
for key in wcs_header :
    hdu.header[key] = wcs_header[key]

hdu.writeto(image_dir + 'ngc7496_f1130w_comp.fits')
"""


############### Selecting the bubbles ###############
image_dir = '/Users/Margaux/Desktop/CLASSES/ISM/Final_project/ngc7496/'
ngc7496_RGB_path = image_dir + "ngc7496_trilogy_1.fits"

ngc7496 = fits_image(ngc7496_RGB_path)
ngc7496.plot_image()
ngc7496.make_hand_made_catalogue()
ngc7496.hand_made_catalogue.make_image_ROI()


ngc7496.hand_made_catalogue.cat.write('./bubbles_3.0.fits', format='fits')


D = 18.7E6 #Mpc
kpc_to_pix = 1E3/D * 180/np.pi / ngc7496.pix_deg_scale
pc_to_pix = kpc_to_pix/1E3



DATA_dir = "./ngc7496/"
image_path = DATA_dir + "ngc7496_trilogy_1.fits"


fig, ax = plot_ellipses(ngc7496.hand_made_catalogue.cat, image_path=ngc7496_RGB_path, facecolor=[0,1,1,0], linewidth=0.4)
fig, ax = plot_ellipses(ngc7496.hand_made_catalogue.cat, image_path=ngc7496_RGB_path, facecolor=[0,0,0,0])
fig, ax = plot_ellipses(ngc7496.hand_made_catalogue.cat, image_path=ngc7496_RGB_path, facecolor=[1,1,1,0])

fig, ax = plot_ellipses(ngc7496.hand_made_catalogue.cat[100:100], image_path=ngc7496_RGB_path, facecolor=[0,1,1,0], linewidth=0.4, make_grid=False)
fig, ax = plot_ellipses(ngc7496.hand_made_catalogue.cat, image_path=ngc7496_RGB_path, facecolor=[0,1,1,0], linewidth=0.4, make_grid=False)
bar_height = 4
scale_bar = matplotlib.patches.Rectangle((690, 1204), kpc_to_pix, bar_height,
                              edgecolor='white', facecolor='white')
ax.add_patch(scale_bar)
ax.text(690 + kpc_to_pix / 2, 1204 + bar_height + 5, f'1 kpc',
        color='white', fontsize=12, ha='center', va='bottom')
scale_bar = matplotlib.patches.Rectangle((353, 575), kpc_to_pix, bar_height,
                              edgecolor='white', facecolor='white')
ax.add_patch(scale_bar)
ax.text(353 + kpc_to_pix / 2, 575 + bar_height + 5, f'1 kpc',
        color='white', fontsize=12, ha='center', va='bottom')
scale_bar = matplotlib.patches.Rectangle((846, 914), kpc_to_pix, bar_height,
                              edgecolor='white', facecolor='white')
ax.add_patch(scale_bar)
ax.text(846 + kpc_to_pix / 2, 914 + bar_height + 5, f'1 kpc',
        color='white', fontsize=12, ha='center', va='bottom')
ax.annotate('', xy=(824, 1360), xytext=(824+40, 1360+40),
            arrowprops=dict(facecolor='white', edgecolor='white', width=2, headwidth=10),
            color='white', fontsize=12, ha='center')
ax.annotate('', xy=(408, 650), xytext=(408-25, 650-25),
            arrowprops=dict(facecolor='white', edgecolor='white', width=2, headwidth=10),
            color='white', fontsize=12, ha='center')
ax.annotate('', xy=(453, 656), xytext=(453+25, 656-25),
            arrowprops=dict(facecolor='white', edgecolor='white', width=2, headwidth=10),
            color='white', fontsize=12, ha='center')
ax.annotate('', xy=(488, 702), xytext=(488+25, 702-25),
            arrowprops=dict(facecolor='white', edgecolor='white', width=2, headwidth=10),
            color='white', fontsize=12, ha='center')
scale_bar = matplotlib.patches.Rectangle((291, 184), kpc_to_pix, bar_height,
                              edgecolor='white', facecolor='white')
ax.add_patch(scale_bar)
ax.text(291 + kpc_to_pix / 2, 184 + bar_height + 5, f'1 kpc',
        color='white', fontsize=12, ha='center', va='bottom')



"""
cst1 = 250.
cst2 = 1
theta_array = np.linspace(0., np.pi/2, 100)
offset = 50*np.pi/180
x = center[0] + cst1*np.exp(theta_array*cst2)*np.cos(theta_array + offset)
y = center[1] + cst1*np.exp(theta_array*cst2)*np.sin(theta_array + offset)
ax.plot(x, y)
"""
"""
cst1 = 500.
theta_array = np.linspace(0., np.pi/2, 100)
offset = 25*np.pi/180
x = cst1*theta_array*np.cos(theta_array + offset)
y = cst1*theta_array*np.sin(theta_array + offset)
x = center[0] + np.concatenate([x, -x])
y = center[1] + np.concatenate([y, -y])
ax.plot(x, y)
"""


"""
bar_length = 250
bar_angle = 45*np.pi/180
bar_x = np.linspace(-bar_length, bar_length, 100) * np.cos(bar_angle)
bar_y = np.linspace(-bar_length, bar_length, 100) * np.sin(bar_angle)

cst1 = 400.
theta_array = np.linspace(0., np.pi/2, 100)
r = bar_length + cst1 * theta_array

x = r*np.cos(theta_array + bar_angle)
y = r*np.sin(theta_array + bar_angle)
x = center[0] + np.concatenate([x, -x, bar_x])
y = center[1] + np.concatenate([y, -y, bar_y])
ax.scatter(x, y)
"""







new_cat = ngc7496.hand_made_catalogue.cat.copy()

reverse_mask = ngc7496.hand_made_catalogue.cat['a'] < ngc7496.hand_made_catalogue.cat['b']
temp_cat = new_cat.copy()
new_cat['a'][reverse_mask] = temp_cat['b'][reverse_mask]
new_cat['b'][reverse_mask] = temp_cat['a'][reverse_mask]
new_cat['theta'][reverse_mask] = temp_cat['theta'][reverse_mask] + 90.

new_cat['theta'] = new_cat['theta']%360


world_radec = WCS.pixel_to_world(ngc7496.wcs, ngc7496.hand_made_catalogue.cat['x'], ngc7496.hand_made_catalogue.cat['y'])
new_cat.add_column(world_radec.ra.deg, name='ra')
new_cat.add_column(world_radec.dec.deg, name='dec')
#world_radec.to_string('hmsdms')
new_cat.add_column(new_cat['a']/pc_to_pix, name='a_pc')
new_cat.add_column(new_cat['b']/pc_to_pix, name='b_pc')

new_cat.remove_column(name='x')
new_cat.remove_column(name='y')
new_cat.remove_column(name='a')
new_cat.remove_column(name='b')


sample_indices = np.int_( np.random.uniform(0,174,size=10) )
sample_cat = new_cat[sample_indices]

sample_cat.write('./bubbles_sample.csv', format='csv', overwrite=True)


centered_x = temp_cat['x'] - center[0]
centered_y = temp_cat['y'] - center[1]
normalization = ( centered_x**2 + centered_y**2 )**0.5
dot_prod = np.cos(new_cat['theta']*np.pi/180) * centered_x/normalization + \
           np.sin(new_cat['theta']*np.pi/180) * centered_y/normalization
ellipticity_mask = new_cat['a_pc']/new_cat['b_pc']>1.1
#ellipticity_mask = new_cat['a_kpc']/new_cat['b_kpc']>1.1
dot_prod_filtered = np.abs( dot_prod[ ellipticity_mask ] )
angles = np.arccos(dot_prod_filtered)*180/np.pi

fig, ax = plt.subplots()
ax.hist(dot_prod_filtered, bins=40, color='blue' )
ax.set_xlabel('correlation with direction to center')
ax.set_ylabel('number of bubbles')

fig, ax = plt.subplots()
ax.hist(angles, bins=25, color='blue' )
ax.set_xlabel("angles between the ellipses and the direction from the galaxy's center")
ax.set_ylabel('number of bubbles')

fig, ax = plt.subplots()
ax.hist(angles, weights=average_sizes[ellipticity_mask], bins=30, color='blue', density=True )
ax.set_xlabel(r"angle $\beta$ between the ellipses and the direction from the galaxy's center (deg)")
ax.set_ylabel('histogram of bubbles weighted by size')

a = new_cat['a_kpc']
b = new_cat['b_kpc']
fig, ax = plt.subplots()
ax.hist(angles, weights=((a-b)/a)[ellipticity_mask], bins=30, color='blue', density=True )
ax.set_xlabel(r"angle $\beta$ between the ellipses and the direction from the galaxy's center (deg)")
ax.set_ylabel('histogram of bubbles weighted by ellipticity')



"""
fig, ax = plt.subplots()
#ax.quiver( temp_cat['x'], temp_cat['y'], (temp_cat['x'] - center[0])/normalization, (temp_cat['y'] - center[1])/normalization, color='red' )
#ax.quiver( temp_cat['x'], temp_cat['y'], np.cos(new_cat['theta']*np.pi/180), np.sin(new_cat['theta']*np.pi/180), color='black' )
ax.quiver( temp_cat['x'][ellipticity_mask], temp_cat['y'][ellipticity_mask], (temp_cat['x'][ellipticity_mask] - center[0])/normalization[ellipticity_mask], (temp_cat['y'][ellipticity_mask] - center[1])/normalization[ellipticity_mask], color='red' )
ax.quiver( temp_cat['x'][ellipticity_mask], temp_cat['y'][ellipticity_mask], np.cos(new_cat['theta'][ellipticity_mask]*np.pi/180), np.sin(new_cat['theta'][ellipticity_mask]*np.pi/180), color='black' )
ax.axis('equal')
strings = np.array([str(element)[0:4] for element in dot_prod_filtered])
for i in range(len(temp_cat[ellipticity_mask])) :
    ax.text(temp_cat['x'][ellipticity_mask][i], temp_cat['y'][ellipticity_mask][i], strings[i] )
"""


a = new_cat['a_pc']
b = new_cat['b_pc']
average_sizes = (a+b)/2
#area = np.pi*a*b
fig, ax = plt.subplots()
ax.hist( (a+b)/2, bins=50, color='blue' )
ax.set_xlabel('ellipse average size (a+b)/2 (pc)')
ax.set_ylabel('number of bubbles')
#ax.set_xscale('log')
#ax.set_yscale('log')
#fig, ax = plt.subplots()
#ax.hist(area)


x = ngc7496.hand_made_catalogue.cat['x']
y = ngc7496.hand_made_catalogue.cat['y']
center = (682, 798)
dist_to_center = ( (x-center[0])**2 + (y-center[1])**2 )**0.5

fig, ax = plt.subplots()
ax.scatter(dist_to_center/kpc_to_pix, average_sizes, color='blue', alpha=0.6, marker='o')
ax.set_xlabel('distance from the center (kpc)')
ax.set_ylabel('ellipse average size (a+b)/2 (pc)')





fig, ax = plt.subplots()
ax.scatter(dist_to_center[ellipticity_mask]/kpc_to_pix, angles, average_sizes[ellipticity_mask], alpha=0.6, color='blue' )
ax.set_xlabel("distance from the center (kpc)")
ax.set_ylabel(r"angle $\beta$")


gal_angle = np.arctan2(centered_y, centered_x)*180/np.pi
gal_angle = gal_angle%360
fig, ax = plt.subplots()
ax.scatter(gal_angle[ellipticity_mask], angles, average_sizes[ellipticity_mask]/3, alpha=0.6, color='blue' )
ax.set_xlabel("galactic angle (deg)")
ax.set_ylabel(r"angle $\beta$ (deg)")


fig, ax = plt.subplots()
ax.scatter(angles, average_sizes[ellipticity_mask], ((a-b)/a)[ellipticity_mask]*300, alpha=0.6, color='blue' )
ax.set_xlabel(r"angle $\beta$ (deg)")
ax.set_ylabel('ellipse size (a+b)/2 (pc)')

fig, ax = plt.subplots()
ax.scatter(angles, average_sizes[ellipticity_mask], alpha=0.6, color='blue' )
ax.set_xlabel(r"angle $\beta$ (deg)")
ax.set_ylabel('ellipse size (a+b)/2 (pc)')

