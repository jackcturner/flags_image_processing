# Adapted from the aperpy code available at 
# https://github.com/astrowhit/aperpy

import copy
import yaml
import os
import subprocess
import cv2

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table, hstack
from astropy.nddata import block_reduce, Cutout2D
from astropy.stats import mad_std, sigma_clip
from astropy.convolution import convolve, convolve_fft
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Linear1D
from astropy.visualization import ImageNormalize, LinearStretch, simple_norm

from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from photutils.detection import find_peaks

from scipy.ndimage import zoom, binary_dilation
from skimage.morphology import disk

import warnings
warnings.resetwarnings()
warnings.filterwarnings('ignore', category = UserWarning, append = True)
np.errstate(invalid = 'ignore')

class PSF():

    def __init__(self, config_filename):
        """
        __init__ method for PSF class.
        
        Arguments
        ---------
        config_filename (str)
            Path to .yml configuration file specifying parameters to use
            at each step
        """

        # Load the config file.
        self.config_filename = config_filename
        with open(self.config_filename, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config = content[0]
        
        # Initalise dictionaries for storing science filename, PSFs and
        # kernels.
        self.filenames = {}
        self.PSFs = {}
        self.Kernels = {}
    
    def measure_curve_of_growth(self, image, radii, position=None, norm=True,
                                show=False):
        """
        Measure the Curve Of Growth of an image based on provided radii.
        
        Arguments
        ---------
        image (numpy.ndarray)
            The 2D image from which to measure the COG.
        radii (List[float])
            The radii in pixels at which to measure the enclosed flux.
        position (None, list[float]) 
            The x,y position of the source centre. If None, measure from 
            moments.
        norm (bool)
            Should the COG be normalised by its maximum value?
        show (bool)
            Should the measured COG be plotted and displayed?

        Returns
        -------
        radii (List[float])
            The radii at which the enclosed energy was measured.
        cog (numpy.ndarray)
            The value of the COG at each radius.
        profile (numpy.ndarray)
            The value of the profile at each radius.
        """

        # Calculate the centroid of the source.
        if type(position) == type(None):
            position = centroid_com(image)

        # Create an aperture for each radius in radii.
        apertures = [CircularAperture(position, r = r) for r in radii]

        # Perform aperture photometry for each aperture.
        phot_table = aperture_photometry(image, apertures)

        # Calculate cumulative aperture fluxes
        cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in 
                        range(len(radii))])

        # Normalise by the maximum COG value.
        if norm == True:
            cog /= max(cog)

        # Get the profile.

        # Area enclosed by each apperture.
        area = np.pi*radii**2 
        # Difference between areas.
        area_cog = np.insert(np.diff(area),0,area[0])
        # Difference between COG elements.
        profile = np.insert(np.diff(cog),0,cog[0])/area_cog 
        # Normalise profile.
        profile /= profile.max()

        # Show the COG and profile if requested.
        if show:
            plt.grid(visible = True, alpha = 0.1)
            plt.xlabel('Radius')
            plt.ylabel('Enclosed')
            plt.scatter(radii, cog, s = 25, alpha = 0.7)
            plt.plot(radii,profile/profile.max())

        # Return the aperture radii, COG and profile.
        return radii, cog, profile
    
    def imshow(self, args, crosshairs=False, **kwargs):
        """Display a series of PSF images as a single plot.

        Arguments
        ---------
        args (List[numpy.ndarray])
            The 2D-images to be plotted.
        crosshairs (bool)
            Should crosshairs be plotted on the images?

        Returns
        -------
        fig (pyplot.figure)
            Pyplot figure object.
        ax (pyplot.axes)
            Pyplot axes object.
        """

        # Base image width.
        width = 30

        # Return if no images given to plot.
        nargs = len(args)
        if nargs == 0: 
            return

        # Set some plotting keywords if not given.

        # Number of cloumns.
        if not (ncol := kwargs.get('ncol')): 
            ncol = int(np.ceil(np.sqrt(nargs)))+1
        # Number of sigma to use in normalisation.
        if not (nsig := kwargs.get('nsig')): 
            nsig = 5
        # Stretching to use.
        if not (stretch := kwargs.get('stretch')): 
            stretch = LinearStretch()

        # Set up the figure.

        # Number of rows to plot.
        nrow = int(np.ceil(nargs/ncol))
        # Width of each panel.
        panel_width = width/ncol 
        fig, ax = plt.subplots(nrows = nrow, ncols = ncol, 
                               figsize = (ncol*panel_width, nrow*panel_width))

        # Ensure compatibility with single images.
        if type(ax) is not np.ndarray: 
            ax = np.array(ax)
        
        usedaxes = []
        for arg, axi in zip(args, ax.flat):
            usedaxes.append(axi)
            # Calculate MAD and use to normalise.
            sig = mad_std(arg[(arg != 0) & np.isfinite(arg)])

            norm = ImageNormalize(np.float32(arg), vmin = -nsig * sig,
                                  vmax = nsig*sig, stretch = stretch)

            # Plot the image.
            axi.imshow(arg, norm = norm, origin = 'lower',
                       interpolation = 'nearest')
            axi.set_axis_off()
            # Add crosshairs if requested.
            if crosshairs:
                axi.plot(50,50, color = 'red', marker = '+', ms = 10, mew = 1)

        # Remove any unused axes.
        for axi in ax.flat:
            if axi not in usedaxes:
                fig.delaxes(axi)

        # If no title given, use the object index.
        if type(title := kwargs.get('title')) is not type(None):
            for fi,axi in zip(title,ax.flat): axi.set_title(fi)

        return fig, ax
    
    def find_stars(self, sci, err, config=None, save_figs=True,
                   science_filename='science_image', outdir = './'):
        """
        Identify stars in an image using peak finding and criteria on the
        quality of COG and centre shift.

        Arguments
        ---------
        sci (numpy.ndarray)
            Science image as a 2D array.
        err (numpy.ndarray)
            Error map as a 2D array. Must be the same shape as sci.
        config (None, dict)
            The config dictionary containing parameter choices for each
            stage.
            If None, use the class attribute config.
        save_figs (bool)
            Should diagnostic figures be saved?
        science_filename (str)
            The filename of the science image. Just used for figure
            names.
        outdir (str)
            The directory in which to save outputs.

        Returns
        -------
        peaks[accept] (stropy.table.table.QTable)
            Information associated with each of the acceptable measured
            peaks.
        cutouts[accept] (numpy.ndarray):
            3D-array containing the cutouts of the acceptable peaks.
        """

        if type(config) == type(None):
            config = self.config

        print((f' Finding peaks {config["NSIG_THRESHOLD"]}x ' 
               f'above the background...'))

        # Identify peaks above the threshold and generate a catalogue.
        peaks = find_peaks(sci, threshold = config["NSIG_THRESHOLD"]*err,
                           npeaks = config["N_PEAKS"])
        
        # Add and update some columns.
        peaks.rename_column('x_peak','x')
        peaks.rename_column('y_peak','y')

        # Will calculate offset from cutout centre
        peaks['x0'] = 0.0
        peaks['y0'] = 0.0
        peaks['minv'] = 0.0 # and the minimum pixel value.

        # The COG and profile within each radius.
        for ir in np.arange(len(config["RADII"])): peaks['r'+str(ir)] = 0.
        for ir in np.arange(len(config["RADII"])): peaks['p'+str(ir)] = 0.

        print(f' Measuring properites of each object...')
        
        # For each peak.
        cutouts = []
        for index, peak in enumerate(peaks):

            # Create cutout around the measured position.
            co = Cutout2D(sci, (peak['x'], peak['y']), config["STAR_SIZE"],
                          mode = 'partial').data

            # Measure offset from cutout centre.
            position = centroid_com(co)
            peaks['x0'][index] = position[0] - config["STAR_SIZE"]//2
            peaks['y0'][index] = position[1] - config["STAR_SIZE"]//2

            # and the minimum pixel value.
            peaks['minv'][index] = np.nanmin(co)

            # Measure the the COG and profile and add to catalogue.
            radii, cog, profile = self.measure_curve_of_growth(
                co, radii = np.array(config["RADII"]), position = position,
                norm = False)
            
            for ir in np.arange(len(config["RADII"])): 
                peaks['r'+str(ir)][index] = cog[ir]
            for ir in np.arange(len(config["RADII"])): 
                peaks['p'+str(ir)][index] = profile[ir]

            cutouts.append(co)

        # Array containing all candidate cutouts and COGs.
        cutouts = np.array(cutouts)

        # Magnitude based on flux at maximum aperture radius.
        peaks['mag'] = (config["MAG_ZP"]
                        -2.5 * np.log10(peaks[f'r{len(config["RADII"])-1}']))

        # Now select only robust star candidates.

        # Magnitude within desired range.
        accept_mag =  ((peaks['mag'] < config["MAG_MIN"])
                       & (peaks['mag'] > config["MAG_MAX"]))
        # Minimum value above threshold.
        accept_min = (peaks['minv'] > config["THRESHOLD_MIN"])
        # COG is well defined.
        accept_phot = ((np.isfinite(peaks[f'r{len(config["RADII"])-1}']))
                       & (np.isfinite(peaks['r0'])))
        # Offset from cutout centre is acceptable.
        accept_shift = ((
            np.sqrt(peaks['x0']**2 + peaks['y0']**2) < config["SHIFT_LIM"])
            & (np.abs(peaks['x0']) < np.sqrt(config["SHIFT_LIM"]))
            & (np.abs(peaks['y0']) < np.sqrt(config["SHIFT_LIM"])))

        # Ratio of COG at maxmium and middle value.
        ratio = (peaks[f'r{len(config["RADII"])-1}']
                 / peaks[f'r{len(config["RADII"]) // 2}'])

        # Bin these values and find the most common radius.
        bins = np.arange(config["RANGE"][0], config["RANGE"][1],
                         config["WIDTH"])
        hist = np.histogram(ratio[(accept_mag)], bins = bins)

        i_mode = np.argmax(hist[0])
        ratio_mode = (hist[1][i_mode] + hist[1][i_mode+1])/2

        # Must be within an acceptable range.
        accept_mode = ((ratio/ratio_mode > config["THRESHOLD_MODE"][0])
                       & (ratio/ratio_mode < config["THRESHOLD_MODE"][1]))
            
        # Full selection array.
        accept = (accept_mag & accept_min & accept_phot & accept_shift
                  & accept_mode)

        # Now fit linear relation to ratio vs mag and remove outliers.
        print(' Fitting and removing outliers...')

        # Set up the fitter object.
        fitter = FittingWithOutlierRemoval(
            LinearLSQFitter(), sigma_clip, sigma = config["SIGMA_FIT"],
            niter = config['ITERATIONS_FIT'])
        # Do the fit.
        lfit, outlier = fitter(Linear1D(), x = peaks['mag'][accept],
                               y = ratio[accept])
        # Flag outliers.
        i_outlier = np.where(accept)[0][outlier]
        accept[i_outlier] = False

        # Set new ids for the accepted objects.
        peaks['id'] = 1
        peaks['id'][accept] = np.arange(1, len(peaks[accept])+1)

        print(f' Selected {sum(accept)} candidate stars.')

        # Produce diagnostic figures.
        if save_figs == True:

            # Construct the main diagnostic plot.
            plt.figure(figsize = (14,8))
            plt.subplot(231)

            # Set the ratio and magnitude limits.
            mags = peaks['mag']
            mlim_plot = np.nanpercentile(mags, [5, 95]) + np.array([-2, 1])
            plt.ylim(min(ratio) - 1, max(ratio) + 1)
            plt.xlim(mlim_plot[0], mlim_plot[1])

            plt.xlabel(f'm$_A{{{len(config["RADII"]) - 1}}}$')
            plt.ylabel(f'A{len(config["RADII"]) // 2}/'
                       f'A{len(config["RADII"]) - 1}')
            plt.grid(visible=True, alpha = 0.1)

            # All sources.
            plt.scatter(mags, ratio, alpha = 0.3, color = 'grey', s = 2,
                        label = 'All peaks')
            # Removed due to bad shift.
            plt.scatter(mags[~accept_shift], ratio[~accept_shift],
                        label = 'Bad shift', c = 'C1', alpha = 0.8, s = 6)
            # Removed by linear fit.
            plt.scatter(mags[i_outlier], ratio[i_outlier], label = 'Outlier',
                        c = 'darkred', alpha = 0.8, s = 6)
            # Accepted
            plt.scatter(mags[accept], ratio[accept], label = 'Accepted',
                        c = 'C2', alpha = 0.8, s = 6)
            # The linear fit.
            plt.plot(np.arange(14,30), lfit(np.arange(14,30)), '--',
                     c = 'k', alpha = 0.3,
                     label = 'Slope = {:.3f}'.format(lfit.slope.value))

            plt.legend()

            # The same plot, but zoomed in to the fit region.
            plt.subplot(232)
            ratio_median = np.nanmedian(ratio[accept])

            plt.ylim(ratio_median-1,ratio_median+1)
            plt.xlim(mlim_plot[0],mlim_plot[1])

            plt.xlabel(f'm$_A{{{len(config["RADII"])-1}}}$')
            plt.ylabel((f'A{len(config["RADII"])//2}/'
                        f'A{len(config["RADII"])-1}'))
            plt.grid(visible=True, alpha = 0.1)

            # All sources.
            plt.scatter(mags, ratio, alpha = 0.3, color = 'grey', s = 2,
                        label = 'All peaks')
            # Removed due to bad shift.
            plt.scatter(mags[~accept_shift], ratio[~accept_shift],
                        label = 'Bad shift', c = 'C1', alpha = 0.8, s = 6)
            # Removed by linear fit.
            plt.scatter(mags[i_outlier], ratio[i_outlier], label = 'Outlier',
                        c = 'darkred', alpha = 0.8, s = 6)
            # Accepted
            plt.scatter(mags[accept], ratio[accept], label = 'Accepted',
                        c = 'C2', alpha = 0.8, s = 6)
            # The linear fit.
            plt.plot(np.arange(14,30), lfit(np.arange(14,30)), '--',
                     c = 'k', alpha = 0.3,
                     label='Slope = {:.3f}'.format(lfit.slope.value))

            # The histogram showing aperture ratios.
            plt.subplot(233)

            # Define the bin edges.
            bins = np.arange(config["RANGE"][0], config["RANGE"][1],
                             config["WIDTH"])
            # For all peaks.
            plt.hist(ratio, bins = bins, alpha = 0.7, color = 'grey')
            # And those that were accepted.
            plt.hist(ratio[accept], bins = bins, color = 'C2', alpha = 1)
            plt.xlabel((f'A{len(config["RADII"]) // 2}/'
                        f'A{len(config["RADII"])-1}'))
            plt.ylabel('N')

            # Ratio of peak value measured in largest aperture to Photutils total.
            plt.subplot(234)
            plt.grid(visible=True, alpha = 0.1)
            # For accepted sources and outliers.
            plt.scatter(config["MAG_ZP"] - 2.5*np.log10(peaks[f'r{len(config["RADII"])-1}'][accept]),(peaks['peak_value']/peaks[f'r{len(config["RADII"])-1}'])[accept], color = 'C2', s = 10, alpha = 0.8)
            plt.scatter(config["MAG_ZP"]-2.5*np.log10(peaks[f'r{len(config["RADII"])-1}'])[i_outlier],(peaks['peak_value'] /peaks[f'r{len(config["RADII"])-1}'])[i_outlier],c='darkred', s = 10, alpha = 0.8)
            plt.ylim(0,1)
            plt.xlabel(f'm$_A{{{len(config["RADII"])-1}}}$')
            plt.ylabel(f'Peak/A{len(config["RADII"])-1}')

            # The offset of each source from the cutout centre.
            plt.subplot(235)
            plt.grid(visible=True, alpha = 0.1)
            # Accepted sources and outliers.
            plt.scatter(peaks['x0'][accept],peaks['y0'][accept],c='C2', alpha = 0.8, s=10)
            plt.scatter(peaks['x0'][i_outlier],peaks['y0'][i_outlier],c='darkred', alpha = 0.8, s =10)
            plt.xlim(-config["SHIFT_LIM"],config["SHIFT_LIM"])
            plt.ylim(-config["SHIFT_LIM"],config["SHIFT_LIM"])
            plt.xlabel('X-offset [pix]')
            plt.ylabel('Y-offset [pix]')

            # The position of the sources in the image.
            plt.subplot(236)
            # Accepted sources and outliers.
            plt.scatter(peaks['x'][accept],peaks['y'][accept],c='C2', alpha = 0.8, s=10)
            plt.scatter(peaks['x'][i_outlier],peaks['y'][i_outlier],c='darkred', alpha = 0.8, s=10)
            plt.axis('scaled')
            plt.tight_layout()
            plt.xlabel('X [pix]')
            plt.ylabel('Y [pix]')
            plt.savefig(f'{outdir}/{os.path.basename(science_filename.replace(".fits", "_diagnostic.pdf"))}')
            plt.close()

            # Show all the PSFs that will be used in stacking.
            title = ['{}: {:.1f} AB, ({:.1f}, {:.1f})'.format(ii, mm,xx,yy) for ii,mm,xx,yy in zip(peaks['id'][accept],mags[accept],peaks['x0'][accept],peaks['y0'][accept])]
            self.imshow(cutouts[accept],nsig=30,title=title)
            plt.tight_layout()
            plt.savefig(f'{outdir}/{os.path.basename(science_filename.replace(".fits", "_star_stamps.pdf"))}')
            plt.close()

        return peaks[accept], cutouts[accept]
    
    def imshift(self, img, ddx, ddy, interpolation=cv2.INTER_CUBIC):
        """
        Recentre an image using an affine transformation.

        Arguments
        ---------
        img (numpy.ndarray):
            2D image array to be recentred.
        ddx (float):
            Shift in the x direction.
        ddy (float):
            Shift in the y direction.
        interpolation (cv2 interpolator):
            Interpolation approach.
        
        Returns
        -------
        recentred (numpy.ndarray):
            The recentred image.
        """

        # Create the transformation matrix.
        M = np.float32([[1,0,ddx],[0,1,ddy]])

        # Output cutout size.
        wxh = img.shape[::-1]

        recentred = cv2.warpAffine(img, M, wxh, flags=interpolation)

        return recentred
    
    def centre(self, star_catalogue, cutouts, config = None, interpolation=cv2.INTER_CUBIC):
        """
        Recentre cutouts and measure contamination.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        cutouts (numpy.ndarray)
            3D-array contaning star candidate cutouts.
        config (None, dict)
            The config dictionary containing parameter choices for each stage.
            If None, use the class attribute config.
        interpolation (cv2 interpolator):
            Interpolation approach.

        Returns
        -------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue updated with recentering information.
        cutouts (numpy.ndarray)
            The recentred star cutouts.
        """

        if type(config) == type(None):
            config = self.config

        # Get the window width and cutout centre.
        window = config['WINDOW']
        cw = window // 2
        c0 = config["PSF_SIZE"] // 2

        pos = []
        # Iterate over the different point sources.
        for i in np.arange(len(cutouts)):

            cutout = cutouts[i,:,:]

            # Measure the COM of the source within the window.
            co_window = Cutout2D(cutout, (c0,c0), window, mode='partial', fill_value=0).data
            co_window[~np.isfinite(co_window)] = 0
            x0, y0 = centroid_com(co_window)

            # Recentre the cutout.
            cutout = self.imshift(cutout, (cw-x0), (cw-y0), interpolation=interpolation)

            # Now measure COM on recentered cutout.
            co_window = Cutout2D(cutout, (c0,c0), window, mode='partial', fill_value=0).data

            # Using small window.
            x1,y1 = centroid_com(co_window)

            # and positive definite in case there are strong ying yang residuals.
            x2,y2 = centroid_com(np.maximum(cutout,0))

            # Record difference in shift between these two cases
            dsh = np.sqrt(((c0-x2)-(cw-x1))**2 + ((c0-y2)-(cw-y1))**2)
            # and the old and new positions.
            pos.append([cw-x0,cw-y0,cw-x1,cw-y1,dsh])

            # Mask infinite or zero values.
            cutout = np.ma.array(cutout, mask = ~np.isfinite(cutout) | (cutout==0))

            # Store the shifted cutout in place of the old one.
            cutouts[i,:,:] = cutout

        # Add these measurements to the star catalogue.
        star_catalogue = hstack([star_catalogue, Table(np.array(pos),names=['x0','y0','x1','y1','dshift'])])
    
        return star_catalogue, cutouts
    
    def measure(self, star_catalogue, cutouts, config = None):
        """
        Measure the photometric properties of stellar sources.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        cutouts (numpy.ndarray)
            3D-array contaning star candidate cutouts.
        config (None, dict)
            The config dictionary containing parameter choices for each stage.
            If None, use the class attribute config.

        Return
        ------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue updated with photometry information.
        cutouts (numpy.ndarray)
            Star cutouts with saturated regions masked.
        """

        print(' Measuring photometric properties...')

        if type(config) == type(None):
            config = self.config

        # Find the peak value in each cutout.
        peaks = np.array([cutout.max()for cutout in cutouts])
        peaks[~np.isfinite(peaks) | (peaks==0)] = 0 # Must be finite and non-zero.

        # Create a mask around the centre of the cutout.
        norm_aper = CircularAperture((config["PSF_SIZE"] // 2, config["PSF_SIZE"] // 2), r=config["NORM_RADIUS"])
        norm_mask = Cutout2D(norm_aper.to_mask(), (config["NORM_RADIUS"], config["NORM_RADIUS"]), self.config["PSF_SIZE"], mode='partial').data

        # Measure the flux within the norm radius for each star.
        phot = [aperture_photometry(cutout, norm_aper)['aperture_sum'][0] for cutout in cutouts]

        # New measurement on unmasked cutout (by casting to array).
        # Used to determine saturation.
        sat =  [aperture_photometry(cutout, norm_aper)['aperture_sum'][0] for cutout in np.array(cutouts)]
        # Minimum unmasked value.
        cmin = [np.nanmin(cutout*norm_mask) for cutout in cutouts]

        # Combine with mask from recentering.
        for i in np.arange(len(cutouts)):
            cutouts[i].mask |= (cutouts[i]*norm_mask) < 0.0

        # Measure the RMS.
        rms_array = []
        for cutout in cutouts:
            rms = mad_std(cutout, ignore_nan=True)
            rms_array.append(rms)

        # Save some information to the catalogue.

        # Fraction of cutout that is masked.
        star_catalogue['frac_mask'] = 0.0
        # Fraction of flux that is within the normalisation radius.
        star_catalogue['phot_frac_mask'] = 1.0

        # New peak value
        star_catalogue['peak'] =  peaks
        # and minimum value.
        star_catalogue['cmin'] =  np.array(cmin)
        # Photometry measured in aperture.
        star_catalogue['phot'] =  np.array(phot)
        # Is the cutout saturated?
        star_catalogue['saturated'] =  np.int32(~np.isfinite(np.array(sat)))
        # The signal to noise ratio.
        star_catalogue['snr'] = 2*np.array(phot)/np.array(rms_array)

        return star_catalogue, cutouts
    
    def select(self, star_catalogue, snr_lim = 800, dshift_lim=3, mask_lim=0.40, phot_frac_mask_lim = 0.85):
        """
        Select objects satisfying given conditions from the catalogue.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        snr (float)
            Minimum accepted SNR.
        dshift (float)
            Maximum accepted difference in shift measured when recentering.
        mask_lim (float)
            Maximum accepted fraction of masked pixels.
        phot_frac_mask_lim (float)
            Minimum accepted fraction of flux within "NORM_RADIUS" of the centre.

        Return
        ------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue with updated selection column.
        """

        # Check which objects in the catalogue satisfy all conditions and add flag to catalogue.
        accept = (star_catalogue['dshift'] < dshift_lim) & (star_catalogue['snr'] > snr_lim) & (star_catalogue['frac_mask'] < mask_lim) & (star_catalogue['phot_frac_mask'] > phot_frac_mask_lim)
        star_catalogue['accept'] = np.int32(accept)

        # Also include individual conditions.
        star_catalogue['accept_shift'] = (star_catalogue['dshift'] < dshift_lim)
        star_catalogue['accept_snr'] = (star_catalogue['snr'] > snr_lim)
        star_catalogue['accept_frac_mask'] = (star_catalogue['frac_mask'] < mask_lim)
        star_catalogue['accept_phot_frac_mask'] = (star_catalogue['phot_frac_mask'] > phot_frac_mask_lim)

        # Format the columns to 3 D.P
        for c in star_catalogue.colnames:
            if 'id' not in c: star_catalogue[c].format='.3g'

        return star_catalogue
    
    def stack(self, star_catalogue, masked_cutouts, cutouts, config = None, save_figs = True, science_filename = 'science_filename', outdir = './'):
        """
        Stack individual PSFs based on a pixelwise sigma clipped mean.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        masked_cutouts (numpy.ndarray)
            3D-array of star cutouts with saturated regions masked.
        cutouts (numpy.ndarray)
            3D-array of unmasked star cutouts.
        config (None, dict)
            The config dictionary containing parameter choices for each stage.
            If None, use the class attribute config.
        save_figs (bool)
            Save figure showing masked regions of each cutout.
        science_filename (str)
            Name of science image file. Only used for saving figure.
        outdir (str)
            Directory in which to save figure.

        Returns
        -------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue updated with stacking information.
        masked_cutouts (numpy.ndarray)
            Star cutouts with masks updated by sigma clipping.
        stack (numpy.ndarray)
            Average 2D PSF measured by sigma-clipped stacking.
        """

        if type(config) == type(None):
            config = self.config

        # Get indexes of acceptable objects.
        i_accept = np.where(star_catalogue['accept'])[0]

        print(f' Stacking {len(i_accept)} robust candidates...')

        # Get flux within normalisation radius.
        norm = star_catalogue['phot'][i_accept]

        # Normalise by flux within the normalisation radius.
        unmasked_cutouts = cutouts[i_accept].copy()
        for i in np.arange(len(unmasked_cutouts)): 
            unmasked_cutouts[i] = unmasked_cutouts[i]/norm[i]

        # Stack the images based on the pixel-wise sigma clipped mean.

        # Make a copy of the data.
        clipped_data = unmasked_cutouts.copy()

        # Perform required number of sigma clipping iterations.
        for i in range(config['MAX_ITERS']):
            clipped_data, lo, hi = sigma_clip(clipped_data, sigma=config['STACK_SIGMA'], maxiters=0, axis=0, masked=True, grow=False, return_bounds=True)
            
            # Grow the mask
            for i in range(len(clipped_data.mask)): 
                clipped_data.mask[i,:,:] = binary_dilation(clipped_data.mask[i,:,:], structure=disk(config['DILATE_RADIUS']), iterations=1)

        # The single averaged PSF.
        stack = np.mean(clipped_data,axis=0)

        for i in np.arange(len(unmasked_cutouts)):
            # Does object satisfy criteria and also have its central pixel unmasked?
            star_catalogue['accept'][i_accept[i]] = star_catalogue['accept'][i_accept[i]] and ~clipped_data[i].mask[config["PSF_SIZE"] // 2, config["PSF_SIZE"] // 2]
            # Use the clipped mask.
            masked_cutouts[i_accept[i]].mask = clipped_data[i].mask
            # The fraction of pixels that are masked.
            mask = masked_cutouts[i_accept[i]].mask
            star_catalogue['frac_mask'][i_accept[i]] = np.size(mask[mask]) / np.size(mask)
    
        if save_figs == True:
            # Save the masked cutouts of all the stacked sources.
            title = ['{}: Mask - {:.1f}%'.format(ii, 100*frac) for ii,frac in zip(star_catalogue['id'][i_accept],star_catalogue['frac_mask'][i_accept])]
            fig, ax = self.imshow(masked_cutouts[i_accept], title=title, nsig=30)
            fig.savefig(f'{outdir}/{os.path.basename(science_filename.replace(".fits", "_masked_cutouts.pdf"))}',dpi=300)
            plt.close()

        # Calculate the fraction of the flux within the normalisation radius.
        aper = CircularAperture((config["PSF_SIZE"] // 2,config["PSF_SIZE"] // 2), r = config["NORM_RADIUS"])
        phot = [aperture_photometry(cutout, aper)['aperture_sum'][0] for cutout in masked_cutouts]
        star_catalogue['phot_frac_mask'] = phot/star_catalogue['phot']

        return star_catalogue, masked_cutouts, stack 
    
    def measure_PSF(self, science_filenames, error_filenames, bands = None, parameters = {}, save_PSF = False, save_figs = False, outdir = './'):
        """
        Run star idenfication and stacking methods to obtain average PSF(s).
        Add generated PSF(s) to internal storage for later use.

        Arguments
        ---------
        science_filenames (str, list)
            Fits filename(s) containing science image(s) from which to identify stars.
        error_filenames (str, list)
            Fits filenames containing error map of each science image.
        bands (str, list, None)
            The broadband filters that these images correspond to.
            If None, use zero based indexing.
        parameters (dict)
            Parameter key-value pairs to update in the config file just for this run.
        save_PSF (bool)
            Should the PSF be saved to a fits file?
        save_figs (bool)
            Should diagnostic figures be saved?
        outdir (str)
            Directory in which to save figures.
        """

        # If single image given, convert to list.
        if type(science_filenames) == str:
            science_filenames = [science_filenames]
            error_filenames = [error_filenames]
        if type(bands) == str:
            bands = [bands]
            
        # If bands are not defined, just use index.
        if bands == None:
            bands = np.arange(0, len(science_filenames))

        # Overwrite some config parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                    hdr[f'HIERARCH {key}'] = str(value)
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', stacklevel=2)         

        for science_filename, error_filename, band in zip(science_filenames, error_filenames, bands):

            print(f'Measuring empirical PSF from {science_filename}...')

            # Get image and corresponding header.
            sci, hdr = fits.getdata(science_filename, header = True)
            err = fits.getdata(error_filename)

            # Get information and cutouts of stars in the image.
            stars, cutouts = self.find_stars(sci, err, config, save_figs=save_figs, science_filename = science_filename, outdir = outdir)

            # Generate new cutouts at the full PSF size.
            psfs = np.array([Cutout2D(sci, (stars['x'][i], stars['y'][i]), config["PSF_SIZE"], mode='partial').data for i in np.arange(len(stars))])
            psfs_masked = np.ma.array(psfs, mask = ~np.isfinite(psfs) | (psfs == 0))

            # Move stars to the centre of the cutouts.
            stars, psfs_masked = self.centre(stars, psfs_masked, config)

            # Measure their flux and SNR.
            stars, psfs_masked = self.measure(stars, psfs_masked, config)

            # Select objects with acceptable shift and SNR.
            stars = self.select(stars, config["SNR_LIM"], config["DSHIFT_LIM"], 0.99, 0.99)

            # Stack the cutouts to create a single PSF.
            stars, psfs_masked, psf_average = self.stack(stars, psfs_masked, psfs, config, save_figs=save_figs, science_filename = science_filename, outdir = outdir)

            # Normalise the PSF and remove mask.
            psf_average = np.array(psf_average)/np.sum(np.array(psf_average))

            if save_PSF == True:
                fits.writeto(f'{outdir}/{os.path.basename(science_filename.replace(".fits", "_EPSF.fits"))}', psf_average/np.sum(psf_average), header = hdr, overwrite=True)

            if save_figs == True:

                fig, ax = plt.subplots()
                sig = mad_std(psf_average[(psf_average != 0) & np.isfinite(psf_average)])
                norm = ImageNormalize(np.float32(psf_average), vmin=-50*sig, vmax=50*sig, stretch=LinearStretch())
                plt.imshow(psf_average, norm = norm, origin='lower', interpolation='none')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(f'{outdir}/{os.path.basename(science_filename.replace(".fits", "_EPSF.pdf"))}')
                plt.close()
            
            if band != None:
                if band in self.PSFs.keys():
                    raise Warning(f'Previously measured {band} PSF overwritten.')
                self.PSFs[band] = psf_average
                self.filenames[band] = [science_filename, error_filename]

            print(f'Done.')

        return
    
    def compare_COG(self, radii, bands = None):
        """
        Create a diagnostic plot, comparing the COGs of measured PSFs.
        
        Arguments
        ---------
        radii (array-like)
            The radii at which to measure the enclosed energy.
        bands (array-like)
            The PSFs of these bands will be plotted.
            If None, plot all measured PSFs.

        Returns
        -------
        fig (pyplot.figure)
            Pyplot figure object.
        ax (pyplot.axes)
            Pyplot axes object.
        """

        # If no bands indicated, use all.
        if bands == None:
            bands = self.PSFs.keys()

        # Set up the plot.
        fig, ax = plt.subplots()
        ax.minorticks_on()
        ax.tick_params(axis='both', which = 'both', direction='in')
        #ax.tick_params(axis='y', direction='in')
        plt.xlabel('Radius [pix]')
        plt.ylabel('Enclosed Energy')
        plt.grid(visible=True, alpha = 0.1)

        # For each PSF.
        for band, psf in self.PSFs.items():

            if band in bands:

                # Measure the COG
                radii, cog, profile = self.measure_curve_of_growth(psf, radii, norm = False)

                ax.plot(radii, cog, label = band, alpha = 0.8)

        plt.legend()
        plt.show()

        return fig, ax
    
    def plot_profile(self, source, target, radii_pix):
        """Plot the profiles of two PSFs.
        
        Arguments
        ---------
        source (np.ndarray)
            The first PSF for which to measure profile.
        target (np.ndarray)
            The second PSF for which to measure profile.
        radii_pix (array-like)
            The radii in pixels at which to measure the enclosed energy.
        
        Returns
        -------
        radii_pix (array-like)
            The radii in pixels at which the enclosed energy was measured.
        flux_source (tuple)
            Profile of the first PSF.
        flux_target (tuple)
            Profile of the second PSF.
        """

        shape = source.shape
        center = (shape[1] // 2, shape[0] // 2)
        apertures = [CircularAperture(center, r=r) for r in radii_pix] #r in pixels

        phot_table = aperture_photometry(source, apertures)
        flux_source = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])

        phot_table = aperture_photometry(target, apertures)
        flux_target = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])

        return radii_pix[:-1], (flux_source)[0:-1], (flux_target)[0:-1]
    
    def generate_kernel(self, target_band, bands = None, parameters = {}, save_kernel = True, save_figs = True, outdir = './'):
        """Create a kernel to match the measured PSF to a target PSF.

        Arguments
        ---------
        target_band (str)
            Key of the target PSF to match to as defined when measured.
        bands (list, None)
            The bands to generate matching kernels for. Will ignore target.
            If None, use measured PSFs.
        parameters (dict)
            Parameter key-value pairs to update in the config file just for this run.
        save_kernel (bool)
            Should the kernel be saved as a fits file?
        save_figs (bool)
            Should diagnostic figures be saved?
        outdir (str)
            The directory in which to store temporary and requested outputs.
        """

        # Overwrite some config parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', stacklevel=2)

        hdr = fits.Header()
        hdr['PIXSCALE'] = (config['PIXEL_SCALE'], 'Pixel scale in arcsec')

        # Get the target PSF.
        target = self.PSFs[target_band]

        # Create sub-dict in Kernels for this target.
        self.Kernels[target_band] = {}
        
        # Oversample if required.
        if config['OVERSAMPLE'] > 1:
            print(f' Oversampling by {config["OVERSAMPLE"]}x...')
            target = zoom(target, config['OVERSAMPLE'])

        # Renormalise
        target /= target.sum()

        # Save to temporary file for passing to PyPHER.
        fits.writeto(f'{outdir}/target.temp.fits', target, header = hdr, overwrite=True)

        # If no bands indicated, measure kernel for all.
        if bands == None:
            bands = self.PSFs.keys()

        # For each band.
        for band, source in self.PSFs.items():

            # Skip the target or omitted bands.
            if (band == target_band) or (band not in bands):
                continue

            # Oversample if required.
            if config['OVERSAMPLE'] > 1:
                source = zoom(source, config['OVERSAMPLE'])
    
            # Renormalise
            source /= source.sum()

            fits.writeto(f'{outdir}/source.temp.fits', source, header = hdr, overwrite=True)

            # Filename of matching kernel.
            match_name = f'{outdir}/{band}_to_{target_band}_kernel.fits'

            # Remove if already exists as pypher will not overwrite.
            if os.path.isfile(match_name):
                os.remove(match_name)

            # Run pypher
            print(' Running pypher...')
            pypherCMD = ['pypher', f'{outdir}/source.temp.fits', f'{outdir}/target.temp.fits', match_name, '-r', str(config["R_PARAMETER"]), '-s', str(config["ANGLE_SOURCE"]), '-t', str(config["ANGLE_TARGET"])]
            p = subprocess.Popen(pypherCMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in p.stderr:
                print(line.decode(encoding="UTF-8"))
            out, err = p.communicate()

            # Remove the temporary source file.
            os.remove(f'{outdir}/source.temp.fits')

            # Load the generated kernel and delete the PyPHER file.
            kernel = fits.getdata(match_name)
            os.remove(match_name)
            os.remove(match_name.replace('.fits', '.log'))

            # If oversampled, renormalise and overwrite saved and stored kernels.
            if config['OVERSAMPLE'] > 1:

                kernel = block_reduce(kernel, block_size=config['OVERSAMPLE'], func=np.sum)
                kernel /= kernel.sum()
                kernel = np.float32(np.array(kernel))

            # Store for later use
            self.Kernels[target_band][band] = kernel

            # and save to fits file if requested.
            if save_kernel == True:

                # Create the header.
                hdr = fits.Header()
                hdr['SOURCE'] = (band, 'Source PSF')
                hdr['TARGET'] = (target_band, 'Target PSF')
                hdr['OVERSAMP'] = (config["OVERSAMPLE"], 'Degree of oversampling')
                hdr['ANGLE_S'] = (config["ANGLE_SOURCE"], 'Angle of source PSF')
                hdr['ANGLE_T'] = (config["ANGLE_TARGET"], 'Angle of target PSF')

                fits.writeto(f'{outdir}/{os.path.basename(self.filenames[band][0]).replace(".fits", f"_kernel_{target_band}.fits")}', kernel, header= hdr, overwrite=True)

            # Construct the diagnostic figure.
            if save_figs == True:
            
                print(f' Plotting kernel checkfile...')

                plt.figure(figsize=(32,4))

                # Normalisation function for images.
                simple = simple_norm(kernel, stretch='linear', power=1, min_cut=-5e-4, max_cut=5e-4)

                # Show the source, target and kernel images.
                plt.subplot(1,7,1)
                plt.title('Source: '+band)
                plt.imshow(source, norm=simple, interpolation='antialiased', origin='lower')
                plt.subplot(1,7,2)
                plt.title('Target: '+target_band)
                plt.imshow(target, norm=simple, interpolation='antialiased', origin='lower')
                plt.subplot(1,7,3)
                plt.title('Kernel')
                plt.imshow(kernel, norm=simple, interpolation='antialiased', origin='lower')

                # Convolve the source with the kernel.
                filt_psf_conv = convolve_fft(source, kernel)

                # Show convolved PSF.
                plt.subplot(1,7,4)
                plt.title("Convolved "+band)
                plt.imshow(filt_psf_conv, norm=simple, interpolation='antialiased', origin='lower')

                # Show the residual after convolution.
                plt.subplot(1,7,5)
                plt.title('Residual')
                res = filt_psf_conv-target
                plt.imshow(res, norm=simple, interpolation='antialiased', origin='lower')

                # Show the COGs of convolved and target PSFs and the ratio.
                plt.subplot(1,7,7)
                r,pf,pt = self.plot_profile(filt_psf_conv,target, np.arange(1,40,1))
                plt.plot(r*self.config["PIXEL_SCALE"], pf/pt)
                plt.ylim(0.95,1.05)
                plt.xlabel('Radius [arcsec]')
                plt.ylabel('EE convolved source / EE target')

                plt.subplot(1,7,6)
                plt.plot(r*self.config["PIXEL_SCALE"],pf,lw=3, label = 'Convolved source')
                plt.plot(r*self.config["PIXEL_SCALE"],pt,'--',alpha=0.7,lw=3, label = 'Target')
                plt.xlabel('Radius [arcsec]')
                plt.ylabel('Enclosed energy')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'{outdir}/{os.path.basename(self.filenames[band][0]).replace(".fits", f"_match_{target_band}_diagnostic.pdf")}')
                plt.close()

            # Remove the target PSF file.
            os.remove(f'{outdir}/target.temp.fits')
        
        return
    
    def convolve_image(self, target_band, bands = None, parameters = {}, outdir = './'):
        """
        Convolve images used to measure PSF with generated matching kernels.

        Arguments
        ---------
        target_band (str)
            The target band for convolution. Matching kernels must already be generated.
        bands (list, None)
            The bands on which to perform convolution. Will ignore target.
            If None, use all measured PSFs.
        parameters (dict)
            Parameters to update in the config file just for this run.
        outdir (str)
            Directory in which to store convolved images.
        """

        # Check that matching kernels have been generated.
        if target_band not in self.Kernels.keys():
            raise KeyError(f'Matching kernels for {target_band} have not been generated. Run generate_kernel first.')
                
        # Overwrite some config parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', stacklevel=2)

        # If no bands indicated, convolve all.
        if bands == None:
            bands = self.PSFs.keys()

        # For each band.
        for band, kernel in self.Kernels[target_band].items():

            print(f'Matching {band} to {target_band}...')

            # Load in the science and error images.
            sci, sci_hdr = fits.getdata(self.filenames[band][0], header = True)
            err, err_hdr = fits.getdata(self.filenames[band][1], header = True)

            # Convolve the images.
            if config["FFT"] == True:
                print(' Convolving science image...')
                convolved_sci = convolve_fft(sci, kernel, allow_huge=True)
                convolved_err = convolve_fft(err, kernel, allow_huge=True)
            else:
                print(' Convolving error image...')
                convolved_sci = convolve(sci, kernel)
                convolved_err = convolve(err, kernel)

            # Add some header keywords.
            sci_hdr['FFT'] = (config["FFT"], 'Convolved by Fast Fourier Transform')
            sci_hdr['KERNEL'] = (target_band, 'Convolved with this kernel')
            err_hdr['FFT'] = (config["FFT"], 'Convolved by Fast Fourier Transform')
            err_hdr['KERNEL'] = (target_band, 'Convolved with this kernel')

            # Ensure off detector region values don't change.
            convolved_sci[np.isnan(err)] = 0
            convolved_err[np.isnan(err)] = np.nan

            # Save the convloved images.
            fits.writeto(f'{outdir}/{os.path.basename(self.filenames[band][0]).replace(".fits", f"_match{target_band}.fits")}', convolved_sci, sci_hdr, overwrite = True)
            fits.writeto(f'{outdir}/{os.path.basename(self.filenames[band][1]).replace(".fits", f"_match{target_band}.fits")}', convolved_err, err_hdr, overwrite = True)

            print(' Done.')
        
        return