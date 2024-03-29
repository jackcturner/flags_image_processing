# Adapted from the ceers-nircam code available at https://github.com/ceers/ceers-nircam 

import os
import yaml
import copy

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import stats as astrostats
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Ring2DKernel, Gaussian2DKernel

from scipy.ndimage import median_filter

from photutils import Background2D, BiweightLocationBackground, BkgIDWInterpolator, BkgZoomInterpolator
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint

import warnings

class Background():
    """Class for performing background subtraction on multiple observations
        based on individual or merged tiered source masks."""

    def __init__(self, background_config):
        """__init__ method for Background_Group
        
        Parameters
        ----------
        background_config (str):
            Path to the background config file.
        """

        # Read in the config file.
        self.config_filepath = background_config
        with open(background_config, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config = content[0]
    
    def replace_masked(self, sci, mask):
        """Replace masked regions of an image with a mean background estimate.
        
        Parameters
        ----------
        sci (numpy.2darray):
            The image for which to replace masked regions.
        mask (numpy.2darray):
            The image mask. 0 if unmasked, 1 if masked.

        Returns
        -------
        sci_filled (numpy.2darray):
            The image with masked regions replaced by the mean background.
        """

        # Replace masked regions of the image with NaN.
        sci_nan = np.choose(mask,(sci,np.nan))

        # Approximate mean background based on unmasked regions.
        robust_mean_background = astrostats.biweight_location(sci_nan,c=6.,ignore_nan=True)

        # Replace the masked regions with the approximate background.
        sci_filled = np.choose(mask,(sci,robust_mean_background))

        return sci_filled
    
    def off_detector(self, err):
        """Return a True or False array indicating if a pixel is off the detector."""

        # True if OFF detector, False if on detector.
        return np.isnan(err)

    def clipped_ring_median_filter(self, sci, mask, config):
        """Remove ring median filtered signal from an image after replacing masked
            pixels with a mean background estimate.
            
        Parameters
        ----------
        sci (numpy.2darray):
            The image to be filtered.
        mask (numpy.2darray):
            Image mask.
            
        Returns
        -------
        sci-filtered (numpy.2darray):
            The science image after subtracting the ring median filtered signal.
        """

        # First make a smooth background.
        bkg = Background2D(sci,
              box_size = config["RING_CLIP_BOX_SIZE"],
              sigma_clip = astrostats.SigmaClip(sigma=config["BG_SIGMA"]),
              filter_size = config["RING_CLIP_FILTER_SIZE"],
              bkg_estimator = BiweightLocationBackground(),
              exclude_percentile = 90,
              mask = mask,
              interpolator = BkgZoomInterpolator())
        
        # Estimate the rms after subtracting smooth background.
        background_rms = astrostats.biweight_scale((sci-bkg.background)[~mask]) 

        # Apply a floating ceiling to the original image
        ceiling = config["RING_CLIP_MAX_SIGMA"] * background_rms + bkg.background
        # Pixels above the ceiling are masked before doing the ring-median filtering
        ceiling_mask = sci > ceiling

        print(f"Ring median filtering with radius, width = ",end='')
        print(f'{config["RING_RADIUS_IN"]}, {config["RING_WIDTH"]}')

        # Replace masked pixels with a mean background estimate.
        sci_filled = self.replace_masked(sci,mask | ceiling_mask)

        # Median filter using a 2D ring kernel.
        ring = Ring2DKernel(config["RING_RADIUS_IN"], config["RING_WIDTH"])
        filtered = median_filter(sci_filled, footprint=ring.array)

        # Return the science image after removing the median filtered signal.
        return sci-filtered
    
    def tier_mask(self, img, mask, config, tiernum = 0):
        """Mask sources from an image using parameters dependent on the tier of masking.
            
        Parameters
        ----------
        img (numpy.2darray):
            Image from which to mask sources.
        mask (numpy.2darray):
            Current image mask.
        tiernum (int):
            The tier of source masking.
            
        Returns
        -------
        mask (numpy.2darray):
            The source mask.
        """

        # Calculate a robust RMS.
        background_rms = astrostats.biweight_scale(img[~mask])

        # Replace the masked pixels by the robust background level so the convolution doesn't smear them
        background_level = astrostats.biweight_location(img[~mask])
        replaced_img = np.choose(mask,(img,background_level))

        if tiernum == 0:
            print(f"  Median of ring-median-filtered image = {np.median(img)}")
            print(f"  Biweight RMS of ring-median-filtered image  = {background_rms}")

        # Convolve the image with a 2D Gaussian kernel.
        convolved_difference = convolve_fft(replaced_img,Gaussian2DKernel(config["TIER_KERNEL_SIZE"][tiernum]),allow_huge=True)

        # First detect the sources, then make masks from the SegmentationImage
        seg_detect = detect_sources(convolved_difference, 
                    threshold=config["TIER_NSIGMA"][tiernum] * background_rms,
                    npixels=config["TIER_NPIXELS"][tiernum], 
                     mask=mask)
        
        # Mask out the sources.
        if config["TIER_DILATE_SIZE"][tiernum] == 0:
            mask = seg_detect.make_source_mask()
        else:
            # Dilate the mask is required.
            footprint = circular_footprint(radius=config["TIER_DILATE_SIZE"][tiernum])
            mask = seg_detect.make_source_mask(footprint=footprint)

        print(f"Tier #{tiernum}:")
        print(f'  Kernel size = {config["TIER_KERNEL_SIZE"][tiernum]}')
        print(f'  N-sigma = {config["TIER_NSIGMA"][tiernum]}')
        print(f'  N-pixels = {config["TIER_NPIXELS"][tiernum]}')
        print(f'  Dilate size = {config["TIER_DILATE_SIZE"][tiernum]}')

        return mask

    def mask_sources(self, img, bitmask, config, starting_bit=1): 
        """Iteratively mask sources using self.tier_mask.

        Parameters
        ----------
        img (numpy.2darray):
            The image to be masked.
        bitmask (numpy.2darray):
            The starting bitmask.
        starting_bit (int):
            Add bits for these masks to an existing bitmask.
        
        Returns
        -------
        bitmask (numpy.2darray):
            The final combined bitmask after all tiers of source masking.
        """

        first_mask = bitmask != 0

        # Iterate over the tiers and combine masks.
        for tiernum in range(len(config["TIER_NSIGMA"])):
            mask = self.tier_mask(img, first_mask, config, tiernum=tiernum)
            bitmask = np.bitwise_or(bitmask,np.left_shift(mask,tiernum+starting_bit))
        return bitmask
    
    def estimate_background(self, img, mask, config):
        """Estimate an image background using 'zoom' interpolation.
        
        Parameters
        ----------
        img (numpy.2darray):
            The image to be background subtracted.
        mask (numpy.2darray):
            The image mask.

        Returns
        -------
        bkg (photutils.Background2D):
            Background object measured from image.
        """

        bkg = Background2D(img, 
                    box_size = config["BG_BOX_SIZE"],
                    sigma_clip = astrostats.SigmaClip(sigma=config["BG_SIGMA"]),
                    filter_size = config["BG_FILTER_SIZE"],
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = config["BG_EXCLUDE_PERCENTILE"],
                    mask = mask,
                    interpolator = BkgZoomInterpolator())
        return bkg
    
    def estimate_background_IDW(self, img, mask, config):
        """Estimate an image background using IDW interpolation.
        
        Parameters
        ----------
        img (numpy.2darray):
            The image to be background subtracted.
        mask (numpy.2darray):
            The image mask.

        Returns
        -------
        bkg (photutils.Background2D):
            Background object measured from image.
        """

        bkg = Background2D(img, 
                    box_size = config["BG_BOX_SIZE"],
                    sigma_clip = astrostats.SigmaClip(sigma=config["BG_SIGMA"]),
                    filter_size = config["BG_FILTER_SIZE"],
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = config["BG_EXCLUDE_PERCENTILE"],
                    mask = mask,
                    interpolator = BkgIDWInterpolator())
        return bkg

    def evaluate_bias(self, bkgd, err, mask):
        """Evaluate the bias between masked and unmasked pixels
        
        Parameters
        ----------
        bkgd (numpy.2darray):
            The background to evaluate.
        err (numpy.2darray):
            The error map belonging to the image 'bkgd' was measured from.
        mask (numpy.2darray):
            The image mask.
        """

        # True if on detector, False if not.
        on_detector = np.logical_not(self.off_detector(err))
    
        # Mean and deviation of background under masked pixels.
        mean_masked = bkgd[mask & on_detector].mean()
        std_masked = bkgd[mask & on_detector].std()
        stderr_masked = mean_masked/(np.sqrt(len(bkgd[mask]))*std_masked)
    
        # Mean and deviation of background in unmasked regions.
        mean_unmasked = bkgd[~mask & on_detector].mean()
        std_unmasked = bkgd[~mask & on_detector].std()
        stderr_unmasked = mean_unmasked/(np.sqrt(len(bkgd[~mask]))*std_unmasked)
        
        # Calculate the significance of the difference in mean values.
        diff = mean_masked - mean_unmasked
        significance = diff/np.sqrt(stderr_masked**2 + stderr_unmasked**2)
        
        print(f"Mean under masked pixels   = {mean_masked:.4f} +- {stderr_masked:.4f}")
        print(f"Mean under unmasked pixels = "
              f"{mean_unmasked:.4f} +- {stderr_unmasked:.4f}")
        print(f"Difference = {diff:.4f} at {significance:.2f} sigma significance")

        return
    
    def check_sci_err(self, science_images, error_images):
        """Check science and error (or background) images have been given in the correct list format.
        
        Parameters
        ----------
        science_images (Any):
            The science images to check.
        error_images (Any):
            The error (or background) images to check.
        """

        # If individual images are given convert to lists.
        if (type(science_images) == str) and (type(error_images) == str):
            science_images = [science_images]
            error_images = [error_images]
        # If not strings or lists raise an error.
        if (type(science_images) != list) and (type(error_images) != list):
            raise KeyError(f'Input images should be filepaths or lists of filepaths but have type {type(science_images)} and {type(error_images)}.')
        
        # Raise an error if the lists are not of the same length.
        if len(science_images) != len(error_images):
            raise KeyError(f'There should be corresponding images of each type but have lengths {len(science_images)} and {len(error_images)}.') 
        
        # Check all the filepaths are valid before starting.
        badpaths = []
        for image in science_images+error_images:
            if os.path.isfile(image) == False:
                badpaths.append(image)
        if len(badpaths) != 0:
            raise KeyError(f'The following are not valid files: {badpaths}')

        return science_images, error_images

    def individual_background(self, science_images, error_images, parameters = {}, replace_sci = False, store_mask = True):
        """Perform individual background subtraction with tiered source masking.
        
        Parameters
        ----------
        science_images (str/list):
            The science image(s) to measure and subtract the background from.
        error_images (str/list):
            The corresponding error images.
        replace_sci (bool):
            Whether to overwrite the science image or create a new file.
        store_mask (bool):
            Whether to store the tiered source mask as an extension. Required for merged masking.
        
        Returns
        -------
        bkgsub_filenames (list):
            List of filenames of the background subtracted images.
        """

        # Get the science and error images in the correct format.
        sci_images, err_images = self.check_sci_err(science_images, error_images)

        # Overwrite some parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', stacklevel=2)

        # Store the filenames of the background subtracted images for later.
        bkgsub_filenames = []
        for sci_filename, err_filename in zip(sci_images, err_images):

            print(f'Measuring background of {sci_filename}...')

            # Load in the images and header.
            sci, hdr = fits.getdata(sci_filename, header = True)            
            err = fits.getdata(err_filename)

            # Set up a bitmask
            bitmask = np.zeros(sci.shape,np.uint32) # Enough for 32 tiers

            # First level is for masking pixels off the detector
            off_detector_mask = self.off_detector(err)
            mask = off_detector_mask 
            bitmask = np.bitwise_or(bitmask,np.left_shift(mask,0))

            # Ring-median filter the image.
            filtered = self.clipped_ring_median_filter(sci, mask, config)
            
            # Mask sources iteratively in tiers
            bitmask = self.mask_sources(filtered, bitmask, config, starting_bit=1)
            mask = (bitmask != 0) 

            # Estimate the background using just unmasked regions
            if config["INTERPOLATOR"] == 'IDW':
                bkg = self.estimate_background_IDW(sci, mask, config)
            else:
                bkg = self.estimate_background(sci, mask, config)
            bkgd = bkg.background

            # Subtract the background
            bkgd_subtracted = sci-bkgd

            # Evaluate the bias under all sources.
            print("Bias under bright sources:")
            self.evaluate_bias(bkgd,err,mask)

            # And just under fainter sources.
            print("\nBias under fainter sources")
            faintmask = np.zeros(sci.shape,bool)
            for t in (3,4):
                faintmask = faintmask | (np.bitwise_and(bitmask,2**t) != 0)
            self.evaluate_bias(bkgd,err,faintmask)
            
            # Overwrite or create new file.
            if replace_sci == True:
                out_filename = sci_filename
            else:
                out_filename = sci_filename.replace(".fits", "_bkgsub.fits")

            # Save the file and append tier mask if needed.
            print(f'Saving background subtracted image to {out_filename}...')

            # Add parameters and function used to header.
            hdr['HIERARCH MASK_TYPE'] = 'Individual'
            for (key, value) in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            fits.writeto(out_filename, bkgd_subtracted, header=hdr, overwrite=True)

            if store_mask == True:
                # Create header for mask.
                wcs = WCS(hdr)
                hdu_mask = fits.ImageHDU(bitmask,header=wcs.to_header(),name='TIERMASK')
                # Append as extension to background subtracted file.
                hdul = fits.open(out_filename)
                hdul.append(hdu_mask)
                hdul.writeto(out_filename, overwrite=True)
                hdul.close()
            
            bkgsub_filenames.append(out_filename)

        # Return a list of background subtracted filenames.
        return bkgsub_filenames

    def merged_background(self, science_images, bkgsub_images, WCS_filter = 0, parameters = {}, merged_name = None):
        """Perform background subtraction using a mask merged from multiple images.
        
        Parameters
        ----------
        science_images (list):
            The science image(s) to measure and subtract the background from.
        bkgsub_images (list):
            The background subtracted images using individual masks.
        WCS_filter (int):
            Index into science_images. Take the WCS information from this image.
        merged_name (str/None):
            The filepath to save the merged mask to. If None, don't save.
        """
            
        print('Calculating background using merged mask:')

        # Do some checks.
        science_images, bkgsub_images = self.check_sci_err(science_images, bkgsub_images)
        if len(science_images) == 1:
            raise KeyError('Only one science image given so not possible to create a merged mask.')
        if (WCS_filter >= len(science_images)) or (WCS_filter<0):
            raise ValueError(f'WCS_filter should index science images but has value {WCS_filter} for {len(science_images)} images.')

        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', stacklevel=2)

        mask = None
        print('Generating mask...')

        # Iterate over each image and get the stored tiered mask.
        for i, bkgimage in enumerate(bkgsub_images):

            with fits.open(bkgimage) as hdu:

                # Get header information from specified filter.
                if i == WCS_filter:
                    wcs = WCS(hdu[0].header)

                # Get the mask.
                input_tiermask = hdu['TIERMASK'].data
                this_source_mask = np.left_shift(np.right_shift(input_tiermask,1),1)

                # Merge the masks.
                if mask is None:
                    mask = this_source_mask
                else:
                    mask = mask | this_source_mask 

        # The full merged mask.
        merged_mask = mask
        mask = mask.astype(np.int32)

        # Save a copy of the merged mask.
        basedir = os.path.dirname(bkgsub_images[0])
        if merged_name == None:
            merged_name = f'{basedir}/merged_mask.temp.fits'
        elif '.fits' not in merged_name:
            merged_name = f'{merged_name}.fits'
        if os.path.dirname(merged_name) != basedir:
            merged_name = f'{basedir}/{os.path.basename(merged_name)}'

        hduout = fits.PrimaryHDU(mask, header=wcs.to_header())
        hduout.writeto(merged_name, overwrite=True)

        # Run final background subtraction on each image using merged mask
        for (image, bkgimage) in zip(science_images, bkgsub_images):
            print(f'Measuring final background for {bkgimage}...')

            # Get tiermask from bgk-subtracted image to get bordermask specific to this image.
            with fits.open(bkgimage) as hdumask:
                bordermask = hdumask['TIERMASK'].data == 1 

            # Combined the merged and border mask.
            merged_mask = fits.getdata(merged_name)
            sourcemask = merged_mask | bordermask
            mask = sourcemask != 0

            # Open the science image and measure the background using the merged mask..
            sci, hdr = fits.getdata(image, header = True)
            wcs = WCS(hdr)

            if config["INTERPOLATOR"] == 'IDW':
                bkg = self.estimate_background_IDW(sci, mask, config)
            else:
                bkg = self.estimate_background(sci, mask, config)
            bkgsub = sci - bkg.background
            bkgsub = np.choose(bordermask, (bkgsub,0.))

            # Overwrite the original background image.
            print(f'Saving background subtracted image to {bkgimage}...')

            # Add parameters and function used to header.
            hdr['HIERARCH MASK_TYPE'] = 'Merged'
            for (key, value) in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            fits.writeto(bkgimage, bkgsub, header=hdr, overwrite=True)  

        # Delete merged mask if required.
        if '.temp' in merged_name:
            os.remove(merged_name)

        return        

    def full_background(self, science_images, error_images, WCS_filter = 0, parameters = {}, merged_name = None):
        """Perform iterative source masking and background subtraction based on merged mask.
        
        Parameters
        ----------
        science_images (list):
            The science image(s) to measure and subtract the background from.
        error_images (list):
            The error images corresponding to each science image.
        WCS_filter (int):
            Index into science_images. Take the WCS information from this image.
        merged_name (str/None):
            The filepath to save the merged mask to. If None, don't save.
        """

        # Measure the individual backgrounds.
        bkgsub_images = self.individual_background(science_images, error_images, parameters, replace_sci = False, store_mask = True)

        # Measure the merged background.
        self.merged_background(science_images, bkgsub_images, WCS_filter, parameters, merged_name)

        return

        