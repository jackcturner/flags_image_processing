# Adapted from the ceers-nircam code available at 
# https://github.com/ceers/ceers-nircam 

import os
import yaml
import copy

import numpy as np

from astropy.io import fits
from astropy import stats as astrostats
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Ring2DKernel, Gaussian2DKernel

from scipy.ndimage import median_filter

from photutils.background import Background2D, BiweightLocationBackground, BkgIDWInterpolator
from photutils.background import BkgZoomInterpolator
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint

import warnings

class Background():
    """
    Perform background subtraction on multiple observations based on 
    individual or merged tiered source masks.
    """

    def __init__(self, background_config):
        """
        __init__ method for Background
        
        Arguments
        ---------
        background_config (str)
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
        """
        Replace masked regions of an image with a mean background 
        estimate.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D image.
        mask (numpy.ndarray)
            The 2D image mask where 0 if unmasked, 1 if masked.

        Returns
        -------
        sci_filled (numpy.ndarray)
            The image with masked regions replaced by the mean 
            background.
        """

        # Replace masked regions of the image with NaN.
        sci_nan = np.choose(mask, (sci, np.nan))

        # Approximate mean background based on unmasked regions.
        robust_mean_background = astrostats.biweight_location(sci_nan, c = 6., ignore_nan = True)

        # Replace the masked regions with the approximate background.
        sci_filled = np.choose(mask, (sci, robust_mean_background))

        return sci_filled

    def clipped_ring_median_filter(self, sci, mask, config):
        """
        Remove ring median filtered signal from an image.
            
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D image to be filtered.
        mask (numpy.ndarray)
            The 2D image mask where 0 if unmasked, 1 if masked.
            
        Returns
        -------
        rmf_image (numpy.ndarray)
            The 2D image after subtracting the ring median filtered 
            signal.
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
        background_rms = astrostats.biweight_scale((sci - bkg.background)[~mask]) 

        # Apply a floating ceiling to the original image
        ceiling = config["RING_CLIP_MAX_SIGMA"] * background_rms + bkg.background
        # Pixels above the ceiling are masked before doing the
        # ring-median filtering
        ceiling_mask = sci > ceiling

        print(f"Ring median filtering with radius, width = ", end = '')
        print(f'{config["RING_RADIUS_IN"]}, {config["RING_WIDTH"]}')

        # Replace masked pixels with a mean background estimate.
        sci_filled = self.replace_masked(sci, mask | ceiling_mask)

        # Median filter using a 2D ring kernel.
        ring = Ring2DKernel(config["RING_RADIUS_IN"], config["RING_WIDTH"])
        filtered = median_filter(sci_filled, footprint = ring.array)

        # Return the science image after removing the median filtered
        # signal.
        rmf_image = sci - filtered

        return rmf_image
    
    def tier_mask(self, img, mask, scaling, config, tiernum=0):
        """
        Update a source mask using parameters dependent on the tier of 
        masking.
            
        Arguments
        ---------
        img (numpy.ndarray)
            The 2D image from which to mask.
        mask (numpy.ndarray)
            The 2D image mask where 0 if unmasked, 1 if masked.
        scaling (numpy.ndarray)
            The 2D image defining the detection threshold scaling of 
            each pixel.
        tiernum (int)
            The tier of source masking.
            
        Returns
        -------
        mask (numpy.ndarray)
            The updated 2D source mask.
        """

        print(f"Tier #{tiernum}:")
        print(f'  Kernel size = {config["TIER_KERNEL_SIZE"][tiernum]}')
        print(f'  N-sigma = {config["TIER_NSIGMA"][tiernum]}')
        print(f'  N-pixels = {config["TIER_NPIXELS"][tiernum]}')
        print(f'  Dilate size = {config["TIER_DILATE_SIZE"][tiernum]}')

        # Calculate a robust RMS.
        background_rms = astrostats.biweight_scale(img[~mask])

        # Replace the masked pixels by the robust background level so the
        # convolution doesn't smear them.
        background_level = astrostats.biweight_location(img[~mask])
        replaced_img = np.choose(mask,(img,background_level))

        print(f"  Median of ring-median-filtered image = {np.median(img[~mask])}")
        print(f"  Biweight RMS of ring-median-filtered image  = {background_rms}")

        # Convolve the image with a 2D Gaussian kernel.
        convolved_difference = convolve_fft(
            replaced_img, Gaussian2DKernel(config["TIER_KERNEL_SIZE"][tiernum]), allow_huge = True)

        # First detect the sources, then make masks from the 
        # SegmentationImage
        seg_detect = detect_sources(convolved_difference, 
                                    threshold = config["TIER_NSIGMA"][tiernum] * background_rms * scaling, 
                                    npixels = config["TIER_NPIXELS"][tiernum],
                                    mask = mask)
        
        # Mask out the sources.
        if config["TIER_DILATE_SIZE"][tiernum] == 0:
            mask = seg_detect.make_source_mask()
        else:
            # Dilate the mask is required.
            footprint = circular_footprint(radius = config["TIER_DILATE_SIZE"][tiernum])
            mask = seg_detect.make_source_mask(footprint = footprint)

        return mask

    def mask_sources(self, img, bitmask, scaling, config, starting_bit=1): 
        """
        Iteratively mask sources using self.tier_mask.

        Arguments
        ----------
        img (numpy.ndarray)
            The 2D image to be masked.
        bitmask (numpy.ndarray)
            The 2D starting bitmask.
        scaling (numpy.ndarray)
            The 2D image defining the detection threshold scaling of 
            each pixel.
        starting_bit (int)
            Add bits for these masks to an existing bitmask.
        
        Returns
        -------
        bitmask (numpy.ndarray)
            The final 2D combined bitmask after all tiers of source
            masking.
        """

        # Iterate over the tiers and combine masks.
        for tiernum in range(len(config["TIER_NSIGMA"])):
                mask = self.tier_mask(img, (bitmask != 0), scaling, config, tiernum = tiernum)
                bitmask = np.bitwise_or(bitmask, np.left_shift(mask, tiernum + starting_bit))  
        return bitmask
    
    def estimate_background(self, img, mask, config):
        """
        Estimate an image background using 'zoom' interpolation.
        
        Arguments
        ---------
        img (numpy.ndarray)
            The 2D image to be background subtracted.
        mask (numpy.ndarray)
            The 2D image mask where 0 if unmasked, 1 if masked.

        Returns
        -------
        bkg (photutils.Background2D)
            Background object measured from the image.
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
        """
        Estimate an image background using 'IDW' interpolation.
        
        Arguments
        ---------
        img (numpy.ndarray)
            The 2D image to be background subtracted.
        mask (numpy.ndarray)
            The 2D image mask where 0 if unmasked, 1 if masked.

        Returns
        -------
        bkg (photutils.Background2D)
            Background object measured from the image.
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
        """Evaluate the bias between masked and unmasked pixels.
        
        Arguments
        ---------
        bkgd (numpy.ndarray)
            The 2D background to evaluate.
        err (numpy.ndarray)
            The error map belonging to the image 'bkgd' was measured 
            from.
        mask (numpy.ndarray)
            The 2D image mask where 0 if unmasked, 1 if masked.
        Returns
        -------
        diff (float)
            The difference in mean values under masked and unmasked
            pixels.
        significance (float)
            The significance of the difference in mean values.
        """

        # True if on detector, False if not.
        on_detector = np.logical_not(np.isnan(err))
    
        # Mean and deviation of background under masked pixels.
        mean_masked = bkgd[mask & on_detector].mean()
        std_masked = bkgd[mask & on_detector].std()
        stderr_masked = mean_masked / (np.sqrt(len(bkgd[mask]))*std_masked)
    
        # Mean and deviation of background in unmasked regions.
        mean_unmasked = bkgd[~mask & on_detector].mean()
        std_unmasked = bkgd[~mask & on_detector].std()
        stderr_unmasked = mean_unmasked / (np.sqrt(len(bkgd[~mask]))*std_unmasked)
        
        # Calculate the significance of the difference in mean values.
        diff = mean_masked - mean_unmasked
        significance = diff / np.sqrt(stderr_masked**2 + stderr_unmasked**2)
        
        print(f"Mean under masked pixels   = {mean_masked:.4f} +- {stderr_masked:.4f}")
        print(f"Mean under unmasked pixels = "
              f"{mean_unmasked:.4f} +- {stderr_unmasked:.4f}")
        print(f"Difference = {diff:.4f} at {significance:.2f} sigma significance")

        return diff, significance

    def individual_background(self, science_images, error_images, weight_images, parameters={}, suffix='bkgsub',
                              replace_sci=False, store_mask=True):
        """
        Perform individual background subtraction with tiered source masking.
        
        Arguments
        ---------
        science_images (str, List[str])
            Filename(s) of science image(s) to subtract the background
            from.
        error_images (str, List[str])
            The corresponding error image(s).
        weight_images (str, List[str])
            The corresponding weight image(s).
        parameters (dict)
            Key-value pairs overwritting parameters given in the config
            file.
        suffix (str)
            Suffix to append to the science filenames when saving.
        replace_sci (bool)
            Whether to overwrite the science image or create a new file.
        store_mask (bool)
            Whether to store the tiered source mask as an extension.
            Required for merged masking.
        
        Returns
        -------
        bkgsub_filenames (List[str])
            Filenames of the generated background subtracted images.
        """

        # If individual images are given convert to lists.
        if type(science_images) == str:
            science_images = [science_images]
        if type(error_images) == str:
            error_images = [error_images]
        if type(weight_images) == str:
            weight_images = [weight_images]

        # Raise an error if the lists are not of the same length.
        if (len(science_images) != len(error_images)) or (len(error_images) != len(weight_images)):
            raise KeyError('There should be corresponding images of each type.')
        
        # Overwrite some parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.',
                                  stacklevel=2)
                    
        # Store the filenames of the background subtracted images for
        # later.
        bkgsub_filenames = []
        for sci_filename, err_filename, wht_filename in zip(science_images, error_images, weight_images):

            print(f'Measuring background of {sci_filename}...')

            # Load in the images and header.
            sci, hdr = fits.getdata(sci_filename, header = True)            
            err = fits.getdata(err_filename)
            wht = fits.getdata(wht_filename)

            # Set up a bitmask
            bitmask = np.zeros(sci.shape,np.uint32) # Enough for 32 tiers

            # First level is for masking pixels off the detector
            off_detector_mask = np.isnan(err)
            mask = off_detector_mask 
            bitmask = np.bitwise_or(bitmask, np.left_shift(mask,0))

            # Scale the detection threshold for low weight regions.

            # First calculate the median weight
            wht_mask = wht != 0
            med_wht = np.median(wht[wht_mask])

            # Find the ratio of median weight to weight of each pixel.
            ratio = np.where(wht == 0, np.nan, med_wht/wht)

            # Default scaling is 1.
            scaling = np.where(np.isnan(ratio), np.nan, 1)

            # If scaling requested.
            if config['SCALE_THRESH'] != 'None':
                print("Scaling threshold based on weight ratios.")

                # Turn on scaling above the provided threshold.
                above_thresh = ratio > config['SCALE_THRESH']

                # Limit scaling to the 99th percentile of these ratios.
                percentile = np.percentile(ratio[above_thresh], 99)
                ratio_capped = np.minimum(ratio[above_thresh], percentile)

                # Calculate the scaling.                                                                                                                                                                                                    
                scaling[above_thresh] = 1 + (ratio_capped - 1) * (config['SCALE_MAX'] - 1) / (percentile - 1)

            # Ring-median filter the image.
            filtered = self.clipped_ring_median_filter(sci, mask, config)
            
            # Mask sources iteratively in tiers
            bitmask = self.mask_sources(filtered, bitmask, scaling, config, starting_bit = 1)
            mask = (bitmask != 0) 

            # Estimate the background using just unmasked regions
            if config["INTERPOLATOR"] == 'IDW':
                bkg = self.estimate_background_IDW(sci, mask, config)
            else:
                bkg = self.estimate_background(sci, mask, config)
            bkgd = bkg.background

            # Subtract the background
            bkgd_subtracted = sci - bkgd

            bkgd_subtracted = np.where(np.isnan(err), 0, bkgd_subtracted)

            # Evaluate the bias under all sources.
            print("Bias under bright sources:")
            bias, sig = self.evaluate_bias(bkgd, err, mask)
            hdr[f'BIAS_B'] = (bias, 'Bias under all sources.')
            hdr[f'SIG_B'] = (sig, 'Significance of bias under all sources.')


            # And just under fainter sources.
            print("\nBias under fainter sources")
            faintmask = np.zeros(sci.shape, bool)
            for t in (3, 4):
                faintmask = faintmask | (np.bitwise_and(bitmask,2**t) != 0)
            bias, sig = self.evaluate_bias(bkgd, err, faintmask)
            hdr[f'BIAS_F'] = (bias, 'Bias under faint sources.')
            hdr[f'SIG_F'] = (sig, 'Significance of bias under faint sources.')

            
            # Overwrite or create new file.
            if replace_sci == True:
                out_filename = sci_filename
            else:
                out_filename = sci_filename.replace(".fits", f"_{suffix}.fits")

            # Save the file and append tier mask if needed.
            print(f'Saving background subtracted image to {out_filename}...')

            # Add parameters and function used to header.
            hdr['HIERARCH MASK_TYPE'] = 'Individual'
            for (key, value) in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            fits.writeto(out_filename, bkgd_subtracted.astype(np.float32), header = hdr, overwrite = True)

            if store_mask == True:
                # Create header for mask.
                wcs = WCS(hdr)
                hdu_mask = fits.ImageHDU(bitmask.astype(np.int32), header = wcs.to_header(), name = 'TIERMASK')
                # Append as extension to background subtracted file.
                hdul = fits.open(out_filename)
                hdul.append(hdu_mask)
                hdul.writeto(out_filename, overwrite = True)
                hdul.close()
            
            bkgsub_filenames.append(out_filename)

        # Return a list of background subtracted filenames.
        return bkgsub_filenames

    def merged_background(self, science_images, bkgsub_images, parameters={}, WCS_filter=0,
                          suffix=None, merged_name=None):
        """
        Perform background subtraction using a mask merged from multiple 
        images.
        
        Arguments
        ---------
        science_images (List[str])
            Filenames of science images to subtract the background
            from.
        bkgsub_images (List[str])
            Filenames background subtracted images using individual 
            masks.
        parameters (dict)
            Key-value pairs overwritting parameters given in the config 
            file.
        WCS_filter (int):
            Index into science_images. Take the WCS information from this
            image.
        suffix (None, str)
            Suffix to append to the science filenames when saving.
        merged_name (str, None):
            Filename for the output merged source mask.
            If None, don't save.
        """
            
        print('Calculating background using merged mask:')

        # Check that more than one image has been provided.
        if len(science_images) == 1:
            raise KeyError('Only one science image given so not possible to create a merged mask.')
        # Check lists are the same length.
        if len(science_images) != len(bkgsub_images):
            raise KeyError('There should be corresponding images of each type.')
        # Check WCS index is acceptable.
        if (WCS_filter >= len(science_images)) or (WCS_filter < 0):
            raise ValueError(f'WCS_filter should index science images but has value {WCS_filter}'
                             f' for {len(science_images)} images.')

        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', 
                                  stacklevel=2)

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
                this_source_mask = np.left_shift(np.right_shift(input_tiermask, 1), 1)

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

        hduout = fits.PrimaryHDU(mask.astype(np.float32), header = wcs.to_header())
        hduout.writeto(merged_name, overwrite = True)

        # Run final background subtraction on each image using merged
        # mask
        for (image, bkgimage) in zip(science_images, bkgsub_images):
            print(f'Measuring final background for {bkgimage}...')

            # Get tiermask from bgk-subtracted image to get bordermask
            # specific to this image.
            with fits.open(bkgimage) as hdumask:
                bordermask = hdumask['TIERMASK'].data == 1 

            # Combined the merged and border mask.
            merged_mask = fits.getdata(merged_name)
            sourcemask = merged_mask | bordermask
            mask = sourcemask != 0

            # Open the science image and measure the background using the
            # merged mask.
            sci, hdr = fits.getdata(image, header = True)
            wcs = WCS(hdr)

            if config["INTERPOLATOR"] == 'IDW':
                bkg = self.estimate_background_IDW(sci, mask, config)
            else:
                bkg = self.estimate_background(sci, mask, config)
            bkgsub = sci - bkg.background
            bkgsub = np.choose(bordermask, (bkgsub, 0.))

            # Overwrite the original background image.
            print(f'Saving background subtracted image to {bkgimage}...')

            # Add parameters and function used to header.
            hdr['HIERARCH MASK_TYPE'] = 'Merged'
            for (key, value) in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            bkgsub = np.where(sci == 0, 0, bkgsub)

            # If no suffix given, overwrite the original background
            # subtracted image.
            if suffix == None:
                fits.writeto(bkgimage, bkgsub.astype(np.float32), header = hdr, overwrite = True)
            # Otherwise create a new file.
            else:
                fits.writeto(image.replace(".fits", f"_{suffix}.fits"), bkgsub.astype(np.float32),
                             header = hdr, overwrite = True) 

        # Delete merged mask if required.
        if '.temp' in merged_name:
            os.remove(merged_name)

        return        

    def full_background(self, science_images, error_images, weight_images, parameters={},
                        suffix='bkgsub', suffix_merged='mbkgsub', WCS_filter=0, merged_name=None):
        """
        Perform iterative source masking on individual images and 
        measure final background from a merged mask.
        
        Arguments
        ---------
        science_images (List[str])
            Filenames of science images to subtract the background
            from.
        error_images (List[str])
            The corresponding error images.
        weight_images (List[str])
            The corresponding weight images.
        parameters (dict)
            Key-value pairs overwritting parameters given in the config
            file.
        suffix (str)
            Suffix to append to the science filenames when saving
            individual backgrounds.
        suffix_merged (str)
            Suffix to append to the science filenames when saving
            merged backgrounds.
        WCS_filter (int)
            Index into science_images. Take the WCS information from this
            image.
        merged_name (str, None)
            The filepath to save the merged mask to. If None, don't save.
        """

        # Measure the individual backgrounds.
        bkgsub_images = self.individual_background(science_images, error_images, weight_images, 
                                                   parameters, suffix, False, True)

        # Measure the merged background.
        self.merged_background(science_images, bkgsub_images, parameters, WCS_filter,
                               suffix_merged, merged_name)

        return