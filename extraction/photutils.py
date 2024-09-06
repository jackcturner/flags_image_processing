import os
import copy
import yaml
import h5py

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma, SigmaClip
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft

import photutils.background as pb
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog

class photuitls():

    def __init__(self, config_file):
        """__init__ method for photutils.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        """

        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config = content[0]

        self.output_names = [
            'area', 'background_centroid', 'background_mean', 'background_sum', 'bbox_xmax',
              'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'centroid','centroid_quad', 'centroid_win', 
              'covar_sigx2', 'covar_sigy2', 'covariance', 'covariance_eigvals', 'cutout_centroid', 
              'cutout_centroid_quad', 'cutout_centroid_win', 'cutout_maxval_index', 
              'cutout_minval_index', 'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 
              'elongation', 'equivalent_radius', 'fwhm', 'gini', 
              'inertia_tensor', 'isscalar', 'kron_aperture', 'kron_flux', 'kron_fluxerr', 
              'kron_radius', 'label', 'labels', 'local_background', 'local_background_aperture', 
              'max_value', 'maxval_index', 'maxval_xindex', 'maxval_yindex', 'min_value', 
              'minval_index', 'minval_xindex', 'minval_yindex', 'moments', 'moments_central', 
              'nlabels', 'orientation', 'perimeter', 'segment_area', 'segment_flux', 
              'segment_fluxerr', 'semimajor_sigma', 'semiminor_sigma', 'skybbox_ll', 'skybox_lr', 
              'skybox_ul', 'skybox_ur', 'sky_centroid', 'sky_centroid_icrs', 'sky_centroid_quad', 
              'sky_centroid_win', 'slices', 'xcentroid', 'xcentroid_quad', 'xcentroid_win', 
              'ycentroid', 'ycentroid_quad', 'ycentroid_win']

    def measure_background(self, sci, wht, config):

        # The interpolation, background and RMS estimators.
        interpolators = {'IDW':pb.BkgIDWInterpolator(), 'Zoom':pb.BkgZoomInterpolator()}
        back_est = {'Mean':pb.MeanBackground(), 'Median':pb.MedianBackground(), 
                    'Mode':pb.ModeEstimatorBackground(),'MMM':pb.MMMBackground(),
                    'SExtractor':pb.SExtractorBackground(),
                    'BiweightLocation':pb.BiweightLocationBackground()}
        rms_est = {'Std':pb.StdBackgroundRMS(), 'MADStd':pb.MADStdBackgroundRMS(), 
                   'BiweightScale':pb.BiweightScaleBackgroundRMS()}
        
        # Mask off detector regions.
        coverage_mask = wht == 0

        # Mask sources if provided.
        if config['SOURCE_MASK'] != None:
            mask = fits.getdata(config['SOURCE_MASK'])
        else:
            mask = None

        # Get the sigma clipping object.
        if config['SIGMA_CLIP'] == True:
            sigma_clip = SigmaClip(
                sigma_lower=config['SIGMA'][0], sigma_upper=config['SIGMA'][1], 
                maxiters=config['MAX_ITERS'])
        else:
            sigma_clip = None

        # Get the interpolation, background and RMS estimators.
        bkg_estimator = back_est.get(config['BACK_ESTIMATOR'])
        bkgrms_estimator = rms_est.get(config['RMS_ESTIMATOR'])
        interpolator = interpolators.get(config['INTERPOLATOR'])

        # Calculate the 2D background.
        print('Measuring the 2D sky background...')
        bkg = pb.Background2D(
            sci, box_size=config['BOX_SIZE'], mask=mask, coverage_mask=coverage_mask, fill_value=0,
            exclude_percentile=config['EXCLUDE_PERCENTILE'], filter_size=config['FILTER_SIZE'],
            filter_threshold=config['FILTER_THRESH'], edge_method=config['EDGE_METHOD'], 
            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, bkgrms_estimator=bkgrms_estimator,
            interpolator=interpolator)
        
        # Subtract the background if requested.
        if config['BKG_SUB'] == True:
            print('Removed background.')
            sci = sci - bkg.background
        else:
            print('Image is already background subtracted.')
        
        return sci, bkg
    
    def filter(self, sci, wht, bkg, config):

        if config['FILTER'] != None:

            # Replace off detector regions with median background so 
            # convolution doesn't smear them.
            sci = np.where(wht == 0, bkg.background_median, sci)

            # Generate kernel based on provided FWHM and convolve.
            if config['FILTER'] == 'Gaussian':
                print('Applying Gaussian detection filter...')
                sigma = config['FWHM'] * gaussian_fwhm_to_sigma
                kernel = Gaussian2DKernel(sigma, x_size=config['SIZE'], y_size=config['SIZE'])
                sci = convolve_fft(sci, kernel, boundary='fill', fill_value=bkg.background_median,
                                   nan_treatment='interpolate', preserve_nan=True, allow_huge=True)
            elif config['FILTER'] == 'Tophat':
                print('Applying Tophat detection filter...')
                sigma = config['FWHM'] / np.sqrt(2)
                kernel = Tophat2DKernel(sigma, x_size=config['SIZE'], y_size=config['SIZE'])
                sci = convolve_fft(sci, kernel, boundary='fill', fill_value=bkg.background_median,
                                   nan_treatment='interpolate', preserve_nan=True, allow_huge=True)
            else:
                raise ValueError('Kernel not supported: {}'.format(config['FILTER']))
            
            # Revert to zeros in the off detector region.
            sci = np.where(wht == 0, 0, sci)
                
        return sci
    
    def segmentation(self, sci, wht, bkg, config):

        # Calculate the detection threshold.
        # Use the computed or provided RMS map.
        if config['RMS_MAP'] == None:
            print('Using internal background RMS map for detection.')
            rms = bkg.background_rms
        else:
            print('Using external background RMS map for detection.')
            rms = fits.getdata(config['RMS_MAP'])

        ##TODO: Use weight information when calculating the threshold.
        threshold = (config['N_SIGMA'] * rms) + bkg.background

        # Mask the image edges.
        mask = wht == 0

        # Generate the segmentation image
        print('Detecting sources...')
        seg_image = detect_sources(sci, threshold=threshold, npixels=config['N_PIXELS'],
                                    connectivity=config['CONNECTIVITY'], mask = mask)
        
        # and then deblend it.
        print('Deblending sources...')
        seg_image = deblend_sources(sci, seg_image, config['N_PIXELS'], nlevels=config['N_LEVELS'],
                                    contrast=config['CONTRAST'], mode=config['MODE'],
                                    connectivity=config['CONNECTIVITY'], relabel=True,
                                    nproc=1, progress_bar=False)
        
        return seg_image
    
    def extract(self, sci_filename, wht_filename, err_filename, parameters = {}, outputs = None, outdir = './'):

        # Make a local copy of the config for updating with provided parameters.
        config = copy.deepcopy(self.config)

        # Update the config with given parameters.
        for key in list(parameters.keys()):
            config[key] = parameters[key]

        # Store the config as is for saving as hdf5 attributes.
        att_config = copy.deepcopy(config)

        # Expand any environment variables and convert string to None.
        for key, value in config.items():
            if type(value) == str:
                config[key] = os.path.expandvars(value)
            if value == 'None':
                config[key] = None

        # Determine if single or double image mode is being used.
        if type(sci_filename) != list:
            print(f'Single image mode. \n D/M: {os.path.basename(sci_filename)}')
            sci_filename = [sci_filename]
            wht_filename = [wht_filename]
            err_filename = [err_filename]
            cat_name = f'{os.path.basename(sci_filename[0]).removesuffix(".fits")}_photutils.hdf5'
        else:
            print(f'Double image mode. \n D: {os.path.basename(sci_filename[0])} \n M: {os.path.basename(sci_filename[1])}')
            cat_name = f'{os.path.basename(sci_filename[1]).removesuffix(".fits")}_photutils.hdf5'

        # First load the detection images.
        sci_d, hdr_d = fits.getdata(sci_filename[0], header=True)
        wht_d = fits.getdata(wht_filename[0])
        err_d = fits.getdata(err_filename[0])

        # Measure background and filter.
        sci_d, bkg_d = self.measure_background(sci_d, wht_d, config)
        sci_d_filt = self.filter(sci_d, wht_d, bkg_d, config)

        # Generate the segmentation map,
        seg_map = self.segmentation(sci_d_filt, wht_d, bkg_d, config)

        # and save if requested.
        if config['SEGMAP'] != None:
            fits.writeto(config['SEGMAP'], seg_map, header=hdr_d)

        # If single image mode measurement is detection.
        if len(sci_filename) == 1:
            # So assign the same properies.
            sci_m = sci_d
            hdr_m = hdr_d
            wht_m = wht_d
            err_m = err_d
            sci_m_filt = sci_d_filt
            bkg_m = bkg_d

        # If double, need to background subtract and filter.
        elif len(sci_filename) == 2:

            # Load the measurement images.
            sci_m, hdr_m = fits.getdata(sci_filename[1], header=True)
            wht_m = fits.getdata(wht_filename[1])
            err_m = fits.getdata(err_filename[1])

            # Measure background
            sci_m, bkg_m = self.measure_background(sci_m, wht_m, config)

            # and only filter if required.
            sci_m_filt = None
            if config['CONVOLVED'] == True:
                sci_m_filt = self.filter(sci_m, wht_m, bkg_m, config)
        else:
            raise ValueError('Incorrect input path shapes.')

        # Get the WCS information from the header.
        wcs = WCS(hdr_m)

        # Mask the off detector regions.
        mask = wht_m == 0

        # Should convolved data be used to measure properties?
        if config['CONVOLVED'] == True:
            convolved_data = sci_m_filt
        else:
            convolved_data = None

        # Measure the photometry.
        print('Measuring source properties...')
        cat = SourceCatalog(
            sci_m, seg_map, convolved_data=convolved_data, error=err_m, mask=mask,
            background=bkg_m.background, wcs=wcs, localbkg_width=config['LOCALBKG_WIDTH'],
            apermask_method=config['APERMASK_METHOD'], kron_params=config['KRON_PARAMS'],
            detection_cat=None, progress_bar=False)
        
        # Measure circular aperture photometry if requested.
        labels = []
        for idx, radius in enumerate(config['RADII']):
            cat.circular_photometry(radius, f'APER_{idx}', overwrite=False)
            labels += [f'APER_{idx}_flux', f'APER_{idx}_fluxerr']
        
        # Get the full list of avilable outputs.
        output_names = self.output_names + labels

        # Now add everything to the hdf5 catalogue.
        with h5py.File(f'{outdir}/{cat_name}', 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')

            # If no outputs requested, use all.
            if outputs == None:
                outputs = output_names

            # Add as a dataset, translating names to SE standard.
            for column in output_names:
                print(column)
                if column in outputs:
                    f[f'photometry/{column}'] = getattr(cat, column)

            # Add the config parameters as attributes.
            for key,value in att_config.items():
                f['photometry'].attrs[key] = value