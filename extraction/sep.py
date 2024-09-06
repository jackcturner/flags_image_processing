import os
import copy
import h5py
import yaml

import numpy as np

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

import sep

class SEP():
    """
    Class for running SEP in dual or single image mode and performing
    Kron aperture photometry.
    """

    def __init__(self, config_file):
        """__init__ method for SEP.

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

        self.translate = {'a':'A_IMAGE', 'b':'B_IMAGE', 'theta':'THETA_IMAGE', 'flux':'FLUX_ISO',
                          'cflux':'cflux', 'peak':'FLUX_MAX', 'cpeak':'c_peak', 'cxx':'CXX_IMAGE',
                          'cxy':'CXY_IMAGE', 'cyy':'CYY_IMAGE', 'errx2':'ERRCXX_IMAGE',
                          'errxy':'ERRCXY_IMAGE', 'erry2':'ERRCYY_IMAGE', 'flag':'FLAGS',
                          'npix':'npix', 'tnpix':'tnpix', 'thresh':'THRESHOLD', 'x':'X_IMAGE',
                          'y':'Y_IMAGE', 'x2':'X2_IMAGE', 'y2':'Y2_IMAGE', 'xy':'XY_IMAGE',
                          'xpeak':'XPEAK_IMAGE', 'ypeak':'YPEAK_IMAGE', 'xcpeak':'xcpeak',
                          'ycpeak':'ycpeak', 'xmax':'XMAX_IMAGE', 'xmin':'XMIN_IMAGE',
                          'ymax':'YMAX_IMAGE', 'ymin':'YMIN_IMAGE','FLUX_AUTO':'FLUX_AUTO',
                          'FLUXERR_AUTO':'FLUXERR_AUTO', 'FLUX_FLAGS':'FLUX_FLAGS',
                          'ALPHA_SKY':'ALPHA_SKY', 'DELTA_SKY':'DELTA_SKY'}

    def measure_background(self, sci, config):
        """
        Measure the background of an image using SEP functionality.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        config (dict)
            Dictionary of background configration arguments.

        Returns
        -------
        bkg (sep.Background)
            The measured SEP background object.
        """

        print('Estimating background...')

        # Load a source mask if provided.
        if config['background_mask'] != None:
            mask = fits.getdata(config['background_mask'])
            mask = mask.byteswap(inplace=True).newbyteorder()
        else:
            mask = None
        bkg = sep.Background(sci, mask, config['background_maskthresh'], config['bw'], config['bh'],
                              config['fw'], config['fh'], config['fthresh'])

        return bkg
                
    def detect_sources(self, sci, hdr, config, weight=None, bkg=None, outdir = './'):
        """
        Use SEP to identify sources in a 2D image.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image from which to identify sources.
        hdr (astropy.io.fits.header.Header)
            The WCS header of the science image.
        config (dict)
            Dictionary of source detection configuration arguments.
        weight (None, str)
            The path to the map for detection weighting.
        bkg (None, sep.Background)
            The background map of sci if already computed.
        outdir (str)
            The directory in which to save the segmentation map.

        Return
        ------
        cat (astropy.table.table.Table)
            Table storing information on detected sources.
        segmap (numpy.ndarray)
            2D image indicating the locations of detected sources.
        """
        print('Detecting sources...')

        # Open the error image if provided.
        if weight != None:
            wht = fits.getdata(weight)
            wht = wht.byteswap(inplace=True).newbyteorder()
            err = np.where(wht == 0, np.nan, 1/np.sqrt(wht))

        # Else use the global background rms.
        else:
            print('No error map provided, using global background RMS for detection threshold.')
            # Calculate if required.
            if bkg == None:
                bkg = self.measure_background(sci, config)
                err = bkg.globalrms
            else:
                err = bkg.globalrms

        # Set up the detection mask.
        if config['detection_mask'] != None:
            # Use the error map.
            if config['detection_mask'] == 'error':
                if weight != None:
                    mask = np.isnan(err)
                else:
                    raise KeyError("No error map provided but detection_mask == 'error'.")
            # Or a provided file.
            else:
                mask = fits.getdata(mask)
                mask = mask.byteswap(inplace=True).newbyteorder()
        else:
            mask = None

        # Set up the filter_kernel.
        if config['filter_kernel'] != None:
            with open(config['filter_kernel'], "r") as file:
                next(file)
                kernel = np.loadtxt(file)
        else:
            kernel = None
        
        # Set some memory limits.
        sep.set_extract_pixstack(config['pixstack'])
        sep.set_sub_object_limit(config['object_limit'])

        # Do the extraction.
        objects, segmap = sep.extract(sci, thresh = config['thresh'], err = err, gain = config['gain'], mask = mask, 
                              maskthresh = config['detection_maskthresh'], minarea = config['minarea'], filter_kernel = kernel,
                              filter_type = config['filter_type'], deblend_nthresh = config['deblend_nthresh'],
                              deblend_cont = config['deblend_cont'], clean = config['clean'], clean_param = config['clean_param'],
                              segmentation_map = True)
        print('Done!')
        
        # Save the segmentation map if requested.
        if config['segmentation_map'] != None:
            print(f'Saving segmentation map to {outdir}/{config["segmentation_map"]}')
            fits.writeto(f'{outdir}/{config["segmentation_map"]}', segmap, hdr, overwrite = True)

        cat = Table(objects)

        # Extract can produce theta values > pi/2, so we need to correct these before performing photometry.
        invalid = cat['theta'] > np.pi / 2
        # Subtract pi from the values where the condition is True
        cat['theta'][invalid] -= np.pi

        # Calculate RA and DEC.
        wcs = WCS(hdr)
        coordinates = pixel_to_skycoord(cat['x'], cat['y'], wcs)
        cat['ALPHA_SKY'] = coordinates.ra.degree
        cat['DELTA_SKY'] = coordinates.dec.degree

        return cat, segmap
    
    def measure_photometry(self, sci, config, segmap, cat, weight=None, bkg=None):
        """
        Measure the photometry of detected sources using Kron apertures.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image from which to identify sources.
        config (dict)
            Dictionary of source detection configuration arguments.
        segmap (numpy.ndarray)
            2D image indicating the locations of detected sources.
        cat (astropy.table.table.Table)
            Table storing information on detected sources.
        weight (None, str)
            The path to the map for detection weighting.
        bkg (None, sep.Background)
            The background map of sci if already computed.

        Return
        ------
        flux (numpy.ndarray)
            The flux of the detected sources in image counts.
        fluxerr (numpy.ndarray)
            The corresponding flux error.
        flag (numpy.ndarray)
            Flag indicating the quality of the measured photometry.
        """
        print('Measuring kron photometry...')

        # Open the error image if provided.
        if weight != None:
            wht = fits.getdata(weight)
            wht = wht.byteswap(inplace=True).newbyteorder()
            err = np.where(wht == 0, np.nan, 1/np.sqrt(wht))

        # Else use the global background rms.
        else:
            print('No error map provided, using global background RMS for detection threshold.')
            # Calculate if required.
            if bkg == None:
                bkg = self.measure_background(sci, config)
                err = bkg.globalrms
            else:
                err = bkg.globalrms

        # Set up the detection mask.
        if config['detection_mask'] != None:
            # Use the error map.
            if config['detection_mask'] == 'error':
                if weight != None:
                    mask = np.isnan(err)
                else:
                    raise KeyError("No error map provided but detection_mask == 'error'.")
            # Or a provided file.
            else:
                mask = fits.getdata(mask)
                mask = mask.byteswap(inplace=True).newbyteorder()
        else:
            mask = None

        # The type of masking requested for kron flux.
        if (config['mask_type'] == None) or (config['mask_type'] == 'NONE'):
            seg_id = None
            seg = None
        elif config['mask_type'] == 'BLANK':
            seg_id = np.arange(1, len(cat)+1, dtype=np.int32)
            seg = segmap
        elif config['mask_type'] == 'SEGMENT':
            seg = segmap
            seg_id = np.arange(1, len(cat)+1, dtype=np.int32) * -1
        else:
            raise KeyError(f"mask_type {config['mask_type']} not recognised. Use 'NONE', 'BLANK' or 'SEGMENT'.")
        
        # Calculate the kron flux.
        kronrad, krflag = sep.kron_radius(sci, cat['x'], cat['y'], cat['a'], cat['b'], cat['theta'], config['int_radius'], 
                                          mask = mask, maskthresh = config['detection_maskthresh'], segmap = seg, seg_id = seg_id)
        flux, fluxerr, flag = sep.sum_ellipse(sci, cat['x'], cat['y'], cat['a'], cat['b'], cat['theta'], config['kron_factor']*kronrad, 
                                              err = err, mask = mask, maskthresh = config['detection_maskthresh'], segmap = seg, seg_id = seg_id, 
                                              gain = config['gain'], subpix=config['subpix'])
        flag |= krflag

        # Use circular aperture if below minimum radius.
        r_min = config['min_radius']
        use_circle = kronrad * np.sqrt(cat['a'] * cat['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(sci, cat['x'], cat['y'], r_min, err = err, mask = mask, 
                                                maskthresh = config['detection_maskthresh'], segmap = seg, seg_id = seg_id, gain = config['gain'], 
                                                subpix=config['subpix'])
        flux[use_circle] = cflux[use_circle]
        fluxerr[use_circle] = cfluxerr[use_circle]
        flag[use_circle] = cflag[use_circle]

        return flux, fluxerr, flag

    
    def extract(self, images, weights=None, parameters = {}, outputs=None, outdir='./'):
        """
        Main function for extracting sources and measuring photometry 
        in a science image.
        
        Arguments
        ---------
        images (str, List[str])
            If string, path to single image from which to detect and 
            measure sources. If List[str], path to detection image as the 
            first entry and measurement as the second.
        weights (None, str, List[str])
            The corresponding weight images for detection.
            If None, use global background RMS for weighting.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (None, List[str])
            The source extraction outputs to save to the catalogue.
            If None, save all available.
        outdir (str)
            The directory in which to store output files.
        """

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

        # Process the science images in advance.
        # Convert to list if not already.
        if isinstance(images, list):
            sci_imgs = images
        else:
            sci_imgs = [images]
        
        # Store headers and measured backgrounds.
        hdrs = []
        bkgs = [None, None]
        for i, img in enumerate(sci_imgs):
            # Read in the image.
            sci, hdr = fits.getdata(img, header=True)
            sci = sci.byteswap(inplace=True).newbyteorder()

            # Remove the background if requested.
            if config['background_sub'] == True:
                bkg = self.measure_background(sci, config)
                sci = sci - bkg
                bkgs[i] = bkg
            sci_imgs[i] = sci
            hdrs.append(hdr)

        # Now work out the type of extraction.

        # Two image with weights.
        if (len(sci_imgs) == 2) and (type(weights) == list):
            print('Starting in dual image mode with weighting.')
            cat_name = f'{outdir}/{os.path.basename(images[1]).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, weights[0], outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[1], config, segmap, cat, weights[1])
        # Two image without weights.
        elif (len(sci_imgs) == 2) and (type(weights) == type(None)):
            print('Starting in dual image mode.')
            cat_name = f'{outdir}/{os.path.basename(images[1]).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, bkg = bkgs[0], outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[1], config, segmap, cat, bkg = bkgs[1])
        # Single image without weight.
        elif (len(sci_imgs) == 1) and (type(weights) == type(None)):
            print('Starting in single image mode.')
            cat_name = f'{outdir}/{os.path.basename(images).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, bkg = bkgs[0], outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[0], config, segmap, cat, bkg = bkgs[0])
        # Single image with weight.
        elif (len(sci_imgs) == 1) and (type(weights) == str):
            print('Starting in single image mode with weighting.')
            cat_name = f'{outdir}/{os.path.basename(images).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, weights, outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[0], config, segmap, cat, weights)

        # # Add these measurements to the catalogue.
        cat['FLUX_AUTO'] = flux
        cat ['FLUXERR_AUTO'] = fluxerr
        cat['FLUX_FLAGS'] = flag

        # Convert to desired flux from counts.
        flux_columns = ['FLUX_AUTO', 'FLUXERR_AUTO', 'cflux', 'flux', 'cpeak', 'peak']
        for name in cat.colnames:
            if name in flux_columns:
                cat[name] = cat[name] * config['flux_conversion']

        # Now add everything to the hdf5 catalogue.
        with h5py.File(cat_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')

            # If no outputs requested, use all.
            if outputs == None:
                outputs = cat.colnames
            for column in cat.colnames:
                if column in outputs:
                    f[f'photometry/{self.translate[column]}'] = cat[column]

            # Add the config parameters as attributes.
            for key,value in att_config.items():
                f['photometry'].attrs[key] = value
            
        print(f'All done and saved to {cat_name}')

        return