import os
import subprocess
import re
import copy
import warnings
import h5py
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

from astropy.table import Table
from astropy.io import ascii, fits
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

import emcee

import photutils.background as pb
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.utils import ImageDepth

import sep

from utils import measure_curve_of_growth

class SExtractor():
    """
    Wrap SExtractor while maintaining the majority of its functionality
    and producing hdf5 catalogues compatible with the FLAGS pipeline.
    """

    def __init__(self, config_file, sexpath='sex'):
        """
        __init__ method for SExtractior.

        Arguments
        ---------
            config_file (str)
                Path to ".yml" configuration file.
            sexpath (str)
                Path to Source Extractor executable.
        """

        # Read the SE configuration file and split into SE and genral
        # config parts.
        self.configfile = config_file
        with open(self.configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.SEconfig, self.config = content

        # Catalogue type is fixed.
        self.SEconfig['CATALOG_TYPE'] = 'ASCII_HEAD'

        # The path to the SExtractor executable.
        self.sexpath = sexpath     

    # Generate the default SExtractor configuration file if none given.
    def generate_default(self, outdir):
        """
        Generate the default SExtractor configuration file.

        Arguments
        ---------
        outdir (str)
            Directory in which to save the configuration file.

        Returns
        -------
        config_path (str)
            Path to the generated configuration file.
        """

        # Pipe the SExtractor output.
        p = subprocess.Popen([self.sexpath, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Save the file.
        config_path = f'{outdir}/default.sex'
        f = open(config_path, 'w')
        f.write(out.decode(encoding = 'UTF-8'))
        f.close()

        return config_path
    
    def get_version(self):
        """
        Retrieve the SExtractor version.

        Returns
        -------
        version (str)
            SExtractor version number used to initalise class.
        """
        
        # Run SExtractor with no inputs.
        p = subprocess.Popen([self.sexpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Search the outputs for the version number.
        version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode(encoding='UTF-8'))
        
        # Raise error if no version found.
        if version_match is False:
            raise RuntimeError("Could not determine SExctractor version, check the output of"
                               " running '%s'" % (self.sexpath))
        
        version = str(version_match.group()[8:])
        assert len(version) != 0

        return version	
     
    def write_params(self, params, outdir):
        """
        Write output parameters in a list to a text file in Sextractor 
        format.

        Returns
        -------
        parameters_filename (str)
            Path to the generated parameter file.
        """

        parameter_filename = f'{outdir}/temporary_parameters.temp.params'
        f = open(parameter_filename, 'w')
        f.write("\n".join(params))
        f.write("\n")
        f.close()

        return parameter_filename
    
    def convert_to_flux(self, cat, config):
        """
        Convert catalogue flux counts to desired unit.

        Arguments
        ---------
        cat (str)
            Path to catalogue file with fluxes to be converted.
        config (dict)
            Config dictonary containing the conversion factor.
        """

        # Return if no conversion required.
        if config['TO_FLUX'] == 0:
            return
        
        print('Converting to flux...')

        catalogue = ascii.read(cat)
        # Search for any flux columns and apply conversion.
        for column in catalogue.colnames:
            if 'FLUX' in column:
                catalogue[column] = catalogue[column] * config['TO_FLUX']
        catalogue.write(cat, format = 'ascii', overwrite = True)

        return
    
    def convert_to_hdf5(self, catalogue, config):
        """
        Converts an SExtractor ascii catalogue to HDF5.

        Arguments
        ---------
        catalogue (str)
            Path to catalogue file to be converted.
        config (dict)
            Config dictonary containing all parameters to be added as 
            attributes.

        Returns
        -------
        hdf5_name (str)
            Path to the generated hdf5 file.
        """

        # Read the ascii catalogue.
        cat = Table.read(catalogue, format = 'ascii')

        hdf5_name = f'{catalogue.removesuffix(".cat")}.hdf5'

        # Create HDF5 file with the same name.
        with h5py.File(hdf5_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')
            for column in cat.colnames:
                f[f'photometry/{column}'] = cat[column]
            
            for key in config:
                f['photometry'].attrs[key] = config[key]
            f['photometry'].attrs['METHOD'] = 'SExtractor'
            f['photometry'].attrs['VERSION'] = self.get_version()

        # Delete the original ascii catalogue.
        os.remove(catalogue)
        
        return hdf5_name
    
    def run_SExtractor(self, basecmd, SEconfig):
        """
        Passes a command to SExtractor on the command line.

        Arguments
        ---------
        basecmd (str)
            String containing the base (file, image, weight) command
            line arguments.
        SEconfig (dict)
            Additional arguments from the SExtractor config file to be 
            added.
        """

        SEcmd = copy.deepcopy(basecmd)

        # Add parameters given in the config file to the base command.
        for (key, value) in SEconfig.items():
            SEcmd.append("-" + str(key))
            SEcmd.append(str(value).replace(' ',''))

        # Run SExtractor and print the outputs.
        p = subprocess.Popen(SEcmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        for line in p.stderr:
            print(line.decode(encoding = "UTF-8"))
        out, err = p.communicate()

        return
        
    def get_aperture_config(self, SEconfig):
        """
        Return an updated config file with parameters appropriate for
        measuring in apertures around set locations.
        
        Arguments
        ---------
        SEconfig (dict)
            Dictonary containing the SExtractor config to be updated.
        
        Returns
        -------
        Seconfig (dict)
            The updated config file.
        """

        SEconfig['DETECT_MINAREA'] = 1
        SEconfig['DETECT_THRESH'] = 1E-12
        SEconfig['FILTER'] = 'N'
        SEconfig['CLEAN'] = 'N'
        SEconfig['MASK_TYPE'] = 'NONE'
        SEconfig['BACK_TYPE'] = 'MANUAL'
        SEconfig['BACK_VALUE'] = 0.0
        SEconfig['CHECKIMAGE_TYPE'] = 'NONE'

        return SEconfig
    
    def get_aperture_locations(self, sci, hdr, mask, radius, napers=10000, overlap=False,
                               overlap_maxiters=50000, outname='aperture_image.fits'):
        """
        Create a detection image with value one at random aperture
        centres.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image in which to place the apertures.
        hdr (astropy.io.fits.Header)
            The header containing image WCS information.
        mask (numpy.ndarray)
            The 2D science image source mask.
        radius (float)
            The radius in pixels of the apertures to place.
        napers (int)
            The maximum number of apertures to place.
        overlap (bool)
            Should the apertures be allowed to overlap.
        overlap_maxiters (int)
            The number of attempts at placing a non-overlapping aperture.
        outname (str)
            The name of the output detection image file.
        
        Returns
        -------
        outname (str)
            The name of the generated detection image file.
        """

        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma = 1.0, napers = napers, niters = 1, overlap = False,
                           overlap_maxiters = overlap_maxiters)
        limits = depth(sci, mask)
        print(f'Placed {int(depth.napers_used)} apertures.')

        # Get the location of the apertures.
        locations = depth.apertures[0].positions

        # Construct the detection image
        det = np.zeros(sci.shape)
        for i in np.round(locations).astype(int):
            det[i[1], i[0]] = 1

        # and save it.
        fits.writeto(outname, det, header = hdr, overwrite = True)

        return outname

    def measure_depth(self, sci_filename, err_filename, mask_filename, radius, psf_filename, 
                      max_apers=50, max_iters=50000, outdir='./'):
        """
        Use randomly placed apertures to measure the 5-sigma min, max
        and average depth of an image.
        
        Arguments
        ---------
        sci_filename (str)
            Filename of science fits image.
        err_filename (str)
            Filename of the coressponding error image.
        mask_filename (str)
            Filename of the source mask fits image.
        radius (float)
            Radius of the random apertures to use in pixels.
        psf_filename (str)
            Filename of the PSF fits image.
        max_apers (int)
            The maximum number of apertures to place.
        max_iters (int)
            The maximun attempts at finding a non overlapping location.
        outdir (str)
            The directory in which to save temporary files.
        
        Retruns
        -------
        depth_5 (float)
            The 5-sigma depth of the image.
        min (float)
            The minimum measured depth.
        max (float)
            The maximum measured depth.
        """
        
        # Get the config parameters.
        SEconfig_depth = copy.deepcopy(self.SEconfig)
        config_depth = copy.deepcopy(self.config)

        SEconfig_depth = self.get_aperture_config(SEconfig_depth)

        # Open the science and error images
        sci, hdr = fits.getdata(sci_filename, header=True)
        err = fits.getdata(err_filename)
        # and the mask.
        source_mask = fits.getdata(mask_filename)
        mask = (source_mask != 0) + (np.isnan(err))

        # Place random apertures and generate detection image.
        det_filename = f'{outdir}det_apertures.temp.fits'
        self.get_aperture_locations(sci, hdr, mask, radius, max_apers, overlap_maxiters = max_iters,
                                     outname = det_filename)

        # Generate default SExtractor config file.
        sexfile = self.generate_default(outdir)

		# Build the base SE command line argument.
        detcmd = [self.sexpath, "-c", sexfile, det_filename, sci_filename, '-WEIGHT_IMAGE',
                  err_filename]

        # Add the correct name and aperture diameters to the command line arguments.
        SEconfig_depth['CATALOG_NAME'] = f'{outdir}det_apertures.temp.cat'
        SEconfig_depth['WEIGHT_TYPE'] = 'MAP_RMS'
        SEconfig_depth['PHOT_APERTURES'] = str(round(radius*2, 2))

        # Write the output parameters to a text file Just need aperture flux and number.
        parameter_filename = f'{outdir}/depth.temp.params'
        f = open(parameter_filename, 'w')
        f.write(f'FLUX_APER') 
        f.write("\n")
        f.write(f'NUMBER') 
        f.write("\n")
        f.close()
        SEconfig_depth['PARAMETERS_NAME'] = parameter_filename

        # Run SE with with the base command and config.
        self.run_SExtractor(detcmd, SEconfig_depth)

        # Remove temporary parameter file and detection image.
        os.remove(parameter_filename)
        os.remove(det_filename)

        # Open the catalogue.
        apps = ascii.read(SEconfig_depth['CATALOG_NAME'])

        # Scale to nJy.
        apps['FLUX_APER'] *= config_depth["TO_FLUX"]

        # Measure the median absolute deviation.
        s = (apps['FLUX_APER'] != 0) & (~np.isnan(apps['FLUX_APER']))
        mad = median_abs_deviation(apps['FLUX_APER'][s], nan_policy = 'omit') * 1.48

        # Measure the PSF curve of growth and interpolate.
        psf = fits.getdata(psf_filename)
        radii = np.arange(0.1, psf.shape[0], 1)
        radii, cog, p = measure_curve_of_growth(psf, radii = radii, position = None, norm = False,
                                                show = False)
        f = lambda r: np.interp(r, radii, cog)

        # Correct by the fraction of the PSF enclosed within the aperture
        # used and convert to 5 sigma.
        mad *= 5/f(radius)

        # Also calculate the minimum and maximum depths.
        max = 5*np.min(apps['FLUX_APER'][s])/f(radius)
        min = 5*np.max(apps['FLUX_APER'][s])/f(radius)

        # Assuming units of nJy, calculate the 5 sigma limit in
        # magnitudes.
        depth_5 = -2.5 * np.log10((mad*1e-9) / 3631)
        max = -2.5 * np.log10((max*1e-9) / 3631)
        min = -2.5 * np.log10((min*1e-9) / 3631)

        # Delete the SExtractor catalog.
        os.remove(SEconfig_depth['CATALOG_NAME'])

        return depth_5, min, max
    
    def measure_uncertainty(self, sci_filename, err_filename, seg_filename, SEconfig, config,
                            outdir):
        """
        Perform empirical uncertainty estimation based on Finkelstein+23.

        Arguments
        ---------
        sci_filename (str)
            Filename of the science image.
        err_filename (str)
            Filename of the corresponding error image.
        seg_filename (str)
            Path to the segmentation file produced by SExtractor.
        SEconfig (dict)
            SExtractor configuration parameters specific to this image.
        config (dict)
            Additional parameters specific to this image.
        outdir (str)
            The directory in which to save outputs.
        """

        print('\nBeginning uncertainty estimation:')

        # Open the image files.
        sci, hdr = fits.getdata(sci_filename, header = True)
        err = fits.getdata(err_filename)
        seg = fits.getdata(seg_filename)

        # Create a copy of the configuration parameters specific to the
        # image.
        err_SEconfig = copy.deepcopy(SEconfig)
        err_config = copy.deepcopy(config)

        # Set some values to those appropriate for uncertainty 
        # estimation.
        err_SEconfig = self.get_aperture_config(err_SEconfig)

        # Mask positive areas of the segmentation map and invalid areas
        # in the error map.
        mask = (seg > 0) + np.isnan(err) + (err < 0) + (err > 1000)

        # Get the range of aperture radii.
        if err_config['RADII_SPACING'] == 'linear':
            radii = np.linspace(err_config['MIN_RADIUS'], err_config['MAX_RADIUS'], 
                                err_config['N_RADII'])
        else:
            radii = np.logspace(np.log10(err_config['MIN_RADIUS']),
                                np.log10(err_config['MAX_RADIUS']), err_config['N_RADII'])

        # Seperate the radii into small and large components based on the
        # median value.
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        print('Getting aperture locations...')

        # Get locations of small and large aperture iterations.
        small_filename = f'{outdir}/small_apertures.temp.fits'
        self.get_aperture_locations(sci, hdr, mask, max(radii[smaller]), err_config['N_SMALL'],
                                    False, err_config['MAX_ITERS'], small_filename)
 
        large_filename = f'{outdir}/large_apertures.temp.fits'
        self.get_aperture_locations(sci, hdr, mask, max(radii), err_config['N_LARGE'],
                                    False, err_config['MAX_ITERS'], large_filename)

        # Generate a default configuration file.
        sexfile = self.generate_default(outdir)

		# Build the base command line arguments for small and large 
        # aperture runs.
        smallcmd = [self.sexpath, "-c", sexfile, small_filename, sci_filename,'-WEIGHT_IMAGE',
                    err_filename]
        largecmd = [self.sexpath, "-c", sexfile, large_filename, sci_filename, '-WEIGHT_IMAGE',
                    err_filename]

        # Add the correct name and aperture diameters to the command line
        # arguments for the small run.
        err_SEconfig['CATALOG_NAME'] = f'{outdir}/small_apertures.temp.cat'
        apertures = ''
        for radius in radii[smaller]:
            apertures += str(round(radius*2, 2)) + ','
        apertures = apertures[:-1]
        err_SEconfig['PHOT_APERTURES'] = apertures

        # Write the output parameter file.
        parameter_filename = self.write_params([f'FLUX_APER({sum(smaller)})'], outdir)
        err_SEconfig['PARAMETERS_NAME'] = parameter_filename

        # Run SE on the smaller apertures.
        print('Running SExtractor on the small apertures...')
        self.run_SExtractor(smallcmd, err_SEconfig)

        # Remove temporary parameter file and detection image.
        os.remove(parameter_filename)
        os.remove(small_filename)

        # Add the correct name and aperture diameters to the command line
        # arguments for the large run.
        err_SEconfig['CATALOG_NAME'] = f'{outdir}/large_apertures.temp.cat'
        apertures = ''
        for radius in radii[larger]:
            apertures += str(round(radius*2, 2)) + ','
        apertures = apertures[:-1]
        err_SEconfig['PHOT_APERTURES'] = apertures

        # Write the output parameter file.
        parameter_filename = self.write_params([f'FLUX_APER({sum(larger)})'], outdir)
        err_SEconfig['PARAMETERS_NAME'] = parameter_filename

        # Run SE on the large apertures.
        print('Running SExtractor on the large apertures...')
        self.run_SExtractor(largecmd, err_SEconfig)

        # Remove temporary parameter file and detection image.
        os.remove(sexfile)
        os.remove(parameter_filename)
        os.remove(large_filename)

        # Read in the catalogues with the different aperture
        # measurements.
        small = ascii.read(f'{outdir}/small_apertures.temp.cat')
        large = ascii.read(f'{outdir}/large_apertures.temp.cat')

        # Calaculate the median-absolute-deviation noise for each 
        # aperture size. 
        # Factor 1.48 converts to Gaussian-like standard deviation.
        medians = []
        s = (small['FLUX_APER'] != 0)
        for column in small.colnames:
            medians.append(median_abs_deviation(small[column][s], nan_policy='omit')*1.48)
        s = (large['FLUX_APER'] != 0)
        for column in large.colnames:
            medians.append(median_abs_deviation(large[column][s], nan_policy='omit')*1.48)

        # The sigma-clipped standard deviation of all non-object pixels.
        sig1 = sigma_clipped_stats(sci, mask)[2]    
        # The number of pixels in each aperture.
        Npix = np.pi * (radii**2)    

        # Defining the model to fit. 
        def model(theta, Npix=Npix):
            a, b, c, d = theta
            return sig1 * (((a / 1E8) * (Npix**b)) + (c * (Npix**(d / 1E1))))
        
        # Using a chi2 log-likelihood function.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x)) / yerr)** 2)
        
        # Setting allowed ranges for the free parameters.
        def lnprior(theta):
            a, b, c, d = theta
            if -1e6 < a < 1e6 and -1e6 < b< 1e6 and -1e6 < c < 1e6 and -1e6 < d < 1e6:
                return 0.0
            return -np.inf
        
        # Set up the MCMC.
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
        
        # The percentage error to use when fitting. 
        # Can help weight low or high regions.
        Merr = err_config['P_ERR']*np.array(medians)

        # Collect the x,y and error data.
        data = (Npix, medians, Merr)

        # Convert intitial parameters to an array.
        initial = np.array(err_config['INITIAL'])

        # Set the step methodology.
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob, args = data)
        print('Running MCMC burn-in...')
        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        print('Running MCMC production...')
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]

        print(f'Most likely parameter values: {theta_max}.')

        # Read original catalogue produced by SExtractor.
        cat = ascii.read(SEconfig['CATALOG_NAME'])

        # Median error value of the whole map.
        median_err = np.median(err[~np.isnan(err)])

        print('Calculating noise for catalogue sources...')

        # Expecting a few NaNs so quiet any warnings.
        with np.errstate(invalid='ignore'):
            
            # For each flux column.
            for column in cat.colnames:

                if column == 'FLUX_AUTO':
                    # Calculate the area of the aperture used.
                    cat['FLUX_AUTO_AREA'] = (np.pi * cat['A_IMAGE'] * cat['B_IMAGE']
                                             * np.power(cat['KRON_RADIUS'], 2))
                    # Extract the corresponding noise predicted by the 
                    # fit and scale based on local value.
                    cat['FLUXERR_AUTO'] = (model(theta_max,cat['FLUX_AUTO_AREA']) 
                                           * (err[cat['Y_IMAGE'].astype(int), 
                                                  cat['X_IMAGE'].astype(int)] / median_err))
                    cat.remove_column('FLUX_AUTO_AREA')
                if column == 'FLUX_APER':
                    cat[f'FLUXAPER_AREA'] = np.pi * np.power(radii[0], 2)
                    cat['FLUXERR_APER'] = (model(theta_max, cat[f'FLUXAPER_AREA']) 
                                           * (err[cat['Y_IMAGE'].astype(int), 
                                                  cat['X_IMAGE'].astype(int)] / median_err))
                    cat.remove_column('FLUXAPER_AREA')

                if 'FLUX_APER_' in column:
                    aper = int(column.split('FLUX_APER_')[1])
                    cat[f'FLUXAPER_{aper}_AREA'] = np.pi * np.power(radii[aper], 2)
                    cat[f'FLUXERR_APER_{aper}'] = (model(theta_max, cat[f'FLUXAPER_{aper}_AREA'])
                                                   * (err[cat['Y_IMAGE'].astype(int),
                                                          cat['X_IMAGE'].astype(int)] / median_err))
                    cat.remove_column(f'FLUXAPER_{aper}_AREA')
            # Overwrite the old catalogue with this one containing the
            # new uncertainties.
            cat.write(SEconfig['CATALOG_NAME'], format='ascii', overwrite = True)

        # Remove remaining aperture files and the segmentation map.
        os.remove(f'{outdir}/small_apertures.temp.cat')
        os.remove(f'{outdir}/large_apertures.temp.cat')
        if '.temp' in seg_filename:
            os.remove(seg_filename)

        # Save a plot of noise vs aperture size.
        if err_config['SAVE_FIG'] == True:

            x = np.linspace(0, max(Npix), 10000)
            fig = plt.figure()
            ax = plt.gca()
            plt.scatter(np.sqrt(Npix), medians,s = 15, color = 'white', edgecolors = 'blue',
                        alpha = 0.8)
            plt.plot(np.sqrt(x), model(theta_max, x), color = 'grey', linestyle = '--',
                     linewidth = 1)  
            plt.xlabel('sqrt(Number of pixels in aperture)')
            plt.ylabel('Noise in aperture [counts]')
            title = os.path.splitext(os.path.basename(SEconfig["CATALOG_NAME"]))[0]
            plt.title(f'{title.split(".cat")[0]}', fontsize = 10)
            plt.minorticks_on()
            ax.tick_params(axis = 'both', direction = 'in', which = 'both')
            plt.savefig(f'{SEconfig["CATALOG_NAME"].split(".cat")[0]}_noise.png')
            plt.close()

        print('Done. \n')

        return

    def SExtract(self, science, weight=None, parameters={}, output=['FLUX_AUTO'], outdir='./'):
        """
        Run SExtractor in any of its standard modes.

        Arguments
        ---------
        science (str, List[str])
            If str, the filename of the image to SExtract.
            If a List[str] filename of detection and measurement images.
        weight (None, str, List[str])
            If str, weight image for single image mode.
            If List[str], weight images for detection and measurement.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        output (list)
            List of output parameters to save.
        outdir (str)
            Directory in which to store outputs. 
        """

        print('Standard SExtraction')
        print('-'*len('Standard SExtraction'))

        # Generate a default SE config file.
        sexfile = self.generate_default(outdir) 
		
        # Create a copy of the configuration parameters specific to the 
        # image.
        img_SEconfig = copy.deepcopy(self.SEconfig)
        img_config = copy.deepcopy(self.config)

        # Overwrite parameters with those given at run time. 
        forbidden_parameters = ['CATALOG_TYPE', 'CATALOG_NAME']
        for (key, value) in parameters.items():
            if (key not in forbidden_parameters):
                if key in img_SEconfig:
                    img_SEconfig[key] = value
                elif key in img_config:
                    img_config[key] = value
                # Raise a warning if the parameter is unknown.
                else:
                    warnings.warn(f'{key} could not be found in the config file.\n If a valid'
                                  ' parameter, please set in the configuration file before'
                                  ' updating. Continuing without updating.', stacklevel = 2)
            # Raise a warning if the parameter cannot be changed.
            else:
                warnings.warn(f'Cannot overwrite configuration parameter {key}. It has a required'
                              ' value.', stacklevel = 2)
            
		# Set the catalogue name based on the measurement image.
        if type(science) == list:
            name = os.path.splitext(os.path.basename(science[1]))[0]
            img_SEconfig['CATALOG_NAME'] = f'{outdir}/{name}.cat'
        else:
            name = os.path.splitext(os.path.basename(science))[0]
            img_SEconfig['CATALOG_NAME'] = f'{outdir}/{name}.cat'
        print(f'SExtracting {os.path.basename(img_SEconfig["CATALOG_NAME"])[:-4]}...')
            
        # Add quantities needed for uncertainty estimation to measurement
        # if required.
        if img_config['EMPIRICAL'] == True:
            for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                if i not in output:
                    output.append(i)

            # Requires segmentation image so set if not requested.
            if img_SEconfig['CHECKIMAGE_TYPE'] != 'SEGMENTATION':
                img_SEconfig['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
                img_SEconfig['CHECKIMAGE_NAME'] = (f'{img_SEconfig["CATALOG_NAME"][:-4]}'
                                                   '.temp_seg.fits')
                warnings.warn('Empirical uncertainty eatimation requires a segmentation image.'
                              ' Other requested images will not be produced.', stacklevel = 2)
        
        # Write the output parameters to a text file.
        parameter_filename = self.write_params(output, outdir)
        img_SEconfig['PARAMETERS_NAME'] = parameter_filename

        # Check that checkimage directory exists.
        if img_SEconfig['CHECKIMAGE_TYPE'] != 'NONE':
            dir_name = os.path.dirname(img_SEconfig['CHECKIMAGE_NAME'])
            if os.path.isdir(dir_name) == False:
                raise KeyError(f'{dir_name} does not exist. Please set CHECKIMAGE_NAME correctly.')

        # Generate the base SExtractor command based on the mode.

        # Two image with weights.
        if (type(science) == list) and (type(weight) == list):
            basecmd = [self.sexpath, "-c", sexfile, science[0], science[1], '-WEIGHT_IMAGE',
                       f'{weight[0]},{weight[1]}']
        # Two image without weights.
        if (type(science) == list) and (type(weight) == type(None)):
            basecmd = [self.sexpath, "-c", sexfile, science[0], science[1]]
        # Single image without weight.
        if (type(science) == str) and (type(weight) == type(None)):
            basecmd = [self.sexpath, "-c", sexfile, science]
        # Single image with weight.
        if (type(science) == str) and (type(weight) == str):
            basecmd = [self.sexpath, "-c", sexfile, science, '-WEIGHT_IMAGE', weight]

        # Run SE using this command and the config parameters.
        self.run_SExtractor(basecmd, img_SEconfig)

        # Remove temporary parameter file.
        os.remove(sexfile)
        os.remove(parameter_filename)

        # Begin uncertainty estimation if needed.
        if img_config['EMPIRICAL'] == True:
            if (type(science) == str):
                self.measure_uncertainty(science, weight, img_SEconfig['CHECKIMAGE_NAME'],
                                         img_SEconfig, img_config, outdir)
            else:
                self.measure_uncertainty(science[1], weight[1], img_SEconfig['CHECKIMAGE_NAME'],
                                         img_SEconfig, img_config, outdir)

        # Convert to flux if required.
        self.convert_to_flux(img_SEconfig['CATALOG_NAME'], img_config)

        # Combine the two config files for saving as attributes.
        full_config = img_SEconfig
        full_config.update(img_config)
        # Convert catalogue to HDF5.
        outname = self.convert_to_hdf5(img_SEconfig['CATALOG_NAME'], full_config)

        print(f'Completed SExtraction and saved to {outname} \n')

        return outname

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
        cat['RA'] = coordinates.ra.degree
        cat['DEC'] = coordinates.dec.degree

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

    
    def extract(self, science, weight=None, parameters = {}, outputs=None, outdir='./'):
        """
        Main function for extracting sources and measuring photometry 
        in a science image.
        
        Arguments
        ---------
        science (str, List[str])
            If string, path to single image from which to detect and 
            measure sources. If List[str], path to detection image as the 
            first entry and measurement as the second.
        weight (None, str, List[str])
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
        if isinstance(science, list):
            sci_imgs = science
        else:
            sci_imgs = [science]
        
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
        if (len(sci_imgs) == 2) and (type(weight) == list):
            print('Starting in dual image mode with weighting.')
            cat_name = f'{outdir}/{os.path.basename(science[1]).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, weight[0], outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[1], config, segmap, cat, weight[1])
        # Two image without weight.
        elif (len(sci_imgs) == 2) and (type(weight) == type(None)):
            print('Starting in dual image mode.')
            cat_name = f'{outdir}/{os.path.basename(science[1]).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, bkg = bkgs[0], outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[1], config, segmap, cat, bkg = bkgs[1])
        # Single image without weight.
        elif (len(sci_imgs) == 1) and (type(weight) == type(None)):
            print('Starting in single image mode.')
            cat_name = f'{outdir}/{os.path.basename(science).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, bkg = bkgs[0], outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[0], config, segmap, cat, bkg = bkgs[0])
        # Single image with weight.
        elif (len(sci_imgs) == 1) and (type(weight) == str):
            print('Starting in single image mode with weighting.')
            cat_name = f'{outdir}/{os.path.basename(science).removesuffix(".fits")}.hdf5'
            cat, segmap = self.detect_sources(sci_imgs[0], hdrs[0], config, weight, outdir = outdir)
            flux, fluxerr, flag = self.measure_photometry(sci_imgs[0], config, segmap, cat, weight)

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
                    f[f'photometry/{column}'] = cat[column]

            # Add the config parameters as attributes.
            for key,value in att_config.items():
                f['photometry'].attrs[key] = value
            
        print(f'All done and saved to {cat_name}')

        return

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

        threshold = config['N_SIGMA'] * rms

        ##TODO: Use the error image at this stage.
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
    
    def extract(self, science, weight, error, parameters = {}, outputs = None, outdir = './'):

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
        if type(science) != list:
            print(f'Single image mode. \n D/M: {os.path.basename(science)}')
            science = [science]
            weight = [weight]
            error = [error]
            cat_name = f'{os.path.basename(science[0]).removesuffix(".fits")}_photutils.hdf5'
        else:
            print(f'Double image mode. \n D: {os.path.basename(science[0])} \n M: {os.path.basename(science[1])}')
            cat_name = f'{os.path.basename(science[1]).removesuffix(".fits")}_photutils.hdf5'

        # First load the detection images.
        sci_d, hdr_d = fits.getdata(science[0], header=True)
        wht_d = fits.getdata(weight[0])
        err_d = fits.getdata(error[0])

        # Measure background and filter.
        sci_d, bkg_d = self.measure_background(sci_d, wht_d, config)
        sci_d_filt = self.filter(sci_d, wht_d, bkg_d, config)

        # Generate the segmentation map,
        seg_map = self.segmentation(sci_d_filt, wht_d, bkg_d, config)

        # and save if requested.
        if config['SEGMAP'] != None:
            fits.writeto(config['SEGMAP'], seg_map, header=hdr_d)

        # If single image mode measurement is detection.
        if len(science) == 1:
            # So assign the same properies.
            sci_m = sci_d
            hdr_m = hdr_d
            wht_m = wht_d
            err_m = err_d
            sci_m_filt = sci_d_filt
            bkg_m = bkg_d

        # If double, need to background subtract and filter.
        elif len(science) == 2:

            # Load the measurement images.
            sci_m, hdr_m = fits.getdata(science[1], header=True)
            wht_m = fits.getdata(weight[1])
            err_m = fits.getdata(error[1])

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
                if column in outputs:
                    f[f'photometry/{column}'] = getattr(cat, column)

            # Add the config parameters as attributes.
            for key,value in att_config.items():
                f['photometry'].attrs[key] = value

        return
    
class ProFound():
    """
    Wrapper around the profound.R ProFound class to allow running 
    through Python.
    """

    def __init__(self, config_file, Rfile_path='./extraction/wrap_profound.R'):
        """
        __init__ method for ProFound.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        Rfile_path (str)
            Path to the file containing the R version of the ProFound 
            class.
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

        # The path to the profound.R file.
        self.Rfile_path = Rfile_path
    
    def extract(self, science, parameters={}, outputs=None, outdir='./'):
        """
        Perform source extraction using Profound. In reality this
        method simply passes the parameters to wrap_profound.R and the 
        extraction and processing is done there.

        Arguments
        ---------
        image (str, List[str])
            If str, the filename of the image to extract.
            If a List[str] filename of detection and measurement images.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (list)
            List of output parameters to save.
        outdir (str)
            Directory in which to store outputs. 

        Returns
        -------
        cat_name (str)
            The location of the generated catalogue.
        """

        # Start with the base command.
        basecmd = [self.Rfile_path, f'config_path={self.configfile}']

        # Add the science images.
        if type(science) == list:
            basecmd.append(f'img1={science[0]}', f'img2={science[1]}')
            cat_name = f'{outdir}/{os.path.basename(science[1]).replace(".fits","_profound.hdf5")}'
        else:
            basecmd.append(f'img1={science}')
            cat_name = f'{outdir}/{os.path.basename(science).replace(".fits","_profound.hdf5")}'

        # Get a comma seperated list of outputs.
        if outputs == None:
            basecmd.append('outputs=None')
        else:
            out_str = ''
            for output in outputs:
                out_str += f'{output},'
            basecmd.append(out_str[:-1])
        
        # Add the overwritten parameters.
        for key, value in parameters.items():
            basecmd.append(f'{key}={value}')

        # Finally the output directory.
        basecmd.append(f'outdir={outdir}')

        # Now run on the command line.     
        p = subprocess.Popen(basecmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        for line in p.stderr:
            print(line.decode(encoding = "UTF-8"))
        out, err = p.communicate()    

        return cat_name