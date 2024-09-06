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
from astropy.stats import sigma_clipped_stats

from photutils.utils import ImageDepth

import emcee

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
                      max_apers=50, max_iters=50000, bins=20, outdir='./'):
        """
        Use randomly placed apertures to measure the 5-sigma depth and
        noise distribution of an image.
        
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
        bins (int)
            The number of bins to use for the noise distribution.
        outdir (str)
            The directory in which to save temporary files.
        
        Retruns
        -------
        depth_5 (float)
            The 5-sigma depth of the image.
        vals (numpy.ndarray)
            Values of the noise histogram bins.
        edges (numpy.ndarray)
            Edges of the noise histogram bins.
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

        # Create a histogram of aperture magnitudes.
        mags = -2.5 * np.log10((apps['FLUX_APER'][s]*1e-9) / 3631)
        bins = np.linspace(min(mags), max(mags), bins)
        vals, edges = np.histogram(mags, bins = bins)

        # Measure the PSF curve of growth and interpolate.
        psf = fits.getdata(psf_filename)
        radii = np.arange(0.1, psf.shape[0], 1)
        radii, cog, p = measure_curve_of_growth(psf, radii = radii, position = None, norm = False,
                                                show = False)
        f = lambda r: np.interp(r, radii, cog)

        # Correct by the fraction of the PSF enclosed within the aperture
        # used and convert to 5 sigma.
        mad *= 5/f(radius)

        # Assuming units of nJy, calculate the 5 sigma limit in
        # magnitudes.
        depth_5 = -2.5 * np.log10((mad*1e-9) / 3631)

        # Delete the SExtractor catalog.
        os.remove(SEconfig_depth['CATALOG_NAME'])

        return depth_5, vals, edges
    
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

    def SExtract(self, image, weight=None, parameters={}, measurement=['FLUX_AUTO'], outdir='./'):
        """
        Run SExtractor in any of its standard modes.

        Arguments
        ---------
        image (str, List[str])
            If str, the filename of the image to SExtract.
            If a List[str] filename of detection and measurement images.
        weight (None, str, List[str])
            If str, weight image for single image mode.
            If List[str], weight images for detection and measurement.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        measurement (list)
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
        if type(image) == list:
            name = os.path.splitext(os.path.basename(image[1]))[0]
            img_SEconfig['CATALOG_NAME'] = f'{outdir}/{name}.cat'
        else:
            name = os.path.splitext(os.path.basename(image))[0]
            img_SEconfig['CATALOG_NAME'] = f'{outdir}/{name}.cat'
        print(f'SExtracting {os.path.basename(img_SEconfig["CATALOG_NAME"])[:-4]}...')
            
        # Add quantities needed for uncertainty estimation to measurement
        # if required.
        if img_config['EMPIRICAL'] == True:
            for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                if i not in measurement:
                    measurement.append(i)

            # Requires segmentation image so set if not requested.
            if img_SEconfig['CHECKIMAGE_TYPE'] != 'SEGMENTATION':
                img_SEconfig['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
                img_SEconfig['CHECKIMAGE_NAME'] = (f'{img_SEconfig["CATALOG_NAME"][:-4]}'
                                                   '.temp_seg.fits')
                warnings.warn('Empirical uncertainty eatimation requires a segmentation image.'
                              ' Other requested images will not be produced.', stacklevel = 2)
        
        # Write the output parameters to a text file.
        parameter_filename = self.write_params(measurement, outdir)
        img_SEconfig['PARAMETERS_NAME'] = parameter_filename

        # Check that checkimage directory exists.
        if img_SEconfig['CHECKIMAGE_TYPE'] != 'NONE':
            dir_name = os.path.dirname(img_SEconfig['CHECKIMAGE_NAME'])
            if os.path.isdir(dir_name) == False:
                raise KeyError(f'{dir_name} does not exist. Please set CHECKIMAGE_NAME correctly.')

        # Generate the base SExtractor command based on the mode.

        # Two image with weights.
        if (type(image) == list) and (type(weight) == list):
            basecmd = [self.sexpath, "-c", sexfile, image[0], image[1], '-WEIGHT_IMAGE',
                       f'{weight[0]},{weight[1]}']
        # Two image without weights.
        if (type(image) == list) and (type(weight) == type(None)):
            basecmd = [self.sexpath, "-c", sexfile, image[0], image[1]]
        # Single image without weight.
        if (type(image) == str) and (type(weight) == type(None)):
            basecmd = [self.sexpath, "-c", sexfile, image]
        # Single image with weight.
        if (type(image) == str) and (type(weight) == str):
            basecmd = [self.sexpath, "-c", sexfile, image, '-WEIGHT_IMAGE', weight]

        # Run SE using this command and the config parameters.
        self.run_SExtractor(basecmd, img_SEconfig)

        # Remove temporary parameter file.
        os.remove(sexfile)
        os.remove(parameter_filename)

        # Begin uncertainty estimation if needed.
        if img_config['EMPIRICAL'] == True:
            if (type(image) == str):
                self.measure_uncertainty(image, weight, img_SEconfig['CHECKIMAGE_NAME'],
                                         img_SEconfig, img_config, outdir)
            else:
                self.measure_uncertainty(image[1], weight[1], img_SEconfig['CHECKIMAGE_NAME'],
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