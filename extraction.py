import os
import signal
import atexit
import traceback
import subprocess
import re
import copy
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
from astropy.coordinates import SkyCoord

import emcee

import photutils
import photutils.background as pb
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.utils import ImageDepth

import sep

from utils import measure_curve_of_growth

import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

class TempFileManager:
    """
    Keep track of temporary files and remove them on code exit.
    """

    def __init__(self):
        """ 
        __init__ method for TempFileManager.
        """

        # Store temporary file paths here.
        self.temp_files = set()

        # Remove files on any kind of exit.
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup_on_signal)
        signal.signal(signal.SIGINT, self.cleanup_on_signal)

    def register(self, path):
        """
        Register a temporary file for cleanup.

        Arguments
        ---------
        path (str)
            File path of the temporary file to be deleted later.
        """
        self.temp_files.add(path)

    def cleanup(self):
        """
        Delete all registered temporary files.
        """
        for file in self.temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f'Warning: Failed to delete {file} ({e})')

    def cleanup_on_signal(self):
        """
        Remove files on signal.
        """        
        self.cleanup()

class SExtractor():
    """
    Wrap Source Extractor (SE) maintaining its key functionality
    and producing hdf5 catalogues.
    """

    def __init__(self, config_file, sexpath):
        """
        __init__ method for SExtractor.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        sexpath (str)
            Path to SE executable.
        """

        # Read the configuration file and split into SE and wrapper
        # specific parts.
        self.configfile = config_file
        with open(self.configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.SEconfig, self.config = content

        # Catalogue type is fixed.
        self.SEconfig['CATALOG_TYPE'] = 'ASCII_HEAD'

        # Path to the SE executable.
        self.sexpath = sexpath    

        # Raise a warning if SE version can't be determined.
        self.version = self.get_version()

        # Keep track of temporary files.
        self._temp_manager = TempFileManager()

        # Will use these later.
        self._outdir = None
        self._prefix = None

    def _generate_default(self):
        """
        Generate the default SE configuration file.

        Returns
        -------
        config_path (str)
            Path to the generated configuration file.
        """

        # Pipe the SExtractor output. -d requests the default parameters.
        p = subprocess.Popen([self.sexpath, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Write these parameters to a file.
        config_path = f'{self._outdir}/{self._prefix}default.temp.sex'
        self._temp_manager.register(config_path)

        f = open(config_path, 'w')
        f.write(out.decode(encoding = 'UTF-8'))
        f.close()

        return config_path
    
    def get_version(self):
        """
        Retrieve the SE version.

        Returns
        -------
        version (str)
            SE version number used to initalise the class.
        """

        # Run SE with no inputs.
        p = subprocess.Popen([self.sexpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Search the outputs for the version number.
        version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode(encoding='UTF-8'))
        
        # Raise error if no version found.
        if version_match is False:
            raise RuntimeError('Could not determine Source Extractor version. Check the output of'
                               f' running {self.sexpath}')
        version = str(version_match.group()[8:])

        return version	
     
    def _write_params(self, params):
        """
        Write output parameters to a text file in SE format.

        Arguments
        ---------
        params (set[str])
            Set of parameters to be written to the file.

        Returns
        -------
        parameter_path (str)
            Path to the generated parameter file.
        """

        # Write the parameters to a file.
        parameter_path = f'{self._outdir}/{self._prefix}temporary_parameters.temp.params'
        self._temp_manager.register(parameter_path)

        f = open(parameter_path, 'w')
        f.write("\n".join(params))
        f.write("\n")
        f.close()

        return parameter_path
        
    def _convert_to_hdf5(self, catalogue, config):
        """
        Converts a SE ascii catalogue to HDF5.

        Arguments
        ---------
        catalogue (str)
            Path to SE catalogue file to be converted.
        config (dict)
            Config containing all parameters to be added as attributes.

        Returns
        -------
        hdf5_name (str)
            Path to the generated hdf5 file.
        """

        print('Saving to hdf5 catalogue.')

        # Read the ascii catalogue.
        cat = Table.read(catalogue, format='ascii')

        # Create HDF5 file with the same name.
        hdf5_name = catalogue.replace(".temp.cat", '.hdf5')
        with h5py.File(hdf5_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')
            for column in cat.colnames:
                if ('FLUX' in column) & (column != 'FLUX_GROWTHSTEP') & (column != 'FLUX_RADIUS'):
                    f[f'photometry/{column}'] = cat[column] * config['TO_FLUX']
                else:
                    f[f'photometry/{column}'] = cat[column]
            
            # Add the parameters used to create the catalogue.
            for key in config:
                f['photometry'].attrs[key] = config[key]
            
            # Record the code used and version.
            f['photometry'].attrs['CODE'] = 'Source Extractor'
            f['photometry'].attrs['VERSION'] = self.version
        
        return hdf5_name
    
    def _run_SExtractor(self, basecmd, SEconfig):
        """
        Passes a command to SExtractor on the command line.

        Arguments
        ---------
        basecmd (str)
            String containing the base command line arguments.
        SEconfig (dict)
            SE config containing the parameter updates to be added.
        """

        # Copy the base SE command.
        SEcmd = copy.deepcopy(basecmd)

        # Add parameters given in the config.
        for (key, value) in SEconfig.items():
            SEcmd.append("-" + str(key))
            SEcmd.append(str(value).replace(' ',''))

        # Run SExtractor and print the outputs.
        self._temp_manager.register(SEconfig['CATALOG_NAME'])
        p = subprocess.Popen(SEcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
        for line in p.stderr:
            clean_line = ansi_escape.sub('', line.decode(encoding = "UTF-8"))
            print(clean_line, end='', flush=True)
        out, err = p.communicate()

        # Tell the user if an error comes from SE rather than 
        # the wrapper itself.
        if p.returncode != 0:
            raise RuntimeError('Source Extractor encountered an error. Check the '
                               'output for further information.')
        return
    
    def _update_config(self, parameters):
        """
        Copy and update the stored SE and wrapper config dictionaries 
        with parameters provided at runtime.

        Arguments
        ---------
        parameters (dict)
            Key-value pairs of parameters to update.
            
        Returns
        -------
        new_SEconfig (dict)
            A new updated Source Extractor config.
        new_config (dict)
            A new updated wrapper config.
        att_config (dict)
            Config with values appropriate for saving to hdf5.
        """

        # Create new copies of the config files.
        new_SEconfig = copy.deepcopy(self.SEconfig)
        new_config = copy.deepcopy(self.config)
                    
        # Overwrite parameters with those given at run time. 
        for (key, value) in parameters.items():
            if key in new_SEconfig:
                new_SEconfig[key] = value
            elif key in new_config:
                new_config[key] = value
            else:
                raise KeyError(f'Parameter {key} not found in config file. It either doesn\'t '
                            'exist, or needs to be defined in the config file before '
                            'overwriting.')
            
        # This is always fixed.
        new_SEconfig['CATALOG_TYPE'] = 'ASCII_HEAD'
        
        # We don't want expanded environment variables in the catalogues.
        # Create combined config file without these.
        att_config = copy.deepcopy(new_SEconfig)
        att_config.update(new_config)
            
        # Expand any environment variables.
        for key, value in new_SEconfig.items():
            if isinstance(value, str):
                new_SEconfig[key] = os.path.expandvars(value)   
            if isinstance(value, list):
                new_SEconfig[key] = ','.join(value)

        return new_SEconfig, new_config, att_config
        
    def _get_aperture_config(self, SEconfig):
        """
        Return an updated config with parameters appropriate for
        measuring in apertures around set locations.
        
        Arguments
        ---------
        SEconfig (dict)
            Dictonary containing the SE config to be updated.
        
        Returns
        -------
        SEconfig (dict)
            The updated config.
        """

        # Set the parameters.
        SEconfig['DETECT_MINAREA'] = 1
        SEconfig['DETECT_THRESH'] = 1E-12
        SEconfig['WEIGHT_TYPE'] = 'NONE'
        SEconfig['FILTER'] = 'N'
        SEconfig['CLEAN'] = 'N'
        SEconfig['MASK_TYPE'] = 'NONE'
        SEconfig['BACK_TYPE'] = 'MANUAL'
        SEconfig['BACK_VALUE'] = 0.0
        SEconfig['CHECKIMAGE_TYPE'] = 'NONE'
        SEconfig['BACKPHOTO_TYPE'] = 'GLOBAL'

        return SEconfig
    
    def _get_aperture_locations(self, sci, hdr, mask, radius, napers=10000, overlap=False,
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
            The name of the output detection image file.
        """

        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma=1.0, napers=napers, niters=1, overlap=overlap,
                           overlap_maxiters=overlap_maxiters)
        limits = depth(sci, mask)
        print(f' Placed {int(depth.napers_used)} apertures.')

        # Get the location of the apertures.
        locations = depth.apertures[0].positions

        # Construct the detection image.
        det = np.zeros(sci.shape)
        for i in np.round(locations).astype(int):
            det[i[1], i[0]] = 1

        # Save the file.
        self._temp_manager.register(outname)
        fits.writeto(outname, det.astype(np.float32), header=hdr, overwrite=True)

        return outname

    def measure_depth(self, science, psf, mask=None, weight=None, parameters={}, radius=3.33, 
                      max_apers=50, max_iters=50000, outdir='./'):
        """
        Use randomly placed apertures to measure the average total
        5-sigma depth of an image.
        
        Arguments
        ---------
        science (str)
            Filename of science fits image.
        psf (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask (None, str)
            Filename of the fits image mask. If None, generate and use
            a SE segmentation map.
        weight (None, str)
            Filename of fits weight map. If None, no weighting will be 
            used if generating a mask and only NaN non-source pixels will
            be masked.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        radius (float)
            Radius of the random apertures to use in pixels.
        max_apers (int)
            The maximum number of apertures to place.
        max_iters (int)
            The maximun attempts at finding a non overlapping location.
        outdir (str)
            The directory in which to save temporary files.
        
        Returns
        -------
        depth (float)
            The average total 5-sigma depth of the image.
        """

        try:

            print(f'Measuring 5-sigma depth of {os.path.basename(science)}.')
        
            # All files will be output here.
            if os.path.isdir(outdir):
                self._outdir = outdir
            else:
                raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an '
                                'existing directory.')            
            
            # Temporary file prefix.
            self._prefix = os.path.splitext(os.path.basename(science))[0]

            # Generate default SE config file.
            sexfile = self._generate_default()

            # Update the config files.
            SEconfig_depth, config_depth, _ = self._update_config(parameters)
        
            # Open the science image.
            sci, hdr = fits.getdata(science, header=True)
                
            # Has a source mask been provided?
            if isinstance(mask, str):
                source_mask = fits.getdata(mask)

            # If not, run SE and use the segmentation map as a mask.
            else:
                print('Generating source mask.')
                SEconfig_depth['CATALOG_NAME'] = f'{self._outdir}/{self._prefix}depth_mask.temp.cat'

                # Set up the command.
                basecmd = [self.sexpath, "-c", sexfile, science]

                # Do we have a weight map?  
                if isinstance(weight, str):
                    if len(SEconfig_depth['WEIGHT_TYPE'].split(',')) > 1:
                        raise ValueError('Ensure only one WEIGHT_TYPE is provided when using single'
                                       f' image mode. Currently {SEconfig_depth["WEIGHT_TYPE"]}.')
                    basecmd += ['-WEIGHT_IMAGE', weight]

                # This will generate the source mask.
                SEconfig_depth['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
                check_name = SEconfig_depth["CATALOG_NAME"].replace(".temp.cat", ".seg.temp.fits")
                SEconfig_depth['CHECKIMAGE_NAME'] = check_name
                self._temp_manager.register(check_name)

                # Need to request at least one output.
                SEconfig_depth['PARAMETERS_NAME'] = self._write_params(['NUMBER'])

                # Run SE with with the base command and config.
                self._run_SExtractor(basecmd, SEconfig_depth)
            
                # Open the mask.
                source_mask = fits.getdata(check_name)

            # Full mask includes sources and off detector regions.
            full_mask = (source_mask != 0) + np.isnan(sci)
            if isinstance(weight, str):
                wht = fits.getdata(weight)
                full_mask += np.isnan(wht)
                full_mask += (wht <= 0)

            # Update the config to work with specific aperture locations.
            SEconfig_depth = self._get_aperture_config(SEconfig_depth) 

            # Place random apertures and save as detection image.
            print('Placing random apertures...')
            det_filename = f'{self._outdir}/{self._prefix}depth_apertures.temp.fits'
            self._get_aperture_locations(sci, hdr, full_mask, radius, max_apers, overlap=False,
                                        overlap_maxiters=max_iters, outname=det_filename)

            # Build the base SE command line argument.
            detcmd = [self.sexpath, "-c", sexfile, det_filename, science]

            # Add the catalog name and aperture diameters.
            SEconfig_depth['CATALOG_NAME'] = f'{self._outdir}/{self._prefix}depth_apertures.temp.cat'
            SEconfig_depth['PHOT_APERTURES'] = str(round(radius*2, 2))
            SEconfig_depth['PARAMETERS_NAME'] = self._write_params(['FLUX_APER', 'NUMBER'])

            # Run SE.
            self._run_SExtractor(detcmd, SEconfig_depth)

            # Open the catalogue and get the fluxes.
            apps = ascii.read(SEconfig_depth['CATALOG_NAME'])
            flux = apps['FLUX_APER'] * config_depth['TO_FLUX']

            # Measure the median absolute deviation.
            s = (flux != 0) & (np.isfinite(flux))
            mad = median_abs_deviation(flux[s], nan_policy='omit', scale='normal')

            # Measure the PSF curve of growth and interpolate.
            psf_ = fits.getdata(psf)
            radii = np.arange(0.1, psf_.shape[0], 1)
            radii, cog, p = measure_curve_of_growth(psf_, radii=radii, position=None, 
                                                    norm=False, show=False)
            f = lambda r: np.interp(r, radii, cog)

            # Correct by the fraction of the PSF enclosed within the 
            # aperture used and convert to 5 sigma.
            depth = 5*mad/f(radius)

            print('Depth calculation completed! \n')

            return depth
        
        except:
            traceback.print_exc()

        finally:
            self._temp_manager.cleanup()
    
    def _empirical_uncertainty(self, science, weight, weight_type, segmap, SEconfig, config):
        """
        Perform empirical uncertainty estimation by fitting the relation
        between aperture size and noise. Based on Finkelstein+23.

        Arguments
        ---------
        science (str)
            Filename of the science image.
        weight (str)
            Filename of the corresponding weight map.
        SEconfig (dict)
            SE configuration parameters specific to this image.
        config (dict)
            Wrapper parameters specific to this image.
        segmap (None, str)
            Path to the segmentation map generated by SE. If None, run
            SE to generate.
        """

        print('\nBeginning uncertainty estimation:')

        # Create a copy of the configuration parameters specific to the
        # image and set some parameters for this case.
        err_SEconfig = copy.deepcopy(SEconfig)
        err_config = copy.deepcopy(config)

        # Generate a default configuration file.
        sexfile = self._generate_default()

        # Update the config to deal with aperture locations.
        err_SEconfig = self._get_aperture_config(err_SEconfig)
        
        # Open the science image and segmap. 
        sci, hdr = fits.getdata(science, header = True)
        seg = fits.getdata(segmap)

        # Open weight file and convert it to an RMS if required.
        err = fits.getdata(weight)
        if weight_type == 'MAP_WEIGHT':
            err = 1/np.sqrt(err)
        elif weight_type == 'MAP_VAR':
            err = np.sqrt(err)
            
        # Mask off detector regions.
        mask = np.isnan(sci) + np.isnan(err) + (err <= 0)

        # Get the aperture radii.
        if err_config['RADII_SPACING'] == 'linear':
            radii = np.linspace(err_config['MIN_RADIUS'], err_config['MAX_RADIUS'], 
                                err_config['N_RADII'])
        else:
            radii = np.logspace(np.log10(err_config['MIN_RADIUS']),
                                np.log10(err_config['MAX_RADIUS']), err_config['N_RADII'])

        # Seperate the radii into small and large components. This way we
        # only need to run SE another two times.
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        # For both runs.
        app_runs = {'small':smaller, 'large':larger}
        medians = []
        for run, s in app_runs.items():
            print(f' Placing {run} apertures...')

            # Get the aperture locations. Add sources to the mask.
            app_filename = f'{self._outdir}/{self._prefix}{run}_apertures.temp.fits'
            self._get_aperture_locations(
                sci, hdr, mask+(seg!=0), max(radii[s]), err_config[f'N_{run.upper()}'], False, 
                err_config['MAX_ITERS'], app_filename)

            # Set the catalogue name.
            err_SEconfig['CATALOG_NAME'] = app_filename.replace('.fits', '.cat')

            # Tell SE which apertures to use.
            apertures = ''
            for radius in radii[s]:
                apertures += str(round(radius, 2)*2) + ','
            apertures = apertures[:-1]
            err_SEconfig['PHOT_APERTURES'] = apertures

            # Write the output parameter file.
            parameter_filename = self._write_params([f'FLUX_APER({sum(s)})'])
            err_SEconfig['PARAMETERS_NAME'] = parameter_filename

            # Define the SE command.
            basecmd = [self.sexpath, "-c", sexfile, app_filename, science]

            # Run SE.
            self._run_SExtractor(basecmd, err_SEconfig)

            # Read the generated catalogue.
            app_cat = ascii.read(err_SEconfig['CATALOG_NAME'])

            # Calculate the MAD in each aperture.
            s = app_cat['FLUX_APER'] != 0
            for column in app_cat.colnames:
                medians.append(median_abs_deviation(app_cat[column][s], nan_policy='omit', 
                                                    scale='normal'))

            # Remove the aperture images ASAP as they can be quite large.
            os.remove(app_filename)

        # Defining the model to fit. 
        sig1 = sigma_clipped_stats(sci, mask+(seg!=0))[2]   
        Npix = np.pi * (radii**2)    
        def model(theta, Npix=Npix):
            a, b = theta
            return sig1 * a * (Npix**b)
        
        # Using a chi2 log-likelihood function.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x)) / yerr)** 2)
        
        # Setting allowed ranges for the free parameters.
        def lnprior(theta):
            a, b = theta
            if -1e9 < a < 1e9 and -1e9 < b< 1e9:
                return 0.0
            return -np.inf
        
        # Set up the MCMC.
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
        
        # The percentage error to use when fitting. 
        # Can help weight small or large apertures.
        Merr = err_config['P_ERR']*np.array(medians)

        # Collect the x,y and error data.
        data = (Npix, medians, Merr)

        # Set the step methodology.
        initial = np.array(err_config['INITIAL'])
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob, args = data)

        print(' Running MCMC...')
        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        print(f' Most likely parameter values: {theta_max}.')

        # Read original catalogue produced by SE.
        cat = ascii.read(SEconfig['CATALOG_NAME'])

        # Median error value of the whole map. Will use this to scale 
        # the errors.
        median_err = np.median(err[~mask])

        # We now want the radii of apertures used for photometry.
        radii = SEconfig.get('PHOT_APERTURES', '0')
        radii = [float(i) for i in radii.split(',')]        

        # Expecting a few NaNs so quiet any warnings.
        with np.errstate(invalid='ignore'):

            # Will scale errors by this relative value.
            rel_e = err[cat['Y_IMAGE'].astype(int)-1, cat['X_IMAGE'].astype(int)-1] / median_err

            # For each flux column, calculate the area based on the type
            # of aperture and extract the noise from the fit.
            for column in cat.colnames:

                if column == 'FLUX_AUTO':
                    area = np.pi * cat['A_IMAGE'] * cat['B_IMAGE'] * (cat['KRON_RADIUS']**2)
                    cat['FLUXERR_AUTO_EMPIRICAL'] = model(theta_max, area) * rel_e 
                    
                if column == 'FLUX_APER':
                    area = np.pi * np.power(radii[0], 2)
                    cat['FLUXERR_APER_EMPIRICAL'] = model(theta_max, area) * rel_e

                if 'FLUX_APER_' in column:
                    aper = int(column.split('FLUX_APER_')[1])
                    area = np.pi * np.power(radii[aper], 2)
                    cat[f'FLUXERR_APER_{aper}_EMPIRICAL'] = model(theta_max, area) * rel_e 

            # Overwrite the old catalogue with this one containing the
            # new uncertainties.
            cat.write(SEconfig['CATALOG_NAME'], format='ascii', overwrite=True)

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
            title = os.path.basename(SEconfig["CATALOG_NAME"]).removesuffix('.temp.cat')
            plt.title(f'{title.split(".cat")[0]}', fontsize = 10)
            plt.minorticks_on()
            ax.tick_params(axis = 'both', direction = 'in', which = 'both')
            plt.savefig(SEconfig["CATALOG_NAME"].replace('.temp.cat', '_noise.png'))
            plt.close()
        
        print(' Empirical errors calculated! \n')

        return

    def extract(self, science, weight=None, parameters={}, output=None, cat_name=None, outdir='./'):
        """
        Run Source Extractor in any of its standard modes.

        Arguments
        ---------
        science (str, List[str])
            If str, the filename of the image to extract.
            If a List[str] filename of detection and measurement images.
        weight (None, str, List[str])
            If None, ignore weighting.
            If str, weight map of the science image.
            If List[str], weight maps for detection and measurement.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        output (None, list)
            List of output parameters to save. If None, return some key 
            values.
        cat_name (None, str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            Directory in which to store outputs. 
        """
		
        try:
            
            # All files will be output here.
            if os.path.isdir(outdir):
                self._outdir = outdir
            else:
                raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an existing'
                                         ' directory.')

            # Create a copy of the config for this run.
            img_SEconfig, img_config, att_config = self._update_config(parameters)

            # Check each checkimage name to ensure the directory exists.
            # The SE error for this is not very helpful.
            check_images = {}
            if img_SEconfig['CHECKIMAGE_TYPE'].strip() != 'NONE':

                for check_type, check_name in zip(img_SEconfig['CHECKIMAGE_TYPE'].split(','), 
                                                  img_SEconfig['CHECKIMAGE_NAME'].split(',')):
                    dir_name = os.path.dirname(check_name)
                    if dir_name == '':
                        check_name = f'{outdir}/{check_name.strip()}'
                    elif os.path.isdir(dir_name) == False:
                        raise NotADirectoryError(f'{dir_name} does not exist. Use an existing '
                                                 ' directory when defining CHECKIMAGE_NAME.')
                    check_images[check_type.strip()] = check_name

            # If no output requested, return only key quantities. 
            if isinstance(output, type(None)):
                output = {'NUMBER', 'X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO'}
            output = set(output)            

            # Check for double mode.
            if isinstance(science, list):
                print('Starting extraction in dual image mode.')
                
                # Get the file prefix and catalogue name.
                self._prefix = os.path.splitext(os.path.basename(science[1]))[0]
                if isinstance(cat_name, type(None)):
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{self._prefix}_sextractor.temp.cat'
                else:
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{os.path.basename(cat_name)}.temp.cat'
                    self._prefix = cat_name
                
                # Generate the base SE parameter file and command.
                sexfile = self._generate_default() 
                basecmd = [self.sexpath, "-c", sexfile, science[0], science[1]]  

                # Check if weights are being used.
                if isinstance(weight, list):
                    if len(weight) == 2:

                        # Are we dealing with relative weights or RMS?
                        split_weight = img_SEconfig.get('WEIGHT_TYPE', 'NONE,NONE').split(',')
                        if len(split_weight) < 2:
                            raise ValueError('When using double image mode, WEIGHT_TYPE should have'
                                             ' the form MAP_{type},MAP_{type}.')
                        weight_type = split_weight[1].strip()

                        # How we set up the command depends on how many 
                        # weights were provided. 
                        s = [i == None for i in weight]

                        # Two weight images.
                        if sum(s) == 0:
                            if 'NONE' in img_SEconfig["WEIGHT_TYPE"]:
                                raise ValueError('Two weight maps provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            basecmd += ['-WEIGHT_IMAGE', f'{weight[0]},{weight[1]}']

                        # One weight image.
                        elif sum(s) == 1:
                            if img_SEconfig['WEIGHT_TYPE'].count('NONE') != 1:
                                raise ValueError('One weight map provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            if s[0] & ('NONE' not in split_weight[0]):
                                raise ValueError('No detection weight provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            if s[1] & ('NONE' not in split_weight[1]):
                                raise ValueError('No measurement weight provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            
                            weight = [item if item is not None else '' for item in weight]
                            basecmd += ['-WEIGHT_IMAGE', f'{weight[0]}{weight[1]}']
                            
                        # If no weight images, we don't need to update 
                        # the command.
                        elif sum(s) == 2:
                            if img_SEconfig.get('WEIGHT_TYPE', 'NONE,NONE').count('NONE') != 2:
                                raise ValueError('No weight images provided. Set ' 
                                            'WEIGHT_TYPE = NONE,NONE')

                    else:
                        raise ValueError('Double image mode requires a list of weight paths of the '
                                       'form [detection, measurement].')  
                    
                # Also allow passing a single None.       
                elif isinstance(weight, type(None)):
                    weight_type = 'NONE'
                    if img_SEconfig['WEIGHT_TYPE'].count('NONE') != 2:
                        raise ValueError('No weight images provided. Set WEIGHT_TYPE = NONE,NONE')
                else:
                    raise ValueError('Double image mode requires a list of weight paths of the '
                                    'form [detection, measurement].') 

            # If not double, hopefully we are in single image mode.
            elif isinstance(science, str):
                print('Starting extraction in single image mode.')

                # Get the file prefix and catalogue name.
                self._prefix = os.path.splitext(os.path.basename(science))[0]
                if isinstance(cat_name, type(None)):
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{self._prefix}_sextractor.temp.cat'
                else:
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{os.path.basename(cat_name)}.temp.cat'
                    self._prefix = cat_name

                # Generate the base SE parameter file and command.
                sexfile = self._generate_default() 
                basecmd = [self.sexpath, "-c", sexfile, science]

                # Check if weights are being used.
                if isinstance(weight, str):

                    # Update the base command.
                    basecmd += ['-WEIGHT_IMAGE', weight]
                    
                    # Are we dealing with relative weights or RMS?
                    if len(img_SEconfig['WEIGHT_TYPE'].split(',')) > 1:
                        raise ValueError('Single image mode but WEIGHT_TYPE = '
                                         f'{img_SEconfig["WEIGHT_TYPE"]}.')
                    elif 'NONE' in img_SEconfig['WEIGHT_TYPE']:
                        raise ValueError('Weight map provided but WEIGHT_TYPE = '
                                         f'{img_SEconfig["WEIGHT_TYPE"]}.')
                    weight_type = img_SEconfig['WEIGHT_TYPE'].strip()

                # If no weights are not being used, ensure types are set 
                # correctly.         
                elif isinstance(weight, type(None)):
                    weight_type = 'NONE'
                    match = re.match(r'^\s*NONE\s*$', img_SEconfig.get('WEIGHT_TYPE', 'NONE'))
                    if not match:
                        raise ValueError('No weight image provided. Set WEIGHT_TYPE = NONE')
                else:
                    raise ValueError('Single image mode requires a string path to a weight map or '
                                   'None for no weighting.')

            # If we get here, the inputs are very wrong.   
            else:
                raise ValueError('Image inputs are not the correct format. Use strings for single '
                               'image mode and lists of the form [detection, measurement] for '
                               'double. Use None for no weighting.')

            # Will uncertainties be estimated empirically?
            if img_config['EMPIRICAL'] == True:
            
                # we will need these quantitites.
                for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                    output.add(i)

                # Will also need a segmentation map.
                if 'SEGMENTATION' in check_images.keys():
                    segmap = check_images['SEGMENTATION']
                else:
                    segmap = img_SEconfig["CATALOG_NAME"].replace(".temp.cat", ".seg.temp.fits")
                    self._temp_manager.register(segmap)
                    check_images['SEGMENTATION'] = segmap
                
                # And possibly a weight map.
                if weight_type == 'NONE':
                    if 'BACKGROUND_RMS' in check_images.keys():
                        weight = check_images['BACKGROUND_RMS']
                    else:
                        weight = img_SEconfig["CATALOG_NAME"].replace(".temp.cat", ".rms.temp.fits")
                        self._temp_manager.register(weight)
                        check_images['BACKGROUND_RMS'] = weight
                    weight_type = 'MAP_RMS'
                elif isinstance(weight, list):
                    weight = weight[1]
            
            # Add all the checkimage requests to the config.
            img_SEconfig['CHECKIMAGE_TYPE'] = ','.join(check_images.keys())
            img_SEconfig['CHECKIMAGE_NAME'] = ','.join(check_images.values())
                    
            # Always need at least 2 outputs.
            if len(output) < 2:
                output.add('NUMBER')
            if len(output) < 2:
                output.add('FLUX_AUTO')

            # Write the full set of output parameters to a text file.
            parameter_filename = self._write_params(output)
            img_SEconfig['PARAMETERS_NAME'] = parameter_filename

            # Run SE using this command and the config parameters.
            self._run_SExtractor(basecmd, img_SEconfig)

            # Begin uncertainty estimation.
            if img_config['EMPIRICAL'] == True:
                if isinstance(science, list):
                    self._empirical_uncertainty(science[1], weight, weight_type, segmap, 
                                                img_SEconfig, img_config)
                else:
                    self._empirical_uncertainty(science, weight, weight_type, segmap, 
                                                img_SEconfig, img_config)                   
            
            # Combine the two config files and save everything to hdf5.
            outname = self._convert_to_hdf5(img_SEconfig['CATALOG_NAME'], att_config)

            print(f'Completed extraction and saved to {outname} \n')

            return outname
        
        except:
            traceback.print_exc()

        finally:
            self._temp_manager.cleanup()

class SEP():
    """
    Class for running SEP in dual or single image mode and performing
    Kron or circular aperture photometry.
    """

    def __init__(self, config_file):
        """
        __init__ method for SEP.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        """
        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            self.config = next(yaml.safe_load_all(file))

        # These are the available outputs.
        self.output_names = ['thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y',
                             'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', 'cxx',
                             'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak', 'xcpeak', 'ycpeak',
                             'xpeak', 'ypeak', 'flag', 'ellipse_flag', 'RA', 'DEC', 'FLUX_AUTO', 
                             'FLUXERR_AUTO', 'FLUX_FLAG']
        
        # May need this later.
        self._cat_name = None

    def _update_config(self, parameters):        
        """
        Copy and update the stored config with parameters provided 
        at runtime.

        Arguments
        ---------
        parameters (dict)
            Key-value pairs of parameters to update.
            
        Returns
        -------
        new_config (dict)
            Updated copy of the config file.
        att_config (dict)
            Config with values appropriate for saving to hdf5.
        """

        # Copy the stored parameter file.
        new_config = copy.deepcopy(self.config)

        # Update with the given parameters.
        new_config.update(parameters)

         # Store the config as is for saving as hdf5 attributes.
        att_config = copy.deepcopy(new_config)

        # Expand any environment variables and convert string to None.
        for key, value in new_config.items():
            if type(value) == str:
                new_config[key] = os.path.expandvars(value)
            if value == 'None':
                new_config[key] = None

        return new_config, att_config

    def _measure_background(self, sci, err, config):
        """
        Measure the background of an image using SEP functionality.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        config (dict)
            Dictionary of background configration arguments.

        Returns
        -------
        bkg (sep.Background)
            The measured SEP background object.
        """

        print('Estimating background.')

        # Load a source mask if provided.
        mask = np.isnan(sci)
        if config['background_mask'] != None:
            mask = mask + fits.getdata(config['background_mask'])

        # Also mask off detector regions if we can.
        if isinstance(err, type(None)) == False:
            mask = mask + (err <= 0) + np.isnan(err)

        # Measure the background.
        bkg = sep.Background(sci, mask, 0, config['bw'], config['bh'], config['fw'], 
                             config['fh'], config['fthresh'])

        return bkg
                
    def _detect_sources(self, sci, err, segmap, config):
        """
        Detect and deblend sources and create an initial catalogue.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        segmap (bool, numpy.ndarray)
            The segmap argument to pass to sep.extract. If bool, should
            the segmap be saved. If numpy.ndarray, the 2D segmentation
            map determined from the detection image.
        config (dict)
            Dictionary of background configration arguments.

        Return
        ------
        cat (astropy.table.Table)
            Astropy table storing information on detected sources.
        segmap (numpy.ndarray)
            2D image indicating the locations of detected sources.
        """

        # Mask off detector regions.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # Generate kernel based on provided FWHM and size.
        kernel_map = {
            'Gaussian': Gaussian2DKernel(x_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         y_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         x_size=config['SIZE'], y_size=config['SIZE']).array,
            'Tophat': Tophat2DKernel(config['FWHM'] / np.sqrt(2), x_size=config['SIZE'], 
                                     y_size=config['SIZE']).array
        }
        kernel = kernel_map.get(config['FILTER'], None)

        # Set some memory limits.
        sep.set_extract_pixstack(config['pixstack'])
        sep.set_sub_object_limit(config['object_limit'])

        # Do the extraction.
        objects, segmap = sep.extract(
            sci, config['thresh'], err=err, gain=config['gain'], mask=mask, maskthresh=0, 
            minarea=config['minarea'], filter_kernel=kernel, filter_type=config['filter_type'], 
            deblend_nthresh=config['deblend_nthresh'], deblend_cont = config['deblend_cont'], 
            clean = config['clean'], clean_param = config['clean_param'],segmentation_map=segmap)
        
        cat = Table(objects)
    
        # Extract can produce theta values > pi/2, so we need to 
        # correct these before performing photometry.
        invalid = cat['theta'] > np.pi / 2
        cat['theta'][invalid] -= np.pi

        return cat, segmap
    
    def _measure_photometry(self, sci, err, segmap, cat, config, type='kron', radius=0):
        """
        Measure the photometry of detected sources using Kron apertures.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image from which to identify sources.
        error (None, str)
            The path to the map for detection weighting.
        segmap (numpy.ndarray)
            2D image indicating the locations of detected sources.
        cat (astropy.table.Table)
            Astropy table storing information on detected sources.
        config (dict)
            Dictionary of source detection configuration arguments.
        type (str)
            The type of photometry to generate, either 'kron' or 
            'circular'.
        radius (float)
            The radius of the circular aperture in pixels. Only used when
            type = 'circular'.

        Return
        ------
        flux (numpy.ndarray)
            The flux of the detected sources in image counts.
        fluxerr (numpy.ndarray)
            The corresponding flux error.
        flag (numpy.ndarray)
            Flag indicating the quality of the measured photometry.
        ap_radius (float/numpy.ndarray)
            The radius of the circular aperture used or the radius of the
            Kron aperture used for each object.
        """

        # Mask off detector regions.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # The type of nearby pixel masking.

        # Do not mask any nearby pixels.
        if (config['mask_type'] == None) or (config['mask_type'] == 'NONE'):
            seg_id = None
            seg = None
        # Mask pixels belonging to other souces.
        elif config['mask_type'] == 'BLANK':
            seg_id = np.arange(1, len(cat)+1, dtype=np.int32)
            seg = segmap
        # Mask all pixels not identified as part of the segment.
        elif config['mask_type'] == 'SEGMENT':
            seg = segmap
            seg_id = np.arange(1, len(cat)+1, dtype=np.int32) * -1
        else:
            raise ValueError(f"mask_type {config['mask_type']} not recognised.")
        
        # Calculate the kron flux.
        if type == 'kron':
            # First get the kron radius.
            ap_radius, krflag = sep.kron_radius(
                sci, cat['x'], cat['y'], cat['a'], cat['b'], cat['theta'], config['int_radius'], 
                mask=mask, maskthresh=0, seg_id=seg_id, segmap=seg)
            
            # Then measure the flux in an elliptical aperture.
            flux, fluxerr, flag = sep.sum_ellipse(
                sci, cat['x'], cat['y'], cat['a'], cat['b'], cat['theta'], 
                config['kron_factor']*ap_radius, err=err, mask=mask, maskthresh=0, seg_id=seg_id, 
                segmap=seg, gain=config['gain'], subpix=config['subpix'])
            
            # Combine the Kron radius and ellipse flags.
            flag += krflag
        
        # Measure circular aperture photometry.
        elif type == 'circular':            
            ap_radius = radius
            flux, fluxerr, flag = sep.sum_circle(
                sci, cat['x'], cat['y'], ap_radius, err=err, mask=mask, maskthresh=0, seg_id=seg_id,
                  segmap=seg, gain=config['gain'], subpix=config['subpix'])
        
        return flux, fluxerr, flag, ap_radius
    
    def _get_aperture_locations(self, sci, mask, radius, napers, overlap=False, 
                                overlap_maxiters=50000):
        """
        Place random apertures in unmasked regions of an image and return
        their centres.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image in which to place the apertures.
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
        
        Returns
        -------
        x (List[float])
            The x-coordinate of the aperture centres.
        y (List[float])
            The y-coordinate of the aperture centres.
        """
        
        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma=1.0, napers=napers, niters=1, overlap=overlap,
                           overlap_maxiters=overlap_maxiters)
        limits = depth(sci, mask)
        print(f' Placed {int(depth.napers_used)} apertures.')

        # Get the location of the apertures.
        locations = depth.apertures[0].positions

        # Extract x-y coordinates from apertures.
        x = []
        y = []
        for i in np.round(locations).astype(int):
            x.append(i[0])
            y.append(i[1])

        return x, y
    
    def measure_depth(self, science, psf, mask=None, error=None, parameters={}, radius=3.33, 
                      max_apers=50, max_iters=50000):
        """
        Use randomly placed apertures to measure the average 
        5-sigma depth of an image.
        
        Arguments
        ---------
        science (str)
            Filename of science fits image.
        psf (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask (None, str)
            Filename of the fits image mask. If None, generate and use
            a SE segmentation map.
        error (None, str)
            Filename of fits error map. If None, no weighting will be 
            used if generating a mask and only NaN non-source pixels will
            be masked.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        radius (float)
            Radius of the random apertures to use in pixels.
        max_apers (int)
            The maximum number of apertures to place.
        max_iters (int)
            The maximun attempts at finding a non overlapping location.
        
        Returns
        -------
        depth (float)
            The 5-sigma depth of the image.
        """

        print(f'Measuring 5-sigma depth of {os.path.basename(science)}.')

        # Update the config file.
        depth_config, _ = self._update_config(parameters)
        depth_config['background_sub'] = False

        # Open the science image.
        sci, hdr = fits.getdata(science, header=True)
        sci = sci.byteswap(inplace=True).newbyteorder()

        # Load RMS map if available or use background RMS.
        if isinstance(error, type(None)):
            bkg = self._measure_background(sci, None, depth_config)
            err = bkg.rms()
        else:
            err = fits.getdata(error)
            err = err.byteswap(inplace=True).newbyteorder()
        
        # Has a source mask been provided?
        if isinstance(mask, str):
            source_mask = fits.getdata(mask)

        # If not, generate it.
        else:
            print('Generating source mask.')
            _, source_mask = self._detect_sources(sci, err, True, depth_config)
        
        # Construct the full source and coverage mask.
        full_mask = (source_mask != 0) | np.isnan(sci) | np.isnan(err) | (err <= 0)

        print('Placing random apertures...')
        x, y = self._get_aperture_locations(sci, full_mask, radius, max_apers, False, max_iters)
        cat = {'x':x, 'y':y}

        depth_config['mask_type'] = 'NONE'
        flux, _, _, _ = self._measure_photometry(sci, err, None, cat, depth_config, 
                                            'circular', radius)
        flux *= depth_config['flux_conversion']

        # Measure the median absolute deviation.
        s = (flux != 0) & (np.isfinite(flux))
        mad = median_abs_deviation(flux[s], nan_policy='omit', scale='normal')

        # Measure the PSF curve of growth and interpolate.
        psf_ = fits.getdata(psf)
        radii = np.arange(0.1, psf_.shape[0], 1)
        radii, cog, p = measure_curve_of_growth(psf_, radii=radii, position=None, 
                                                norm=False, show=False)
        f = lambda r: np.interp(r, radii, cog)

        # Correct by the fraction of the PSF enclosed within the 
        # aperture used and convert to 5 sigma.
        depth = 5*mad/f(radius)

        print('Depth calculation completed! \n')

        return depth
            
    def _empirical_uncertainty(self, sci, err, seg, cat, config):
        """
        Perform empirical uncertainty estimation by fitting the relation
        between aperture size and noise. Based on Finkelstein+23.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D science image.
        err (numpy.ndarray)
            RMS map matching the shape of sci.
        seg (numpy.ndarray)
            Segmentation map measured from sci.
        cat (astropy.table.table.Table)
            Detection catalogue generated by SEP from sci.
        config (dict)
            Dictionary of SEP configuration parameters.
        
        Returns
        -------
        cat (astropy.table.table.Table)
            SEP detection catalogue updated with empirical uncertainties.
        """

        print('\nBeginning uncertainty estimation:')

        # Make a local copy of the config.
        err_config = copy.deepcopy(config)
        err_config['mask_type'] = 'NONE'

        # Mask off detector regions and sources.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # Get the aperture radii.
        if err_config['RADII_SPACING'] == 'linear':
            radii = np.linspace(err_config['MIN_RADIUS'], err_config['MAX_RADIUS'], 
                                err_config['N_RADII'])
        else:
            radii = np.logspace(np.log10(err_config['MIN_RADIUS']),
                                np.log10(err_config['MAX_RADIUS']), err_config['N_RADII'])
        
        # Seperate the radii into small and large components. This way we
        # only need to run SEP twice.
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        app_runs = {'small':smaller, 'large':larger}
        medians = []
        for run, s in app_runs.items():
            print(f' Placing {run} apertures...')

            # Get the random locations for the apertures.
            x, y = self._get_aperture_locations(
                sci, mask+(seg!=0), max(radii[s]), err_config[f'N_{run.upper()}'], False, 
                err_config['MAX_ITERS'])
            ap_cat = {'x':x, 'y':y}

            # Measure the median flux in each aperture size.
            for r in radii[s]:
                flux, _, _, _ = self._measure_photometry(sci, err, seg, ap_cat, err_config, 
                                                         'circular', round(r,2))
                s_ = flux != 0
                medians.append(median_abs_deviation(flux[s_], nan_policy='omit', scale='normal'))

        # Defining the model to fit. 
        sig1 = sigma_clipped_stats(sci, mask+(seg!=0))[2]   
        Npix = np.pi * (radii**2)    
        def model(theta, Npix=Npix):
            a, b = theta
            return sig1 * a * (Npix**b)
        
        # Using a chi2 log-likelihood function.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x)) / yerr)** 2)
        
        # Setting allowed ranges for the free parameters.
        def lnprior(theta):
            a, b = theta
            if -1e9 < a < 1e9 and -1e9 < b < 1e9:
                return 0.0
            return -np.inf
        
        # Set up the MCMC.
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
    
        # The percentage error to use when fitting. 
        Merr = err_config['P_ERR']*np.array(medians)

        # Collect the x,y and error data.
        data = (Npix, medians, Merr)

        # Set the step methodology.
        initial = np.array(err_config['INITIAL'])
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob, args = data)

        print(' Running MCMC...')
        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        print(f' Most likely parameter values: {theta_max}')

        # Median error value of the whole map. Will use this to scale 
        # the errors.
        median_err = np.median(err[~mask])

        # We now want the radii of the apertures used for photometry.
        radii = err_config['radii']

        # Expecting a few NaNs so quiet any warnings.
        with np.errstate(invalid='ignore'):

            # Will scale errors by this relative value.
            rel_e = err[cat['y'].astype(int), cat['x'].astype(int)] / median_err

            # For each flux column, calculate the area based on the type
            # of aperture and extract the noise from the fit.
            for column in cat.colnames:

                if column == 'FLUX_AUTO':
                    area = np.pi * cat['a'] * cat['b'] * np.power(cat['KRON_RADIUS'] * 
                                                                  err_config['kron_factor'], 2)
                    cat['FLUXERR_AUTO_EMPIRICAL'] = (model(theta_max, area) * rel_e * 
                                                     err_config['flux_conversion'])

                    usec = (cat['KRON_RADIUS'] * err_config['kron_factor'] * 
                            np.sqrt(cat['a'] * cat['b']) < err_config['min_radius'])
                    area = np.pi * np.power(err_config['min_radius'], 2)
                    cat['FLUXERR_AUTO_EMPIRICAL'][usec] = (model(theta_max, area) * rel_e[usec] * 
                                                           err_config['flux_conversion'])

                if 'FLUX_APER_' in column:
                    aper = int(column.split('FLUX_APER_')[1])
                    area = np.pi * np.power(radii[aper], 2)
                    cat[f'FLUXERR_APER_{aper}_EMPIRICAL'] = (model(theta_max, area) * rel_e * 
                                                             err_config['flux_conversion'])

        # Save a plot of noise vs aperture size.
        if err_config['SAVE_FIG'] == True:

            x = np.linspace(0, max(Npix), 10000)
            fig = plt.figure()
            ax = plt.gca()
            plt.scatter(np.sqrt(Npix), medians,s = 15, color = 'white', edgecolors = 'blue',
                        alpha = 0.8)
            plt.plot(np.sqrt(x), model(theta_max, x), color = 'grey', linestyle = '--',
                     linewidth = 1)  
            title = os.path.basename(self._cat_name).removesuffix('.hdf5')
            plt.title(title, fontsize = 10)
            plt.xlabel('sqrt(Number of pixels in aperture)')
            plt.ylabel('Noise in aperture [counts]')
            plt.minorticks_on()
            ax.tick_params(axis = 'both', direction = 'in', which = 'both')
            plt.savefig(self._cat_name.replace('.hdf5', '_noise.png'))
            plt.close()

        print(' Empirical errors calculated! \n')

        return cat

    def extract(self, science, error=None, parameters={}, outputs=None, cat_name=None, outdir='./'):
        """
        Main function for extracting sources and measuring photometry 
        in a science image.
        
        Arguments
        ---------
        science (str, List[str])
            If string, path to single image from which to detect and 
            measure sources. If List[str], path to detection image as the 
            first entry and measurement as the second.
        error (None, str, List[str])
            The corresponding error images for detection.
            If None, use global background RMS.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (None, List[str])
            The source extraction outputs to save to the catalogue.
            If None, save all available.
        cat_name (None, str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            The directory in which to store output files.

        Returns
        -------
        cat_name (str)
            The filepath of the generated hdf5 catalogue.
        """

        # All files will be output here.
        if os.path.isdir(outdir) == False:
            raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an existing'
                                        ' directory.')

        # Update the config file with the given parameters.
        config, att_config = self._update_config(parameters)

        # Are we in double image mode?
        single_mode = False
        if isinstance(science, list):
            if len(science) == 2:
                print('Starting extraction in double image mode. \n')
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                    'form [detection, measurement].') 
             
            # Have errors been provided.
            if isinstance(error, list):
                if len(error) != 2:
                    raise ValueError('Double image mode requires a list of weight paths of the '
                                    'form [detection, measurement].')
                
                # If not, warn the user that the background RMS will be
                # used instead.
                s = [i == None for i in error]  
                if sum(s) == 2:
                    print('No RMS maps provided. Will use measured background RMS for weighting. \n')
                elif sum(s) == 1:
                    print(f'No RMS map provided for {np.array(["detection", "measurement"])[s][0]}.'
                          ' Will use measured background RMS for weighting. \n')
            elif isinstance(error, type(None)):
                print('No RMS maps provided. Will use measured background RMS for weighting. \n')
                error = [error] * 2
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                                'form [detection, measurement], or None for no weighting.') 
            
        # If not, we should be in single image mode.
        elif isinstance(science, str):
            print('Starting extraction in single image mode. \n')
            single_mode = True
            
            if isinstance(error, type(None)):
                print('No RMS map provided. Will use measured background RMS for weighting. \n')
            elif isinstance(error, str) == False:
                raise ValueError('Single image mode requires a string path to a weight map or None '
                               'for no weighting.') 
            
            # Duplicate the inputs to match double image format.
            science = [science] * 2
            error = [error] * 2

        # If we get here, the inputs are very wrong.   
        else:
            raise ValueError('Image inputs are not the correct format. Use strings for single image'
                           ' mode and lists of the form [detection, measurement] for double. '
                           'Use None for no weighting.')
        
        # Name the catalogue after the measurement image.
        if isinstance(cat_name, type(None)):
            cat_name = f'{outdir}/{os.path.basename(science[1]).removesuffix(".fits")}_sep.hdf5'
        else:
            cat_name = f'{outdir}/{os.path.basename(cat_name)}.hdf5'
        self._cat_name = cat_name

        # Load detection image.
        if not single_mode: print(f'Processing {os.path.basename(science[0])}:')
        sci_d, hdr_d = fits.getdata(science[0], header=True)
        sci_d = sci_d.byteswap(inplace=True).newbyteorder()

        # Load RMS map if available.
        if isinstance(error[0], type(None)) == False:
            err_d = fits.getdata(error[0])
            err_d = err_d.byteswap(inplace=True).newbyteorder()
        else:
            err_d = None

        # Measure the background if needed.
        if isinstance(err_d, type(None)) or config['background_sub']:
            bkg = self._measure_background(sci_d, err_d, config)
            if isinstance(err_d, type(None)):
                err_d = bkg.rms()
            if config['background_sub']:
                print('Subtracting the background.')
                bkg.subfrom(sci_d)

        # Create a segmentation map and an initial catalogue.
        print('Detecting sources.')
        cat, segmap = self._detect_sources(sci_d, err_d, True, config)

        # Save the segmentation map if requested.
        if config['segmap_name'] != None:
            print(f'Saving segmentation map to {outdir}/{config["segmap_name"]}.fits')
            fits.writeto(f'{outdir}/{config["segmap_name"]}.fits', segmap, hdr_d, overwrite=True)

        # If in single image mode, measurement is detection.
        if single_mode:
            sci_m, err_m = (sci_d, err_d)
        
        # Otherwise need to load and process the measurement images.
        else:
            print(f'Processing {os.path.basename(science[1])}:')
            sci_m = fits.getdata(science[1])
            sci_m = sci_m.byteswap(inplace=True).newbyteorder()

            # Load RMS map if available.
            if isinstance(error[1], type(None)) == False:
                err_m = fits.getdata(error[1])
                err_m = err_m.byteswap(inplace=True).newbyteorder()
            else:
                err_m = None

            # Measure the background if needed.
            if isinstance(err_m, type(None)) or config['background_sub']:
                bkg = self._measure_background(sci_m, err_m, config)
                if isinstance(err_m, type(None)):
                    err_m = bkg.rms()
                if config['background_sub']:
                    print('Subtracting the background.')
                    bkg.subfrom(sci_m)

            # Rerun detect_sources with the previously measured segmap to
            # get a new catalogue.
            cat, _ = self._detect_sources(sci_m, err_m, segmap, config)

        # Can't perform aperture photometry if a or b couldn't be 
        # measured. Will set these to zero and flag.
        s_a = ~np.isfinite(cat['a'])
        cat['a'][s_a] = 0

        s_b = ~np.isfinite(cat['b'])
        cat['b'][s_b] = 0

        cat['ellipse_flag'] = s_a + s_b

        # Calculate RA and DEC.
        wcs = WCS(hdr_d)
        coordinates = pixel_to_skycoord(cat['x'], cat['y'], wcs)
        cat['RA'] = coordinates.ra.degree
        cat['DEC'] = coordinates.dec.degree

        # Measure the photometry in Kron apertures.
        print('Measuring Kron photometry.')
        kflux, kfluxerr, kflag, kron = self._measure_photometry(sci_m, err_m, segmap, cat, config, 
                                                            'kron')

        # Also measure in circular apertures, with the minimum Kron 
        # radius defined in the config.
        r_min = config['min_radius']
        cflux, cfluxerr, cflag, _ = self._measure_photometry(sci_m, err_m, segmap, cat, config, 
                                                            'circular', r_min)

        # Only use this photometry when kron radius is less than minimum.
        use_circle = kron * np.sqrt(cat['a'] * cat['b']) < r_min

        # Replace Kron flux measurements with circular ones.
        kflux[use_circle] = cflux[use_circle]
        kfluxerr[use_circle] = cfluxerr[use_circle]
        kflag[use_circle] = cflag[use_circle]

        # Convert to desired flux unit and add to catalogue.
        cat['FLUX_AUTO'] = kflux * config['flux_conversion']
        cat ['FLUXERR_AUTO'] = kfluxerr * config['flux_conversion']
        cat['FLUX_FLAG'] = kflag
        cat['KRON_RADIUS'] = kron

        # Also measure flux in user defined circular apertures.
        for idx, radius in enumerate(config['radii']):
            if idx == 0: print('Measuring aperture photometry.')
            flux, fluxerr, _, _ = self._measure_photometry(sci_m, err_m, segmap, cat, config, 
                                                          'circular', radius)
            cat[f'FLUX_APER_{idx}'] = flux * config['flux_conversion']
            cat[f'FLUXERR_APER_{idx}'] = fluxerr * config['flux_conversion']

        # Perfrom empirical uncertaninty estimation.
        if config['EMPIRCIAL']:
            self._empirical_uncertainty(sci_m, err_m, segmap, cat, config)

        # Need to convert other quantities to chosen flux unit.
        flux_columns = ['cflux', 'flux', 'cpeak', 'peak']
        for name in flux_columns:
            cat[name] = cat[name] * config['flux_conversion']

        # Now add everything to the hdf5 catalogue.
        with h5py.File(cat_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')

            # If no outputs requested, use all.
            if outputs == None:
                outputs = cat.colnames
            outputs = set(outputs)
                        
            # Add the outputs to the hdf5 catalogue.
            for output in outputs:
                if output in cat.colnames:
                    f[f'photometry/{output}'] = cat[output]
                else:
                    print(f'Skipping {output} as it is not a recognised output quantity. ' 
                          'Check SEP.output_names for available outputs.')

            # Add the config parameters as attributes.
            for key,value in att_config.items():
                f['photometry'].attrs[key] = value
            f['photometry'].attrs['CODE'] = 'SEP'
            f['photometry'].attrs['VERSION'] = sep.__version__
            
        print(f'Completed extraction and saved to {cat_name} \n')

        return cat_name

class Photutils():

    def __init__(self, config_file):
        """
        __init__ method for Photutils.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        """

        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            self.config = next(yaml.safe_load_all(file))

        # List of outputs produced by SourceCatalogue.
        self.output_names = [
            'area', 'background_centroid', 'background_mean', 'background_sum', 'bbox_xmax',
              'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'centroid','centroid_quad', 'centroid_win', 
              'covar_sigx2', 'covar_sigy2', 'covariance', 'covariance_eigvals', 'cutout_centroid', 
              'cutout_centroid_quad', 'cutout_centroid_win', 'cutout_maxval_index', 
              'cutout_minval_index', 'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 
              'elongation', 'equivalent_radius', 'fwhm', 'gini', 'inertia_tensor', 
              'kron_flux', 'kron_fluxerr', 'kron_radius', 'label', 'labels', 
              'local_background', 'max_value', 'maxval_index', 
              'maxval_xindex', 'maxval_yindex', 'min_value', 'minval_index', 'minval_xindex', 
              'minval_yindex', 'moments', 'moments_central', 'orientation', 'perimeter',
              'segment_area', 'segment_flux', 'segment_fluxerr', 'semimajor_sigma', 
              'semiminor_sigma', 'sky_bbox_ll', 'sky_bbox_lr', 'sky_bbox_ul', 'sky_bbox_ur', 
              'sky_centroid', 'sky_centroid_icrs', 'sky_centroid_quad', 'sky_centroid_win', 
              'xcentroid', 'xcentroid_quad', 'xcentroid_win', 'ycentroid', 
              'ycentroid_quad', 'ycentroid_win', 'RA', 'DEC', 'background', 'convdata', 'data',
              'error', 'segment']
        
        # May need this later.
        self._cat_name = None

    def _update_config(self, parameters):
        """
        Copy and update the stored config with parameters provided 
        at runtime.

        Arguments
        ---------
        parameters (dict)
            Key-value pairs of parameters to update.
            
        Returns
        -------
        new_config (dict)
            Updated copy of the config file.
        att_config (dict)
            Config with values appropriate for saving to hdf5.
        """

        # Copy the stored parameter file.
        new_config = copy.deepcopy(self.config)

        # Update with the given parameters.
        new_config.update(parameters)

         # Store the config as is for saving as hdf5 attributes.
        att_config = copy.deepcopy(new_config)

        # Expand any environment variables and convert string to None.
        for key, value in new_config.items():
            if type(value) == str:
                new_config[key] = os.path.expandvars(value)
            if value == 'None':
                new_config[key] = None

        return new_config, att_config
        
    def _measure_background(self, sci, err, config):
        """
        Measure and remove the background from a science image.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D array of science image values.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        config (dict)
            Key value pairs defining the background measurement 
            parameters.

        Returns
        -------
        bkg (photutils.background.Background2D)
            Photutils background object measured from the science image.
        """

        # Translate interpolation, background and RMS estimators.
        interpolators = {'IDW':pb.BkgIDWInterpolator(), 'Zoom':pb.BkgZoomInterpolator()}
        back_est = {'Mean':pb.MeanBackground(), 'Median':pb.MedianBackground(), 
                    'Mode':pb.ModeEstimatorBackground(),'MMM':pb.MMMBackground(),
                    'SExtractor':pb.SExtractorBackground(),
                    'BiweightLocation':pb.BiweightLocationBackground()}
        rms_est = {'Std':pb.StdBackgroundRMS(), 'MADStd':pb.MADStdBackgroundRMS(), 
                   'BiweightScale':pb.BiweightScaleBackgroundRMS()}
        
        # Set up the coverage mask.
        coverage_mask = np.isnan(sci)
        if isinstance(err, type(None)) == False:
            coverage_mask += (err <= 0) + np.isnan(err)
        
        # Use source mask if provided.
        mask = None
        if config['SOURCE_MASK'] != None:
            mask = fits.getdata(config['SOURCE_MASK'])

        # Get the sigma clipping object.
        sigma_clip = None
        if config['SIGMA_CLIP'] == True:
            sigma_clip = SigmaClip(sigma_lower=config['SIGMA'][0], sigma_upper=config['SIGMA'][1], 
                                   maxiters=config['MAX_ITERS'])

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
        
        return bkg
    
    def _filter(self, sci, err, bkg, config):
        """
        Filter an image using a Guassian or Tophat kernel.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D array of science image values.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        bkg (photutils.background.Background2D)
            Photutils background object measured from the science image.
        config (dict)
            Key value pairs defining the filtering parameters.

        Returns
        -------
        sci (numpy.ndarray)
            The filtered science image.
        """

        # Replace off detector regions with median background so 
        # convolution doesn't smear them.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci) + (sci == 0)
        sci = np.where(mask == True, bkg.background_median, sci)

        # Generate kernel based on provided FWHM and size.
        kernel_map = {
            'Gaussian': Gaussian2DKernel(x_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         y_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         x_size=config['SIZE'], y_size=config['SIZE']).array,
            'Tophat': Tophat2DKernel(config['FWHM'] / np.sqrt(2), x_size=config['SIZE'], 
                                     y_size=config['SIZE']).array
        }
        kernel = kernel_map[config['FILTER']]

        # Generate kernel based on provided FWHM and convolve.
        sci = convolve_fft(sci, kernel, boundary='fill', fill_value=bkg.background_median,
                            nan_treatment='interpolate', preserve_nan=True, allow_huge=True)
            
        # Revert to zeros in the off detector region.
        sci = np.where(mask == True, 0, sci)
                
        return sci
    
    def _segmentation(self, sci, err, config):
        """
        Segment and deblend sources in a sicence image.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D array of science image values.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        config (dict)
            Key value pairs defining the detection parameters.

        Returns
        -------
        seg_image (photutils.segmentation.SegmentationImage)
            A segmentation image, with the same shape as sci, where 
            sources are marked by different positive integer values. 
        """

        # Compute the detection threshold.
        threshold = config['N_SIGMA'] * err

        # Mask off detector regions.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # Generate the segmentation image.
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
    
    def _get_aperture_locations(self, sci, mask, radius, napers, overlap=False, 
                                overlap_maxiters=50000):
        """
        Place random apertures in unmasked regions of an image and create
        a detection image based on their centres.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image in which to place the apertures.
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
        
        Returns
        -------
        det (numpy.ndarray)
            Array matching the shape of sci with value 1 at aperture 
            centres and zero otherwise.
        """
                        
        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma=1.0, napers=napers, niters=1, overlap=overlap,
                           overlap_maxiters=overlap_maxiters)
        limits = depth(sci, mask)
        print(f' Placed {int(depth.napers_used)} apertures.')

        # Get the location of the apertures.
        locations = depth.apertures[0].positions

        # Construct the detection image.
        x = []
        y = []
        for i in np.round(locations).astype(int):
            x.append(i[0])
            y.append(i[1])

        det = np.zeros(sci.shape)
        for i in np.round(locations).astype(int):
            det[i[1], i[0]] = 1

        return det
    
    def measure_depth(self, science, psf, mask=None, error=None, parameters={}, radius=3.33, 
                      max_apers=50, max_iters=50000):
        """
        Use randomly placed apertures to measure the average 
        5-sigma depth of an image.
        
        Arguments
        ---------
        science (str)
            Filename of science fits image.
        psf (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask (None, str)
            Filename of the fits image mask. If None, generate and use
            a SE segmentation map.
        error (None, str)
            Filename of fits error map. If None, no weighting will be 
            used if generating a mask and only NaN non-source pixels will
            be masked.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        radius (float)
            Radius of the random apertures to use in pixels.
        max_apers (int)
            The maximum number of apertures to place.
        max_iters (int)
            The maximun attempts at finding a non overlapping location.
        
        Returns
        -------
        depth (float)
            The 5-sigma depth of the image.
        """

        print(f'Measuring 5-sigma depth of {os.path.basename(science)}.')
        
        # Update the config file with given parameters.
        depth_config, _ = self._update_config(parameters)
        depth_config['BKG_SUB'] = False

        sci, hdr = fits.getdata(science, header=True)

        # Load RMS map if available.
        bkg = None
        if isinstance(error, type(None)):
            bkg = self._measure_background(sci, None, depth_config)
            err = bkg.background_rms
        else:
            err = fits.getdata(error)

        # Has a source mask been provided?
        if isinstance(mask, str):
            source_mask = fits.getdata(mask)

        # If not, generate it.
        else:
            print('Generating source mask.')

            # Filter the image if required.
            if depth_config['FILTER'] != None:

                # Measure background if we haven't already.
                if isinstance(bkg, type(None)):
                    bkg = self._measure_background(sci, err, depth_config)

                sci_filt = self._filter(sci, err, bkg, depth_config)
                source_mask = self._segmentation(sci_filt, err, depth_config).data

            else:
                source_mask = self._segmentation(sci, err, depth_config).data

        # Construct the full source and coverage mask.
        mask = np.isnan(sci) | np.isnan(err) | (err <= 0)  
        full_mask = mask | (source_mask != 0)  

        # Get the random aperture locations and construct and image with
        # ones at these coordinates.
        print('Placing random apertures...')
        det = self._get_aperture_locations(sci, full_mask, radius, max_apers, False, max_iters) 

        # Perform aperture photometry.
        det_seg = detect_sources(det, threshold=1E-12, npixels=1, mask=mask)
        ap_cat = SourceCatalog(sci, det_seg, error=err, mask=mask)
        ap_cat.circular_photometry(radius, 'APER_0')

        # Calculate the Gaussian-like MAD of the fluxes.
        flux = getattr(ap_cat, f'APER_0_flux')*depth_config['CONVERSION']
        s = (flux != 0) & (np.isfinite(flux))
        mad = median_abs_deviation(flux, nan_policy='omit', scale='normal')

        # Measure the PSF curve of growth and interpolate.
        psf_ = fits.getdata(psf)
        radii = np.arange(0.1, psf_.shape[0], 1)
        radii, cog, p = measure_curve_of_growth(psf_, radii=radii, position=None, 
                                                norm=False, show=False)
        f = lambda r: np.interp(r, radii, cog)

        # Correct by the fraction of the PSF enclosed within the 
        # aperture used and convert to 5 sigma.
        depth = 5*mad/f(radius)

        print('Depth calculation completed! \n')

        return depth
    
    def _empirical_uncertainty(self, sci, err, seg, cat, config):
        """
        Perform empirical uncertainty estimation by fitting the relation
        between aperture size and noise. Based on Finkelstein+23.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D science image.
        err (numpy.ndarray)
            RMS map matching the shape of sci.
        seg (numpy.ndarray)
            Segmentation map measured from sci.
        cat (photutils.segmentation.catalog.SourceCatalog)
            SourceCatalogue generated by Photutils from sci.
        config (dict)
            Dictionary of SEP configuration parameters.
        
        Returns
        -------
        cat (photutils.segmentation.catalog.SourceCatalog)
            SourceCatalogue updated with empirical uncertainties.
        """

        print('\nBeginning uncertainty estimation:')

        # Make a local copy of the config.
        err_config = copy.deepcopy(config)
        err_config['FILTER'] = None
        err_config['N_SIGMA'] = 1E-12
        err_config['N_PIXELS'] = 1
        err_config['APERMASK_METHOD'] = None
        err_config['BKG_SUB'] = False 

        # Mask off detector regions and sources.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # Get the aperture radii.
        if err_config['RADII_SPACING'] == 'linear':
            radii = np.linspace(err_config['MIN_RADIUS'], err_config['MAX_RADIUS'], 
                                err_config['N_RADII'])
        else:
            radii = np.logspace(np.log10(err_config['MIN_RADIUS']),
                                np.log10(err_config['MAX_RADIUS']), err_config['N_RADII'])
        
        # Seperate the radii into small and large components. This way we
        # only need to run SEP twice.
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        app_runs = {'small':smaller, 'large':larger}
        medians = []
        for run, s in app_runs.items():

            # Get the random locations for the apertures.
            det = self._get_aperture_locations(
                sci, mask+(seg!=0), max(radii[s]), err_config[f'N_{run.upper()}'], False, 
                err_config['MAX_ITERS'])

            det_seg = detect_sources(det, threshold=1E-12, npixels=err_config['N_PIXELS'],
                                        connectivity=err_config['CONNECTIVITY'], mask = mask)
    
            ap_cat = SourceCatalog(
                sci, det_seg, convolved_data=None, error=err, mask=mask,
                background=None, wcs=None, localbkg_width=config['LOCALBKG_WIDTH'],
                apermask_method=config['APERMASK_METHOD'], kron_params=config['KRON_PARAMS'],
                detection_cat=None, progress_bar=False)
        
            # Measure the median flux in each aperture size.
            for i, r in enumerate(radii[s]):
                ap_cat.circular_photometry(r, f'APER_{i}', overwrite=False)
                medians.append(median_abs_deviation(getattr(ap_cat, f'APER_{i}_flux'), 
                                                    nan_policy='omit', scale='normal'))

        # Defining the model to fit. 
        sig1 = sigma_clipped_stats(sci, mask+(seg!=0))[2]   
        Npix = np.pi * (radii**2)    
        def model(theta, Npix=Npix):
            a, b = theta
            return sig1 * a * (Npix**b)
        
        # Using a chi2 log-likelihood function.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x)) / yerr)** 2)
        
        # Setting allowed ranges for the free parameters.
        def lnprior(theta):
            a, b = theta
            if -1e9 < a < 1e9 and -1e9 < b < 1e9:
                return 0.0
            return -np.inf
        
        # Set up the MCMC.
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
    
        # The percentage error to use when fitting. 
        # Can help weight small or large apertures.
        Merr = err_config['P_ERR']*np.array(medians)

        # Collect the x,y and error data.
        data = (Npix, medians, Merr)

        # Set the step methodology.
        initial = np.array(err_config['INITIAL'])
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob, args = data)

        print(' Running MCMC...')
        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        print(f' Most likely parameter values: {theta_max}.')

        # We now want the radii of the apertures used for photometry.
        radii = err_config['RADII']

        # Median error value of the whole map. Will use this to scale 
        # the errors.
        median_err = np.median(err[~mask])

        # Expecting a few NaNs so quiet any warnings.
        labels = []
        with np.errstate(invalid='ignore'):

            # Will scale errors by this relative value.
            rel_e = err[cat.ycentroid.astype(int), cat.xcentroid.astype(int)] / median_err

            # Scale Kron flux,
            area = np.pi * (cat.semimajor_sigma * cat.semiminor_sigma * 
                            np.power(cat.kron_radius * err_config['KRON_PARAMS'][0], 2))
            cat.add_extra_property('kron_fluxerr_empirical', model(theta_max, area) * rel_e)
            labels.append('kron_fluxerr_empirical')

            # Segment flux,
            area = cat.segment_area
            cat.add_extra_property('segment_fluxerr_empirical',  model(theta_max, area) * rel_e)
            labels.append('segment_fluxerr_empirical')

            # and any aperture fluxes.
            for idx, radius in enumerate(radii):
                area = np.pi * np.power(radius, 2)
                cat.add_extra_property(f'APER_{idx}_fluxerr_empirical', 
                                       model(theta_max, area) * rel_e)
                labels.append(f'APER_{idx}_fluxerr_empirical')

        # Save a plot of noise vs aperture size.
        if err_config['SAVE_FIG'] == True:

            x = np.linspace(0, max(Npix), 10000)
            fig = plt.figure()
            ax = plt.gca()
            plt.scatter(np.sqrt(Npix), medians,s = 15, color = 'white', edgecolors = 'blue',
                        alpha = 0.8)
            plt.plot(np.sqrt(x), model(theta_max, x), color = 'grey', linestyle = '--',
                     linewidth = 1)  
            title = os.path.basename(self._cat_name).removesuffix('.hdf5')
            plt.title(title, fontsize = 10)
            plt.xlabel('sqrt(Number of pixels in aperture)')
            plt.ylabel('Noise in aperture [counts]')
            plt.minorticks_on()
            ax.tick_params(axis = 'both', direction = 'in', which = 'both')
            plt.savefig(self._cat_name.replace('.hdf5', '_noise.png'))
            plt.close()

        return labels
    
    def extract(self, science, error, parameters={}, outputs=None, cat_name=None, outdir='./'):
        """
        Perform background subtraction, filtering, detection and source
        photometry on a science image and save to a hdf5 catalogue.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D array of science image values.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        parameters (dict)
            Keys defining parameters to be overwritten in the config file
            and their value.
        outputs (list[str])
            The quantities to output. See Photutils.output_names for 
            available parameters.
        cat_name (None, str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            Directory in which to save the output catalogue.

        Returns
        -------
        cat_name (str)
            Path to the hdf5 file containing the measured photometry.
        """

        # Make a local copy of the config for updating with provided 
        # parameters.
        config = copy.deepcopy(self.config)

        # Update the config with given parameters.
        config.update(parameters)

        # Store the config as is for saving as hdf5 attributes.
        att_config = copy.deepcopy(config)

        # Expand any environment variables and convert string to None.
        for key, value in config.items():
            if type(value) == str:
                config[key] = os.path.expandvars(value)
            if value == 'None':
                config[key] = None

        # Are we in double image mode?
        single_mode = False
        if isinstance(science, list):
            if len(science) == 2:
                print('Starting extraction in double image mode.')
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                    'form [detection, measurement].') 
             
            # Have errors been provided.
            if isinstance(error, list):
                if len(error) != 2:
                    raise ValueError('Double image mode requires a list of weight paths of the '
                                    'form [detection, measurement].')
                
                # If not, warn the user that the background RMS will be 
                # used instead.
                s = [i == None for i in error]  
                if sum(s) == 2:
                    print('No RMS maps provided. Will use measured background RMS for weighting. \n')
                elif sum(s) == 1:
                    print(f'No RMS map provided for {np.array(["detection", "measurement"])[s][0]}.'
                          ' Will use measured background RMS for weighting. \n')
            elif isinstance(error, type(None)):
                print('No RMS maps provided. Will use measured background RMS for weighting. \n')
                error = [error] * 2
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                                'form [detection, measurement], or None for no weighting.') 
            
        # If not, we should be in single image mode.
        elif isinstance(science, str):
            print('Starting extraction in single image mode.')
            single_mode = True
            
            if isinstance(error, type(None)):
                print('No RMS map provided. Will use measured background RMS for weighting. \n')
            elif isinstance(error, str) == False:
                raise ValueError('Single image mode requires a string path to a weight map or None '
                               'for no weighting.') 
            
            # Duplicate the inputs to match double image format.
            science = [science] * 2
            error = [error] * 2

        # If we get here, the inputs are very wrong.   
        else:
            raise ValueError('Image inputs are not the correct format. Use strings for single image '
                           'mode and lists of the form [detection, measurement] for double. '
                           'Use None for no weighting.')

        # Name the catalogue after the measurement image.
        if isinstance(cat_name, type(None)):
            cat_name = f'{outdir}/{os.path.basename(science[1]).removesuffix(".fits")}_photutils.hdf5'
        else:
            cat_name = f'{outdir}/{cat_name.split(".")[0]}.hdf5'
        self._cat_name = cat_name

        # Load detection image.
        print(f'Processing {os.path.basename(science[0])}:')
        sci, hdr = fits.getdata(science[0], header=True)

        # Load RMS map if available.
        if isinstance(error[0], type(None)) == False:
            err = fits.getdata(error[0])
        else:
            err = None

        # Measure the background.
        bkg = self._measure_background(sci, err, config)
        if isinstance(err, type(None)):
            err = bkg.background_rms
        if config['BKG_SUB']:
            sci -= bkg.background

        # Filter the image if required.
        if config['FILTER'] != None:
            sci_filt = self._filter(sci, err, bkg, config)
        else:
            sci_filt = sci

        # Identify the sources and save segmentation map.
        segmap = self._segmentation(sci_filt, err, config)
        if config['SEGMAP'] != None:
            print(f' Saving segmentation map to {outdir}/{config["SEGMAP"]}.fits')
            fits.writeto(f'{outdir}/{config["SEGMAP"]}.fits', segmap.data, hdr, overwrite=True)

        # Get the WCS information from the header.
        wcs = WCS(hdr)

        # Mask the off detector regions.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # Should convolved data be used to measure properties?
        convolved_data = None
        if config['CONVOLVED'] == True:
            if config['FILTER'] != None:
                convolved_data = sci_filt
            else:
                raise ValueError('Requested filtered image be used to measure source properties'
                                ' but filtering is turned off')

        # Measure the properties of the sources.
        cat = SourceCatalog(
            sci, segmap, convolved_data=convolved_data, error=err, mask=mask,
            background=bkg.background, wcs=wcs, localbkg_width=config['LOCALBKG_WIDTH'],
            apermask_method=config['APERMASK_METHOD'], kron_params=config['KRON_PARAMS'],
            detection_cat=None, progress_bar=False)

        # If in double image mode, repeat with the measurement images.
        if not single_mode :
            print(f'Processing {os.path.basename(science[1])}:')
            sci = fits.getdata(science[1])

            # Load RMS map if available.
            if isinstance(error[1], type(None)) == False:
                err = fits.getdata(error[1])
            else:
                err = None

            # Measure the background.
            bkg = self._measure_background(sci, err, config)
            if isinstance(err, type(None)):
                err = bkg.background_rms
            if config['BKG_SUB']:
                sci -= bkg.background

            # Filter the image if required.
            if config['FILTER'] != None:
                sci_filt = self._filter(sci, err, bkg, config)
            else:
                sci_filt = sci

            # Mask the off detector regions.
            mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

            # Should convolved data be used to measure properties?
            convolved_data = None
            if config['CONVOLVED'] == True:
                if config['FILTER'] != None:
                    convolved_data = sci_filt
                else:
                    raise ValueError('Requested filtered image be used to measure source properties'
                                    ' but filtering is turned off')

            # Measure the photometry.
            print('Measuring source properties...')
            cat = SourceCatalog(
                sci, segmap, convolved_data=convolved_data, error=err, mask=mask,
                background=bkg.background, localbkg_width=config['LOCALBKG_WIDTH'],
                detection_cat=cat, progress_bar=False)
        
        # Selection array to only keep objects with a position 
        # measurement.
        s = np.isfinite(cat.xcentroid) & np.isfinite(cat.ycentroid)
        
        # Calculate RA and DEC of the centroids.
        ra_dec = wcs.pixel_to_world(cat.xcentroid, cat.ycentroid)
        cat.add_extra_property('RA', ra_dec.ra.deg)
        cat.add_extra_property('DEC', ra_dec.dec.deg)
        
        # Measure circular aperture photometry if requested.
        labels = []
        for idx, radius in enumerate(config['RADII']):
            cat.circular_photometry(radius, f'APER_{idx}', overwrite=False)
            labels += [f'APER_{idx}_flux', f'APER_{idx}_fluxerr']

        if config['EMPIRICAL']:
            labels_ = self._empirical_uncertainty(sci, err, segmap.data, cat, config)
            labels += labels_

        # Get the full list of avilable outputs.
        output_names = set(self.output_names + labels)

        # Now add everything to the hdf5 catalogue.
        with h5py.File(cat_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')

            # If no outputs requested, use all bar the image cutouts.
            if outputs == None:
                outputs = output_names
                outputs.difference_update(['background', 'convdata', 'data', 'error', 'segment'])
            outputs = set(outputs)

            # Flux values need to be converted to desired units.
            flux_quantities = ['background_centroid', 'background_mean', 'background_sum', 
                               'kron_flux', 'kron_fluxerr', 'local_background', 'max_value',
                               'min_value', 'segment_flux', 'segment_fluxerr'] + labels
            
            # Add quantities to hdf5 catalogue.
            for output in outputs:
                if output in output_names:

                    # Need to get things in the right format.
                    attr = getattr(cat, output)
                    if isinstance(attr, SkyCoord):
                        attr = np.array([list(coord) for coord in zip(attr.ra.deg, attr.dec.deg)])

                    # Add to catalogue.
                    if output in flux_quantities:
                        f[f'photometry/{output}'] = attr[s] * config['CONVERSION']
                    else:
                        f[f'photometry/{output}'] = attr[s]
                else:
                    print(f'Skipping {output} as it is not a recognised output quantity. ' 
                          'Check Photutils.output_names for available outputs.')

            # Add the config parameters as attributes.
            for key, value in att_config.items():
                f['photometry'].attrs[key] = value
            f['photometry'].attrs['CODE'] = 'Photutils'
            f['photometry'].attrs['VERSION'] = photutils.__version__

        print(f'Completed extraction and saved to {cat_name}')

        return cat_name
    
class ProFound():
    """
    Wrapper around the ProFound.R ProFound class (via run_ProFound.R) 
    to allow running through Python.
    """

    def __init__(self, config_file, flags_path='./'):
        """
        __init__ method for ProFound.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        Rfile_path (str)
            Path to the directory containing the ProFound.R and 
            run_ProFound.R scripts.
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
        self.flags_path = flags_path

    def measure_depth(self, science, psf, mask=None, error=None, parameters={}, radius=3.33,
                      max_apers=50, max_iters=50000):
        """
        Measure the 5-sigma point source depth of an image using 
        ProFound. This method simply passes the parameters to 
        wrap_profound.R.
        
        Arguments
        ---------
        science (str)
            Filename of science fits image.
        psf (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask (None, str)
            Filename of the fits image mask. If None, generate and use
            a ProFound segmentation map.
        error (None, str)
            Filename of fits RMS map. If None, no weighting will be 
            used if generating a mask and only NaN non-source pixels will
            be masked.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        radius (float)
            Radius of the random apertures to use in pixels.
        max_apers (int)
            The maximum number of apertures to place.
        max_iters (int)
            The maximun attempts at finding a non overlapping location.

        Returns
        -------
        depth (float)
            The 5-sigma depth of the image.
        """

        # Contruct the base command for running ProFound in depth mode.
        basecmd = [f'Rscript', 'run_ProFound.R', 'type=depth', f'config_path={self.configfile}', 
                   f'flags_path={self.flags_path}', f'img1={science}', f'psf={psf}', 
                   f'radius={radius}', f'max_apers={max_apers}', f'max_iters={max_iters}']
        
        # Add source mask.
        if isinstance(mask, type(None)):
            basecmd.append('mask=None')
        else:
            basecmd.append(f'mask={mask}')
        
        # Add error map.
        if isinstance(error, type(None)):
            basecmd.append('error=None')
        else:
            basecmd.append(f'error={error}')
        
        # Add the overwritten parameters.
        for key, value in parameters.items():
            basecmd.append(f'{key}={value}')

        # Now run on the command line.     
        p = subprocess.Popen(basecmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
        for line in p.stderr:
            print(line)
        out, err = p.communicate()       

        # Get the depth and return it.
        if p.returncode == 0:
            if 'Depth:' in out:
                depth = out.split('Depth:')[1].strip()
                return float(depth)
        else:
            raise RuntimeError('ProFound encountered an error. Check the '
                               'output for further information.')

    def extract(self, science, parameters={}, outputs=None, cat_name=None, outdir='./'):
        """
        Perform source extraction and photometry using Profound. 
        This method simply passes the parameters to wrap_profound.R.

        Arguments
        ---------
        science (str, List[str])
            If str, the filename of the image to extract.
            If a List[str] filename of detection and measurement images.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (list)
            List of output parameters to save.
        cat_name (None, str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            Directory in which to store outputs. 

        Returns
        -------
        out_name (str)
            Path to the generated catalogue.
        """

        # Construct the base command for running ProFound extraction.
        basecmd = [f'Rscript', 'run_ProFound.R', 'type=extract', f'config_path={self.configfile}', 
                   f'flags_path={self.flags_path}']

        # Add the science images.
        if type(science) == list:
            if len(science) == 2:
                basecmd += [f'img1={science[0]}', f'img2={science[1]}']
                name = os.path.basename(science[1]).replace(".fits","_profound.hdf5")
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                    'form [detection, measurement].') 
            
        elif isinstance(science, str):
            basecmd.append(f'img1={science}')
            name = os.path.basename(science).replace(".fits","_profound.hdf5")

        else:
            raise ValueError('Image inputs are not the correct format. Use strings for single image'
                           ' mode and lists of the form [detection, measurement] for double. '
                           'Use None for no weighting.')

        if isinstance(cat_name, type(None)):
            cat_name = name

        out_name = f'{outdir}/{cat_name}'

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
        
        # Add the catalogue name.
        basecmd.append(f'cat_name={cat_name}')

        # Finally the output directory.
        basecmd.append(f'outdir={outdir}')

        # Now run on the command line.     
        p = subprocess.Popen(basecmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
        for line in p.stderr:
            print(line)
        out, err = p.communicate()  

        # Get the catalogue name and return it.
        if p.returncode == 0:
            if 'out_name:' in out:
                out_name = out.split('out_name:')[1].strip()
                return out_name
        else:
            raise RuntimeError('ProFound encountered an error. Check the '
                    'output for further information.')  