import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import Regions

import scipy.ndimage as nd
import scipy.optimize as opt

from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, aperture_photometry

from skimage.measure import block_reduce

def multiband_catalogue(catalogues, labels, new_catalogue = 'combined_catalogue.hdf5'):
    """Combine multiple hdf5 catalogues produced by SExtractor into a single catalogue
    
    Arguments
    ---------
    catalogues (list):
        A list of paths to the catalogues to be combined.
    labels (list):
        List of group names for each band added to the new catalogue.
    new_catalogue (str):
        Name of the new catalogue to output.
    """

    # Create the new catalogue and its base photometry group.
    with h5py.File(new_catalogue, 'w') as newcat:
        newcat.create_group('photometry')

        # Iterate over the different catalogues.
        for label, catalogue in zip(labels, catalogues):

            with h5py.File(catalogue, 'r') as cat:

                # Check for photometry group.
                if 'photometry' not in cat.keys():
                    raise KeyError('Catalogue should have base "photometry" group.')
                
                # Create new group in catalogue to store the data for this band.
                group = f'photometry/{label}'
                newcat.create_group(group)

                # Add each dataset to the new catalogue.
                for key in cat['photometry'].keys():
                    # Have consistent naming convention for apertures.
                    if (key == 'FLUX_APER') or (key == 'FLUXERR_APER'):
                        newcat[f'{group}/{key}_0'] = cat[f'photometry/{key}'][()]
                    else:
                        newcat[f'{group}/{key}'] = cat[f'photometry/{key}'][()]
                # And copy over the config information.
                for key in cat['photometry'].attrs.keys():
                    newcat[f'{group}'].attrs[key] = cat['photometry'].attrs[key]
    
    print(f'Created a multiband catalogue and save to {new_catalogue}')

    return

def weight_to_error(weight_image, error_filename = None):
    """Convert a weight image to an error map.

    Arguments
    ---------
    weight_image (str):
        Path to the weight image to convert.
    error_filename (None, str):
        Name of the error image to output. If None, append "to_err" to weight filename.
    """

    # Load the weight image and header.
    wht, hdr = fits.getdata(weight_image, header = True)

    # Convert the weight image to error map.
    err = np.where(wht==0, np.nan, 1/np.sqrt(wht))

    # If no name for new file given, use weight filename as base.
    if error_filename == None:
        error_filename = f'{weight_image.remove(".fits")}_to_err.fits'

    # Add header keyword to indicate how the error map was generated.
    hdr['CONVERTED'] = ('T', 'Converted to error image from weight image.')

    # Save the error image to a new file.
    fits.writeto(error_filename,err,header=hdr,overwrite=True)

    return

def error_to_weight(error_image, weight_filename = None):
    """Convert an error image to an weight map.

    Arguments
    ---------
    error_image (str):
        Path to the error image to convert.
    weight_filename (None, str):
        Name of the weight image to output. If None, append "to_wht" to error filename.
    """

    # Load the weight image and header.
    err, hdr = fits.getdata(error_image, header = True)

    # Convert the weight image to error map.
    wht = np.where(np.isnan(err), 0, 1/(err**2))

    # If no name for new file given, use weight filename as base.
    if weight_filename == None:
        weight_filename = f'{error_image.remove(".fits")}_to_wht.fits'

    # Add header keyword to indicate how the error map was generated.
    hdr['CONVERTED'] = ('T', 'Converted to weight image from error image.')

    # Save the error image to a new file.
    fits.writeto(weight_filename,wht,header=hdr,overwrite=True)

    return

def generate_error(science, weight, exposure, outname = None):
    """Generate an error image including Poisson noise from science, weight and exposure images.
        Assumes exposure map is downsampled by factor 4 relative to science and weight images.

    Arguments
    ---------
    science (str):
        Path to the science image.
    weights (str):
        Path to the weight image.
    exposure (str):
        Path to the exposure time map.
    outname (None, str):
        Output filename of the generated error image. If none, append "_err" to science filename.
    """

    if outname == None:
        outname = f'{science.remove_suffix(".fits")}_err.fits'

    sci = fits.getdata(science)
    exp, exp_header = fits.getdata(exposure, header = True)
    wht, wht_header = fits.getdata(weight, header = True)

    # Grow the exposure map to the original frame.
    full_exp = np.zeros(sci.shape, dtype=int)
    full_exp[2::4,2::4] += exp*1
    full_exp = nd.maximum_filter(full_exp, 4)

    # Determine multiplicative factors that have been applied since the original count-rate images.
    phot_scale = 1.

    for k in ['PHOTMJSR','PHOTSCAL']:
        print(f'{k} {exp_header[k]:.3f}')
        phot_scale /= exp_header[k]

    # Unit and pixel area scale factors
    if 'OPHOTFNU' in exp_header:
        phot_scale *= exp_header['PHOTFNU'] / exp_header['OPHOTFNU']

    # "effective_gain" = electrons per DN of the mosaic
    effective_gain = (phot_scale * full_exp)

    # Poisson variance in mosaic DN
    var_poisson_dn = np.maximum(sci, 0) / effective_gain

    # Original variance from the `wht` image = RNOISE + BACKGROUND
    var_wht = 1/wht

    # New total variance
    var_total = var_wht + var_poisson_dn
    full_wht = 1 / var_total

    # Null weights
    full_wht[var_total <= 0] = 0

    err = np.where(full_wht==0, 0, 1/np.sqrt(full_wht))

    fits.writeto(outname, err, header = wht_header)

    return

def pc2cd(hdr, key=' '):
    """Convert a PC matrix to a CD matrix. Used when rebinning to preserve WCS information.

    Arguments
    ---------
    hdr: (astropy.io.fits.Header)
        Header containing PC matrix to be converted.
    key: (str)
        Additional key attached to the PC keywords.
    """

    key = key.strip()
    cdelt1 = hdr.pop(f'CDELT1{key:.1s}', 1)
    cdelt2 = hdr.pop(f'CDELT2{key:.1s}', 1)
    hdr[f'CD1_1{key:.1s}'] = (cdelt1 * hdr.pop(f'PC1_1{key:.1s}', 1),
                              'partial of first axis coordinate w.r.t. x')
    hdr[f'CD1_2{key:.1s}'] = (cdelt1 * hdr.pop(f'PC1_2{key:.1s}', 0),
                              'partial of first axis coordinate w.r.t. y')
    hdr[f'CD2_1{key:.1s}'] = (cdelt2 * hdr.pop(f'PC2_1{key:.1s}', 0),
                              'partial of second axis coordinate w.r.t. x')
    hdr[f'CD2_2{key:.1s}'] = (cdelt2 * hdr.pop(f'PC2_2{key:.1s}', 1),
                              'partial of second axis coordinate w.r.t. y')
    return hdr

def rebin_image(input_fits, source_scale, target_scale, method = 'sum', outname = None):
    """Rebin an image to a lower resolution pixel scale using integer scaling.
    
    Arguments
    ---------
    input_fits: (str)
        The image to be rebinned.
    source_scale: (float)
        The pixel scale of the input image in arcseconds.
    target_scale: (float)
        The output pixel scale in arcseconds.
    method: (str)
        The method to use when summing pixels. Either 'sum' for linear or 'quad' for quadratic.
    outname: (None, str)
        """

    print(f'Rebinning {input_fits}...')
    if outname == None:
        outname = f'{input_fits.removesuffix(".fits")}_rebinned.fits'

    # Read the FITS file
    with fits.open(input_fits) as hdul:
        img = hdul[1].data
        wcs = WCS(hdul[1].header)

    # Calculate the scale factor for resizing
    scale_factor = int(target_scale/source_scale)

    # Create a new WCS for the rebinned image
    wcs_rebinned = wcs.slice((np.s_[:None:int(scale_factor)], np.s_[:None:int(scale_factor)]))
    wcs_header = wcs_rebinned.to_header()
    pc2cd(wcs_header)

    # Define the block size for rebinning.
    block_size = (scale_factor, scale_factor)

    print(block_size)

    # Rebin the image.
    if method == 'sum':
        rebinned_image = block_reduce(img, block_size, np.sum)
    if method == 'quad':
        rebinned_image = np.sqrt(block_reduce(img**2, block_size, np.sum))

    # Save the rebinned image as a new FITS file with updated WCS information
    hdu = fits.PrimaryHDU(rebinned_image)
    hdu.header.update(wcs_header)
    hdul_rebinned = fits.HDUList([hdu])
    hdu.header["REBIN"] = (source_scale, 'Image has been rebinned from this scale')
    hdul_rebinned.writeto(outname, overwrite=True)

    print(f'Rebinned and saved to {outname}.')

    return

def create_stack(sci_images, wht_images, stack_name = 'stacked_image.fits'):
    """Create a stacked image for detection.

    Arguments
    ---------
    sci_images (list):
        A list of science image file paths to stack.
        The header used in the final images will be taken from the first image in the list.
    wht_images (list):
        A list of corresponding weight image file paths.
    stack_name (str):
        Filename of the output stacked image.
        Suffixes _sci and _wht will be added before the fits extension.
    """

    if len(sci_images) != len(wht_images):
        raise KeyError('The number of science and weight images must be equal.')

    # Get the image size and headers from the first image.
    first_image, sci_hdr = fits.getdata(sci_images[0], header=True)
    wht_hdr = fits.getheader(wht_images[0])

    shape = first_image.data.shape
    stack_sci = np.zeros(shape)
    stack_wht = np.zeros(shape)

    # Stack the images. 
    for sci, wht in zip(sci_images, wht_images):
        wht_ = fits.getdata(wht)
        stack_sci += fits.getdata(sci) * wht_
        stack_wht += wht_

    stack_sci /= stack_wht

    # Save the images.
    fits.writeto({stack_name.replace('.fits', '_sci.fits')}, stack_sci, header = sci_hdr, overwrite=True)
    fits.writeto({stack_name.replace('.fits', '_wht.fits')}, stack_wht, header = wht_hdr, overwrite=True)

    return

def Gaussian_2D(coord, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """2D Gaussian fitting function. Use for determining FWHM of PSFs.
    
    Arguments
    ---------
    coord (list):
        The (x,y) coordinate at which to evaluate the Gaussian.
    xo (float):
        The x-coordinate of the centre.
    yo (float):
        The y-coordinate of the centre.
    sigma_x (float):
        Standard deviation in pixels in the x direction.
    sigma_y (float):
        Standard deviation in pixels the y direction.
    amplitude (float):
        Amplitude of the gaussian.
    offset (float):
        Offset to apply to the Gaussian values.

    Returns
    -------
    Gaussian (1D array):
        Flattened Gaussian distribution.
    """

    g = offset + amplitude*np.exp( - (((coord[0]-float(xo))**2)/(2*sigma_x**2) + ((coord[1]-float(yo))**2)/(2*sigma_y**2)))

    return g.ravel()

def get_PSF_FWHM(psf_path):
    """Get FWHM in x and y directions of a PSF using Gaussian fitting.

    Arguments
    ---------
    psf_path (str):
        Path to PSF array fits file.

    Returns
    -------
    FWHM (list):
        List containing measured FWHM in x and y directions.
    """

    # Read the PSF array from the fits file.
    img = fits.getdata(psf_path)

    # Create an x and y grid.
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Some parameter inital guesses
    initial_guess = [img.shape[1]/2,img.shape[0]/2,10,10,1,0]

    # Fit with a Gaussian model.
    popt, pcov = opt.curve_fit(Gaussian_2D, (x, y), 
                               img.ravel(), p0 = initial_guess)
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]

    # Convert the standard deviations to FWHM.
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))

    return [FWHM_x, FWHM_y]

def powspace(start, stop, power=0.5, num = 30, **kwargs):
    """Generate a square-root spaced array with a specified number of points between two endpoints.
        Used to measure PSF curves of growth.

    Parameters
    ----------
    start (float):
        The starting value of the range.
    stop (float):
        The ending value of the range.
    power (float): 
        Power of distribution, defaults to sqrt.
    num (int):
        The number of points to generate in the array. Default is 30.

    Returns
    -------
    Array (1D array):
        A 1D array of 'num' values spaced equally in square-root space
        between 'start' and 'stop'.
    """

    return np.linspace(start**power, stop**power, num=num, **kwargs)**(1/power)

def measure_curve_of_growth(image, position=None, radii=None, rnorm='auto', nradii=30, rbackg=True, showme=False, verbose=False):
    """Measure a curve of growth from cumulative circular aperture photometry on a list of radii 
        centered on the center of mass of a source in a 2D image.

    Parameters
    ----------
    image (numpy.ndarray):
        2D image array.
    position (astropy.coordinates.SkyCoord/None):
        Position of the centre of the source. If 'None', it will be measured.
    radii (numpy.ndarray/None):
        Array of aperture radii. If None, use 'nradii'.
    rnorm (float):
        The radius to use for normalisation. Must be in 'radii'.
    nradii (int):
        Number of aperture radii to get from self.powspace. Only used if radii==None.
    rbackg (bool):
        Whether to perform backgound subtraction. 
    showme (bool):
        Whether to save COG and profile figure. 
    verbose (bool):
        Whether to print progress information.
    
    Returns
    -------
    radii (numpy.ndarray):
        The aperture radii used.
    cog (numpy.ndarray):
        The measured curve of growth.
    profile (numpy.ndarray):
        The measured profile.
    """

    # Default to a sqaure root spaced array.
    if type(radii) is type(None):
        radii = powspace(0.5,image.shape[1]/2,num=nradii)

    # Calculate the centroid of the source in the image if not given.
    if type(position) is type(None):
        position = centroid_com(image)

    # Create an aperture for each radius in radii.
    apertures = [CircularAperture(position, r=r) for r in radii]

    # Remove background if requested.
    if rbackg == True:
        bg_mask = apertures[-1].to_mask().to_image(image.shape) == 0
        # Background is median of image unmasked by apertures.
        bg = np.nanmedian(image[bg_mask])
        if verbose: print('background',bg)
    else:
        bg = 0.

    # Perform aperture photometry for each aperture
    phot_table = aperture_photometry(image-bg, apertures)
    # Calculate cumulative aperture fluxes
    cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

    # Normalise at some radius.
    if rnorm == 'auto': rnorm = image.shape[1]/2.0
    if rnorm:
        rnorm_indx = np.searchsorted(radii, rnorm)
        cog /= cog[rnorm_indx]

    # Get the profile.
    area = np.pi*radii**2 # Area enclosed by each apperture.
    area_cog = np.insert(np.diff(area),0,area[0]) # Difference between areas.
    profile = np.insert(np.diff(cog),0,cog[0])/area_cog # Difference between COG elements.
    profile /= profile.max() # Normalise profile.

    # Show the COG and profile if requested.
    if showme:
        plt.scatter(radii, cog, s = 25, alpha = 0.7)
        plt.plot(radii,profile/profile.max())
        plt.xlabel('Radius [pix]')
        plt.ylabel('Curve of Growth')

    # Return the aperture radii, COG and profile.
    return radii, cog, profile

def flag_stars(catalogue, PSF_FWHM, efficiency = 0.15, mag_limit = 26.5, min_bands = 2, mag_name = 'MAG_AUTO', fwhm_name = 'FWHM_WORLD'):

    with h5py.File(catalogue, 'r+') as f:

        # Get list of instruments.
        instruments = list(f['photometry'].keys())
        # Remove the detection image.
        if 'detection' in instruments:
            instruments.remove('detection')


        # Get list of filters.
        bands = []
        for instrument in instruments:
            bands_ = list(f[f'photometry/{instrument}'].keys())
            for i in bands_:
                bands_[bands_.index(i)] = f'{instrument}/{i}'
            bands += bands_
        
        stars = []
        for band in bands:
            mag = f[f'photometry/{band}/{mag_name}'][:]
            fwhm = f[f'photometry/{band}/{fwhm_name}'][:] * 3600 # Converting to arcseconds.

            # Spurious sources also have FWHM less than that of PSF.
            below_psf = (fwhm < PSF_FWHM)

            # Selection region for stars.
            s = ~below_psf & (fwhm< PSF_FWHM * (1 + efficiency)) & (mag < mag_limit)

            # Create selection array for stars.
            if type(stars) == list:
                stars = s.astype(int)
            else:
                stars += s.astype(int)
        
        # Only select stars identified as such in the required number of bands.
        selection = stars >= min_bands
        print(f'Identified {sum(selection)} stars.')

        # Add a star flag to the catalogue.
        if 'STAR' in f['photometry/detection'].keys():
            del f['photometry/detection/STAR']
        f['photometry/detection/STAR'] = selection

        return

def match_gaia_sources(catalogue, bands, gaia_catalogue, tolerance, ra_name = 'ALPHA_SKY', dec_name = 'DELTA_SKY'):
    """Match sources in a GAIA star catalogue to those in an hdf5 file
    
    Arguments
    ---------
    catalogue (str):
        Path to hdf5 catalogue containing the sources to be matched.
    bands (list):
        List containing the bands (catalogue groups) to match.
    gaia_catalogue (str):
        Path to the fits catalogue containing GAIA star information.
    tolerance (float):
        The matching tolerance in arcseconds.
    ra_name (str): 
        Name of the RA dataset in the catalogue.
    dec_name (str):
        Name of the DEC dataset in the catalogue."""

    # Calculate the matching tolerance in arcseconds.
    arcsec_tolerance = tolerance*u.arcsec

    # Load the GAIA catalogue and get source positions.
    gaia_cat = Table.read(gaia_catalogue)
    gaia_coord = SkyCoord(ra=np.array(gaia_cat['ra'])*u.degree, dec=np.array(gaia_cat['dec'])*u.degree)

    # For each band requested.
    with h5py.File(catalogue, 'r+') as cat:

        for band in bands:

            # Arrays for storing the star quasar and parallax flags.
            star_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))
            quasar_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))
            p_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))

            # Get the positions of the sources.
            cat_coord = SkyCoord(ra=cat[f'photometry/{band}/{ra_name}'][:]*u.degree, dec=cat[f'photometry/{band}/{dec_name}'][:]*u.degree)

            # Find the source closest to each gaia source.
            idx, d2d, d3d = gaia_coord.match_to_catalog_sky(cat_coord)
            d2d = d2d.to('arcsec')

            # Only accept matches within the given tolerance.
            s = (d2d < arcsec_tolerance)

            # Add the parallax measurement/error, star and quasar classification to the catalogue.
            for index, class_s in zip(idx[s], gaia_cat['classprob_dsc_combmod_star'][s]):
                star_flag[index] = class_s
            for index, class_q in zip(idx[s], gaia_cat['classprob_dsc_combmod_quasar'][s]):
                quasar_flag[index] = class_q
            for index, p_over_e in zip(idx[s], gaia_cat['parallax_over_error'][s]):
                p_flag[index] = p_over_e

            # Add the flag array
            if 'GAIA_STAR' in cat[f'photometry/{band}'].keys():
                del cat[f'photometry/{band}/GAIA_STAR']
            cat[f'photometry/{band}/GAIA_STAR'] = star_flag

            if 'GAIA_QUASAR' in cat[f'photometry/{band}'].keys():
                del cat[f'photometry/{band}/GAIA_QUASAR']
            cat[f'photometry/{band}/GAIA_QUASAR'] = quasar_flag

            if 'GAIA_POE' in cat[f'photometry/{band}'].keys():
                del cat[f'photometry/{band}/GAIA_POE']
            cat[f'photometry/{band}/GAIA_POE'] = p_flag
    
    return

def mask_regions(image, regions, outname = None):
    """Convert a DS9 region file to an image mask.
    
    Arguments
    ---------
    image (str):
        Path to fits image containing the image to be masked.
    regions (str):
        Path to DS9 ".reg" file containing the masking regions.
    outname (None, str):
        Output name for the generated mask. If None, append "_mask" to image name.
    """

    # Set the output name.
    if outname == None:
        outname = image.replace(".fits", "_mask.fits")

    # Extract the WCS information from the image being masked
    img, hdr = fits.getdata(image, header=True)
    wcs = WCS(hdr)

    # and read the region file.
    regions = Regions.read(regions, format='ds9')

    # Use WCS to convert regions to pixel coordinates.
    pixcoords = [scoord.to_pixel(wcs) for scoord in regions]
    combined_region = pixcoords[0]

    # Create one unified region and convert it to a mask.
    for region in pixcoords[1:]:
        combined_region = combined_region.union(region)

    mask = combined_region.to_mask()
    mask = mask.to_image(shape = img.shape)

    # Save as a fits file.
    fits.writeto(outname, mask, hdr)

    return

def flag_mask(catalogue, mask, bands, label = 'MASK', X_name = 'X_IMAGE', Y_name = 'Y_IMAGE'):
    """Flag sources with centres within a masked region.

    WARNING: Assume SExtractor coordinates so X -> Y, Y -> X.
        Can be overwritten be specifying X_name, Y_name accordingly.
    
    Arguments
    ---------
    catalogue (str):
        Path to hdf5 catalogue with sources to be masked.
    mask (str):
        Path to fits file containing the mask
    bands (list): 
        List of photometry sub groups to be considered.
    label (str):
        Name to use for the flag dataset.
    X_name (str):
        Name of the X coordinate dataset.
    Y_name (str):
        Name of the Y coordinate dataset."""

    # Read in the catalogue.
    with h5py.File(catalogue, 'r+') as f:

        # For each band.
        for band in bands:

            # Round object centres to the nearest pixel.
            xcen = np.round(f[f'photometry/{band}/{X_name}'][:])
            ycen = np.round(f[f'photometry/{band}/{Y_name}'][:])

            flag = []

            # If the centre of an object is within the edge region, flag it.
            for x, y in zip(xcen, ycen):
                if mask[int(y), int(x)] == True:
                    flag.append(1)
                else:
                    flag.append(0)

            # Add the flag to catalogue, remvoing any previous iteration.
            if label in f[f'photometry/{band}'].keys():
                del f[f'photometry/{band}/{label}']
            f[f'photometry/{band}/{label}'] = flag

    return