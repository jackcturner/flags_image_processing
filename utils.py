import os
import h5py
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
import random
import math

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import Regions

import scipy.ndimage as nd
import scipy.optimize as opt
import scipy.stats as st
from scipy.spatial import cKDTree

from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, aperture_photometry

from skimage.measure import block_reduce

from ned_extinction_calc import request_extinctions

def poisson_confidence_interval(counts, p=0.68):
    """ 
    Return the upper and lower Poisson confidence limits on a count.
    
    Arguments
    ---------
    counts (numpy.ndarray)
        1D array of counts.
    p (float)
        The confidence limit to return.
        
    Returns
    -------
    intervals (numpy.ndarray)
        2D array of upper and lower confidence limits.
    """
    
    lower = []
    upper = []

    for n in counts:
    
        if n>0:   
            interval=(st.chi2.ppf((1.-p)/2.,2*n)/2.,st.chi2.ppf(p+(1.-p)/2.,2*(n+1))/2.)       
        
        else:
            
            #this bit works out the case for n=0
            
            ul=(1.-p)/2.
            
            prev=1.0
            for a in np.arange(0.,5.0,0.001):
            
                cdf=st.poisson.cdf(n,a)
            
                if cdf<ul and prev>ul:
                    i=a
            
                prev=cdf
            
            interval=(0.,i)
        
        lower.append(interval[0])
        upper.append(interval[1])
    
    intervals = np.column_stack([lower, upper])
    
    return intervals

def merge_catalogues(catalogues, labels, new_catalogue = 'merged_catalogue.hdf5'):
    """
    Combine multiple hdf5 catalogues produced by SExtractor into a single 
    catalogue where each catalogue has its own group within the
    photometry group.
    
    Arguments
    ---------
    catalogues (List[str])
        A list of paths to the catalogues to be combined.
    labels (List[str])
        List of group names for each catalogue added to the new
        catalogue.
    new_catalogue (str)
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
                    raise KeyError(('Catalogue should have a base "photometry" group.'
                                    'Was this catalogue produced by FLAGS?'))
                
                # Create new group in catalogue to store the data for this catalogue.
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
    
    print(f'Merged {len(catalogues)} and saved to {new_catalogue}')

    return

def weight_to_error(weight_image, error_filename = None):
    """
    Convert a weight image to an error map.

    Arguments
    ---------
    weight_image (str)
        Path to the weight fits image to convert.
    error_filename (None, str):
        Name of the error fits image to output.
        If None, append "_to_err" to weight filename.
    """

    # Load the weight image and header.
    wht, hdr = fits.getdata(weight_image, header = True)

    # Convert the weight image to error map.
    err = np.where(wht==0, np.nan, 1/np.sqrt(wht))

    # If no name for new file given, use weight filename as base.
    if error_filename == None:
        error_filename = f'{weight_image.remove(".fits")}_to_err.fits'

    # Add header keyword to indicate how the error map was generated.
    hdr['FROMWHT'] = ('T', 'Converted to RMS from weight.')

    # Save the error image to a new file.
    fits.writeto(error_filename , err.astype(np.float32), header = hdr, overwrite = True)

    return

def error_to_weight(error_image, weight_filename = None):
    """
    Convert an error image to an weight map.

    Arguments
    ---------
    error_image (str)
        Path to the error fits image to convert.
    weight_filename (None, str)
        Name of the weight fits image to output.
        If None, append "_to_wht" to error filename.
    """

    # Load the error image and header.
    err, hdr = fits.getdata(error_image, header = True)

    # Convert the error image to a weight map.
    wht = np.where(np.isnan(err), 0, 1/(err**2))

    # If no name for new file given, use error filename as base.
    if weight_filename == None:
        weight_filename = f'{error_image.remove(".fits")}_to_wht.fits'

    # Add header keyword to indicate how the weight map was generated.
    hdr['FROMERR'] = ('T', 'Converted to weight from RMS.')

    # Save the weight image to a new file.
    fits.writeto(weight_filename, wht.astype(np.float32), header = hdr, overwrite = True)

    return

def generate_error(science, weight, exposure, grow = True, outname = None):
    """
    Generate an error map including Poisson noise from science, weight
    and exposure images. See:
    https://dawn-cph.github.io/dja/blog/2023/07/18/image-data-products/

    Arguments
    ---------
    science (str)
        Path to the science fits image.
    weights (str)
        Path to the weight fits image.
    exposure (str):
        Path to the exposure fits image.
    grow (bool)
        Is the exposure image downsampled by a factor 4. (For DJA images)
    outname (None, str):
        Output filename of the generated error fits image.
        If none, append "_err" to science filename.
    """

    # Set output filename.
    if outname == None:
        outname = f'{science.remove_suffix(".fits")}_err.fits'

    # Load in each image.
    sci = fits.getdata(science)
    exp, exp_header = fits.getdata(exposure, header = True)
    wht, wht_header = fits.getdata(weight, header = True)

    # Grow the exposure map to the original frame if required.
    if grow == True:
        full_exp = np.zeros(sci.shape, dtype=int)
        full_exp[2::4,2::4] += exp*1
        full_exp = nd.maximum_filter(full_exp, 4)

    # Determine multiplicative factors that have been applied since the
    # original count-rate images.

    phot_scale = 1.

    for k in ['PHOTMJSR','PHOTSCAL']:
        print(f'{k} {exp_header[k]:.3f}')
        phot_scale /= exp_header[k]

    # Unit and pixel area scale factors.
    if 'OPHOTFNU' in exp_header:
        phot_scale *= exp_header['PHOTFNU'] / exp_header['OPHOTFNU']

    # Electrons per DN of the mosaic.
    effective_gain = (phot_scale * full_exp)

    # Poisson variance in mosaic DN.
    var_poisson_dn = np.maximum(sci, 0) / effective_gain

    # Original variance from the weight image.
    var_wht = 1/wht

    # New total variance.
    var_total = var_wht + var_poisson_dn
    full_wht = 1 / var_total

    # Null weights.
    full_wht[var_total <= 0] = 0

    # Convert the full weight image to an error map
    err = np.where(full_wht==0, 0, 1/np.sqrt(full_wht))

    # and save to a fits image.
    fits.writeto(outname, err.astype(np.float32), header = wht_header)

    return

def pc2cd(hdr, key=' '):
    """
    Convert a PC matrix to a CD matrix.

    Arguments
    ---------
    hdr (astropy.io.fits.Header)
        Astropy header containing PC matrix to be converted.
    key (str)
        Additional key attached to the PC keywords.

    Returns
    -------
    hdr (astropy.io.fits.Header)
        Astropy table including generated CD matrix.
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
    """
    Rebin an image to a lower resolution pixel scale by combining pixels.
    
    Arguments
    ---------
    input_fits (str)
        Filename of the fits image to be rebinned.
    source_scale (float)
        The pixel scale of the input image in arcseconds.
    target_scale (float)
        The desired output pixel scale in arcseconds.
    method (str)
        The method to use when combining pixels.
        Either 'sum' for linear or 'quad' for quadratic.
    outname (None, str)
        Filename of the rebinned image.
        If None, append "_rebinned" to input filename.
    """

    print(f'Rebinning {input_fits}...')

    # Set the output filename.
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

    # Rebin the image.
    if method == 'sum':
        rebinned_image = block_reduce(img, block_size, np.sum)
    if method == 'quad':
        rebinned_image = np.sqrt(block_reduce(img**2, block_size, np.sum))

    # Save the rebinned image as a new FITS file with updated WCS information
    hdu = fits.PrimaryHDU(rebinned_image.astype(np.float32))
    hdu.header.update(wcs_header)
    hdul_rebinned = fits.HDUList([hdu])
    hdu.header["REBIN"] = (source_scale, 'Image has been rebinned from this scale')
    hdul_rebinned.writeto(outname, overwrite=True)

    print(f'Rebinned and saved to {outname}.')

    return

def create_stack(sci_images, wht_images, hdr_index = 0, stack_name = 'stacked_image.fits'):
    """Create a variance weighted stacked image for source detection.

    Arguments
    ---------
    sci_images (List[str])
        Filenames of science images to stack.
    wht_images (List[str])
        Filenames of the corresponding weight images.
    hdr_index (int)
        Index into sci_images. Use the header from this element in the
        final image.
    stack_name (str)
        Filename of the output stacked image.
        Suffixes _sci and _wht will be added before the fits extension.
    """

    if len(sci_images) != len(wht_images):
        raise KeyError('The number of science and weight images must be equal.')

    # Get the image size and headers from the first image.
    img, sci_hdr = fits.getdata(sci_images[hdr_index], header=True)
    wht_hdr = fits.getheader(wht_images[hdr_index])

    shape = img.data.shape
    stack_sci = np.zeros(shape)
    stack_wht = np.zeros(shape)

    # Stack the images. 
    for sci, wht in zip(sci_images, wht_images):
        wht_ = fits.getdata(wht)
        stack_sci += fits.getdata(sci) * wht_
        stack_wht += wht_

    stack_sci /= stack_wht

    # Save the images.
    fits.writeto({stack_name.replace('.fits', '_sci.fits')}, stack_sci.astype(np.float32), header = sci_hdr, overwrite=True)
    fits.writeto({stack_name.replace('.fits', '_wht.fits')}, stack_wht.astype(np.float32), header = wht_hdr, overwrite=True)

    return

def Gaussian_2D(coord, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """
    2D Gaussian fitting function.
    
    Arguments
    ---------
    coord (List[float])
        The x,y coordinate at which to evaluate the Gaussian.
    xo (float)
        The x-coordinate of the centre.
    yo (float)
        The y-coordinate of the centre.
    sigma_x (float)
        Standard deviation in the x direction in pixels.
    sigma_y (float)
        Standard deviation in the y direction in pixels.
    amplitude (float)
        Amplitude of the gaussian.
    offset (float)
        Offset to apply to the Gaussian values.

    Returns
    -------
    flat_gaussian (numpy.ndarray)
        1D flattened Gaussian distribution.
    """

    gaussian = offset + amplitude*np.exp( - (((coord[0]-float(xo))**2)/(2*sigma_x**2)
                                             + ((coord[1]-float(yo))**2)/(2*sigma_y**2)))

    flat_gaussian = gaussian.ravel()

    return flat_gaussian

def get_PSF_FWHM(psf_path):
    """
    Get FWHM of a PSF in x and y directions using Gaussian fitting.

    Arguments
    ---------
    psf_path (str)
        Filename of fits image PSF.

    Returns
    -------
    fwhm (List[float])
        Measured FWHM in x and y directions.
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

    fwhm = [FWHM_x, FWHM_y]

    return fwhm

def measure_curve_of_growth(image, radii, position=None, norm=True, show=False):
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
    cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

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

def flag_stars(catalogue, PSF_FWHM, efficiency=0.15, mag_limit=26.5, min_bands=2,
               mag_name='MAG_AUTO', fwhm_name='FWHM_WORLD', flag_name = 'STAR'):
    """
    Identify a flag stars based on SExtractor measurements and the 
    FWHM of the PSF.
    
    Arguments
    ---------
    catalogue (str)
        Path to hdf5 catalogue file produced by FLAGS.
    PSF_FWHM (float)
        FWHM of the image PSF this catalogue was determined from.
    efficiency (float)
        The efficiency of star identification.
    mag_limit (float)
        Stars must be brighter than this limit.
    min_bands (int)
        Objects must be identified as stars in this many bands.
    mag_name (str)
        Name of the magnitude dataset.
    fwhm_name (str)
        Name of the FWHM dataset.
    flag_name (str)
        Name of the star flag dataset to create.
    """

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
        if flag_name in f['photometry/detection'].keys():
            del f[f'photometry/detection/{flag_name}']
        f[f'photometry/detection/{flag_name}'] = selection

        return

def match_gaia_sources(catalogue, bands, gaia_catalogue, tolerance, ra_name='ALPHA_SKY',
                       dec_name='DELTA_SKY'):
    """
    Match and flag sources in a FLAGS catalogue to those in a GAIA
    star catalogue
    
    Arguments
    ---------
    catalogue (str)
        Filename of FLAGS hdf5 catalogue.
    bands (List[str])
        List containing the catalogue groups to match.
    gaia_catalogue (str)
        Filename of the fits GAIA star catalogue.
    tolerance (float)
        The matching tolerance in arcseconds.
    ra_name (str):
        Name of the RA dataset in the catalogue.
    dec_name (str)
        Name of the DEC dataset in the catalogue.
    """

    # Calculate the matching tolerance in arcseconds.
    arcsec_tolerance = tolerance*u.arcsec

    # Load the GAIA catalogue and get source positions.
    gaia_cat = Table.read(gaia_catalogue)
    gaia_coord = SkyCoord(ra=np.array(gaia_cat['ra'])*u.degree,
                          dec=np.array(gaia_cat['dec'])*u.degree)

    # For each band requested.
    with h5py.File(catalogue, 'r+') as cat:

        for band in bands:

            # Arrays for storing the star quasar and parallax flags.
            star_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))
            quasar_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))
            p_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))

            # Get the positions of the sources.
            cat_coord = SkyCoord(ra=cat[f'photometry/{band}/{ra_name}'][:]*u.degree,
                                 dec=cat[f'photometry/{band}/{dec_name}'][:]*u.degree)

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

            # Add the flag array.
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

def regions_to_mask(image, regions, outname = None):
    """
    Convert a DS9 region file to an image mask.
    
    Arguments
    ---------
    image (str)
        Filename of fits image to be masked.
    regions (str)
        Path to DS9 ".reg" file containing the masking regions.
    outname (None, str)
        Output name for the generated mask.
        If None, append "_mask" to image name.
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
    if len(pixcoords) > 1:
        for region in pixcoords[1:]:
            combined_region = combined_region.union(region)

    mask = combined_region.to_mask()
    mask = mask.to_image(shape = img.shape)

    # Save as a fits file.
    fits.writeto(outname, mask.astype(np.int32), hdr, overwrite = True)

    return

def create_edge_mask(images, off_image=0, buffer_size=5, threshold=0.1, n_pixels=50,
                     outname='combined_edge_mask.fits'):
    """
    Use binaray hole filling and sobel filters to identify and mask
    image edges and merge multiple mask into a single combined mask.
    
    Arguments
    ---------
    images (str, List[str])
        The image(s) for which to create the edge mask.
    off_image (float)
        The value indicating an off detector region.
    buffer_size (int)
        The width of buffer around the image array edge within which to 
        ignore edges. 
    threshold (float)
        Threshold for edge identification.
    n_pixels (int)
        Number of  pixels to use when dilating the edge mask.
    outname (str)
        Filename for the saved edge mask.

    Returns
    -------
    combined_mask (numpy.ndarray)
        2D array where True indicates an edge in one of the 
        provided images.
    """

    # Convert string to list if required.
    if type(images) == str:
        images = [images]

    masks = []
    for image in images:

        print(f'Finding edges in {image}...')

        sci , hdr = fits.getdata(image, header=True)

        # Fill any holes that may be identified as edges.
        data = nd.binary_fill_holes(sci)

        # Identify off-image regions
        off_image_mask = (data == off_image)

        # Do not create a mask if the edge identified is with this many
        # pixels of the image edge.
        # Can create spurious masks if not set.
        buffer_mask = np.zeros_like(data, dtype=bool)
        buffer_mask[:buffer_size, :] = True  # Top buffer
        buffer_mask[-buffer_size:, :] = True  # Bottom buffer
        buffer_mask[:, :buffer_size] = True  # Left buffer
        buffer_mask[:, -buffer_size:] = True  # Right buffer

        # Detect the edges.
        edges_x = nd.sobel(data, axis=0)
        edges_y = nd.sobel(data, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)

        # Reequire a minimum threshold.
        edge_mask = edges > threshold

        # Combine off-image mask and dilated edge mask
        comb_mask = np.logical_and(off_image_mask, edge_mask)
        # Exclude buffer zone from masking
        comb_mask[buffer_mask] = False  

        # Dilate the edge mask
        final_mask = nd.binary_dilation(comb_mask, iterations=n_pixels) 

        masks.append(final_mask)
    
    # Merge masks if required.
    if len(masks) > 1:
        print('Merging...')
        combined_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask)
    else:
        combined_mask = masks[0]
    combined_mask = combined_mask.astype(np.uint8)

    hdr = fits.getheader(images[0])
    fits.writeto(outname, combined_mask.astype(np.float32), hdr, overwrite = True)

    return combined_mask

def flag_mask(catalogue, mask, bands, label='MASK', X_name='X_IMAGE', Y_name='Y_IMAGE'):
    """Flag sources with centres within a masked region.

    WARNING: Assumes SExtractor coordinates so X -> Y, Y -> X.
        Can be overwritten be specifying X_name, Y_name accordingly.
    
    Arguments
    ---------
    catalogue (str)
        Filename of hdf5 catalogue with sources to be masked.
    mask (str)
        Filename fits image mask.
    bands (List[str])
        List of photometry sub groups to be considered.
    label (str)
        Name to use for the flag dataset.
    X_name (str)
        Name of the X coordinate dataset.
    Y_name (str)
        Name of the Y coordinate dataset.
    """

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

def correct_extinction(catalogue, replace = False, suffix = '_EXT'):
    """
    Query the NED extinction calculator using mean RA and DEC location
    and apply correction to FLAGS catalogue.
    
    Arguments
    ---------
    catalogue (str)
        Path to FLAGS hdf5 catalogue. Flux units should be nJy.
    replace (bool)
        Should the original flux values be replaced?
    suffix (str)
        If original values are not replaced, create new dataset using
        this suffix
    """

    # Translate commonly used filters to the closest match on NED.
    translate = {'JWST/NIRCam.F070W': 'WFPC2 F702W','JWST/NIRCam.F090W': 'ACS F850LP','JWST/NIRCam.F115W': 'WFC3 F110W',
                    'JWST/NIRCam.F150W': 'WFC3 F160W','JWST/NIRCam.F200W': 'WFC3 F160W','JWST/NIRCam.F140M': 'WFC3 F140W',
                    'JWST/NIRCam.F162M': 'UKIRT H','JWST/NIRCam.F182M': 'UKIRT H','JWST/NIRCam.F210M': 'UKIRT K',
                    'JWST/NIRCam.F277W': 'UKIRT K','JWST/NIRCam.F356W': "UKIRT L",'JWST/NIRCam.F444W': "UKIRT L",
                    'JWST/NIRCam.F250M': 'UKIRT K','JWST/NIRCam.F300M': "UKIRT L",'JWST/NIRCam.F335M': "UKIRT L",
                    'JWST/NIRCam.F360M': "UKIRT L",'JWST/NIRCam.F410M': "UKIRT L",'JWST/NIRCam.F430M': "UKIRT L",
                    'JWST/NIRCam.F460M': "UKIRT L",'JWST/NIRCam.F480M': "UKIRT L",'HST/ACS_WFC.F435W': 'ACS F435W',
                    'HST/ACS_WFC.F475W': 'ACS F475W','HST/ACS_WFC.F555W': 'ACS F555W','HST/ACS_WFC.F606W': 'ACS	F606W',
                    'HST/ACS_WFC.F625W': 'ACS F625W','HST/ACS_WFC.F775W': 'ACS F775W','HST/ACS_WFC.F814W': 'ACS F814W',
                    'HST/WFC3_IR.F098M': 'LSST y','HST/WFC3_IR.F105W': 'WFC3 F105W','HST/WFC3_IR.F110W': 'WFC3 F110W',
                    'HST/WFC3_IR.F125W': 'WFC3 F125W','HST/WFC3_IR.F140W': 'WFC3 F140W','HST/WFC3_IR.F160W': 'WFC3 F160W'}

    # Read the catalogue.
    with h5py.File(catalogue, 'r+') as f:

        # Get list of instruments.
        instruments = list(f['photometry'].keys())
        # Remove the detection image.
        if 'detection' in instruments:
            instruments.remove('detection')

        # Get list of filters.
        filters = []
        for instrument in instruments:
            filters_ = list(f[f'photometry/{instrument}'].keys())
            for i in filters_:
                filters_[filters_.index(i)] = f'{instrument}/{i}'
            filters += filters_
        
        # For each filter.
        for filter in filters:

            # Get the mean RA and DEC.
            ra = str(np.mean(f[f'photometry/{filter}/ALPHA_SKY'][:]))
            dec = str(np.mean(f[f'photometry/{filter}/DELTA_SKY'][:]))

            # Determine the closest matching filter.
            corr_filt = translate[filter]

            # Get the extinction in magnitudes.
            Alam = request_extinctions(ra, dec, filters=corr_filt)

            # Apply the correction to each flux value.
            keys = f[f'photometry/{filter}'].keys()
            for key in keys:

                if ('FLUX' in key) & ('ERR' not in key):

                    # Get the flux in nJy.
                    flux = f[f'photometry/{filter}/{key}'][:]

                    # Convert to magnitude and apply correction.
                    mag = (-2.5 * np.log10(flux * 1e-9)) + 8.90
                    mag -= Alam

                    # Convert back to flux in nJy.
                    flux_corr = (10**((mag-8.90)/-2.5)) * 1e9

                    # Keep original values if not detected.
                    flux_corr_ = np.where(flux <= 0, flux, flux_corr)

                    # Add to the catalogue.
                    if replace == True:
                        f[f'photometry/{filter}/{key}'] = flux_corr_
                    else:
                        f[f'photometry/{filter}/{key}{suffix}'] = flux_corr_

                    # If error is provided, correct by maintaining
                    # signal to noise ratio.
                    err_name = f'FLUXERR{key.split("FLUX")[1]}'
                    if err_name in keys:

                        # The error in nJy.
                        err = f[f'photometry/{filter}/{err_name}'][:]

                        # Calculate original signal to noise.
                        s_n = flux/err

                        # Scale error to maintain this.
                        err_corr = np.where(flux <= 0, err, flux_corr/s_n)
                        print(err_corr)
                        print(flux_corr_/err_corr)

                        # Add to the catalogue.
                        if replace == True:
                            f[f'photometry/{filter}/{err_name}'] = err_corr
                        else:
                            f[f'photometry/{filter}/{err_name}{suffix}'] = err_corr
                

    return