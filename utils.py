import os
import h5py
from astropy.table import Table
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import scipy.ndimage as nd
import scipy.optimize as opt
import pyregion


def multiband_catalogue(catalogues, labels, new_catalogue = 'combined_catalogue.hdf5'):

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

def multifield_catalogue(catalogues, new_catalogue, numbering = None, replace = False):
    """Combine multiple hdf5 multiband catalogues produced by pyex into a multifield catalogue.

    Parameters
    ----------
    catalogues (list):
        A list of paths to the catalogues to be combined, in pointing number order.
    new_catalogue (str):
        Path to the new catalogue.
    numbering (list):
        Numbering to use to name pointings. If None, use the position of the catalogue in catalogues.
    replace (bool):
        Whether to remove the individual field catalogues.
    """

    # Create the new catalogue.
    with h5py.File(new_catalogue, 'a') as new_cat: 

        # Iterate over the multiband catalogues.
        for catalogue in catalogues:
            with h5py.File(catalogue, 'r+') as cat:

                cat_group = cat['photometry']

                # Create a dataset indicating the source pointing.
                if numbering == None:
                    cat_group['FIELD'] = [catalogues.index(catalogue)+1]*len(cat[f'photometry/{list(cat["photometry"].keys())[0]}'])
                else:
                    cat_group['FIELD'] = [numbering[catalogues.index(catalogue)]]*len(cat[f'photometry/{list(cat["photometry"].keys())[0]}'])
                
                # Create the photometry group in the new catalogue.
                if 'photometry' in new_cat:
                    new_cat_group = new_cat['photometry']
                else:
                    new_cat_group = new_cat.create_group('photometry')

                # Iterate over datasets within the group
                for dataset_name, dataset in cat_group.items():
                    cat_data = dataset[()]

                    # If the dataset already exists in the destination group, concatenate it
                    if dataset_name in new_cat_group:
                        new_cat_dataset = new_cat_group[dataset_name]
                        new_cat_data = new_cat_dataset[()]  # Read the data from the destination dataset
                        combined_data = np.concatenate((new_cat_data, cat_data))
                        
                        # Delete the existing dataset and create a new one with the combined data
                        del new_cat_group[dataset_name]
                        new_cat_group.create_dataset(dataset_name, data=combined_data)

                    # If the dataset doesn't exist in the destination group, create it
                    else:
                        new_cat_group.create_dataset(dataset_name, data=cat_data)

                # Remove the field group from the orginal catalogue.
                del cat_group['FIELD']

                # Remove the orginal catalogue if required.
                if replace == True:
                    os.remove(catalogue)

    return

def weight_to_error(weight_image, error_filename = None):
    """Convert a weight image to an error map.

    Parameters
    ----------
    weight_image (str):
        Path to the weight image to convert.
    error_filename (str):
        Name of the error image to output. If None, append "to_error" to weight filename.
    """

    # Load the weight image and header.
    wht, hdr = fits.open(weight_image, header = True)

    # Convert the weight image to error map.
    err = np.where(wht==0, 0, 1/np.sqrt(wht))

    # If no name for new file given, use weight filename as base.
    if error_filename == None:
        error_filename = f'{weight_image.remove(".fits")}_to_error.fits'

    # Add header keyword to indicate how the error map was generated.
    hdr['CONVERTED'] = ('T', 'Converted to error image from weight image.')

    # Save the error image to a new file.
    fits.writeto(err,error_filename,header=hdr,overwrite=True)

    return

def create_stack(sci_images, wht_images, stack_name = 'pyex_stacked'):
    """Create a stacked image for detection.

    Parameters
    ----------
    sci_images (list):
        A list of science image file paths to stack.
    wht_images (list):
        A list of corresponding weight image file paths.
    stack_name (str):
        Filename of the output stacked image.

    The header used in the final images will be taken from the first image in the corresponding list.
    """

    if len(sci_images) != len(wht_images):
        raise ValueError('The number of science and weight images must be equal.')

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
    fits.writeto(f'{stack_name}_sci.fits', stack_sci, header = sci_hdr, overwrite=True)
    fits.writeto(f'{stack_name}_wht.fits', stack_wht, header = wht_hdr, overwrite=True)

    return

def generate_error(science, weight, exposure, outname = None):
    """Generate an error image from a science, weight and exposure image.

    Parameters
    ----------
    science (str):
        Path to the science image.
    weights (str):
        Path to the weight image.
    exposure (str):
        Path to the exposure time map.
    outname (str/None):
        Output filename of the generated error image. If none use default.
    """

    if outname == None:
        outname = f'{science.remove_suffix(".fits")}_err.fits'

    sci = fits.getdata(science)
    exp, exp_header = fits.getdata(exposure, header = True)
    wht, wht_header = fits.getdata(weight, header = True)

    # Grow the exposure map to the original frame
    full_exp = np.zeros(sci.shape, dtype=int)
    full_exp[2::4,2::4] += exp*1
    full_exp = nd.maximum_filter(full_exp, 4)

    # Multiplicative factors that have been applied since the original count-rate images
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
    """
    Convert a PC matrix to a CD matrix.

    WCSLIB (and PyWCS) recognizes CD keywords as input
    but converts them and works internally with the PC matrix.
    to_header() returns the PC matrix even if the input was a
    CD matrix. To keep input and output consistent we check
    for has_cd and convert the PC back to CD.

    Parameters
    ----------
    hdr: `astropy.io.fits.Header`

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

    print(f'Rebinning {input_fits}...')
    if outname == None:
        outname = f'{input_fits.removesuffix(".fits")}_rebinned.fits'

    # Read the FITS file
    with fits.open(input_fits) as hdul:
        img_array = hdul[0].data
        wcs = WCS(hdul[0].header)

    # Calculate the scale factor for resizing
    scale_factor = source_scale / target_scale

    # Calculate the new dimensions for the rebinned image
    new_height = int(img_array.shape[0] * scale_factor)
    new_width = int(img_array.shape[1] * scale_factor)

    # Initialize an empty array for the rebinned image
    rebinned_image = np.zeros((new_height, new_width), dtype=np.float32)

    # Create a new WCS for the rebinned image
    wcs_rebinned = wcs.slice((np.s_[:None:int(1/scale_factor)], np.s_[:None:int(1/scale_factor)]))
    wcs_header = wcs_rebinned.to_header()
    pc2cd(wcs_header)

    # Iterate over groups of pixels and average them to get the rebinned value
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the corresponding region in the original image
            start_i = int(i / scale_factor)
            end_i = int((i + 1) / scale_factor)
            start_j = int(j / scale_factor)
            end_j = int((j + 1) / scale_factor)

            # Extract the region from the original image
            region = img_array[start_i:end_i, start_j:end_j]

            # Take the average value of the region
            if method == 'sum':
                rebinned_value = np.sum(region)
            # or sum in quadrature.
            if method == 'quad':
                rebinned_value = np.sqrt(sum([val*val for val in region.flatten()]))

            # Assign the rebinned value to the new image
            rebinned_image[i, j] = rebinned_value

    # Save the rebinned image as a new FITS file with updated WCS information
    hdu = fits.PrimaryHDU(rebinned_image)
    hdu.header.update(wcs_header)
    hdul_rebinned = fits.HDUList([hdu])
    hdu.header["REBIN"] = (source_scale, 'Image has been rebinned from this scale')
    hdul_rebinned.writeto(outname, overwrite=True)

    print(f'Rebinned and saved to {outname}.')

    return

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
        
        print(bands)

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
        for band in bands:
            f[f'photometry/{band}/STAR'] = selection
        f['photometry/detection/STAR'] = selection

        return
    
def twoD_GaussianScaledAmp(coord, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude*np.exp( - (((coord[0]-xo)**2)/(2*sigma_x**2) + ((coord[1]-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def get_PSF_FWHM(psf_path):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """

    img = fits.getdata(psf_path)

    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = [img.shape[1]/2,img.shape[0]/2,10,10,1,0]
    # subtract background and rescale image into [0,1], with floor clipping

    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y), 
                               img.ravel(), p0 = initial_guess)
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)

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

def flag_mask(catalogue, detection, regions, X_name = 'X_IMAGE', Y_name = 'Y_IMAGE'):

    # Open the regions.
    r = pyregion.open(regions)

    # Use detection image header for WCS information.
    hdul = fits.open(detection)
    mask = r.get_mask(hdul[0])
    hdul.close()

    # Open the catalogue.
    with h5py.File(catalogue, 'r+') as f:

        # Round object centres to the nearest pixel.
        xcen = np.round(f[f'photometry/detection/{X_name}'][:])
        ycen = np.round(f[f'photometry/detection/{Y_name}'][:])

        flag = []

        # If the centre of an object is within a masked region, flag it.
        for x, y in zip(xcen, ycen):
            if mask[int(y), int(x)] == True:
                flag.append(1)
            else:
                flag.append(0)

        # Add the flag to catalogue, remvoing any previous iteration.
        if 'MASKED' in f['photometry/detection'].keys():
            del f['photometry/detection/MASKED']
        f['photometry/detection/MASKED'] = flag

    return

def flag_edges(catalogue, detection, border_size = 10, X_name = 'X_IMAGE', Y_name = 'Y_IMAGE'):

    # Load the detection image.
    image_data = fits.getdata(detection)

    # Get pixels within the specified distance from the edge
    valid_mask = ~np.isnan(image_data)
    distance_transform = np.ones_like(image_data) * np.inf

    for i in range(-border_size, border_size + 1):
        for j in range(-border_size, border_size + 1):
            if i**2 + j**2 <= border_size**2:
                shifted = np.roll(np.roll(valid_mask, i, axis=0), j, axis=1)
                distance_transform = np.minimum(distance_transform, shifted)

    pixels_near_edge = np.where(distance_transform)

    selected_coordinates = tuple(zip(pixels_near_edge[0], pixels_near_edge[1]))

    # Create a mask based on the selected coordinates
    mask = np.zeros(image_data.shape, dtype=bool)
    mask[tuple(zip(*selected_coordinates))] = True

    # Open the catalogue.
    with h5py.File(catalogue, 'r+') as f:

        # Round object centres to the nearest pixel.
        xcen = np.round(f[f'photometry/detection/{X_name}'][:])
        ycen = np.round(f[f'photometry/detection/{Y_name}'][:])

        flag = []

        # If the centre of an object is within the edge region, flag it.
        for x, y in zip(xcen, ycen):
            if mask[int(y), int(x)] == True:
                flag.append(0)
            else:
                flag.append(1)

        # Add the flag to catalogue, remvoing any previous iteration.
        if 'EDGE' in f['photometry/detection'].keys():
            del f['photometry/detection/EDGE']
        f['photometry/detection/EDGE'] = flag

    return








