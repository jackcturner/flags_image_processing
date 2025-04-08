import os
import h5py

import numpy as np
from numpy.random import uniform
import random
import math

from astropy.io import fits
from astropy.table import Table

from scipy.spatial import cKDTree
import scipy.ndimage as nd

from utils import create_edge_mask, poisson_confidence_interval
from extraction import SExtractor

def find_matches(small_cat, large_cat):
    """
    Return indicies into a larger catalogue from X-Y matches to a smaller
    catalogue. Matches are not unique.
    
    Arguments
    ---------
    small_cat (numpy.ndarray)
        The X-Y coordinates of sources in the smaller catalogue.
    large_cat (numpy.ndarray)
        The X-Y coordinates of sources in the larger catalogue.
        
    Returns
    -------
    indices (List[int])
        For each object in small_cat, the index of the closest match in 
        large_cat.
    distances (List[float])
        The distance between matches in pixels."""

    # Create KD-tree for the larger catalogue.
    large_tree = cKDTree(large_cat)
    
    # Query the KD-tree with the positions from the smaller catalogue.
    distances, indices = large_tree.query(small_cat)

    # Sort indices and reorder distances accordingly.
    sorted_indices = np.argsort(indices)
    indices = indices[sorted_indices]
    distances = distances[sorted_indices]
    
    # Return the matched indices and distances.
    return indices, distances

def measure_completeness(science, weight, psf, bins, config, parameters={}, sex_path='sex', 
                         mask=None, conversion=1/21.15, dilate=0, border=50, min_sources=1500, 
                         density=5, offset=6.66, flux_limits=[0.5, 1.5], min_sn=2, pixel_scale=0.03, 
                         outdir='./'):
    """
    Measure the completeness of an image by inserting synthetic sources
    in provided magnitude bins. 

    Arguments
    ---------
    science (str)
        Path to fits image from which to measure completeness.
    wht_name (str)
        Path to the fits image to use for weighting.
    psf (str)
        Path to the fits PSF image to insert as a synthetic source.
    bins (numpy.ndarray)
        1D array defining the AB-magnitude bin edges.
    config (str)
        Path to the Source Extractor configuration file to use.
    parameters (dict)
        Key-value pairs to overwrite the configuration file.
    sex_path (str)
        Path to the source extractor executable.
    mask (str/None)
        Path to fits source mask. If None, compute with SE.
    conversion (float)
        Multiplicative factor to convert nJy to image units.
    dilate (int)
        The number of source mask dilation iterations.
    border_width (int)
        Width of edge mask to generate.
    min_sources (int)
        The minimum number of sources to generate.
    density (int)
        The number of sources per arcmin that can be inserted.
    offset (float)
        The maximum offset in pixels allowed between the inserted and
        recovered source.
    flux_limits (List[float])
        The minimum and maximum flux ratio of recovered and inserted 
        sources.
    min_sn (float)
        The minimum S/N of recovered sources.
    pixel_scale (float)
        The pixel scale in arcseconds. Only used if PIXAR_A2 not found
        in the header.
    outdir (str)
        Directory in which to save temporary files.

    Returns
    -------
    complete (List[float])
        The estimated completeness in each magnitude bin.
    error (List[numpy.ndarray])
        The 1-sigma upper and lower confidence limits.
    """
    
    sci_name = os.path.basename(science).removesuffix('fits')
    print(f'Measuring completeness in {sci_name}...')

    # Convert the bin magnitudes to nJy.
    bins = np.array([(10**((m-8.90)/-2.5))*1e9 for m in bins])
    bins_info = np.column_stack(((bins[:-1] + bins[1:]) / 2, bins[:-1], bins[1:]))

    # Initalise the SExtractor class.
    se_run = SExtractor(config, sex_path)

    # Create an edge mask to remove noisy regions.
    edges = create_edge_mask(science, n_pixels=border)

    # If no mask provided, generate a segmentation map.
    delete = False
    if mask == None:
        print('Generating source mask...')
        mask = f'{outdir}/{sci_name}_completeness_mask.fits'
        parameters['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
        parameters['CHECKIMAGE_NAME'] = mask
        cat = se_run.extract(science, weight, parameters, outdir=outdir)
        os.remove(cat)
        delete = True

    # Get a list of pixels that are on the detector and unmasked.
    unmasked = (edges == 0)
    with fits.open(weight) as wht:
            unmasked = unmasked & (wht[0].data > 0) & (~np.isnan(wht[0].data))
    with fits.open(mask) as seg:
            seg_mask = (seg[0].data != 0)
            if dilate > 0:
                seg_mask = nd.binary_dilation(seg_mask, iterations=dilate)
            unmasked = unmasked & (seg_mask == 0)

    # Find indices of unmasked pixels.
    unmasked_pixels = np.where(unmasked)
    unmasked_coordinates = list(zip(unmasked_pixels[0], unmasked_pixels[1]))

    # Get the total unmasked area in arcmin.
    hdr = fits.getheader(science)
    try:
        total_area = np.sum(unmasked)*(hdr['PIXAR_A2']/3600)
    except:
        total_area = np.sum(unmasked)*((pixel_scale**2)/3600)

    # Load an normalise the psf
    psf = fits.getdata(psf)
    psf /= np.sum(psf)

    # Number of sources that can be placed in each image.
    n_sources = math.ceil(total_area*density) 
    # The number of mosaics needed for minimum sources.
    n_img_max = math.ceil(min_sources/n_sources)    
    # The total number of sources to be placed.
    total_sources = n_sources*n_img_max

    print(f'Placing {n_sources} synthetic sources in {n_img_max} mosaics, ' 
          f'totalling {total_sources}.')

    # Before running SE fix some parameters.
    parameters['CHECKIMAGE_TYPE'] = 'NONE'
    parameters['EMPIRICAL'] = False
    parameters['TO_FLUX'] = 1/conversion

    # Store the completeness here.
    complete = []
    error = []

    # For each bin.
    for bin in bins_info:

        print(f'Working on bin with central flux {round(bin[0])} nJy...')

        # Keep track of the number of recovered sources.
        n_recovered = 0

        # For the required number of mosaics.
        n_img = 0
        while n_img < n_img_max:

            # Store source information in a table.
            source_table = Table(names = ['INDEX', 'X_IMAGE', 'Y_IMAGE', 'FLUX'])

            # Open a new mosaic.
            img = fits.getdata(science)

            # Get the random locations of the sources.
            indices = random.sample(range(len(unmasked_coordinates)), n_sources)
            locations = [unmasked_coordinates[i] for i in indices]

            # For each source.
            for i, location in enumerate(locations):

                # Scale the PSF to the desired total flux in image units.
                # Flux is selected uniformly within bin.
                flux_psf = uniform(bin[1], bin[2])
                psf_ = psf * flux_psf * conversion

                # Calculate the bounding box for the source image within 
                # the mosaic.
                x_start = location[0] - psf_.shape[0]//2  
                x_end = x_start + psf_.shape[0]
                y_start = location[1] - psf_.shape[1]//2
                y_end = y_start + psf_.shape[1]  

                # Ensure the bounding box is within the bounds of 
                # the mosaic.
                x_start = max(x_start, 0)
                x_end = min(x_end, img.shape[0])
                y_start = max(y_start, 0)
                y_end = min(y_end, img.shape[1])

                # Add the source to the mosaic.
                img[x_start:x_end, y_start:y_end] += psf_

                # add to the table with the flux in nJy.
                source_table.add_row([i, location[1], location[0], flux_psf])
                        
            # Save the image.
            img_name = f'{outdir}/{sci_name}_completeness_{n_img}_{len(locations)}.fits'
            fits.writeto(img_name, img, hdr, overwrite = True)

            # Run the SExtraction on this image.
            cat = se_run.extract(img_name, weight, parameters, output = ['FLUX_AUTO', 'FLUXERR_AUTO', 'X_IMAGE', 'Y_IMAGE'], outdir=outdir)

            with h5py.File(cat) as f:

                # Stack the measure X and Y coordinates and the true values.
                cat_xy = np.column_stack((f['photometry/X_IMAGE'][:], f['photometry/Y_IMAGE'][:]))
                syn_xy = np.column_stack((source_table['X_IMAGE'], source_table['Y_IMAGE']))

                # Match true to measured sources and return distances and
                # indicies into the measured catalogue.
                indices, distances = find_matches(syn_xy, cat_xy)

                # Apply distance criterion.
                s = distances < offset

                # Search objects passing this criterion for duplicate matches.
                unique_indices, unique_pos = np.unique(indices, return_inverse=True)
                duplicate_mask = np.zeros_like(indices, dtype=bool)
    
                for i in range(len(unique_indices)):
                    duplicate_indices = np.where(unique_pos == i)[0]
                    # If duplicates are identified
                    if len(duplicate_indices) > 1:
                        min_dist_idx = np.argmin(distances[duplicate_indices])
                        # Only keep the match with the smallest distance.
                        for j in range(len(duplicate_indices)):
                            if j != min_dist_idx:
                                duplicate_mask[duplicate_indices[j]] = True

                # Otherwise the criterion is failed.
                s[duplicate_mask] = False

                print(f'Number of matches within distance threshold: {sum(s)}')

                # Apply the distance criterion to indices
                filtered_indices = indices[s]

                # and sort them.
                sorted_order = np.argsort(filtered_indices)
                sorted_indices = filtered_indices[sorted_order]

                # Get measured fluxes and S/N of passing objects
                flux = f['photometry/FLUX_AUTO'][sorted_indices]
                err = f['photometry/FLUXERR_AUTO'][sorted_indices]
                sn = flux / err

                # and the true flux.
                true_flux = source_table['FLUX'][s]
                true_flux = true_flux[sorted_order]

                # Apply flux criteria.
                s_ = ((flux / true_flux < flux_limits[1]) & (flux / true_flux > flux_limits[0]) & 
                      (sn > min_sn))
                print(f"Number of sources matching flux criteria: {sum(s_)}")

            # Record the number of recovered objects.
            n_recovered += sum(s_)

            # Remove files for this image.
            os.remove(cat)
            os.remove(img_name)

            n_img += 1

        complete.append(n_recovered/total_sources)
        error.append(poisson_confidence_interval([n_recovered])/total_sources)

    # If source mask was generated, remove it.
    if delete == True:
        os.remove(mask)

    return complete, error