import os
import h5py

import numpy as np
from numpy.random import uniform
import random
import math

from astropy.io import fits
from astropy.table import Table

from scipy.spatial import cKDTree

from utils import create_edge_mask, poisson_confidence_interval
from extraction.SExtractor_pipeline import SExtractor

import webbpsf
os.environ['WEBBPSF_PATH'] = "/Users/jt458/jwst_data/webbpsf-data"

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

    # Create KD-tree for the larger catalogue
    large_tree = cKDTree(large_cat)
    
    # Query the KD-tree with the positions from the smaller catalogue
    distances, indices = large_tree.query(small_cat)

    # Sort indices and reorder distances accordingly
    sorted_indices = np.argsort(indices)
    indices = indices[sorted_indices]
    distances = distances[sorted_indices]
    
    # Return the matched indices and distances
    return indices, distances

def measure_completeness(sci_name, wht_name, config_name, psf_name=None, filter=None, seg_name=None,
                         sex_path='sex', min_sources=1500, density=5, max_distance=6.66,
                         max_flux=1.5, min_flux=0.5, min_sn=2, border_width=50,
                         conversion=1/21.15, bins=np.arange(start = 19.75, stop = 32.25, step=0.5)):
    """
    Measure the completeness of an image by inserting synthetic sources
    in a range of magnitude bins.

    Arguments
    ---------
    sci_name (str)
        Path to the fits science image for which to measure completeness.
    wht_name (str)
        Path the corresponding fits weight map.
    config_name (str)
        Path to the Sextractor configuration file to use.
    psf_name (str/ None)
        If str, path to fits PSF image to use as synthetic source.
        If None, use WebbPSF to genrate the source.
    filter (str, None)
        If psf_name = None, the PSF to generate with WebbPSF.
    seg_name (str, None)
        If str, path to Sextractor segmentation map to use as source
        mask. If None, generate using provided parameters.
    sex_path (str)
        Path to Sextractor executable.
    min_sources (int)
        The minimum number of sources to generate.
    density (int)
        The number of sources per arcmin that can be inserted.
    max_distance (float)
        The maximum acceptable distance in pixels for a match.
    max_flux (float)
        The maximum accepted flux ratio for a match.
    min_flux (float)
        The minimum accepted flux ratio for a match.
    min_sn (float)
        The minimum accepted S/N for a match.
    border_width (int)
        Width of edge mask to generate.
    conversion (float)
        Multiplicative factor for converting nJy to image units.
    bins (numpy.ndarray)
        1D array defining the magnitude bin edges.

    Returns
    -------
    complete (List[float])
        The estimated completeness in each magnitude bin.
    error (List[numpy.ndarray])
        The 1-sigma upper and lower confidence limits.
    """
    
    print(f'Measuring completeness in {os.path.basename(sci_name)}...')

    # Convert the bin centre magnitudes to nJy.
    bins = np.array([(10**((m-8.90)/-2.5))*1e9 for m in bins])
    # Store the centres and edges of each bin.
    bins_info = np.column_stack(((bins[:-1] + bins[1:]) / 2, bins[:-1], bins[1:]))

    # Initalise the SExtractor class.
    se_run = SExtractor(config_name, sex_path)

    # Create an edge mask to remove noisy regions.
    edges = create_edge_mask(sci_name, n_pixels=border_width)
    os.remove('combined_edge_mask.fits')

    # If no mask provided, generate a segmentation map.
    delete = False
    if seg_name == None:
        print('Generating source mask...')
        cat = se_run.SExtract(sci_name, wht_name, parameters = {'CHECKIMAGE_TYPE':'SEGMENTATION',
                                                            'CHECKIMAGE_NAME':'./completeness_mask.fits'})
        seg_name = './completeness_mask.fits'
        os.remove(cat)
        delete = True

    # Get a list of pixels that are on the detector and unmasked.
    unmasked = (edges == 0)
    with fits.open(wht_name) as wht:
            unmasked = unmasked & (wht[0].data != 0)
    with fits.open(seg_name) as seg:
            unmasked = unmasked & (seg[1].data == 0)

    # Find indices of unmasked pixels.
    unmasked_pixels = np.where(unmasked)
    unmasked_coordinates = list(zip(unmasked_pixels[0], unmasked_pixels[1]))

    # Get the total unmasked area.
    hdr = fits.getheader(sci_name)
    total_area = np.sum(unmasked)*(hdr['PIXAR_A2']/3600)

    # If no PSF provided, generate using WebbPSF.
    if psf_name == None:
        print('Using WebbPSF generated PSF.')
        nc = webbpsf.NIRCam()
        nc.pixelscale = np.sqrt(hdr['PIXAR_A2'])
        nc.filter = filter
        psf = nc.calc_psf()
        psf = psf[3].data
        del nc
    else:
        psf = fits.getdata(psf_name)

    # Ensure the PSF is normalised.
    psf /= np.sum(psf)

    # Number of sources that can be placed in each image.
    n_sources = math.ceil(total_area*density) 
    # The number of mosaics needed for minimum sources.
    n_img_max = math.ceil(min_sources/n_sources)    
    # The total number of sources to be placed.
    total_sources = n_sources*n_img_max

    print(f"Placing {n_sources} synthetic sources in {n_img_max} mosaics, totalling {total_sources}.")

    # Store the completeness here.
    complete = []
    error = []

    # For each bin.
    for bin in bins_info:

        print(f"Working on bin with central flux {round(bin[0])} nJy...")

        # Keep track of the number of recovered sources.
        n_recovered = 0

        # For the required number of mosaics.
        n_img = 0
        while n_img < n_img_max:

            # Store source information in a table.
            source_table = Table(names = ['INDEX', 'X_IMAGE', 'Y_IMAGE', 'FLUX'])

            # Open a new mosaic.
            img = fits.getdata(sci_name)

            # Get the random locations of the sources.

            # Select unique random entries using random.sample
            indices = random.sample(range(len(unmasked_coordinates)), n_sources)

            # Get the corresponding locations
            locations = [unmasked_coordinates[i] for i in indices]

            #locations = unmasked_coordinates[np.random.choice(len(unmasked_coordinates), n_sources, replace=False)]

            # For each source.
            for i, location in enumerate(locations):

                # Scale the PSF to the desired total flux in image units.
                # Flux is selected uniformly within bin.
                psf_ = (psf * (uniform(bin[1], bin[2])/np.sum(psf)))*conversion
                flux_psf = np.sum(psf_)
                 
                # Calculate the bounding box for the source image within the mosaic.
                x_start = location[0] - psf_.shape[0]//2  
                x_end = x_start + psf_.shape[0]
                y_start = location[1] - psf_.shape[1]//2
                y_end = y_start + psf_.shape[1]  

                # Ensure the bounding box is within the bounds of the mosaic.
                x_start = max(x_start, 0)
                x_end = min(x_end, img.shape[0])
                y_start = max(y_start, 0)
                y_end = min(y_end, img.shape[1])

                # Add the source to the mosaic.
                img[x_start:x_end, y_start:y_end] += psf_

                # add to the table
                source_table.add_row([i, location[1], location[0], flux_psf])
                        
            # Save the image.
            fits.writeto(f'completeness_{n_img}_{len(locations)}.fits', img, hdr, overwrite = True)

            # Run the SExtraction on this image.
            cat = se_run.SExtract(f'completeness_{n_img}_{len(locations)}.fits', wht_name, parameters = {'TO_FLUX':1/conversion}, measurement = ['FLUX_AUTO', 'FLUXERR_AUTO', 'X_IMAGE', 'Y_IMAGE'])

            with h5py.File(cat) as f:

                # Stack the measure X and Y coordinates and the true values.
                cat_xy = np.column_stack((f['photometry/X_IMAGE'][:], f['photometry/Y_IMAGE'][:]))
                syn_xy = np.column_stack((source_table['X_IMAGE'], source_table['Y_IMAGE']))

                # Match true to measured sources and return distances and
                # indicies into the measured catalogue.
                indices, distances = find_matches(syn_xy, cat_xy)

                # Apply distance criterion.
                s = distances < max_distance

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

                print(f"Number of matches within distance threshold: {sum(s)}")

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
                s_ = (true_flux/conversion / flux < max_flux) & \
                    (true_flux/conversion / flux > min_flux) & (sn > min_sn)
                print(f"Number of sources matching flux criteria: {sum(s_)}")

            # Record the number of recovered objects.
            n_recovered += sum(s_)

            # Remove files for this image.
            os.remove(cat)
            os.remove(f'completeness_{n_img}_{len(locations)}.fits')

            n_img += 1

        complete.append(n_recovered/total_sources)
        error.append(poisson_confidence_interval([n_recovered])/total_sources)

    # If segmentation map was generated, remove it.
    if delete == True:
        os.remove(seg_name)

    return complete, error