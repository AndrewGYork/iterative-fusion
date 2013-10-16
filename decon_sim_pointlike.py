import numpy as np
from simple_tif import tif_to_array, array_to_tif
from scipy.ndimage import gaussian_filter, interpolation
from scipy.signal import fftconvolve
import time

msim_data_radius = 16
illumination_sigma = 4.
emission_sigma = illumination_sigma
num_iterations = 200
intensity_scaling = 0.5

def density_to_msim_data(density, out=None):
    """
    Takes a 2D image input, returns a 4D MSIM dataset output
    """
    if out == None:
        msim_data = np.zeros(density.shape + (2*msim_data_radius+1,
                                              2*msim_data_radius+1),
                             dtype=np.float64)
    else:
        msim_data = out

    illumination = np.zeros(msim_data.shape[2:], dtype=np.float64)
    illumination[msim_data_radius, msim_data_radius] = 1
    gaussian_filter(illumination, sigma=illumination_sigma, output=illumination)

    for row in range(density.shape[0]):
        #print "Row", row
        """
        If we're too close to the edge, skip to the next row
        """
        if row <= msim_data_radius:
            continue
        if row + msim_data_radius >= density.shape[0]:
            break
        for col in range(density.shape[1]):
            """
            If we're too close to the edge, skip to the next column
            """
            if col <= msim_data_radius:
                continue
            if col + msim_data_radius >= density.shape[1]:
                continue
            msim_data[row, col, :, :] = density[
                row-msim_data_radius:row+msim_data_radius+1,
                col-msim_data_radius:col+msim_data_radius+1]
            msim_data[row, col, :, :] *= illumination
            gaussian_filter(
                msim_data[row, col, :, :],
                sigma=emission_sigma,
                output=msim_data[row, col, :, :])
    return msim_data

def msim_data_to_density(msim_data, out=None):
    """
    The transpose of the density_to_msim_data operation we perform above.
    """
    if out == None:
        density = np.zeros(msim_data.shape[:2], dtype=np.float64)
    else:
        density = out
        density.fill(0)

    illumination = np.zeros(msim_data.shape[2:], dtype=np.float64)
    illumination[msim_data_radius, msim_data_radius] = 1
    gaussian_filter(illumination, sigma=illumination_sigma, output=illumination)
    temp = np.zeros_like(illumination)
    
    for row in range(density.shape[0]):
        #print "Row", row
        """
        If we're too close to the edge, skip to the next row
        """
        if row <= msim_data_radius:
            continue
        if row + msim_data_radius >= density.shape[0]:
            break
        for col in range(density.shape[1]):
            """
            If we're too close to the edge, skip to the next column
            """
            if col <= msim_data_radius:
                continue
            if col + msim_data_radius >= density.shape[1]:
                continue
            gaussian_filter(
                msim_data[row, col, :, :],
                sigma=emission_sigma,
                output=temp)
            temp *= illumination
            density[
                row-msim_data_radius:row+msim_data_radius+1,
                col-msim_data_radius:col+msim_data_radius+1
                ] += temp
    density *= 1.0 / (msim_data.shape[2] * msim_data.shape[3])
    return density

def msim_data_to_2D_visualization(msim_data, outfile=None):
    enderlein_data_image = np.zeros(
        (1, msim_data.shape[0]*msim_data.shape[2],
        msim_data.shape[1]*msim_data.shape[3]),
        dtype=np.float64)
    for row in range(msim_data.shape[0]):
        #print "Row", row
        for col in range(msim_data.shape[1]):
            enderlein_data_image[
                0,
                msim_data.shape[2]*row:msim_data.shape[2]*(row+1),
                msim_data.shape[3]*col:msim_data.shape[3]*(col+1)
                ] = msim_data[row, col, :, :]
    if outfile is not None:
        array_to_tif(enderlein_data_image.astype(np.float32), outfile)
    return enderlein_data_image

def msim_data_to_3D_visualization(msim_data, outfile=None):
    enderlein_data_stack = np.zeros(
        (msim_data.shape[2]*msim_data.shape[3],
         msim_data.shape[0],
         msim_data.shape[1]),
        dtype=np.float64)
    for row in range(msim_data.shape[2]):
        #print "Row", row
        for col in range(msim_data.shape[3]):
            enderlein_data_stack[
                row*msim_data.shape[3] + col, :, :
                ] = msim_data[:, :, row, col]
    if outfile is not None:
        array_to_tif(enderlein_data_stack.astype(np.float32), outfile)
    return enderlein_data_stack

def shift_msim_data(msim_data, out=None):
    """
    We need to compare to the standard method for processing pointlike
    SIM data.
    """
    if out is None:
        shifted_msim_data = np.zeros_like(msim_data)
    else:
        shifted_msim_data = out
    for m in range(msim_data.shape[2]):
        for n in range(msim_data.shape[3]):
            interpolation.shift(
                input=msim_data[:, :, m, n],
                shift=(0.5*(m - 0.5*msim_data.shape[2]),
                       0.5*(n - 0.5*msim_data.shape[3])),
                output=shifted_msim_data[:, :, m, n],
                order=3)
    shifted_msim_data[shifted_msim_data < 0] = 0 #Interpolation can produce zeros
    return shifted_msim_data

"""
Load and truncate the object
"""
print "Loading resolution_target.tif..."
actual_object = tif_to_array('ladder_complete.tif'
                             )[0, :, :].astype(np.float64)
print "Done loading."
print "Apodizing resolution target..."
mask = np.zeros_like(actual_object)
trim_size = 20#40
blur_size = 2#10
mask[trim_size:-trim_size, trim_size:-trim_size] = 1
gaussian_filter(mask, sigma=blur_size, output=mask)
array_to_tif(
    mask.reshape((1,)+mask.shape).astype(np.float32), outfile='mask.tif')
np.multiply(actual_object, mask, out=actual_object)
print "Done apodizing."

"""
Generate noiseless data
"""
print "Generating msim data from resolution target..."
noisy_msim_data = density_to_msim_data(actual_object)
print "Done generating."
print "Saving visualization..."
msim_data_to_2D_visualization(noisy_msim_data, outfile='msim_data.tif')
msim_data_to_3D_visualization(noisy_msim_data, outfile='msim_data_stack.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(noisy_msim_data.sum(axis=(2, 3)
                                 ).reshape((1,) + noisy_msim_data.shape[:2]
                                           ).astype(np.float32),
             outfile='noiseless_image.tif')
print "Done saving"

"""
Add noise
"""
print "Adding noise to msim data..."
np.random.seed(0) #Repeatably random, for now
for row in range(noisy_msim_data.shape[0]):
    noisy_msim_data[row, :, :, :] = np.random.poisson(
        lam=intensity_scaling * noisy_msim_data[row, :, :, :])
print "Done adding noise."
print "Saving visualization..."
msim_data_to_2D_visualization(noisy_msim_data, outfile='noisy_msim_data.tif')
msim_data_to_3D_visualization(
    noisy_msim_data, outfile='noisy_msim_data_stack.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(noisy_msim_data.sum(axis=(2, 3)
                                 ).reshape((1,) + noisy_msim_data.shape[:2]
                                           ).astype(np.float32),
             outfile='noisy_image.tif')
print "Done saving"

"""
Time for deconvolution!!!
"""
estimate = np.ones(actual_object.shape, dtype=np.float64)
expected_data = np.zeros_like(noisy_msim_data)
correction_factor = np.zeros_like(estimate)
history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
history[0, :, :] = estimate
for i in range(num_iterations):
    print "Iteration", i
    """
    Construct the expected data from the estimate
    """
    print "Constructing estimated data..."
    density_to_msim_data(estimate, out=expected_data)
    msim_data_to_3D_visualization(
        expected_data, outfile='expected_data_stack.tif')    
    "Done constructing."
    """
    Take the ratio between the measured data and the expected data.
    Store this ratio in 'expected_data'
    """
    expected_data += 1e-6 #Don't want to divide by 0!
    np.divide(noisy_msim_data, expected_data, out=expected_data)
    """
    Apply the transpose of the expected data operation to the correction factor
    """
    msim_data_to_density(expected_data, out=correction_factor)
    """
    Multiply the old estimate by the correction factor to get the new estimate
    """
    np.multiply(estimate, correction_factor, out=estimate)
    """
    Update the history
    """
    print "Saving..."
    history[i+1, :, :] = estimate
    array_to_tif(history.astype(np.float32), outfile='history.tif')
    array_to_tif(estimate.reshape((1,) + estimate.shape
                                  ).astype(np.float32), outfile='estimate.tif')
    print "Done saving."
##    raw_input('Hit enter to continue...')
print "Done deconvolving"

"""
If our new deonvolution is worth anything, it has to compare favorably
to existing methods. The standard way to process MSIM data is using
Enderlein's trick, followed by standard deconvolution.

First, execute Enderlein's trick on the noisy MSIM data.
"""
aligned_msim_data = shift_msim_data(noisy_msim_data)
enderlein_image = aligned_msim_data.sum(axis=(2, 3))
array_to_tif(enderlein_image.reshape((1,) + enderlein_image.shape
                                     ).astype(np.float32),
             outfile='enderlein_image.tif')
"""
Next, make MSIM data for a pointlike object, and execute Enderlein's
trick to get an MSIM PSF
"""
pointlike_object = np.zeros_like(actual_object)
pointlike_object[pointlike_object.shape[0]//2,
                 pointlike_object.shape[1]//2] = 1
pointlike_msim_data = density_to_msim_data(pointlike_object)
aligned_pointlike_msim_data = shift_msim_data(pointlike_msim_data)
enderlein_psf = aligned_pointlike_msim_data.sum(axis=(2, 3))
array_to_tif(enderlein_psf.reshape((1,) + enderlein_psf.shape
                                     ).astype(np.float32),
             outfile='enderlein_psf.tif')

"""
Now process the Enderlein image with standard Richardson-Lucy deconvolution.
"""
measurement = enderlein_image
print measurement.min(), measurement.max(), measurement.dtype
estimate = np.ones_like(enderlein_image)
blurred_estimate = np.zeros_like(estimate)
correction_factor = np.zeros_like(estimate)

estimate_history = np.zeros((num_iterations + 1,) + enderlein_image.shape,
                               dtype=np.float64)
estimate_history[0, :, :] = estimate
print "Deconvolving Enderlein image..."
for i in range(num_iterations):
    print " Iteration", i
##    blurred_estimate = fftconvolve(estimate, enderlein_psf, mode='same')
    gaussian_filter(
        estimate,
        sigma=np.sqrt(1.0/((1.0/emission_sigma**2) +
                           (1.0/illumination_sigma**2))),
        output=blurred_estimate)
    print " Done blurring."
    print " Computing correction ratio..."
    np.divide(measurement, blurred_estimate + 1e-6, out=correction_factor)
    print " Blurring correction ratio..."
##    correction_factor = fftconvolve(
##        correction_factor, enderlein_psf, mode='same')
    gaussian_filter(
        correction_factor,
        sigma=np.sqrt(1.0/((1.0/emission_sigma**2) +
                           (1.0/illumination_sigma**2))),
        output=correction_factor)
    print " Done blurring."
    np.multiply(estimate, correction_factor, out=estimate)
    estimate_history[i+1, :, :] = estimate
    for i in range(3):
        try:
            print " Saving history..."
            array_to_tif(estimate_history.astype(np.float32),
                         'estimate_history_enderlein.tif')
            print " Saving estimate..."
            array_to_tif(estimate.reshape((1,)+estimate.shape
                                          ).astype(np.float32),
                         'estimate_enderlein.tif')
            break
        except IOError:
            print "IO Error, trying again..."
            time.sleep(1)
            continue
    else:
        raise UserWarning("Three consecutive IO errors. :(")
print "Done deconvolving"

