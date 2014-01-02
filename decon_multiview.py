import numpy as np
from simple_tif import tif_to_array, array_to_tif
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

sigmas = ((6, 2), #6, 2
          (2, 6))
num_iterations = 100
intensity_scaling = 0.1 #0.003

##def construct_psf(density, sigma):
##    y = np.arange(density.shape[0]).reshape(density.shape[0], 1)
##    x = np.arange(density.shape[1]).reshape(1, density.shape[1])
##    k = 2 * np.pi / illumination_period
##    for t, theta in enumerate(np.arange(0, 2*np.pi, 2*np.pi/num_rotations)):
##        k_x = k * np.cos(theta)
##        k_y = k * np.sin(theta)
##        for p, phase in enumerate(np.arange(0, 2*np.pi, 2*np.pi/num_phases)):
##            illumination[t, p, :, :] = 1 + np.sin(k_x * x +k_y * y +phase)
##    return illumination
##
##    return psf

def density_to_multiview_data(density, out=None):
    """
    Takes a 2D image input, returns a stack of multiview data
    """
    if out == None:
        multiview_data = np.zeros(
            (len(sigmas),) + density.shape, dtype=np.float64)
    else:
        multiview_data = out

    """
    Simulate the imaging process by applying multiple blurs
    """
    for s, sig in enumerate(sigmas):
        gaussian_filter(density, sigma=sig, output=multiview_data[s, :, :])
    return multiview_data

def multiview_data_to_density(multiview_data, out=None):
    """
    The transpose of the density_to_multiview_data operation we perform above.
    """
    if out == None:
        density = np.zeros(multiview_data.shape[2:], dtype=np.float64)
    else:
        density = out
        density.fill(0)

    for s, sig in enumerate(sigmas):
        density += gaussian_filter(multiview_data[s, :, :], sigma=sig)
    density *= 1.0 / len(sigmas)
    return density

def multiview_data_to_visualization(multiview_data, outfile=None):
    if outfile is not None:
        array_to_tif(multiview_data.astype(np.float32), outfile)
    return multiview_data

"""
Load and truncate the object
"""
print "Loading resolution_target.tif..."
actual_object = tif_to_array('resolution_target.tif'
                             )[0, :, :].astype(np.float64)
print "Done loading."
print "Apodizing resolution target..."
mask = np.zeros_like(actual_object)
trim_size = 40
blur_size = 10
mask[trim_size:-trim_size, trim_size:-trim_size] = 1
gaussian_filter(mask, sigma=blur_size, output=mask)
array_to_tif(
    mask.reshape((1,)+mask.shape).astype(np.float32), outfile='mask.tif')
np.multiply(actual_object, mask, out=actual_object)
print "Done apodizing."

"""
Generate noiseless data
"""
print "Generating multiview data from resolution target..."
noisy_multiview_data = density_to_multiview_data(actual_object)
print "Done generating."
print "Saving visualization..."
multiview_data_to_visualization(
    noisy_multiview_data, outfile='multiview_data.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(
    noisy_multiview_data.sum(axis=0).reshape(
        (1,) + noisy_multiview_data.shape[1:]).astype(np.float32),
    outfile='noiseless_image.tif')
print "Done saving"

"""
Add noise
"""
print "Adding noise to sim data..."
np.random.seed(0) #Repeatably random, for now
for s in range(len(sigmas)):
    noisy_multiview_data[s, :, :] = np.random.poisson(
        lam=intensity_scaling * noisy_multiview_data[s, :, :])
print "Done adding noise."
print "Saving visualization..."
multiview_data_to_visualization(
    noisy_multiview_data, outfile='noisy_multiview_data.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(
    noisy_multiview_data.sum(axis=0).reshape(
        (1,) + noisy_multiview_data.shape[1:]).astype(np.float32),
    outfile='noisy_image.tif')
print "Done saving"

"""
Time for deconvolution!!!
"""
estimate = np.ones(actual_object.shape, dtype=np.float64)
expected_data = np.zeros_like(noisy_multiview_data)
correction_factor = np.zeros_like(estimate)
history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
history[0, :, :] = estimate
for i in range(num_iterations):
    print "Iteration", i
    """
    Construct the expected data from the estimate
    """
    print "Constructing estimated data..."
    density_to_multiview_data(estimate, out=expected_data)
    multiview_data_to_visualization(expected_data, outfile='expected_data.tif')
    "Done constructing."
    """
    Take the ratio between the measured data and the expected data.
    Store this ratio in 'expected_data'
    """
    expected_data += 1e-6 #Don't want to divide by 0!
    np.divide(noisy_multiview_data, expected_data, out=expected_data)
    """
    Apply the transpose of the expected data operation to the correction factor
    """
    multiview_data_to_density(expected_data, out=correction_factor)
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
If this deonvolution is worth anything, it has to compare favorably to
existing methods. For example, if we just add the two views, and
deconvolve with an appropriate PSF, do we get the same image?

First, make multiview data for a pointlike object, and sum the images.
"""
pointlike_object = np.zeros_like(actual_object)
pointlike_object[pointlike_object.shape[0]//2,
                 pointlike_object.shape[1]//2] = 1
pointlike_multiview_data = density_to_multiview_data(pointlike_object)
summed_multiview_psf = pointlike_multiview_data.mean(axis=0)
array_to_tif(summed_multiview_psf.reshape((1,) + summed_multiview_psf.shape
                                     ).astype(np.float32),
             outfile='summed_multiview_psf.tif')

"""
Now process the summed multiview image with standard Richardson-Lucy
deconvolution.
"""
measurement = noisy_multiview_data.mean(axis=0)
print measurement.min(), measurement.max(), measurement.dtype
estimate = np.ones_like(actual_object)
correction_factor = np.zeros_like(estimate)

estimate_history = np.zeros((num_iterations + 1,) + estimate.shape,
                               dtype=np.float64)
estimate_history[0, :, :] = estimate
print "Deconvolving summed multiview image..."
for i in range(num_iterations):
    print " Iteration", i
    blurred_estimate = fftconvolve(estimate, summed_multiview_psf, mode='same')
    print " Done blurring."
    print " Computing correction ratio..."
    np.divide(measurement, blurred_estimate + 1e-6, out=correction_factor)
    print " Blurring correction ratio..."
    correction_factor = fftconvolve(
        correction_factor, summed_multiview_psf, mode='same')
    print " Done blurring."
    np.multiply(estimate, correction_factor, out=estimate)
    estimate_history[i+1, :, :] = estimate
    print " Saving history..."
    array_to_tif(estimate_history.astype(np.float32),
                 'estimate_history_summed_multiview.tif')
    print " Saving estimate..."
    array_to_tif(estimate.reshape((1,)+estimate.shape).astype(np.float32),
                 'estimate_summed_multiview.tif')
print "Done deconvolving"
