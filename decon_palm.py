import numpy as np
from simple_tif import tif_to_array, array_to_tif
from scipy.ndimage import gaussian_filter

palm_sigma = 0.7
widefield_sigma = 3.5
num_iterations = 300
palm_intensity_scaling = 0.0001
widefield_intensity_scaling = .1

def density_to_palm_and_widefield_data(density, out=None):
    """
    Takes a 2D image input, returns a 3D PALM/widefield dataset output
    """
    if out == None:
        palm_and_widefield_data = np.zeros(
            (2,) + density.shape, dtype=np.float64)
    else:
        palm_and_widefield_data = out

    """
    Simulate the imaging process
    """
    palm_and_widefield_data[0, :, :] = (
        palm_intensity_scaling *
        gaussian_filter(density, sigma=palm_sigma))
    palm_and_widefield_data[1, :, :] = (
        widefield_intensity_scaling *
        gaussian_filter(density, sigma=widefield_sigma))
    return palm_and_widefield_data

def palm_and_widefield_data_to_density(palm_and_widefield_data, out=None):
    """
    The transpose of the density_to_palm_and_widefield_data operation
    we perform above.
    """
    if out == None:
        density = np.zeros(palm_and_widefield_data.shape[1:], dtype=np.float64)
    else:
        density = out
        density.fill(0)
    density += (
        0.5*palm_intensity_scaling *
        gaussian_filter(palm_and_widefield_data[0, :, :],
                        sigma=palm_sigma))
    density += (
        0.5*widefield_intensity_scaling *
        gaussian_filter(palm_and_widefield_data[1, :, :],
                        sigma=widefield_sigma))
    return density

"""
Load and truncate the object
"""
print "Loading resolution_target.tif..."
actual_object = tif_to_array('resolution_target-2.tif'
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
print "Generating sim data from resolution target..."
noisy_palm_and_widefield_data = density_to_palm_and_widefield_data(
    actual_object.astype(np.float32))
print "Done generating."
print "Saving visualization..."
array_to_tif(noisy_palm_and_widefield_data.astype(np.float32), outfile='palm_and_widefield_data.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(noisy_palm_and_widefield_data.sum(
    axis=0).reshape((1,) + noisy_palm_and_widefield_data.shape[1:]
                    ).astype(np.float32),
             outfile='noiseless_image.tif')
print "Done saving"

"""
Add noise
"""
print "Adding noise to sim data..."
np.random.seed(0) #Repeatably random, for now
noisy_palm_and_widefield_data[:] = np.random.poisson(
    lam=noisy_palm_and_widefield_data)
print "Done adding noise."
print "Saving visualization..."
array_to_tif(noisy_palm_and_widefield_data.astype(np.float32),
             outfile='noisy_palm_and_widefield_data.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(noisy_palm_and_widefield_data.sum(
    axis=0).reshape((1,) + noisy_palm_and_widefield_data.shape[1:]
                    ).astype(np.float32),
             outfile='noisy_image.tif')
print "Done saving"

"""
Time for deconvolution!!!
"""
estimate = np.ones(actual_object.shape, dtype=np.float64)
expected_data = np.zeros_like(noisy_palm_and_widefield_data)
correction_factor = np.zeros_like(estimate)
history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
history[0, :, :] = estimate
for i in range(num_iterations):
    print "Iteration", i
    """
    Construct the expected data from the estimate
    """
    print "Constructing estimated data..."
    density_to_palm_and_widefield_data(estimate, out=expected_data)
    array_to_tif(expected_data.astype(np.float32), outfile='expected_data.tif')
    "Done constructing."
    """
    Take the ratio between the measured data and the expected data.
    Store this ratio in 'expected_data'
    """
    expected_data += 1e-6 #Don't want to divide by 0!
    np.divide(noisy_palm_and_widefield_data, expected_data, out=expected_data)
    """
    Apply the transpose of the expected data operation to the correction factor
    """
    palm_and_widefield_data_to_density(expected_data, out=correction_factor)
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
If our fusion method is any good, the results better improve on decon
of the PALM alone or the widefield alone.
"""
estimate = np.ones((2,) + actual_object.shape, dtype=np.float64)
expected_data = np.zeros_like(estimate)
correction_factor = np.zeros_like(estimate)
palm_history = np.zeros(
    ((1+num_iterations,) + estimate.shape[1:]), dtype=np.float64)
widefield_history = np.zeros(
    ((1+num_iterations,) + estimate.shape[1:]), dtype=np.float64)
palm_history[0, :, :] = estimate[0, :, :]
widefield_history[0, :, :] = estimate[1, :, :]
for i in range(num_iterations):
    print "Iteration", i
    """
    Construct the expected data from the estimate
    """
    print "Constructing estimated data..."
    gaussian_filter(estimate[0, :, :],
                    sigma=palm_sigma,
                    output=expected_data[0, :, :])
    gaussian_filter(estimate[1, :, :],
                    sigma=widefield_sigma,
                    output=expected_data[1, :, :])
    print expected_data.min(), expected_data.max()
    array_to_tif(expected_data.astype(np.float32), outfile='expected_data.tif')
    print "Done constructing."
    """
    Take the ratio between the measured data and the expected data.
    Store this ratio in 'expected_data'
    """
    expected_data += 1e-6 #Don't want to divide by 0!
    np.divide(noisy_palm_and_widefield_data, expected_data, out=expected_data)
    print expected_data.min(), expected_data.max()
    """
    Apply the transpose of the expected data operation to the correction factor
    """
    gaussian_filter(expected_data[0, :, :],
                    sigma=palm_sigma,
                    output=correction_factor[0, :, :])
    gaussian_filter(expected_data[1, :, :],
                    sigma=widefield_sigma,
                    output=correction_factor[1, :, :])
    """
    Multiply the old estimate by the correction factor to get the new estimate
    """
    np.multiply(estimate, correction_factor, out=estimate)
    """
    Update the history
    """
    print "Saving..."
    palm_history[i+1, :, :] = estimate[0, :, :]
    array_to_tif(palm_history.astype(np.float32),
                 outfile='history_palm.tif')
    widefield_history[i+1, :, :] = estimate[1, :, :]
    array_to_tif(widefield_history.astype(np.float32),
                 outfile='history_widefield.tif')
    print "Done saving."
##    raw_input('Hit enter to continue...')
print "Done deconvolving"


