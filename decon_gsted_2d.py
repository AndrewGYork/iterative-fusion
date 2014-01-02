import numpy as np
from simple_tif import tif_to_array, array_to_tif
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

print "Loading resolution_target.tif..."
actual_object = tif_to_array('resolution_target.tif'
                             )[0, :, :].astype(np.float64)
pointlike_object = np.zeros_like(actual_object)
pointlike_object[pointlike_object.shape[0]//2,
                 pointlike_object.shape[1]//2] = 1
print "Done loading."

num_timepoints = 10
sigma_range = np.linspace(7.5, 2, num_timepoints)
signal_range = np.linspace(1, .1, num_timepoints)
blurred_object = np.zeros(
    (num_timepoints,) + actual_object.shape, dtype=np.float64)
blurred_pointlike_object = np.zeros_like(blurred_object)
print "Blurring..."
for i in range(num_timepoints):
    print " Blurring timepoint", i
    gaussian_filter(signal_range[i] * actual_object,
                    sigma=sigma_range[i],
                    output=blurred_object[i, :, :])
    gaussian_filter(signal_range[i] * pointlike_object,
                    sigma=sigma_range[i],
                    output=blurred_pointlike_object[i, :, :])
print "Done blurring."
print "Saving blurred_object.tif"
array_to_tif(blurred_object.astype(np.float32), 'blurred_object.tif')
print "Done saving."
print "Saving blurred_pointlike_object.tif"
array_to_tif(blurred_pointlike_object.astype(np.float32),
             'blurred_pointlike_object.tif')
print "Done saving."

noisy_object = np.zeros(blurred_object.shape, dtype=np.float64)
print "Adding noise..."
print " Seeding random number generator with value: 0"
np.random.seed(0) #For now, we want repeatably random
for p in range(num_timepoints):
    print " Adding Poisson noise to slice", p
    noisy_object[p, :, :] = np.random.poisson(lam=0.5*blurred_object[p, :, :])
print "Saving noisy_object.tif"
array_to_tif(noisy_object.astype(np.float32), 'noisy_object.tif')
print "Done saving."

print "Saving cumulative_sum.tif"
cumulative_sum = np.cumsum(noisy_object[::-1, :, :], axis=0)
print cumulative_sum.dtype
array_to_tif(cumulative_sum.astype(np.float32), 'cumulative_sum.tif')
print "Done saving"
print "Saving cumulative_sum_psf.tif"
cumulative_sum_psf = np.cumsum(blurred_pointlike_object[::-1, :, :], axis=0)
array_to_tif(cumulative_sum_psf.astype(np.float32), 'cumulative_sum_psf.tif')
print "Done saving"

measurement = noisy_object
estimate = np.ones(actual_object.shape, dtype=np.float64)
##np.mean(noisy_object, axis=0, out=estimate)

blurred_estimate = np.zeros(blurred_object.shape, dtype=np.float64)
correction_factor = np.zeros_like(blurred_estimate)

num_iterations = 30
estimate_history = np.zeros((num_iterations + 1,) + actual_object.shape,
                               dtype=np.float64)
estimate_history[0, :, :] = estimate

print "Deconvolving noisy object..."
for i in range(num_iterations):
    print " Iteration", i
    for t in range(num_timepoints):
##        print " Blurring timepoint", t
        gaussian_filter(signal_range[t] * estimate,
                        sigma=sigma_range[t],
                        output=blurred_estimate[t, :, :])
    print " Done blurring."
    print " Computing correction ratio..."
    np.divide(measurement, blurred_estimate, out=correction_factor)
    print " Blurring correction ratio..."
    for t in range(num_timepoints):
##        print "  Blurring timepoint", t
        gaussian_filter(signal_range[t] * correction_factor[t, :, :],
                        sigma=sigma_range[t],
                        output=correction_factor[t, :, :])
    print " Done blurring."
    np.multiply(estimate, correction_factor.mean(axis=0), out=estimate)
    estimate_history[i+1, :, :] = estimate
    print " Saving history..."
    array_to_tif(estimate_history.astype(np.float32), 'estimate_history.tif')
print "Done deconvolving"

"""
For reference, we want to  compare our new type of deconvolution to
standard deconvolution of the same data. Since we don't know which
slice of the cumulative sum will loook the best, we'll deconvolve them
all. If our new type of decon is any good, it'll look better than any
of these frames.
"""
measurement = cumulative_sum
estimate = np.ones_like(cumulative_sum)
blurred_estimate = np.zeros_like(estimate)
correction_factor = np.zeros_like(blurred_estimate)

history_slice = 2
estimate_history = np.zeros((num_iterations + 1,) + actual_object.shape,
                               dtype=np.float64)
estimate_history[0, :, :] = estimate[history_slice, :, :]
print "Deconvolving cumulative sum..."
for i in range(num_iterations):
    print " Iteration", i
    for t in range(num_timepoints):
##        print " Blurring timepoint", t
        blurred_estimate[t, :, :] = fftconvolve(
            estimate[t, :, :],
            cumulative_sum_psf[t, :, :],
            mode='same')
    print " Done blurring."
    print " Computing correction ratio..."
    np.divide(measurement, blurred_estimate, out=correction_factor)
    print " Blurring correction ratio..."
    for t in range(num_timepoints):
##        print "  Blurring timepoint", t
        correction_factor[t, :, :] = fftconvolve(
            correction_factor[t, :, :],
            cumulative_sum_psf[t, :, :],
            mode='same')
    print " Done blurring."
    np.multiply(estimate, correction_factor, out=estimate)
    estimate_history[i+1, :, :] = estimate[history_slice, :, :]
    print " Saving history..."
    array_to_tif(estimate_history.astype(np.float32),
                 'estimate_history_cumsum.tif')
    print " Saving estimate..."
    array_to_tif(estimate.astype(np.float32),
                 'estimate_cumsum.tif')
print "Done deconvolving"

