import numpy as np
from simple_tif import tif_to_array, array_to_tif
from scipy.ndimage import gaussian_filter

num_rotations = 3
num_phases = 5
illumination_period = 15.
emission_fwhm = 0.5 * illumination_period
num_iterations = 200
intensity_scaling = 0.03

emission_sigma = emission_fwhm * 1.0 / (2*np.sqrt(2*np.log(2)))

def generate_illumination(density):
    y = np.arange(density.shape[0]).reshape(density.shape[0], 1)
    x = np.arange(density.shape[1]).reshape(1, density.shape[1])
    illumination = np.zeros((num_rotations, num_phases) + density.shape,
                            dtype=np.float64)
    k = 2 * np.pi / illumination_period
    for t, theta in enumerate(np.arange(0, 2*np.pi, 2*np.pi/num_rotations)):
        k_x = k * np.cos(theta)
        k_y = k * np.sin(theta)
        for p, phase in enumerate(np.arange(0, 2*np.pi, 2*np.pi/num_phases)):
            illumination[t, p, :, :] = 1 + np.sin(k_x * x +k_y * y +phase)
    return illumination

def density_to_sim_data(density, out=None):
    """
    Takes a 2D image input, returns a 4D SIM dataset output
    """
    if out == None:
        sim_data = np.zeros(
            (num_rotations, num_phases) + density.shape, dtype=np.float64)
    else:
        sim_data = out

    """
    Construct the illumination pattern
    """
    illumination = generate_illumination(density)
    sim_data_to_visualization(illumination, 'illumination.tif')

    """
    Simulate the imaging process: multiply by the illumination, and blur
    """
    for t in range(num_rotations):
        for p in range(num_phases):
            sim_data[t, p, :, :] = illumination[t, p, :, :] * density
            gaussian_filter(sim_data[t, p, :, :],
                            sigma=emission_sigma,
                            output=sim_data[t, p, :, :])
    return sim_data

def sim_data_to_density(sim_data, out=None):
    """
    The transpose of the density_to_sim_data operation we perform above.
    """
    if out == None:
        density = np.zeros(sim_data.shape[2:], dtype=np.float64)
    else:
        density = out
        density.fill(0)

    illumination = generate_illumination(density)
    for t in range(num_rotations):
        for p in range(num_phases):
            density += (illumination[t, p, :, :] *
                        gaussian_filter(sim_data[t, p, :, :],
                                        sigma=emission_sigma))
    return density

def sim_data_to_visualization(sim_data, outfile=None):
    image_stack = np.zeros(
        (num_rotations*num_phases,
         sim_data.shape[2],
         sim_data.shape[3]),
        dtype=np.float64)
    s = -1
    for t in range(num_rotations):
        for p in range(num_phases):
            s += 1
            image_stack[s, :, :] = sim_data[t, p, :, :]
    if outfile is not None:
        array_to_tif(image_stack.astype(np.float32), outfile)
    return image_stack

"""
Load and truncate the object
"""
print "Loading resolution_target.tif..."
actual_object = tif_to_array('resolution_target_qbf.tif'
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
noisy_sim_data = density_to_sim_data(actual_object)
print "Done generating."
print "Saving visualization..."
sim_data_to_visualization(noisy_sim_data, outfile='sim_data.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(noisy_sim_data.sum(axis=(0, 1)
                                 ).reshape((1,) + noisy_sim_data.shape[2:]
                                           ).astype(np.float32),
             outfile='noiseless_image.tif')
print "Done saving"

"""
Add noise
"""
print "Adding noise to sim data..."
np.random.seed(0) #Repeatably random, for now
for t in range(num_rotations):
    for p in range(num_phases):
        noisy_sim_data[t, p, :, :] = np.random.poisson(
            lam=intensity_scaling * noisy_sim_data[t, p, :, :])
print "Done adding noise."
print "Saving visualization..."
sim_data_to_visualization(noisy_sim_data, outfile='noisy_sim_data.tif')
print "Done saving"
print "Saving unprocessed image..."
array_to_tif(noisy_sim_data.sum(axis=(0, 1)
                                 ).reshape((1,) + noisy_sim_data.shape[2:]
                                           ).astype(np.float32),
             outfile='noisy_image.tif')
print "Done saving"

"""
Time for deconvolution!!!
"""
estimate = np.ones(actual_object.shape, dtype=np.float64)
expected_data = np.zeros_like(noisy_sim_data)
correction_factor = np.zeros_like(estimate)
history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
history[0, :, :] = estimate
for i in range(num_iterations):
    print "Iteration", i
    """
    Construct the expected data from the estimate
    """
    print "Constructing estimated data..."
    density_to_sim_data(estimate, out=expected_data)
    sim_data_to_visualization(expected_data, outfile='expected_data.tif')
    "Done constructing."
    """
    Take the ratio between the measured data and the expected data.
    Store this ratio in 'expected_data'
    """
    expected_data += 1e-6 #Don't want to divide by 0!
    np.divide(noisy_sim_data, expected_data, out=expected_data)
    """
    Apply the transpose of the expected data operation to the correction factor
    """
    sim_data_to_density(expected_data, out=correction_factor)
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
Make SIM data for a pointlike object
"""
pointlike_object = np.zeros_like(actual_object)
pointlike_object[pointlike_object.shape[0]//2,
                 pointlike_object.shape[1]//2] = 1
array_to_tif(pointlike_object.reshape((1,) + pointlike_object.shape
                                      ).astype(np.float32),
             outfile='pointlike_object.tif')
pointlike_sim_data = density_to_sim_data(pointlike_object)
sim_data_to_visualization(pointlike_sim_data, outfile='pointlike_sim_data.tif')
