#! /feynman/work/dap/lcs/as274094/.conda/envs/psf/bin/python

import numpy as np
# import wf_psf as wf_psf
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count, parallel_backend

from wf_psf.utils.utils import add_noise
from wf_psf.sims.psf_simulator import PSFSimulator
from wf_psf.sims.spatial_varying_psf import SpatialVaryingPSF
from wf_psf.sims.spatial_varying_psf import ZernikeHelper

import sys
import time

#######################################
#      PARALLEL      FUNCTIONS        #
#######################################
# Function to get (i,j) from id
def unwrap_id(id, n_cpus):
    i = int(id/n_cpus)
    j = int(id - i * n_cpus)
    return i, j

def print_status(star_id, i, j):
    print('\nStar ' +str(star_id)+ ' done!' + '   index=('+str(i)+','+str(j)+')')

# Get batches from a list
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# Funcion to get mono PSFs
def generate_multi_mono_PSF(PSF_toolkit, SED, n_bins=35):
    """Generate varios monochromatic PSFs.

    The wavelength space will be the Euclid VIS instrument band:
    [550,900]nm and will be sampled in ``n_bins``.

    """
    # Calculate the feasible values of wavelength and the corresponding
    # SED interpolated values
    feasible_wv, SED_norm = PSF_toolkit.calc_SED_wave_values(SED, n_bins)
    mono_psfs = []
    stacked_psf = 0

    # Generate the required monochromatic PSFs
    for it in range(feasible_wv.shape[0]):
        PSF_toolkit.generate_mono_PSF(lambda_obs=feasible_wv[it])
        # Append the non-weighted mono PSF
        mono_psfs.append( PSF_toolkit.get_psf())
        stacked_psf += PSF_toolkit.get_psf() * SED_norm[it]

    return mono_psfs, stacked_psf

# Function to get one PSF
def simulate_star(star_id, sim_PSF_toolkit_):
    i_,j_ = unwrap_id(star_id, n_cpus)
    sim_PSF_toolkit_[j_].set_z_coeffs(zks[star_id])
    mono_psfs, _psf = generate_multi_mono_PSF(sim_PSF_toolkit_[j_],SED_list[star_id], n_bins)
    # # Change output parameters to get the super resolved PSF
    # sim_PSF_toolkit_[j_].output_Q = super_out_Q
    # sim_PSF_toolkit_[j_].output_dim = super_out_res
    # super_psf = sim_PSF_toolkit_[j_].generate_poly_PSF(SED_list[star_id], n_bins)
    # # Put back original parameters
    # sim_PSF_toolkit_[j_].output_Q = original_out_Q
    # sim_PSF_toolkit_[j_].output_dim = original_out_dim
    return (star_id, mono_psfs, _psf, zks[star_id])#, super_psf)

# Paths
wf_psf_dir = '/feynman/work/dap/lcs/ec270266/wf-psf/'
output_dir = '/feynman/work/dap/lcs/ec270266/sed_spectral_classification/output/'
# wf_psf_dir = '/Users/ec270266/Documents/Phd/Euclid/dev/wf-psf/'
# output_dir = '/Users/ec270266/Documents/Phd/Euclid/dev/feature-sed-pred/sed_spectral_classification/output/'
# SED folder path
SED_path = wf_psf_dir+'data/SEDs/save_SEDs/'
# Reference dataset PATH
reference_data_train = 'train_10000_stars_id_001_8bins.npy'
reference_data_test = 'test_1000_stars_id_001_8bins.npy'

# Output saving path (in node05 of candide or $WORK space on feynman)
# output_folder = '/feynman/work/dap/lcs/ec270266/output/interp_SEDs/'
output_folder = output_dir+'psf_dataset/'

# Reference dataset PATH
# reference_data = '../interp_SED_data/reference_dataset/'
# reference_data = wf_psf_dir+'data/coherent_euclid_dataset/'
# ref_train = 'train_Euclid_res_2000_TrainStars_id_001.npy'
# ref_test  = 'test_Euclid_res_id_001.npy'
#selected_id_SED_path = 'selected_id_SED.npy'

# Number of cpus to use for parallelization
n_cpus = 32 #verify that it doesn't reach the N of actual CPUs

# Save output prints to logfile
old_stdout = sys.stdout
log_file = open(output_folder + 'PSF_estimate_output.log','w')
sys.stdout = log_file
print('Starting the log file.')

# Dataset ID
dataset_id = 1
dataset_id_str = '%03d'%(dataset_id)

# This list must be in order from bigger to smaller
n_star_list = [10000, 2000, 500]
n_test_stars = 1000
# n_star_list = [100]
# n_test_stars = 10
# Total stars
n_stars = n_star_list[0] + n_test_stars
# Max train stars
tot_train_stars = n_star_list[0]

# Parameters
d_max = 4
max_order = 66
ZK_label = '_{}zks'.format(max_order)
x_lims = [0, 1e3]
y_lims = [0, 1e3]
grid_points = [4, 4]
n_bins = 8

oversampling_rate = 3.
output_Q = 3.

# max_wfe_rms = 0.1
output_dim = 32
LP_filter_length = 2
euclid_obsc = True

snr_max = 50
snr_min = 10
SNR_label = ''

# Values for getting 3xEuclid_resolution PSFs outputs.
original_out_Q = output_Q
original_out_dim = output_dim
super_out_Q = 1
super_out_res = 64

# Desired WFE resolutions
pupil_diameter = 256

print('\nInit dataset generation')

# Load the SEDs
stellar_SEDs = np.load(SED_path + 'SEDs.npy', allow_pickle=True)
stellar_lambdas = np.load(SED_path + 'lambdas.npy', allow_pickle=True)

# Load reference dataset
train_dataset_ref = np.load(output_folder+reference_data_train , allow_pickle=True)[()]
test_dataset_ref = np.load(output_folder+reference_data_test, allow_pickle=True)[()]

# Load all the stars positions
pos_np = np.vstack((train_dataset_ref['positions'],test_dataset_ref['positions']))

# Assign preselected SEDs
SED_id_list = train_dataset_ref['SED_ids']+test_dataset_ref['SED_ids']
SED_list = []
for it in range(n_stars):
    concat_SED_wv = np.concatenate((
        stellar_lambdas.reshape(-1,1),
        stellar_SEDs[SED_id_list[it],:].reshape(-1,1)
    ), axis=1)
    SED_list.append(concat_SED_wv)

# # Choose the locations randomly
# pos_np = np.random.rand(n_stars, 2)

# pos_np[:,0] = pos_np[:,0]*(x_lims[1] - x_lims[0]) + x_lims[0]
# pos_np[:,1] = pos_np[:,1]*(y_lims[1] - y_lims[0]) + y_lims[0]    

# # Select random SEDs
# SED_list = []
# SED_id_list = []
# for it in range(n_stars):
#     selected_id_SED = np.random.randint(low=0, high=13)
#     concat_SED_wv = np.concatenate((
#         stellar_lambdas.reshape(-1,1),
#         stellar_SEDs[selected_id_SED,:].reshape(-1,1)), axis=1)
#     SED_id_list.append(selected_id_SED)
#     SED_list.append(concat_SED_wv)

# Load and assign the C_poly matrix
C_poly = train_dataset_ref['C_poly'][:max_order,:]

# Generate aproximated PSF model for each WFE error level
desired_wfe_rms = np.array([100, 50, 10, 5, 1, 0.5, 0.1]) * 1e-3
desired_wfe_rms = np.array([5, 10]) * 1e-3

for wfe_rms in desired_wfe_rms:
    print('##############################################')
    print('Generating dataset for WFE RMS = {:.0e}'.format(wfe_rms))
    print('##############################################')
    # Initialize PSF simulator for each cpu available 
    # (no euclid obscurations and wfr_rms init)
    sim_PSF_toolkit = [PSFSimulator(
        max_order=max_order,
        max_wfe_rms=wfe_rms,
        oversampling_rate=oversampling_rate,
        output_Q=output_Q,
        output_dim=output_dim,
        pupil_diameter=pupil_diameter,
        euclid_obsc=euclid_obsc,
        LP_filter_length=LP_filter_length
    ) for j in range(n_cpus)]

    obscurations = sim_PSF_toolkit[0].obscurations   

    error_field = SpatialVaryingPSF(
        psf_simulator=sim_PSF_toolkit[0],
        d_max=d_max,
        grid_points=grid_points,
        max_order=max_order,
        x_lims=x_lims,
        y_lims=y_lims,
        n_bins=n_bins,
        lim_max_wfe_rms=wfe_rms,
    )

    print('\nStar positions selected')

    # Compute zernikes for each star
    zks = ZernikeHelper.calculate_zernike(pos_np[:,0], pos_np[:,1], x_lims, y_lims, d_max, error_field.polynomial_coeffs+C_poly).T

    print('\nZernikes calculated')

    #######################################
    #            PARALELLIZED             #
    #######################################

    # Total number of stars
    n_procs = n_stars

    # Print some info
    cpu_info = ' - Number of available CPUs: {}'.format(cpu_count())
    cpu_use = ' - Number of selected CPUs: {}'.format(n_cpus)
    proc_info = ' - Total number of processes: {}'.format(n_procs)
    print(cpu_info)
    print(cpu_use)
    print(proc_info)

    # Generate star list
    star_id_list = [id_ for id_ in range(n_stars)]

    # Measure time
    start_time = time.time()

    index_i_list = []
    psf_i_list = []
    z_coef_i_list = []
    mono_psfs_i_list = []
    for batch in chunker(star_id_list, n_cpus):
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_cpus, verbose=100)(delayed(simulate_star)(_star_id, sim_PSF_toolkit)
                                                for _star_id in batch)
        index_batch, mono_psfs_batch, psf_batch,z_coef_batch = zip(*results)
        index_i_list.extend(index_batch)
        psf_i_list.extend(psf_batch)
        z_coef_i_list.extend(z_coef_batch)
        mono_psfs_i_list.extend(mono_psfs_batch)

    index = np.array(index_i_list)
    poly_psf = np.array( psf_i_list)
    zernike_coef = np.array(z_coef_i_list)
    mono_psfs = np.array(mono_psfs_i_list)

    end_time = time.time()
    print('\nAll stars generated in '+ str(end_time-start_time) +' seconds')

    #######################################
    #            END PARALLEL             #
    #######################################

    # Add noise to generated train star PSFs and save datasets

    # SNR varying randomly from snr_min to snr_max - shared over all WFE resolutions
    rand_SNR_train = (np.random.rand(tot_train_stars) * (snr_max-snr_min)) + snr_min
    # Copy the training stars
    train_stars = np.copy(poly_psf[:tot_train_stars, :, :])
    # Add Gaussian noise to the observations
    noisy_train_stars = np.stack([
        add_noise(_im, desired_SNR=_SNR) 
        for _im, _SNR in zip(train_stars, rand_SNR_train)], axis=0)
    # # Generate Gaussian noise patterns to be shared over all datasets (but not every star)
    # noisy_train_patterns = noisy_train_stars - train_stars


    # Add noise to generated test star PSFs and save datasets

    # SNR varying randomly from snr_min to snr_max - shared over all WFE resolutions
    rand_SNR_test = (np.random.rand(n_test_stars) * (snr_max-snr_min)) + snr_min
    # Copy the test stars
    test_stars = np.copy(poly_psf[tot_train_stars:, :, :])
    # Add Gaussian noise to the observations
    noisy_test_stars = np.stack([
        add_noise(_im, desired_SNR=_SNR) 
        for _im, _SNR in zip(test_stars, rand_SNR_test)], axis=0)
    # Generate Gaussian noise patterns to be shared over all datasets (but not every star)
    # noisy_test_patterns = noisy_test_stars - test_stars


    # Generate datasets
    # Generate numpy array from the SED list
    SED_np = np.array(SED_list)

    # Save only one test dataset
    # Build param dicitionary
    dataset_params = {
        'd_max':d_max,
        'max_order':max_order,
        'x_lims':x_lims,
        'y_lims':y_lims,
        'grid_points':grid_points,
        'n_bins':n_bins,
        'max_wfe_rms':wfe_rms,
        'oversampling_rate':oversampling_rate,
        'output_Q':output_Q,
        'output_dim':output_dim,
        'LP_filter_length':LP_filter_length,
        'pupil_diameter':pupil_diameter,
        'euclid_obsc':euclid_obsc,
        'n_stars':n_test_stars
    }

    test_psf_dataset = {
        'stars' : poly_psf[tot_train_stars:, :, :],
        'noisy_stars': noisy_test_stars,
        'mono_psfs' : mono_psfs[tot_train_stars:, :, :],
        'positions' : pos_np[tot_train_stars:, :],
        'SEDs' : SED_np[tot_train_stars:, :, :],
        'zernike_coef' : zernike_coef[tot_train_stars:, :],
        'C_poly' : train_dataset_ref['C_poly'],
        'C_poly_err' : C_poly+error_field.polynomial_coeffs,
        'parameters': dataset_params,
        'SED_ids':SED_id_list[tot_train_stars:],
        'SNR': rand_SNR_test
    }

    np.save(
        output_folder + 'test_' + str(n_test_stars) + '_stars_id_' + dataset_id_str + '_' + 
        str(n_bins)+'bins'+ZK_label+'_rms_' + '{:.0e}'.format(wfe_rms)+'.npy',
        test_psf_dataset,
        allow_pickle=True
    )



    # Save the different train datasets
    for it_glob in range(len(n_star_list)):

        n_train_stars = n_star_list[it_glob]

        # Build param dicitionary
        dataset_params = {
            'd_max':d_max,
            'max_order':max_order,
            'x_lims':x_lims,
            'y_lims':y_lims,
            'grid_points':grid_points,
            'n_bins':n_bins,
            'max_wfe_rms':wfe_rms,
            'oversampling_rate':oversampling_rate,
            'output_Q':output_Q,
            'output_dim':output_dim,
            'LP_filter_length':LP_filter_length,
            'pupil_diameter':pupil_diameter,
            'euclid_obsc':euclid_obsc,
            'n_stars':n_train_stars
        }

        train_psf_dataset = {
            'stars' : poly_psf[:n_train_stars, :, :],
            'noisy_stars': noisy_train_stars[:n_train_stars, :, :],
            'mono_psfs' : mono_psfs[:n_train_stars, :, :],
            'positions' : pos_np[:n_train_stars, :],
            'SEDs' : SED_np[:n_train_stars, :, :],
            'zernike_coef' : zernike_coef[:n_train_stars, :],
            'C_poly' : train_dataset_ref['C_poly'],
            'C_poly_err' : C_poly+error_field.polynomial_coeffs,
            'parameters': dataset_params,
            'SED_ids' : SED_id_list[:n_train_stars],
            'SNR': rand_SNR_train
        }


        np.save(
            output_folder + 'train_' + str(n_train_stars) + '_stars_id_' + dataset_id_str + '_' + 
            str(n_bins)+'bins'+ZK_label+'_rms_' + '{:.0e}'.format(wfe_rms)+'.npy',
            train_psf_dataset,
            allow_pickle=True
        )

print('\nDone!')

# Close log file
sys.stdout = old_stdout
log_file.close()
