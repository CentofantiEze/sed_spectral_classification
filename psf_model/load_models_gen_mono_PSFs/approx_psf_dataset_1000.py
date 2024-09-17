from wf_psf.psf_models import psf_models
from wf_psf.utils.read_config import read_stream
from wf_psf.utils.configs_handler import *
import logging.config
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import tensorflow as tf

# Pre-defined colormap
top = mpl.colormaps['Oranges_r'].resampled(128)
bottom = mpl.colormaps['Blues'].resampled(128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

model_id = 1000

# Path to the pretrained model

pretrained_models_path = '/feynman/work/dap/lcs/ec270266/sed_spectral_classification/psf_model/outputs/wf-outputs/pretrained_models/'
# pretrained_models_path = '/Users/ec270266/Documents/Phd/Euclid/dev/feature-sed-pred/sed_spectral_classification/psf_model/outputs/wf-outputs/pretrained_models/'
config_files_folder = pretrained_models_path + 'config/'
# Config files
model_config_file = pretrained_models_path + 'config/configs_{}.yaml'.format(model_id)
model_training_config_file = pretrained_models_path + 'config/training_config_{}.yaml'.format(model_id)
model_data_config_file = pretrained_models_path + 'config/data_config_{}.yaml'.format(model_id)

checkpoint_dir = pretrained_models_path + 'psf_model/psf_{}/'.format(model_id)
repo_dir = '/feynman/work/dap/lcs/ec270266/wf-psf/'
# output_dir = '../outputs/'
output_dir = '/feynman/work/dap/lcs/ec270266/sed_spectral_classification/psf_model/outputs/'

plot = False

# Load the model configuration
configs_path = os.path.dirname(model_config_file)
configs = read_stream(model_config_file)
configs_file = os.path.basename(model_config_file)

file_handler = FileIOHandler(repo_dir, output_dir, configs_path)
file_handler.setup_outputs()
file_handler.copy_conffile_to_output_dir(configs_file)

for i, conf in enumerate(configs):
    # print(i, conf)
    for k, v in conf.items():
        # print(k)
        # print(v)
        # do nothing
        pass

config_class = get_run_config(k, os.path.join(config_files_folder, v), file_handler)
train_conf_handler = TrainingConfigHandler(model_training_config_file, file_handler)
data_conf_handler = DataConfigHandler(model_data_config_file, train_conf_handler.training_conf.training.model_params)

########################
# Create the PSF model #
########################
psf_model = psf_models.get_psf_model(
        train_conf_handler.training_conf.training.model_params,
        train_conf_handler.training_conf.training.training_hparams,
        data_conf_handler,
    )

######################
# Load model weights #
######################

chkp = tf.train.latest_checkpoint(checkpoint_dir)
psf_model.load_weights(chkp).expect_partial()

inputs_0 = data_conf_handler.test_data.dataset["positions"]
inputs_1 = data_conf_handler.test_data.sed_data

gt_psfs = data_conf_handler.test_data.dataset['stars']

zks = data_conf_handler.test_data.dataset['zernike_coef']
zks_tf = zks.reshape(zks.shape[0], -1, 1, 1)
zks_tf.shape
gt_wfe = psf_model.tf_zernike_OPD(zks_tf)

if plot == True:
    idx=np.random.randint(0, inputs_0.shape[0])

    opd_test =psf_model.predict_opd(inputs_0[idx:idx+1])
    psf = psf_model.predict([inputs_0[idx:idx+1], inputs_1[idx:idx+1]])

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    cax = ax[1][0].imshow(opd_test[0], cmap=newcmp, vmin=-.2, vmax=.2)
    fig.colorbar(cax, ax=ax[1][0], orientation='vertical')
    cax = ax[1][1].imshow(gt_wfe[idx], cmap=newcmp, vmin=-.2, vmax=.2)
    fig.colorbar(cax, ax=ax[1][1], orientation='vertical')
    cax = ax[1][2].imshow(np.abs(opd_test[0] - gt_wfe[idx]), cmap='viridis')
    fig.colorbar(cax, ax=ax[1][2], orientation='vertical')
    
    ax[0][0].imshow(psf[0], cmap='gist_stern')
    ax[0][1].imshow(gt_psfs[idx], cmap='gist_stern')
    cax = ax[0][2].imshow(np.abs(psf[0] - gt_psfs[idx]), cmap='viridis')
    fig.colorbar(cax, ax=ax[0][2], orientation='vertical')

    ax[1][0].set_title('Predicted WFE')
    ax[1][1].set_title('Ground Truth WFE')
    ax[1][2].set_title('WFE residual')
    ax[0][0].set_title('Predicted PSF')
    ax[0][1].set_title('Ground Truth PSF')
    ax[0][2].set_title('PSF residual')
    plt.show()

########################
# Load the 12k dataset #
########################
idx_offset = 0
n_stars = 12000

dataset_10k_path = '/feynman/work/dap/lcs/ec270266/sed_spectral_classification/output/psf_dataset/train_12000_stars_id_002_8bins.npy'
# dataset_10k_path = '/Users/ec270266/Documents/Phd/Euclid/dev/feature-sed-pred/sed_spectral_classification/output/psf_dataset/train_12000_stars_id_002_8bins.npy'
dataset_10k = np.load(dataset_10k_path, allow_pickle=True)[()]
print(dataset_10k.keys())
packed_SEDs = np.array([dataset_10k['packed_SEDs'][i].T for i in dataset_10k['SED_ids']])
packed_SEDs.shape

inputs_0 = dataset_10k['positions'][idx_offset:idx_offset+n_stars]
inputs_1 = packed_SEDs[idx_offset:idx_offset+n_stars]

gt_psfs = dataset_10k['stars'][idx_offset:idx_offset+n_stars]

zks = dataset_10k['zernike_coef'][idx_offset:idx_offset+1]
#zks_tf = zks.reshape(zks.shape[0], -1, 1, 1)
# time consuming (kernel might crash)#
#gt_wfe = psf_model.tf_zernike_OPD(zks_tf)
print('opd_skipped')

if plot == True:
    idx=0 #np.random.randint(0, inputs_0.shape[0])
    opd_test =psf_model.predict_opd(inputs_0[idx:idx+1])
    psf = psf_model.predict([inputs_0[idx:idx+1], inputs_1[idx:idx+1]])

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    cax = ax[1][0].imshow(opd_test[0], cmap=newcmp, vmin=-.2, vmax=.2)
    fig.colorbar(cax, ax=ax[1][0], orientation='vertical')
    cax = ax[1][1].imshow(gt_wfe[idx], cmap=newcmp, vmin=-.2, vmax=.2)
    fig.colorbar(cax, ax=ax[1][1], orientation='vertical')
    cax = ax[1][2].imshow(np.abs(opd_test[0] - gt_wfe[idx]), cmap='viridis')
    fig.colorbar(cax, ax=ax[1][2], orientation='vertical')
    
    ax[0][0].imshow(psf[0], cmap='gist_stern')
    ax[0][1].imshow(gt_psfs[idx], cmap='gist_stern')
    cax = ax[0][2].imshow(np.abs(psf[0] - gt_psfs[idx]), cmap='viridis')
    fig.colorbar(cax, ax=ax[0][2], orientation='vertical')

    ax[1][0].set_title('Predicted WFE')
    ax[1][1].set_title('Ground Truth WFE')
    ax[1][2].set_title('WFE residual')
    ax[0][0].set_title('Predicted PSF')
    ax[0][1].set_title('Ground Truth PSF')
    ax[0][2].set_title('PSF residual')
    plt.show()

#############################
# Predict the 10k mono PSFs #
#############################

#mono_psfs = [[psf_model.predict_mono_psfs(inputs_0[idx:idx+1], float(inputs_1[idx:idx+1,wv,1]), int(inputs_1[idx:idx+1,wv,0]))[0] for wv in range(dataset_10k['parameters']['n_bins'])] for idx in range(n_stars)]
#mono_psfs = np.array(mono_psfs)

n_batch = 12000//32
batch_size = 32
mono_psfs = np.empty((0, 8, 32, 32))
inputs_0 = np.array(inputs_0)
inputs_1 = np.array(inputs_1)
for i in range(n_batch):
	print('predict batch number:', i)
# 	mono_psfs_batch = np.array([psf_model.predict_mono_psfs(tf.convert_to_tensor(inputs_0[i*batch_size:(i+1)*batch_size]), in_1[1], int(in_1[0])) for in_1 in inputs_1[0,:,:2]])
# 	print('batch {} done'.format(i))
# 	mono_psfs = np.append(mono_psfs, np.swapaxes(mono_psfs_batch, 0, 1), axis=0)
# print(mono_psfs.shape)

# plot mono psfs and the poly psf
if plot == True:
    im, ax = plt.subplots(2, dataset_10k['parameters']['n_bins'], figsize=(14, 4))

    for i in range(dataset_10k['parameters']['n_bins']):
        ax[0][i].imshow(dataset_10k['mono_psfs'][2000,i], cmap='gist_stern')
        ax[0][i].axis('off')
        ax[1][i].imshow(mono_psfs[0,i], cmap='gist_stern')
        ax[1][i].axis('off')

######################
# Save the mono PSFs #
######################

# dataset_10k['mono_psfs_approx'] = mono_psfs
# np.save(dataset_10k_path[:-4]+'_approx_{}_stars.npy'.format(model_id), dataset_10k, allow_pickle=True)

#########################
# Predict the test PSFs #
#########################

# Load the test dataset

dataset_test_path = '/feynman/work/dap/lcs/ec270266/sed_spectral_classification/output/psf_dataset/test_1000_stars_id_002_8bins.npy'
# dataset_test_path = '/Users/ec270266/Documents/Phd/Euclid/dev/feature-sed-pred/sed_spectral_classification/output/psf_dataset/test_1000_stars_id_002_8bins.npy'
dataset_test = np.load(dataset_test_path, allow_pickle=True)[()]
print(dataset_test.keys())
packed_SEDs = np.array([dataset_test['packed_SEDs'][i].T for i in dataset_test['SED_ids']])
packed_SEDs.shape

inputs_0 = dataset_test['positions']
inputs_1 = packed_SEDs

print('Predict the test PSFs')
# Predict the mono PSFs
batch_size = 20
n_batch = 1000//batch_size
mono_psfs = np.empty((0, 8, 32, 32))
inputs_0 = np.array(inputs_0)
inputs_1 = np.array(inputs_1)
for i in range(n_batch):
	print('predict batch number:', i)
	mono_psfs_batch = np.array([psf_model.predict_mono_psfs(tf.convert_to_tensor(inputs_0[i*batch_size:(i+1)*batch_size]), in_1[1], int(in_1[0])) for in_1 in inputs_1[0,:,:2]])
	print('batch {} done'.format(i))
	mono_psfs = np.append(mono_psfs, np.swapaxes(mono_psfs_batch, 0, 1), axis=0)
print(mono_psfs.shape)

# Save the mono PSFs

dataset_test['mono_psfs_approx'] = mono_psfs
np.save(dataset_test_path[:-4]+'_approx_{}_stars.npy'.format(model_id), dataset_test, allow_pickle=True)

