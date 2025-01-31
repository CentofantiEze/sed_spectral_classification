[![arXiv:2501.16151](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2203.04908-B31B1B.svg)](https://arxiv.org/abs/2501.16151) [![License](https://img.shields.io/badge/License-MIT-brigthgreen.svg)](https://github.com/CentofantiEze/sed_spectral_classification/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14780747.svg)](https://doi.org/10.5281/zenodo.14780747)

# SED Spectral Classification
Spectral star classifier from single wide-band images. The classifiers were trained and tested with simulated star images.

## Models
This repository contains three star spectral classification models:


The [PCA+MLP](https://github.com/CentofantiEze/sed_spectral_classification/blob/main/notebooks/classifiers/PCA_pixel_classifier_data_002.ipynb) classifier is based on a Principal Component Analysis (PCA) to reduce the dimensionality of the input images and a Multi-Layer Perceptron (MLP) to classify the stars into 13 spectral classes.


The [CNN+MLP](https://github.com/CentofantiEze/sed_spectral_classification/blob/main/notebooks/classifiers/CNN_pixel_compression_classifier_data_002.ipynb) classifier uses a Convolutional Neural Network (CNN) to extract features from the input images and a Multi-Layer Perceptron (MLP) to classify the stars into 13 spectral classes.

The [SVM+PSF](https://github.com/CentofantiEze/sed_spectral_classification/blob/main/notebooks/classifiers/svm_classifier_data_002.ipynb) model is a PSF-aware classifier that takes into account the spectral variation of the telescope's Point Spread Function (PSF) for breaking the degeneracy between the stellar type and the PSF size, hence enhancing the classification accuracy.

### Classification results

| Model         | F1-score  | Accuracy  | Top-two accuracy |
|---------------|-----------|-----------|------------------|
| PCA+MLP       | 0.366     | 0.370     | 0.757            |
| CNN+MLP       | 0.385     | 0.391     | 0.746            |
| **SVM+PSFGT** | **0.546** | **0.549** | **0.910**        |

## Data
The star images used for training and testing the classifiers as well as the PSF models, the trained models, and the results are available in [Zenodo](https://doi.org/10.5281/zenodo.14780747).

The data is organised as follows:
```
datasets/
├── Classification_datasets/
├── PSF_modelling_datasets/
├── Approximated_PSF_datasets/
└── Extra_stars_datasets/

classification_metrics/

PSF_models/
├── checkpoint/
├── metrics/
├── ...
└── psf_model/

Final_PSF_improvement/
└── metrics/
```

## Project outline
This repository contains all the necessary scripts and notebooks to reproduce the results presented in the paper. The project can be divided into the following steps:

### Training and testing the classifiers
1. Generate classification datasets:
    - 10.000 stars for training.
    - 1.000 stars for testing.

2. Train and test the pixel-only classifiers:
    - PCA+MLP
    - CNN+MLP

3. Generate PSF modelling datasets:
    - Nested datasets of 50, 100, 200, 500, 1.000, and 2.000 stars.

4. Train the PSF models.

5. Use the PSF models to predict aproximated PSFs for the classification training and testing stars.

6. Train and test the PSF-aware classifier:
    - SVM+PSF

### Improving the final PSF model
7. Baseline PSF model and dataset: PSF trained with 50 stars.

7. Use the SVM+PSF classifier to predict the SED of stars not included in the nested datasets.

8. Extend the baseline dataset with the predicted SEDs stars.

9. Train the final PSF model with the extended datasets.