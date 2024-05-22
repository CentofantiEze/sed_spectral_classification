#!/bin/bash
#SBATCH --job-name="psf_est_gen"
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=END
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=psf_est_gen-%j.log

# Activate conda environment
module load anaconda
source activate $ANACONDA_DIR
conda activate wavediff

# echo des commandes lancees
set -x

# Change location
cd $WORK/sed_spectral_classification/scripts

# Run code
srun python gen-estimated-PSF-parallel-8bins.py

# Return exit code
exit 0
