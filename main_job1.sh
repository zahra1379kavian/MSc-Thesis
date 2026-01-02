#!/bin/bash
#SBATCH --job-name=fmri_opt
#SBATCH --account=st-mmckeown-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=zkavian@student.ubc.ca
#SBATCH --mail-type=END,FAIL

module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11

export HOME=/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis
source /scratch/st-mmckeown-1/zkavian/fmri_models/myenv/bin/activate
cd /scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/

# python mian_second_obj_25.py
# python tmp.py
python trial_roi_beta_analysis.py
