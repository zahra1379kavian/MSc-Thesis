#!/bin/bash
#SBATCH --job-name=fmri_bigmem
#SBATCH --array=0-11
#SBATCH --account=st-mmckeown-1
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --mail-user=zkavian@student.ubc.ca
#SBATCH --mail-type=END,FAIL

module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11

export HOME=/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis
source /scratch/st-mmckeown-1/zkavian/fmri_models/myenv/bin/activate
cd /scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/

python mian_second_object_25folds.py --combo-idx "${SLURM_ARRAY_TASK_ID}"
# python matrix_diagnostics.py
