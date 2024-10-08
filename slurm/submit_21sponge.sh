#!/bin/bash
#SBATCH --chdir="/home/twenger/tigress_hi"
#SBATCH --job-name="21sponge"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-56

eval "$(conda shell.bash hook)"
conda activate caribou_hi

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $SLURM_ARRAY_TASK_ID"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python scripts/run_21sponge.py $SLURM_ARRAY_TASK_ID
rm -rf $tmpdir
