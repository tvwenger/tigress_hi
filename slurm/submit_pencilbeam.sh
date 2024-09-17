#!/bin/bash
#SBATCH --chdir="/home/twenger/tigress_hi"
#SBATCH --job-name="pencilbeam"
#SBATCH --output="logs/HI_joint_spectra_logs/%x.%j.%N.out"
#SBATCH --error="logs/HI_joint_spectra_logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-999%175

# 81920 spectra = 1000 jobs of 82 spectra
# limit to 175 tasks to limit resource usage
DIR_NAME="HI_joint_spectra"
PER_JOB=82
NUM_SPEC=81920
START_IDX=$(( $SLURM_ARRAY_TASK_ID * $PER_JOB ))
END_IDX=$(( ( $SLURM_ARRAY_TASK_ID + 1 ) * $PER_JOB ))

eval "$(conda shell.bash hook)"
conda activate caribou_hi

for (( idx=$START_IDX; idx<$END_IDX; idx++ )); do
    if [ $idx -ge $NUM_SPEC ]; then
	break
    fi
    
    fmtidx=$(printf "%06d" $idx)
    # check if result already exists, then skip
    if [ -f "results/${DIR_NAME}_results/$fmtidx.pkl" ]; then
        echo "results/${DIR_NAME}_results/$fmtidx.pkl already exists!"
        continue
    fi

    # temporary pytensor compiledir
    tmpdir=`mktemp -d`
    echo "starting to analyze $idx"
    PYTENSOR_FLAGS="base_compiledir=$tmpdir" python scripts/run_pencilbeam.py $DIR_NAME $idx
    rm -rf $tmpdir
done
