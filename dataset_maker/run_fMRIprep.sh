#!/bin/bash
#SBATCH --account=brainof
#SBATCH --job-name=fmriprep
#SBATCH --nodes=1
#SBATCH --array=0-30
#SBATCH --cpus-per-task=40
#SBATCH --partition=booster
#SBATCH --output=/path/to/fmriprep/dataset/slurm-%j.out
#SBATCH --error=/path/to/fmriprep/dataset/slurm-%j.err
#SBATCH --time=15:00:00


# --- Paths ---
export BIDS_DIR="/path/to/dataset"
export OUTPUT_DIR="${BIDS_DIR}/derivatives/fmriprep"
export WORK_DIR="/path/to/dataset_Work"
export LICENSE_FILE="/path/to/FreeSurfer/license.txt"
export CONTAINER="/path/to/fmriprep-25.0.0.sif"

# --- Config ---
export N_PARALLEL=4
export THREADS_PER_SUBJECT=20
export OMP_NUM_THREADS=18
export SUBJECTS_PER_BATCH=$((SLURM_NTASKS * N_PARALLEL)) # n nodes × 4 subjects per node
export MEM_MB_PER_SUBJECT=40000

# --- Subject list ---
SUBJECTS=($(ls -d ${BIDS_DIR}/sub-* | xargs -n 1 basename))

export N_SUBJECTS=${#SUBJECTS[@]}
N_BATCHES=$(((N_SUBJECTS + SUBJECTS_PER_BATCH - 1) / SUBJECTS_PER_BATCH))  # Ceiling of division

echo "Total subjects: $N_SUBJECTS"


START=$((SLURM_ARRAY_TASK_ID * N_PARALLEL))
BATCH_SUBJECTS=("${SUBJECTS[@]:$START:$N_PARALLEL}")

for SUB in "${BATCH_SUBJECTS[@]}"; do
    echo "[$(hostname)] Start $SUB" &

    /usr/bin/apptainer run --cleanenv \
      -B ${BIDS_DIR}:/data \
      -B ${OUTPUT_DIR}:/out \
      -B ${WORK_DIR}:/work \
      -B ${LICENSE_FILE}:/license.txt \
      ${CONTAINER} \
      /data /out participant \
      --participant-label ${SUB} \
      --fs-license-file /license.txt \
      --fs-subjects-dir /out/${SUB}/freesurfer \
      --work-dir /work/${SUB} \
      --nthreads ${THREADS_PER_SUBJECT} \
      --omp-nthreads ${OMP_NUM_THREADS} \
      --mem_mb ${MEM_MB_PER_SUBJECT} \
      --output-spaces MNI152NLin6Asym:res-2 MNI152NLin2009cAsym \
      --skip-bids-validation &
done
wait

echo "All batches completed."
