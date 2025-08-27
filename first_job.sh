#!/bin/bash
#SBATCH --job-name=first_job    # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --time=23:59:59
#SBATCH --partition=A100 
#SBATCH --nodes=1
#SBATCH --gpus=3                   

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job
JSON_FILE="./results/DZ_IT/candidates_test_top50_en.json"
DATASET_PATH="./test_data/DZ_IT"
OUTPUT_DIR="./results/DZ_IT/"
MODEL_ID="mistralai/Mixtral-8x7B-v0.1"
N_CANDIDATES=20
HF_TOKEN=""  

cd /home/infres/XXX/MHEL-LLAMO

source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm

# Execute the Python script with specific arguments
srun python prompt_llm.py \
    --json_f "$JSON_FILE" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --n_candidates $N_CANDIDATES \
    --hf_token "$HF_TOKEN"

# Print job completion time
echo "Job finished at: $(date)"