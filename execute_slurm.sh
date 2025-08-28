#!/bin/bash
#SBATCH --job-name=filter_and_prompt    # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --time=23:59:59
#SBATCH --partition=A100 
#SBATCH --nodes=1
#SBATCH --gpus=1                   

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job
JSON_FILE="./results/HIPE_FR/candidates_test_top50_fr.json"
DATASET_PATH="./test_data/HIPE_FR"
OUTPUT_DIR="./results/HIPE_FR/llama_3.1_8B_van_k20_en"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
N_CANDIDATES=20
HF_TOKEN=""  

cd /home/infres/xxx/MHEL-LLAMO

source /home/infres/xxx/anaconda3/etc/profile.d/conda.sh
conda activate llm

# Execute the Python script with specific arguments
srun python filter_and_prompt.py \
    --json_f "$JSON_FILE" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --n_candidates $N_CANDIDATES \
    --hf_token "$HF_TOKEN"

# Print job completion time
echo "Job finished at: $(date)"