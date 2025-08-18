# MHEL-LLAMO
Multilingual Historical Entity Linking with Large Language Models (LLMs)

This project aims to provide a benchmark for open-source LLMs in Historical Entity Linking by using a simple approach: using prompt engineering to filter candidates returned by a smaller bi-encoder model, i.e. BELA.


## Install Requirements

Due to dependency issues, the bi-encoder requires a different huggingface version than LLMs. For this reason, we suggest to create two different conda environments.

### Create BELA (bi-encoder) environment.

```
conda create -n bela39 -y python=3.9 && conda activate bela39
pip install -r requirements_bela.txt
```

```
conda create -n mhel-llamo -y python=3.9 && conda activate mhel-llamo
pip install -r requirements_llms.txt
```

## Perform Candidate Retrieval with BELA

```
conda activate bela39

python 0_get_candidates.py --dataset_path ./test_data/DZ_IT --output_dir ./results/DZ_IT --top_k 20 --lang it
```

## Perform Candidate Selection with LLM and Compute Metrics
```
conda activate mhel-llamo

python 1_prompt_llm.py --json_f results/DZ_IT/candidates_top20.json --dataset_path ./test_data/DZ_IT \
--output_dir ./results/DZ_IT/ --model_id meta-llama/Llama-3.1-8B-Instruct --hf_token your_secret_token

python 2_eval.py --path_data ./test_data/DZ_IT --path_results ./results/DZ_IT/
```