# MHEL-LLAMO
Multilingual Historical Entity Linking with Large Language Models (LLMs)

This project aims to provide a benchmark for open-source LLMs in Historical Entity Linking by using an ensemble approach which combines a multilingual bi-encoder model, i.e. BELA, for candidate retrieval with prompt engineering for NIL prediction and candidate selection.


## Install Requirements

Due to dependency issues, the bi-encoder requires a different huggingface version than LLMs. For this reason, we suggest to create two different conda environments.

### Create BELA (bi-encoder) environment

```
conda create -n bela39 -y python=3.9 && conda activate bela39
pip install -r requirements_bela.txt
```

### Create LLM environment
```
conda create -n llm -y python=3.9 && conda activate llm
pip install -r requirements_llms.txt
```

## Perform Candidate Retrieval with BELA

```
conda activate bela39

python get_candidates.py --dataset_path ./test_data/HIPE_EN --output_dir ./results/HIPE_EN --top_k 50 --lang en
```

## Perform Candidate Selection with LLM and Compute Metrics
```
conda activate llm

python filter_and_prompt_chain.py \
--json_f results/HIPE_EN/candidates_test_top50_en.json \
--dataset_path ./test_data/HIPE_EN \
--output_dir ./results/HIPE_EN \ 
--threshold 21.24 \ # optional
--n_candidates 20 \ # optional
--model_id mistralai/Mistral-Small-24B-Instruct-2501 \
--hf_token your_secret_token # only with gated models

python eval.py --path_data ./test_data/HIPE_EN --path_results ./results/HIPE_EN
```

For more accurate parameters, see our publication.
