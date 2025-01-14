# bela_experiments
scripts to run bela on ENEIDE dataset

## How to run experiments


### Step 1: Clone BELA Repository and download models

``` 
git clone https://github.com/facebookresearch/BELA.git

cd BELA

!./download_models.sh
``` 

### Step 2: Installing requirements (suggested Python 3.9)

``` 
conda create -n bela39 python=3.9

conda activate bela39

pip install -r requirements.txt

pip install --upgrade numpy==1.26.4
pip install --upgrade transformers==4.20.0

conda install pytorch::faiss-gpu
``` 

### Step 3: Running experiments

``` 

git clone https://github.com/sntcristian/bela_experiments.git

cd bela_experiments

python run_bela.py

``` 

The output will be shown in the `DZ_output.csv` file.