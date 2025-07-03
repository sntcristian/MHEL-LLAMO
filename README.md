# MultiEL
Multilingual Entity Linking model by BELA model

This project want to create easy-to-use Multilingual Entity Linking model by BELA model for entity linking in 98 languages.

**Origin Project**

- Bi-encoder Entity Linking Architecture (BELA): [https://github.com/facebookresearch/BELA](https://github.com/facebookresearch/BELA)
- Multilingual End to End Entity Linking: [https://arxiv.org/abs/2306.08896](https://arxiv.org/abs/2306.08896)


## Install

### 1. Create conda environment and install requirements

(optional) It might be a good idea to use a separate conda environment. Python 3.9 is recommended. It can be created by running:
```
conda create -n bela39 -y python=3.9 && conda activate blink39
pip install -r requirements.txt
```

### 2. Download the BLINK models

The BELA pretrained models can be downloaded using the following script:
```console
chmod +x download_models.sh
./download_models.sh
```

To run this implementation it is necessary to build the [FAISS](https://github.com/facebookresearch/faiss) indexer, which enables efficient exact/approximate retrieval for biencoder model.


To build and save FAISS index yourself, run
`python build_faiss.py`


## Test

`python entity_disambiguator.py`



## License

MIT license and the model is MIT license. ([BELA is MIT licensed](https://github.com/facebookresearch/BELA/blob/main/LICENSE))
