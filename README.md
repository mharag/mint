# MIni Neural Translator

This repository was created as a part of project for the course ZPJa at FIT VUT.

This is a simple from scratch implementation of a neural translator.
It's based on Transformer architecture (with small modifications).

This repository contains:
- `experiments/` - contains multiple notebooks for training, evaluation, preparing data, etc.
- `mint/` - contains the implementation of model, translator, trainer...
- `pretrained/` - contains pretrained models
- `tokenizers/` - contains tokenizers for different languages

# Usage

### Install the requirements:
```bash
pip install -r requirements.txt
```
You also need pytorch installed. You can install it from [official website](https://pytorch.org/).

### Inference

For inference check the `experiments/translate.ipynb` notebook.

### Training

For training check the `experiments/train.ipynb` notebook.

You will need to prepare dataset to the correct format.
For that you can use the `experiments/prepare_dataset.ipynb` notebook.

### Evaluation

To evaluate the model you can use the `experiments/evaluate.ipynb` notebook.

# Pretrained models

You can find some pretrained models in the `pretrained/` directory.
All models were trained on subset of the [CCMatrix dataset](https://opus.nlpl.eu/CCMatrix/en&sk/v1/CCMatrix).
I trained it on single GPU (RTX 3060). Training of the bigger model with batch_size=8 took around 2 hours for 100k samples.
I didn't really tune the hyperparameters or tried to optimize the performance so don't expect any miracles :). 
This is more of a demonstration of the model, training process and the translation.


| Model        | Size  | Total training samples | BLEU | chrF2 |
|--------------|-------|------------------------|------|-------|
| en_sk_small  | 126M  | 200k                   | 0.00 | 1.07  | 
| en_sk_medium | 289M  | ...                    | ...  | ...   |