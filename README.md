# Language Modeling with Phrase Induction

This repository contains the code used for the following paper:
+ [Improving Neural Language Models by Segmenting, Attending, and Predicting the Future](#)
This code and the README file are based on
+ [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm)
+ [Transformer-XL](https://github.com/kimiyoung/transformer-xl)

The model comes with instructions to train:
+ word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets

+ character level language models over the Penn Treebank (PTBC) and Hutter Prize dataset (enwik8)

The model can be composed of an LSTM or a [Quasi-Recurrent Neural Network](https://github.com/salesforce/pytorch-qrnn/) (QRNN) which is two or more times faster than the cuDNN LSTM in this setup while achieving equivalent or better accuracy.

+ Install PyTorch 0.4
+ Run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets
+ Train the base model using `main.py`
+ (Optionally) Finetune the model using `finetune.py`
+ (Optionally) Apply the [continuous cache pointer](https://arxiv.org/abs/1612.04426) to the finetuned model using `pointer.py`

If you use this code or our results in your research, please cite as appropriate:

## Software Requirements

Python 3 and PyTorch 0.4 are required for the current codebase.

Included below are hyper parameters to get equivalent or better results to those included in the original paper.

If you need to use an earlier version of the codebase, the original code and hyper parameters accessible at the [PyTorch==0.1.12](https://github.com/salesforce/awd-lstm-lm/tree/PyTorch%3D%3D0.1.12) release, with Python 3 and PyTorch 0.1.12 are required.
If you are using Anaconda, installation of PyTorch 0.1.12 can be achieved via:
`conda install pytorch=0.1.12 -c soumith`.

## Experiments

For data setup, run `./getdata.sh`.
This script collects the Mikolov pre-processed Penn Treebank and the WikiText-2 datasets and places them in the `data` directory.

### Word level Penn Treebank (PTB) with LSTM

The instruction below trains a PTB model that without finetuning achieves perplexities of approximately `61.2` / `58.8` (validation / testing), with finetuning achieves perplexities of approximately `58.8` / `56.5`, and with the continuous cache pointer augmentation achieves perplexities of approximately `53.2` / `52.5`.

+ `python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
+ `python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
+ `python pointer.py --data data/penn --save PTB.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000`

### Word level WikiText-2 (WT2) with LSTM
The instruction below trains a PTB model that without finetuning achieves perplexities of approximately `68.7` / `65.6` (validation / testing), with finetuning achieves perplexities of approximately `67.4` / `64.7`, and with the continuous cache pointer augmentation achieves perplexities of approximately `52.2` / `50.6`.

+ `python main.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ `python finetune.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ `python pointer.py --save WT2.pt --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt 2000 --data data/wikitext-2`

### Details of the LSTM optimization

All the augmentations to the LSTM, including our variant of [DropConnect (Wan et al. 2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) termed weight dropping which adds recurrent dropout, allow for the use of NVIDIA's cuDNN LSTM implementation.
PyTorch will automatically use the cuDNN backend if run on CUDA with cuDNN installed.
This ensures the model is fast to train even when convergence may take many hundreds of epochs.
