
# Language Modeling with Phrase Induction

This repository contains the code used for the following paper:

[Improving Neural Language Models by Segmenting, Attending, and Predicting the Future](#)

```
@InProceedings{belinkov:2017:acl,
  author    = {Luo, Hongyin and Jiang, Lan and Belinkov, Yonatan and Glass, James},
  title     = {Improving Neural Language Models by Segmenting, Attending, and Predicting the Future},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  month     = {July},
  year      = {2019},
  address   = {Florence},
  publisher = {Association for Computational Linguistics},
}
```

This code is based on
+ [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm)
+ [Transformer-XL](https://github.com/kimiyoung/transformer-xl)

If you use this code or our results in your research, please cite as appropriate:

## Software Requirements

+ Python 3 and PyTorch 0.4 are required for the LSTM language models for PTB and Wikitext-2.
+ Python 2 and Tensorflow 1.12.0 are required for the Transformer-XL language model on Wikitext-3.

## Experiments

For data setup, run `./getdata.sh`.
This script collects the Mikolov pre-processed Penn Treebank and the WikiText-2 datasets and places them in the `data` directory.

### Word level Penn Treebank (PTB) with LSTM

You can train an LSTM language model on PTB using the following command. The checkpoint will be stored in ./models/
```
./train_span.sh MODEL_FILE_NAME
```
You will get a language model achieving perplexities of approximately 59.6 / 57.5 running this.

The finetuning process can be done with the following command,
```
./finetune_ptb.sh MODEL_FILE_NAME
```
The finetuning process can produce a language model achieves 57.8 / 55.7 perplexities.

### Word level WikiText-2 (WT2) with LSTM

You can train an LSTM language model on WT2 using the following command. The checkpoint will be stored in ./models/
```
./train_span_wt2.sh MODEL_FILE_NAME
```
You will get a language model achieving perplexities of approximately 68.4 / 65.2 running this.

The finetuning process can be done with the following command,
```
./finetune_ptb.sh MODEL_FILE_NAME
```
The finetuning process can produce a language model achieves 66.9 / 64.1 perplexities.

### Word level WikiText-103 (WT103) with Transformer-XL

CODE COMING SOON

Download the pretrained Transformer-XL + Phrase Induction model [here](https://drive.google.com/open?id=1aySA0MYa3oqHYycXhXjYGUZKnYdXphOM) to reproduce the 17.4 perplexity on the test set of WT103.
