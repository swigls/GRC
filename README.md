This repository contains the original code of Gated Recurrent Context (paper links:[arXiv](https://arxiv.org/pdf/2007.05214.pdf), [IEEE Transactions on ASLP](https://ieeexplore.ieee.org/document/9314198))().

This repository includes a pipeline scripts for training & inference of AED (Attention-based Encoder-Decoder) models for speech recognition task on [LibriSpeech](http://www.openslr.org/12/) dataset.

All the included scripts are largely based on the [RETURNN](https://github.com/rwth-i6/returnn) toolkit.

# Installation
Installation process includes data download & preparation.

Basically follow the same process as [installation guide](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention/librispeech/full-setup-attention) of original setup by Albert Zeyer,
only with a little modification as follows. 

### Modification in RETURNN scripts
As this repository already contains RETURNN toolkit scripts in returnn/ directory, 
the '03_git_clone_returnn.sh' script do not have to be executed.

If you want to use up-to-date version of RETURNN scripts rather than the RETURNN originally included in this repository,
please check the instructions in '03_git_clone_returnn.sh' and follow them.  

### Installation steps
Just execute scripts '01_pip_install_requirements.sh' to '21_prepare_train.sh', in index-order of pipeline bash-scripts. 


# Training
```bash
CUDA_VISIBLE_DEVICES=[n] bash 22_train.sh [expname]
```
[expname] denotes the name of each experiment (e.g., 'E1.GSA_BiLSTM').

[n] denotes index of a GPU (e.g., '0')

(Only single-GPU experiments had been conducted.

Multi-GPU training is possible by using Horovod within RETURNN training, but not tested yet.)

# Inference

### Beam search
```bash
CUDA_VISIBLE_DEVICES=[n] bash 23_recog.sh [expname] 0
```

### Check WER
```bash
CUDA_VISIBLE_DEVICES=[n] bash 24_report_best_recog.sh [expname]
```


