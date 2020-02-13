This repository includes a pipeline scripts for training & inference of AED (Attention-based Encoder-Decoder) models for speech recognition task on [LibriSpeech](http://www.openslr.org/12/) dataset.

All the included scripts are largely based on the [RETURNN](https://github.com/rwth-i6/returnn) toolkit.

## Installation
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


## Training
```bash
bash 22_train.sh [expname]
```
[expname] denotes the name of each experiment (e.g., 'E1.GSA_BiLSTM').

## Inference

### Beam search
```bash
bash 23_recog.sh [expname] 0
```

### Check WER
```bash
bash 24_report_best_recog.sh [expname]
```


