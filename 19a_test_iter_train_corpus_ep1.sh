#!/bin/bash

set -exv

# Running this is not mandatory.
# Or you can also break it in the middle.
# This is just for testing, whether the dataset was prepared correctly.
# This will use the training config and iterate through epoch 1.
# For interactive terminals, you should see some progress bar like:
# 353/822, 0:00:04 [|||....]

#./returnn/tools/dump-dataset.py returnn.config --endseq -1 --type null
./returnn/tools/dump-dataset.py E1.GSA_BiLSTM.config --endseq -1 --type null
