#!/usr/bin/env bash

#OLD_DIR=$(pwd)
cd results/datasets

# if not having this, I will end up creating a link inside the existing folder/symbolic link, and can be dangerous.
if [ ! -e TIMITcorpus ]; then
    ln -s /data2/leelab/standard_datasets/LDC/LDC93S1/TIMITcorpus TIMITcorpus
fi
