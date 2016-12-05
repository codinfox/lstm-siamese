#!/usr/bin/env bash

#OLD_DIR=$(pwd)
cd 3rdparty
tar -zxvf sox-14.4.2.tar.gz
cd sox-14.4.2
./configure
make -s

