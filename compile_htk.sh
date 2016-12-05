#!/usr/bin/env bash

#OLD_DIR=$(pwd)
cd 3rdparty
tar -zxvf HTK-3.5.beta-2.tar.gz
cd htk
cd HTKLib && make -f MakefileCPU all && cd ..
cd HLMLib && make -f MakefileCPU all && cd ..
cd HTKTools && make -f MakefileCPU all && cd ..
cd HLMTools && make -f MakefileCPU all && cd ..
# install
cd HTKTools && make -f MakefileCPU install && cd ..
cd HLMTools && make -f MakefileCPU install && cd ..
