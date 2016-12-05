#!/usr/bin/env bash
# -T 1 is for trace level (debugging)
# -A is for verbosity
# -D is also for verbosity
# check Section 4.4 of HTK book.
../3rdparty/htk/bin.cpu/HCopy -A -D -T 1 -C mfcc_config_debug -S hcopy_script.scp
