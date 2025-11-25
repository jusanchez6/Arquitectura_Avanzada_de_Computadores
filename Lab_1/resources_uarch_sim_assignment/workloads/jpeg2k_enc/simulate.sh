#! /bin/bash

cd "$(dirname $0)"
GEM5PATH=~/mySimTools/gem5/build/ARM
SCRIPTDIR=../../scripts/CortexA76_scripts_gem5

MOREOPTIONS="--l1i_size=32kB --l1d_size=128kB"

$GEM5PATH/gem5.fast $SCRIPTDIR/CortexA76.py --cmd=jpg2k_enc --options="-i jpg2kenc_testfile.bmp -o jpg2kenc_outfile.j2k" $MOREOPTIONS
