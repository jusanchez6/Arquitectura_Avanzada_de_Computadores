#! /bin/bash

cd "$(dirname $0)"
GEM5PATH=~/mySimTools/gem5/build/ARM
SCRIPTDIR=../../scripts/CortexA76_scripts_gem5

MOREOPTIONS="--l1i_size=32kB --l1d_size=128kB"

$GEM5PATH/gem5.fast $SCRIPTDIR/CortexA76.py --cmd=h264_dec --options="h264dec_testfile.264 h264dec_outfile.yuv" $MOREOPTIONS
