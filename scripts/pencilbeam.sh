#!/bin/bash

source activate
conda activate caribou_hi

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $idx"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python pencilbeam.py $1
rm -rf $tmpdir
