#!/bin/bash

source activate
conda activate caribou_hi

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1 $2 $3"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python run_caribou.py $1 $2 $3
rm -rf $tmpdir
