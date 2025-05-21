#!/bin/bash

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1 $2 $3"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python run_caribou.py $1
rm -rf $tmpdir
