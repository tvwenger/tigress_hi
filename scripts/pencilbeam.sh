#!/bin/bash

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $idx"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python pencilbeam.py $1
rm -rf $tmpdir
