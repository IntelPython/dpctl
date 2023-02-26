#!/usr/bin/bash

local_vendors=$PREFIX/etc/OpenCL/vendors/
icd_fn=$local_vendors/intel-ocl-gpu.icd

if [[ -L $icd_fn ]]; then
    rm $icd_fn
fi
