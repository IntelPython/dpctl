#!/usr/bin/bash

systemwide_icd=/etc/OpenCL/vendors/intel.icd
local_vendors=$PREFIX/etc/OpenCL/vendors/
icd_fn=$local_vendors/intel-ocl-gpu.icd

if [[ -f $systemwide_icd && -d $local_vendors && ! -f $icd_fn ]]; then
    ln -s $systemwide_icd $icd_fn
fi
