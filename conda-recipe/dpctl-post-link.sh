#!/usr/bin/bash

systemwide_icd=/etc/OpenCL/vendors/intel.icd
local_vendors=$CONDA_PREFIX/etc/OpenCL/vendors/

if [[ -f $systemwide_icd && -d $local_vendors && ! -f $local_vendors/intl-ocl-gpu.icd ]]; then
    ln -s $systemwide_icd  $local_vendors/intel-ocl-gpu.icd
fi
