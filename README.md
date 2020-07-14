What?
====
A lightweight Python package exposing a subset of OpenCL and SYCL 
functionalities.


How to install?
===
1. Install oneAPI and OpenCL HD graphics drivers.
2. Install conda or minicconda (you can use the conda that comes with oneAPI).
3. [Optional] Create and activate a conda environment:

    `bash conda env create -n dppl-env -f environment.yml`
    
    `conda activate dppl-env`
4. run `./build_for_conda.sh`

Examples:
===
   Run create_sycl_queues.py under examples.

   `python examples/create_sycl_queues.py`
