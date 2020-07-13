What?
====
A minimal Python module exposing a subset of OpenCL and SYCL functionalities.

How to install?
===
   1. Change to the "sycllite" branch

   1. Modify the build_for_develop scipt to change these two variables
      ```
      ONEAPI_ROOT=<your-oneapi-install-directory>
      OpenCL_LIBDIR=<your-opencl-library-directory>
      ```

   1. run `./build_for_develop`

      Note: the build_for_conda script is broken.

   1. Add the directory the following directory created by the build script
      to your LD_LIBRARY_PATH. (Temporary step till the build_for_conda is fixed)

      `export LD_LIBRARY_PATH=<dppl-root-dir>/build/install/lib/:${LD_LIBRARY_PATH}`

Examples:
===
   Run create_sycl_queues.py under examples.

   `python examples/create_sycl_queues.py`

Development
===========

Install `conda`.
Create and activate build environment:
```bash
conda env create -n dppl-env -f environment.yml
conda activate dppl-env
```
Run `build_for_conda.sh`.
