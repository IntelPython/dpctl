A minimal Python module exposing a subset of OpenCL and SYCL functionalities.


To install:
   0) Change to the "sycllite" branch

   1) Modify the build_for_develop scipt to change these two variables
   
      ONEAPI_ROOT=<your-oneapi-install-directory>
      OpenCL_LIBDIR=<your-opencl-library-directory>

   2) run `./build_for_develop`

      Note: the build_for_conda script is broken.

   3) Add the directory the following directory created by the build script
      to your LD_LIBRARY_PATH. (Temporary step till the build_for_conda is fixed)

      export LD_LIBRARY_PATH=<dppy-root-dir>/build/install/lib/:${LD_LIBRARY_PATH}

To check if the install worked:
   Run dppy_example1.py under examples.
