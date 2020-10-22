import os
import subprocess

if os.environ["ONEAPI_ROOT"] != None:
    ONEAPI_ROOT = os.environ["ONEAPI_ROOT"]
    os.system("source" + " " + ONEAPI_ROOT + "/compiler/latest/env/vars.sh")
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
else:
    print("DPCPP is needed to build DPPL. Abort!")

build_cmake_dir = "build_cmake"
path = os.path.join(os.getcwd(), build_cmake_dir)
if os.path.exists(path):
    os.rmdir(path)
os.mkdir(path)

INSTALL_PREFIX = path + "/../install"
if os.path.exists(INSTALL_PREFIX):
    os.rmdir(INSTALL_PREFIX)

import distutils.sysconfig
import numpy

PYTHON_INC = distutils.sysconfig.get_python_inc()
NUMPY_INC = numpy.get_include()
DPCPP_ROOT = ONEAPI_ROOT + "/compiler/latest/linux/"

# subprocess.check_call(
#     [
#         "cmake",
#         "-DCMAKE_BUILD_TYPE=Release",
#         "-DCMAKE_INSTALL_PREFIX=" + INSTALL_PREFIX,
#         "-DCMAKE_PREFIX_PATH=" + INSTALL_PREFIX,
#         "-DDPCPP_ROOT=" + DPCPP_ROOT,
#         "-DPYTHON_INCLUDE_DIR=" + PYTHON_INC,
#         "-DNUMPY_INCLUDE_DIR=" + NUMPY_INC,
#         "/backends",
#     ]
# )

os.system(
    "cmake "
    + "-DCMAKE_BUILD_TYPE=Release "
    + "-DCMAKE_INSTALL_PREFIX="
    + INSTALL_PREFIX
    + " "
    + "-DCMAKE_PREFIX_PATH="
    + INSTALL_PREFIX
    + " "
    + "-DDPCPP_ROOT="
    + DPCPP_ROOT
    + " "
    + "-DPYTHON_INCLUDE_DIR="
    + PYTHON_INC
    + " "
    + "-DNUMPY_INCLUDE_DIR="
    + NUMPY_INC
    + " "
    + os.getcwd()
    + "/backends",
)

# subprocess.check_call(["make", "-j", "4"])
# subprocess.check_call(["make", "install"])
os.system("make -j 4 && make install")

os.system("cp " + os.getcwd() + "/install/lib/*.so" + os.getcwd() + "/dpctl/")

if not os.path.exists(os.getcwd() + "/dpctl/include"):
    os.mkdir(os.getcwd() + "/dpctl/include")
os.system(
    "cp -r " + os.getcwd() + "/backends/include/* " + os.getcwd() + "/dpctl/include"
)

os.environ["DPPL_OPENCL_INTERFACE_LIBDIR"] = "dpctl"
os.environ["DPPL_OPENCL_INTERFACE_INCLDIR"] = "dpctl/include"
os.environ["OpenCL_LIBDIR"] = DPCPP_ROOT + "/lib"

os.environ["DPPL_SYCL_INTERFACE_LIBDIR"] = "dpctl"
os.environ["DPPL_SYCL_INTERFACE_INCLDIR"] = "dpctl/include"

if os.environ["CFLAGS"] == None:
    os.environ["CFLAGS"] = " "
# os.environ["CFLAGS"] = "-fPIC -O3" + os.environ["CFLAGS"]
os.environ["CFLAGS"] = "-fPIC -O3"
os.environ["LDFLAGS"] = (
    "-L " + os.environ["OpenCL_LIBDIR"] + " " + os.environ["LDFLAGS"]
)

subprocess.check_call([os.environ["PYTHON"] + "setup.py" + "clean" + "--all"])
subprocess.check_call([os.environ["PYTHON"] + "setup.py" + "build" + "install"])
