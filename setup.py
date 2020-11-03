##===---------- setup.py - dpctl.ocldrv interface -----*- Python -*-----===##
##
##               Data Parallel Control Library (dpCtl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
###
### \file
### This file builds the dpctl and dpctl.ocldrv extension modules.
##===----------------------------------------------------------------------===##
import os
import os.path
import sys
import versioneer
import subprocess

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy as np

requirements = [
    "cffi>=1.0.0",
    "cython",
]

IS_WIN = False
IS_MAC = False
IS_LIN = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform == "darwin":
    IS_MAC = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True
else:
    assert False, sys.platform + " not supported"

if IS_LIN:
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    os.environ["DPCPP_ROOT"] = os.environ["ONEAPI_ROOT"] + "/compiler/latest/linux/"
    os.environ["DPPL_OPENCL_INTERFACE_LIBDIR"] = "dpctl"
    os.environ["DPPL_OPENCL_INTERFACE_INCLDIR"] = "dpctl/include"
    os.environ["OpenCL_LIBDIR"] = os.environ["DPCPP_ROOT"] + "/lib"
    os.environ["DPPL_SYCL_INTERFACE_LIBDIR"] = "dpctl"
    os.environ["DPPL_SYCL_INTERFACE_INCLDIR"] = "dpctl/include"

elif IS_WIN:
    os.environ["CC"] = "clang-cl.exe"
    os.environ["CXX"] = "dpcpp.exe"
    os.environ["DPCPP_ROOT"] = "%ONEAPI_ROOT%\compiler\latest\windows"
    os.environ["DPPL_OPENCL_INTERFACE_LIBDIR"] = "dpctl"
    os.environ["DPPL_OPENCL_INTERFACE_INCLDIR"] = "dpctl\include"
    os.environ["OpenCL_LIBDIR"] = os.environ["DPCPP_ROOT"] + "\lib"
    os.environ["DPPL_SYCL_INTERFACE_LIBDIR"] = "dpctl"
    os.environ["DPPL_SYCL_INTERFACE_INCLDIR"] = "dpctl\include"

dppl_sycl_interface_lib = os.environ["DPPL_SYCL_INTERFACE_LIBDIR"]
dppl_sycl_interface_include = os.environ["DPPL_SYCL_INTERFACE_INCLDIR"]
sycl_lib = os.environ["ONEAPI_ROOT"] + "\compiler\latest\windows\lib"


def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        return [
            "-fstack-protector",
            "-fPIC",
            "-D_FORTIFY_SOURCE=2",
            "-Wformat",
            "-Wformat-security",
        ]
    elif IS_WIN:
        return []


def get_sdl_ldflags():
    if IS_LIN:
        return [
            "-Wl,-z,noexecstack,-z,relro,-z,now",
        ]
    elif IS_MAC:
        return []
    elif IS_WIN:
        return ["/NXCompat", "/DynamicBase"]


def get_other_cxxflags():
    if IS_LIN:
        return ["-O3", "-fsycl", "-std=c++17"]
    elif IS_MAC:
        return []
    elif IS_WIN:
        # FIXME: These are specific to MSVC and we should first make sure
        # what compiler we are using.
        return ["/Ox", "/std:c++17"]


def build_backend():
    if IS_LIN:
        subprocess.check_call(["/bin/bash", "-c", "scripts/build_backend.sh"])
    elif IS_WIN:
        subprocess.check_call(["cmd.exe", "/c", "scripts\\build_backend.bat"])


def extensions():
    build_backend()
    # Security flags
    eca = get_sdl_cflags()
    ela = get_sdl_ldflags()
    libs = []
    librarys = []

    if IS_LIN:
        libs += ["rt", "DPPLSyclInterface"]
    elif IS_MAC:
        pass
    elif IS_WIN:
        libs += ["DPPLSyclInterface", "sycl"]

    if IS_LIN:
        librarys = [dppl_sycl_interface_lib]
    elif IS_WIN:
        librarys = [dppl_sycl_interface_lib, sycl_lib]
    elif IS_MAC:
        librarys = [dppl_sycl_interface_lib]

    if IS_LIN or IS_MAC:
        runtime_library_dirs = ["$ORIGIN"]
    elif IS_WIN:
        runtime_library_dirs = []

    extension_args = {
        "depends": [
            dppl_sycl_interface_include,
        ],
        "include_dirs": [np.get_include(), dppl_sycl_interface_include],
        "extra_compile_args": eca + get_other_cxxflags(),
        "extra_link_args": ela,
        "libraries": libs,
        "library_dirs": librarys,
        "runtime_library_dirs": runtime_library_dirs,
        "language": "c++",
    }

    extensions = [
        Extension(
            "dpctl._sycl_core",
            [
                os.path.join("dpctl", "sycl_core.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl._memory",
            [
                os.path.join("dpctl", "_memory.pyx"),
            ],
            **extension_args
        ),
    ]

    exts = cythonize(extensions)
    return exts


setup(
    name="dpctl",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A lightweight Python wrapper for a subset of OpenCL and SYCL.",
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpCtl",
    packages=find_packages(include=["*"]),
    include_package_data=True,
    ext_modules=extensions(),
    setup_requires=requirements,
    cffi_modules=["./dpctl/opencl_core.py:ffi"],
    install_requires=requirements,
    keywords="dpctl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
