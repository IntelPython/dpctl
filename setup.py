#               Data Parallel Control Library (dpCtl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path
import sys
import versioneer
import subprocess

import setuptools.command.install as orig_install
import setuptools.command.develop as orig_develop

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy as np

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
    DPCPP_ROOT = os.environ["ONEAPI_ROOT"] + "/compiler/latest/linux"
    os.environ["DPCTL_SYCL_INTERFACE_LIBDIR"] = "dpctl"
    os.environ["DPCTL_SYCL_INTERFACE_INCLDIR"] = "dpctl/include"
    os.environ["CFLAGS"] = "-fPIC"

elif IS_WIN:
    os.environ["DPCTL_SYCL_INTERFACE_LIBDIR"] = "dpctl"
    os.environ["DPCTL_SYCL_INTERFACE_INCLDIR"] = "dpctl\include"

dpctl_sycl_interface_lib = os.environ["DPCTL_SYCL_INTERFACE_LIBDIR"]
dpctl_sycl_interface_include = os.environ["DPCTL_SYCL_INTERFACE_INCLDIR"]
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
        return ["-O3", "-std=c++17"]
    elif IS_MAC:
        return []
    elif IS_WIN:
        # FIXME: These are specific to MSVC and we should first make sure
        # what compiler we are using.
        return ["/Ox", "/std:c++17"]


def get_suppressed_warning_flags():
    if IS_LIN:
        # PEP 590 renamed "tp_print" to "tp_vectorcall" and this causes a flood
        # of deprecation warnings in the Cython generated module. This flag
        # temporarily suppresses the warnings. The flag should not be needed
        # once we move to Python 3.9 and/or Cython 0.30.
        return ["-Wno-deprecated-declarations"]
    elif IS_WIN:
        return []


def build_backend():
    build_script = os.path.join(os.getcwd(), "scripts", "build_backend.py")
    subprocess.check_call([sys.executable, build_script])


def extensions():
    # Security flags
    eca = get_sdl_cflags()
    ela = get_sdl_ldflags()
    libs = []
    librarys = []
    CODE_COVERAGE = os.environ.get("CODE_COVERAGE")

    if IS_LIN:
        libs += ["rt", "DPCTLSyclInterface"]
    elif IS_MAC:
        pass
    elif IS_WIN:
        libs += ["DPCTLSyclInterface", "sycl"]

    if IS_LIN:
        librarys = [dpctl_sycl_interface_lib]
    elif IS_WIN:
        librarys = [dpctl_sycl_interface_lib, sycl_lib]
    elif IS_MAC:
        librarys = [dpctl_sycl_interface_lib]

    if IS_LIN or IS_MAC:
        runtime_library_dirs = ["$ORIGIN"]
    elif IS_WIN:
        runtime_library_dirs = []

    extension_args = {
        "depends": [
            dpctl_sycl_interface_include,
        ],
        "include_dirs": [np.get_include(), dpctl_sycl_interface_include],
        "extra_compile_args": eca
        + get_other_cxxflags()
        + get_suppressed_warning_flags(),
        "extra_link_args": ela,
        "libraries": libs,
        "library_dirs": librarys,
        "runtime_library_dirs": runtime_library_dirs,
        "language": "c++",
    }

    if CODE_COVERAGE:
        extension_args.update(
            {
                "define_macros": [
                    ("CYTHON_TRACE", "1"),
                ]
            }
        )

    extensions = [
        Extension(
            "dpctl._sycl_context",
            [
                os.path.join("dpctl", "_sycl_context.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl._sycl_device",
            [
                os.path.join("dpctl", "_sycl_device.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl._sycl_device_factory",
            [
                os.path.join("dpctl", "_sycl_device_factory.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl._sycl_event",
            [
                os.path.join("dpctl", "_sycl_event.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl._sycl_queue",
            [
                os.path.join("dpctl", "_sycl_queue.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl._sycl_queue_manager",
            [
                os.path.join("dpctl", "_sycl_queue_manager.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl.memory._memory",
            [
                os.path.join("dpctl", "memory", "_memory.pyx"),
            ],
            **extension_args
        ),
        Extension(
            "dpctl.program._program",
            [
                os.path.join("dpctl", "program", "_program.pyx"),
            ],
            **extension_args
        ),
    ]
    if CODE_COVERAGE:
        exts = cythonize(extensions, compiler_directives={"linetrace": True})
    else:
        exts = cythonize(extensions)
    return exts


class install(orig_install.install):
    def run(self):
        build_backend()
        return super().run()


class develop(orig_develop.develop):
    def run(self):
        build_backend()
        return super().run()


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
    cmdclass["install"] = install
    cmdclass["develop"] = develop
    return cmdclass


setup(
    name="dpctl",
    version=versioneer.get_version(),
    cmdclass=_get_cmdclass(),
    description="A lightweight Python wrapper for a subset of OpenCL and SYCL.",
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpCtl",
    packages=find_packages(include=["*"]),
    include_package_data=True,
    ext_modules=extensions(),
    zip_safe=False,
    setup_requires=["Cython"],
    keywords="dpctl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
