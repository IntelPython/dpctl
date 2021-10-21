#               Data Parallel Control Library (dpctl)
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

import glob
import os
import os.path
import shutil
import sys

import numpy as np
import setuptools.command.build_ext as orig_build_ext
import setuptools.command.develop as orig_develop
import setuptools.command.install as orig_install
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import versioneer

IS_WIN = False
IS_LIN = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True
else:
    assert False, "We currently do not build for " + sys.platform

# global variable used to pass value of --coverage option of develop command
# to build_ext command
_coverage = False
dpctl_sycl_interface_lib = "dpctl"
dpctl_sycl_interface_include = r"dpctl/include"

# Get long description
with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


def remove_empty(li):
    return [el for el in li if el]


def get_sdl_cflags():
    cflags = []
    if IS_LIN:
        cflags = [
            "-fstack-protector",
            "-fPIC",
            "-D_FORTIFY_SOURCE=2",
            "-Wformat",
            "-Wformat-security",
        ]
    # Add cflags from environment
    cflags += remove_empty(os.getenv("CFLAGS", "").split(" "))

    return cflags


def get_sdl_ldflags():
    ldflags = []
    if IS_LIN:
        ldflags = ["-Wl,-z,noexecstack,-z,relro,-z,now"]
    elif IS_WIN:
        ldflags = [r"/NXCompat", r"/DynamicBase"]
    # Add ldflags from environment
    ldflags += remove_empty(os.getenv("LDFLAGS", "").split(" "))

    return ldflags


def get_other_cxxflags():
    if IS_LIN:
        return ["-O3", "-std=c++17"]
    elif IS_WIN:
        # FIXME: These are specific to MSVC and we should first make sure
        # what compiler we are using.
        return [r"/Ox", r"/std:c++17"]


def get_suppressed_warning_flags():
    if IS_LIN:
        # PEP 590 renamed "tp_print" to "tp_vectorcall" and this causes a flood
        # of deprecation warnings in the Cython generated module. This flag
        # temporarily suppresses the warnings. The flag should not be needed
        # once we move to Python 3.9 and/or Cython 0.30.
        return ["-Wno-deprecated-declarations"]
    elif IS_WIN:
        return []


def build_backend(l0_support, coverage, sycl_compiler_prefix):
    import os.path
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "build_backend", os.path.join("scripts", "build_backend.py")
    )
    builder_module = module_from_spec(spec)
    spec.loader.exec_module(builder_module)
    builder_module.build_backend(
        l0_support=l0_support,
        code_coverage=coverage,
        sycl_compiler_prefix=sycl_compiler_prefix,
    )


def extensions():
    # Security flags
    eca = get_sdl_cflags()
    ela = get_sdl_ldflags()
    libs = []
    libraries = []

    if IS_LIN:
        libs += ["rt", "DPCTLSyclInterface"]
        libraries = [dpctl_sycl_interface_lib]
        runtime_library_dirs = ["$ORIGIN"]
    elif IS_WIN:
        libs += ["DPCTLSyclInterface"]
        libraries = [dpctl_sycl_interface_lib]
        runtime_library_dirs = []

    extension_args = {
        "depends": [
            dpctl_sycl_interface_include,
        ],
        "include_dirs": [np.get_include(), dpctl_sycl_interface_include],
        "extra_compile_args": (
            eca + get_other_cxxflags() + get_suppressed_warning_flags()
        ),
        "extra_link_args": ela,
        "libraries": libs,
        "library_dirs": libraries,
        "runtime_library_dirs": runtime_library_dirs,
        "language": "c++",
        "define_macros": [],
    }

    extensions = [
        Extension(
            "dpctl._sycl_context",
            [
                os.path.join("dpctl", "_sycl_context.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl._sycl_device",
            [
                os.path.join("dpctl", "_sycl_device.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl._sycl_device_factory",
            [
                os.path.join("dpctl", "_sycl_device_factory.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl._sycl_event",
            [
                os.path.join("dpctl", "_sycl_event.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl._sycl_platform",
            [
                os.path.join("dpctl", "_sycl_platform.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl._sycl_queue",
            [
                os.path.join("dpctl", "_sycl_queue.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl._sycl_queue_manager",
            [
                os.path.join("dpctl", "_sycl_queue_manager.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl.memory._memory",
            [
                os.path.join("dpctl", "memory", "_memory.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl.program._program",
            [
                os.path.join("dpctl", "program", "_program.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl.utils._compute_follows_data",
            [
                os.path.join("dpctl", "utils", "_compute_follows_data.pyx"),
            ],
            **extension_args,
        ),
        Extension(
            "dpctl.tensor._usmarray",
            [
                os.path.join("dpctl", "tensor", "_usmarray.pyx"),
            ],
            depends=extension_args["depends"]
            + [os.path.join("libtensor", "include", "usm_array.hpp")],
            language="c++",
            include_dirs=(
                extension_args["include_dirs"]
                + [os.path.join("libtensor", "include")]
            ),
            extra_compile_args=extension_args["extra_compile_args"],
            extra_link_args=extension_args["extra_link_args"],
            libraries=extension_args["libraries"],
            library_dirs=extension_args["library_dirs"],
            runtime_library_dirs=extension_args["runtime_library_dirs"],
            define_macros=extension_args["define_macros"],
        ),
    ]
    return extensions


class build_ext(orig_build_ext.build_ext):
    description = "Build dpctl native extensions"

    def finalize_options(self):
        if _coverage:
            pre_d = getattr(self, "define", None)
            if pre_d is None:
                self.define = "CYTHON_TRACE"
            else:
                self.define = ",".join((pre_d, "CYTHON_TRACE"))
        super().finalize_options()

    def run(self):
        return super().run()


def get_build_py(orig_build_py):
    class build_py(orig_build_py):
        def run(self):
            dpctl_src_dir = self.get_package_dir("dpctl")
            dpctl_build_dir = os.path.join(self.build_lib, "dpctl")
            os.makedirs(dpctl_build_dir, exist_ok=True)
            if IS_LIN:
                for fn in glob.glob(os.path.join(dpctl_src_dir, "*.so*")):
                    # Check if the file already exists before copying.
                    # The check is needed when dealing with symlinks.
                    if not os.path.exists(
                        os.path.join(dpctl_build_dir, os.path.basename(fn))
                    ):
                        shutil.copy(
                            src=fn,
                            dst=dpctl_build_dir,
                            follow_symlinks=False,
                        )
            elif IS_WIN:
                for fn in glob.glob(os.path.join(dpctl_src_dir, "*.lib")):
                    shutil.copy(src=fn, dst=dpctl_build_dir)

                for fn in glob.glob(os.path.join(dpctl_src_dir, "*.dll")):
                    shutil.copy(src=fn, dst=dpctl_build_dir)
            else:
                raise NotImplementedError("Unsupported platform")
            return super().run()

    return build_py


class install(orig_install.install):
    description = "Installs dpctl into Python prefix"
    user_options = orig_install.install.user_options + [
        (
            "level-zero-support=",
            None,
            "Whether to enable support for program creation "
            "for Level-zero backend",
        ),
        (
            "sycl-compiler-prefix=",
            None,
            "Path to SYCL compiler installation. None means "
            "read it off ONEAPI_ROOT environment variable or fail.",
        ),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.level_zero_support = "True"
        self.sycl_compiler_prefix = None

    def finalize_options(self):
        if isinstance(self.level_zero_support, str):
            self.level_zero_support = self.level_zero_support.capitalize()
        if self.level_zero_support in ["True", "False", "0", "1"]:
            self.level_zero_support = bool(eval(self.level_zero_support))
        else:
            raise ValueError(
                "--level-zero-support value is invalid, use True/False"
            )
        if isinstance(self.sycl_compiler_prefix, str):
            if not os.path.exists(os.path.join(self.sycl_compiler_prefix)):
                raise ValueError(
                    "--sycl-compiler-prefix expects a path "
                    "to an existing directory"
                )
        elif self.sycl_compiler_prefix is None:
            pass
        else:
            raise ValueError(
                "--sycl-compiler-prefix value is invalid, use a "
                "path to compiler installation. To use oneAPI, use the "
                "default value, but remember to activate the compiler "
                "environment"
            )
        super().finalize_options()

    def run(self):
        build_backend(self.level_zero_support, False, self.sycl_compiler_prefix)
        if _coverage:
            pre_d = getattr(self, "define", None)
            if pre_d is None:
                self.define = "CYTHON_TRACE"
            else:
                self.define = ",".join((pre_d, "CYTHON_TRACE"))
        cythonize(self.distribution.ext_modules)
        ret = super().run()
        if IS_LIN:
            dpctl_build_dir = os.path.join(
                os.path.dirname(__file__), self.build_lib, "dpctl"
            )
            dpctl_install_dir = os.path.join(self.install_libbase, "dpctl")
            for fn in glob.glob(
                os.path.join(dpctl_install_dir, "*DPCTLSyclInterface.so*")
            ):
                os.remove(fn)
                shutil.copy(
                    src=os.path.join(dpctl_build_dir, os.path.basename(fn)),
                    dst=dpctl_install_dir,
                    follow_symlinks=False,
                )
        return ret


class develop(orig_develop.develop):
    description = "Installs dpctl in place"
    user_options = orig_develop.develop.user_options + [
        (
            "level-zero-support=",
            None,
            "Whether to enable support for program creation "
            "for Level-zero backend",
        ),
        (
            "sycl-compiler-prefix=",
            None,
            "Path to SYCL compiler installation. None means "
            "read it off ONEAPI_ROOT environment variable or fail.",
        ),
        (
            "coverage=",
            None,
            "Whether to generate coverage report "
            "when building the backend library",
        ),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.level_zero_support = "True"
        self.coverage = "False"
        self.sycl_compiler_prefix = None

    def finalize_options(self):
        if isinstance(self.level_zero_support, str):
            self.level_zero_support = self.level_zero_support.capitalize()
        if self.level_zero_support in ["True", "False", "0", "1"]:
            self.level_zero_support = bool(eval(self.level_zero_support))
        else:
            raise ValueError(
                "--level-zero-support value is invalid, use True/False"
            )
        if isinstance(self.coverage, str):
            self.coverage = self.coverage.capitalize()
        if self.coverage in ["True", "False", "0", "1"]:
            self.coverage = bool(eval(self.coverage))
            global _coverage
            _coverage = self.coverage
        else:
            raise ValueError("--coverage value is invalid, use True/False")
        if isinstance(self.sycl_compiler_prefix, str):
            if not os.path.exists(os.path.join(self.sycl_compiler_prefix)):
                raise ValueError(
                    "--sycl-compiler-prefix expects a path "
                    "to an existing directory"
                )
        elif self.sycl_compiler_prefix is None:
            pass
        else:
            raise ValueError(
                "--sycl-compiler-prefix value is invalid, use a "
                "path to compiler installation. To use oneAPI, use the "
                "default value, but remember to activate the compiler "
                "environment"
            )
        super().finalize_options()

    def run(self):
        build_backend(
            self.level_zero_support, self.coverage, self.sycl_compiler_prefix
        )
        if _coverage:
            pre_d = getattr(self, "define", None)
            if pre_d is None:
                self.define = "CYTHON_TRACE"
            else:
                self.define = ",".join((pre_d, "CYTHON_TRACE"))
        cythonize(self.distribution.ext_modules)
        return super().run()


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
    cmdclass["build_py"] = get_build_py(cmdclass["build_py"])
    cmdclass["install"] = install
    cmdclass["develop"] = develop
    cmdclass["build_ext"] = build_ext
    return cmdclass


setup(
    name="dpctl",
    version=versioneer.get_version(),
    cmdclass=_get_cmdclass(),
    description="A lightweight Python wrapper for a subset of SYCL.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpctl",
    packages=find_packages(include=["*"]),
    include_package_data=True,
    ext_modules=extensions(),
    zip_safe=False,
    setup_requires=["Cython"],
    install_requires=[
        "numpy",
    ],
    keywords="dpctl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
