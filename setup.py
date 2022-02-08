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

import os.path
import pathlib
import shutil

import skbuild
import skbuild.setuptools_wrap
import skbuild.utils
from setuptools import find_packages

import versioneer

# Get long description
with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
    return cmdclass


def cleanup_destination(cmake_manifest):
    """Delete library files from dpctl/ folder before
    letting skbuild copy them over to avoid errors.
    """
    _to_unlink = []
    for fn in cmake_manifest:
        bn = os.path.basename(fn)
        # delete
        if "DPCTLSyclInterface" in bn:
            lib_fn = os.path.join("dpctl", bn)
            if os.path.exists(lib_fn):
                _to_unlink.append(lib_fn)
    for fn in _to_unlink:
        pathlib.Path(fn).unlink()
    return cmake_manifest


def _patched_copy_file(src_file, dest_file, hide_listing=True):
    """Copy ``src_file`` to ``dest_file`` ensuring parent directory exists.

    By default, message like `creating directory /path/to/package` and
    `copying directory /src/path/to/package -> path/to/package` are displayed
    on standard output. Setting ``hide_listing`` to False avoids message from
    being displayed.

    NB: Patched here to not follows symbolic links
    """
    # Create directory if needed
    dest_dir = os.path.dirname(dest_file)
    if dest_dir != "" and not os.path.exists(dest_dir):
        if not hide_listing:
            print("creating directory {}".format(dest_dir))
        skbuild.utils.mkdir_p(dest_dir)

    # Copy file
    if not hide_listing:
        print("copying {} -> {}".format(src_file, dest_file))
    shutil.copyfile(src_file, dest_file, follow_symlinks=False)
    shutil.copymode(src_file, dest_file, follow_symlinks=False)


skbuild.setuptools_wrap._copy_file = _patched_copy_file


skbuild.setup(
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
    package_data={"dpctl": ["tests/*.*", "tests/helper/*.py"]},
    include_package_data=True,
    zip_safe=False,
    setup_requires=["Cython"],
    install_requires=[
        "numpy",
    ],
    extras_require={
        "docs": [
            "Cython",
            "sphinx",
            "sphinx_rtd_theme",
            "pydot",
            "graphviz",
            "sphinxcontrib-programoutput",
        ],
        "coverage": ["Cython", "pytest", "pytest-cov", "coverage", "tomli"],
    },
    keywords="dpctl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    cmake_process_manifest_hook=cleanup_destination,
)
