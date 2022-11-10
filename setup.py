#               Data Parallel Control Library (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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
import os.path
import pathlib
import shutil
import sys

import skbuild
import skbuild.setuptools_wrap
import skbuild.utils
from skbuild.command.build_py import build_py as _skbuild_build_py
from skbuild.command.install import install as _skbuild_install

import versioneer

# Get long description
with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


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


def _patched_copy_file(
    src_file, dest_file, hide_listing=True, preserve_mode=True
):
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


class BuildPyCmd(_skbuild_build_py):
    def copy_file(self, src, dst, preserve_mode=True):
        _patched_copy_file(src, dst, preserve_mode=preserve_mode)
        return (dst, 1)

    def build_package_data(self):
        """Copy data files into build directory"""
        for package, src_dir, build_dir, filenames in self.data_files:
            for filename in filenames:
                target = os.path.join(build_dir, filename)
                self.mkpath(os.path.dirname(target))
                srcfile = os.path.join(src_dir, filename)
                outf, copied = self.copy_file(srcfile, target)
                srcfile = os.path.abspath(srcfile)


class InstallCmd(_skbuild_install):
    def run(self):
        ret = super().run()
        if "linux" in sys.platform:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            dpctl_build_dir = os.path.join(this_dir, self.build_lib, "dpctl")
            dpctl_install_dir = os.path.join(self.install_libbase, "dpctl")
            sofiles = glob.glob(
                os.path.join(dpctl_build_dir, "*DPCTLSyclInterface.so*")
            )
            # insert actual file at the beginning of the list
            pos = [i for i, fn in enumerate(sofiles) if not os.path.islink(fn)]
            if pos:
                hard_file = sofiles.pop(pos[0])
                sofiles.insert(0, hard_file)
            for fn in sofiles:
                base_fn = os.path.basename(fn)
                src_file = os.path.join(dpctl_build_dir, base_fn)
                dst_file = os.path.join(dpctl_install_dir, base_fn)
                os.remove(dst_file)
                _patched_copy_file(src_file, dst_file)
        return ret


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass(
        cmdclass={
            "build_py": BuildPyCmd,
            "install": InstallCmd,
        }
    )
    return cmdclass


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
    packages=[
        "dpctl",
        "dpctl.memory",
        "dpctl.tensor",
        "dpctl.program",
        "dpctl.utils",
    ],
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
