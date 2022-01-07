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

from setuptools import find_packages
from skbuild import setup

import versioneer

# Get long description
with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
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
    package_data={"dpctl": ["tests/*.*", "tests/helper/*.py"]},
    include_package_data=True,
    zip_safe=False,
    setup_requires=["Cython"],
    install_requires=[
        "numpy",
    ],
    keywords="dpctl",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
