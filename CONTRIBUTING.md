# Mechanical Source Issues

## Source Code Formatting

### C/C++ code style

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) code formatter.

Install: `conda install clang-tools`

- Revision: `10.0.1`
- See the default configuration used by dpCtl in `.clang-format`.

Run before each commit: `clang-format -style=file -i dpctl-capi/include/*.h dpctl-capi/include/Support/*.h dpctl-capi/source/*.cpp dpctl-capi/tests/*.cpp dpctl-capi/helper/include/*.h dpctl-capi/helper/source/*.cpp`

### Python code style

We use [black](https://black.readthedocs.io/en/stable/) code formatter.

- Revision: `20.8b1` or branch `stable`.
- See configuration in `pyproject.toml`.

Run before each commit: `black .`

### C/C++ File Headers

Every C API source file should have a header on it that describes the basic
purpose of the file. The standard header looks like this:

```
//===----- dpctl_sycl_event_interface.h - C API for sycl::event  -*-C++-*- ===//
//
//                      Data Parallel Control (dpCtl)
//
// Copyright 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header declares a C API to a sub-set of the sycl::event interface.
///
//===----------------------------------------------------------------------===//
```
Few things to note about this format:
- The `-*- C++ -*-` string on the first line is needed to tell Emacs that
  the file is a C++ file. The string is only needed for `*.h` headers and
  should be omitted for `*.cpp` files. Without the string Emacs assumes that
  file is a C header.
- The copyright year should be updated every calendar year.
- Each comment line should be a max of 80 chars.
- A Doxygen `\file` tag describing the contents of the file must be provided.
  Also note that the `\file` tag is inside a Doxygen comment block (defined by `///`
  comment marker instead of the `//` comment marker used in the rest of the header.

### Python File Headers

Every Python and Cython file should only include the following license header:

```
#                      Data Parallel Control (dpCtl)
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
```
The copyright year should be updated every calendar year.

## Security

### Bandit

We use [Bandit](https://github.com/PyCQA/bandit) to find common security issues in Python code.

Install: `pip install bandit`

- Revision: `1.7.0`

Run before each commit: `bandit -r dpctl -lll`

## Code Coverage

Implement python, cython and c++ file coverage using `coverage` and `llvm-cov` packages on Linux.

### Using Code Coverage

You need to install additional packages and add an environment variable to cover:
- conda install cmake
- conda install coverage
- conda install conda-forge::lcov
- conda install conda-forge::gtest
- export CODE_COVERAGE=ON

CODE_COVERAGE allows you to build a debug version of dpctl and enable string tracing, which allows you to analyze strings to create a coverage report.
It was added for the convenience of configuring the CI in the future.

Installing the dpctl package:
- python setup.py develop

It is important that there are no files of the old build in the folder.
Use `git clean -xdf` to clean up the working tree.

The coverage report will appear during the build in the console. This report contains information about c++ file coverage.
The `dpctl-c-api-coverage` folder will appear in the root folder after installation.
The folder contains a report on the coverage of c++ files in html format.

You need to run tests to cover the cython and python files:
- coverage run -m unittest dpctl.tests

The required flags for the command coverage run are contained in the file `.coveragerc`.

The simplest reporting is a textual summary produced with report:
- coverage report

For each module executed, the report shows the count of executable statements, the number of those statements missed, and the resulting coverage, expressed as a percentage.

The `-m` flag also shows the line numbers of missing statements:
- coverage report -m

To create an annotated HTML listings with coverage results:
- coverage html

The `htmlcov` folder will appear in the root folder of the project. It contains reports on the coverage of python and cython files in html format.

Erase previously collected coverage data:
- coverage erase

### Error in the build process

An error occurs during the dcptl build with the CODE_COVERAGE environment variable:
```
error: '../compat/unistd.h' file not found, did you mean 'compat/unistd.h'?
#   include "../compat/unistd.h"
            ^
1 error generated.
```
The error is related to the `tcl` package.
You need to remove the tcl package to resolve this error.
