# Contributing <!-- omit in toc -->

We welcome your contributions!

To contribute, do one of the following:
- create an [issue](https://github.com/IntelPython/dpctl/issues/new)
- participate in [discussions](https://github.com/IntelPython/dpctl/discussions)
- open a [pull request](https://github.com/IntelPython/dpctl/compare) from changes committed to your fork
of `dpctl`

> **NOTE:** Make sure to check the box ``"[x] Allow edits from maintainers"`` to allow the proper functioning
> of automation bots. See [Working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork) for more details.

# Table of Contents <!-- omit in toc -->
- [Mechanical Source Issues](#mechanical-source-issues)
  - [Source Code Formatting](#source-code-formatting)
    - [C/C++ Code Style](#cc-code-style)
    - [Python Code Style](#python-code-style)
    - [Setting Up a Pre-commit](#setting-up-a-pre-commit)
    - [C/C++ File Headers](#cc-file-headers)
    - [Python File Headers](#python-file-headers)
  - [Security](#security)
    - [Bandit](#bandit)
  - [Code Coverage](#code-coverage)
  - [Error Reporting and Logging](#error-reporting-and-logging)
    - [Optional use of the Google logging library (glog)](#optional-use-of-the-google-logging-library-glog)


# Mechanical Source Issues

## Source Code Formatting

### C/C++ Code Style

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for C++ code formatting.

To install, run:
```bash
conda install clang-tools
```

See the default configuration used by dpctl in `.clang-format`.

Before each commit, run:

```bash
clang-format -style=file -i         \
     libsyclinterface/include/*.h         \
     libsyclinterface/include/Support/*.h \
     libsyclinterface/source/*.cpp        \
     libsyclinterface/tests/*.cpp         \
     libsyclinterface/helper/include/*.h  \
     libsyclinterface/helper/source/*.cpp
```

> **NOTE:** It is recommended to use `pre-commit` that invokes `clang-format` among other linters as configured.

### Python Code Style

We use the following Python code style tools:
- [black](https://black.readthedocs.io/en/stable/) code formatter.
- [flake8](https://flake8.pycqa.org/en/latest/) linter.
- [isort](https://pycqa.github.io/isort/) import sorter.

> **NOTE:** Refer:
>  * `pyproject.toml` and `.flake8` config files for current configurations
> * `.pre-commit-config.yaml` file for the versions of the tools

Run these three tools before each commit. 

> **TIP:** Refer your IDE docs to automate these checks or set up `pre-commit` to add git hooks.

### Setting Up a Pre-commit
                                     
A `.pre-commit-config.yaml` is included to run various checks before you
commit your code. 

To setup `pre-commit` in your workflow, install:

- `pre-commit`: `pip install pre-commit`
- the git hook scripts: `pre-commit install`


### C/C++ File Headers

Every C API source file should have a header that describes the file’s basic purpose.
The standard header looks like this:

```
//===----- dpctl_sycl_event_interface.h - C API for sycl::event  -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
---
> **NOTE:**
>- The `-*- C++ -*-` string on the first line tells Emacs* that
  it is a C++ file. The string is only needed for `*.h` headers and
  should be omitted for `*.cpp` files. Without the string, Emacs assumes the
  file is a C header.
>- The copyright year must be updated every calendar year.
>- Each comment line should be a max of 80 chars.
>- A Doxygen `\file` tag describing the contents of the file must be provided.
  Note that the `\file` tag is inside a Doxygen comment block. It is
  defined by the `///` comment marker instead of the `//` comment marker used in the
  rest of the header.

---


### Python File Headers

Every Python and Cython file should only include the following license header:

```
#                      Data Parallel Control (dpctl)
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
```

The copyright year should be updated every calendar year.

## Security

### Bandit

We use [Bandit](https://github.com/PyCQA/bandit) to find common security issues
in the Python code.

To install, run:
```bash
pip install bandit
```

Bandit revision used: `1.7.0`

Before each commit, run:
```bash
bandit -r dpctl -lll
```

## Code Coverage

Code coverage, for both C and Python sources in dpctl, is generated for each
pull request (PR). A PR cannot be merged if it leads to a drop in the code
coverage by more than five percentage points. Therefore, write
unit tests for your changes. 

To check the code coverage for your code, follow these steps:

1. Install dependencies for C/C++ source.

    For C/C++ source,  `lcov`, `llvm-cov` (>=11.0), and `llvm-profdata` (>=11.0) are required. If you
    have multiple `llvm-cov` tools installed, set the `LLVM_TOOLS_HOME`
    environment variable to make sure the correct one is used to generate
    coverage.

2. Install dependencies for Python sources.

    To generate the coverage data for dpctl Python sources,
    install `coverage`:

    ```bash
    python -m pip install coverage[toml]
    ```

3. Build dpctl with code coverage support.

    ```bash
    python scripts/gen_coverage.py --oneapi
    coverage html
    ```

    The code coverage builds the C sources with debug symbols. For this
    reason, the coverage script builds the package in `develop` mode of
    `setup.py`.

    The coverage results for the C and Python sources are printed to the
    terminal during the build (`libsyclinterface`) and pytest execution (Python).
    The detailed coverage reports for the `libsyclinterface` library are saved to the
    `dpctl-c-api-coverage` directory. The Python coverage reports are saved to
    the `htmlcov` directory.

    The coverage data for every PR is also available online at
    [coveralls.io](https://coveralls.io/github/IntelPython/dpctl).

> **_NOTE:_**
> 1. Run `git clean -xdf` to clean up the working tree before running
> a fresh build with code coverage data generation.
> 2. You may encounter the following error when generating coverage data:
>    ```
>    error: '../compat/unistd.h' file not found, did you mean 'compat/unistd.h'?
>    #   include "../compat/unistd.h"
>        ^
>    1 error generated.
>    ```
>    The error is related to the `tcl` package. Uninstall the `tcl`
> package to resolve it.

## Error Reporting and Logging

The `libsyclinterface` library responds to the `DPCTL_VERBOSITY` environment variable that controls the severity level of errors printed to the console.
Specify one of the following severity levels (in increasing order of severity): `warning` and `error` by running:

```bash
export DPCTL_VERBOSITY=warning
```

Messages of a given severity are shown not only in the console for that severity but also for the higher severity. For example, the severity level `warning` outputs severity errors for `error` and `warning` to the console.

### Optional use of the Google logging library (glog)

The dpctl error handler for libsyclinterface can be optionally configured to use [glog](https://github.com/google/glog). 

To use glog, complete the following steps:

1. Install glog package of the latest version

```bash
conda install glog
```
2. Build dpctl with the glog support

```bash
python scripts/build_locally.py --oneapi --glog
```

3. Use the `dpctl._diagnostics.syclinterface_diagnostics(verbosity="warning", log_dir=None)` context manager to switch library diagnostics on for a block of a Python code.
Use `DPCTLService_InitLogger` and `DPCTLService_ShutdownLogger` library C functions during library development to initialize Google's logging library and de-initialize accordingly:

```python
from dpctl._diagnostics import syclinterface_diagnostics
import dpctl

with syclinterface_diagnostics():
    code
```

```c
DPCTLService_InitLogger(const char *app_name, const char *log_dir);
DPCTLService_ShutdownLogger();
```
Where:
 - `*app_name` - the name of the executable file (prefix for logs of various levels).
 - `*log_dir` - a directory path for writing log files. Specifying `NULL` results in logging to ``std::cerr``.

> **_NOTE:_**
> If `InitGoogleLogging` is not called before the first use of the glog, the library self-initializes to the `logtostderr` mode, and log files are not generated.
