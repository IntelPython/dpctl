# Mechanical Source Issues

## Source Code Formatting

### C/C++ code style

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) code formatter.

Install: `conda install clang-tools`

- Revision: `10.0.1`
- See the default configuration used by dpctl in `.clang-format`.

Run before each commit:

```bash
clang-format -style=file -i         \
     libsyclinterface/include/*.h         \
     libsyclinterface/include/Support/*.h \
     libsyclinterface/source/*.cpp        \
     libsyclinterface/tests/*.cpp         \
     libsyclinterface/helper/include/*.h  \
     libsyclinterface/helper/source/*.cpp
```

> **_NOTE:_** A much simpler option is to use `pre-commit` and the
> `clang-format` hook that we provide.

### Python code style

We use the following Python code style tools:
- [black](https://black.readthedocs.io/en/stable/) code formatter.
    - Revision: `20.8b1`.
- [flake8](https://flake8.pycqa.org/en/latest/) linter.
    - Revision `3.9.1`.
- [isort](https://pycqa.github.io/isort/) import sorter.
    - Revision `5.8.0`.

- Refer `pyproject.toml` and `.flake8` config files for current configurations.

Please run these three tools before each commit. Although, you may choose to
do so manually, but it is much easier and preferable to automate these checks.
Refer your IDE docs to set them up in your IDE, or you can set up `pre-commit`
to add git hooks.

### Setting up pre-commit

A `.pre-commit-config.yaml` is included to run various check before you
commit your code. Here are the steps to setup `pre-commit` in your workflow:

- Install `pre-commit`: `pip install pre-commit`
- Install the git hook scripts: `pre-commit install`

That should be it!

### C/C++ File Headers

Every C API source file should have a header on it that describes the basic
purpose of the file. The standard header looks like this:

```
//===----- dpctl_sycl_event_interface.h - C API for sycl::event  -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
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
  Also note that the `\file` tag is inside a Doxygen comment block (
  defined by `///` comment marker instead of the `//` comment marker used in the
  rest of the header.

### Python File Headers

Every Python and Cython file should only include the following license header:

```
#                      Data Parallel Control (dpctl)
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

We use [Bandit](https://github.com/PyCQA/bandit) to find common security issues
in Python code.

Install: `pip install bandit`

- Revision: `1.7.0`

Run before each commit: `bandit -r dpctl -lll`

## Code Coverage

Code coverage for both C and Python sources in dpctl is generated for each
pull request (PR). A PR cannot be merged if it leads to a drop in the code
coverage by more than five percentage points. *Ergo, do not forget to write
unit tests for your changes.* To check the code coverage for your code, follow
these steps:

1. Install dependencies for C/C++ source.

    For C/C++ source we require `lcov`, `llvm-cov`, and `llvm-profdata`. Note
    that `llvm-cov` and `llvm-profdata` should be version 11.0 or higher. If you
    have multiple `llvm-cov` tools installed, most likely because you have
    multiple llvm installations, you should set the `LLVM_TOOLS_HOME`
    environment variable to make sure the correct one is used to generate
    coverage.

2. Install dependencies for Python sources.

    To generate the coverage data for dpctl's Python sources, you only need to
    install `coverage`.

    ```bash
    python -m pip install coverage[toml]
    ```

3. Build dpctl with code coverage support.

    ```bash
    python scripts/gen_coverage.py --oneapi
    coverage html
    ```

    Note that code coverage builds the C sources with debug symbols. For this
    reason, the coverage script builds the package in `develop` mode of
    `setup.py`.

    The coverage results for the C and Python sources will be printed to the
    terminal during the build (C API) and during the pytest execution (Python).
    The detailed coverage reports for the C API is saved to the
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
>    The error is related to the `tcl` package. You should uninstall the `tcl`
> package to resolve the error.

## Error Reporting and Logging

The SyclInterface library responds to `DPCTL_VERBOSITY` environment variable that controls the severity level of errors printed to console.
One can specify one of the following severity levels (in increasing order of severity): `warning` and `error`.

```bash
export DPCTL_VERBOSITY=warning
```

Messages of a given severity are shown not only in the console for that severity, but also for the higher severity. For example, the severity level `warning` will output severity errors for `error` and `warning` to the console.

### Optional use of the Google logging library (glog)

Dpctl's error handler for libsyclinterface can be optionally configured to use [glog](https://github.com/google/glog). To use glog, follow the following steps:

1. Install glog package of the latest version (0.5.0)

```bash
conda install glog
```
2. Build dpctl with glog support

```bash
python scripts/build_locally.py --oneapi --glog
```

3. Use `dpctl._diagnostics.syclinterface_diagnostics(verbosity="warning", log_dir=None)` context manager to switch library diagnostics on for a block of Python code.
Use `DPCTLService_InitLogger` and `DPCTLService_ShutdownLogger` library C functions during library development to initialize the Google's logging library and de-initialize accordingly

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

 - `*app_name` - name of the executable file (prefix for logs of various levels).
 - `*log_dir` - directory path for writing log files. Specifying `NULL` results in logging to ``std::cerr``.

> **_NOTE:_**
>
> If `InitGoogleLogging` is not called before first use of glog, the library will self-initialize to `logtostderr` mode and log files will not be generated.
