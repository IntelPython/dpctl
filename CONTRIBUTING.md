# Mechanical Source Issues¶

## Source Code Formatting¶

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

Every Python and Cython File should only include the following license header:

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
