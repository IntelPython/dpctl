##===---------- __init__.py - dpctl.experimentl module ----*- Python -*----===##
##
##                      Data Parallel Control (dpCtl)
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
##
## \file
## This file contains various experimental features defined through
## `dpctl.experimental`.
##
##===----------------------------------------------------------------------===##
"""
    **Data Parallel Control Experimental**

    `dpctl.experimental` is a staging area for features that may at some point
    be included into `dpctl`. Presently, the module provides a way to create a
    SYCL kernel from either a source string or SPIR-V binary file.

"""
from ._program import *
from ._program import __all__ as _program__all__

__all__ = _program__all__
