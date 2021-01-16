//===---- dpctl-capi/Config/dpCtl-config.h - dpctl-C API  -------*- C++ -*-===//
//
//               Data Parallel Control Library (dpCtl)
//
// Copyright 2020 Intel Corporation
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
/// This file exports a set of dpCtl C API configurations.
///
//===----------------------------------------------------------------------===//

#pragma once

/* Defined when dpCtl was built with level zero program creation enabled. */
#define DPCTL_ENABLE_LO_PROGRAM_CREATION ON

/* The DPCPP version used to build dpCtl */
#define DPCTL_DPCPP_VERSION "Intel(R) oneAPI DPC++ Compiler 2021.1.2 (2020.10.0.1214)"
