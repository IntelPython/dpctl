//===------------------- dppl_utils.h - dpctl-C_API ---*--- C++ -----*-----===//
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
/// This file defines common helper functions used in other places in DPPL.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
using namespace cl::sycl;

std::string DDPL_StrToDeviceType(info::device_type devTy);
info::device_type DPPL_DeviceTypeToStr(std::string devTyStr);
