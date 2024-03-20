//=== syclinterace.h - single include header for libsyclinterface  -*-C-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
/// This file includes all the headers of syclinterface/
//===----------------------------------------------------------------------===//

#pragma once

// clang-format off
#include "syclinterface/dpctl_sycl_types.h"
#include "syclinterface/dpctl_sycl_enum_types.h"
#include "syclinterface/dpctl_service.h"
#include "syclinterface/dpctl_vector.h"
#include "syclinterface/dpctl_utils.h"
#include "syclinterface/dpctl_sycl_device_selector_interface.h"
#include "syclinterface/dpctl_sycl_context_interface.h"
#include "syclinterface/dpctl_sycl_device_interface.h"
#include "syclinterface/dpctl_sycl_event_interface.h"
#include "syclinterface/dpctl_sycl_platform_interface.h"
#include "syclinterface/dpctl_sycl_queue_interface.h"
#include "syclinterface/dpctl_sycl_usm_interface.h"
#include "syclinterface/dpctl_sycl_device_manager.h"
#include "syclinterface/dpctl_sycl_platform_manager.h"
#include "syclinterface/dpctl_sycl_kernel_bundle_interface.h"
#include "syclinterface/dpctl_sycl_kernel_interface.h"
// clang-format on
