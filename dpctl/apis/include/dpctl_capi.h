//===----------- dpctl_capi.h - Headers for dpctl's       C-API   -*-C-*- ===//
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
/// This file defines imports for dcptl's Python C-API
//===----------------------------------------------------------------------===//

#pragma once

// clang-format off
// Ordering of includes is important here. dpctl_sycl_types defines types
// used by dpctl's Python C-API headers.
#include "syclinterface/dpctl_sycl_types.h"
#include "../_sycl_device.h"
#include "../_sycl_device_api.h"
#include "../_sycl_context.h"
#include "../_sycl_context_api.h"
#include "../_sycl_event.h"
#include "../_sycl_event_api.h"
#include "../_sycl_queue.h"
#include "../_sycl_queue_api.h"
#include "../memory/_memory.h"
#include "../memory/_memory_api.h"
#include "../tensor/_usmarray.h"
#include "../tensor/_usmarray_api.h"
// clang-format on

/*
 * Function to import dpctl and make C-API functions available.
 * C functions can use dpctl's C-API functions without linking to
 * shared objects defining this symbols, if they call `import_dpctl()`
 * prior to using those symbols.
 */
static inline void import_dpctl(void)
{
    import_dpctl___sycl_device();
    import_dpctl___sycl_context();
    import_dpctl___sycl_event();
    import_dpctl___sycl_queue();
    import_dpctl__memory___memory();
    import_dpctl__tensor___usmarray();
    return;
}
