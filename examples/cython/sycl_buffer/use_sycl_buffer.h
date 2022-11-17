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

#include "dpctl_sycl_interface.h"
#include <CL/sycl.hpp>

extern int c_columnwise_total(DPCTLSyclQueueRef q,
                              size_t n,
                              size_t m,
                              double *mat,
                              double *ct);
extern int c_columnwise_total_no_mkl(DPCTLSyclQueueRef q,
                                     size_t n,
                                     size_t m,
                                     double *mat,
                                     double *ct);
