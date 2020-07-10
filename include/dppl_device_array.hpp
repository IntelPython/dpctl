//===-- dppl_device_array.hpp - DPPL-SYCL interface -------*- C++ -*-------===//
//
//                     Data Parallel Python (DPPL)
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
/// This file contains the declaration of a NumPy ndarray drop in replacement
/// that is based on sycl::buffers.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "dppl_error_codes.hpp"
#include <cstdint>

namespace dppl
{
/*! \class DppyDeviceArray
 *
 */
template <typename T>
class DppyDeviceArray
{
public:
    using value_type = T;

private:
    void        *buff_;
    std::size_t *shape_;
    int         ndims_;

public:

    int64_t getBuffer (void **buff) const;

    DppyDeviceArray (T *hostData, int ndims, const std::size_t *shape);

    DppyDeviceArray (const T* hostData, int ndims, const std::size_t *shape);

    DppyDeviceArray (int ndims, const std::size_t *shape);

    // TODO : Copy, Move Ctors, copy assign operators
    virtual ~DppyDeviceArray ();
};

}
