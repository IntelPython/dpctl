//===-- dppy_device_array.cpp - DPPY-SYCL interface -------*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
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
/// This file implements the DppyDeviceArray class.
///
//===----------------------------------------------------------------------===//
#include "dppy_device_array.hpp"
#include "dppy_error_codes.hpp"
#include <algorithm>
#include <iterator>
#include <stdexcept>

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace std;
using namespace cl::sycl;
using namespace dppy;

template <typename T>
DppyDeviceArray<T>::DppyDeviceArray (T* hostData, int ndims, const size_t *dims)
{
    auto size = 1ul;

    /* Flatten the array into a 1D buffer */
    for (auto d = 0; d < ndims; ++d)
        size *= dims[d];

    auto r = range(size);
    buff_ = new buffer<T>(hostData, r);

    shape_ = new size_t[ndims];
    std::copy(dims, dims+ndims, shape_);
}


template <typename T>
DppyDeviceArray<T>::DppyDeviceArray (const T* hostData, int ndims,
                                     const size_t *dims)
    : ndims_(ndims)
{
    auto size = 1ul;

    /* Flatten the array into a 1D buffer */
    for (auto d = 0; d < ndims; ++d)
        size *= dims[d];

    auto r = range(size);
    buff_ = new buffer<T>(hostData, r);

    shape_ = new size_t[ndims];
    std::copy(dims, dims+ndims, shape_);
    shape_ = new size_t[ndims];
    std::copy(dims, dims+ndims, shape_);
}


template <typename T>
DppyDeviceArray<T>::DppyDeviceArray (int ndims, const size_t *dims)
    : ndims_(ndims)
{
    auto size = 1ul;

    /* Flatten the array into a 1D buffer */
    for (auto d = 0; d < ndims; ++d)
        size *= dims[d];

    auto r = range(size);
    buff_ = new buffer<T>(r);

    shape_ = new size_t[ndims];
    std::copy(dims, dims+ndims, shape_);
}


template <typename T>
DppyDeviceArray<T>::~DppyDeviceArray()
{
    delete shape_;
    delete static_cast<buffer<T>>(buff_);
}


template <typename T>
int64_t DppyDeviceArray<T>::getBuffer (void **buff) const
{
    *buff = new buffer<T>(buff_);
    return DPPY_SUCCESS;
}
