//===----------- usm_array.hpp - class representing an array  -*-C++-*- ===//
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
/// This file defines classes for strided_array, and usm_array
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl_sycl_types.h"
#include <cstdlib>

namespace usm_array
{

class strided_array
{
public:
    /* strided_array is data only class encapsulating information about
     * type homogeneous nd-array.
     *    ptr     : pointer to memory block storing array values
     *    nd      : number of indices needed to reference an array element
     *    shape   : pointer to C-array of length `nd` of array dimensions
     *    strides : pointer to C-array of length `nd` of memory displacements
     *              for unit increment of each index
     *    typenum : an integer (enum), encoding value type of array elements
     *    flags   : field to encode additional array attributes
     */
    explicit strided_array(char *ptr, int nd, size_t *shape, int typenum)
        : ptr_(ptr), nd_(nd), shape_(shape), typenum_(typenum){};
    explicit strided_array(char *ptr,
                           int nd,
                           size_t *shape,
                           std::ptrdiff_t *strides,
                           int typenum)
        : ptr_(ptr), nd_(nd), shape_(shape), strides_(strides),
          typenum_(typenum){};
    explicit strided_array(char *ptr,
                           int nd,
                           size_t *shape,
                           std::ptrdiff_t *strides,
                           int typenum,
                           int flags)
        : ptr_(ptr), nd_(nd), shape_(shape), strides_(strides),
          typenum_(typenum), flags_(flags){};

    // member access functions
    char *get_data_ptr() const
    {
        return ptr_;
    }
    int ndim() const
    {
        return nd_;
    }
    size_t *get_shape_ptr() const
    {
        return shape_;
    }
    std::ptrdiff_t *get_strides_ptr() const
    {
        return strides_;
    }
    int typenum() const
    {
        return typenum_;
    }
    int flags() const
    {
        return flags_;
    }

    size_t get_shape(int i) const
    {
        return shape_[i];
    }
    std::ptrdiff_t get_stride(int i) const
    {
        return strides_[i];
    }

private:
    char *ptr_{0};
    int nd_{0};
    size_t *shape_{0};
    std::ptrdiff_t *strides_{0};
    int typenum_{0};
    int flags_{0};
};

class usm_array : public strided_array
{
public:
    /*
     * usm_array additionally carries DPCTLSyclQueueRef
     * recording Sycl context the data USM pointer is bound to
     */
    explicit usm_array(char *data,
                       int nd,
                       size_t *shape,
                       std::ptrdiff_t *strides,
                       int typenum,
                       int flags,
                       DPCTLSyclQueueRef qref)
        : strided_array(data, nd, shape, strides, typenum, flags), q_(qref){};

    DPCTLSyclQueueRef get_queue_ref() const
    {
        return q_;
    }

private:
    DPCTLSyclQueueRef q_{0};
};

} // namespace usm_array
