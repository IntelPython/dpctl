//===-- dppy_oneapi_interface.hpp - DPPY-SYCL interface ---*- C++ -*-------===//
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
/// This file contains the declaration of a C++ API to expose a lightweight SYCL
/// interface for the Python dppy package.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>                /* SYCL headers   */
#include <deque>
#include <variant>


namespace dppy
{

enum : int64_t
{
    DPPY_FAILURE = -1,
    DPPY_SUCCESS
};


////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIBuffer //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! \class DppyOneAPIBuffer1D
 *
 */
template <typename T>
class DppyOneAPIBuffer
{
public:
    using buff_variant = std::variant<
                             cl::sycl::buffer<T, 1>,
                             cl::sycl::buffer<T, 2>,
                             cl::sycl::buffer<T, 3>
                         >;
private:

    buff_variant buff_;

    // Stores the size of the buffer_ptr (e.g sizeof(cl_mem))
    size_t sizeof_buffer_ptr_;
    size_t ndims_;
    size_t *dims_;

public:

    DppyOneAPIBuffer (T *hostData, size_t ndims, const size_t dims[],
                      const sycl::property_list & propList = {});

    DppyOneAPIBuffer (const T* hostData, size_t ndims, const size_t dims[],
                      const sycl::property_list & propList = {});

    DppyOneAPIBuffer (size_t ndims, const size_t dims[],
                      const sycl::property_list& propList = {});

    // TODO : Copy, Move Ctors, copy assign operators

    ~DppyOneAPIBuffer ();
};


////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIRuntime /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*! \class DppyOneAPIRuntime
 *  \brief A runtime and context factory class
 *
 */
class DppyOneAPIRuntime
{
    size_t                                  num_platforms_;
    cl::sycl::vector_class<cl::sycl::queue> cpu_queues_;
    cl::sycl::vector_class<cl::sycl::queue> gpu_queues_;
    std::deque<cl::sycl::queue>             active_queues_;

public:
    int64_t getNumPlatforms (size_t *platforms) const;
    int64_t getCurrentQueue (cl::sycl::queue **Q) const;
    int64_t getQueue (cl::sycl::queue **Q,
                      cl::sycl::info::device_type DeviceTy,
                      size_t DNum = 0) const;
    int64_t resetGlobalQueue (cl::sycl::info::device_type DeviceTy,
                              size_t DNum = 0);
    /*!
     * Push a new sycl queue to the top of the activate_queues deque. The
     * newly activated queue is returned to caller inside the Q object.
     */
    int64_t activateQueue (cl::sycl::queue **Q,
                           cl::sycl::info::device_type DeviceTy,
                           size_t DNum);
    int64_t deactivateCurrentQueue ();
    int64_t dump () const;
    int64_t dump_queue (const cl::sycl::queue *Q) const;

    DppyOneAPIRuntime();
    ~DppyOneAPIRuntime();
};


int64_t deleteQueue (void *Q);

} /* end of namespace dppy */
