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
//#include <CL/cl.h>                    /* OpenCL headers */
#include <deque>
#include <memory>
#include <variant>


namespace dppy
{

enum : int64_t
{
    DPPY_FAILURE = -1,
    DPPY_SUCCESS
};

#if 0
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIBuffer //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! \class DppyOneAPIBuffer1D
 *
 */
template <typename T>
class DppyOneAPIBuffer
{
    std::variant<
        cl::sycl::buffer<T, 1>,
        cl::sycl::buffer<T, 2>,
        cl::sycl::buffer<T, 3>
    > buffer;

    // Stores the size of the buffer_ptr (e.g sizeof(cl_mem))
    size_t sizeof_buffer_ptr_;
public:

    DppyOneAPIBuffer (T *hostData, const property_list& propList = {});
};

#endif
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIContext /////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! \class DppyOneAPIContext
 *  \brief A convenience wrapper encapsulating a SYCL queue
 *
 */
class DppyOneAPIContext
{
    std::shared_ptr<cl::sycl::queue> queue_;

public:
    int64_t getSyclQueue (std::shared_ptr<cl::sycl::queue> * Queue) const;

#if 0
    int64_t getSyclContext (cl::sycl::context * Context) const;
    int64_t getSyclDevice (cl::sycl::device * Device) const;
    int64_t getOpenCLQueue (cl_command_queue * Cl_Queue) const;
    int64_t getOpenCLContext (cl_context * CL_Context) const;
    int64_t getOpenCLDevice (cl_device_id * CL_Device) const;
#endif

    int64_t dump ();
    virtual ~DppyOneAPIContext () = default;
    DppyOneAPIContext (const cl::sycl::device_selector & DeviceSelector);
    DppyOneAPIContext (const cl::sycl::device & Device);
    DppyOneAPIContext (const DppyOneAPIContext & Ctx);
    DppyOneAPIContext (DppyOneAPIContext && Ctx);
    DppyOneAPIContext& operator=(const DppyOneAPIContext & Ctx);
    DppyOneAPIContext& operator=(DppyOneAPIContext && other);
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
    size_t                                         num_platforms_;
    cl::sycl::vector_class<cl::sycl::device>       cpu_devices_;
    cl::sycl::vector_class<cl::sycl::device>       gpu_devices_;
    std::deque<std::shared_ptr<DppyOneAPIContext>> active_contexts_;

public:
    int64_t getNumPlatforms (size_t *platforms) const;
    int64_t getCurrentContext (std::shared_ptr<DppyOneAPIContext> * C) const;
    int64_t setGlobalContextWithGPU (size_t DNum);
    int64_t setGlobalContextWithCPU (size_t DNum);
    /*!
     * Try to create a new DppyOneAPIContext and add it to the front of the
     * deque. If a new DppyOneAPIContext, return it encapsulated within the
     * passed in shared_ptr. Caller should check the return flag to check if
     * a new DppyOneAPIContext was created, before using the shared_ptr.
     *
     */
    int64_t pushGPUContext (std::shared_ptr<DppyOneAPIContext> * C,
                            size_t DNum);
    int64_t pushCPUContext (std::shared_ptr<DppyOneAPIContext> * C,
                            size_t DNum);
    int64_t popContext ();
    int64_t dump () const;

    DppyOneAPIRuntime();
    ~DppyOneAPIRuntime();
};

} /* end of namespace dppy_rt */
