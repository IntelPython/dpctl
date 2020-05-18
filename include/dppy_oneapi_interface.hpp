//===-- dppy_oneapi_interface.hpp - DPPY-SYCL interface ---*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
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

#ifndef DPPY_ONEAPI_INTERFACE_HPP_
#define DPPY_ONEAPI_INTERFACE_HPP_

#include <deque>
#include <memory>
#include <CL/sycl.hpp>                /* SYCL headers   */
//#include <CL/cl.h>                    /* OpenCL headers */

namespace dppy_rt
{
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIContext /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

enum class ErrorCode
{
    DPPY_SUCCESS, DPPY_FAILURE
};

/*! \class DppyOneAPIContext
 *  \brief A convenience wrapper encapsulating a SYCL queue
 *
 */
class DppyOneAPIContext
{
    cl::sycl::queue queue_;

public:

    ErrorCode getSyclQueue (cl::sycl::queue * Queue) const;
    ErrorCode getSyclContext (cl::sycl::context * Context) const;
    ErrorCode getSyclDevice (cl::sycl::device * Device) const;
#if 0
    ErrorCode getOpenCLQueue (cl_command_queue * Cl_Queue) const;
    ErrorCode getOpenCLContext (cl_context * CL_Context) const;
    ErrorCode getOpenCLDevice (cl_device_id * CL_Device) const;
#endif
    ErrorCode dump ();

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
    std::deque<std::shared_ptr<DppyOneAPIContext>> contexts_;

public:
    ErrorCode getCurrentContext (std::shared_ptr<DppyOneAPIContext> & C) const;
    ErrorCode setCurrentContext (cl::sycl::info::device_type ty,
                                 size_t device_num);
    ErrorCode resetCurrentContext ();
    ErrorCode dump () const;

    DppyOneAPIRuntime();
    ~DppyOneAPIRuntime();
};

} /* end of namespace dppy_rt */

#endif /*--- DPPY_ONEAPI_INTERFACE_HPP_ ---*/
