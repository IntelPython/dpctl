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
class DppyOneAPIBuffer1D
{
    // 1D sycl::buffer
    void *buffer_ptr;
    // Stores the size of the buffer_ptr (e.g sizeof(cl_mem))
    size_t sizeof_buffer_ptr;
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIContext /////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! \class DppyOneAPIContext
 *  \brief A convenience wrapper encapsulating a SYCL queue
 *
 */
class DppyOneAPIContext
{
    cl::sycl::queue queue_;

public:

    int64_t getSyclQueue (cl::sycl::queue * Queue) const;
    int64_t getSyclContext (cl::sycl::context * Context) const;
    int64_t getSyclDevice (cl::sycl::device * Device) const;
#if 0
    int64_t getOpenCLQueue (cl_command_queue * Cl_Queue) const;
    int64_t getOpenCLContext (cl_context * CL_Context) const;
    int64_t getOpenCLDevice (cl_device_id * CL_Device) const;
#endif
    int64_t dump ();

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
    int64_t getCurrentContext (std::shared_ptr<DppyOneAPIContext> & C) const;
    int64_t setCurrentContext (cl::sycl::info::device_type ty,
                                 size_t device_num);
    int64_t resetCurrentContext ();
    int64_t dump () const;

    DppyOneAPIRuntime();
    ~DppyOneAPIRuntime();
};

} /* end of namespace dppy_rt */

#endif /*--- DPPY_ONEAPI_INTERFACE_HPP_ ---*/
