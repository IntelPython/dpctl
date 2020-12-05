//===-------- test_sycl_program_interface.cpp - dpctl-C_API -*- C++ ---*---===//
//
//               Data Parallel Control Library (dpCtl)
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
/// This file has unit test cases for functions defined in
/// dpctl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_program_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_utils.h"

#include <array>
#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;


struct TestDPCTLSyclKernelInterface : public ::testing::Test
{
    const char *CLProgramStr = R"CLC(
        kernel void add(global int* a, global int* b, global int* c) {
            size_t index = get_global_id(0);
            c[index] = a[index] + b[index];
        }

        kernel void axpy(global int* a, global int* b, global int* c, int d) {
            size_t index = get_global_id(0);
            c[index] = a[index] + d*b[index];
        }
    )CLC";
    const char *CompileOpts ="-cl-fast-relaxed-math";

    size_t nOpenCLGpuQ = 0;
    TestDPCTLSyclKernelInterface ()
        : nOpenCLGpuQ(DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU))
    {  }
};

TEST_F (TestDPCTLSyclKernelInterface, CheckGetFunctionName)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    auto CtxRef   = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    auto fnName1 = DPCTLKernel_GetFunctionName(AddKernel);
    auto fnName2 = DPCTLKernel_GetFunctionName(AxpyKernel);

    ASSERT_STREQ("add", fnName1);
    ASSERT_STREQ("axpy", fnName2);

    DPCTLCString_Delete(fnName1);
    DPCTLCString_Delete(fnName2);

    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
}

TEST_F (TestDPCTLSyclKernelInterface, CheckGetNumArgs)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    auto CtxRef   = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    ASSERT_EQ(DPCTLKernel_GetNumArgs(AddKernel), 3ul);
    ASSERT_EQ(DPCTLKernel_GetNumArgs(AxpyKernel), 4ul);

    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
}

