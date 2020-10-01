//===---- test_sycl_program_interface.cpp - DPPL-SYCL interface -*- C++ -*-===//
//
//               Python Data Parallel Processing Library (PyDPPL)
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
/// dppl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_kernel_interface.h"
#include "dppl_sycl_program_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"
#include "dppl_utils.h"

#include <array>
#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;


struct TestDPPLSyclKernelInterface : public ::testing::Test
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
    TestDPPLSyclKernelInterface ()
        : nOpenCLGpuQ(DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_GPU))
    {  }
};

TEST_F (TestDPPLSyclKernelInterface, CheckGetFunctionName)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
    auto CtxRef   = DPPLQueue_GetContext(QueueRef);
    auto PRef = DPPLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    auto AddKernel = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");

    auto fnName1 = DPPLKernel_GetFunctionName(AddKernel);
    auto fnName2 = DPPLKernel_GetFunctionName(AxpyKernel);

    ASSERT_STREQ("add", fnName1);
    ASSERT_STREQ("axpy", fnName2);

    DPPLCString_Delete(fnName1);
    DPPLCString_Delete(fnName2);

    DPPLQueue_Delete(QueueRef);
    DPPLContext_Delete(CtxRef);
    DPPLProgram_Delete(PRef);
    DPPLKernel_Delete(AddKernel);
    DPPLKernel_Delete(AxpyKernel);
}

TEST_F (TestDPPLSyclKernelInterface, CheckGetNumArgs)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
    auto CtxRef   = DPPLQueue_GetContext(QueueRef);
    auto PRef = DPPLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    auto AddKernel = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");

    ASSERT_EQ(DPPLKernel_GetNumArgs(AddKernel), 3);
    ASSERT_EQ(DPPLKernel_GetNumArgs(AxpyKernel), 4);

    DPPLQueue_Delete(QueueRef);
    DPPLContext_Delete(CtxRef);
    DPPLProgram_Delete(PRef);
    DPPLKernel_Delete(AddKernel);
    DPPLKernel_Delete(AxpyKernel);
}

int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
