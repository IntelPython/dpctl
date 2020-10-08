//===---------- test_sycl_program_interface.cpp - dpctl-C_API --*-- C++ -*-===//
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
/// dppl_sycl_program_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_kernel_interface.h"
#include "dppl_sycl_program_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"

#include <array>
#include <fstream>
#include <experimental/filesystem>
#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;
using fs = std::fs;

namespace
{
    const size_t SIZE = 1024;

    void add_kernel_checker (queue *syclQueue, DPPLSyclKernelRef AddKernel)
    {
        range<1> a_size{SIZE};
        std::array<int, SIZE> a, b, c;

        for (int i = 0; i<SIZE; ++i) {
            a[i] = i;
            b[i] = i;
            c[i] = 0;
        }

        {
            buffer<int, 1> a_device(a.data(), a_size);
            buffer<int, 1> b_device(b.data(), a_size);
            buffer<int, 1> c_device(c.data(), a_size);
            buffer<int, 1> buffs[3] = {a_device, b_device, c_device};
            syclQueue->submit([&](handler& cgh) {
                for (auto buff : buffs) {
                    auto arg = buff.get_access<access::mode::read_write>(cgh);
                    cgh.set_args(arg);
                }
                auto syclKernel = reinterpret_cast<kernel*>(AddKernel);
                cgh.parallel_for(range<1>{SIZE}, *syclKernel);
            });
        }

        // Validate the data
        for(auto i = 0ul; i < SIZE; ++i) {
            EXPECT_EQ(c[i], i + i);
        }
    }

    void axpy_kernel_checker (queue *syclQueue, DPPLSyclKernelRef AxpyKernel)
    {
        range<1> a_size{SIZE};
        std::array<int, SIZE> a, b, c;

        for (int i = 0; i<SIZE; ++i) {
            a[i] = i;
            b[i] = i;
            c[i] = 0;
        }
        int d = 10;
        {
            buffer<int, 1> a_device(a.data(), a_size);
            buffer<int, 1> b_device(b.data(), a_size);
            buffer<int, 1> c_device(c.data(), a_size);
            buffer<int, 1> buffs[3] = {a_device, b_device, c_device};
            syclQueue->submit([&](handler& cgh) {
                for (auto i = 0ul; i < 3; ++i) {
                    auto arg = buffs[i].get_access<access::mode::read_write>(cgh);
                    cgh.set_arg(i, arg);
                }
                cgh.set_arg(3, d);
                auto syclKernel = reinterpret_cast<kernel*>(AxpyKernel);
                cgh.parallel_for(range<1>{SIZE}, *syclKernel);
            });
        }

        // Validate the data
        for(auto i = 0ul; i < SIZE; ++i) {
            EXPECT_EQ(c[i], i + d*i);
        }
    }
}

struct TestDPPLSyclProgramInterface : public ::testing::Test
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
    std::ifstream spirvFile;
    size_t spirvFileSize = 0;
    std::vector<char> spirvBuffer;
    size_t nOpenCLGpuQ = 0;

    TestDPPLSyclProgramInterface () :
        nOpenCLGpuQ(DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_GPU)),
        spirvFile{"./multi_kernel.spv", std::ios::binary | std::ios::ate},
        spirvFileSize(fs::file_size("./multi_kernel.spv")),
        spirvBuffer(spirvFileSize)
    {
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer.data(), spirvFileSize);
    }

    ~TestDPPLSyclProgramInterface ()
    {
        spirvFile.close();
    }
};

TEST_F (TestDPPLSyclProgramInterface, CheckCreateFromOCLSource)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
    auto CtxRef = DPPLQueue_GetContext(QueueRef);
    auto PRef = DPPLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "axpy"));

    DPPLQueue_Delete(QueueRef);
    DPPLContext_Delete(CtxRef);
    DPPLProgram_Delete(PRef);
}

TEST_F (TestDPPLSyclProgramInterface, CheckCreateFromOCLSpirv)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
    auto CtxRef = DPPLQueue_GetContext(QueueRef);
    auto PRef = DPPLProgram_CreateFromOCLSpirv(CtxRef, spirvBuffer.data(),
                                               spirvFileSize);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "axpy"));

    DPPLQueue_Delete(QueueRef);
    DPPLContext_Delete(CtxRef);
    DPPLProgram_Delete(PRef);
}

TEST_F (TestDPPLSyclProgramInterface, CheckGetKernelOCLSource)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
    auto CtxRef = DPPLQueue_GetContext(QueueRef);
    auto PRef = DPPLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    auto AddKernel = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");

    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);

    auto syclQueue = reinterpret_cast<queue*>(QueueRef);

    add_kernel_checker(syclQueue, AddKernel);
    axpy_kernel_checker(syclQueue, AxpyKernel);

    DPPLKernel_Delete(AddKernel);
    DPPLKernel_Delete(AxpyKernel);
    DPPLQueue_Delete(QueueRef);
    DPPLContext_Delete(CtxRef);
    DPPLProgram_Delete(PRef);
}

TEST_F (TestDPPLSyclProgramInterface, CheckGetKernelOCLSpirv)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
    auto CtxRef = DPPLQueue_GetContext(QueueRef);
    auto PRef = DPPLProgram_CreateFromOCLSpirv(CtxRef, spirvBuffer.data(),
                                               spirvFileSize);
    auto AddKernel = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");

    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);

    auto syclQueue = reinterpret_cast<queue*>(QueueRef);

    add_kernel_checker(syclQueue, AddKernel);
    axpy_kernel_checker(syclQueue, AxpyKernel);

    DPPLKernel_Delete(AddKernel);
    DPPLKernel_Delete(AxpyKernel);
    DPPLQueue_Delete(QueueRef);
    DPPLContext_Delete(CtxRef);
    DPPLProgram_Delete(PRef);
}

int
main (int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
