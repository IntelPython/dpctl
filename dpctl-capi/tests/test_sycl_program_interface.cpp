//===---------- test_sycl_program_interface.cpp - dpctl-C_API ---*- C++ -*-===//
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
/// dpctl_sycl_program_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_program_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "Config/dpctl_config.h"
#include <array>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace
{
const int SIZE = 1024;

void add_kernel_checker (queue *syclQueue, DPCTLSyclKernelRef AddKernel)
{
    range<1> a_size{SIZE};
    std::array<int, SIZE> a, b, c;

    for (int i = 0; i < SIZE; ++i) {
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
    for(int i = 0; i < SIZE; ++i) {
        EXPECT_EQ(c[i], i + i);
    }
}

void axpy_kernel_checker (queue *syclQueue, DPCTLSyclKernelRef AxpyKernel)
{
    range<1> a_size{SIZE};
    std::array<int, SIZE> a, b, c;

    for (int i = 0; i < SIZE; ++i) {
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
    for(int i = 0; i < SIZE; ++i) {
        EXPECT_EQ(c[i], i + d*i);
    }
}

} /* end of anonymous namespace */

struct TestDPCTLSyclProgramInterface : public ::testing::Test
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

    TestDPCTLSyclProgramInterface () :
        spirvFile{"./multi_kernel.spv", std::ios::binary | std::ios::ate},
        spirvFileSize(std::filesystem::file_size("./multi_kernel.spv")),
        spirvBuffer(spirvFileSize),
        nOpenCLGpuQ(DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU))
    {
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer.data(), spirvFileSize);
    }

    ~TestDPCTLSyclProgramInterface ()
    {
        spirvFile.close();
    }
};

TEST_F (TestDPCTLSyclProgramInterface, CheckCreateFromOCLSource)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    auto CtxRef = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                 CompileOpts);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));

    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
}

TEST_F (TestDPCTLSyclProgramInterface, CheckCreateFromSpirvOCL)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    auto CtxRef = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromSpirv(CtxRef, spirvBuffer.data(),
                                             spirvFileSize,
                                             nullptr);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));

    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
}

#ifdef DPCTL_ENABLE_LO_PROGRAM_CREATION
TEST_F (TestDPCTLSyclProgramInterface, CheckCreateFromSpirvL0)
{
    auto nL0GpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_LEVEL_ZERO, DPCTL_GPU));
    if(!nL0GpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU, 0);
    auto CtxRef = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromSpirv(CtxRef, spirvBuffer.data(),
                                             spirvFileSize,
                                             nullptr);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));

    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
}
#endif

TEST_F (TestDPCTLSyclProgramInterface, CheckGetKernelOCLSource)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    auto CtxRef = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                                CompileOpts);
    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);

    auto syclQueue = reinterpret_cast<queue*>(QueueRef);

    add_kernel_checker(syclQueue, AddKernel);
    axpy_kernel_checker(syclQueue, AxpyKernel);

    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
}

TEST_F (TestDPCTLSyclProgramInterface, CheckGetKernelSpirv)
{
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto QueueRef = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    auto CtxRef = DPCTLQueue_GetContext(QueueRef);
    auto PRef = DPCTLProgram_CreateFromSpirv(CtxRef, spirvBuffer.data(),
                                             spirvFileSize, nullptr);
    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);

    auto syclQueue = reinterpret_cast<queue*>(QueueRef);

    add_kernel_checker(syclQueue, AddKernel);
    axpy_kernel_checker(syclQueue, AxpyKernel);

    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
    DPCTLQueue_Delete(QueueRef);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
}

