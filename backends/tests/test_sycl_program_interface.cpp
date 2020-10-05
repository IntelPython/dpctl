//===---- test_sycl_program_interface.cpp - DPPL-SYCL interface -*- C++ -*-===//
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
#include <filesystem>
#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

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

    DPPLSyclContextRef CtxRef   = nullptr;
    DPPLSyclQueueRef   QueueRef = nullptr;
    DPPLSyclProgramRef PRef     = nullptr;
    DPPLSyclProgramRef PRef2    = nullptr;

    TestDPPLSyclProgramInterface ()
    {
        QueueRef = DPPLQueueMgr_GetQueue(DPPL_GPU, 0);
        CtxRef   = DPPLQueue_GetContext(QueueRef);
        PRef = DPPLProgram_CreateFromOCLSource(CtxRef, CLProgramStr,
                                               CompileOpts);

        // Create a program from a SPIR-V file
        std::ifstream file{"./multi_kernel.spv",
                           std::ios::binary | std::ios::ate};
        auto fileSize = std::filesystem::file_size("./multi_kernel.spv");
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);
        PRef2 = DPPLProgram_CreateFromOCLSpirv(CtxRef, buffer.data(),
                                               fileSize);
    }

    ~TestDPPLSyclProgramInterface ()
    {
        DPPLQueue_Delete(QueueRef);
        DPPLContext_Delete(CtxRef);
        DPPLProgram_Delete(PRef);
        DPPLProgram_Delete(PRef2);
    }
};

TEST_F (TestDPPLSyclProgramInterface, CheckCreateFromOCLSource)
{
    ASSERT_TRUE(PRef != nullptr);
}

TEST_F (TestDPPLSyclProgramInterface, CheckCreateFromOCLSpirv)
{
    ASSERT_TRUE(PRef2 != nullptr);
}

TEST_F (TestDPPLSyclProgramInterface, CheckHasKernelOCLSource)
{
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "axpy"));
}

TEST_F (TestDPPLSyclProgramInterface, CheckHasKernelSpirvSource)
{
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "axpy"));
}

TEST_F (TestDPPLSyclProgramInterface, CheckGetKernelOCLSource)
{
    auto AddKernel = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");
    auto syclQueue = reinterpret_cast<queue*>(QueueRef);

    add_kernel_checker(syclQueue, AddKernel);
    axpy_kernel_checker(syclQueue, AxpyKernel);

    DPPLKernel_Delete(AddKernel);
    DPPLKernel_Delete(AxpyKernel);
}

TEST_F (TestDPPLSyclProgramInterface, CheckGetKernelOCLSpirv)
{
    auto AddKernel = DPPLProgram_GetKernel(PRef2, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef2, "axpy");
    auto syclQueue = reinterpret_cast<queue*>(QueueRef);

    add_kernel_checker(syclQueue, AddKernel);
    axpy_kernel_checker(syclQueue, AxpyKernel);

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
