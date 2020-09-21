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
/// dppl_sycl_program_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_kernel_interface.h"
#include "dppl_sycl_program_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"

#include <array>
#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace
{
    const size_t SIZE = 1024;
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

    DPPLSyclContextRef CurrCtxRef   = nullptr;
    DPPLSyclQueueRef   CurrQueueRef = nullptr;
    DPPLSyclProgramRef PRef         = nullptr;

    TestDPPLSyclProgramInterface ()
    {
        CurrQueueRef = DPPLQueueMgr_GetCurrentQueue();
        CurrCtxRef   = DPPLQueue_GetContext(CurrQueueRef);
        PRef = DPPLProgram_CreateFromOCLSource(CurrCtxRef, CLProgramStr,
                                               CompileOpts);
    }

    ~TestDPPLSyclProgramInterface ()
    {
        DPPLQueue_Delete(CurrQueueRef);
        DPPLContext_Delete(CurrCtxRef);
        DPPLProgram_Delete(PRef);
    }
};

TEST_F (TestDPPLSyclProgramInterface, CheckCreateFromOCLSource)
{
    ASSERT_TRUE(PRef != nullptr);
}


TEST_F (TestDPPLSyclProgramInterface, CheckHasKernel)
{
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "axpy"));
}

TEST_F (TestDPPLSyclProgramInterface, CheckGetKernel)
{
    auto AddKernel = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");
    auto syclQueue = reinterpret_cast<queue*>(CurrQueueRef);

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

    DPPLKernel_DeleteKernelRef(AddKernel);
    DPPLKernel_DeleteKernelRef(AxpyKernel);
}

int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
