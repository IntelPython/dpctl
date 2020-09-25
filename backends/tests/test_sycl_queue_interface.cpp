//===---- test_sycl_queue_interface.cpp - DPPL-SYCL interface -*- C++ --*--===//
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
/// dppl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_event_interface.h"
#include "dppl_sycl_kernel_interface.h"
#include "dppl_sycl_program_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"
#include "dppl_sycl_usm_interface.h"

#include "Support/CBindingWrapping.h"

#include <gtest/gtest.h>

namespace
{
    constexpr size_t SIZE = 1024;

    DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPPLSyclUSMRef);

    void add_kernel_checker (const float *a, const float *b, const float *c)
    {
        // Validate the data
        for(auto i = 0ul; i < SIZE; ++i) {
            EXPECT_EQ(c[i], a[i] + b[i]);
        }
    }

    void axpy_kernel_checker (const float *a, const float *b, const float *c,
                              float d)
    {
        for(auto i = 0ul; i < SIZE; ++i) {
            EXPECT_EQ(c[i], a[i] + d*b[i]);
        }
    }
}

struct TestDPPLSyclQueueInterface : public ::testing::Test
{
    const char *CLProgramStr = R"CLC(
        kernel void init_arr (global float *a) {
            size_t index = get_global_id(0);
            a[index] = (float)index;
        }

        kernel void add (global float* a, global float* b, global float* c) {
            size_t index = get_global_id(0);
            c[index] = a[index] + b[index];
        }

        kernel void axpy (global float* a, global float* b,
                          global float* c, float d) {
            size_t index = get_global_id(0);
            c[index] = a[index] + d*b[index];
        }
    )CLC";
    const char *CompileOpts ="-cl-fast-relaxed-math";

    DPPLSyclContextRef CurrCtxRef   = nullptr;
    DPPLSyclQueueRef   CurrQueue    = nullptr;
    DPPLSyclProgramRef PRef         = nullptr;
    DPPLSyclProgramRef PRef2        = nullptr;
    TestDPPLSyclQueueInterface ()
    {
        CurrQueue = DPPLQueueMgr_GetCurrentQueue();
        CurrCtxRef   = DPPLQueue_GetContext(CurrQueue);
        PRef = DPPLProgram_CreateFromOCLSource(CurrCtxRef, CLProgramStr,
                                               CompileOpts);
    }

    ~TestDPPLSyclQueueInterface ()
    {
        DPPLQueue_Delete(CurrQueue);
        DPPLContext_Delete(CurrCtxRef);
        DPPLProgram_Delete(PRef);
    }
};

TEST_F (TestDPPLSyclQueueInterface, CheckSubmit)
{
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "init_arr"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPPLProgram_HasKernel(PRef, "axpy"));

    auto InitKernel = DPPLProgram_GetKernel(PRef, "init_arr");
    auto AddKernel  = DPPLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPPLProgram_GetKernel(PRef, "axpy");

    // Create the input args
    auto a = DPPLmalloc_shared(SIZE, CurrQueue);
    ASSERT_TRUE(a != nullptr);
    auto b = DPPLmalloc_shared(SIZE, CurrQueue);
    ASSERT_TRUE(b != nullptr);
    auto c = DPPLmalloc_shared(SIZE, CurrQueue);
    ASSERT_TRUE(c != nullptr);

    // Initialize a,b
    DPPLKernelArgType argTypes[] = {VOID_PTR};
    size_t Range[] = {SIZE};
    void *arg1[1] = { unwrap(a) };
    void *arg2[1] = { unwrap(b) };

    auto E1 = DPPLQueue_Submit(InitKernel, CurrQueue, arg1, argTypes, 1,
                               Range, 1);
    auto E2 = DPPLQueue_Submit(InitKernel, CurrQueue, arg2, argTypes, 1,
                               Range, 1);
    ASSERT_TRUE(E1 != nullptr);
    ASSERT_TRUE(E2 != nullptr);

    DPPLQueue_Wait(CurrQueue);

    // Submit the add kernel
    void *args[3] = { unwrap(a), unwrap(b), unwrap(c) };
    DPPLKernelArgType addKernelArgTypes[] = {VOID_PTR, VOID_PTR, VOID_PTR};

    auto E3 = DPPLQueue_Submit(AddKernel, CurrQueue, args, addKernelArgTypes, 3,
                               Range, 1);
    ASSERT_TRUE(E3 != nullptr);
    DPPLQueue_Wait(CurrQueue);

    // Verify the result of "add"
    add_kernel_checker((float*)a, (float*)b, (float*)c);

    // Create kernel args for axpy
    float d = 10.0;
    void *args2[4] = { unwrap(a), unwrap(b), unwrap(c) , (void*)&d};
    DPPLKernelArgType addKernelArgTypes2[] = { VOID_PTR,
                                               VOID_PTR,
                                               VOID_PTR,
                                               FLOAT
                                             };
    auto E4 = DPPLQueue_Submit(AxpyKernel, CurrQueue, args2,
                               addKernelArgTypes2, 4, Range, 1);
    ASSERT_TRUE(E4 != nullptr);
    DPPLQueue_Wait(CurrQueue);

    // Verify the result of "axpy"
    axpy_kernel_checker((float*)a, (float*)b, (float*)c, d);

    // clean ups
    DPPLEvent_Delete(E1);
    DPPLEvent_Delete(E2);
    DPPLEvent_Delete(E3);
    DPPLEvent_Delete(E4);

    DPPLKernel_Delete(AddKernel);
    DPPLKernel_Delete(AxpyKernel);
    DPPLKernel_Delete(InitKernel);

    DPPLfree_with_queue((DPPLSyclUSMRef)a, CurrQueue);
    DPPLfree_with_queue((DPPLSyclUSMRef)b, CurrQueue);
    DPPLfree_with_queue((DPPLSyclUSMRef)c, CurrQueue);
}

int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
