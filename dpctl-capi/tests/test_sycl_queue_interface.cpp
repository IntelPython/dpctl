//===------ test_sycl_queue_interface.cpp - Test cases for queue interface ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
/// dpctl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_program_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_sycl_usm_interface.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
constexpr size_t SIZE = 1024;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPCTLSyclUSMRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPCTLSyclQueueRef);

void add_kernel_checker(const float *a, const float *b, const float *c)
{
    // Validate the data
    for (auto i = 0ul; i < SIZE; ++i) {
        EXPECT_EQ(c[i], a[i] + b[i]);
    }
}

void axpy_kernel_checker(const float *a,
                         const float *b,
                         const float *c,
                         float d)
{
    for (auto i = 0ul; i < SIZE; ++i) {
        EXPECT_EQ(c[i], a[i] + d * b[i]);
    }
}

bool has_devices()
{
    bool ret = false;
    for (auto &p : platform::get_platforms()) {
        if (p.is_host())
            continue;
        if (!p.get_devices().empty()) {
            ret = true;
            break;
        }
    }
    return ret;
}

} /* End of anonymous namespace */

struct TestDPCTLSyclQueueInterface : public ::testing::Test
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
    const char *CompileOpts = "-cl-fast-relaxed-math";

    TestDPCTLSyclQueueInterface() {}

    ~TestDPCTLSyclQueueInterface() {}
};

struct TestDPCTLQueueMemberFunctions
    : public ::testing::TestWithParam<const char *>
{
protected:
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;

    TestDPCTLQueueMemberFunctions()
    {
        DSRef = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
        QRef =
            DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    }

    void SetUp()
    {
        if (!QRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLQueueMemberFunctions()
    {
        DPCTLQueue_Delete(QRef);
        DPCTLDeviceSelector_Delete(DSRef);
        DPCTLDevice_Delete(DRef);
    }
};

TEST_F(TestDPCTLSyclQueueInterface, Check_CreateForDevice)
{
    /* We are testing that we do not crash even when input is garbage. */
    DPCTLSyclQueueRef QRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(
        QRef = DPCTLQueue_CreateForDevice(nullptr, nullptr, 0));
    ASSERT_TRUE(QRef == nullptr);
}

TEST_F(TestDPCTLSyclQueueInterface, Check_Copy)
{
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;
    EXPECT_NO_FATAL_FAILURE(Q1 = DPCTLQueueMgr_GetCurrentQueue());
    EXPECT_NO_FATAL_FAILURE(Q2 = DPCTLQueue_Copy(Q1));
    EXPECT_TRUE(bool(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
}

TEST_F(TestDPCTLSyclQueueInterface, CheckAreEq)
{
    auto nOclGPU = DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_GPU);
    if (!nOclGPU)
        GTEST_SKIP_("Skipping: No OpenCL GPUs available.\n");

    auto Q1 = DPCTLQueueMgr_GetCurrentQueue();
    auto Q2 = DPCTLQueueMgr_GetCurrentQueue();

    EXPECT_TRUE(DPCTLQueue_AreEq(Q1, Q2));

    auto FSRef = DPCTLFilterSelector_Create("opencl:gpu:0");
    auto DRef = DPCTLDevice_CreateFromSelector(FSRef);
    auto Q3 = DPCTLQueue_CreateForDevice(DRef, nullptr, 0);
    auto Q4 = DPCTLQueue_CreateForDevice(DRef, nullptr, 0);

    // These are different queues
    EXPECT_FALSE(DPCTLQueue_AreEq(Q3, Q4));

    auto C0 = DPCTLQueue_GetContext(Q3);
    auto C1 = DPCTLQueue_GetContext(Q4);

    // All the queues should share the same context
    EXPECT_TRUE(DPCTLContext_AreEq(C0, C1));

    DPCTLContext_Delete(C0);
    DPCTLContext_Delete(C1);
    DPCTLQueue_Delete(Q1);
    DPCTLQueue_Delete(Q2);
    DPCTLQueue_Delete(Q3);
    DPCTLQueue_Delete(Q4);
    DPCTLDeviceSelector_Delete(FSRef);
    DPCTLDevice_Delete(DRef);
}

TEST_F(TestDPCTLSyclQueueInterface, CheckAreEq2)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOclGPU = DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_GPU);
    auto nOclCPU = DPCTLDeviceMgr_GetNumDevices(
        DPCTLSyclBackendType::DPCTL_OPENCL | DPCTL_CPU);
    if (!nOclGPU || !nOclCPU)
        GTEST_SKIP_("OpenCL GPUs and CPU not available.\n");

    auto FSRef = DPCTLFilterSelector_Create("opencl:gpu:0");
    auto DRef = DPCTLDevice_CreateFromSelector(FSRef);
    auto FSRef2 = DPCTLFilterSelector_Create("opencl:cpu:0");
    auto DRef2 = DPCTLDevice_CreateFromSelector(FSRef2);
    auto GPU_Q =
        DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    auto CPU_Q =
        DPCTLQueue_CreateForDevice(DRef2, nullptr, DPCTL_DEFAULT_PROPERTY);

    EXPECT_FALSE(DPCTLQueue_AreEq(GPU_Q, CPU_Q));

    DPCTLQueue_Delete(GPU_Q);
    DPCTLQueue_Delete(CPU_Q);
    DPCTLDeviceSelector_Delete(FSRef);
    DPCTLDevice_Delete(DRef);
    DPCTLDeviceSelector_Delete(FSRef2);
    DPCTLDevice_Delete(DRef2);
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckGetBackend)
{
    auto q = unwrap(QRef);
    auto Backend = q->get_device().get_platform().get_backend();
    auto Bty = DPCTLQueue_GetBackend(QRef);
    switch (Bty) {
    case DPCTL_CUDA:
        EXPECT_TRUE(Backend == backend::cuda);
        break;
    case DPCTL_HOST:
        EXPECT_TRUE(Backend == backend::host);
        break;
    case DPCTL_LEVEL_ZERO:
        EXPECT_TRUE(Backend == backend::level_zero);
        break;
    case DPCTL_OPENCL:
        EXPECT_TRUE(Backend == backend::opencl);
        break;
    default:
        FAIL();
    }
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckGetContext)
{
    auto Ctx = DPCTLQueue_GetContext(QRef);
    ASSERT_TRUE(Ctx != nullptr);
    DPCTLContext_Delete(Ctx);
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckGetDevice)
{
    auto D = DPCTLQueue_GetDevice(QRef);
    ASSERT_TRUE(D != nullptr);
    DPCTLDevice_Delete(D);
}

INSTANTIATE_TEST_SUITE_P(DPCTLQueueMemberFuncTests,
                         TestDPCTLQueueMemberFunctions,
                         ::testing::Values("opencl:gpu:0",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0"));

TEST_F(TestDPCTLSyclQueueInterface, CheckSubmit)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOpenCLGpuQ = DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_GPU);

    if (!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping: No OpenCL GPU device.\n");

    auto FSRef = DPCTLFilterSelector_Create("opencl:gpu:0");
    auto DRef = DPCTLDevice_CreateFromSelector(FSRef);
    auto Queue =
        DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    auto CtxRef = DPCTLQueue_GetContext(Queue);
    auto PRef =
        DPCTLProgram_CreateFromOCLSource(CtxRef, CLProgramStr, CompileOpts);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "init_arr"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));

    auto InitKernel = DPCTLProgram_GetKernel(PRef, "init_arr");
    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE, Queue);
    ASSERT_TRUE(a != nullptr);
    auto b = DPCTLmalloc_shared(SIZE, Queue);
    ASSERT_TRUE(b != nullptr);
    auto c = DPCTLmalloc_shared(SIZE, Queue);
    ASSERT_TRUE(c != nullptr);

    // Initialize a,b
    DPCTLKernelArgType argTypes[] = {DPCTL_VOID_PTR};
    size_t Range[] = {SIZE};
    void *arg1[1] = {unwrap(a)};
    void *arg2[1] = {unwrap(b)};

    auto E1 = DPCTLQueue_SubmitRange(InitKernel, Queue, arg1, argTypes, 1,
                                     Range, 1, nullptr, 0);
    auto E2 = DPCTLQueue_SubmitRange(InitKernel, Queue, arg2, argTypes, 1,
                                     Range, 1, nullptr, 0);
    ASSERT_TRUE(E1 != nullptr);
    ASSERT_TRUE(E2 != nullptr);

    DPCTLQueue_Wait(Queue);

    // Submit the add kernel
    void *args[3] = {unwrap(a), unwrap(b), unwrap(c)};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR};

    auto E3 = DPCTLQueue_SubmitRange(AddKernel, Queue, args, addKernelArgTypes,
                                     3, Range, 1, nullptr, 0);
    ASSERT_TRUE(E3 != nullptr);
    DPCTLQueue_Wait(Queue);

    // Verify the result of "add"
    add_kernel_checker((float *)a, (float *)b, (float *)c);

    // Create kernel args for axpy
    float d = 10.0;
    void *args2[4] = {unwrap(a), unwrap(b), unwrap(c), (void *)&d};
    DPCTLKernelArgType addKernelArgTypes2[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                               DPCTL_VOID_PTR, DPCTL_FLOAT};
    auto E4 = DPCTLQueue_SubmitRange(
        AxpyKernel, Queue, args2, addKernelArgTypes2, 4, Range, 1, nullptr, 0);
    ASSERT_TRUE(E4 != nullptr);
    DPCTLQueue_Wait(Queue);

    // Verify the result of "axpy"
    axpy_kernel_checker((float *)a, (float *)b, (float *)c, d);

    // clean ups
    DPCTLEvent_Delete(E1);
    DPCTLEvent_Delete(E2);
    DPCTLEvent_Delete(E3);
    DPCTLEvent_Delete(E4);

    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
    DPCTLKernel_Delete(InitKernel);

    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, Queue);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)b, Queue);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)c, Queue);

    DPCTLQueue_Delete(Queue);
    DPCTLContext_Delete(CtxRef);
    DPCTLProgram_Delete(PRef);
    DPCTLDeviceSelector_Delete(FSRef);
    DPCTLDevice_Delete(DRef);
}
