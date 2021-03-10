//===------ test_sycl_queue_interface.cpp - Test cases for queue interface ===//
//
//                      Data Parallel Control (dpCtl)
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

} // namespace

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

TEST_F(TestDPCTLSyclQueueInterface, CheckAreEq)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOclGPU = DPCTLQueueMgr_GetNumQueues(
        DPCTLSyclBackendType::DPCTL_OPENCL, DPCTLSyclDeviceType::DPCTL_GPU);
    if (!nOclGPU)
        GTEST_SKIP_("Skipping: No OpenCL GPUs available.\n");

    auto Q1 = DPCTLQueueMgr_GetCurrentQueue();
    auto Q2 = DPCTLQueueMgr_GetCurrentQueue();
    EXPECT_TRUE(DPCTLQueue_AreEq(Q1, Q2));

    auto Def_Q = DPCTLQueueMgr_SetAsDefaultQueue(
        DPCTLSyclBackendType::DPCTL_OPENCL, DPCTLSyclDeviceType::DPCTL_GPU, 0);
    auto OclGPU_Q0 = DPCTLQueueMgr_PushQueue(DPCTLSyclBackendType::DPCTL_OPENCL,
                                             DPCTLSyclDeviceType::DPCTL_GPU, 0);
    auto OclGPU_Q1 = DPCTLQueueMgr_PushQueue(DPCTLSyclBackendType::DPCTL_OPENCL,
                                             DPCTLSyclDeviceType::DPCTL_GPU, 0);
    EXPECT_TRUE(DPCTLQueue_AreEq(Def_Q, OclGPU_Q0));
    EXPECT_TRUE(DPCTLQueue_AreEq(Def_Q, OclGPU_Q1));
    EXPECT_TRUE(DPCTLQueue_AreEq(OclGPU_Q0, OclGPU_Q1));
    DPCTLQueue_Delete(Def_Q);
    DPCTLQueue_Delete(OclGPU_Q0);
    DPCTLQueue_Delete(OclGPU_Q1);
    DPCTLQueueMgr_PopQueue();
    DPCTLQueueMgr_PopQueue();
}

TEST_F(TestDPCTLSyclQueueInterface, CheckAreEq2)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOclGPU = DPCTLQueueMgr_GetNumQueues(
        DPCTLSyclBackendType::DPCTL_OPENCL, DPCTLSyclDeviceType::DPCTL_GPU);
    auto nOclCPU = DPCTLQueueMgr_GetNumQueues(
        DPCTLSyclBackendType::DPCTL_OPENCL, DPCTLSyclDeviceType::DPCTL_CPU);
    if (!nOclGPU || !nOclCPU)
        GTEST_SKIP_("OpenCL GPUs and CPU not available.\n");
    auto GPU_Q = DPCTLQueueMgr_PushQueue(DPCTLSyclBackendType::DPCTL_OPENCL,
                                         DPCTLSyclDeviceType::DPCTL_GPU, 0);
    auto CPU_Q = DPCTLQueueMgr_PushQueue(DPCTLSyclBackendType::DPCTL_OPENCL,
                                         DPCTLSyclDeviceType::DPCTL_CPU, 0);
    EXPECT_FALSE(DPCTLQueue_AreEq(GPU_Q, CPU_Q));
    DPCTLQueueMgr_PopQueue();
    DPCTLQueueMgr_PopQueue();
}

TEST_F(TestDPCTLSyclQueueInterface, CheckGetBackend)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto Q1 = DPCTLQueueMgr_GetCurrentQueue();
    auto BE = DPCTLQueue_GetBackend(Q1);
    EXPECT_TRUE((BE == DPCTL_OPENCL) || (BE == DPCTL_LEVEL_ZERO) ||
                (BE == DPCTL_CUDA) || (BE == DPCTL_HOST));
    DPCTLQueue_Delete(Q1);
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
        EXPECT_TRUE(DPCTLQueue_GetBackend(Q) == DPCTL_OPENCL);
        DPCTLQueue_Delete(Q);
        DPCTLQueueMgr_PopQueue();
    }
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
        EXPECT_TRUE(DPCTLQueue_GetBackend(Q) == DPCTL_OPENCL);
        DPCTLQueue_Delete(Q);
        DPCTLQueueMgr_PopQueue();
    }
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_LEVEL_ZERO, DPCTL_GPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU, 0);
        EXPECT_TRUE(DPCTLQueue_GetBackend(Q) == DPCTL_LEVEL_ZERO);
        DPCTLQueue_Delete(Q);
        DPCTLQueueMgr_PopQueue();
    }
}

TEST_F(TestDPCTLSyclQueueInterface, CheckGetContext)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto Q1 = DPCTLQueueMgr_GetCurrentQueue();
    auto Ctx = DPCTLQueue_GetContext(Q1);
    ASSERT_TRUE(Ctx != nullptr);
    DPCTLQueue_Delete(Q1);
    DPCTLContext_Delete(Ctx);

    if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
        auto OclGpuCtx = DPCTLQueue_GetContext(Q);
        ASSERT_TRUE(OclGpuCtx != nullptr);
        DPCTLQueue_Delete(Q);
        DPCTLContext_Delete(OclGpuCtx);
        DPCTLQueueMgr_PopQueue();
    }
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
        auto OclCpuCtx = DPCTLQueue_GetContext(Q);
        ASSERT_TRUE(OclCpuCtx != nullptr);
        DPCTLQueue_Delete(Q);
        DPCTLContext_Delete(OclCpuCtx);
        DPCTLQueueMgr_PopQueue();
    }
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_LEVEL_ZERO, DPCTL_GPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU, 0);
        auto L0Ctx = DPCTLQueue_GetContext(Q);
        ASSERT_TRUE(Ctx != nullptr);
        DPCTLQueue_Delete(Q);
        DPCTLContext_Delete(L0Ctx);
        DPCTLQueueMgr_PopQueue();
    }
}

TEST_F(TestDPCTLSyclQueueInterface, CheckGetDevice)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto Q1 = DPCTLQueueMgr_GetCurrentQueue();
    auto D = DPCTLQueue_GetDevice(Q1);
    ASSERT_TRUE(D != nullptr);
    DPCTLQueue_Delete(Q1);
    DPCTLDevice_Delete(D);

    if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
        auto OCLGPU_D = DPCTLQueue_GetDevice(Q);
        ASSERT_TRUE(OCLGPU_D != nullptr);
        EXPECT_TRUE(DPCTLDevice_IsGPU(OCLGPU_D));
        DPCTLQueue_Delete(Q);
        DPCTLDevice_Delete(OCLGPU_D);
        DPCTLQueueMgr_PopQueue();
    }
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
        auto OCLCPU_D = DPCTLQueue_GetDevice(Q);
        ASSERT_TRUE(OCLCPU_D != nullptr);
        EXPECT_TRUE(DPCTLDevice_IsCPU(OCLCPU_D));
        DPCTLQueue_Delete(Q);
        DPCTLDevice_Delete(OCLCPU_D);
        DPCTLQueueMgr_PopQueue();
    }
    if (DPCTLQueueMgr_GetNumQueues(DPCTL_LEVEL_ZERO, DPCTL_GPU)) {
        auto Q = DPCTLQueueMgr_PushQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU, 0);
        auto L0GPU_D = DPCTLQueue_GetDevice(Q);
        ASSERT_TRUE(L0GPU_D != nullptr);
        EXPECT_TRUE(DPCTLDevice_IsGPU(L0GPU_D));
        DPCTLQueue_Delete(Q);
        DPCTLDevice_Delete(L0GPU_D);
        DPCTLQueueMgr_PopQueue();
    }
}

TEST_F(TestDPCTLSyclQueueInterface, CheckSubmit)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOpenCLGpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU);

    if (!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping: No OpenCL GPU device.\n");

    auto Queue = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
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
}
