//===-- test_sycl_queue_submit.cpp - Test cases for kernel submission fns. ===//
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
/// This file has unit test cases for the various submit functions defined
/// inside dpctl_sycl_queue_interface.cpp.
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_kernel_bundle_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_usm_interface.h"
#include <CL/sycl.hpp>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace
{
constexpr size_t SIZE = 1024;
static_assert(SIZE % 8 == 0);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPCTLSyclUSMRef);
} /* end of anonymous namespace */

struct TestQueueSubmit : public ::testing::Test
{
    std::ifstream spirvFile;
    size_t spirvFileSize;
    std::vector<char> spirvBuffer;

    TestQueueSubmit()
    {
        spirvFile.open("./multi_kernel.spv", std::ios::binary | std::ios::ate);
        spirvFileSize = std::filesystem::file_size("./multi_kernel.spv");
        spirvBuffer.reserve(spirvFileSize);
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer.data(), spirvFileSize);
    }

    ~TestQueueSubmit()
    {
        spirvFile.close();
    }
};

TEST_F(TestQueueSubmit, CheckSubmitRange_saxpy)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    DPCTLDeviceMgr_PrintDeviceInfo(DRef);
    ASSERT_TRUE(DRef);
    auto QRef =
        DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    ASSERT_TRUE(QRef);
    auto CRef = DPCTLQueue_GetContext(QRef);
    ASSERT_TRUE(CRef);
    auto KBRef = DPCTLKernelBundle_CreateFromSpirv(
        CRef, DRef, spirvBuffer.data(), spirvFileSize, nullptr);
    ASSERT_TRUE(KBRef != nullptr);
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, "axpy"));
    auto AxpyKernel = DPCTLKernelBundle_GetKernel(KBRef, "axpy");

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(a != nullptr);
    auto b = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(b != nullptr);
    auto c = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(c != nullptr);

    auto a_ptr = reinterpret_cast<float *>(unwrap(a));
    auto b_ptr = reinterpret_cast<float *>(unwrap(b));
    // Initialize a,b
    for (auto i = 0ul; i < SIZE; ++i) {
        a_ptr[i] = i + 1.0;
        b_ptr[i] = i + 2.0;
    }

    // Create kernel args for axpy
    float d = 10.0;
    size_t Range[] = {SIZE};
    void *args2[4] = {unwrap(a), unwrap(b), unwrap(c), (void *)&d};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR, DPCTL_FLOAT};
    auto ERef = DPCTLQueue_SubmitRange(
        AxpyKernel, QRef, args2, addKernelArgTypes, 4, Range, 1, nullptr, 0);
    ASSERT_TRUE(ERef != nullptr);
    DPCTLQueue_Wait(QRef);

    // clean ups
    DPCTLEvent_Delete(ERef);
    DPCTLKernel_Delete(AxpyKernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)b, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)c, QRef);
    DPCTLQueue_Delete(QRef);
    DPCTLContext_Delete(CRef);
    DPCTLKernelBundle_Delete(KBRef);
    DPCTLDevice_Delete(DRef);
    DPCTLDeviceSelector_Delete(DSRef);
}

TEST_F(TestQueueSubmit, CheckSubmitNDRange_saxpy)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    DPCTLDeviceMgr_PrintDeviceInfo(DRef);
    ASSERT_TRUE(DRef);
    auto QRef =
        DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    ASSERT_TRUE(QRef);
    auto CRef = DPCTLQueue_GetContext(QRef);
    ASSERT_TRUE(CRef);
    auto KBRef = DPCTLKernelBundle_CreateFromSpirv(
        CRef, DRef, spirvBuffer.data(), spirvFileSize, nullptr);
    ASSERT_TRUE(KBRef != nullptr);
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, "axpy"));
    auto AxpyKernel = DPCTLKernelBundle_GetKernel(KBRef, "axpy");

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(a != nullptr);
    auto b = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(b != nullptr);
    auto c = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(c != nullptr);

    auto a_ptr = reinterpret_cast<float *>(unwrap(a));
    auto b_ptr = reinterpret_cast<float *>(unwrap(b));
    // Initialize a,b
    for (auto i = 0ul; i < SIZE; ++i) {
        a_ptr[i] = i + 1.0;
        b_ptr[i] = i + 2.0;
    }

    // Create kernel args for axpy
    float d = 10.0;
    size_t gRange[] = {1, 1, SIZE};
    size_t lRange[] = {1, 1, 8};
    void *args2[4] = {unwrap(a), unwrap(b), unwrap(c), (void *)&d};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR, DPCTL_FLOAT};
    DPCTLSyclEventRef events[1];
    events[0] = DPCTLEvent_Create();

    auto ERef =
        DPCTLQueue_SubmitNDRange(AxpyKernel, QRef, args2, addKernelArgTypes, 4,
                                 gRange, lRange, 3, events, 1);
    ASSERT_TRUE(ERef != nullptr);
    DPCTLQueue_Wait(QRef);

    // clean ups
    DPCTLEvent_Delete(ERef);
    DPCTLKernel_Delete(AxpyKernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)b, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)c, QRef);
    DPCTLQueue_Delete(QRef);
    DPCTLContext_Delete(CRef);
    DPCTLKernelBundle_Delete(KBRef);
    DPCTLDevice_Delete(DRef);
    DPCTLDeviceSelector_Delete(DSRef);
}

struct TestQueueSubmitBarrier : public ::testing::Test
{
    DPCTLSyclQueueRef QRef = nullptr;

    TestQueueSubmitBarrier()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;

        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        EXPECT_NO_FATAL_FAILURE(QRef = DPCTLQueue_CreateForDevice(
                                    DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
    ~TestQueueSubmitBarrier()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));
    }
};

TEST_F(TestQueueSubmitBarrier, ChkSubmitBarrier)
{
    DPCTLSyclEventRef ERef = nullptr;

    ASSERT_TRUE(QRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(ERef = DPCTLQueue_SubmitBarrier(QRef));
    ASSERT_TRUE(ERef != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
}

TEST_F(TestQueueSubmitBarrier, ChkSubmitBarrierWithEvents)
{
    DPCTLSyclEventRef ERef = nullptr;
    DPCTLSyclEventRef DepsERefs[2] = {nullptr, nullptr};

    EXPECT_NO_FATAL_FAILURE(DepsERefs[0] = DPCTLEvent_Create());
    EXPECT_NO_FATAL_FAILURE(DepsERefs[1] = DPCTLEvent_Create());

    ASSERT_TRUE(QRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_SubmitBarrierForEvents(QRef, DepsERefs, 2));

    ASSERT_TRUE(ERef != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(DepsERefs[0]));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(DepsERefs[1]));
}
