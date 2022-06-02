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
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_sycl_usm_interface.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPCTLSyclQueueRef);

void error_handler_fn(int /*err*/)
{
    return;
}

struct TestDPCTLQueueMemberFunctions
    : public ::testing::TestWithParam<
          std::tuple<const char *, DPCTLQueuePropertyType, bool>>
{
protected:
    DPCTLSyclQueueRef QRef = nullptr;

    TestDPCTLQueueMemberFunctions()
    {
        auto param_tuple = GetParam();
        auto DS = DPCTLFilterSelector_Create(std::get<0>(param_tuple));
        DPCTLSyclDeviceRef DRef = nullptr;
        if (DS) {
            EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DS));
            EXPECT_NO_FATAL_FAILURE(
                QRef = DPCTLQueue_CreateForDevice(
                    DRef,
                    (std::get<2>(param_tuple)) ? &error_handler_fn : nullptr,
                    std::get<1>(param_tuple)));
        }
        DPCTLDevice_Delete(DRef);
        DPCTLDeviceSelector_Delete(DS);
    }

    void SetUp()
    {
        if (!QRef) {
            auto param_tuple = GetParam();
            auto message = "Skipping as no device of type " +
                           std::string(std::get<0>(param_tuple)) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLQueueMemberFunctions()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));
    }
};

} /* End of anonymous namespace */

TEST(TestDPCTLSyclQueueInterface, CheckCreateForDevice)
{
    /* We are testing that we do not crash even when input is NULL. */
    DPCTLSyclQueueRef QRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(
        QRef = DPCTLQueue_CreateForDevice(nullptr, nullptr, 0));
    ASSERT_TRUE(QRef == nullptr);
}

TEST(TestDPCTLSyclQueueInterface, CheckCopy)
{
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_Create());
    EXPECT_NO_FATAL_FAILURE(
        Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    ASSERT_TRUE(Q1);
    EXPECT_NO_FATAL_FAILURE(Q2 = DPCTLQueue_Copy(Q1));
    EXPECT_TRUE(bool(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckCopyInvalid)
{
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;

    EXPECT_NO_FATAL_FAILURE(Q2 = DPCTLQueue_Copy(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
}

TEST(TestDPCTLSyclQueueInterface, CheckAreEqFalse)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(
        Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_NO_FATAL_FAILURE(
        Q2 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_FALSE(DPCTLQueue_AreEq(Q1, Q2));
    EXPECT_FALSE(DPCTLQueue_Hash(Q1) == DPCTLQueue_Hash(Q2));
    auto C0 = DPCTLQueue_GetContext(Q1);
    auto C1 = DPCTLQueue_GetContext(Q2);
    // All the queues should share the same context
    EXPECT_TRUE(DPCTLContext_AreEq(C0, C1));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(C0));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(C1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckAreEqTrue)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(
        Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_NO_FATAL_FAILURE(Q2 = DPCTLQueue_Copy(Q1));
    EXPECT_TRUE(DPCTLQueue_AreEq(Q1, Q2));
    EXPECT_TRUE(DPCTLQueue_Hash(Q1) == DPCTLQueue_Hash(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckAreEqInvalid)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;

    EXPECT_FALSE(DPCTLQueue_AreEq(Q1, Q2));
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(
        Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_FALSE(DPCTLQueue_AreEq(Q1, Q2));
    EXPECT_FALSE(DPCTLQueue_Hash(Q1) == DPCTLQueue_Hash(Q2));

    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckHashInvalid)
{
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;
    EXPECT_TRUE(DPCTLQueue_Hash(Q1) == 0);
    EXPECT_TRUE(DPCTLQueue_Hash(Q2) == 0);
}

TEST(TestDPCTLSyclQueueInterface, CheckGetBackendInvalid)
{
    DPCTLSyclQueueRef Q = nullptr;
    DPCTLSyclBackendType Bty = DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(Bty = DPCTLQueue_GetBackend(Q));
    EXPECT_TRUE(Bty == DPCTL_UNKNOWN_BACKEND);
}

TEST(TestDPCTLSyclQueueInterface, CheckGetContextInvalid)
{
    DPCTLSyclQueueRef Q = nullptr;
    DPCTLSyclContextRef CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLQueue_GetContext(Q));
    EXPECT_TRUE(CRef == nullptr);
}

TEST(TestDPCTLSyclQueueInterface, CheckGetDeviceInvalid)
{
    DPCTLSyclQueueRef Q = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLQueue_GetDevice(Q));
    EXPECT_TRUE(DRef == nullptr);
}

TEST(TestDPCTLSyclQueueInterface, CheckIsInOrder)
{
    bool ioq = true;
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(
        Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_NO_FATAL_FAILURE(ioq = DPCTLQueue_IsInOrder(Q1));
    EXPECT_FALSE(ioq);

    EXPECT_NO_FATAL_FAILURE(
        Q2 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_IN_ORDER));
    EXPECT_NO_FATAL_FAILURE(ioq = DPCTLQueue_IsInOrder(Q2));
    EXPECT_TRUE(ioq);

    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckIsInOrderInvalid)
{
    bool ioq = true;
    DPCTLSyclQueueRef Q1 = nullptr;
    EXPECT_NO_FATAL_FAILURE(ioq = DPCTLQueue_IsInOrder(Q1));
    EXPECT_FALSE(ioq);
}

TEST(TestDPCTLSyclQueueInterface, CheckHasEnableProfiling)
{
    bool ioq = true;
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef Q1 = nullptr;
    DPCTLSyclQueueRef Q2 = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(
        Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_NO_FATAL_FAILURE(ioq = DPCTLQueue_HasEnableProfiling(Q1));
    EXPECT_FALSE(ioq);

    EXPECT_NO_FATAL_FAILURE(
        Q2 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_ENABLE_PROFILING));
    EXPECT_NO_FATAL_FAILURE(ioq = DPCTLQueue_HasEnableProfiling(Q2));
    EXPECT_TRUE(ioq);

    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q1));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(Q2));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckHasEnableProfilingInvalid)
{
    bool ioq = true;
    DPCTLSyclQueueRef Q1 = nullptr;
    EXPECT_NO_FATAL_FAILURE(ioq = DPCTLQueue_HasEnableProfiling(Q1));
    EXPECT_FALSE(ioq);
}

TEST(TestDPCTLSyclQueueInterface, CheckPropertyHandling)
{
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));

    ::testing::internal::CaptureStderr();
    EXPECT_NO_FATAL_FAILURE(QRef = DPCTLQueue_CreateForDevice(
                                DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    std::string capt1 = ::testing::internal::GetCapturedStderr();
    ASSERT_TRUE(capt1.empty());
    ASSERT_FALSE(DPCTLQueue_IsInOrder(QRef));
    ASSERT_FALSE(DPCTLQueue_HasEnableProfiling(QRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));

    QRef = nullptr;
    ::testing::internal::CaptureStderr();
    int invalid_prop = -1;
    EXPECT_NO_FATAL_FAILURE(
        QRef = DPCTLQueue_CreateForDevice(DRef, nullptr, invalid_prop));
    std::string capt2 = ::testing::internal::GetCapturedStderr();
    ASSERT_TRUE(!capt2.empty());
    ASSERT_TRUE(DPCTLQueue_IsInOrder(QRef) ==
                bool((invalid_prop & DPCTL_IN_ORDER)));
    ASSERT_TRUE(DPCTLQueue_HasEnableProfiling(QRef) ==
                bool((invalid_prop & DPCTL_ENABLE_PROFILING)));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST(TestDPCTLSyclQueueInterface, CheckMemOpsZeroQRef)
{
    DPCTLSyclQueueRef QRef = nullptr;
    void *p1 = nullptr;
    void *p2 = nullptr;
    size_t n_bytes = 0;
    DPCTLSyclEventRef ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Memcpy(QRef, p1, p2, n_bytes));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Prefetch(QRef, p1, n_bytes));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_MemAdvise(QRef, p1, n_bytes, 0));
    ASSERT_FALSE(bool(ERef));
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckGetBackend)
{
    auto q = unwrap(QRef);
    auto Backend = q->get_device().get_platform().get_backend();
    auto Bty = DPCTLQueue_GetBackend(QRef);
    switch (Bty) {
    case DPCTL_CUDA:
        EXPECT_TRUE(Backend == backend::ext_oneapi_cuda);
        break;
    case DPCTL_HOST:
        EXPECT_TRUE(Backend == backend::host);
        break;
    case DPCTL_LEVEL_ZERO:
        EXPECT_TRUE(Backend == backend::ext_oneapi_level_zero);
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
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(Ctx));
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckGetDevice)
{
    auto D = DPCTLQueue_GetDevice(QRef);
    ASSERT_TRUE(D != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(D));
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckMemOpsNullPtr)
{
    void *p1 = nullptr;
    void *p2 = nullptr;
    size_t n_bytes = 256;
    DPCTLSyclEventRef ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Memcpy(QRef, p1, p2, n_bytes));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Prefetch(QRef, p1, n_bytes));
    if (ERef) {
        ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
        ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
        ERef = nullptr;
    }

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_MemAdvise(QRef, p1, n_bytes, 0));
    if (ERef) {
        ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
        ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
        ERef = nullptr;
    }
}

TEST(TestDPCTLSyclQueueInterface, CheckMemsetNullQRef)
{
    DPCTLSyclQueueRef QRef = nullptr;
    void *p = nullptr;
    uint8_t val8 = 0;
    DPCTLSyclEventRef ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Memset(QRef, p, val8, 1));
    ASSERT_FALSE(bool(ERef));
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckMemset)
{
    DPCTLSyclUSMRef p = nullptr;
    DPCTLSyclEventRef ERef = nullptr;
    uint8_t val = 73;
    size_t nbytes = 256;
    uint8_t *host_arr = new uint8_t[nbytes];

    ASSERT_FALSE(host_arr == nullptr);

    ASSERT_NO_FATAL_FAILURE(p = DPCTLmalloc_device(nbytes, QRef));
    ASSERT_FALSE(p == nullptr);

    ASSERT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_Memset(QRef, (void *)p, val, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Memcpy(QRef, host_arr, p, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ASSERT_NO_FATAL_FAILURE(DPCTLfree_with_queue(p, QRef));

    for (size_t i = 0; i < nbytes; ++i) {
        ASSERT_TRUE(host_arr[i] == val);
    }
    delete[] host_arr;
}

TEST(TestDPCTLSyclQueueInterface, CheckFillNullQRef)
{
    DPCTLSyclQueueRef QRef = nullptr;
    void *p = nullptr;
    uint8_t val8 = 0;
    uint16_t val16 = 0;
    uint32_t val32 = 0;
    uint64_t val64 = 0;
    uint64_t val128[2] = {0, 0};
    DPCTLSyclEventRef ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Fill8(QRef, p, val8, 1));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Fill16(QRef, p, val16, 1));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Fill32(QRef, p, val32, 1));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Fill64(QRef, p, val64, 1));
    ASSERT_FALSE(bool(ERef));

    ASSERT_NO_FATAL_FAILURE(ERef = DPCTLQueue_Fill128(QRef, p, val128, 1));
    ASSERT_FALSE(bool(ERef));
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckFill8)
{
    using T = uint8_t;
    DPCTLSyclUSMRef p = nullptr;
    DPCTLSyclEventRef ERef = nullptr;
    T val = static_cast<T>(0xB);
    size_t nelems = 256;
    T *host_arr = new T[nelems];
    size_t nbytes = nelems * sizeof(T);

    ASSERT_FALSE(host_arr == nullptr);

    ASSERT_NO_FATAL_FAILURE(p = DPCTLmalloc_device(nbytes, QRef));
    ASSERT_FALSE(p == nullptr);

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Fill8(QRef, (void *)p, val, nelems));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Memcpy(QRef, host_arr, p, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ASSERT_NO_FATAL_FAILURE(DPCTLfree_with_queue(p, QRef));

    for (size_t i = 0; i < nelems; ++i) {
        ASSERT_TRUE(host_arr[i] == val);
    }
    delete[] host_arr;
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckFill16)
{
    using T = uint16_t;

    DPCTLSyclUSMRef p = nullptr;
    DPCTLSyclEventRef ERef = nullptr;
    T val = static_cast<T>(0xAB);
    size_t nelems = 256;
    T *host_arr = new T[nelems];
    size_t nbytes = nelems * sizeof(T);

    ASSERT_FALSE(host_arr == nullptr);

    ASSERT_NO_FATAL_FAILURE(p = DPCTLmalloc_device(nbytes, QRef));
    ASSERT_FALSE(p == nullptr);

    ASSERT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_Fill16(QRef, (void *)p, val, nelems));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Memcpy(QRef, host_arr, p, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ASSERT_NO_FATAL_FAILURE(DPCTLfree_with_queue(p, QRef));

    for (size_t i = 0; i < nelems; ++i) {
        ASSERT_TRUE(host_arr[i] == val);
    }
    delete[] host_arr;
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckFill32)
{
    using T = uint32_t;

    DPCTLSyclUSMRef p = nullptr;
    DPCTLSyclEventRef ERef = nullptr;
    T val = static_cast<T>(0xABCD);
    size_t nelems = 256;
    T *host_arr = new T[nelems];
    size_t nbytes = nelems * sizeof(T);

    ASSERT_FALSE(host_arr == nullptr);

    ASSERT_NO_FATAL_FAILURE(p = DPCTLmalloc_device(nbytes, QRef));
    ASSERT_FALSE(p == nullptr);

    ASSERT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_Fill32(QRef, (void *)p, val, nelems));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Memcpy(QRef, host_arr, p, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ASSERT_NO_FATAL_FAILURE(DPCTLfree_with_queue(p, QRef));

    for (size_t i = 0; i < nelems; ++i) {
        ASSERT_TRUE(host_arr[i] == val);
    }
    delete[] host_arr;
}

TEST_P(TestDPCTLQueueMemberFunctions, CheckFill64)
{
    using T = uint64_t;

    DPCTLSyclUSMRef p = nullptr;
    DPCTLSyclEventRef ERef = nullptr;
    T val = static_cast<T>(0xABCDEF73);
    size_t nelems = 256;
    T *host_arr = new T[nelems];
    size_t nbytes = nelems * sizeof(T);

    ASSERT_FALSE(host_arr == nullptr);

    ASSERT_NO_FATAL_FAILURE(p = DPCTLmalloc_device(nbytes, QRef));
    ASSERT_FALSE(p == nullptr);

    ASSERT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_Fill64(QRef, (void *)p, val, nelems));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Memcpy(QRef, host_arr, p, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ASSERT_NO_FATAL_FAILURE(DPCTLfree_with_queue(p, QRef));

    for (size_t i = 0; i < nelems; ++i) {
        ASSERT_TRUE(host_arr[i] == val);
    }
    delete[] host_arr;
}

namespace
{
struct value128_t
{
    uint64_t first;
    uint64_t second;

    value128_t() : first(0), second(0) {}
    value128_t(uint64_t v0, uint64_t v1) : first(v0), second(v1) {}
};

static_assert(sizeof(value128_t) == 2 * sizeof(uint64_t));
} // namespace

TEST_P(TestDPCTLQueueMemberFunctions, CheckFill128)
{
    using T = value128_t;

    DPCTLSyclUSMRef p = nullptr;
    DPCTLSyclEventRef ERef = nullptr;
    T val{static_cast<uint64_t>(0xABCDEF73), static_cast<uint64_t>(0x3746AF05)};
    size_t nelems = 256;
    T *host_arr = new T[nelems];
    size_t nbytes = nelems * sizeof(T);

    ASSERT_FALSE(host_arr == nullptr);

    ASSERT_NO_FATAL_FAILURE(p = DPCTLmalloc_device(nbytes, QRef));
    ASSERT_FALSE(p == nullptr);

    ASSERT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_Fill128(QRef, (void *)p,
                                  reinterpret_cast<uint64_t *>(&val), nelems));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ERef = nullptr;

    ASSERT_NO_FATAL_FAILURE(ERef =
                                DPCTLQueue_Memcpy(QRef, host_arr, p, nbytes));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    ASSERT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));

    ASSERT_NO_FATAL_FAILURE(DPCTLfree_with_queue(p, QRef));

    for (size_t i = 0; i < nelems; ++i) {
        ASSERT_TRUE(host_arr[i].first == val.first);
        ASSERT_TRUE(host_arr[i].second == val.second);
    }
    delete[] host_arr;
}

INSTANTIATE_TEST_SUITE_P(
    DPCTLQueueMemberFuncTests,
    TestDPCTLQueueMemberFunctions,
    ::testing::Combine(
        ::testing::Values("opencl:gpu", "opencl:cpu", "level_zero:gpu"),
        ::testing::Values(DPCTL_DEFAULT_PROPERTY,
                          DPCTL_ENABLE_PROFILING,
                          DPCTL_IN_ORDER,
                          static_cast<DPCTLQueuePropertyType>(
                              DPCTL_ENABLE_PROFILING | DPCTL_IN_ORDER)),
        ::testing::Bool()));
