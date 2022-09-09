//===---------- test_sycl_usm_interface.cpp - Test cases for USM interface ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// dpctl_sycl_usm_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_sycl_usm_interface.h"
#include <CL/sycl.hpp>
#include <cstring>
#include <gtest/gtest.h>

using namespace sycl;

namespace
{
constexpr size_t SIZE = 1024;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPCTLSyclUSMRef);

void common_test_body(size_t nbytes,
                      const DPCTLSyclUSMRef Ptr,
                      const DPCTLSyclQueueRef Q,
                      const char *expected)
{
    auto Ctx = DPCTLQueue_GetContext(Q);

    auto kind = DPCTLUSM_GetPointerType(Ptr, Ctx);
    EXPECT_TRUE(0 == std::strncmp(kind, expected, 4));

    auto Dev = DPCTLUSM_GetPointerDevice(Ptr, Ctx);
    auto QueueDev = DPCTLQueue_GetDevice(Q);
    EXPECT_TRUE(DPCTLDevice_AreEq(Dev, QueueDev));

    DPCTLSyclEventRef E1Ref = nullptr, E2Ref = nullptr, E3Ref = nullptr;
    EXPECT_NO_FATAL_FAILURE(E1Ref = DPCTLQueue_Prefetch(Q, Ptr, nbytes));
    EXPECT_TRUE(E1Ref != nullptr);
    EXPECT_NO_FATAL_FAILURE(E2Ref = DPCTLQueue_MemAdvise(Q, Ptr, nbytes, 0));
    EXPECT_TRUE(E2Ref != nullptr);

    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(E1Ref));
    DPCTLEvent_Delete(E1Ref);
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(E2Ref));
    DPCTLEvent_Delete(E2Ref);
    try {
        unsigned short *host_ptr = new unsigned short[nbytes];
        EXPECT_NO_FATAL_FAILURE(
            E3Ref = DPCTLQueue_Memcpy(Q, host_ptr, Ptr, nbytes));
        EXPECT_TRUE(E3Ref != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(E3Ref));
        DPCTLEvent_Delete(E3Ref);
        delete[] host_ptr;
    } catch (std::bad_alloc const &ba) {
        // pass
    }

    DPCTLDevice_Delete(QueueDev);
    DPCTLDevice_Delete(Dev);
    DPCTLContext_Delete(Ctx);
}

} // end of namespace

struct TestDPCTLSyclUSMInterface : public ::testing::Test
{

    TestDPCTLSyclUSMInterface() {}

    ~TestDPCTLSyclUSMInterface() {}
};

TEST_F(TestDPCTLSyclUSMInterface, MallocShared)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(Q);
    const size_t nbytes = SIZE;
    auto Ptr = DPCTLmalloc_shared(nbytes, Q);
    EXPECT_TRUE(bool(Ptr));
    common_test_body(nbytes, Ptr, Q, "shared");
    DPCTLfree_with_queue(Ptr, Q);
    DPCTLQueue_Delete(Q);
}

TEST_F(TestDPCTLSyclUSMInterface, MallocDevice)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(Q);
    const size_t nbytes = SIZE;
    auto Ptr = DPCTLmalloc_device(nbytes, Q);
    EXPECT_TRUE(bool(Ptr));
    common_test_body(nbytes, Ptr, Q, "device");
    DPCTLfree_with_queue(Ptr, Q);
    DPCTLQueue_Delete(Q);
}

TEST_F(TestDPCTLSyclUSMInterface, MallocHost)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(Q);
    const size_t nbytes = SIZE;
    auto Ptr = DPCTLmalloc_host(nbytes, Q);
    EXPECT_TRUE(bool(Ptr));
    common_test_body(nbytes, Ptr, Q, "host");
    DPCTLfree_with_queue(Ptr, Q);
    DPCTLQueue_Delete(Q);
}

TEST_F(TestDPCTLSyclUSMInterface, AlignedAllocShared)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(Q);
    const size_t nbytes = SIZE;
    auto Ptr = DPCTLaligned_alloc_shared(64, nbytes, Q);
    EXPECT_TRUE(bool(Ptr));
    common_test_body(nbytes, Ptr, Q, "shared");
    DPCTLfree_with_queue(Ptr, Q);
    DPCTLQueue_Delete(Q);
}

TEST_F(TestDPCTLSyclUSMInterface, AlignedAllocDevice)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(Q);
    const size_t nbytes = SIZE;
    auto Ptr = DPCTLaligned_alloc_device(64, nbytes, Q);
    EXPECT_TRUE(bool(Ptr));
    common_test_body(nbytes, Ptr, Q, "device");
    DPCTLfree_with_queue(Ptr, Q);
    DPCTLQueue_Delete(Q);
}

TEST_F(TestDPCTLSyclUSMInterface, AlignedAllocHost)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(Q);
    const size_t nbytes = SIZE;
    auto Ptr = DPCTLaligned_alloc_host(64, nbytes, Q);
    EXPECT_TRUE(bool(Ptr));
    common_test_body(nbytes, Ptr, Q, "host");
    DPCTLfree_with_queue(Ptr, Q);
}

struct TestDPCTLSyclUSMNullArgs : public ::testing::Test
{
};

TEST_F(TestDPCTLSyclUSMNullArgs, ChkMalloc)
{
    DPCTLSyclQueueRef Null_QRef = nullptr;
    void *ptr = nullptr;

    EXPECT_NO_FATAL_FAILURE(ptr = DPCTLmalloc_shared(512, Null_QRef));
    ASSERT_TRUE(ptr == nullptr);

    EXPECT_NO_FATAL_FAILURE(ptr = DPCTLmalloc_device(512, Null_QRef));
    ASSERT_TRUE(ptr == nullptr);

    EXPECT_NO_FATAL_FAILURE(ptr = DPCTLmalloc_host(512, Null_QRef));
    ASSERT_TRUE(ptr == nullptr);
}

TEST_F(TestDPCTLSyclUSMNullArgs, ChkAlignedAlloc)
{
    DPCTLSyclQueueRef Null_QRef = nullptr;
    void *ptr = nullptr;

    EXPECT_NO_FATAL_FAILURE(ptr =
                                DPCTLaligned_alloc_shared(64, 512, Null_QRef));
    ASSERT_TRUE(ptr == nullptr);

    EXPECT_NO_FATAL_FAILURE(ptr =
                                DPCTLaligned_alloc_device(64, 512, Null_QRef));
    ASSERT_TRUE(ptr == nullptr);

    EXPECT_NO_FATAL_FAILURE(ptr = DPCTLaligned_alloc_host(64, 512, Null_QRef));
    ASSERT_TRUE(ptr == nullptr);
}

TEST_F(TestDPCTLSyclUSMNullArgs, ChkFree)
{
    DPCTLSyclQueueRef Null_QRef = nullptr;
    DPCTLSyclUSMRef ptr = nullptr;

    EXPECT_NO_FATAL_FAILURE(DPCTLfree_with_queue(ptr, Null_QRef));

    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    EXPECT_NO_FATAL_FAILURE(QRef = DPCTLQueue_CreateForDevice(
                                DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));

    EXPECT_NO_FATAL_FAILURE(DPCTLfree_with_queue(ptr, QRef));

    DPCTLSyclContextRef Null_CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DPCTLfree_with_context(ptr, Null_CRef));

    DPCTLSyclContextRef CRef = DPCTLQueue_GetContext(QRef);
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLfree_with_context(ptr, CRef));

    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_F(TestDPCTLSyclUSMNullArgs, ChkPointerQueries)
{
    DPCTLSyclContextRef Null_CRef = nullptr;
    DPCTLSyclUSMRef Null_MRef = nullptr;
    const char *t = nullptr;
    auto is_unknown = [=](const char *t) -> bool {
        return strncmp(t, "unknown", 7) == 0;
    };

    EXPECT_NO_FATAL_FAILURE(t = DPCTLUSM_GetPointerType(Null_MRef, Null_CRef));
    ASSERT_TRUE(is_unknown(t));

    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    ASSERT_TRUE(bool(DRef));

    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));

    t = nullptr;
    EXPECT_NO_FATAL_FAILURE(t = DPCTLUSM_GetPointerType(Null_MRef, CRef));
    ASSERT_TRUE(is_unknown(t));

    DPCTLSyclDeviceRef D2Ref = nullptr;
    EXPECT_NO_FATAL_FAILURE(D2Ref = DPCTLUSM_GetPointerDevice(Null_MRef, CRef));
    ASSERT_TRUE(D2Ref == nullptr);

    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));

    EXPECT_NO_FATAL_FAILURE(
        D2Ref = DPCTLUSM_GetPointerDevice(Null_MRef, Null_CRef));
    ASSERT_TRUE(D2Ref == nullptr);
}
