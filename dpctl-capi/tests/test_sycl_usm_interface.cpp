//===---------- test_sycl_usm_interface.cpp - Test cases for USM interface ===//
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
/// dpctl_sycl_usm_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_event_interface.h"
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

    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Prefetch(Q, Ptr, nbytes));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_MemAdvise(Q, Ptr, nbytes, 0));

    try {
        unsigned short *host_ptr = new unsigned short[nbytes];
        EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Memcpy(Q, host_ptr, Ptr, nbytes));
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
