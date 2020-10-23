//===-------- test_sycl_usm_interface.cpp - dpctl-C_API ---*--- C++ --*--===//
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
/// dppl_sycl_usm_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_device_interface.h"
#include "dppl_sycl_event_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"
#include "dppl_sycl_usm_interface.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
constexpr size_t SIZE = 1024;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPPLSyclUSMRef);

bool has_devices ()
{
    bool ret = false;
    for (auto &p : platform::get_platforms()) {
        if (p.is_host())
            continue;
        if(!p.get_devices().empty()) {
            ret = true;
            break;
        }
    }
    return ret;
}

void common_test_body(size_t nbytes, const DPPLSyclUSMRef Ptr, const DPPLSyclQueueRef Q, const char *expected) {

    auto Ctx = DPPLQueue_GetContext(Q);

    auto kind = DPPLUSM_GetPointerType(Ptr, Ctx);
    EXPECT_TRUE(0 == std::strncmp(kind, expected, 4));

    auto Dev = DPPLUSM_GetPointerDevice(Ptr, Ctx);
    auto QueueDev = DPPLQueue_GetDevice(Q);
    EXPECT_TRUE(DPPLDevice_AreEq(Dev, QueueDev));

    DPPLQueue_Prefetch(Q, Ptr, nbytes);
}
    
}

struct TestDPPLSyclUSMInterface : public ::testing::Test
{

    TestDPPLSyclUSMInterface ()
    {  }

    ~TestDPPLSyclUSMInterface ()
    {  }
};

TEST_F(TestDPPLSyclUSMInterface, MallocShared)
{
    if (!has_devices())
	GTEST_SKIP_("Skipping: No Sycl Devices.\n");

    auto Q = DPPLQueueMgr_GetCurrentQueue();
    const size_t nbytes = 1024;

    auto Ptr = DPPLmalloc_shared(nbytes, Q);
    EXPECT_TRUE(bool(Ptr));

    common_test_body(nbytes, Ptr, Q, "shared");
    DPPLfree_with_queue(Ptr, Q);
}

TEST_F(TestDPPLSyclUSMInterface, MallocDevice)
{
    if (!has_devices())
	GTEST_SKIP_("Skipping: No Sycl Devices.\n");

    auto Q = DPPLQueueMgr_GetCurrentQueue();
    const size_t nbytes = 1024;

    auto Ptr = DPPLmalloc_device(nbytes, Q);
    EXPECT_TRUE(bool(Ptr));

    common_test_body(nbytes, Ptr, Q, "device");	
    DPPLfree_with_queue(Ptr, Q);
}

TEST_F(TestDPPLSyclUSMInterface, MallocHost)
{
    if (!has_devices())
	GTEST_SKIP_("Skipping: No Sycl Devices.\n");

    auto Q = DPPLQueueMgr_GetCurrentQueue();
    const size_t nbytes = 1024;

    auto Ptr = DPPLmalloc_host(nbytes, Q);
    EXPECT_TRUE(bool(Ptr));

    common_test_body(nbytes, Ptr, Q, "host");
    DPPLfree_with_queue(Ptr, Q);
}

TEST_F(TestDPPLSyclUSMInterface, AlignedAllocShared)
{
    if (!has_devices())
	GTEST_SKIP_("Skipping: No Sycl Devices.\n");

    auto Q = DPPLQueueMgr_GetCurrentQueue();
    const size_t nbytes = 1024;

    auto Ptr = DPPLaligned_alloc_shared(64, nbytes, Q);
    EXPECT_TRUE(bool(Ptr));

    common_test_body(nbytes, Ptr, Q, "shared");
    DPPLfree_with_queue(Ptr, Q);
}

TEST_F(TestDPPLSyclUSMInterface, AlignedAllocDevice)
{
    if (!has_devices())
	GTEST_SKIP_("Skipping: No Sycl Devices.\n");

    auto Q = DPPLQueueMgr_GetCurrentQueue();
    const size_t nbytes = 1024;

    auto Ptr = DPPLaligned_alloc_device(64, nbytes, Q);
    EXPECT_TRUE(bool(Ptr));

    common_test_body(nbytes, Ptr, Q, "device");
    DPPLfree_with_queue(Ptr, Q);
}

TEST_F(TestDPPLSyclUSMInterface, AlignedAllocHost)
{
    if (!has_devices())
	GTEST_SKIP_("Skipping: No Sycl Devices.\n");

    auto Q = DPPLQueueMgr_GetCurrentQueue();
    const size_t nbytes = 1024;

    auto Ptr = DPPLaligned_alloc_host(64, nbytes, Q);
    EXPECT_TRUE(bool(Ptr));

    common_test_body(nbytes, Ptr, Q, "host");
    DPPLfree_with_queue(Ptr, Q);
}

int
main (int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
