//===------- test_sycl_queue_manager.cpp - Test cases for queue manager    ===//
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
/// dpctl_sycl_queue_interface.h and dpctl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <thread>

using namespace std;
using namespace cl::sycl;

namespace
{

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPCTLSyclQueueRef);

void foo(size_t &num)
{
    auto DS1 = DPCTLFilterSelector_Create("opencl:gpu");
    auto DS2 = DPCTLFilterSelector_Create("opencl:cpu");
    auto D1 = DPCTLDevice_CreateFromSelector(DS1);
    auto D2 = DPCTLDevice_CreateFromSelector(DS2);
    auto Q1 = DPCTLQueue_CreateForDevice(D1, nullptr, DPCTL_DEFAULT_PROPERTY);
    auto Q2 = DPCTLQueue_CreateForDevice(D2, nullptr, DPCTL_DEFAULT_PROPERTY);
    DPCTLQueueMgr_PushQueue(Q2);
    DPCTLQueueMgr_PushQueue(Q1);

    // Capture the number of active queues in first
    num = DPCTLQueueMgr_GetQueueStackSize();
    DPCTLQueueMgr_PopQueue();
    DPCTLQueueMgr_PopQueue();
    DPCTLQueue_Delete(Q1);
    DPCTLQueue_Delete(Q2);
    DPCTLDeviceSelector_Delete(DS1);
    DPCTLDeviceSelector_Delete(DS2);
    DPCTLDevice_Delete(D1);
    DPCTLDevice_Delete(D2);
}

void bar(size_t &num)
{
    auto DS1 = DPCTLFilterSelector_Create("opencl:gpu");
    auto D1 = DPCTLDevice_CreateFromSelector(DS1);
    auto Q1 = DPCTLQueue_CreateForDevice(D1, nullptr, DPCTL_DEFAULT_PROPERTY);
    DPCTLQueueMgr_PushQueue(Q1);
    // Capture the number of active queues in second
    num = DPCTLQueueMgr_GetQueueStackSize();
    DPCTLQueueMgr_PopQueue();
    DPCTLQueue_Delete(Q1);
    DPCTLDeviceSelector_Delete(DS1);
    DPCTLDevice_Delete(D1);
}

} /* end of anonymous namespace */

struct TestDPCTLSyclQueueManager : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = DPCTLFilterSelector_Create(GetParam());
    DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef);

    TestDPCTLSyclQueueManager()
    {
        DSRef = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLSyclQueueManager()
    {
        DPCTLDeviceSelector_Delete(DSRef);
        DPCTLDevice_Delete(DRef);
    }
};

TEST_P(TestDPCTLSyclQueueManager, CheckDPCTLGetCurrentQueue)
{
    DPCTLSyclQueueRef q = DPCTLQueueMgr_GetCurrentQueue();
    ASSERT_TRUE(q != nullptr);
}

TEST_P(TestDPCTLSyclQueueManager, CheckIsCurrentQueue)
{
    auto Q0 = DPCTLQueueMgr_GetCurrentQueue();
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q0));
    auto Q1 = DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    DPCTLQueueMgr_PushQueue(Q1);
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    DPCTLQueue_Delete(Q1);
    DPCTLQueueMgr_PopQueue();
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q0));
    DPCTLQueue_Delete(Q0);
}

TEST(TestDPCTLSyclQueueManager, CheckGetNumActivatedQueues)
{
    if (!(DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_GPU) &&
          DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_CPU)))
        GTEST_SKIP_("Both OpenCL gpu and cpu drivers needed for this test.");

    size_t num0, num1, num2, num4;
    auto DS1 = DPCTLFilterSelector_Create("opencl:gpu");
    auto D1 = DPCTLDevice_CreateFromSelector(DS1);
    auto Q1 = DPCTLQueue_CreateForDevice(D1, nullptr, DPCTL_DEFAULT_PROPERTY);
    DPCTLQueueMgr_PushQueue(Q1);

    std::thread first(foo, std::ref(num1));
    std::thread second(bar, std::ref(num2));

    // synchronize threads:
    first.join();
    second.join();

    // Capture the number of active queues in first
    num0 = DPCTLQueueMgr_GetQueueStackSize();
    DPCTLQueueMgr_PopQueue();
    num4 = DPCTLQueueMgr_GetQueueStackSize();

    // Verify what the expected number of activated queues each time a thread
    // called getNumActivatedQueues.
    EXPECT_EQ(num0, 1ul);
    EXPECT_EQ(num1, 2ul);
    EXPECT_EQ(num2, 1ul);
    EXPECT_EQ(num4, 0ul);

    DPCTLQueue_Delete(Q1);
    DPCTLDeviceSelector_Delete(DS1);
    DPCTLDevice_Delete(D1);
}

TEST(TestDPCTLSyclQueueManager, CheckIsCurrentQueue2)
{
    if (!(DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_GPU) &&
          DPCTLDeviceMgr_GetNumDevices(DPCTL_OPENCL | DPCTL_CPU)))
        GTEST_SKIP_("Both OpenCL gpu and cpu drivers needed for this test.");

    auto DS1 = DPCTLFilterSelector_Create("opencl:gpu");
    auto DS2 = DPCTLFilterSelector_Create("opencl:cpu");
    auto D1 = DPCTLDevice_CreateFromSelector(DS1);
    auto D2 = DPCTLDevice_CreateFromSelector(DS2);
    auto Q1 = DPCTLQueue_CreateForDevice(D1, nullptr, DPCTL_DEFAULT_PROPERTY);
    DPCTLQueueMgr_PushQueue(Q1);
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    auto Q2 = DPCTLQueue_CreateForDevice(D2, nullptr, DPCTL_DEFAULT_PROPERTY);
    DPCTLQueueMgr_PushQueue(Q2);
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q2));
    EXPECT_FALSE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    DPCTLQueue_Delete(Q2);
    DPCTLQueueMgr_PopQueue();
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    DPCTLQueue_Delete(Q1);
    DPCTLQueueMgr_PopQueue();
    DPCTLDeviceSelector_Delete(DS1);
    DPCTLDeviceSelector_Delete(DS2);
    DPCTLDevice_Delete(D1);
    DPCTLDevice_Delete(D2);
}

INSTANTIATE_TEST_SUITE_P(QueueMgrFunctions,
                         TestDPCTLSyclQueueManager,
                         ::testing::Values("opencl:gpu:0",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0"));
