//===------- test_sycl_queue_manager.cpp - Test cases for queue manager    ===//
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
using namespace sycl;

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

INSTANTIATE_TEST_SUITE_P(QueueMgrFunctions,
                         TestDPCTLSyclQueueManager,
                         ::testing::Values("opencl:gpu:0",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0"));

struct TestDPCTLQueueMgrFeatures : public ::testing::Test
{
    TestDPCTLQueueMgrFeatures() {}
    ~TestDPCTLQueueMgrFeatures() {}
};

TEST_F(TestDPCTLQueueMgrFeatures, CheckGetNumActivatedQueues)
{
    size_t num0, num1, num2, num4;
    DPCTLSyclDeviceSelectorRef CPU_DSRef = nullptr, GPU_DSRef = nullptr;
    DPCTLSyclDeviceRef CPU_DRef = nullptr, GPU_DRef = nullptr;

    GPU_DSRef = DPCTLFilterSelector_Create("opencl:gpu");
    GPU_DRef = DPCTLDevice_CreateFromSelector(GPU_DSRef);
    CPU_DSRef = DPCTLFilterSelector_Create("opencl:cpu");
    CPU_DRef = DPCTLDevice_CreateFromSelector(CPU_DSRef);

    if (!(CPU_DRef && GPU_DRef)) {
        DPCTLDeviceSelector_Delete(GPU_DSRef);
        DPCTLDevice_Delete(GPU_DRef);
        DPCTLDeviceSelector_Delete(CPU_DSRef);
        DPCTLDevice_Delete(CPU_DRef);
        GTEST_SKIP_(
            "OpenCL GPU and CPU devices are needed, but were not found.");
    }
    else {
        auto Q1 = DPCTLQueue_CreateForDevice(GPU_DRef, nullptr,
                                             DPCTL_DEFAULT_PROPERTY);
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

        // Verify what the expected number of activated queues each time a
        // thread called getNumActivatedQueues.
        EXPECT_EQ(num0, 1ul);
        EXPECT_EQ(num1, 2ul);
        EXPECT_EQ(num2, 1ul);
        EXPECT_EQ(num4, 0ul);

        DPCTLQueue_Delete(Q1);
        DPCTLDeviceSelector_Delete(GPU_DSRef);
        DPCTLDevice_Delete(GPU_DRef);
        DPCTLDeviceSelector_Delete(CPU_DSRef);
        DPCTLDevice_Delete(CPU_DRef);
    }
}

TEST_F(TestDPCTLQueueMgrFeatures, CheckIsCurrentQueue2)
{
    DPCTLSyclDeviceSelectorRef DS1 = nullptr, DS2 = nullptr;
    DPCTLSyclDeviceRef D1 = nullptr, D2 = nullptr;

    DS1 = DPCTLFilterSelector_Create("opencl:gpu");
    DS2 = DPCTLFilterSelector_Create("opencl:cpu");
    D1 = DPCTLDevice_CreateFromSelector(DS1);
    D2 = DPCTLDevice_CreateFromSelector(DS2);

    if (!(D1 && D2)) {
        DPCTLDeviceSelector_Delete(DS1);
        DPCTLDeviceSelector_Delete(DS2);
        DPCTLDevice_Delete(D1);
        DPCTLDevice_Delete(D2);
        GTEST_SKIP_(
            "OpenCL GPU and CPU devices are needed, but were not found.");
    }

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

TEST_F(TestDPCTLQueueMgrFeatures, CheckSetGlobalQueueForSubDevices)
{
    DPCTLSyclDeviceSelectorRef DS = nullptr;
    DPCTLSyclDeviceRef RootCpu_DRef = nullptr;
    size_t max_eu_count = 0;
    DPCTLSyclDeviceRef SubDev0_DRef = nullptr;
    DPCTLSyclDeviceRef SubDev1_DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DS = DPCTLFilterSelector_Create("opencl:cpu"));
    EXPECT_NO_FATAL_FAILURE(RootCpu_DRef = DPCTLDevice_CreateFromSelector(DS));
    DPCTLDeviceSelector_Delete(DS);
    EXPECT_TRUE(RootCpu_DRef);
    EXPECT_NO_FATAL_FAILURE(max_eu_count =
                                DPCTLDevice_GetMaxComputeUnits(RootCpu_DRef));
    size_t n1 = max_eu_count / 2;
    size_t n2 = max_eu_count - n1;

    if (n1 > 0 && n2 > 0) {
        size_t counts[2] = {n1, n2};
        DPCTLDeviceVectorRef DVRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLDevice_CreateSubDevicesByCounts(
                                    RootCpu_DRef, counts, 2));
        EXPECT_NO_FATAL_FAILURE(SubDev0_DRef =
                                    DPCTLDeviceVector_GetAt(DVRef, 0));
        EXPECT_NO_FATAL_FAILURE(SubDev1_DRef =
                                    DPCTLDeviceVector_GetAt(DVRef, 1));
        EXPECT_NO_FATAL_FAILURE(
            QRef = DPCTLQueue_CreateForDevice(SubDev0_DRef, nullptr,
                                              DPCTL_DEFAULT_PROPERTY));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(SubDev1_DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(SubDev0_DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLQueueMgr_SetGlobalQueue(QRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));
    }
    else {
        DPCTLDevice_Delete(RootCpu_DRef);
        GTEST_SKIP_("OpenCL CPU devices are needed, but were not found.");
    }
}

TEST_F(TestDPCTLQueueMgrFeatures,
       CheckSetGlobalQueueForSubDevicesMultiDeviceContext)
{
    DPCTLSyclDeviceSelectorRef DS = nullptr;
    DPCTLSyclDeviceRef RootCpu_DRef = nullptr;
    size_t max_eu_count = 0;
    DPCTLSyclDeviceRef SubDev0_DRef = nullptr;
    DPCTLSyclDeviceRef SubDev1_DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DS = DPCTLFilterSelector_Create("opencl:cpu"));
    EXPECT_NO_FATAL_FAILURE(RootCpu_DRef = DPCTLDevice_CreateFromSelector(DS));
    DPCTLDeviceSelector_Delete(DS);
    EXPECT_TRUE(RootCpu_DRef);
    EXPECT_NO_FATAL_FAILURE(max_eu_count =
                                DPCTLDevice_GetMaxComputeUnits(RootCpu_DRef));
    size_t n1 = max_eu_count / 2;
    size_t n2 = max_eu_count - n1;

    if (n1 > 0 && n2 > 0) {
        size_t counts[2] = {n1, n2};
        DPCTLDeviceVectorRef DVRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLDevice_CreateSubDevicesByCounts(
                                    RootCpu_DRef, counts, 2));
        EXPECT_NO_FATAL_FAILURE(SubDev0_DRef =
                                    DPCTLDeviceVector_GetAt(DVRef, 0));
        EXPECT_NO_FATAL_FAILURE(SubDev1_DRef =
                                    DPCTLDeviceVector_GetAt(DVRef, 1));
        EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_CreateFromDevices(
                                    DVRef, nullptr, DPCTL_DEFAULT_PROPERTY));
        EXPECT_NO_FATAL_FAILURE(
            QRef = DPCTLQueue_Create(CRef, SubDev0_DRef, nullptr,
                                     DPCTL_DEFAULT_PROPERTY));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(SubDev1_DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(SubDev0_DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLQueueMgr_SetGlobalQueue(QRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
    }
    else {
        DPCTLDevice_Delete(RootCpu_DRef);
        GTEST_SKIP_("OpenCL CPU devices are needed, but were not found.");
    }
}

struct TestDPCTLQueueMgrNullArgs : public ::testing::Test
{
    DPCTLSyclQueueRef Null_QRef = nullptr;

    TestDPCTLQueueMgrNullArgs() {}
    ~TestDPCTLQueueMgrNullArgs() {}
};

TEST_F(TestDPCTLQueueMgrNullArgs, ChkGlobalQueueIsCurrent)
{
    bool res = true;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLQueueMgr_GlobalQueueIsCurrent());
    ASSERT_TRUE(res == true || res == false);
}

TEST_F(TestDPCTLQueueMgrNullArgs, ChkIsCurrentQueue)
{
    bool res = true;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLQueueMgr_IsCurrentQueue(Null_QRef));
    ASSERT_FALSE(res);
}

#if 0
TEST_F(TestDPCTLQueueMgrNullArgs, ChkSetGlobalQueue)
{
    EXPECT_DEATH(DPCTLQueueMgr_SetGlobalQueue(Null_QRef), "*");
}

TEST_F(TestDPCTLQueueMgrNullArgs, ChkPushGlobalQueue)
{
    EXPECT_DEATH(DPCTLQueueMgr_SetGlobalQueue(Null_QRef), "*");
}
#endif

TEST_F(TestDPCTLQueueMgrNullArgs, ChkGetQueueStackSize)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLQueueMgr_GetQueueStackSize());
    ASSERT_TRUE(n < size_t(-1));
}
