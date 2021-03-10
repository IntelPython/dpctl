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
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include <gtest/gtest.h>
#include <thread>

#include <CL/sycl.hpp>

using namespace std;
using namespace cl::sycl;

namespace
{
void foo(size_t &num)
{
    auto q1 = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
    auto q2 = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    // Capture the number of active queues in first
    num = DPCTLQueueMgr_GetNumActivatedQueues();
    DPCTLQueueMgr_PopQueue();
    DPCTLQueueMgr_PopQueue();
    DPCTLQueue_Delete(q1);
    DPCTLQueue_Delete(q2);
}

void bar(size_t &num)
{
    auto q1 = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    // Capture the number of active queues in second
    num = DPCTLQueueMgr_GetNumActivatedQueues();
    DPCTLQueueMgr_PopQueue();
    DPCTLQueue_Delete(q1);
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

struct TestDPCTLSyclQueueManager : public ::testing::Test
{
};

TEST_F(TestDPCTLSyclQueueManager, CheckDPCTLGetCurrentQueue)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    DPCTLSyclQueueRef q = nullptr;
    ASSERT_NO_THROW(q = DPCTLQueueMgr_GetCurrentQueue());
    ASSERT_TRUE(q != nullptr);
}

TEST_F(TestDPCTLSyclQueueManager, CheckDPCTLGetOpenCLCpuQ)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOpenCLCpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU);
    if (!nOpenCLCpuQ)
        GTEST_SKIP_("Skipping: No OpenCL CPU device found.");

    auto q = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
    EXPECT_TRUE(q != nullptr);
    auto sycl_q = reinterpret_cast<queue *>(q);
    auto be = sycl_q->get_context().get_platform().get_backend();
    EXPECT_EQ(be, backend::opencl);
    auto devty = sycl_q->get_device().get_info<info::device::device_type>();
    EXPECT_EQ(devty, info::device_type::cpu);

    auto non_existent_device_num = nOpenCLCpuQ + 1;
    // Non-existent device number should return nullptr
    auto null_q = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_CPU,
                                         non_existent_device_num);
    ASSERT_TRUE(null_q == nullptr);
}

TEST_F(TestDPCTLSyclQueueManager, CheckDPCTLGetOpenCLGpuQ)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nOpenCLGpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU);
    if (!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping: No OpenCL GPU device found.\n");

    auto q = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    EXPECT_TRUE(q != nullptr);
    auto sycl_q = reinterpret_cast<queue *>(q);
    auto be = sycl_q->get_context().get_platform().get_backend();
    EXPECT_EQ(be, backend::opencl);
    auto devty = sycl_q->get_device().get_info<info::device::device_type>();
    EXPECT_EQ(devty, info::device_type::gpu);

    auto non_existent_device_num = nOpenCLGpuQ + 1;
    // Non-existent device number should return nullptr
    auto null_q = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU,
                                         non_existent_device_num);
    ASSERT_TRUE(null_q == nullptr);
}

TEST_F(TestDPCTLSyclQueueManager, CheckDPCTLGetLevel0GpuQ)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    auto nL0GpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_LEVEL_ZERO, DPCTL_GPU);
    if (!nL0GpuQ)
        GTEST_SKIP_("Skipping: No OpenCL GPU device found.\n");

    auto q = DPCTLQueueMgr_GetQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU, 0);
    EXPECT_TRUE(q != nullptr);
    auto sycl_q = reinterpret_cast<queue *>(q);
    auto be = sycl_q->get_context().get_platform().get_backend();
    EXPECT_EQ(be, backend::level_zero);
    auto devty = sycl_q->get_device().get_info<info::device::device_type>();
    EXPECT_EQ(devty, info::device_type::gpu);

    auto non_existent_device_num = nL0GpuQ + 1;
    // Non-existent device number should return nullptr
    auto null_q = DPCTLQueueMgr_GetQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU,
                                         non_existent_device_num);
    ASSERT_TRUE(null_q == nullptr);
}

TEST_F(TestDPCTLSyclQueueManager, CheckGetNumActivatedQueues)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");

    size_t num0, num1, num2, num4;

    auto nOpenCLCpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU);
    auto nOpenCLGpuQ = DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU);

    // Add a queue to main thread
    if (!nOpenCLCpuQ || !nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto q = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_CPU, 0);

    std::thread first(foo, std::ref(num1));
    std::thread second(bar, std::ref(num2));

    // synchronize threads:
    first.join();
    second.join();

    // Capture the number of active queues in first
    num0 = DPCTLQueueMgr_GetNumActivatedQueues();
    DPCTLQueueMgr_PopQueue();
    num4 = DPCTLQueueMgr_GetNumActivatedQueues();

    // Verify what the expected number of activated queues each time a thread
    // called getNumActivatedQueues.
    EXPECT_EQ(num0, 1ul);
    EXPECT_EQ(num1, 2ul);
    EXPECT_EQ(num2, 1ul);
    EXPECT_EQ(num4, 0ul);

    DPCTLQueue_Delete(q);
}

TEST_F(TestDPCTLSyclQueueManager, CheckDPCTLDumpDeviceInfo)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");
    auto q = DPCTLQueueMgr_GetCurrentQueue();
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_DumpInfo(DPCTLQueue_GetDevice(q)));
    EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(q));
}

TEST_F(TestDPCTLSyclQueueManager, CheckIsCurrentQueue)
{
    if (!has_devices())
        GTEST_SKIP_("Skipping: No Sycl devices.\n");
    if (!DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU))
        GTEST_SKIP_("Skipping: No OpenCL GPU.\n");

    auto Q0 = DPCTLQueueMgr_GetCurrentQueue();
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q0));
    auto Q1 = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    DPCTLQueue_Delete(Q1);
    DPCTLQueueMgr_PopQueue();
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q0));
    DPCTLQueue_Delete(Q0);
}

TEST_F(TestDPCTLSyclQueueManager, CheckIsCurrentQueue2)
{
    if (!DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU) ||
        !DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU))
        GTEST_SKIP_("Skipping: No OpenCL GPU and OpenCL CPU.\n");

    auto Q1 = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    auto Q2 = DPCTLQueueMgr_PushQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q2));
    EXPECT_FALSE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    DPCTLQueue_Delete(Q2);
    DPCTLQueueMgr_PopQueue();
    EXPECT_TRUE(DPCTLQueueMgr_IsCurrentQueue(Q1));
    DPCTLQueue_Delete(Q1);
    DPCTLQueueMgr_PopQueue();
}

TEST_F(TestDPCTLSyclQueueManager, CreateQueueFromDeviceAndContext)
{
    auto Q = DPCTLQueueMgr_GetCurrentQueue();
    auto D = DPCTLQueue_GetDevice(Q);
    auto C = DPCTLQueue_GetContext(Q);

    auto Q2 = DPCTLQueueMgr_GetQueueFromContextAndDevice(C, D);
    auto D2 = DPCTLQueue_GetDevice(Q2);
    auto C2 = DPCTLQueue_GetContext(Q2);

    EXPECT_TRUE(DPCTLDevice_AreEq(D, D2));
    EXPECT_TRUE(DPCTLContext_AreEq(C, C2));

    DPCTLQueue_Delete(Q2);
    DPCTLQueue_Delete(Q);
}
