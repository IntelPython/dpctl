//===--- test_sycl_queue_manager.cpp - DPPL-SYCL interface --*- C++ ---*---===//
//
//               Python Data Parallel Processing Library (PyDPPL)
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
/// dppl_sycl_queue_interface.h and dppl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//
#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_device_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"
#include <gtest/gtest.h>
#include <thread>

using namespace std;

namespace
{
    void foo (size_t & num)
    {
        auto q1 = DPPLQueueMgr_PushQueue(DPPL_OPENCL, DPPL_CPU, 0);
        auto q2 = DPPLQueueMgr_PushQueue(DPPL_OPENCL, DPPL_GPU, 0);
        // Capture the number of active queues in first
        num = DPPLQueueMgr_GetNumActivatedQueues();
        DPPLQueueMgr_PopQueue();
        DPPLQueueMgr_PopQueue();
        DPPLQueue_Delete(q1);
        DPPLQueue_Delete(q2);
    }

    void bar (size_t & num)
    {
        auto q1 = DPPLQueueMgr_PushQueue(DPPL_OPENCL, DPPL_GPU, 0);
        // Capture the number of active queues in second
        num = DPPLQueueMgr_GetNumActivatedQueues();
        DPPLQueueMgr_PopQueue();
        DPPLQueue_Delete(q1);
    }
}

struct TestDPPLSyclQueueManager : public ::testing::Test
{ };


TEST_F (TestDPPLSyclQueueManager, CheckDPPLGetCurrentQueue)
{
    DPPLSyclQueueRef q;
    ASSERT_NO_THROW(q = DPPLQueueMgr_GetCurrentQueue());
    ASSERT_TRUE(q != nullptr);
}


TEST_F (TestDPPLSyclQueueManager, CheckDPPLGetOpenCLCpuQ)
{
    auto nOpenCLCpuQ = DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_CPU);
    if(!nOpenCLCpuQ)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    EXPECT_TRUE(DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_CPU, 0) != nullptr);
    auto non_existent_device_num = nOpenCLCpuQ+1;
    // Non-existent device number should return nullptr
    if (DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_CPU, non_existent_device_num)) {
        FAIL() << "SYCL OpenCL CPU device " << non_existent_device_num
               << "should not exist.";
    }
}

TEST_F (TestDPPLSyclQueueManager, CheckDPPLGetOpenCLGpuQ)
{
    auto nOpenCLGpuQ = DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_GPU);
    if(!nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    EXPECT_TRUE(DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0) != nullptr);
    auto non_existent_device_num = nOpenCLGpuQ+1;
    // Non-existent device number should return nullptr
    if (DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, non_existent_device_num)) {
        FAIL() << "SYCL OpenCL GPU device " << non_existent_device_num
               << "should not exist.";
    }
}

TEST_F (TestDPPLSyclQueueManager, CheckDPPLGetLevel0GpuQ)
{
    auto nL0GpuQ = DPPLQueueMgr_GetNumQueues(DPPL_LEVEL_ZERO, DPPL_GPU);
    if(!nL0GpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    EXPECT_TRUE(DPPLQueueMgr_GetQueue(DPPL_LEVEL_ZERO, DPPL_GPU, 0) != nullptr);
    auto non_existent_device_num = nL0GpuQ+1;
    // Non-existent device number should return nullptr
    if (DPPLQueueMgr_GetQueue(DPPL_LEVEL_ZERO, DPPL_GPU,
                              non_existent_device_num))
    {
        FAIL() << "SYCL OpenCL GPU device " << non_existent_device_num
               << "should not exist.";
    }
}


TEST_F (TestDPPLSyclQueueManager, CheckGetNumActivatedQueues)
{
    size_t num0, num1, num2, num4;

    auto nOpenCLCpuQ = DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_CPU);
    auto nOpenCLGpuQ = DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_GPU);
    auto nL0GpuQ     = DPPLQueueMgr_GetNumQueues(DPPL_LEVEL_ZERO, DPPL_GPU);

    // Add a queue to main thread
    if(!nOpenCLCpuQ || !nOpenCLGpuQ)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto q = DPPLQueueMgr_PushQueue(DPPL_OPENCL, DPPL_CPU, 0);

    std::thread first (foo, std::ref(num1));
    std::thread second (bar, std::ref(num2));

    // synchronize threads:
    first.join();
    second.join();

    // Capture the number of active queues in first
    num0 = DPPLQueueMgr_GetNumActivatedQueues();
    DPPLQueueMgr_PopQueue();
    num4 = DPPLQueueMgr_GetNumActivatedQueues();

    // Verify what the expected number of activated queues each time a thread
    // called getNumActivatedQueues.
    EXPECT_EQ(num0, 1);
    EXPECT_EQ(num1, 2);
    EXPECT_EQ(num2, 1);
    EXPECT_EQ(num4, 0);

    DPPLQueue_Delete(q);
}


TEST_F (TestDPPLSyclQueueManager, CheckDPPLDumpDeviceInfo)
{
    auto q = DPPLQueueMgr_GetCurrentQueue();
    EXPECT_NO_FATAL_FAILURE(DPPLDevice_DumpInfo(DPPLQueue_GetDevice(q)));
    EXPECT_NO_FATAL_FAILURE(DPPLQueue_Delete(q));
}


int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
