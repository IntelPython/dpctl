#include "dppl_sycl_queue_interface.h"
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

using namespace std;

namespace
{
    void foo (size_t & num)
    {
        auto q1 = DPPLPushSyclQueue(DPPL_CPU, 0);
        auto q2 = DPPLPushSyclQueue(DPPL_GPU, 0);
        // Capture the number of active queues in first
        num = DPPLGetNumActivatedQueues();
        DPPLPopSyclQueue();
        DPPLPopSyclQueue();
        DPPLDeleteQueue(q1);
        DPPLDeleteQueue(q2);
    }

    void bar (size_t & num)
    {
        auto q1 = DPPLPushSyclQueue(DPPL_GPU, 0);
        // Capture the number of active queues in second
        num = DPPLGetNumActivatedQueues();
        DPPLPopSyclQueue();
        DPPLDeleteQueue(q1);
    }
}

struct TestDPPLSyclQueuemanager : public ::testing::Test
{

};

TEST_F (TestDPPLSyclQueuemanager, CheckGetNumPlatforms)
{
    auto nplatforms = DPPLGetNumPlatforms();
    EXPECT_GE(nplatforms, 0);
}


TEST_F (TestDPPLSyclQueuemanager, CheckDPPLGetCurrentQueue)
{
    DPPLSyclQueueRef q;
    ASSERT_NO_THROW(q = DPPLGetCurrentQueue());
    ASSERT_TRUE(q != nullptr);
}


TEST_F (TestDPPLSyclQueuemanager, CheckDPPLGetQueue)
{
    auto numCpuQueues = DPPLGetNumCPUQueues();
    auto numGpuQueues = DPPLGetNumGPUQueues();
    if(numCpuQueues > 0) {
        EXPECT_TRUE(DPPLGetQueue(DPPL_CPU, 0) != nullptr);
        auto non_existent_device_num = numCpuQueues+1;
        try {
            DPPLGetQueue(DPPL_CPU, non_existent_device_num);
            FAIL() << "SYCL CPU device " << non_existent_device_num
                   << "not found on system.";
        }
        catch (...) { }
    }
    if(numGpuQueues > 0) {
        EXPECT_TRUE(DPPLGetQueue(DPPL_GPU, 0) != nullptr);
        auto non_existent_device_num = numGpuQueues+1;
        try {
            DPPLGetQueue(DPPL_GPU, non_existent_device_num);
            FAIL() << "SYCL GPU device " << non_existent_device_num
                   << "not found on system.";
        }
        catch (...) { }
    }
}


TEST_F (TestDPPLSyclQueuemanager, CheckGetNumActivatedQueues)
{
    size_t num0, num1, num2, num4;

    // Add a queue to main thread
    auto q = DPPLPushSyclQueue(DPPL_CPU, 0);

    std::thread first (foo, std::ref(num1));
    std::thread second (bar, std::ref(num2));

    // synchronize threads:
    first.join();
    second.join();

    // Capture the number of active queues in first
    num0 = DPPLGetNumActivatedQueues();
    DPPLPopSyclQueue();
    num4 = DPPLGetNumActivatedQueues();

    // Verify what the expected number of activated queues each time a thread
    // called getNumActivatedQueues.
    EXPECT_EQ(num0, 1);
    EXPECT_EQ(num1, 2);
    EXPECT_EQ(num2, 1);
    EXPECT_EQ(num4, 0);

    DPPLDeleteQueue(q);
}


TEST_F (TestDPPLSyclQueuemanager, CheckDPPLDumpPlatformInfo)
{
    EXPECT_NO_FATAL_FAILURE(DPPLDumpPlatformInfo());
}


TEST_F (TestDPPLSyclQueuemanager, CheckDPPLDumpDeviceInfo)
{
    auto q = DPPLGetCurrentQueue();
    EXPECT_NO_FATAL_FAILURE(DPPLDumpDeviceInfo(q));
    EXPECT_NO_FATAL_FAILURE(DPPLDeleteQueue(q));
}


int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
