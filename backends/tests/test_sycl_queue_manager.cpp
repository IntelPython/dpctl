#include "dppl_sycl_queue_interface.hpp"
#include "dppl_error_codes.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

using namespace std;
using namespace dppl;

namespace
{
    void foo (DpplSyclQueueManager & qmgr, size_t & num)
    {
        void *q1, *q2;

        qmgr.setAsCurrentQueue(&q1, sycl_device_type::DPPL_CPU, 0);
        qmgr.setAsCurrentQueue(&q2, sycl_device_type::DPPL_GPU, 0);
        // Capture the number of active queues in first
        qmgr.getNumActivatedQueues(num);
        qmgr.removeCurrentQueue();
        qmgr.removeCurrentQueue();
    }

    void bar (DpplSyclQueueManager & qmgr, size_t & num)
    {
        void *q1;

        qmgr.setAsCurrentQueue(&q1, sycl_device_type::DPPL_GPU, 0);
        // Capture the number of active queues in second
        qmgr.getNumActivatedQueues(num);
        qmgr.removeCurrentQueue();
    }
}

struct TestDPPLSyclQueuemanager : public ::testing::Test
{
protected:
  DpplSyclQueueManager qmgr;
};

TEST_F (TestDPPLSyclQueuemanager, CheckGetNumPlatforms)
{
    size_t nPlatforms;
    auto ret = qmgr.getNumPlatforms(nPlatforms);
    EXPECT_EQ(DPPL_SUCCESS, ret);
}

TEST_F (TestDPPLSyclQueuemanager, CheckGetNumActivatedQueues)
{
    size_t num0, num1, num2, num4;
    void *q;

    // Add a queue to main thread
    qmgr.setAsCurrentQueue(&q, sycl_device_type::DPPL_CPU, 0);

    std::thread first (foo, std::ref(qmgr), std::ref(num1));
    std::thread second (bar, std::ref(qmgr), std::ref(num2));

    // synchronize threads:
    first.join();
    second.join();

    // Capture the number of active queues in first
    qmgr.getNumActivatedQueues(num0);
    qmgr.removeCurrentQueue();
    qmgr.getNumActivatedQueues(num4);

    // Verify what the expected number of activated queues each time a thread
    // called getNumActivatedQueues.
    EXPECT_EQ(num0, 1);
    EXPECT_EQ(num1, 2);
    EXPECT_EQ(num2, 1);
    EXPECT_EQ(num4, 0);

    deleteQueue(q);
}

int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
