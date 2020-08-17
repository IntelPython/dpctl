#include "dppl_sycl_interface.hpp"
#include "dppl_error_codes.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace std;
using namespace dppl;

struct TestDPPLSyclQueuemanager : public ::testing::Test
{
//   TestDPPLSyclQueuemanager ()
//   {
//     qmgr = DpplSyclQueueManager();
//   }

protected:
  DpplSyclQueueManager qmgr;
};

TEST_F (TestDPPLSyclQueuemanager, CheckGetNumPlatforms)
{
    size_t nPlatforms;
    EXPECT_EQ(DPPL_SUCCESS, qmgr.getNumPlatforms(&nPlatforms));
}

TEST_F (TestDPPLSyclQueuemanager, CheckMultipleInstancesOfQMgr)
{
    void * q1, * q2;
    DpplSyclQueueManager qmgr2;
    qmgr.getCurrentQueue(&q1);
    qmgr2.getCurrentQueue(&q2);
}

int
main (int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}