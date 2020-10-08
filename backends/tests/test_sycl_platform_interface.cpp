//===------- test_sycl_platform_interface.cpp - dpctl-C_API --*-- C++ --*--===//
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
/// dppl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//
#include "dppl_sycl_platform_interface.h"
#include <gtest/gtest.h>

struct TestDPPLSyclPlatformInterface : public ::testing::Test
{ };

TEST_F (TestDPPLSyclPlatformInterface, CheckGetNumPlatforms)
{
    auto nplatforms = DPPLPlatform_GetNumNonHostPlatforms();
    EXPECT_GE(nplatforms, 0);
}

TEST_F (TestDPPLSyclPlatformInterface, GetNumBackends)
{
    auto nbackends = DPPLPlatform_GetNumNonHostBackends();
    EXPECT_GE(nbackends, 0);
}

TEST_F (TestDPPLSyclPlatformInterface, GetListOfBackends)
{
    auto nbackends = DPPLPlatform_GetNumNonHostBackends();

    if(!nbackends)
      GTEST_SKIP_("No non host backends available");

    auto backends = DPPLPlatform_GetListOfNonHostBackends();
	  EXPECT_TRUE(backends != nullptr);
    for(auto i = 0ul; i < nbackends; ++i) {
        EXPECT_TRUE(
          backends[i] == DPPLSyclBackendType::DPPL_CUDA   ||
          backends[i] == DPPLSyclBackendType::DPPL_OPENCL ||
          backends[i] == DPPLSyclBackendType::DPPL_LEVEL_ZERO
          );
    }
	DPPLPlatform_DeleteListOfBackends(backends);
}

TEST_F (TestDPPLSyclPlatformInterface, CheckDPPLPlatformDumpInfo)
{
    EXPECT_NO_FATAL_FAILURE(DPPLPlatform_DumpInfo());
}

int
main (int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
