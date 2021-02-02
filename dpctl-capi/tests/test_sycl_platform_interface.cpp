//===-- test_sycl_platform_interface.cpp - Test cases for platform interface =//
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
/// dpctl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//
#include "dpctl_sycl_platform_interface.h"
#include <gtest/gtest.h>

struct TestDPCTLSyclPlatformInterface : public ::testing::Test
{ };

TEST_F (TestDPCTLSyclPlatformInterface, CheckGetNumPlatforms)
{
    auto nplatforms = DPCTLPlatform_GetNumNonHostPlatforms();
    EXPECT_GE(nplatforms, 0ul);
}

TEST_F (TestDPCTLSyclPlatformInterface, GetNumBackends)
{
    auto nbackends = DPCTLPlatform_GetNumNonHostBackends();
    EXPECT_GE(nbackends, 0ul);
}

TEST_F (TestDPCTLSyclPlatformInterface, GetListOfBackends)
{
    auto nbackends = DPCTLPlatform_GetNumNonHostBackends();

    if(!nbackends)
      GTEST_SKIP_("No non host backends available");

    auto backends = DPCTLPlatform_GetListOfNonHostBackends();
	  EXPECT_TRUE(backends != nullptr);
    for(auto i = 0ul; i < nbackends; ++i) {
        EXPECT_TRUE(
          backends[i] == DPCTLSyclBackendType::DPCTL_CUDA   ||
          backends[i] == DPCTLSyclBackendType::DPCTL_OPENCL ||
          backends[i] == DPCTLSyclBackendType::DPCTL_LEVEL_ZERO
          );
    }
	DPCTLPlatform_DeleteListOfBackends(backends);
}

TEST_F (TestDPCTLSyclPlatformInterface, CheckDPCTLPlatformDumpInfo)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_DumpInfo());
}
