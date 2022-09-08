//===--- test_service.cpp - Test cases for sevice functions  ===//
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
/// dpctl_service.h.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_service.h"
#include <gtest/gtest.h>
#include <string>

#define ASSTR(a) TOSTR(a)
#define TOSTR(a) #a

TEST(TestServicesFns, ChkDPCPPVersion)
{
    auto c_ver = DPCTLService_GetDPCPPVersion();
    std::string ver = std::string(c_ver);
    ASSERT_TRUE(ver.length() > 0);

    std::string ver_from_cmplr(ASSTR(__VERSION__));
    std::size_t found = ver_from_cmplr.find(ver);

    // version returned by DPCTLService_GetDPCPPVersion
    // should occur as a substring in the version obtained
    // from compiler
    ASSERT_TRUE(found != std::string::npos);
}
