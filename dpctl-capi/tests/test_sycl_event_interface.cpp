//===------ test_sycl_event_interface.cpp - Test cases for event interface ===//
//
//                      Data Parallel Control (dpctl)
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
/// dpctl_sycl_event_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_event_interface.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(event, DPCTLSyclEventRef)
} // namespace

struct TestDPCTLSyclEventInterface : public ::testing::Test
{
    DPCTLSyclEventRef ERef = nullptr;

    TestDPCTLSyclEventInterface()
    {
        EXPECT_NO_FATAL_FAILURE(ERef = DPCTLEvent_Create());
    }

    void SetUp()
    {
        ASSERT_TRUE(ERef);
    }

    ~TestDPCTLSyclEventInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
    }
};

TEST_F(TestDPCTLSyclEventInterface, CheckEvent_Copy)
{
    DPCTLSyclEventRef Copied_ERef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_ERef = DPCTLEvent_Copy(ERef));
    EXPECT_TRUE(bool(Copied_ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(Copied_ERef));
}

TEST_F(TestDPCTLSyclEventInterface, CheckCopy_Invalid)
{
    DPCTLSyclEventRef E1 = nullptr;
    DPCTLSyclEventRef E2 = nullptr;
    EXPECT_NO_FATAL_FAILURE(E2 = DPCTLEvent_Copy(E1));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(E1));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(E2));
}
