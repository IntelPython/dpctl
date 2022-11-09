//===------ test_sycl_event_interface.cpp - Test cases for event interface ===//
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
/// dpctl_sycl_event_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_types.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <vector>

using namespace sycl;

namespace
{
sycl::event produce_event(sycl::queue &Q, sycl::buffer<int> &data)
{
    int N = data.get_range()[0];

    auto e1 = Q.submit([&](sycl::handler &h) {
        sycl::accessor a{data, h, sycl::write_only, sycl::no_init};
        h.parallel_for(N, [=](sycl::id<1> i) { a[i] = 1; });
    });

    auto e2 = Q.submit([&](sycl::handler &h) {
        sycl::accessor a{data, h};
        h.single_task([=]() {
            for (int i = 1; i < N; i++)
                a[0] += a[i];
        });
    });

    return e2;
}
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

TEST_F(TestDPCTLSyclEventInterface, CheckEvent_Wait)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
}

TEST_F(TestDPCTLSyclEventInterface, CheckWait_Invalid)
{
    DPCTLSyclEventRef E = nullptr;
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(E));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(E));
}

TEST_F(TestDPCTLSyclEventInterface, CheckEvent_WaitAndThrow)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_WaitAndThrow(ERef));
}

TEST_F(TestDPCTLSyclEventInterface, CheckWaitAndThrow_Invalid)
{
    DPCTLSyclEventRef E = nullptr;
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_WaitAndThrow(E));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(E));
}

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

TEST_F(TestDPCTLSyclEventInterface, CheckEvent_GetBackend)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(BTy = DPCTLEvent_GetBackend(ERef));
    EXPECT_TRUE([BTy] {
        switch (BTy) {
        case DPCTLSyclBackendType::DPCTL_CUDA:
            return true;
        case DPCTLSyclBackendType::DPCTL_HOST:
            return true;
        case DPCTLSyclBackendType::DPCTL_LEVEL_ZERO:
            return true;
        case DPCTLSyclBackendType::DPCTL_OPENCL:
            return true;
        default:
            return false;
        }
    }());
}

TEST_F(TestDPCTLSyclEventInterface, CheckGetBackend_Invalid)
{
    DPCTLSyclEventRef E = nullptr;
    DPCTLSyclBackendType Bty = DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(Bty = DPCTLEvent_GetBackend(E));
    EXPECT_TRUE(Bty == DPCTL_UNKNOWN_BACKEND);
}

TEST_F(TestDPCTLSyclEventInterface, ChkGetCommandExecutionStatus)
{
    DPCTLSyclEventStatusType ESTy =
        DPCTLSyclEventStatusType::DPCTL_UNKNOWN_STATUS;
    EXPECT_NO_FATAL_FAILURE(ESTy = DPCTLEvent_GetCommandExecutionStatus(ERef));
    EXPECT_TRUE(ESTy != DPCTLSyclEventStatusType::DPCTL_UNKNOWN_STATUS);
    EXPECT_TRUE(ESTy == DPCTLSyclEventStatusType::DPCTL_COMPLETE);
}

TEST_F(TestDPCTLSyclEventInterface, CheckGetProfiling)
{
    property_list propList{property::queue::enable_profiling()};

#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER
    queue Q(cpu_selector_v, propList);
#else
    queue Q(cpu_selector(), propList);
#endif
    auto eA = Q.submit(
        [&](handler &h) { h.parallel_for(1000, [=](id<1>) { /*...*/ }); });
    DPCTLSyclEventRef ERef = reinterpret_cast<DPCTLSyclEventRef>(&eA);

    auto eStart = DPCTLEvent_GetProfilingInfoStart(ERef);
    auto eEnd = DPCTLEvent_GetProfilingInfoEnd(ERef);
    auto eSubmit = DPCTLEvent_GetProfilingInfoSubmit(ERef);

    EXPECT_TRUE(eStart);
    EXPECT_TRUE(eEnd);
    EXPECT_TRUE(eSubmit);
}

TEST_F(TestDPCTLSyclEventInterface, CheckGetProfiling_Invalid)
{
    auto eStart = DPCTLEvent_GetProfilingInfoStart(ERef);
    auto eEnd = DPCTLEvent_GetProfilingInfoEnd(ERef);
    auto eSubmit = DPCTLEvent_GetProfilingInfoSubmit(ERef);

    EXPECT_FALSE(eStart);
    EXPECT_FALSE(eEnd);
    EXPECT_FALSE(eSubmit);
}

TEST_F(TestDPCTLSyclEventInterface, CheckGetWaitList)
{
    DPCTLEventVectorRef EVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(EVRef = DPCTLEvent_GetWaitList(ERef));
    ASSERT_TRUE(EVRef);
    EXPECT_NO_FATAL_FAILURE(DPCTLEventVector_Clear(EVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEventVector_Delete(EVRef));
}

TEST_F(TestDPCTLSyclEventInterface, CheckGetWaitListSYCL)
{
    sycl::queue q;
    sycl::buffer<int> data{42};
    sycl::event eD;
    DPCTLEventVectorRef EVRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(eD = produce_event(q, data));
    DPCTLSyclEventRef ERef = reinterpret_cast<DPCTLSyclEventRef>(&eD);
    EXPECT_NO_FATAL_FAILURE(EVRef = DPCTLEvent_GetWaitList(ERef));
    ASSERT_TRUE(DPCTLEventVector_Size(EVRef) > 0);
    DPCTLEventVector_Delete(EVRef);
}
