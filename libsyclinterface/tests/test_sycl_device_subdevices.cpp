
//===--- test_sycl_device_interface.cpp - Test cases for device interface  ===//
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
/// dpctl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_utils.h"
#include "dpctl_utils_helper.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef);

const DPCTLPartitionAffinityDomainType a_dpctl_domain =
    DPCTLPartitionAffinityDomainType::not_applicable;

struct TestDPCTLSyclDeviceInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceRef DRef = nullptr;

    TestDPCTLSyclDeviceInterface()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        if (DS) {
            EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DS));
        }
        DPCTLDeviceSelector_Delete(DS);
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLSyclDeviceInterface()
    {
        DPCTLDevice_Delete(DRef);
    }
};

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesEqually)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    uint32_t maxCUs = 0;

    EXPECT_NO_FATAL_FAILURE(maxCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (maxCUs) {
        int count = maxCUs / 2;
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesEqually(DRef, count));
        if (DVRef) {
            DPCTLSyclDeviceRef pDRef = nullptr;
            DPCTLSyclDeviceRef sDRef = nullptr;
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) > 0);
            EXPECT_NO_FATAL_FAILURE(sDRef = DPCTLDeviceVector_GetAt(DVRef, 0));
            EXPECT_NO_FATAL_FAILURE(pDRef = DPCTLDevice_GetParentDevice(sDRef));
            EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(sDRef));
            EXPECT_TRUE(DPCTLDevice_AreEq(DRef, pDRef));
            EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(pDRef));
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesEqually(DRef, 0));
        EXPECT_TRUE(DVRef == nullptr);
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByCounts)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    uint32_t maxCUs = 0;

    EXPECT_NO_FATAL_FAILURE(maxCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (maxCUs) {
        size_t count = maxCUs / 2;
        size_t *counts = nullptr;
        int n = 2;
        counts = new size_t[n];
        for (auto i = 0; i < n; ++i) {
            counts[i] = count;
        }
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByCounts(DRef, counts, n));
        if (DVRef) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) > 0);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
            DVRef = nullptr;
        }
        counts[n - 1] = 0;
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByCounts(DRef, counts, n));
        EXPECT_TRUE(DVRef == nullptr);
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByAffinityNotApplicable)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::not_applicable;
    DPCTLPartitionAffinityDomainType dpctl_domain =
        DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain);

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            auto expected_size = subDevices.size();

            if (DVRef) {
                EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
                EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
            }
        } catch (exception const &e) {
        }
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByAffinityNuma)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::numa;
    DPCTLPartitionAffinityDomainType dpctl_domain = a_dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::exception const &e) {
            std::cerr << e.what() << std::endl;
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByAffinityL4Cache)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L4_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain = a_dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::exception const &e) {
            std::cerr << e.what() << std::endl;
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByAffinityL3Cache)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L3_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain = a_dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::exception const &e) {
            std::cerr << e.what() << std::endl;
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByAffinityL2Cache)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L2_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain = a_dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::exception const &e) {
            std::cerr << e.what() << std::endl;
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkCreateSubDevicesByAffinityL1Cache)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L1_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain = a_dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::exception const &e) {
            std::cerr << e.what() << std::endl;
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }
}

TEST_P(TestDPCTLSyclDeviceInterface,
       ChkCreateSubDevicesByAffinityNextPartitionable)
{
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::next_partitionable;
    DPCTLPartitionAffinityDomainType dpctl_domain = a_dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::exception const &e) {
            std::cerr << e.what() << std::endl;
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(DPCTLDeviceFns,
                         TestDPCTLSyclDeviceInterface,
                         ::testing::Values("opencl",
                                           "opencl:gpu",
                                           "opencl:cpu",
                                           "opencl:gpu:0",
                                           "gpu",
                                           "cpu",
                                           "level_zero",
                                           "level_zero:gpu",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0",
                                           "gpu:0",
                                           "gpu:1",
                                           "1"));
