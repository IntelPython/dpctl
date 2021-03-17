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

#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_utils.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef);

struct TestDPCTLSyclDeviceInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;

    TestDPCTLSyclDeviceInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(GetParam()));
    }

    void SetUp()
    {
        if (!DSRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLSyclDeviceInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
};

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesEqually)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;
    uint32_t maxCUs = 0;

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    EXPECT_NO_FATAL_FAILURE(maxCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (maxCUs) {
        int count = maxCUs / 2;
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesEqually(DRef, count));
        if (DVRef) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) > 0);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesByCounts)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;
    uint32_t maxCUs = 0;

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

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
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface,
       Chk_CreateSubDevicesByAffinityNotApplicable)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::not_applicable;
    DPCTLPartitionAffinityDomainType dpctl_domain =
        DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain);

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

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
        } catch (runtime_error const &re) {
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesByAffinityNuma)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::numa;
    DPCTLPartitionAffinityDomainType dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesByAffinityL4Cache)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L4_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesByAffinityL3Cache)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L3_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesByAffinityL2Cache)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L2_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_CreateSubDevicesByAffinityL1Cache)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::L1_cache;
    DPCTLPartitionAffinityDomainType dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface,
       Chk_CreateSubDevicesByAffinityNextPartitionable)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    info::partition_affinity_domain domain =
        info::partition_affinity_domain::next_partitionable;
    DPCTLPartitionAffinityDomainType dpctl_domain;
    EXPECT_NO_FATAL_FAILURE(
        dpctl_domain = DPCTL_SyclPartitionAffinityDomainToDPCTLType(domain));

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    if (dpctl_domain) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesByAffinity(DRef, dpctl_domain));

        auto D = unwrap(DRef);
        size_t expected_size = 0;
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            expected_size = subDevices.size();
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }

        if (DVRef && expected_size) {
            EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == expected_size);
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
        }
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

INSTANTIATE_TEST_SUITE_P(DPCTLDevice_Fns,
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
