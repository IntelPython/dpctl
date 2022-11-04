//===--- test_sycl_device_aspects.cpp - Test cases for device interface ===//
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
/// dpctl_sycl_device_interface.h concernings device aspects.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <utility>

namespace
{

template <typename...> struct are_same : std::true_type
{
};

template <typename T> struct are_same<T> : std::true_type
{
};

template <typename T1, typename T2, typename... Types>
struct are_same<T1, T2, Types...>
    : std::integral_constant<bool,
                             (std::is_same<T1, T2>::value &&
                              are_same<T1, Types...>::value)>
{
};

template <typename T,
          typename... Ts,
          typename std::enable_if<are_same<T, Ts...>::value>::type * = nullptr>
constexpr auto get_param_list(Ts... args)
{
    std::array<T, sizeof...(Ts)> params{{args...}};
    return params;
}

template <typename T1, typename T2, size_t S1, size_t S2>
constexpr auto build_param_pairs(const std::array<T1, S1> &arr1,
                                 const std::array<T2, S2> &arr2)
{
    std::array<std::pair<T1, T2>, S1 * S2> paramPairs;
    auto n = 0ul;

    for (auto &p1 : arr1) {
        for (auto &p2 : arr2) {
            paramPairs[n] = {p1, p2};
            ++n;
        }
    }

    return paramPairs;
}

template <typename PArr, std::size_t... I>
auto build_gtest_values_impl(const PArr &arr, std::index_sequence<I...>)
{
    return ::testing::Values(arr[I]...);
}

template <typename T1,
          typename T2,
          size_t N,
          typename Indices = std::make_index_sequence<N>>
auto build_gtest_values(const std::array<std::pair<T1, T2>, N> &params)
{
    return build_gtest_values_impl(params, Indices());
}

auto build_params()
{
    constexpr auto param_1 = get_param_list<const char *>(
        "opencl:gpu", "opencl:cpu", "level_zero:gpu", "host");

    constexpr auto param_2 =
        get_param_list<std::pair<const char *, sycl::aspect>>(
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
            std::make_pair("host", sycl::aspect::host),
#endif
            std::make_pair("cpu", sycl::aspect::cpu),
            std::make_pair("gpu", sycl::aspect::gpu),
            std::make_pair("accelerator", sycl::aspect::accelerator),
            std::make_pair("custom", sycl::aspect::custom),
            std::make_pair("fp16", sycl::aspect::fp16),
            std::make_pair("fp64", sycl::aspect::fp64),
            std::make_pair("atomic64", sycl::aspect::atomic64),
            std::make_pair("online_compiler", sycl::aspect::online_compiler),
            std::make_pair("online_linker", sycl::aspect::online_linker),
            std::make_pair("queue_profiling", sycl::aspect::queue_profiling),
            std::make_pair("usm_device_allocations",
                           sycl::aspect::usm_device_allocations),
            std::make_pair("usm_host_allocations",
                           sycl::aspect::usm_host_allocations),
            std::make_pair("usm_shared_allocations",
                           sycl::aspect::usm_shared_allocations),
            std::make_pair("usm_restricted_shared_allocations",
                           sycl::aspect::usm_restricted_shared_allocations),
            std::make_pair("usm_system_allocations",
                           sycl::aspect::usm_system_allocations),
            std::make_pair("usm_atomic_host_allocations",
                           sycl::aspect::usm_atomic_host_allocations),
            std::make_pair("usm_atomic_shared_allocations",
                           sycl::aspect::usm_atomic_shared_allocations),
            std::make_pair("host_debuggable", sycl::aspect::host_debuggable));

    auto pairs =
        build_param_pairs<const char *, std::pair<const char *, sycl::aspect>,
                          param_1.size(), param_2.size()>(param_1, param_2);

    return build_gtest_values(pairs);
}

using namespace dpctl::syclinterface;

} // namespace

struct TestDPCTLSyclDeviceInterfaceAspects
    : public ::testing::TestWithParam<
          std::pair<const char *, std::pair<const char *, sycl::aspect>>>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    bool hasAspect = false;

    TestDPCTLSyclDeviceInterfaceAspects()
    {
        auto filterstr = GetParam().first;
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(filterstr));
    }

    void SetUp()
    {
        if (!DSRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam().first) + ".";
            GTEST_SKIP_(message.c_str());
        }

        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        if (!DRef)
            GTEST_SKIP_("Device not found");
        auto D = unwrap<sycl::device>(DRef);
        auto syclAspect = GetParam().second.second;
        try {
            hasAspect = D->has(syclAspect);
        } catch (sycl::exception const &e) {
        }
    }

    ~TestDPCTLSyclDeviceInterfaceAspects()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
};

TEST_P(TestDPCTLSyclDeviceInterfaceAspects, ChkHasAspect)
{
    bool actual = false;
    auto dpctlAspect = DPCTL_StrToAspectType(GetParam().second.first);
    auto AspectTy = DPCTL_SyclAspectToDPCTLAspectType(dpctlAspect);
    EXPECT_NO_FATAL_FAILURE(actual = DPCTLDevice_HasAspect(DRef, AspectTy));
    EXPECT_TRUE(hasAspect == actual);
}

INSTANTIATE_TEST_SUITE_P(DPCTLSyclDeviceInterfaceAspects,
                         TestDPCTLSyclDeviceInterfaceAspects,
                         build_params());
