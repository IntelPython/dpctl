#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_enum_types.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <utility>

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::device, DPCTLSyclDeviceRef);

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
    constexpr auto param_2 = get_param_list<DPCTLSyclAspectType>(
        host, cpu, gpu, accelerator, custom, fp16, fp64, int64_base_atomics,
        int64_extended_atomics, online_compiler, online_linker, queue_profiling,
        usm_device_allocations, usm_host_allocations, usm_shared_allocations,
        usm_restricted_shared_allocations, usm_system_allocator);

    auto pairs =
        build_param_pairs<const char *, DPCTLSyclAspectType, param_1.size(),
                          param_2.size()>(param_1, param_2);

    return build_gtest_values(pairs);
}

} // namespace

struct TestDPCTLSyclDeviceInterfaceAspects
    : public ::testing::TestWithParam<
          std::pair<const char *, DPCTLSyclAspectType>>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    bool actual = false;

    TestDPCTLSyclDeviceInterfaceAspects()
    {
        auto params = GetParam();
        auto filterstr = params.first;
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(filterstr));
    }

    void SetUp()
    {
        if (!DSRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam().first) + ".";
            GTEST_SKIP_(message.c_str());
        }
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        if (!DRef)
            GTEST_SKIP_("Device not found");
        auto D = unwrap(DRef);
        try {
            actual = D->has(
                DPCTL_DPCTLAspectTypeToSyclAspectType(GetParam().second));
        } catch (...) {
        }
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }

    ~TestDPCTLSyclDeviceInterfaceAspects()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
};

TEST_P(TestDPCTLSyclDeviceInterfaceAspects, Chk_HasAspect)
{
    bool expected = false;
    auto aspectTy = GetParam().second;
    DPCTLSyclDeviceRef DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");
    EXPECT_NO_FATAL_FAILURE(expected = DPCTLDevice_HasAspect(DRef, aspectTy));
    EXPECT_TRUE(expected == actual);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

INSTANTIATE_TEST_SUITE_P(DPCTLSyclDeviceInterfaceAspects,
                         TestDPCTLSyclDeviceInterfaceAspects,
                         build_params());
