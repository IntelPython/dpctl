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

    constexpr auto param_2 =
        get_param_list<std::pair<const char *, cl::sycl::aspect>>(
            // clang-format off
            std::make_pair("host", cl::sycl::aspect::host), 
            std::make_pair("cpu", cl::sycl::aspect::cpu), 
            std::make_pair("gpu", cl::sycl::aspect::gpu),
            std::make_pair("accelerator", cl::sycl::aspect::accelerator),
            std::make_pair("custom", cl::sycl::aspect::custom),
            std::make_pair("fp16", cl::sycl::aspect::fp16), 
            std::make_pair("fp64", cl::sycl::aspect::fp64),
            std::make_pair("int64_base_atomics", cl::sycl::aspect::int64_base_atomics),
            std::make_pair("int64_extended_atomics", cl::sycl::aspect::int64_extended_atomics),
            std::make_pair("online_compiler", cl::sycl::aspect::online_compiler),
            std::make_pair("online_linker", cl::sycl::aspect::online_linker),
            std::make_pair("queue_profiling", cl::sycl::aspect::queue_profiling),
            std::make_pair("usm_device_allocations", cl::sycl::aspect::usm_device_allocations),
            std::make_pair("usm_host_allocations", cl::sycl::aspect::usm_host_allocations),
            std::make_pair("usm_shared_allocations", cl::sycl::aspect::usm_shared_allocations),
            std::make_pair("usm_restricted_shared_allocations", cl::sycl::aspect::usm_restricted_shared_allocations),
            std::make_pair("usm_system_allocator", cl::sycl::aspect::usm_system_allocator));
    // clang-format on

    auto pairs =
        build_param_pairs<const char *,
                          std::pair<const char *, cl::sycl::aspect>,
                          param_1.size(), param_2.size()>(param_1, param_2);

    return build_gtest_values(pairs);
}

} // namespace

struct TestDPCTLSyclDeviceInterfaceAspects
    : public ::testing::TestWithParam<
          std::pair<const char *, std::pair<const char *, cl::sycl::aspect>>>
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
        auto D = unwrap(DRef);
        auto syclAspect = GetParam().second.second;
        try {
            hasAspect = D->has(syclAspect);
        } catch (cl::sycl::runtime_error const &re) {
        }
    }

    ~TestDPCTLSyclDeviceInterfaceAspects()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
};

TEST_P(TestDPCTLSyclDeviceInterfaceAspects, Chk_HasAspect)
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
