//===---- test_sycl_queue_submit_raw_kernel_arg - Test raw kernel arg -----===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file contains tests for kernel submit using the raw_kernel_arg
/// SYCL extension.
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_kernel_bundle_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_sycl_usm_interface.h"

#include <stddef.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

namespace
{
static constexpr std::size_t SIZE = 320;

static_assert(SIZE % 10 == 0);

using namespace dpctl::syclinterface;

template <typename T> struct Params
{
    T mul;
    T add;
};

template <typename T>
void submit_kernel(DPCTLSyclQueueRef QRef,
                   DPCTLSyclKernelBundleRef KBRef,
                   std::vector<char> spirvBuffer,
                   std::size_t spirvFileSize,
                   std::string kernelName)
{
    if (!DPCTLRawKernelArg_Available()) {
        GTEST_SKIP() << "Skipping raw_kernel_arg test since the compiler does "
                        "not support this feature";
        return;
    }

    static constexpr std::size_t NARGS = 2;
    static constexpr std::size_t RANGE_NDIMS = 1;

    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, kernelName.c_str()));
    auto kernel = DPCTLKernelBundle_GetKernel(KBRef, kernelName.c_str());

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(T), QRef);
    ASSERT_TRUE(a != nullptr);
    auto a_ptr = static_cast<T *>(unwrap<void>(a));
    for (auto i = 0ul; i < SIZE; ++i) {
        a_ptr[i] = T{1};
    }

    // Create kernel args for vector_add
    std::size_t lws = SIZE / 10;
    std::size_t gRange[] = {SIZE};
    std::size_t lRange[] = {lws};

    Params<T> p{T{4}, T{5}};
    auto rka = DPCTLRawKernelArg_Create(&p, sizeof(Params<T>));
    ASSERT_TRUE(rka != nullptr);
    auto *rka_raw = unwrap<std::vector<unsigned char>>(rka);
    ASSERT_TRUE(rka_raw != nullptr);
    void *args_1d[NARGS] = {unwrap<void>(a), rka};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR,
                                              DPCTL_RAW_KERNEL_ARG};

    DPCTLSyclEventRef E1Ref = DPCTLQueue_SubmitNDRange(
        kernel, QRef, args_1d, addKernelArgTypes, NARGS, gRange, lRange,
        RANGE_NDIMS, nullptr, 0);
    ASSERT_TRUE(E1Ref != nullptr);

    DPCTLSyclEventRef DepEv1[] = {E1Ref};
    void *args_2d[NARGS] = {unwrap<void>(a), rka};

    DPCTLSyclEventRef E2Ref =
        DPCTLQueue_SubmitNDRange(kernel, QRef, args_2d, addKernelArgTypes,
                                 NARGS, gRange, lRange, RANGE_NDIMS, DepEv1, 1);
    ASSERT_TRUE(E2Ref != nullptr);

    DPCTLSyclEventRef DepEv2[] = {E1Ref, E2Ref};
    void *args_3d[NARGS] = {unwrap<void>(a), rka};

    DPCTLSyclEventRef E3Ref =
        DPCTLQueue_SubmitNDRange(kernel, QRef, args_3d, addKernelArgTypes,
                                 NARGS, gRange, lRange, RANGE_NDIMS, DepEv2, 2);
    ASSERT_TRUE(E3Ref != nullptr);

    DPCTLEvent_Wait(E3Ref);

    std::cout << a_ptr[0] << std::endl;
    ASSERT_TRUE(a_ptr[0] == T(169));

    // clean ups
    DPCTLEvent_Delete(E1Ref);
    DPCTLEvent_Delete(E2Ref);
    DPCTLEvent_Delete(E3Ref);
    DPCTLRawKernelArg_Delete(rka);
    DPCTLKernel_Delete(kernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
}

} /* end of anonymous namespace */

/*
// The work_group_memory_kernel spv files were generated from the SYCL program
// included in this comment. The program can be compiled using
// `icpx -fsycl raw_kernel_arg_kernel.cpp`. After that if the generated
// executable is run with the environment variable `SYCL_DUMP_IMAGES=1`, icpx
// runtime will dump all offload sections of fat binary to the current working
// directory. When tested with DPC++ 2025.1 the kernels are split across two
// separate SPV files. One contains all kernels for integers and FP32
// data type, and another contains the kernel for FP64.
//
// Note that, `SYCL_DUMP_IMAGES=1` will also generate extra SPV files that
// contain the code for built in functions such as indexing and barriers. To
// figure which SPV file contains the kernels, use `spirv-dis` from the
// spirv-tools package to translate the SPV binary format to a human-readable
// textual format.
#include <iostream>
#include <sstream>
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

template <typename T>
struct Params{ T mul; T add; };

template <typename T>
class SyclKernel_RKA
{
private:
    T *a_ = nullptr;
    Params<T> p_;

public:
  SyclKernel_RKA(T *a, Params<T> p)
        : a_(a), p_(p)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        int i = it.get_global_id();
        a_[i] = (a_[i] * p_.mul) + p_.add;
    }
};

template <typename T>
sycl::event
submit_kernel(sycl::queue q, const unsigned long N, T *a, T mul, T add)
{
  auto gws = N;
  auto lws = (N/10);

  sycl::range<1> gRange{gws};
  sycl::range<1> lRange{lws};
  sycl::nd_range<1> ndRange{gRange, lRange};

  Params<T> p{mul, add};

  sycl::event e =
    q.submit([&](auto &h)
    {
        h.parallel_for(
            ndRange,
            SyclKernel_RKA<T>(a, p));
    });

  return e;
}

template <typename T>
void driver(std::size_t N)
{
    sycl::queue q;
    auto *a = sycl::malloc_shared<T>(N, q);
    submit_kernel(q, N, a, T{4}, T{5}).wait();
    sycl::free(a, q);
}

int main(int argc, const char **argv)
{
    std::size_t N = 0;
    std::cout << "Enter problem size in N:\n";
    std::cin >> N;
    std::cout << "Executing with N = " << N << std::endl;

    driver<int8_t>(N);
    driver<uint8_t>(N);
    driver<int16_t>(N);
    driver<uint16_t>(N);
    driver<int32_t>(N);
    driver<uint32_t>(N);
    driver<int64_t>(N);
    driver<uint64_t>(N);
    driver<float>(N);
    driver<double>(N);

    return 0;
}
*/

struct TestQueueSubmitWithRawKernelArg : public ::testing::Test
{
    std::ifstream spirvFile;
    std::size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitWithRawKernelArg()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;
        const char *test_spv_fn = "./raw_kernel_arg_kernel_inttys_fp32.spv";

        spirvFile.open(test_spv_fn, std::ios::binary | std::ios::ate);
        spirvFileSize_ = std::filesystem::file_size(test_spv_fn);
        spirvBuffer_.reserve(spirvFileSize_);
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer_.data(), spirvFileSize_);

        DSRef = DPCTLDefaultSelector_Create();
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
        QRef =
            DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
        auto CRef = DPCTLQueue_GetContext(QRef);

        KBRef = DPCTLKernelBundle_CreateFromSpirv(
            CRef, DRef, spirvBuffer_.data(), spirvFileSize_, nullptr);
        DPCTLDevice_Delete(DRef);
        DPCTLDeviceSelector_Delete(DSRef);
    }

    ~TestQueueSubmitWithRawKernelArg()
    {
        spirvFile.close();
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

struct TestQueueSubmitWithRawKernelArgFP64 : public ::testing::Test
{
    std::ifstream spirvFile;
    std::size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitWithRawKernelArgFP64()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        const char *test_spv_fn = "./raw_kernel_arg_kernel_fp64.spv";

        spirvFile.open(test_spv_fn, std::ios::binary | std::ios::ate);
        spirvFileSize_ = std::filesystem::file_size(test_spv_fn);
        spirvBuffer_.reserve(spirvFileSize_);
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer_.data(), spirvFileSize_);
        DSRef = DPCTLDefaultSelector_Create();
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
        QRef =
            DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
        auto CRef = DPCTLQueue_GetContext(QRef);

        KBRef = DPCTLKernelBundle_CreateFromSpirv(
            CRef, DRef, spirvBuffer_.data(), spirvFileSize_, nullptr);
        DPCTLDeviceSelector_Delete(DSRef);
    }

    ~TestQueueSubmitWithRawKernelArgFP64()
    {
        spirvFile.close();
        DPCTLDevice_Delete(DRef);
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForInt8)
{
    submit_kernel<std::int8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                               "_ZTS14SyclKernel_RKAIaE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForUInt8)
{
    submit_kernel<std::uint8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                "_ZTS14SyclKernel_RKAIhE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForInt16)
{
    submit_kernel<std::int16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                "_ZTS14SyclKernel_RKAIsE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForUInt16)
{
    submit_kernel<std::uint16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                 "_ZTS14SyclKernel_RKAItE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForInt32)
{
    submit_kernel<std::int32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                "_ZTS14SyclKernel_RKAIiE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForUInt32)
{
    submit_kernel<std::uint32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                 "_ZTS14SyclKernel_RKAIjE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForInt64)
{
    submit_kernel<std::int64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                "_ZTS14SyclKernel_RKAIlE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForUInt64)
{
    submit_kernel<std::uint64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                 "_ZTS14SyclKernel_RKAImE");
}

TEST_F(TestQueueSubmitWithRawKernelArg, CheckForFloat)
{
    submit_kernel<float>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                         "_ZTS14SyclKernel_RKAIfE");
}

TEST_F(TestQueueSubmitWithRawKernelArgFP64, CheckForDouble)
{
    if (DPCTLDevice_HasAspect(DRef, DPCTLSyclAspectType::fp64)) {
        submit_kernel<double>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                              "_ZTS14SyclKernel_RKAIdE");
    }
}
