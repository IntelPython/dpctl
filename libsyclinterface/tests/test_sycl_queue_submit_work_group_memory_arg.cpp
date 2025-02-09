//===-- test_sycl_queue_submit.cpp - Test cases for kernel submission fns. ===//
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
/// This file has unit test cases for the various submit functions defined
/// inside dpctl_sycl_queue_interface.cpp.
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
constexpr std::size_t SIZE = 320;

static_assert(SIZE % 10 == 0);

using namespace dpctl::syclinterface;

template <typename T>
void submit_kernel(DPCTLSyclQueueRef QRef,
                   DPCTLSyclKernelBundleRef KBRef,
                   std::vector<char> spirvBuffer,
                   std::size_t spirvFileSize,
                   DPCTLKernelArgType kernelArgTy,
                   std::string kernelName)
{

#if SYCL_EXT_ONEAPI_WORK_GROUP_MEMORY

    constexpr std::size_t NARGS = 2;
    constexpr std::size_t RANGE_NDIMS = 1;

    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, kernelName.c_str()));
    auto kernel = DPCTLKernelBundle_GetKernel(KBRef, kernelName.c_str());

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(T), QRef);
    ASSERT_TRUE(a != nullptr);
    auto a_ptr = static_cast<T *>(unwrap<void>(a));
    for (auto i = 0ul; i < SIZE; ++i) {
        a_ptr[i] = 0;
    }

    // Create kernel args for vector_add
    std::size_t lws = SIZE / 10;
    std::size_t gRange[] = {SIZE};
    std::size_t lRange[] = {lws};
    std::uintptr_t wgm_sz = lws * sizeof(T);
    void *args_1d[NARGS] = {unwrap<void>(a), (void *)wgm_sz};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR,
                                              DPCTL_WORK_GROUP_MEMORY};

    DPCTLSyclEventRef E1Ref = DPCTLQueue_SubmitNDRange(
        kernel, QRef, args_1d, addKernelArgTypes, NARGS, gRange, lRange,
        RANGE_NDIMS, nullptr, 0);
    ASSERT_TRUE(E1Ref != nullptr);

    DPCTLSyclEventRef DepEv1[] = {E1Ref};
    void *args_2d[NARGS] = {unwrap<void>(a), (void *)wgm_sz};

    DPCTLSyclEventRef E2Ref =
        DPCTLQueue_SubmitNDRange(kernel, QRef, args_2d, addKernelArgTypes,
                                 NARGS, gRange, lRange, RANGE_NDIMS, DepEv1, 1);
    ASSERT_TRUE(E2Ref != nullptr);

    DPCTLSyclEventRef DepEv2[] = {E1Ref, E2Ref};
    void *args_3d[NARGS] = {unwrap<void>(a), (void *)wgm_sz};

    DPCTLSyclEventRef E3Ref =
        DPCTLQueue_SubmitNDRange(kernel, QRef, args_3d, addKernelArgTypes,
                                 NARGS, gRange, lRange, RANGE_NDIMS, DepEv2, 2);
    ASSERT_TRUE(E3Ref != nullptr);

    DPCTLEvent_Wait(E3Ref);

    ASSERT_TRUE(a_ptr[0] == T(lws * 2));

    // clean ups
    DPCTLEvent_Delete(E1Ref);
    DPCTLEvent_Delete(E2Ref);
    DPCTLEvent_Delete(E3Ref);
    DPCTLKernel_Delete(kernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
#else
    GTEST_SKIP() << "Skipping work-group-memory test since the compiler does "
                    "not support this feature";
    return;
#endif
}

} /* end of anonymous namespace */

/*
// The work_group_memory_kernel spv files were generated from the SYCL program
// included in this comment. The program can be compiled using
// `icpx -fsycl work_group_memory_kernel.cpp`. After that if the generated
// executable is run with the environment variable `SYCL_DUMP_IMAGES=1`, icpx
// runtime will dump all offload sections of fat binary to the current working
// directory. When tested with DPC++ 2024.0 the kernels are split across two
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
class SyclKernel_WGM
{
private:
    T N_;
    T *a_ = nullptr;
    syclexp::work_group_memory<T[]> wgm_;

public:
  SyclKernel_WGM(T *a, syclexp::work_group_memory<T[]> wgm)
        : a_(a), wgm_(wgm)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        int i = it.get_global_id();
        int j = it.get_local_id();
        wgm_[j] = 2;
        auto g = it.get_group();
        group_barrier(g);
        auto temp = 0;
        for (auto idx = 0ul; idx < it.get_local_range(0); ++idx)
            temp += wgm_[idx];
        a_[i] = temp * (i + 1);
    }
};

template <typename T>
sycl::event
submit_kernel(sycl::queue q, const unsigned long N, T *a)
{
  auto gws = N;
  auto lws = (N/10);

  sycl::range<1> gRange{gws};
  sycl::range<1> lRange{lws};
  sycl::nd_range<1> ndRange{gRange, lRange};

  sycl::event e =
    q.submit([&](auto &h)
    {
        syclexp::work_group_memory<T[]> wgm(lws, h);
        h.parallel_for(
            ndRange,
            SyclKernel_WGM<T>(a, wgm));
    });

  return e;
}

template <typename T>
void driver(std::size_t N)
{
    sycl::queue q;
    auto *a = sycl::malloc_shared<T>(N, q);
    submit_kernel(q, N, a).wait();
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

struct TestQueueSubmitWithWorkGroupMemory : public ::testing::Test
{
    std::ifstream spirvFile;
    std::size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitWithWorkGroupMemory()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;
        const char *test_spv_fn = "./work_group_memory_kernel_inttys_fp32.spv";

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

    ~TestQueueSubmitWithWorkGroupMemory()
    {
        spirvFile.close();
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

struct TestQueueSubmitWithWorkGroupMemoryFP64 : public ::testing::Test
{
    std::ifstream spirvFile;
    std::size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitWithWorkGroupMemoryFP64()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        const char *test_spv_fn = "./work_group_memory_kernel_fp64.spv";

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

    ~TestQueueSubmitWithWorkGroupMemoryFP64()
    {
        spirvFile.close();
        DPCTLDevice_Delete(DRef);
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForInt8)
{
    submit_kernel<std::int8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                               DPCTLKernelArgType::DPCTL_INT8_T,
                               "_ZTS14SyclKernel_WGMIaE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForUInt8)
{
    submit_kernel<std::uint8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                DPCTLKernelArgType::DPCTL_UINT8_T,
                                "_ZTS14SyclKernel_WGMIhE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForInt16)
{
    submit_kernel<std::int16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                DPCTLKernelArgType::DPCTL_INT16_T,
                                "_ZTS14SyclKernel_WGMIsE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForUInt16)
{
    submit_kernel<std::uint16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                 DPCTLKernelArgType::DPCTL_UINT16_T,
                                 "_ZTS14SyclKernel_WGMItE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForInt32)
{
    submit_kernel<std::int32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                DPCTLKernelArgType::DPCTL_INT32_T,
                                "_ZTS14SyclKernel_WGMIiE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForUInt32)
{
    submit_kernel<std::uint32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                 DPCTLKernelArgType::DPCTL_UINT32_T,
                                 "_ZTS14SyclKernel_WGMIjE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForInt64)
{
    submit_kernel<std::int64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                DPCTLKernelArgType::DPCTL_INT64_T,
                                "_ZTS14SyclKernel_WGMIlE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForUInt64)
{
    submit_kernel<std::uint64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                                 DPCTLKernelArgType::DPCTL_UINT64_T,
                                 "_ZTS14SyclKernel_WGMImE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemory, CheckForFloat)
{
    submit_kernel<float>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                         DPCTLKernelArgType::DPCTL_FLOAT32_T,
                         "_ZTS14SyclKernel_WGMIfE");
}

TEST_F(TestQueueSubmitWithWorkGroupMemoryFP64, CheckForDouble)
{
    if (DPCTLDevice_HasAspect(DRef, DPCTLSyclAspectType::fp64)) {
        submit_kernel<double>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                              DPCTLKernelArgType::DPCTL_FLOAT64_T,
                              "_ZTS14SyclKernel_WGMIdE");
    }
}
