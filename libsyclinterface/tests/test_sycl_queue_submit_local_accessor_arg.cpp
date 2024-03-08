//===-- test_sycl_queue_submit.cpp - Test cases for kernel submission fns. ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <utility>

namespace
{
constexpr size_t SIZE = 100;

using namespace dpctl::syclinterface;

typedef struct MDLocalAccessorTy
{
    size_t ndim;
    DPCTLKernelArgType dpctl_type_id;
    size_t dim0;
    size_t dim1;
    size_t dim2;
} MDLocalAccessor;

template <typename T>
void submit_kernel(DPCTLSyclQueueRef QRef,
                   DPCTLSyclKernelBundleRef KBRef,
                   std::vector<char> spirvBuffer,
                   size_t spirvFileSize,
                   DPCTLKernelArgType kernelArgTy,
                   std::string kernelName)
{
    constexpr size_t NARGS = 2;
    constexpr size_t RANGE_NDIMS = 1;

    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, kernelName.c_str()));
    auto kernel = DPCTLKernelBundle_GetKernel(KBRef, kernelName.c_str());

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(T), QRef);
    ASSERT_TRUE(a != nullptr);
    auto a_ptr = static_cast<T *>(unwrap<void>(a));
    for (auto i = 0ul; i < SIZE; ++i) {
        a_ptr[i] = 0;
    }

    auto la = MDLocalAccessor{1, kernelArgTy, SIZE / 10, 1, 1};

    // Create kernel args for vector_add
    size_t gRange[] = {SIZE};
    size_t lRange[] = {SIZE / 10};
    void *args[NARGS] = {unwrap<void>(a), (void *)&la};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR,
                                              DPCTL_LOCAL_ACCESSOR};

    auto ERef =
        DPCTLQueue_SubmitNDRange(kernel, QRef, args, addKernelArgTypes, NARGS,
                                 gRange, lRange, RANGE_NDIMS, nullptr, 0);
    ASSERT_TRUE(ERef != nullptr);
    DPCTLQueue_Wait(QRef);

    if (kernelArgTy != DPCTL_FLOAT32_T && kernelArgTy != DPCTL_FLOAT64_T)
        ASSERT_TRUE(a_ptr[0] == 20);
    else
        ASSERT_TRUE(a_ptr[0] == 20.0);

    // clean ups
    DPCTLEvent_Delete(ERef);
    DPCTLKernel_Delete(kernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
}

} /* end of anonymous namespace */

/*
// The local_accessor_kernel spv files were generated from the SYCL program
// included in this comment. The program can be compiled using
// `icpx -fsycl local_accessor_kernel.cpp`. After that if the generated
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
#include <CL/sycl.hpp>
#include <iostream>
#include <sstream>

template <typename T>
class SyclKernel_SLM
{
private:
    T N_;
    T *a_ = nullptr;
    sycl::local_accessor<T, 1> slm_;

public:
    SyclKernel_SLM(T *a, sycl::local_accessor<T, 1> slm)
        : a_(a), slm_(slm)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        int i = it.get_global_id();
        int j = it.get_local_id();
        slm_[j] = 2;
        auto g = it.get_group();
        group_barrier(g);
        auto temp = 0;
        for (auto idx = 0ul; idx < it.get_local_range(0); ++idx)
            temp += slm_[idx];
        a_[i] = temp * (i + 1);
    }
};

template <typename T>
void submit_kernel(sycl::queue q, const unsigned long N, T *a)
{
    q.submit([&](auto &h)
             {
        sycl::local_accessor<T, 1> slm(sycl::range(N/10), h);
        h.parallel_for(sycl::nd_range(sycl::range{N}, sycl::range{N/10}),
                              SyclKernel_SLM<T>(a, slm)); });
}

template <typename T>
void driver(size_t N)
{
    sycl::queue q;
    auto *a = sycl::malloc_shared<T>(N, q);
    submit_kernel(q, N, a);
    q.wait();
    sycl::free(a, q);
}

int main(int argc, const char **argv)
{
    size_t N = 0;
    std::cout << "Enter problem size in N:\n";
    std::cin >> N;
    std::cout << "Executing with N = " << N << std::endl;

    driver<int8_t>(N);
    driver<uint8_t>(N);
    driver<int16_t>(N);
    driver<int32_t>(N);
    driver<int32_t>(N);
    driver<uint32_t>(N);
    driver<int64_t>(N);
    driver<uint64_t>(N);
    driver<float>(N);
    driver<double>(N);

    return 0;
}

*/

struct TestQueueSubmitWithLocalAccessor : public ::testing::Test
{
    std::ifstream spirvFile;
    size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitWithLocalAccessor()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;

        spirvFile.open("./local_accessor_kernel_inttys_fp32.spv",
                       std::ios::binary | std::ios::ate);
        spirvFileSize_ = std::filesystem::file_size(
            "./local_accessor_kernel_inttys_fp32.spv");
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

    ~TestQueueSubmitWithLocalAccessor()
    {
        spirvFile.close();
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

struct TestQueueSubmitWithLocalAccessorFP64 : public ::testing::Test
{
    std::ifstream spirvFile;
    size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitWithLocalAccessorFP64()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;

        spirvFile.open("./local_accessor_kernel_fp64.spv",
                       std::ios::binary | std::ios::ate);
        spirvFileSize_ =
            std::filesystem::file_size("./local_accessor_kernel_fp64.spv");
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

    ~TestQueueSubmitWithLocalAccessorFP64()
    {
        spirvFile.close();
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForInt8)
{
    submit_kernel<int8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                          DPCTLKernelArgType::DPCTL_INT8_T,
                          "_ZTS14SyclKernel_SLMIaE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForUInt8)
{
    submit_kernel<uint8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_UINT8_T,
                           "_ZTS14SyclKernel_SLMIhE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForInt16)
{
    submit_kernel<int16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_INT16_T,
                           "_ZTS14SyclKernel_SLMIsE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForUInt16)
{
    submit_kernel<uint16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                            DPCTLKernelArgType::DPCTL_UINT16_T,
                            "_ZTS14SyclKernel_SLMItE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForInt32)
{
    submit_kernel<int32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_INT32_T,
                           "_ZTS14SyclKernel_SLMIiE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForUInt32)
{
    submit_kernel<uint32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                            DPCTLKernelArgType::DPCTL_UINT32_T,
                            "_ZTS14SyclKernel_SLMIjE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForInt64)
{
    submit_kernel<int64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_INT64_T,
                           "_ZTS14SyclKernel_SLMIlE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForUInt64)
{
    submit_kernel<uint64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                            DPCTLKernelArgType::DPCTL_UINT64_T,
                            "_ZTS14SyclKernel_SLMImE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForFloat)
{
    submit_kernel<float>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                         DPCTLKernelArgType::DPCTL_FLOAT32_T,
                         "_ZTS14SyclKernel_SLMIfE");
}

TEST_F(TestQueueSubmitWithLocalAccessorFP64, CheckForDouble)
{
    submit_kernel<double>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                          DPCTLKernelArgType::DPCTL_FLOAT64_T,
                          "_ZTS14SyclKernel_SLMIdE");
}

TEST_F(TestQueueSubmitWithLocalAccessor, CheckForUnsupportedArgTy)
{
    size_t gRange[] = {SIZE};
    size_t lRange[] = {SIZE / 10};
    size_t RANGE_NDIMS = 1;
    constexpr size_t NARGS = 2;

    auto la = MDLocalAccessor{1, DPCTL_UNSUPPORTED_KERNEL_ARG, SIZE / 10, 1, 1};
    auto kernel = DPCTLKernelBundle_GetKernel(KBRef, "_ZTS14SyclKernel_SLMImE");
    void *args[NARGS] = {unwrap<void>(nullptr), (void *)&la};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR,
                                              DPCTL_LOCAL_ACCESSOR};
    auto ERef =
        DPCTLQueue_SubmitNDRange(kernel, QRef, args, addKernelArgTypes, NARGS,
                                 gRange, lRange, RANGE_NDIMS, nullptr, 0);

    ASSERT_TRUE(ERef == nullptr);
}
