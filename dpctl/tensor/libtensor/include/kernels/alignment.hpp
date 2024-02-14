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

#pragma once

#include <cstddef>
#include <cstdint>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace alignment_utils
{

static constexpr size_t required_alignment = 64;

template <std::uintptr_t alignment, typename Ptr> bool is_aligned(Ptr p)
{
    return !(reinterpret_cast<std::uintptr_t>(p) % alignment);
}

template <typename KernelName> class disabled_sg_loadstore_wrapper_krn;

} // end of namespace alignment_utils
} // end of namespace kernels
} // end of namespace tensor
} // end of namespace dpctl
