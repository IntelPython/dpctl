#pragma once

#include <type_traits>

#include "reductions/reduction_atomic_support.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{
namespace atomic_support
{

template <typename fnT, typename T> struct DotAtomicSupportFactory
{
    fnT get()
    {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<T>::value) {
            return atomic_support::fixed_decision<false>;
        }
        else {
            return atomic_support::check_atomic_support<T>;
        }
    }
};

} // namespace atomic_support
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
