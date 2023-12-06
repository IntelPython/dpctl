#pragma once

#include "utils/math_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

template <typename cT> struct ComplexLess
{
    bool operator()(const cT &v1, const cT &v2) const
    {
        using dpctl::tensor::math_utils::less_complex;

        return less_complex(v1, v2);
    }
};

template <typename cT> struct ComplexGreater
{
    bool operator()(const cT &v1, const cT &v2) const
    {
        using dpctl::tensor::math_utils::greater_complex;

        return greater_complex(v1, v2);
    }
};

template <typename argTy> struct AscendingSorter
{
    using type = std::less<argTy>;
};

template <typename T> struct AscendingSorter<std::complex<T>>
{
    using type = ComplexLess<std::complex<T>>;
};

template <typename argTy> struct DescendingSorter
{
    using type = std::greater<argTy>;
};

template <typename T> struct DescendingSorter<std::complex<T>>
{
    using type = ComplexGreater<std::complex<T>>;
};

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
