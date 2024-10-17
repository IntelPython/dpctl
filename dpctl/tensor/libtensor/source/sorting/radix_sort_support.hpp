#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

template <typename Ty, typename ArgTy>
struct TypeDefinedEntry : std::bool_constant<std::is_same_v<Ty, ArgTy>>
{
    static constexpr bool is_defined = true;
};

struct NotDefinedEntry : std::true_type
{
    static constexpr bool is_defined = false;
};

template <typename T> struct RadixSortSupportVector
{
    using resolver_t =
        typename std::disjunction<TypeDefinedEntry<T, bool>,
                                  TypeDefinedEntry<T, std::int8_t>,
                                  TypeDefinedEntry<T, std::uint8_t>,
                                  TypeDefinedEntry<T, std::int16_t>,
                                  TypeDefinedEntry<T, std::uint16_t>,
                                  TypeDefinedEntry<T, std::int32_t>,
                                  TypeDefinedEntry<T, std::uint32_t>,
                                  TypeDefinedEntry<T, std::int64_t>,
                                  TypeDefinedEntry<T, std::uint64_t>,
                                  TypeDefinedEntry<T, sycl::half>,
                                  TypeDefinedEntry<T, float>,
                                  TypeDefinedEntry<T, double>,
                                  NotDefinedEntry>;

    static constexpr bool is_defined = resolver_t::is_defined;
};

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
