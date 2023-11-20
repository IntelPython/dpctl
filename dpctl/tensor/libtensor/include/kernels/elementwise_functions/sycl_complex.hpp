#pragma once
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
#define SYCL_EXT_ONEAPI_COMPLEX
#if __has_include(<sycl/ext/oneapi/experimental/sycl_complex.hpp>)
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#else
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#endif
#endif

#ifdef USE_SYCL_FOR_COMPLEX_TYPES
namespace exprm_ns = sycl::ext::oneapi::experimental;
#endif
