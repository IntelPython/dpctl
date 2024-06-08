#pragma once

#define SYCL_EXT_ONEAPI_COMPLEX
#if __has_include(<sycl/ext/oneapi/experimental/sycl_complex.hpp>)
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#else
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#endif

namespace exprm_ns = sycl::ext::oneapi::experimental;
