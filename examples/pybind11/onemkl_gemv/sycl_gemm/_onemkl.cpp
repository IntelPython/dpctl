// clang-format off
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dpctl4pybind11.hpp"
// clang-format on

namespace py = pybind11;

using dpctl::utils::keep_args_alive;

std::pair<sycl::event, sycl::event>
gemv(sycl::queue q,
     dpctl::tensor::usm_ndarray matrix,
     dpctl::tensor::usm_ndarray vector,
     dpctl::tensor::usm_ndarray result,
     const std::vector<sycl::event> &depends = {})
{
    if (matrix.get_ndim() != 2 || vector.get_ndim() != 1 ||
        result.get_ndim() != 1) {
        throw std::runtime_error(
            "Inconsistent dimensions, expecting matrix and a vector");
    }

    py::ssize_t n = matrix.get_shape(0); // get 0-th element of the shape
    py::ssize_t m = matrix.get_shape(1);

    py::ssize_t v_dim = vector.get_shape(0);
    py::ssize_t r_dim = result.get_shape(0);
    if (v_dim != m || r_dim != n) {
        throw std::runtime_error("Inconsistent shapes.");
    }

    int mat_flags = matrix.get_flags();
    int v_flags = vector.get_flags();
    int r_flags = result.get_flags();

    if (!((mat_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (v_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (r_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS))))
    {
        throw std::runtime_error("Arrays must be contiguous.");
    }

    int mat_typenum = matrix.get_typenum();
    int v_typenum = vector.get_typenum();
    int r_typenum = result.get_typenum();

    if ((mat_typenum != v_typenum) || (r_typenum != v_typenum) ||
        !((v_typenum == UAR_DOUBLE) || (v_typenum == UAR_FLOAT) ||
          (v_typenum == UAR_CDOUBLE) || (v_typenum == UAR_CFLOAT)))
    {
        std::cout << "Found: [" << mat_typenum << ", " << v_typenum << ", "
                  << r_typenum << "]" << std::endl;
        std::cout << "Expected: [" << UAR_DOUBLE << ", " << UAR_FLOAT << ", "
                  << UAR_CDOUBLE << ", " << UAR_CFLOAT << "]" << std::endl;
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    char *mat_typeless_ptr = matrix.get_data();
    char *v_typeless_ptr = vector.get_data();
    char *r_typeless_ptr = result.get_data();

    sycl::event res_ev;
    if (v_typenum == UAR_DOUBLE) {
        using T = double;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (v_typenum == UAR_FLOAT) {
        using T = float;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (v_typenum == UAR_CDOUBLE) {
        using T = std::complex<double>;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (v_typenum == UAR_CFLOAT) {
        using T = std::complex<float>;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
    }

    sycl::event ht_event =
        keep_args_alive(q, {matrix, vector, result}, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

PYBIND11_MODULE(_onemkl, m)
{
    // Import the dpctl extensions
    import_dpctl();
    m.def("gemv", &gemv, "Uses oneMKL to compute dot(matrix, vector)");
}
