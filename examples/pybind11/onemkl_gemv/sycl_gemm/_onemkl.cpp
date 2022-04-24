// clang-format off
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
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

template <typename T> class sub_kern;

template <typename T>
sycl::event sub_impl(sycl::queue q,
                     size_t n,
                     const char *v1_i,
                     const char *v2_i,
                     char *r_i,
                     const std::vector<sycl::event> &depends = {})
{
    const T *v1 = reinterpret_cast<const T *>(v1_i);
    const T *v2 = reinterpret_cast<const T *>(v2_i);
    T *r = reinterpret_cast<T *>(r_i);

    sycl::event r_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<sub_kern<T>>(sycl::range<1>{n}, [=](sycl::id<1> id) {
            auto i = id.get(0);
            r[i] = v1[i] - v2[i];
        });
    });

    return r_ev;
}

// out_r = in_v1 - in_v2
std::pair<sycl::event, sycl::event>
sub(sycl::queue q,
    dpctl::tensor::usm_ndarray in_v1,
    dpctl::tensor::usm_ndarray in_v2,
    dpctl::tensor::usm_ndarray out_r,
    const std::vector<sycl::event> &depends = {})
{
    if (in_v1.get_ndim() != 1 || in_v2.get_ndim() != 1 || out_r.get_ndim() != 1)
    {
        throw std::runtime_error("Inconsistent dimensions, expecting vectors");
    }

    py::ssize_t n = in_v1.get_shape(0); // get length of the vector

    if (n != in_v2.get_shape(0) || n != out_r.get_shape(0)) {
        throw std::runtime_error("Vectors must have the same length");
    }

    int in_v1_flags = in_v1.get_flags();
    int in_v2_flags = in_v2.get_flags();
    int out_r_flags = out_r.get_flags();

    if (!((in_v1_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (in_v2_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (out_r_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS))))
    {
        throw std::runtime_error("Vectors must be contiguous.");
    }

    int in_v1_typenum = in_v1.get_typenum();
    int in_v2_typenum = in_v2.get_typenum();
    int out_r_typenum = out_r.get_typenum();

    if ((in_v2_typenum != in_v1_typenum) || (out_r_typenum != in_v1_typenum) ||
        !((in_v1_typenum == UAR_DOUBLE) || (in_v1_typenum == UAR_FLOAT) ||
          (in_v1_typenum == UAR_CDOUBLE) || (in_v1_typenum == UAR_CFLOAT)))
    {
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    const char *in_v1_typeless_ptr = in_v1.get_data();
    const char *in_v2_typeless_ptr = in_v2.get_data();
    char *out_r_typeless_ptr = out_r.get_data();

    sycl::event res_ev;
    if (out_r_typenum == UAR_DOUBLE) {
        using T = double;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (out_r_typenum == UAR_FLOAT) {
        using T = float;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (out_r_typenum == UAR_CDOUBLE) {
        using T = std::complex<double>;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (out_r_typenum == UAR_CFLOAT) {
        using T = std::complex<float>;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
    }

    sycl::event ht_event = keep_args_alive(q, {in_v1, in_v2, out_r}, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

template <typename T> class axbpy_inplace_kern;

template <typename T>
sycl::event axbpy_inplace_impl(sycl::queue q,
                               size_t nelems,
                               py::object pyobj_a,
                               const char *x_typeless,
                               py::object pyobj_b,
                               char *y_typeless,
                               const std::vector<sycl::event> depends = {})
{
    T a = py::cast<T>(pyobj_a);
    T b = py::cast<T>(pyobj_b);

    const T *x = reinterpret_cast<const T *>(x_typeless);
    T *y = reinterpret_cast<T *>(y_typeless);

    sycl::event res_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<axbpy_inplace_kern<T>>(sycl::range<1>{nelems},
                                                [=](sycl::id<1> id) {
                                                    auto i = id.get(0);
                                                    y[i] = a * x[i] + b * y[i];
                                                });
    });

    return res_ev;
}

// y = a * x + b * y
std::pair<sycl::event, sycl::event>
axbpy_inplace(sycl::queue q,
              py::object a,
              dpctl::tensor::usm_ndarray x,
              py::object b,
              dpctl::tensor::usm_ndarray y,
              const std::vector<sycl::event> &depends = {})
{

    if (x.get_ndim() != 1 || y.get_ndim() != 1) {
        throw std::runtime_error("Inconsistent dimensions, expecting vectors");
    }

    py::ssize_t n = x.get_shape(0); // get length of the vector

    if (n != y.get_shape(0)) {
        throw std::runtime_error("Vectors must have the same length");
    }

    int x_flags = x.get_flags();
    int y_flags = y.get_flags();

    if (!((x_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (y_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS))))
    {
        throw std::runtime_error("Vectors must be contiguous.");
    }

    int x_typenum = x.get_typenum();
    int y_typenum = y.get_typenum();

    if ((x_typenum != y_typenum) ||
        !((x_typenum == UAR_DOUBLE) || (x_typenum == UAR_FLOAT) ||
          (x_typenum == UAR_CDOUBLE) || (x_typenum == UAR_CFLOAT)))
    {
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    const char *x_typeless_ptr = x.get_data();
    char *y_typeless_ptr = y.get_data();

    sycl::event res_ev;
    if (x_typenum == UAR_DOUBLE) {
        using T = double;
        res_ev = axbpy_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (x_typenum == UAR_FLOAT) {
        using T = float;
        res_ev = axbpy_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (x_typenum == UAR_CDOUBLE) {
        using T = std::complex<double>;
        res_ev = axbpy_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (x_typenum == UAR_CFLOAT) {
        using T = std::complex<float>;
        res_ev = axbpy_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
    }

    sycl::event ht_event = keep_args_alive(q, {x, y}, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

template <typename T> class norm_squared_blocking_kern;

template <typename T>
T norm_squared_blocking_impl(sycl::queue q,
                             size_t nelems,
                             const char *r_typeless,
                             const std::vector<sycl::event> &depends = {})
{
    const T *r = reinterpret_cast<const T *>(r_typeless);

    sycl::buffer<T, 1> sum_sq_buf(sycl::range<1>{1});

    q.submit([&](sycl::handler &cgh) {
         cgh.depends_on(depends);
         auto sum_sq_reduction = sycl::reduction(
             sum_sq_buf, cgh, sycl::plus<T>(),
             {sycl::property::reduction::initialize_to_identity{}});
         cgh.parallel_for<norm_squared_blocking_kern<T>>(
             sycl::range<1>{nelems}, sum_sq_reduction,
             [=](sycl::id<1> id, auto &sum_sq) {
                 auto i = id.get(0);
                 sum_sq += r[i] * r[i];
             });
     }).wait_and_throw();

    sycl::host_accessor ha(sum_sq_buf);
    return ha[0];
}

py::object norm_squared_blocking(sycl::queue q,
                                 dpctl::tensor::usm_ndarray r,
                                 const std::vector<sycl::event> depends = {})
{
    if (r.get_ndim() != 1) {
        throw std::runtime_error("Expecting a vector");
    }

    py::ssize_t n = r.get_shape(0); // get length of the vector

    int r_flags = r.get_flags();

    if (!(r_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS))) {
        throw std::runtime_error("Vector must be contiguous.");
    }

    int r_typenum = r.get_typenum();
    if ((r_typenum != UAR_DOUBLE) && (r_typenum != UAR_FLOAT) &&
        (r_typenum != UAR_CDOUBLE) && (r_typenum != UAR_CFLOAT))
    {
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    const char *r_typeless_ptr = r.get_data();
    py::object res;

    if (r_typenum == UAR_DOUBLE) {
        using T = double;
        T n_sq = norm_squared_blocking_impl<T>(q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (r_typenum == UAR_FLOAT) {
        using T = float;
        T n_sq = norm_squared_blocking_impl<T>(q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
#if 0
    else if (r_typenum == UAR_CDOUBLE) {
        using T = std::complex<double>;
        double n_sq = norm_squared_blocking_impl<T>(
	    q, n, r_typeless_ptr,
	    depends);
	res = py::float_(n_sq);
    }
    else if (r_typenum == UAR_CFLOAT) {
        using T = std::complex<float>;
        float n_sq = norm_squared_blocking_impl<T>(
	    q, n, r_typeless_ptr,
	    depends);
	res = py::float_(n_sq);
    }
#endif
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
    }

    return res;
}

PYBIND11_MODULE(_onemkl, m)
{
    // Import the dpctl extensions
    import_dpctl();
    m.def("gemv", &gemv, "Uses oneMKL to compute dot(matrix, vector)",
          py::arg("exec_queue"), py::arg("Amatrix"), py::arg("xvec"),
          py::arg("resvec"), py::arg("depends") = py::list());
    m.def("sub", &sub, "Subtraction: out = v1 - v2", py::arg("exec_queue"),
          py::arg("in1"), py::arg("in2"), py::arg("out"),
          py::arg("depends") = py::list());
    m.def("axbpy_inplace", &axbpy_inplace, "y = a * x + b * y",
          py::arg("exec_queue"), py::arg("a"), py::arg("x"), py::arg("b"),
          py::arg("y"), py::arg("depends") = py::list());
    m.def("norm_squared_blocking", &norm_squared_blocking, "norm(r)**2",
          py::arg("exec_queue"), py::arg("r"), py::arg("depends") = py::list());
}
