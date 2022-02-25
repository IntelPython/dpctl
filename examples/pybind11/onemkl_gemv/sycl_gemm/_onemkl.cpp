#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/* DPCTL C-API for usm_ndarray
     UsmNDArray_GetData
     UsmNDArray_GetNDim
     UsmNDArray_GetShape
     UsmNDArray_GetStrides
     UsmNDArray_GetTypenum
     UsmNDArray_GetFlags
     UsmNDArray_GetQueueRef
 */

sycl::event keep_args_alive(sycl::queue q,
                            py::object o1,
                            py::object o2,
                            py::object o3,
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event ht_event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        std::shared_ptr<py::handle> shp1 = std::make_shared<py::handle>(o1);
        std::shared_ptr<py::handle> shp2 = std::make_shared<py::handle>(o2);
        std::shared_ptr<py::handle> shp3 = std::make_shared<py::handle>(o3);
        shp1->inc_ref();
        shp2->inc_ref();
        shp3->inc_ref();
        cgh.host_task([=]() {
            bool guard = (Py_IsInitialized() && !_Py_IsFinalizing());
            if (guard) {
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                shp1->dec_ref();
                shp2->dec_ref();
                shp3->dec_ref();
                PyGILState_Release(gstate);
            }
        });
    });
    return ht_event;
}

std::pair<sycl::event, sycl::event>
gemv(sycl::queue q,
     py::object matrix,
     py::object vector,
     py::object result,
     const std::vector<sycl::event> &depends = {})
{
    PyObject *m_src = matrix.ptr();
    if (!PyObject_TypeCheck(m_src, &PyUSMArrayType)) {
        throw std::runtime_error("Matrix is not a dpctl.tensor.usm_ndarray");
    }

    PyObject *v_src = vector.ptr();
    if (!PyObject_TypeCheck(v_src, &PyUSMArrayType)) {
        throw std::runtime_error("Vector is not a dpctl.tensor.usm_ndarray");
    }

    PyObject *r_src = result.ptr();
    if (!PyObject_TypeCheck(r_src, &PyUSMArrayType)) {
        throw std::runtime_error("Result is not a dpctl.tensor.usm_ndarray");
    }

    PyUSMArrayObject *m_usm_ary = reinterpret_cast<PyUSMArrayObject *>(m_src);
    PyUSMArrayObject *v_usm_ary = reinterpret_cast<PyUSMArrayObject *>(v_src);
    PyUSMArrayObject *r_usm_ary = reinterpret_cast<PyUSMArrayObject *>(r_src);

    if (UsmNDArray_GetNDim(m_usm_ary) != 2 ||
        UsmNDArray_GetNDim(v_usm_ary) != 1 ||
        UsmNDArray_GetNDim(r_usm_ary) != 1)
    {
        throw std::runtime_error(
            "Inconsistent dimensions, expecting matrix and a vector");
    }

    py::ssize_t *m_sh = UsmNDArray_GetShape(m_usm_ary);
    py::ssize_t n = m_sh[0];
    py::ssize_t m = m_sh[1];

    py::ssize_t *v_sh = UsmNDArray_GetShape(v_usm_ary);
    py::ssize_t *r_sh = UsmNDArray_GetShape(r_usm_ary);
    if (v_sh[0] != m || r_sh[0] != n) {
        throw std::runtime_error("Inconsistent shapes.");
    }

    int mat_flags = UsmNDArray_GetFlags(m_usm_ary);
    int v_flags = UsmNDArray_GetFlags(v_usm_ary);
    int r_flags = UsmNDArray_GetFlags(r_usm_ary);

    if (!((mat_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (v_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)) &&
          (r_flags & (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS))))
    {
        throw std::runtime_error("Arrays must be contiguous.");
    }

    int mat_typenum = UsmNDArray_GetTypenum(m_usm_ary);
    int v_typenum = UsmNDArray_GetTypenum(v_usm_ary);
    int r_typenum = UsmNDArray_GetTypenum(r_usm_ary);

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

    char *mat_typeless_ptr = UsmNDArray_GetData(m_usm_ary);
    char *v_typeless_ptr = UsmNDArray_GetData(v_usm_ary);
    char *r_typeless_ptr = UsmNDArray_GetData(r_usm_ary);

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

    sycl::event ht_event = keep_args_alive(q, matrix, vector, result, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

PYBIND11_MODULE(_onemkl, m)
{
    // Import the dpctl extensions
    import_dpctl();
    m.def("gemv", &gemv, "Uses oneMKL to compute dot(matrix, vector)");
}
