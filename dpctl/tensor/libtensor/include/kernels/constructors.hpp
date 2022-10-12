//=== constructors.hpp -  -----------------------------------*-C++-*--/===//
//===              Implementation of tensor constructors kernels ------===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// This file defines kernels for tensor constructors.
//===----------------------------------------------------------------------===//

#pragma once
#include "utils/strided_iters.hpp"
#include "utils/type_utils.hpp"
#include <CL/sycl.hpp>
#include <complex>
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace constructors
{

/*!
  @defgroup CtorKernels
 */

template <typename Ty> class linear_sequence_step_kernel;
template <typename Ty, typename wTy> class linear_sequence_affine_kernel;
template <typename Ty> class eye_kernel;

namespace py = pybind11;

/* =========== Unboxing Python scalar =============== */

/*!
 * @brief Cast pybind11 class managing Python object to specified type `T`.
 * @defgroup CtorKernels
 */
template <typename T> T unbox_py_scalar(py::object o)
{
    return py::cast<T>(o);
}

template <> inline sycl::half unbox_py_scalar<sycl::half>(py::object o)
{
    float tmp = py::cast<float>(o);
    return static_cast<sycl::half>(tmp);
}

// Constructor to populate tensor with linear sequence defined by
// start and step data

typedef sycl::event (*lin_space_step_fn_ptr_t)(
    sycl::queue,
    size_t, // num_elements
    py::object start,
    py::object step,
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

template <typename Ty> class LinearSequenceStepFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty step_v;

public:
    LinearSequenceStepFunctor(char *dst_p, Ty v0, Ty dv)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<Ty>::value) {
            p[i] = Ty{start_v.real() + i * step_v.real(),
                      start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + i * step_v;
        }
    }
};

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified by typed starting value and
 * increment.
 *
 * @param q  Sycl queue to which the kernel is submitted
 * @param nelems Length of the sequence
 * @param start_v Typed starting value of the sequence
 * @param step_v  Typed increment of the sequence
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_step_impl(sycl::queue exec_q,
                                size_t nelems,
                                Ty start_v,
                                Ty step_v,
                                char *array_data,
                                const std::vector<sycl::event> &depends)
{
    sycl::event lin_space_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<linear_sequence_step_kernel<Ty>>(
            sycl::range<1>{nelems},
            LinearSequenceStepFunctor<Ty>(array_data, start_v, step_v));
    });

    return lin_space_step_event;
}

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified by starting value and increment
 * given as Python objects.
 *
 * @param q  Sycl queue to which the kernel is submitted
 * @param nelems Length of the sequence
 * @param start Starting value of the sequence as Python object. Must be
 * convertible to array element data type `Ty`.
 * @param step  Increment of the sequence as Python object. Must be convertible
 * to array element data type `Ty`.
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_step_impl(sycl::queue exec_q,
                                size_t nelems,
                                py::object start,
                                py::object step,
                                char *array_data,
                                const std::vector<sycl::event> &depends)
{
    Ty start_v;
    Ty step_v;
    try {
        start_v = unbox_py_scalar<Ty>(start);
        step_v = unbox_py_scalar<Ty>(step);
    } catch (const py::error_already_set &e) {
        throw;
    }

    auto lin_space_step_event = lin_space_step_impl<Ty>(
        exec_q, nelems, start_v, step_v, array_data, depends);

    return lin_space_step_event;
}

/*!
 * @brief  Factor to get function pointer of type `fnT` for array with elements
 * of type `Ty`.
 * @defgroup CtorKernels
 */
template <typename fnT, typename Ty> struct LinSpaceStepFactory
{
    fnT get()
    {
        fnT f = lin_space_step_impl<Ty>;
        return f;
    }
};

// Constructor to populate tensor with linear sequence defined by
// start and and data

typedef sycl::event (*lin_space_affine_fn_ptr_t)(
    sycl::queue,
    size_t, // num_elements
    py::object start,
    py::object end,
    bool include_endpoint,
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

template <typename Ty, typename wTy> class LinearSequenceAffineFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty end_v;
    size_t n;

public:
    LinearSequenceAffineFunctor(char *dst_p, Ty v0, Ty v1, size_t den)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), end_v(v1),
          n((den == 0) ? 1 : den)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        wTy wc = wTy(i) / n;
        wTy w = wTy(n - i) / n;
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<Ty>::value) {
            auto _w = static_cast<typename Ty::value_type>(w);
            auto _wc = static_cast<typename Ty::value_type>(wc);
            auto re_comb = start_v.real() * _w + end_v.real() * _wc;
            auto im_comb = start_v.imag() * _w + end_v.imag() * _wc;
            Ty affine_comb = Ty{re_comb, im_comb};
            p[i] = affine_comb;
        }
        else {
            using dpctl::tensor::type_utils::convert_impl;
            auto affine_comb = start_v * w + end_v * wc;
            p[i] = convert_impl<Ty, decltype(affine_comb)>(affine_comb);
        }
    }
};

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified by typed starting and end values.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence.
 * @param start_v Stating value of the sequence.
 * @param end_v   End-value of the sequence.
 * @param include_endpoint  Whether the end-value is included in the sequence.
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_affine_impl(sycl::queue exec_q,
                                  size_t nelems,
                                  Ty start_v,
                                  Ty end_v,
                                  bool include_endpoint,
                                  char *array_data,
                                  const std::vector<sycl::event> &depends)
{
    bool device_supports_doubles = exec_q.get_device().has(sycl::aspect::fp64);
    sycl::event lin_space_affine_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        if (device_supports_doubles) {
            cgh.parallel_for<linear_sequence_affine_kernel<Ty, double>>(
                sycl::range<1>{nelems},
                LinearSequenceAffineFunctor<Ty, double>(
                    array_data, start_v, end_v,
                    (include_endpoint) ? nelems - 1 : nelems));
        }
        else {
            cgh.parallel_for<linear_sequence_affine_kernel<Ty, float>>(
                sycl::range<1>{nelems},
                LinearSequenceAffineFunctor<Ty, float>(
                    array_data, start_v, end_v,
                    (include_endpoint) ? nelems - 1 : nelems));
        }
    });

    return lin_space_affine_event;
}

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified  by starting and end values given
 * as Python objects.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param start Stating value of the sequence as Python object. Must be
 * convertible to array data element type `Ty`.
 * @param end   End-value of the sequence as Python object. Must be convertible
 * to array data element type `Ty`.
 * @param include_endpoint  Whether the end-value is included in the sequence
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_affine_impl(sycl::queue exec_q,
                                  size_t nelems,
                                  py::object start,
                                  py::object end,
                                  bool include_endpoint,
                                  char *array_data,
                                  const std::vector<sycl::event> &depends)
{
    Ty start_v, end_v;
    try {
        start_v = unbox_py_scalar<Ty>(start);
        end_v = unbox_py_scalar<Ty>(end);
    } catch (const py::error_already_set &e) {
        throw;
    }

    auto lin_space_affine_event = lin_space_affine_impl<Ty>(
        exec_q, nelems, start_v, end_v, include_endpoint, array_data, depends);

    return lin_space_affine_event;
}

/*!
 * @brief Factory to get function pointer of type `fnT` for array data type
 * `Ty`.
 */
template <typename fnT, typename Ty> struct LinSpaceAffineFactory
{
    fnT get()
    {
        fnT f = lin_space_affine_impl<Ty>;
        return f;
    }
};

/* ================ Full ================== */

typedef sycl::event (*full_contig_fn_ptr_t)(sycl::queue,
                                            size_t,
                                            py::object,
                                            char *,
                                            const std::vector<sycl::event> &);

/*!
 * @brief Function to submit kernel to fill given contiguous memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param fill_v  Value to fill the array with
 * @param dst_p Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_contig_impl(sycl::queue q,
                             size_t nelems,
                             dstTy fill_v,
                             char *dst_p,
                             const std::vector<sycl::event> &depends)
{
    sycl::event fill_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        dstTy *p = reinterpret_cast<dstTy *>(dst_p);
        cgh.fill<dstTy>(p, fill_v, nelems);
    });

    return fill_ev;
}

/*!
 * @brief Function to submit kernel to fill given contiguous memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param py_value Python object representing the value to fill the array with.
 * Must be convertible to `dstTy`.
 * @param dst_p Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_contig_impl(sycl::queue exec_q,
                             size_t nelems,
                             py::object py_value,
                             char *dst_p,
                             const std::vector<sycl::event> &depends)
{
    dstTy fill_v;
    try {
        fill_v = unbox_py_scalar<dstTy>(py_value);
    } catch (const py::error_already_set &e) {
        throw;
    }

    sycl::event fill_ev =
        full_contig_impl<dstTy>(exec_q, nelems, fill_v, dst_p, depends);

    return fill_ev;
}

template <typename fnT, typename Ty> struct FullContigFactory
{
    fnT get()
    {
        fnT f = full_contig_impl<Ty>;
        return f;
    }
};

/* ================ Eye ================== */

typedef sycl::event (*eye_fn_ptr_t)(sycl::queue,
                                    size_t nelems, // num_elements
                                    py::ssize_t start,
                                    py::ssize_t end,
                                    py::ssize_t step,
                                    char *, // dst_data_ptr
                                    const std::vector<sycl::event> &);

template <typename Ty> class EyeFunctor
{
private:
    Ty *p = nullptr;
    py::ssize_t start_v;
    py::ssize_t end_v;
    py::ssize_t step_v;

public:
    EyeFunctor(char *dst_p,
               const py::ssize_t v0,
               const py::ssize_t v1,
               const py::ssize_t dv)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), end_v(v1), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        Ty set_v = 0;
        py::ssize_t i = static_cast<py::ssize_t>(wiid.get(0));
        if (i >= start_v and i <= end_v) {
            if ((i - start_v) % step_v == 0) {
                set_v = 1;
            }
        }
        p[i] = set_v;
    }
};

/*!
 * @brief Function to populate 2D array with eye matrix.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Number of elements to assign.
 * @param start   Position of the first non-zero value.
 * @param end     Position of the last non-zero value.
 * @param step    Number of array elements between non-zeros.
 * @param array_data Kernel accessible USM pointer for the destination array.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return  Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event eye_impl(sycl::queue exec_q,
                     size_t nelems,
                     const py::ssize_t start,
                     const py::ssize_t end,
                     const py::ssize_t step,
                     char *array_data,
                     const std::vector<sycl::event> &depends)
{
    sycl::event eye_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<eye_kernel<Ty>>(
            sycl::range<1>{nelems},
            EyeFunctor<Ty>(array_data, start, end, step));
    });

    return eye_event;
}

/*!
 * @brief  Factory to get function pointer of type `fnT` for data type `Ty`.
 * @ingroup CtorKernels
 */
template <typename fnT, typename Ty> struct EyeFactory
{
    fnT get()
    {
        fnT f = eye_impl<Ty>;
        return f;
    }
};

/* =========================== Tril and triu ============================== */

// define function type
typedef sycl::event (*tri_fn_ptr_t)(sycl::queue,
                                    py::ssize_t,   // inner_range  //py::ssize_t
                                    py::ssize_t,   // outer_range
                                    char *,        // src_data_ptr
                                    char *,        // dst_data_ptr
                                    py::ssize_t,   // nd
                                    py::ssize_t *, // shape_and_strides
                                    py::ssize_t,   // k
                                    const std::vector<sycl::event> &,
                                    const std::vector<sycl::event> &);

/*!
 * @brief Function to copy triangular matrices from source stack to destination
 * stack.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param inner_range  Number of elements in each matrix.
 * @param outer_range  Number of matrices to copy.
 * @param src_p  Kernel accessible USM pointer for the source array.
 * @param dst_p  Kernel accessible USM pointer for the destination array.
 * @param nd  The array dimensionality of source and destination arrays.
 * @param shape_and_strides  Kernel accessible USM pointer to packed shape and
 * strides of arrays.
 * @param k Position of the diagonal above/below which to copy filling the rest
 * with zero elements.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 * @param additional_depends  List of additional events to wait for before
 * starting computations, if any.
 *
 * @return  Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty, bool> class tri_kernel;
template <typename Ty, bool upper>
sycl::event tri_impl(sycl::queue exec_q,
                     py::ssize_t inner_range,
                     py::ssize_t outer_range,
                     char *src_p,
                     char *dst_p,
                     py::ssize_t nd,
                     py::ssize_t *shape_and_strides,
                     py::ssize_t k,
                     const std::vector<sycl::event> &depends,
                     const std::vector<sycl::event> &additional_depends)
{
    constexpr int d2 = 2;
    py::ssize_t src_s = nd;
    py::ssize_t dst_s = 2 * nd;
    py::ssize_t nd_1 = nd - 1;
    py::ssize_t nd_2 = nd - 2;
    Ty *src = reinterpret_cast<Ty *>(src_p);
    Ty *dst = reinterpret_cast<Ty *>(dst_p);

    sycl::event tri_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);
        cgh.parallel_for<tri_kernel<Ty, upper>>(
            sycl::range<1>(inner_range * outer_range), [=](sycl::id<1> idx) {
                py::ssize_t outer_gid = idx[0] / inner_range;
                py::ssize_t inner_gid = idx[0] - inner_range * outer_gid;

                py::ssize_t src_inner_offset, dst_inner_offset;
                bool to_copy;

                {
                    // py::ssize_t inner_gid = idx.get_id(0);
                    CIndexer_array<d2, py::ssize_t> indexer_i(
                        {shape_and_strides[nd_2], shape_and_strides[nd_1]});
                    indexer_i.set(inner_gid);
                    const std::array<py::ssize_t, d2> &inner = indexer_i.get();
                    src_inner_offset =
                        inner[0] * shape_and_strides[src_s + nd_2] +
                        inner[1] * shape_and_strides[src_s + nd_1];
                    dst_inner_offset =
                        inner[0] * shape_and_strides[dst_s + nd_2] +
                        inner[1] * shape_and_strides[dst_s + nd_1];

                    if constexpr (upper)
                        to_copy = (inner[0] + k >= inner[1]);
                    else
                        to_copy = (inner[0] + k <= inner[1]);
                }

                py::ssize_t src_offset = 0;
                py::ssize_t dst_offset = 0;
                {
                    // py::ssize_t outer_gid = idx.get_id(1);
                    CIndexer_vector<py::ssize_t> outer(nd - d2);
                    outer.get_displacement(
                        outer_gid, shape_and_strides, shape_and_strides + src_s,
                        shape_and_strides + dst_s, src_offset, dst_offset);
                }

                src_offset += src_inner_offset;
                dst_offset += dst_inner_offset;

                dst[dst_offset] = (to_copy) ? src[src_offset] : Ty(0);
            });
    });
    return tri_ev;
}

/*!
 * @brief  Factory to get function pointer of type `fnT` for data type `Ty`.
 * @ingroup CtorKernels
 */
template <typename fnT, typename Ty> struct TrilGenericFactory
{
    fnT get()
    {
        fnT f = tri_impl<Ty, /*tril*/ true>;
        return f;
    }
};

/*!
 * @brief  Factory to get function pointer of type `fnT` for data type `Ty`.
 * @ingroup CtorKernels
 */
template <typename fnT, typename Ty> struct TriuGenericFactory
{
    fnT get()
    {
        fnT f = tri_impl<Ty, /*triu*/ false>;
        return f;
    }
};

} // namespace constructors
} // namespace kernels
} // namespace tensor
} // namespace dpctl
