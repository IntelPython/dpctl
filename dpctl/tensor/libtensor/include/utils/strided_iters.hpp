//===-- strided_iters.cpp - CIndexer classes for strided iteration ---*-C++-*-
//===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
/// This file defines CIndexer_array, and CIndexer_vector classes, as well
/// iteration space simplifiers.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm> // sort
#include <array>
#include <numeric> // std::iota
#include <vector>

/* An N-dimensional array can be stored in a single
 * contiguous chunk of memory by contiguously laying
 * array elements in lexicographinc order of their
 * array indices. Such a layout is called C-contiguous.
 *
 * E.g. for (2, 3, 2) array `a` with zero-based indexing convention
 * the C-array's elements are
 * { a[0,0,0], a[0,0,1], a[0,1,0], a[0,1,1], a[0,2,0], a[0,2,1],
 *   a[1,0,0], a[1,0,1], a[1,1,0], a[1,1,1], a[1,2,0], a[1,2,1] }
 *
 * Indexer maps zero-based index in C-array to a multi-index
 * for the purpose of computing element displacement in the
 * strided array, i.e. in the above example for k = 5, the displacement
 * is (s0*0 + s1*2 + s2*1), and for k = 7 it is (s0*1 + s1*0 + s2*1)
 * for N-dimensional array with strides (s0, s1, s2).
 *
 * Cindexer_vector need not know array rank `dim` at compile time.
 * Shape and strides are stored in std::vector, which are not trivially
 * copyable.
 *
 * For the class to be trivially copyable for offloading displacement
 * computation methods take accessor/pointer arguments of type T for
 * shape and stride and modify displacement argument passed by reference.
 */
template <typename indT = std::ptrdiff_t> class CIndexer_vector
{
    static_assert(std::is_integral<indT>::value, "Integral type is required");
    static_assert(std::is_signed<indT>::value,
                  "Signed integral type is required");
    int nd;

public:
    CIndexer_vector(int dim) : nd(dim) {}

    template <class ShapeTy> indT size(ShapeTy shape) const
    {
        indT s = static_cast<indT>(1);
        for (int i = 0; i < nd; ++i) {
            s *= shape[i];
        }
        return s;
    }

    template <class ShapeTy, class StridesTy>
    void
    get_displacement(indT i, ShapeTy shape, StridesTy stride, indT &disp) const
    {
        if (nd == 1) {
            disp = i * stride[0];
            return;
        }

        indT i_ = i;
        indT d = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            d += r * stride[dim];
            i_ = q;
        }
        disp = d + i_ * stride[0];
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(indT i,
                          ShapeTy shape,
                          StridesTy stride1,
                          StridesTy stride2,
                          indT &disp1,
                          indT &disp2) const
    {
        if (nd == 1) {
            disp1 = i * stride1[0];
            disp2 = i * stride2[0];
            return;
        }

        indT i_ = i;
        indT d1 = 0, d2 = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            i_ = q;
            d1 += r * stride1[dim];
            d2 += r * stride2[dim];
        }
        disp1 = d1 + i_ * stride1[0];
        disp2 = d2 + i_ * stride2[0];
        return;
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(indT i,
                          ShapeTy shape,
                          StridesTy stride1,
                          StridesTy stride2,
                          StridesTy stride3,
                          indT &disp1,
                          indT &disp2,
                          indT &disp3) const
    {
        if (nd == 1) {
            disp1 = i * stride1[0];
            disp2 = i * stride2[0];
            disp3 = i * stride3[0];
            return;
        }

        indT i_ = i;
        indT d1 = 0, d2 = 0, d3 = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            i_ = q;
            d1 += r * stride1[dim];
            d2 += r * stride2[dim];
            d3 += r * stride3[dim];
        };
        disp1 = d1 + i_ * stride1[0];
        disp2 = d2 + i_ * stride2[0];
        disp3 = d3 + i_ * stride3[0];
        return;
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(indT i,
                          ShapeTy shape,
                          StridesTy stride1,
                          StridesTy stride2,
                          StridesTy stride3,
                          StridesTy stride4,
                          indT &disp1,
                          indT &disp2,
                          indT &disp3,
                          indT &disp4) const
    {
        if (nd == 1) {
            disp1 = i * stride1[0];
            disp2 = i * stride2[0];
            disp3 = i * stride3[0];
            disp4 = i * stride4[0];
            return;
        }

        indT i_ = i;
        indT d1 = 0, d2 = 0, d3 = 0, d4 = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            i_ = q;
            d1 += r * stride1[dim];
            d2 += r * stride2[dim];
            d3 += r * stride3[dim];
            d4 += r * stride4[dim];
        }
        disp1 = d1 + i_ * stride1[0];
        disp2 = d2 + i_ * stride2[0];
        disp3 = d3 + i_ * stride3[0];
        disp4 = d4 + i_ * stride4[0];
        return;
    }

    template <class ShapeTy, class StridesTy, int nstrides>
    void get_displacement(indT i,
                          ShapeTy shape,
                          const std::array<StridesTy, nstrides> strides,
                          std::array<indT, nstrides> &disps) const
    {
        if (nd == 1) {
            for (int k = 0; k < nstrides; ++k) {
                disps[k] = i * strides[k][0];
            }
            return;
        }

        indT i_ = i;
        std::array<indT, nstrides> ds;
        for (int k = 0; k < nstrides; ++k) {
            ds[k] = 0;
        }

        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            for (int k = 0; k < nstrides; ++k) {
                ds[k] += r * strides[k][dim];
            }
            i_ = q;
        };
        for (int k = 0; k < nstrides; ++k) {
            disps[k] = ds[k] + i_ * strides[k][0];
        }
        return;
    }
};

/*
 * CIndexer is for arrays whose array-rank is known at compile time.
 * Statically allocated shape and multi_index arrays are members of
 * the class instance, and it remains trivially copyable.
 *
 * Method `set(k)` populates work-item private array multi_index, which
 * can be accessed using `get()` to compute the displacement as needed.
 */

template <int _ndim, typename indT = std::ptrdiff_t> class CIndexer_array
{
    static const int ndim = _ndim;

    static_assert(std::is_integral<indT>::value, "Integral type is required");
    static_assert(std::is_signed<indT>::value,
                  "Signed integral type is required");
    static_assert(ndim > 0, "Dimensionality must be positive");

private:
    typedef std::array<indT, ndim> index_t;

    indT elem_count;
    index_t shape;
    index_t multi_index;

public:
    CIndexer_array() : elem_count(0), shape{}, multi_index{} {}

    explicit CIndexer_array(const index_t &input_shape)
        : elem_count(0), shape{}, multi_index{}
    {
        indT s = static_cast<std::ptrdiff_t>(1);
        for (int i = 0; i < ndim; ++i) {
            shape[i] = input_shape[i];
            s *= input_shape[i];
        }
        elem_count = s;
    }

    indT size() const
    {
        return elem_count;
    }
    indT rank() const
    {
        return ndim;
    }

    void set(indT i)
    {
        if (ndim == 1) {
            multi_index[0] = i;
            return;
        }

        indT i_ = i;
        for (int dim = ndim; --dim > 0;) {
            indT si = shape[dim];
            indT q = i_ / si;
            multi_index[dim] = i_ - q * si;
            i_ = q;
        }
        multi_index[0] = i_;
    }

    const index_t &get() const
    {
        return multi_index;
    }
};

/*
    For purposes of iterating over elements of array with
    `shape` and `strides` given as pointers
    `simplify_iteration_strides(nd, shape_ptr, strides_ptr, disp)`
    may modify memory and returns new length of these arrays.

    The new shape and new strides, as well as the offset
    `(new_shape, new_strides, disp)` are such that iterating over
    them will traverse the same elements, possibly in
    different order.

    ..Example: python
        import itertools
        # for some array Y over whose elements we iterate
        csh, cst, cp = contract_iter(Y.shape, Y.strides)
        def pointers_set(sh, st, p):
            citers = itertools.product(*map(lambda s: range(s), sh))
            dot = lambda st, it: sum(st[k]*it[k] for k in range(len(st)))
            return set(p + dot(st, it) for it in citers)
        ps1 = pointers_set(csh, cst, cp)
        ps2 = pointers_set(Y.shape, Y.strides, 0)
        assert ps1 == ps2

 */
template <class ShapeTy, class StridesTy>
int simplify_iteration_stride(const int nd,
                              ShapeTy *shape,
                              StridesTy *strides,
                              StridesTy &disp)
{
    disp = std::ptrdiff_t(0);
    if (nd < 2)
        return nd;

    std::vector<int> pos(nd);
    std::iota(pos.begin(), pos.end(), 0);

    std::stable_sort(
        pos.begin(), pos.end(), [&strides, &shape](int i1, int i2) {
            auto abs_str1 = (strides[i1] < 0) ? -strides[i1] : strides[i1];
            auto abs_str2 = (strides[i2] < 0) ? -strides[i2] : strides[i2];
            return (abs_str1 > abs_str2) ||
                   (abs_str1 == abs_str2 && shape[i1] > shape[i2]);
        });

    std::vector<ShapeTy> shape_w;
    std::vector<StridesTy> strides_w;
    int nd_ = nd;
    shape_w.reserve(nd_);
    strides_w.reserve(nd_);

    for (int i = 0; i < nd; ++i) {
        auto p = pos[i];
        auto sh_p = shape[p];
        auto str_p = strides[p];
        shape_w.push_back(sh_p);
        if (str_p < 0) {
            disp += str_p * (sh_p - 1);
            str_p = -str_p;
        }
        strides_w.push_back(str_p);
    }

    {
        bool changed;
        do {
            changed = false;
            for (int i = 0; i + 1 < nd_; ++i) {
                StridesTy step = strides_w[i + 1];
                StridesTy jump = strides_w[i] - (shape_w[i + 1] - 1) * step;
                if (jump == step) {
                    changed = true;
                    for (int k = i; k + 1 < nd_; ++k) {
                        strides_w[k] = strides_w[k + 1];
                    }
                    shape_w[i] *= shape_w[i + 1];
                    for (int k = i + 1; k + 1 < nd_; ++k) {
                        shape_w[k] = shape_w[k + 1];
                    }
                    --nd_;
                }
            }
        } while (changed);
    }

    for (int i = 0; i < nd_; ++i) {
        shape[i] = shape_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides[i] = strides_w[i];
    }

    return nd_;
}

/*
    For purposes of iterating over pairs of elements of two arrays
    with  `shape` and strides `strides1`, `strides2` given as pointers
    `simplify_iteration_two_strides(nd, shape_ptr, strides1_ptr,
    strides2_ptr, disp1, disp2)`
    may modify memory and returns new length of these arrays.

    The new shape and new strides, as well as the offset
    `(new_shape, new_strides1, disp1, new_stride2, disp2)` are such that
    iterating over  them will traverse the same pairs of elements, possibly in
    different order.

 */
template <class ShapeTy, class StridesTy>
int simplify_iteration_two_strides(const int nd,
                                   ShapeTy *shape,
                                   StridesTy *strides1,
                                   StridesTy *strides2,
                                   StridesTy &disp1,
                                   StridesTy &disp2)
{
    disp1 = std::ptrdiff_t(0);
    disp2 = std::ptrdiff_t(0);
    if (nd < 2)
        return nd;

    std::vector<int> pos(nd);
    std::iota(pos.begin(), pos.end(), 0);

    std::stable_sort(
        pos.begin(), pos.end(), [&strides1, &shape](int i1, int i2) {
            auto abs_str1 = (strides1[i1] < 0) ? -strides1[i1] : strides1[i1];
            auto abs_str2 = (strides1[i2] < 0) ? -strides1[i2] : strides1[i2];
            return (abs_str1 > abs_str2) ||
                   (abs_str1 == abs_str2 && shape[i1] > shape[i2]);
        });

    std::vector<ShapeTy> shape_w;
    std::vector<StridesTy> strides1_w;
    std::vector<StridesTy> strides2_w;

    bool contractable = true;
    for (int i = 0; i < nd; ++i) {
        auto p = pos[i];
        auto sh_p = shape[p];
        auto str1_p = strides1[p];
        auto str2_p = strides2[p];
        shape_w.push_back(sh_p);
        if (str1_p < 0 && str2_p < 0) {
            disp1 += str1_p * (sh_p - 1);
            str1_p = -str1_p;
            disp2 += str2_p * (sh_p - 1);
            str2_p = -str2_p;
        }
        if (str1_p < 0 || str2_p < 0) {
            contractable = false;
        }
        strides1_w.push_back(str1_p);
        strides2_w.push_back(str2_p);
    }
    int nd_ = nd;
    while (contractable) {
        bool changed = false;
        for (int i = 0; i + 1 < nd_; ++i) {
            StridesTy str1 = strides1_w[i + 1];
            StridesTy str2 = strides2_w[i + 1];
            StridesTy jump1 = strides1_w[i] - (shape_w[i + 1] - 1) * str1;
            StridesTy jump2 = strides2_w[i] - (shape_w[i + 1] - 1) * str2;

            if (jump1 == str1 and jump2 == str2) {
                changed = true;
                shape_w[i] *= shape_w[i + 1];
                for (int j = i; j < nd_; ++j) {
                    strides1_w[j] = strides1_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides2_w[j] = strides2_w[j + 1];
                }
                for (int j = i + 1; j + 1 < nd_; ++j) {
                    shape_w[j] = shape_w[j + 1];
                }
                --nd_;
                break;
            }
        }
        if (!changed)
            break;
    }
    for (int i = 0; i < nd_; ++i) {
        shape[i] = shape_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides1[i] = strides1_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides2[i] = strides2_w[i];
    }

    return nd_;
}
