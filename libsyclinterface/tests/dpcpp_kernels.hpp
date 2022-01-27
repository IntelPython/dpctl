#pragma once
#ifndef __SYCL_INTERNAL_API
// make sure that sycl::program is defined and implemented
#define __SYCL_INTERNAL_API
#endif
#include <CL/sycl.hpp>

namespace dpcpp_kernels
{

namespace
{
template <typename T> class populate_a;

template <typename T> class populate_b;

template <typename T, typename scT> class mad_kern;

template <typename name, class kernelFunc>
auto make_cgh_function(int n, kernelFunc func)
{
    auto Kernel = [&](sycl::handler &cgh) {
        cgh.parallel_for<name>(sycl::range<1>(n), func);
    };
    return Kernel;
};

template <typename Ty, typename scT> struct MadFunc
{
    const Ty *in1, *in2;
    Ty *out;
    scT val;
    MadFunc(const Ty *a, const Ty *b, Ty *c, scT d)
        : in1(a), in2(b), out(c), val(d)
    {
    }
    void operator()(sycl::id<1> myId) const
    {
        auto gid = myId[0];
        out[gid] = in1[gid] + val * in2[gid];
        return;
    }
};

template <typename T> struct FillFunc
{
    T *out;
    T val;
    FillFunc(T *a, T val) : out(a), val(val) {}
    void operator()(sycl::id<1> myId) const
    {
        auto gid = myId[0];
        out[gid] = val;
        return;
    };
};

template <typename T> struct RangeFunc
{
    T *out;
    RangeFunc(T *b) : out(b) {}
    void operator()(sycl::id<1> myId) const
    {
        auto gid = myId[0];
        out[gid] = T(gid);
        return;
    };
};

} // namespace

template <typename T>
sycl::kernel get_fill_kernel(sycl::queue &q, size_t n, T *out, T fill_val)
{
    // out[i] = fill_val
    sycl::program program(q.get_context());

    [[maybe_unused]] auto cgh_fn =
        make_cgh_function<class populate_a<T>>(n, FillFunc<T>(out, fill_val));

    program.build_with_kernel_type<populate_a<T>>();
    return program.get_kernel<populate_a<T>>();
};

template <typename T>
sycl::kernel get_range_kernel(sycl::queue &q, size_t n, T *b)
{
    // b[i] = i
    sycl::program program(q.get_context());

    [[maybe_unused]] auto cgh_fn =
        make_cgh_function<class populate_b<T>>(n, RangeFunc<T>(b));

    program.build_with_kernel_type<populate_b<T>>();
    return program.get_kernel<populate_b<T>>();
};

template <typename T, typename scT>
sycl::kernel
get_mad_kernel(sycl::queue &q, size_t n, T *in1, T *in2, T *out, scT val)
{
    // c[i] = a[i] + b[i] * val
    sycl::program program(q.get_context());

    [[maybe_unused]] auto cgh_fn = make_cgh_function<class mad_kern<T, scT>>(
        n, MadFunc<T, scT>(in1, in2, out, val));

    program.build_with_kernel_type<mad_kern<T, scT>>();
    return program.get_kernel<mad_kern<T, scT>>();
};

template <typename name,
          typename localAccessorT,
          class KernelFuncArgs,
          class KernelFunctor>
auto make_cgh_nd_function_with_local_memory(const sycl::nd_range<1> &nd_range,
                                            size_t slm_size,
                                            KernelFuncArgs kern_params)
{
    auto Kernel = [&](sycl::handler &cgh) {
        localAccessorT lm(slm_size, cgh);
        cgh.parallel_for<name>(nd_range, KernelFunctor(kern_params, lm));
    };
    return Kernel;
};

template <typename name, class KernelFunctor>
auto make_cgh_nd_function(const sycl::nd_range<1> &nd_range, KernelFunctor kern)
{
    auto Kernel = [&](sycl::handler &cgh) {
        cgh.parallel_for<name>(nd_range, kern);
    };
    return Kernel;
};

template <typename T> struct LocalSortArgs
{
    T *arr;
    size_t global_array_size;
    size_t wg_chunk_size;
    LocalSortArgs(T *arr, size_t arr_len, size_t wg_len)
        : arr(arr), global_array_size(arr_len), wg_chunk_size(wg_len)
    {
    }
    ~LocalSortArgs() {}

    T *get_array_pointer() const
    {
        return arr;
    }
    size_t get_array_size() const
    {
        return global_array_size;
    }
    size_t get_chunk_size() const
    {
        return wg_chunk_size;
    }
};

template <typename T, typename localAccessorT> struct LocalSortFunc
{
    /*

     */
    T *arr;
    size_t global_array_size;
    size_t wg_chunk_size;
    localAccessorT lm;
    LocalSortFunc(T *arr, size_t arr_len, size_t wg_len, localAccessorT lm)
        : arr(arr), global_array_size(arr_len), wg_chunk_size(wg_len), lm(lm)
    {
    }
    template <class paramsT>
    LocalSortFunc(paramsT params, localAccessorT lm)
        : arr(params.get_array_pointer()),
          global_array_size(params.get_array_size()),
          wg_chunk_size(params.get_chunk_size()), lm(lm)
    {
    }
    ~LocalSortFunc() {}
    void operator()(sycl::nd_item<1> item) const
    {
        /* Use odd-even merge sort to sort lws chunk of array */
        size_t group_id = item.get_group_linear_id();
        size_t chunk_size =
            sycl::min((group_id + 1) * wg_chunk_size, global_array_size) -
            group_id * wg_chunk_size;

        // compute the greatest power of 2 less than chunk_size
        size_t sp2 = 1;
        while (sp2 < chunk_size) {
            sp2 <<= 1;
        }
        sp2 >>= 1;

        size_t gid = item.get_global_linear_id();
        size_t lid = item.get_local_linear_id();

        if (gid < global_array_size) {
            lm[lid] = arr[gid];
        }
        item.barrier(sycl::access::fence_space::local_space);

        for (size_t p = sp2; p > 0; p >>= 1) {
            size_t q = sp2;
            size_t r = 0;
            for (size_t d = p; d > 0; d = q - p, q >>= 1, r = p) {
                if ((lid < chunk_size - d) && (lid & p) == r) {
                    size_t i = lid;
                    size_t j = i + d;
                    T v1 = lm[i];
                    T v2 = lm[j];
                    if (v1 > v2) {
                        lm[i] = v2;
                        lm[j] = v1;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
        }
        if (gid < global_array_size) {
            arr[gid] = lm[lid];
        }
    };
};

template <typename T> class local_sort_kern;

template <typename T>
sycl::kernel get_local_sort_kernel(sycl::queue &q,
                                   size_t gws,
                                   size_t lws,
                                   T *arr,
                                   size_t arr_len)
{
    sycl::program program(q.get_context());

    using local_accessor_t =
        sycl::accessor<T, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>;

    [[maybe_unused]] auto cgh_fn = make_cgh_nd_function_with_local_memory<
        local_sort_kern<T>, local_accessor_t, LocalSortArgs<T>,
        LocalSortFunc<T, local_accessor_t>>(
        sycl::nd_range<1>(gws, lws), lws, LocalSortArgs<T>(arr, arr_len, lws));

    program.build_with_kernel_type<local_sort_kern<T>>();
    return program.get_kernel<local_sort_kern<T>>();
};

template <typename T> struct LocalCountExceedanceFunc
{
    T *arr;
    size_t arr_len;
    T threshold_val;
    int *count_arr;
    LocalCountExceedanceFunc(T *arr,
                             size_t arr_len,
                             T threshold_val,
                             int *count_arr)
        : arr(arr), arr_len(arr_len), threshold_val(threshold_val),
          count_arr(count_arr)
    {
    }

    void operator()(sycl::nd_item<1> item) const
    {
        /* count number of array elements in group chunk that
           exceeds the threshold value */
        size_t gid = item.get_global_linear_id();
        int partial_sum = sycl::ONEAPI::reduce(
            item.get_group(),
            (gid < arr_len) ? int(arr[gid] > threshold_val) : int(0),
            std::plus<int>());
        count_arr[item.get_group_linear_id()] = partial_sum;
    }
};

template <typename T> class local_exceedance_kern;

template <typename T>
sycl::kernel get_local_count_exceedance_kernel(sycl::queue &q,
                                               size_t gws,
                                               size_t lws,
                                               T *arr,
                                               size_t arr_len,
                                               T threshold_val,
                                               int *counts)
{
    sycl::program program(q.get_context());

    [[maybe_unused]] auto cgh_fn =
        make_cgh_nd_function<local_exceedance_kern<T>,
                             LocalCountExceedanceFunc<T>>(
            sycl::nd_range<1>(gws, lws),
            LocalCountExceedanceFunc<T>(arr, arr_len, threshold_val, counts));

    program.build_with_kernel_type<local_exceedance_kern<T>>();
    return program.get_kernel<local_exceedance_kern<T>>();
};

} // namespace dpcpp_kernels
