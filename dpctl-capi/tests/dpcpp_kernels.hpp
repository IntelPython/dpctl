#pragma once
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

} // namespace dpcpp_kernels
