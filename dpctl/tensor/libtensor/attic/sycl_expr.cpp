#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>
#include <iostream>

enum typenum_t : int
{
    BOOL = 0,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    HALF,
    SINGLE,
    DOUBLE,
    CSINGLE,
    CDOUBLE
};
constexpr int num_types = 14; // number of elements in typenum_t

struct typed_vector
{
private:
    char *data;
    size_t n;
    typenum_t type;

public:
    typed_vector() : data(nullptr), n(0), type(typenum_t::INT32) {}
    explicit typed_vector(char *data, size_t n, typenum_t type)
        : data(data), n(n), type(type)
    {
    }

    typenum_t get_type() const
    {
        return type;
    }
    char *get_data() const
    {
        return data;
    }
    size_t get_size() const
    {
        return n;
    }
};

template <class T> struct is_complex : std::false_type
{
};
template <class T> struct is_complex<std::complex<T>> : std::true_type
{
};

template <typename dstTy, typename srcTy>
inline dstTy convert_impl(const srcTy &v)
{
    if constexpr (std::is_same_v<dstTy, bool> && is_complex<srcTy>::value) {
        // bool(complex_v) == (complex_v.real() != 0) && (complex_v.imag() !=0)
        return (convert_impl<bool, typename srcTy::value_type>(v.real()) ||
                convert_impl<bool, typename srcTy::value_type>(v.imag()));
    }
    else if constexpr (is_complex<srcTy>::value && !is_complex<dstTy>::value) {
        // real_t(complex_v) == real_t(complex_v.real())
        return convert_impl<dstTy, typename srcTy::value_type>(v.real());
    }
    else {
        return static_cast<dstTy>(v);
    }
}

template <typename srcT, typename dstT> class Caster
{
public:
    Caster() = default;
    void operator()(char *src,
                    std::ptrdiff_t src_offset,
                    char *dst,
                    std::ptrdiff_t dst_offset)
    {
        srcT *src_ = reinterpret_cast<srcT *>(src) + src_offset;
        dstT *dst_ = reinterpret_cast<dstT *>(dst) + dst_offset;
        *dst_ = convert_impl<dstT, srcT>(*src_);
    }
};

template <typename CastFnT> class CopyFunctor
{
    char *src = nullptr;
    char *dst = nullptr;

public:
    CopyFunctor(char *src, char *dst) : src(src), dst(dst) {}

    void operator()(sycl::id<1> wiid) const
    {
        std::ptrdiff_t i = wiid.get(0);
        CastFnT fn{};
        fn(src, i, dst, i);
    }
};

typedef sycl::event (*copy_and_cast_fn_ptr_t)(sycl::queue,
                                              size_t,
                                              char *,
                                              char *,
                                              const std::vector<sycl::event> &);

template <typename srcTy, typename dstTy>
sycl::event copy_and_cast(sycl::queue q,
                          size_t n,
                          char *p_src,
                          char *p_dst,
                          const std::vector<sycl::event> &depends)
{
    sycl::event cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        CopyFunctor<Caster<srcTy, dstTy>> copy_krn(p_src, p_dst);
        cgh.parallel_for(sycl::range<1>{n}, copy_krn);
    });

    return cast_ev;
}

template <typename dstTy>
static std::initializer_list<copy_and_cast_fn_ptr_t>
    template_copy_and_cast_funcs_row = {
        copy_and_cast<bool, dstTy>,
        copy_and_cast<int8_t, dstTy>,
        copy_and_cast<uint8_t, dstTy>,
        copy_and_cast<int16_t, dstTy>,
        copy_and_cast<uint16_t, dstTy>,
        copy_and_cast<int32_t, dstTy>,
        copy_and_cast<uint32_t, dstTy>,
        copy_and_cast<int64_t, dstTy>,
        copy_and_cast<uint64_t, dstTy>,
        copy_and_cast<sycl::half, dstTy>,
        copy_and_cast<float, dstTy>,
        copy_and_cast<double, dstTy>,
        copy_and_cast<std::complex<float>, dstTy>,
        copy_and_cast<std::complex<double>, dstTy>,
};

static copy_and_cast_fn_ptr_t copy_and_cast_dispatch_table[num_types]
                                                          [num_types];

void init_dispatch_table(void)
{
    {
        const auto the_whole_copy_and_cast_thing = {
            template_copy_and_cast_funcs_row<bool>,
            template_copy_and_cast_funcs_row<int8_t>,
            template_copy_and_cast_funcs_row<uint8_t>,
            template_copy_and_cast_funcs_row<int16_t>,
            template_copy_and_cast_funcs_row<uint16_t>,
            template_copy_and_cast_funcs_row<int32_t>,
            template_copy_and_cast_funcs_row<uint32_t>,
            template_copy_and_cast_funcs_row<int64_t>,
            template_copy_and_cast_funcs_row<uint64_t>,
            template_copy_and_cast_funcs_row<sycl::half>,
            template_copy_and_cast_funcs_row<float>,
            template_copy_and_cast_funcs_row<double>,
            template_copy_and_cast_funcs_row<std::complex<float>>,
            template_copy_and_cast_funcs_row<std::complex<double>>,
        };

        int count1 = 0;
        for (auto &row : the_whole_copy_and_cast_thing) {
            int count2 = 0;
            for (auto &fn_ptr : row) {
                copy_and_cast_dispatch_table[count1][count2] = fn_ptr;
                ++count2;
            }
            ++count1;
        }
    }
}

int main(void)
{
    init_dispatch_table();
    sycl::queue q;

    size_t n = 17;
    int32_t *i32_arr = sycl::malloc_device<int32_t>(n, q);
    int64_t *i64_arr = sycl::malloc_device<int64_t>(n, q);

    sycl::event pop32_ev = q.fill<int32_t>(i32_arr, 12374, n);
    sycl::event pop64_ev = q.fill<int64_t>(i64_arr, 938472423, n);

    typed_vector v32(reinterpret_cast<char *>(i32_arr), n, typenum_t::INT32);
    typed_vector v64(reinterpret_cast<char *>(i64_arr), n, typenum_t::INT64);

    typed_vector src = v32;
    typed_vector dst = v64;

    sycl::event cast_ev =
        copy_and_cast_dispatch_table[dst.get_type()][src.get_type()](
            q, src.get_size(), src.get_data(), dst.get_data(),
            {pop32_ev, pop64_ev});

    int64_t *res = new int64_t[n];

    sycl::event backcopy_ev = q.copy<int64_t>(
        reinterpret_cast<int64_t *>(v64.get_data()), res, n, {cast_ev});
    backcopy_ev.wait();

    for (size_t i = 0; i < n; ++i) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
