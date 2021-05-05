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

struct typed_value
{
private:
    char *data;
    typenum_t type;

public:
    typed_value() : data(nullptr), type(typenum_t::INT32) {}
    explicit typed_value(char *data, typenum_t type) : data(data), type(type) {}
    typenum_t get_type() const
    {
        return type;
    }
    char *get_data() const
    {
        return data;
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

template <typename dstTy, typename srcTy>
void copy_impl(char *dst_char_p, char *src_char_p)
{
    dstTy *dst_p = reinterpret_cast<dstTy *>(dst_char_p);
    srcTy *src_p = reinterpret_cast<srcTy *>(src_char_p);

    *dst_p = convert_impl<dstTy, srcTy>(*src_p);
    return;
}

typedef void (*fn_ptr_t)(char *, char *);

template <typename dstTy>
static std::initializer_list<fn_ptr_t> template_funcs_row = {
    copy_impl<dstTy, bool>,
    copy_impl<dstTy, int8_t>,
    copy_impl<dstTy, uint8_t>,
    copy_impl<dstTy, int16_t>,
    copy_impl<dstTy, uint16_t>,
    copy_impl<dstTy, int32_t>,
    copy_impl<dstTy, uint32_t>,
    copy_impl<dstTy, int64_t>,
    copy_impl<dstTy, uint64_t>,
    copy_impl<dstTy, sycl::half>,
    copy_impl<dstTy, float>,
    copy_impl<dstTy, double>,
    copy_impl<dstTy, std::complex<float>>,
    copy_impl<dstTy, std::complex<double>>,
};

static fn_ptr_t dispatch_table[14][14];

void copy_with_dispatch(typed_value dst, typed_value src)
{
    auto dst_ty = dst.get_type();
    auto src_ty = src.get_type();
    dispatch_table[dst_ty][src_ty](dst.get_data(), src.get_data());
}

int main_dynamic_type_casting(void)
{

    std::cout << "Size of bool is " << sizeof(bool) << std::endl;
    std::cout << "Size of int8_t is " << sizeof(int8_t) << std::endl;
    std::cout << "Size of uint8_t is " << sizeof(uint8_t) << std::endl;
    std::cout << "Size of int16_t is " << sizeof(int16_t) << std::endl;
    std::cout << "Size of uint16_t is " << sizeof(uint16_t) << std::endl;
    std::cout << "Size of int32_t is " << sizeof(int32_t) << std::endl;
    std::cout << "Size of uint32_t is " << sizeof(uint32_t) << std::endl;
    std::cout << "Size of int64_t is " << sizeof(int64_t) << std::endl;
    std::cout << "Size of uint64_t is " << sizeof(uint64_t) << std::endl;
    std::cout << "Size of half is " << sizeof(sycl::half) << std::endl;
    std::cout << "Size of float is " << sizeof(float) << std::endl;
    std::cout << "Size of double is " << sizeof(double) << std::endl;
    std::cout << "Size of std::complex<float> is "
              << sizeof(std::complex<float>) << std::endl;
    std::cout << "Size of std::complex<double> is "
              << sizeof(std::complex<double>) << std::endl;

    auto the_whole_thing = {
        template_funcs_row<bool>,
        template_funcs_row<int8_t>,
        template_funcs_row<uint8_t>,
        template_funcs_row<int16_t>,
        template_funcs_row<uint16_t>,
        template_funcs_row<int32_t>,
        template_funcs_row<uint32_t>,
        template_funcs_row<int64_t>,
        template_funcs_row<uint64_t>,
        template_funcs_row<sycl::half>,
        template_funcs_row<float>,
        template_funcs_row<double>,
        template_funcs_row<std::complex<float>>,
        template_funcs_row<std::complex<double>>,
    };

    int count1 = 0;
    for (auto &row : the_whole_thing) {
        int count2 = 0;
        for (auto &fn_ptr : row) {
            dispatch_table[count1][count2] = fn_ptr;
            ++count2;
        }
        ++count1;
    }

    bool b_v(false);
    int8_t i8_v(-17);
    uint8_t u8_v(17);
    int16_t i16_v(-317);
    uint16_t u16_v(317);
    int32_t i32_v(-123317);
    uint32_t u32_v(123317);
    int64_t i64_v(-1232433317);
    uint64_t u64_v(1223333317);
    sycl::half h_v(-8.2);
    float f_v(0.2);
    double d_v(3.2);
    std::complex<float> c8_v(1, 3);
    std::complex<double> c16_v(-1, 2);

    typed_value double_val(reinterpret_cast<char *>(&d_v), typenum_t::DOUBLE);
    typed_value int64_val(reinterpret_cast<char *>(&i64_v), typenum_t::INT64);
    typed_value half_val(reinterpret_cast<char *>(&h_v), typenum_t::HALF);

    copy_impl<int64_t, double>(int64_val.get_data(), double_val.get_data());

    std::cout << "Integer value : " << i64_v << std::endl;

    copy_with_dispatch(int64_val, half_val);

    std::cout << "Integer value : " << i64_v << std::endl;

    return 0;
}
