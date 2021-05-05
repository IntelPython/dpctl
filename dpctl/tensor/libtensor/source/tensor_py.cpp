#include "dpctl4pybind11.hpp"
#include "utils/strided_iters.hpp"
#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{
enum typenum_t : int
{
    BOOL = 0,
    INT8, // 1
    UINT8,
    INT16,
    UINT16,
    INT32, // 5
    UINT32,
    INT64,
    UINT64,
    HALF,
    FLOAT, // 10
    DOUBLE,
    CFLOAT,
    CDOUBLE, // 13
};
constexpr int num_types = 14; // number of elements in typenum_t

template <typename T> struct same_size
{
    template <typename U> using as = std::bool_constant<sizeof(T) == sizeof(U)>;
};

template <typename Concrete> constexpr int platform_lookup()
{
    return -1;
}

// Lookup a type according to its size, and return a value corresponding to the
// NumPy typenum.
template <typename Concrete, typename T, typename... Ts, typename... Ints>
constexpr int platform_lookup(int I, Ints... Is)
{
    return sizeof(Concrete) == sizeof(T)
               ? I
               : platform_lookup<Concrete, Ts...>(Is...);
}

// Platform-dependent normalization
int UAR_INT8 = -1;
int UAR_UINT8 = -1;
int UAR_INT16 = -1;
int UAR_UINT16 = -1;
int UAR_INT32 = -1;
int UAR_UINT32 = -1;
int UAR_INT64 = -1;
int UAR_UINT64 = -1;

typenum_t typenum_to_lookup_id(int typenum)
{
    if (typenum == UAR_DOUBLE) {
        return typenum_t::DOUBLE;
    }
    else if (typenum == UAR_INT64) {
        return typenum_t::INT64;
    }
    else if (typenum == UAR_INT32) {
        return typenum_t::INT32;
    }
    else if (typenum == UAR_BOOL) {
        return typenum_t::BOOL;
    }
    else if (typenum == UAR_CDOUBLE) {
        return typenum_t::CDOUBLE;
    }
    else if (typenum == UAR_FLOAT) {
        return typenum_t::FLOAT;
    }
    else if (typenum == UAR_INT16) {
        return typenum_t::INT16;
    }
    else if (typenum == UAR_INT8) {
        return typenum_t::INT8;
    }
    else if (typenum == UAR_UINT64) {
        return typenum_t::UINT64;
    }
    else if (typenum == UAR_UINT32) {
        return typenum_t::UINT32;
    }
    else if (typenum == UAR_UINT16) {
        return typenum_t::UINT16;
    }
    else if (typenum == UAR_UINT8) {
        return typenum_t::UINT8;
    }
    else if (typenum == UAR_CFLOAT) {
        return typenum_t::CFLOAT;
    }
    else if (typenum == UAR_HALF) {
        return typenum_t::HALF;
    }
    else {
        throw std::runtime_error("Unrecogized typenum " +
                                 std::to_string(typenum) + " encountered.");
    }
}

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
                    std::ptrdiff_t dst_offset) const
    {
        srcT *src_ = reinterpret_cast<srcT *>(src) + src_offset;
        dstT *dst_ = reinterpret_cast<dstT *>(dst) + dst_offset;
        *dst_ = convert_impl<dstT, srcT>(*src_);
    }
};

template <typename CastFnT> class CopyFunctor
{
private:
    char *src_ = nullptr;
    char *dst_ = nullptr;
    py::ssize_t *shape_strides_ = nullptr;
    int nd_ = 0;

public:
    CopyFunctor(char *src_cp, char *dst_cp, py::ssize_t *shape_strides, int nd)
        : src_(src_cp), dst_(dst_cp), shape_strides_(shape_strides), nd_(nd)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        std::ptrdiff_t src_offset = 0;
        std::ptrdiff_t dst_offset = 0;
        CIndexer_vector indxr(nd_);
        indxr.get_displacement(
            wiid.get(0),
            static_cast<const py::ssize_t *>(shape_strides_), // common shape
            static_cast<const py::ssize_t *>(shape_strides_ +
                                             nd_), // src strides
            static_cast<const py::ssize_t *>(shape_strides_ +
                                             2 * nd_), // dst strides
            src_offset,                                // modified by reference
            dst_offset                                 // modified by reference
        );
        CastFnT fn{};
        fn(src_, src_offset, dst_, dst_offset);
    }
};

template <typename srcT, typename dstT> class copy_cast_generic_kernel;

typedef sycl::event (*copy_and_cast_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    py::ssize_t *,
    char *,
    char *,
    const std::vector<sycl::event> &,
    const std::initializer_list<sycl::event> &);

template <typename srcTy, typename dstTy>
sycl::event copy_and_cast_generic_impl(
    sycl::queue q,
    size_t nelems,
    int nd,
    py::ssize_t *shape_and_strides,
    char *src_p,
    char *dst_p,
    const std::vector<sycl::event> &depends,
    const std::initializer_list<sycl::event> &additional_depends)
{
    sycl::event copy_and_cast_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);
        cgh.parallel_for<copy_cast_generic_kernel<srcTy, dstTy>>(
            sycl::range<1>(nelems), CopyFunctor<Caster<srcTy, dstTy>>(
                                        src_p, dst_p, shape_and_strides, nd));
    });

    return copy_and_cast_ev;
}

template <typename dstTy>
static std::initializer_list<copy_and_cast_fn_ptr_t>
    template_copy_and_cast_funcs_row_for_dstT = {
        copy_and_cast_generic_impl<dstTy, bool>,
        copy_and_cast_generic_impl<dstTy, int8_t>,
        copy_and_cast_generic_impl<dstTy, uint8_t>,
        copy_and_cast_generic_impl<dstTy, int16_t>,
        copy_and_cast_generic_impl<dstTy, uint16_t>,
        copy_and_cast_generic_impl<dstTy, int32_t>,
        copy_and_cast_generic_impl<dstTy, uint32_t>,
        copy_and_cast_generic_impl<dstTy, int64_t>,
        copy_and_cast_generic_impl<dstTy, uint64_t>,
        copy_and_cast_generic_impl<dstTy, sycl::half>,
        copy_and_cast_generic_impl<dstTy, float>,
        copy_and_cast_generic_impl<dstTy, double>,
        copy_and_cast_generic_impl<dstTy, std::complex<float>>,
        copy_and_cast_generic_impl<dstTy, std::complex<double>>,
};

static copy_and_cast_fn_ptr_t copy_and_cast_generic_dispatch_table[num_types]
                                                                  [num_types];

void init_copy_and_cast_generic_dispatch_table(void)
{
    const auto copy_and_cast_map_by_dstT = {
        template_copy_and_cast_funcs_row_for_dstT<bool>,
        template_copy_and_cast_funcs_row_for_dstT<int8_t>,
        template_copy_and_cast_funcs_row_for_dstT<uint8_t>,
        template_copy_and_cast_funcs_row_for_dstT<int16_t>,
        template_copy_and_cast_funcs_row_for_dstT<uint16_t>,
        template_copy_and_cast_funcs_row_for_dstT<int32_t>,
        template_copy_and_cast_funcs_row_for_dstT<uint32_t>,
        template_copy_and_cast_funcs_row_for_dstT<int64_t>,
        template_copy_and_cast_funcs_row_for_dstT<uint64_t>,
        template_copy_and_cast_funcs_row_for_dstT<sycl::half>,
        template_copy_and_cast_funcs_row_for_dstT<float>,
        template_copy_and_cast_funcs_row_for_dstT<double>,
        template_copy_and_cast_funcs_row_for_dstT<std::complex<float>>,
        template_copy_and_cast_funcs_row_for_dstT<std::complex<double>>,
    };

    int dst_id = 0;
    for (auto &row : copy_and_cast_map_by_dstT) {
        int src_id = 0;
        for (auto &fn_ptr : row) {
            copy_and_cast_generic_dispatch_table[dst_id][src_id] = fn_ptr;
            ++src_id;
        }
        ++dst_id;
    }
    return;
}

} // namespace

namespace
{

using vecT = std::vector<py::ssize_t>;
std::tuple<vecT, vecT, py::size_t> contract_iter(vecT shape, vecT strides)
{
    const size_t dim = shape.size();
    if (dim != strides.size()) {
        throw py::value_error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides = strides;
    py::ssize_t disp(0);

    int nd = simplify_iteration_stride(dim, out_shape.data(),
                                       out_strides.data(), disp);
    out_shape.resize(nd);
    out_strides.resize(nd);
    return std::make_tuple(out_shape, out_strides, disp);
}

bool usm_ndarray_check_(py::object o)
{
    PyObject *raw_o = o.ptr();
    return ((raw_o != nullptr) &&
            static_cast<bool>(PyObject_TypeCheck(raw_o, &PyUSMArrayType)));
}

int usm_ndarray_ndim_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetNDim(raw_ar);
}

const py::ssize_t *usm_ndarray_shape_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetShape(raw_ar);
}

const py::ssize_t *usm_ndarray_strides_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetStrides(raw_ar);
}

std::pair<py::ssize_t, py::ssize_t> usm_ndarray_minmax_offsets_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    int nd = UsmNDArray_GetNDim(raw_ar);
    const py::ssize_t *shape = UsmNDArray_GetShape(raw_ar);
    const py::ssize_t *strides = UsmNDArray_GetStrides(raw_ar);

    py::ssize_t offset_min = 0;
    py::ssize_t offset_max = 0;
    if (strides == nullptr) {
        py::ssize_t stride(1);
        for (int i = 0; i < nd; ++i) {
            offset_max += stride * (shape[i] - 1);
            stride *= shape[i];
        }
    }
    else {
        offset_min = UsmNDArray_GetOffset(raw_ar);
        offset_max = offset_min;
        for (int i = 0; i < nd; ++i) {
            py::ssize_t delta = strides[i] * (shape[i] - 1);
            if (strides[i] > 0) {
                offset_max += delta;
            }
            else {
                offset_min += delta;
            }
        }
    }
    return std::make_pair(offset_min, offset_max);
}

sycl::queue usm_ndarray_get_queue_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    DPCTLSyclQueueRef QRef = UsmNDArray_GetQueueRef(raw_ar);
    return *(reinterpret_cast<sycl::queue *>(QRef));
}

int usm_ndarray_get_typenum_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetTypenum(raw_ar);
}

char *usm_ndarray_get_data_(py::object ar)
{
    PyObject *raw_o = ar.ptr();
    PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

    return UsmNDArray_GetData(raw_ar);
}

sycl::event
copy_usm_ndarray_into_usm_ndarray(py::object src,
                                  py::object dst,
                                  sycl::queue exec_q,
                                  const std::vector<sycl::event> &depends = {})
{
    if (!usm_ndarray_check_(src) || !usm_ndarray_check_(dst)) {
        throw py::type_error("Arguments of type usm_ndarray expected");
    }

    // array dimensions must be the same
    int src_nd = usm_ndarray_ndim_(src);
    int dst_nd = usm_ndarray_ndim_(dst);
    if (src_nd != dst_nd) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // shapes must be the same
    const py::ssize_t *src_shape = usm_ndarray_shape_(src);
    const py::ssize_t *dst_shape = usm_ndarray_shape_(dst);
    bool shapes_equal(true);
    size_t src_nelems(1);
    for (int i = 0; i < src_nd; ++i) {
        src_nelems *= static_cast<size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    if (src_nelems == 0) {
        // nothing to do
        return sycl::event();
    }

    // destination must be ample enough to accomodate all elements
    auto dst_offsets = usm_ndarray_minmax_offsets_(dst);
    if (static_cast<size_t>(dst_offsets.second - dst_offsets.first) + 1 <
        src_nelems)
    {
        throw py::value_error("Destination array can not accomodate all the "
                              "elements of source array.");
    }

    // check same contexts
    sycl::queue src_q = usm_ndarray_get_queue_(src);
    sycl::queue dst_q = usm_ndarray_get_queue_(dst);

    sycl::context exec_ctx = exec_q.get_context();
    if (src_q.get_context() != exec_ctx || dst_q.get_context() != exec_ctx) {
        throw py::value_error(
            "Execution queue context is not the same as allocation contexts");
    }

    int src_typenum = usm_ndarray_get_typenum_(src);
    int dst_typenum = usm_ndarray_get_typenum_(dst);

    // TODO: check can cast

    const py::ssize_t *src_strides = usm_ndarray_strides_(src);
    const py::ssize_t *dst_strides = usm_ndarray_strides_(dst);
    // TODO: check for applicability of special cases:
    //           (same type && (both C-contiguous || both F-contiguous)

    // TODO: use contract_iter2

    // TODO: optimization: use specialized kernels for dim <=3

    // Generic implementation:

    //   If shape/strides are accessed with accessors, buffer destructor
    //   will force syncronization.
    py::ssize_t *shape_strides =
        sycl::malloc_device<py::ssize_t>(3 * src_nd, exec_q);

    sycl::event copy_shape_ev =
        exec_q.copy<py::ssize_t>(src_shape, shape_strides, src_nd);

    sycl::event copy_src_strides_ev =
        exec_q.copy<py::ssize_t>(src_strides, shape_strides + src_nd, src_nd);

    sycl::event copy_dst_strides_ev = exec_q.copy<py::ssize_t>(
        dst_strides, shape_strides + 2 * src_nd, src_nd);

    copy_and_cast_fn_ptr_t copy_and_cast_fn = nullptr;
    {
        int src_id = typenum_to_lookup_id(src_typenum);
        int dst_id = typenum_to_lookup_id(dst_typenum);
        copy_and_cast_fn = copy_and_cast_generic_dispatch_table[dst_id][src_id];
    }

    char *src_data = usm_ndarray_get_data_(src);
    char *dst_data = usm_ndarray_get_data_(dst);
    sycl::event copy_ev = copy_and_cast_fn(
        exec_q, src_nelems, src_nd, shape_strides, src_data, dst_data, depends,
        {copy_shape_ev, copy_src_strides_ev, copy_dst_strides_ev});

    // async free of shape_strides temporary
    auto ctx = exec_q.get_context();
    exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_ev);
        cgh.host_task(
            [ctx, shape_strides]() { sycl::free(shape_strides, ctx); });
    });

    return copy_ev;
}

} // namespace

PYBIND11_MODULE(_tensor_impl, m)
{

    init_copy_and_cast_generic_dispatch_table();

    import_dpctl();

    UAR_INT8 = UAR_BYTE;
    UAR_UINT8 = UAR_UBYTE;
    UAR_INT16 = UAR_SHORT;
    UAR_UINT16 = UAR_USHORT;
    UAR_INT32 = platform_lookup<std::int32_t, long, int, short>(
        UAR_LONG, UAR_INT, UAR_SHORT);
    UAR_UINT32 =
        platform_lookup<std::uint32_t, unsigned long, unsigned int,
                        unsigned short>(UAR_ULONG, UAR_UINT, UAR_USHORT);
    UAR_INT64 = platform_lookup<std::int64_t, long, long long, int>(
        UAR_LONG, UAR_LONGLONG, UAR_INT);
    UAR_UINT64 =
        platform_lookup<std::uint64_t, unsigned long, unsigned long long,
                        unsigned int>(UAR_ULONG, UAR_ULONGLONG, UAR_UINT);

    m.def(
        "fp64_default_device",
        [](void) -> sycl::device {
            sycl::device d;
            try {
                d = sycl::device(sycl::default_selector{});
            } catch (const std::exception &e) {
                throw std::runtime_error("");
            }
            return d;
        },
        "Return default selected device that supports double precision "
        "computation");

    m.def(
        "_contract_iter", &contract_iter,
        "Simplifies iteration of array of given shape & stride. Returns "
        "a triple: shape, stride and offset for the new iterator of possible "
        "smaller dimension, which traverses the same elements as the original "
        "iterator, possibly in a different order.");

    m.def("_copy_usm_ndarray_into_usm_ndarray",
          &copy_usm_ndarray_into_usm_ndarray,
          "Copies into usm_ndarray `src` from usm_ndarray `dst`.",
          py::arg("src"), py::arg("dst"),
          py::arg("queue") = py::cast<py::none>(Py_None),
          py::arg("depends") = py::list());
}
