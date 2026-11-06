#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for NumPy data types as kernel arguments, local accessor and work
group memory types."""

import subprocess
import sys

import numpy as np
import pytest

import dpctl
import dpctl.compiler as dpc
import dpctl.memory as dpm

# layout of std::complex<sycl::half>
complex_half = np.dtype([("real", np.float16), ("imag", np.float16)])


def _opencl_queue_or_skip():
    try:
        return dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")


def _sycl_source_queue_or_skip():
    if not dpc.is_sycl_source_compilation_available():
        pytest.skip("SYCL source compilation extension not available")
    for filter_str in ("level_zero", "opencl:gpu", "opencl:cpu"):
        try:
            q = dpctl.SyclQueue(filter_str)
        except dpctl.SyclQueueCreationError:
            continue
        if q.sycl_device.can_compile("sycl"):
            return q
    pytest.skip("No device supporting SYCL source compilation is available")


def _skip_if_compilation_crashes(q, src):
    # some DPC++ runtimes crash compiling SYCL source using bfloat16
    code = (
        "import sys, dpctl, dpctl.compiler as dpc; "
        "q = dpctl.SyclQueue(sys.argv[1]); "
        "dpc.create_kernel_bundle_from_sycl_source(q, sys.argv[2])"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, q.sycl_device.filter_string, src],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip(f"Compilation of source failed for {q.sycl_device}")


def _skip_if_no_aspect(q, ctype):
    if "half" in ctype and not q.sycl_device.has_aspect_fp16:
        pytest.skip("Device does not support half precision floating point")
    if "double" in ctype and not q.sycl_device.has_aspect_fp64:
        pytest.skip("Device does not support double precision floating point")


def _pragmas(ctype):
    src = ""
    if "half" in ctype:
        src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    if "double" in ctype:
        src += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    return src


def _to_device(q, host):
    usm = dpm.MemoryUSMDevice(host.nbytes, queue=q)
    q.memcpy(dest=usm, src=host, count=host.nbytes)
    return usm


@pytest.mark.parametrize(
    "ctype,dtype",
    [
        ("char", "i1"),
        ("uchar", "u1"),
        ("short", "i2"),
        ("ushort", "u2"),
        ("int", "i4"),
        ("uint", "u4"),
        ("long", "i8"),
        ("ulong", "u8"),
        ("half", "f2"),
        ("float", "f4"),
        ("double", "f8"),
    ],
)
def test_numpy_scalar_arg(ctype, dtype):
    q = _opencl_queue_or_skip()
    _skip_if_no_aspect(q, ctype)
    src = _pragmas(ctype) + (
        f"kernel void axpy(global {ctype} *a, global {ctype} *b, {ctype} d) {{"
        "   size_t i = get_global_id(0);"
        "   b[i] = d * a[i] + b[i];"
        "}"
    )
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("axpy")

    n = 64
    a = (np.arange(n) % 5).astype(dtype)
    b = np.ones(n, dtype=dtype)
    a_usm, b_usm = _to_device(q, a), _to_device(q, b)

    d = np.dtype(dtype).type(3)
    q.submit(krn, [a_usm, b_usm, d], [n]).wait()
    q.memcpy(dest=b, src=b_usm, count=b.nbytes)
    assert np.array_equal(b, (a * d + np.ones(n, dtype=dtype)).astype(dtype))


def _complex_half_scalar(kind, value):
    if kind == "complex32":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        return ml_dtypes.complex32(value)
    return np.array((value.real, value.imag), dtype=complex_half)[()]


@pytest.mark.parametrize(
    "ctype,kind",
    [
        ("half2", "complex32"),
        ("half2", "structured"),
        ("float2", "c8"),
        ("double2", "c16"),
    ],
)
def test_complex_scalar_arg(ctype, kind):
    q = _opencl_queue_or_skip()
    _skip_if_no_aspect(q, ctype)
    src = _pragmas(ctype) + (
        f"kernel void cmul(global {ctype} *a, global {ctype} *b, {ctype} d) {{"
        "   size_t i = get_global_id(0);"
        f"  {ctype} ai = a[i];"
        f"  b[i] = ({ctype})(ai.x * d.x - ai.y * d.y, ai.x * d.y + ai.y * d.x);"
        "}"
    )
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("cmul")

    n = 64
    a = (np.arange(n) % 4) - 1j * (np.arange(n) % 3)
    d = 2 - 1j
    expected = a * d

    if ctype == "half2":
        # host arrays are kept as float16 pairs
        a_host = a.astype("c8").view("f4").astype("f2")
        d_arg = _complex_half_scalar(kind, d)
    else:
        a_host = a.astype(kind)
        d_arg = np.dtype(kind).type(d)
    b_host = np.zeros_like(a_host)
    a_usm, b_usm = _to_device(q, a_host), _to_device(q, b_host)

    if ctype == "half2":
        assert d_arg.dtype.itemsize == 2 * a_host.itemsize
    q.submit(krn, [a_usm, b_usm, d_arg], [n]).wait()
    q.memcpy(dest=b_host, src=b_usm, count=b_host.nbytes)

    if ctype == "half2":
        res = b_host.astype("f4").view("c8")
    else:
        res = b_host
    assert np.array_equal(res, expected.astype(res.dtype))


def test_bcomplex32_scalar_arg():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    q = _opencl_queue_or_skip()
    src = (
        "kernel void widen(global float *out, ushort2 d) {"
        "   out[0] = as_float(((uint)d.x) << 16);"
        "   out[1] = as_float(((uint)d.y) << 16);"
        "}"
    )
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("widen")
    out = np.zeros(2, dtype="f4")
    out_usm = _to_device(q, out)
    q.submit(krn, [out_usm, ml_dtypes.bcomplex32(1.5 - 2.25j)], [1]).wait()
    q.memcpy(dest=out, src=out_usm, count=out.nbytes)
    assert out.tolist() == [1.5, -2.25]


@pytest.mark.parametrize(
    "ctype,value",
    [("uchar", True), ("long", -(2**40)), ("double", 0.1), ("double2", 1 - 2j)],
)
def test_python_scalar_arg(ctype, value):
    q = _opencl_queue_or_skip()
    _skip_if_no_aspect(q, ctype)
    src = _pragmas(ctype) + (
        f"kernel void put(global {ctype} *out, {ctype} d) {{ out[0] = d; }}"
    )
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("put")
    expected = {
        bool: np.bool_,
        int: np.int64,
        float: np.float64,
        complex: np.complex128,
    }[type(value)](value)
    out = np.zeros(1, dtype=expected.dtype)
    out_usm = _to_device(q, out)
    q.submit(krn, [out_usm, value], [1]).wait()
    q.memcpy(dest=out, src=out_usm, count=out.nbytes)
    assert out[0] == expected


def test_python_int_arg_out_of_range():
    q = _opencl_queue_or_skip()
    src = "kernel void put(global long *out, long d) { out[0] = d; }"
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("put")
    out_usm = dpm.MemoryUSMDevice(8, queue=q)
    with pytest.raises(OverflowError):
        q.submit(krn, [out_usm, 2**63], [1])


def test_bfloat16_scalar_arg():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    q = _opencl_queue_or_skip()
    # a bfloat16 is the upper half of a float with the same value
    src = (
        "kernel void scale(global float *a, global float *b, ushort d) {"
        "   size_t i = get_global_id(0);"
        "   b[i] = a[i] * as_float(((uint)d) << 16);"
        "}"
    )
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("scale")

    n = 64
    a = np.arange(n, dtype="f4")
    b = np.zeros_like(a)
    a_usm, b_usm = _to_device(q, a), _to_device(q, b)

    d = ml_dtypes.bfloat16(1.5)
    q.submit(krn, [a_usm, b_usm, d], [n]).wait()
    q.memcpy(dest=b, src=b_usm, count=b.nbytes)
    assert np.array_equal(b, a * np.float32(d))


def test_bfloat16_sycl_source_arg():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    q = _sycl_source_queue_or_skip()
    src = """
    #include <sycl/sycl.hpp>
    #include <sycl/ext/oneapi/bfloat16.hpp>

    namespace syclext = sycl::ext::oneapi::experimental;
    using bfloat16_t = sycl::ext::oneapi::bfloat16;

    extern "C" SYCL_EXTERNAL
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclext::nd_range_kernel<1>))
    void scale_bfloat16(bfloat16_t *a, bfloat16_t *b, bfloat16_t d) {
        sycl::nd_item<1> it =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>();
        size_t i = it.get_global_linear_id();
        float p = static_cast<float>(a[i]) * static_cast<float>(d);
        // truncated on the bits, since float to bfloat16 conversion gives
        // wrong results on some Intel GPUs; the products are exact
        b[i] = sycl::bit_cast<bfloat16_t>(
            std::uint16_t(sycl::bit_cast<std::uint32_t>(p) >> 16));
    }
    """
    _skip_if_compilation_crashes(q, src)
    try:
        kb = dpc.create_kernel_bundle_from_sycl_source(q, src)
    except Exception:
        pytest.skip(f"Compilation of source failed for {q.sycl_device}")
    krn = kb.get_sycl_kernel("scale_bfloat16")

    lws = 32
    n = lws * 4
    a = (0.5 * (np.arange(n) % 7)).astype(ml_dtypes.bfloat16)
    # bfloat16 arrays do not support the buffer protocol
    a_bits = a.view(np.uint16)
    b_bits = np.zeros_like(a_bits)
    a_usm, b_usm = _to_device(q, a_bits), _to_device(q, b_bits)

    d = ml_dtypes.bfloat16(3.0)
    q.submit(krn, [a_usm, b_usm, d], [n], [lws]).wait()
    q.memcpy(dest=b_bits, src=b_usm, count=b_bits.nbytes)
    assert np.array_equal(b_bits.view(ml_dtypes.bfloat16), a * d)


def test_structured_scalar_raw_kernel_arg():
    if not dpctl.RawKernelArg.is_available():
        pytest.skip("Raw kernel arg extension not available")
    q = _sycl_source_queue_or_skip()
    src = """
    #include <sycl/sycl.hpp>

    namespace syclext = sycl::ext::oneapi::experimental;

    struct Coeffs {
        float scale;
        float offset;
        int repeats;
    };

    extern "C" SYCL_EXTERNAL
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclext::nd_range_kernel<1>))
    void apply_coeffs(float *a, float *b, Coeffs c) {
        sycl::nd_item<1> it =
            sycl::ext::oneapi::this_work_item::get_nd_item<1>();
        size_t i = it.get_global_linear_id();
        float acc = a[i];
        for (int r = 0; r < c.repeats; ++r) {
            acc = acc * c.scale + c.offset;
        }
        b[i] = acc;
    }
    """
    try:
        kb = dpc.create_kernel_bundle_from_sycl_source(q, src)
    except Exception:
        pytest.skip(f"Compilation of source failed for {q.sycl_device}")
    krn = kb.get_sycl_kernel("apply_coeffs")

    coeffs_t = np.dtype([("scale", "f4"), ("offset", "f4"), ("repeats", "i4")])
    assert coeffs_t.itemsize == 12

    lws = 16
    n = lws * 4
    a = np.arange(n, dtype="f4")
    b = np.zeros_like(a)
    a_usm, b_usm = _to_device(q, a), _to_device(q, b)

    c = np.array((2.0, 1.0, 3), dtype=coeffs_t)[()]
    q.submit(krn, [a_usm, b_usm, c], [n], [lws]).wait()
    q.memcpy(dest=b, src=b_usm, count=b.nbytes)

    expected = a.copy()
    for _ in range(3):
        expected = expected * np.float32(2.0) + np.float32(1.0)
    assert np.array_equal(b, expected)


@pytest.mark.parametrize(
    "ctype,dtype,base",
    [
        ("half2", complex_half, "f2"),
        ("float2", np.dtype("c8"), "f4"),
        ("long2", np.dtype([("x", "i8"), ("y", "i8")]), "i8"),
    ],
)
def test_local_accessor_arg(ctype, dtype, base):
    q = _opencl_queue_or_skip()
    _skip_if_no_aspect(q, ctype)
    src = _pragmas(ctype) + (
        f"kernel void rev(global {ctype} *a, local {ctype} *tmp) {{"
        "   size_t lid = get_local_id(0);"
        "   size_t lws = get_local_size(0);"
        "   tmp[lid] = a[get_global_id(0)];"
        "   barrier(CLK_LOCAL_MEM_FENCE);"
        "   a[get_global_id(0)] = tmp[lws - 1 - lid];"
        "}"
    )
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("rev")

    lws = 16
    n = lws * 4
    # host arrays are kept in the base type of the data type
    x = np.arange(2 * n, dtype=base)
    x_usm = _to_device(q, x)

    la = dpctl.LocalAccessor(dtype, (lws,))
    q.submit(krn, [x_usm, la], [n], [lws]).wait()
    res = np.empty_like(x)
    q.memcpy(dest=res, src=x_usm, count=res.nbytes)

    expected = x.reshape(n // lws, lws, 2)[:, ::-1, :].reshape(-1)
    assert np.array_equal(res, expected)


@pytest.mark.parametrize(
    "dtype",
    ["i1", "u2", "f2", "f4", "c8", "c16", "float", np.float16, complex_half],
)
def test_local_accessor_dtypes(dtype):
    assert isinstance(dpctl.LocalAccessor(dtype, (8,)), dpctl.LocalAccessor)


def test_local_accessor_bfloat16():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    for dtype in (ml_dtypes.bfloat16, "bfloat16", ml_dtypes.complex32):
        la = dpctl.LocalAccessor(dtype, (8,))
        assert isinstance(la, dpctl.LocalAccessor)


@pytest.mark.parametrize(
    "dtype",
    [
        "not_a_type",
        "U4",
        "S4",
        "M8",
        "f16",
        object,
        [("a", "f4"), ("b", "f4"), ("c", "f4")],
        None,
        np.dtype("f4").newbyteorder(),
    ],
)
def test_local_accessor_invalid_dtypes(dtype):
    with pytest.raises(ValueError):
        dpctl.LocalAccessor(dtype, (8,))


def test_work_group_memory_dtypes():
    if not dpctl.WorkGroupMemory.is_available():
        pytest.skip("Workgroup memory extension not available")
    for dtype in ("f4", np.float16, complex_half, "c16"):
        assert dpctl.WorkGroupMemory(dtype, 8)._ref != 0
    # any size is allowed
    assert dpctl.WorkGroupMemory([("a", "f4"), ("b", "i8")], 8)._ref != 0
    for dtype in ("not_a_type", "U4", object, None):
        with pytest.raises(TypeError):
            dpctl.WorkGroupMemory(dtype, 8)


@pytest.mark.parametrize(
    "arg", [np.str_("ab"), np.datetime64("2026-01-01"), "ab", None]
)
def test_unsupported_scalar_args(arg):
    q = _opencl_queue_or_skip()
    src = "kernel void k(global int *a, int d) { a[get_global_id(0)] = d; }"
    krn = dpc.create_kernel_bundle_from_source(q, src).get_sycl_kernel("k")
    a_usm = dpm.MemoryUSMDevice(4 * 8, queue=q)
    with pytest.raises(TypeError):
        q.submit(krn, [a_usm, arg], [8])
