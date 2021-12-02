#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

"""Defines unit test cases for the Memory classes in _memory.pyx.
"""

import numpy as np
import pytest
from helper import has_cpu, has_gpu, has_sycl_platforms

import dpctl
from dpctl.memory import (
    MemoryUSMDevice,
    MemoryUSMHost,
    MemoryUSMShared,
    as_usm_memory,
)


class Dummy(MemoryUSMShared):
    """
    Class that exposes ``__sycl_usm_array_interface__`` with
    SYCL context for sycl object, instead of Sycl queue.
    """

    @property
    def __sycl_usm_array_interface(self):
        iface = super().__sycl_usm_array_interface__
        iface["syclob"] = iface["syclobj"].get_sycl_context()
        return iface


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_memory_create(memory_ctor):
    import sys

    nbytes = 1024
    queue = dpctl.SyclQueue()
    mobj = memory_ctor(nbytes, alignment=64, queue=queue)
    assert mobj.nbytes == nbytes
    assert hasattr(mobj, "__sycl_usm_array_interface__")
    assert len(mobj) == nbytes
    assert mobj.size == nbytes
    assert mobj._context == queue.sycl_context
    assert mobj._queue == queue
    assert mobj.sycl_queue == queue
    assert type(repr(mobj)) is str
    assert type(bytes(mobj)) is bytes
    assert sys.getsizeof(mobj) > nbytes


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_memory_create_with_np():
    nbytes = 16384
    mobj = dpctl.memory.MemoryUSMShared(np.int64(nbytes))
    assert mobj.nbytes == nbytes
    assert hasattr(mobj, "__sycl_usm_array_interface__")


def _create_memory():
    nbytes = 1024
    queue = dpctl.SyclQueue()
    mobj = MemoryUSMShared(nbytes, alignment=64, queue=queue)
    return mobj


def _create_host_buf(nbytes):
    ba = bytearray(nbytes)
    for i in range(nbytes):
        ba[i] = (i % 32) + ord("a")
    return ba


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_memory_without_context():
    mobj = _create_memory()

    # Without context
    assert mobj.get_usm_type() == "shared"
    assert mobj.get_usm_type(syclobj=dpctl.SyclContext()) == "shared"


@pytest.mark.skipif(not has_cpu(), reason="No SYCL CPU device available.")
def test_memory_cpu_context():
    mobj = _create_memory()

    # type respective to the context in which
    # memory was created
    usm_type = mobj.get_usm_type()
    assert usm_type == "shared"

    cpu_queue = dpctl.SyclQueue("cpu")
    # type as view from CPU queue
    usm_type = mobj.get_usm_type(cpu_queue)
    # type can be unknown if current queue is
    # not in the same SYCL context
    assert usm_type in ["unknown", "shared"]


@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
def test_memory_gpu_context():
    mobj = _create_memory()

    # GPU context
    usm_type = mobj.get_usm_type()
    assert usm_type == "shared"
    gpu_queue = dpctl.SyclQueue("opencl:gpu")
    usm_type = mobj.get_usm_type(gpu_queue)
    assert usm_type in ["unknown", "shared"]


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_buffer_protocol():
    mobj = _create_memory()
    mv1 = memoryview(mobj)
    mv2 = memoryview(mobj)
    assert mv1 == mv2


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_copy_host_roundtrip():
    mobj = _create_memory()
    host_src_obj = _create_host_buf(mobj.nbytes)
    mobj.copy_from_host(host_src_obj)
    host_dest_obj = mobj.copy_to_host()
    del mobj
    assert host_src_obj == host_dest_obj


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_zero_copy():
    mobj = _create_memory()
    mobj2 = type(mobj)(mobj)

    assert mobj2.reference_obj is mobj
    mobj_data = mobj.__sycl_usm_array_interface__["data"]
    mobj2_data = mobj2.__sycl_usm_array_interface__["data"]
    assert mobj_data == mobj2_data


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_pickling(memory_ctor):
    import pickle

    mobj = memory_ctor(1024, alignment=64)
    host_src_obj = _create_host_buf(mobj.nbytes)
    mobj.copy_from_host(host_src_obj)

    mobj_reconstructed = pickle.loads(pickle.dumps(mobj))
    assert type(mobj) == type(
        mobj_reconstructed
    ), "Pickling should preserve type"
    assert (
        mobj.tobytes() == mobj_reconstructed.tobytes()
    ), "Pickling should preserve buffer content"
    assert (
        mobj._pointer != mobj_reconstructed._pointer
    ), "Pickling/unpickling should be changing pointer"


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_pickling_reconstructor_invalid_type(memory_ctor):
    import pickle

    mobj = memory_ctor(1024, alignment=64)
    good_pickle_bytes = pickle.dumps(mobj)
    usm_types = expected_usm_type(memory_ctor).encode("utf-8")
    i = good_pickle_bytes.rfind(usm_types)
    bad_pickle_bytes = good_pickle_bytes[:i] + b"u" + good_pickle_bytes[i + 1 :]
    with pytest.raises(ValueError):
        pickle.loads(bad_pickle_bytes)


@pytest.fixture(params=[MemoryUSMShared, MemoryUSMDevice, MemoryUSMHost])
def memory_ctor(request):
    return request.param


def expected_usm_type(ctor):
    mapping = {
        MemoryUSMShared: "shared",
        MemoryUSMDevice: "device",
        MemoryUSMHost: "host",
    }
    return mapping.get(ctor, "unknown")


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_create_with_size_and_alignment_and_queue(memory_ctor):
    q = dpctl.SyclQueue()
    m = memory_ctor(1024, alignment=64, queue=q)
    assert m.nbytes == 1024
    assert m.get_usm_type() == expected_usm_type(memory_ctor)


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_create_with_size_and_queue(memory_ctor):
    q = dpctl.SyclQueue()
    m = memory_ctor(1024, queue=q)
    assert m.nbytes == 1024
    assert m.get_usm_type() == expected_usm_type(memory_ctor)


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_create_with_size_and_alignment(memory_ctor):
    m = memory_ctor(1024, alignment=64)
    assert m.nbytes == 1024
    assert m.get_usm_type() == expected_usm_type(memory_ctor)


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_create_with_only_size(memory_ctor):
    m = memory_ctor(1024)
    assert m.nbytes == 1024
    assert m.get_usm_type() == expected_usm_type(memory_ctor)


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_sycl_usm_array_interface(memory_ctor):
    m = memory_ctor(256)
    m2 = Dummy(m.nbytes)
    hb = np.random.randint(0, 256, size=256, dtype="|u1")
    m2.copy_from_host(hb)
    # test that USM array interface works with SyclContext as 'syclobj'
    m.copy_from_device(m2)
    assert np.array_equal(m.copy_to_host(), hb)


class View:
    def __init__(
        self, buf, shape, strides, offset, syclobj=None, transf_fn=None
    ):
        self.buffer_ = buf
        self.shape_ = shape
        self.strides_ = strides
        self.offset_ = offset
        self.syclobj_ = syclobj
        self.transf_fn_ = transf_fn

    @property
    def __sycl_usm_array_interface__(self):
        sua_iface = self.buffer_.__sycl_usm_array_interface__
        sua_iface["offset"] = self.offset_
        sua_iface["shape"] = self.shape_
        sua_iface["strides"] = self.strides_
        if self.syclobj_:
            sua_iface["syclobj"] = self.syclobj_
        if self.transf_fn_:
            sua_iface = self.transf_fn_(sua_iface)
        return sua_iface


def test_suai_non_contig_1D(memory_ctor):
    """
    Test of zero-copy using sycl_usm_array_interface with non-contiguous
    data.
    """

    try:
        buf = memory_ctor(32)
    except Exception:
        pytest.skip("{} could not be allocated".format(memory_ctor.__name__))
    host_canary = np.full((buf.nbytes,), 77, dtype="|u1")
    buf.copy_from_host(host_canary)
    n1d = 10
    step_1d = 2
    offset = 8
    v = View(buf, shape=(n1d,), strides=(step_1d,), offset=offset)
    buf2 = memory_ctor(v)
    expected_nbytes = (
        np.flip(
            host_canary[offset : offset + n1d * step_1d : step_1d]
        ).ctypes.data
        + 1
        - host_canary[offset:].ctypes.data
    )
    assert buf2.nbytes == expected_nbytes
    inset_canary = np.arange(0, buf2.nbytes, dtype="|u1")
    buf2.copy_from_host(inset_canary)
    res = buf.copy_to_host()
    del buf
    del buf2
    expected_res = host_canary.copy()
    expected_res[offset : offset + (n1d - 1) * step_1d + 1] = inset_canary
    assert np.array_equal(res, expected_res)


def test_suai_non_contig_2D(memory_ctor):
    try:
        buf = memory_ctor(20)
    except Exception:
        pytest.skip("{} could not be allocated".format(memory_ctor.__name__))
    host_canary = np.arange(20, dtype="|u1")
    buf.copy_from_host(host_canary)
    shape_2d = (2, 2)
    strides_2d = (10, -2)
    offset = 9
    idx = []
    for i0 in range(shape_2d[0]):
        for i1 in range(shape_2d[1]):
            idx.append(offset + i0 * strides_2d[0] + i1 * strides_2d[1])
    idx.sort()
    v = View(buf, shape=shape_2d, strides=strides_2d, offset=offset)
    buf2 = memory_ctor(v)
    expected_nbytes = idx[-1] - idx[0] + 1
    assert buf2.nbytes == expected_nbytes
    inset_canary = np.full((buf2.nbytes), 255, dtype="|u1")
    buf2.copy_from_host(inset_canary)
    res = buf.copy_to_host()
    del buf
    del buf2
    expected_res = host_canary.copy()
    expected_res[idx[0] : idx[-1] + 1] = inset_canary
    assert np.array_equal(res, expected_res)


def test_suai_invalid_suai():
    n_bytes = 2 * 3 * 5 * 128
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queue")
    try:
        buf = MemoryUSMShared(n_bytes, queue=q)
    except Exception:
        pytest.skip("USM-shared allocation failed")

    # different syclobj values
    class DuckSyclObject:
        def __init__(self, syclobj):
            self.syclobj = syclobj

        def _get_capsule(self):
            return self.syclobj._get_capsule()

    ctx = q.sycl_context
    for syclobj in [
        q,
        DuckSyclObject(q),
        q._get_capsule(),
        ctx,
        DuckSyclObject(ctx),
        ctx._get_capsule(),
    ]:
        v = View(buf, shape=(n_bytes,), strides=(1,), offset=0, syclobj=syclobj)
        MemoryUSMShared(v)
        with pytest.raises(ValueError):
            MemoryUSMDevice(v)
        with pytest.raises(ValueError):
            MemoryUSMHost(v)

    # version validation
    def invalid_version(suai_iface):
        "Set version to invalid"
        suai_iface["version"] = 0
        return suai_iface

    v = View(
        buf, shape=(n_bytes,), strides=(1,), offset=0, transf_fn=invalid_version
    )
    with pytest.raises(ValueError):
        MemoryUSMShared(v)

    # data validation
    def invalid_data(suai_iface):
        "Set data to invalid"
        suai_iface["data"] = tuple()
        return suai_iface

    v = View(
        buf, shape=(n_bytes,), strides=(1,), offset=0, transf_fn=invalid_data
    )
    with pytest.raises(ValueError):
        MemoryUSMShared(v)
    # set shape to a negative value
    v = View(buf, shape=(-n_bytes,), strides=(2,), offset=0)
    with pytest.raises(ValueError):
        MemoryUSMShared(v)
    v = View(buf, shape=(-n_bytes,), strides=None, offset=0)
    with pytest.raises(ValueError):
        MemoryUSMShared(v)
    # shape validation
    v = View(buf, shape=None, strides=(1,), offset=0)
    with pytest.raises(ValueError):
        MemoryUSMShared(v)

    # typestr validation
    def invalid_typestr(suai_iface):
        suai_iface["typestr"] = "invalid"
        return suai_iface

    v = View(
        buf, shape=(n_bytes,), strides=(1,), offset=0, transf_fn=invalid_typestr
    )
    with pytest.raises(ValueError):
        MemoryUSMShared(v)

    def unsupported_typestr(suai_iface):
        suai_iface["typestr"] = "O"
        return suai_iface

    v = View(
        buf,
        shape=(n_bytes,),
        strides=(1,),
        offset=0,
        transf_fn=unsupported_typestr,
    )
    with pytest.raises(ValueError):
        MemoryUSMShared(v)
    # set strides to invalid value
    v = View(buf, shape=(n_bytes,), strides=Ellipsis, offset=0)
    with pytest.raises(ValueError):
        MemoryUSMShared(v)


def check_view(v):
    """
    Memory object created from duck __sycl_usm_array_interface__ argument
    should be consistent with the buffer from which the argument was constructed
    """
    assert type(v) is View
    buf = v.buffer_
    m = as_usm_memory(v)
    assert m.get_usm_type() == buf.get_usm_type()
    assert m._pointer == buf._pointer
    assert m.sycl_device == buf.sycl_device


def test_with_constructor(memory_ctor):
    try:
        buf = memory_ctor(64)
    except Exception:
        pytest.skip("{} could not be allocated".format(memory_ctor.__name__))
    # reuse queue from buffer's SUAI
    v = View(buf, shape=(64,), strides=(1,), offset=0)
    check_view(v)
    # Use SyclContext
    v = View(buf, shape=(64,), strides=(1,), offset=0, syclobj=buf.sycl_context)
    check_view(v)
    # Use queue capsule
    v = View(
        buf,
        shape=(64,),
        strides=(1,),
        offset=0,
        syclobj=buf.sycl_queue._get_capsule(),
    )
    check_view(v)
    # Use context capsule
    v = View(
        buf,
        shape=(64,),
        strides=(1,),
        offset=0,
        syclobj=buf.sycl_context._get_capsule(),
    )
    check_view(v)
    # Use filter string
    v = View(
        buf,
        shape=(64,),
        strides=(1,),
        offset=0,
        syclobj=buf.sycl_device.filter_string,
    )
    check_view(v)


@pytest.mark.skipif(
    not has_sycl_platforms(),
    reason="No SYCL devices except the default host device.",
)
def test_cpython_api(memory_ctor):
    import ctypes
    import sys

    mobj = memory_ctor(1024)
    mod = sys.modules[mobj.__class__.__module__]
    # get capsules storing function pointers
    mem_ptr_fn_cap = mod.__pyx_capi__["get_usm_pointer"]
    mem_ctx_fn_cap = mod.__pyx_capi__["get_context"]
    mem_nby_fn_cap = mod.__pyx_capi__["get_nbytes"]
    # construct Python callable to invoke "get_usm_pointer"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    mem_ptr_fn_ptr = cap_ptr_fn(
        mem_ptr_fn_cap, b"DPCTLSyclUSMRef (struct Py_MemoryObject *)"
    )
    mem_ctx_fn_ptr = cap_ptr_fn(
        mem_ctx_fn_cap, b"DPCTLSyclContextRef (struct Py_MemoryObject *)"
    )
    mem_nby_fn_ptr = cap_ptr_fn(
        mem_nby_fn_cap, b"size_t (struct Py_MemoryObject *)"
    )
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_ptr_fn = callable_maker(mem_ptr_fn_ptr)
    get_ctx_fn = callable_maker(mem_ctx_fn_ptr)
    get_nby_fn = callable_maker(mem_nby_fn_ptr)

    capi_ptr = get_ptr_fn(mobj)
    direct_ptr = mobj._pointer
    assert capi_ptr == direct_ptr
    capi_ctx_ref = get_ctx_fn(mobj)
    direct_ctx_ref = mobj._context.addressof_ref()
    assert capi_ctx_ref == direct_ctx_ref
    capi_nbytes = get_nby_fn(mobj)
    direct_nbytes = mobj.nbytes
    assert capi_nbytes == direct_nbytes


def test_memory_construction_from_other_memory_objects():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default queue could not be created")
    m_sh = MemoryUSMShared(256, queue=q)
    m_de = MemoryUSMDevice(256, queue=q)
    m_ho = MemoryUSMHost(256, queue=q)
    with pytest.raises(ValueError):
        MemoryUSMDevice(m_sh)
    with pytest.raises(ValueError):
        MemoryUSMHost(m_de)
    with pytest.raises(ValueError):
        MemoryUSMShared(m_ho)
    m1 = MemoryUSMDevice(m_sh, copy=True)
    m2 = MemoryUSMHost(m_de, copy=True)
    m3 = MemoryUSMShared(m_de, copy=True)
    assert bytes(m1) == bytes(m_sh)
    assert bytes(m2) == bytes(m3)


def test_memory_copy_between_contexts():
    try:
        q = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("CPU queue could not be created")
    d = q.sycl_device
    n = d.max_compute_units
    n_half = n // 2
    d0, d1 = d.create_sub_devices(partition=[n_half, n - n_half])
    q0 = dpctl.SyclQueue(d0)
    q1 = dpctl.SyclQueue(d1)
    m0 = MemoryUSMDevice(256, queue=q0)
    m1 = MemoryUSMDevice(256, queue=q1)
    host_buf = b"abcd" * 64
    m0.copy_from_host(host_buf)
    m1.copy_from_device(m0)
    copy_buf = bytearray(256)
    m1.copy_to_host(copy_buf)
    assert host_buf == copy_buf
