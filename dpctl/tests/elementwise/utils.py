import dpctl
import dpctl.tensor._type_utils as tu

_all_dtypes = [
    "b1",
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]
_usm_types = ["device", "shared", "host"]


def _map_to_device_dtype(dt, dev):
    return tu._to_device_supported_dtype(dt, dev)


def _compare_dtypes(dt, ref_dt, sycl_queue=None):
    assert isinstance(sycl_queue, dpctl.SyclQueue)
    dev = sycl_queue.sycl_device
    expected_dt = _map_to_device_dtype(ref_dt, dev)
    return dt == expected_dt


__all__ = [
    "_all_dtypes",
    "_usm_types",
    "_map_to_device_dtype",
    "_compare_dtypes",
]
