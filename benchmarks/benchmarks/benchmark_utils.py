from asv_runner.benchmarks.mark import SkipNotImplemented

import dpctl.tensor as dpt


def skip_unsupported_dtype(q, dtype):
    """
    Skip the benchmark if the device does not support the given data type.
    """
    if (
        (dtype == dpt.float64 or dtype.name == dpt.complex128)
        and not q.sycl_device.has_aspect_fp64
    ) or (dtype == dpt.float16 and not q.sycl_device.has_aspect_fp16):
        raise SkipNotImplemented(
            f"Skipping benchmark for {dtype.name} on this device"
            + " as it is not supported."
        )
