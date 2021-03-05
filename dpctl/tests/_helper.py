import dpctl


def has_gpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="gpu"))

def has_cpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="cpu"))
