import dpctl


def has_gpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="gpu"))


def has_cpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="cpu"))


def has_sycl_platforms():
    return bool(len(dpctl.get_platforms()))


def create_invalid_capsule():
    """Creates an invalid capsule for the purpose of testing dpctl
    constructors that accept capsules.
    """
    import ctypes

    ctor = ctypes.pythonapi.PyCapsule_New
    ctor.restype = ctypes.py_object
    ctor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return ctor(id(ctor), b"invalid", 0)
