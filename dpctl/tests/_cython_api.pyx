# cython: language=c++
# cython: language_level=3

cimport dpctl as c_dpctl

import dpctl


def call_create_from_context_and_devices():
    cdef c_dpctl.SyclQueue q
    d = dpctl.SyclDevice()
    ctx = dpctl.SyclContext(d)
    # calling static method
    q = c_dpctl.SyclQueue._create_from_context_and_device(
        <c_dpctl.SyclContext> ctx,
        <c_dpctl.SyclDevice> d
    )
    return q
