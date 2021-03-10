#                      Data Parallel Control (dpCtl)
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

# cython: language_level=3
# distutils: language=c++

cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpctl_mem
cimport numpy as cnp
from cython cimport floating

import dpctl
import numpy as np

cdef extern from "sycl_blackscholes.hpp":
    cdef void cpp_blackscholes[T](c_dpctl.DPCTLSyclQueueRef, size_t n_opts, T* option_params, T* callput)
    cdef void cpp_populate_params[T](c_dpctl.DPCTLSyclQueueRef, size_t n_opts, T* option_params, T pl, T ph, T sl, T sh, T tl, T th, T rl, T rh, T vl, T vh, int seed)

def black_scholes_price(floating[:, ::1] option_params):
    cdef size_t n_opts = option_params.shape[0]
    cdef size_t n_params = option_params.shape[1]
    cdef c_dpctl.SyclQueue q
    cdef c_dpctl.DPCTLSyclQueueRef q_ptr
    cdef c_dpctl_mem.MemoryUSMShared mobj
    cdef floating[:, :] call_put_prices
    cdef cnp.ndarray callput_arr
    cdef double* dp1
    cdef double* dp2
    cdef float* fp1
    cdef float* fp2

    if (n_params != 5):
        raise ValueError((
            "Array of option parameters has unexpected number of columns {} != 5. "
            "Each row must specify (current_price, strike_price, maturity, interest_rate, volatility)."
            ).format(n_params))

    q = c_dpctl.get_current_queue()
    q_ptr = q.get_queue_ref()
    if (floating is double):
        mobj = c_dpctl_mem.MemoryUSMShared(nbytes=2*n_opts * sizeof(double))
        callput_arr = np.ndarray((n_opts, 2), buffer=mobj, dtype='d')
        call_put_prices = callput_arr
        dp1 = &option_params[0,0]
        dp2 = &call_put_prices[0,0];
        cpp_blackscholes[double](q_ptr, n_opts, dp1, dp2)
    elif (floating is float):
        mobj = c_dpctl_mem.MemoryUSMShared(nbytes=2*n_opts * sizeof(float))
        callput_arr = np.ndarray((n_opts, 2), buffer=mobj, dtype='f')
        call_put_prices = callput_arr
        fp1 = &option_params[0,0]
        fp2 = &call_put_prices[0,0]
        cpp_blackscholes[float](q_ptr, n_opts, fp1, fp2)

    return callput_arr

def populate_params(floating[:, ::1] option_params, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, int seed):
    cdef size_t n_opts = option_params.shape[0]
    cdef size_t n_params = option_params.shape[1]

    cdef c_dpctl.SyclQueue q
    cdef c_dpctl.DPCTLSyclQueueRef q_ptr
    cdef double* dp
    cdef float* fp

    if (n_params != 5):
        raise ValueError((
            "Array of option parameters has unexpected number of columns {} != 5. "
            "Each row must specify (current_price, strike_price, maturity, interest_rate, volatility)."
            ).format(n_params))

    q = c_dpctl.get_current_queue()
    q_ptr = q.get_queue_ref()
    if (floating is double):
        dp = &option_params[0,0]
        cpp_populate_params[double](q_ptr, n_opts, dp, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed)
    elif (floating is float):
        fp = &option_params[0,0]
        cpp_populate_params[float](q_ptr, n_opts, fp, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed)
