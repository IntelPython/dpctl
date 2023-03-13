#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

cimport numpy as cnp
from cython cimport floating

cimport dpctl as c_dpctl
cimport dpctl.tensor as c_dpt
from dpctl.sycl cimport queue as dpcpp_queue
from dpctl.sycl cimport unwrap_queue

import numpy as np

import dpctl
import dpctl.tensor as dpt


cdef extern from "sycl_blackscholes.hpp":
    cdef void cpp_blackscholes[T](
        dpcpp_queue, size_t n_opts, T* option_params, T* callput
    ) except +
    cdef void cpp_populate_params[T](
        dpcpp_queue,
        size_t n_opts,
        T* option_params,
        T pl,
        T ph,
        T sl,
        T sh,
        T tl,
        T th,
        T rl,
        T rh,
        T vl,
        T vh,
        int seed
    ) except +


def black_scholes_price(c_dpt.usm_ndarray option_params_arr):
    """black_scholes_price(params)

    Applies Black-Scholes-Merton formula to compute call and put European option prices.

    Args:
        option_params_arr: usm_ndarray
            Floating point array with shape (n_opts, 5) containing
            (price, strike, maturity, rate, volatility) per each option.
    Returns:
        usm_ndarray
            Floating point array with shape (n_opts, 2) containing (call_price, put_price)
            per each option.
    """
    cdef size_t n_opts = 0
    cdef size_t n_params = 0
    cdef size_t n_bytes = 0
    cdef c_dpctl.SyclQueue q
    cdef dpcpp_queue* exec_q_ptr = NULL
    cdef c_dpt.usm_ndarray call_put_prices
    cdef double* dp1 = NULL
    cdef double* dp2 = NULL
    cdef float* fp1 = NULL
    cdef float* fp2 = NULL
    cdef int flags_ = 0
    cdef int typenum_ = 0

    if option_params_arr.get_ndim() != 2:
        raise ValueError("Option parameter array must be 2-dimensional")

    n_opts = option_params_arr.get_shape()[0]
    n_params = option_params_arr.get_shape()[1]

    if (n_params != 5):
        raise ValueError((
            "Array of option parameters has unexpected number of "
            "columns {} != 5. Each row must specify (current_price, "
            "strike_price, maturity, interest_rate, volatility)."
            ).format(n_params)
        )

    flags_ = option_params_arr.get_flags()
    if (not (flags_ & c_dpt.USM_ARRAY_C_CONTIGUOUS)):
        raise ValueError("Only C-contiguous arrays are supported")

    q = option_params_arr.get_sycl_queue()
    exec_q_ptr = unwrap_queue(q.get_queue_ref())
    typenum_ = option_params_arr.get_typenum()

    if (typenum_ == c_dpt.UAR_DOUBLE):
        call_put_prices = dpt.empty((n_opts, 2), dtype='d', sycl_queue=q)
        dp1 = <double *>option_params_arr.get_data()
        dp2 = <double *>call_put_prices.get_data()
        cpp_blackscholes[double](exec_q_ptr[0], n_opts, dp1, dp2)
    elif (typenum_ == c_dpt.UAR_FLOAT):
        call_put_prices = dpt.empty((n_opts, 2), dtype='f', sycl_queue=q)
        fp1 = <float *>option_params_arr.get_data()
        fp2 = <float *>call_put_prices.get_data()
        cpp_blackscholes[float](exec_q_ptr[0], n_opts, fp1, fp2)
    else:
        raise ValueError("Unsupported data-type")

    return call_put_prices


def populate_params(
        c_dpt.usm_ndarray option_params_arr,
        pl,
        ph,
        sl,
        sh,
        tl,
        th,
        rl,
        rh,
        vl,
        vh,
        int seed
):
    """ populate_params(params, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed)

    Args:
        params: usm_narray
            Array of shape (n_opts, 5) to populate with price, strike, time to
            maturity, interest rate, volatility rate per option using uniform
            distribution with provided distribution parameters.
        pl: float
            Lower bound for distribution of option price parameter
        ph: float
            Upper bound for distribution of option price parameter
        sl: float
            Lower bound for distribution of option strike parameter
        sh: float
            Upper bound for distribution of option strike parameter
        tl: float
            Lower bound for distribution of option time to maturity parameter
        th: float
            Upper bound for distribution of option time to maturity parameter
        rl: float
            Lower bound for distribution of option interest rate parameter
        rh: float
            Upper bound for distribution of option interest rate parameter
        vl: float
            Lower bound for distribution of option volatility parameter
        vh: float
            Upper bound for distribution of option volatility parameter
        seed: int
            Pseudo-random number generator parameter
    """
    cdef size_t n_opts = 0
    cdef size_t n_params = 0
    cdef c_dpctl.SyclQueue sycl_queue
    cdef dpcpp_queue* exec_q_ptr = NULL
    cdef double* dp = NULL
    cdef float* fp = NULL
    cdef int typenum_ = 0
    cdef int flags_ = 0

    if option_params_arr.get_ndim() != 2:
        raise ValueError("Option parameter array must be 2-dimensional")

    n_opts = option_params_arr.get_shape()[0]
    n_params = option_params_arr.get_shape()[1]

    if (n_params != 5):
        raise ValueError(
            "Array of option parameters has unexpected number of "
            "columns {} != 5. Each row must specify (current_price, "
            "strike_price, maturity, interest_rate, volatility).".format(
                n_params
            )
        )

    flags_ = option_params_arr.get_flags()
    if (not (flags_ & c_dpt.USM_ARRAY_C_CONTIGUOUS)):
        raise ValueError("Only C-contiguous arrays are supported")

    exec_q_ptr = unwrap_queue(option_params_arr.get_queue_ref())

    typenum_ = option_params_arr.get_typenum()

    if (typenum_ == c_dpt.UAR_DOUBLE):
        dp = <double *>option_params_arr.get_data()
        cpp_populate_params[double](
            exec_q_ptr[0], n_opts, dp, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed
        )
    elif (typenum_ == c_dpt.UAR_FLOAT):
        fp = <float *>option_params_arr.get_data()
        cpp_populate_params[float](
            exec_q_ptr[0], n_opts, fp, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed
        )
    else:
        raise ValueError("Unsupported data-type")
