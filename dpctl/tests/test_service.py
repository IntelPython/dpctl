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

""" Defines unit test cases for miscellaneous functions.
"""

import ctypes
import ctypes.util
import glob
import os
import os.path
import re

import pytest

import dpctl


def _get_mkl_version_if_present():
    class MKLVersion(ctypes.Structure):
        _fields_ = [
            ("MajorVersion", ctypes.c_int),
            ("MinorVersion", ctypes.c_int),
            ("UpdateVersion", ctypes.c_int),
            ("ProductStatus", ctypes.c_char_p),
            ("Build", ctypes.c_char_p),
            ("Processor", ctypes.c_char_p),
            ("Platform", ctypes.c_char_p),
        ]

    lib = ctypes.util.find_library("mkl_rt")
    if lib is None:
        return None
    try:
        lib = ctypes.cdll.LoadLibrary(lib)
        get_ver_fn = lib.mkl_get_version
    except Exception:
        return None
    get_ver_fn.argtypes = []
    get_ver_fn.restype = MKLVersion
    mkl_ver = get_ver_fn()
    return ".".join(
        [
            str(mkl_ver.MajorVersion),
            str(mkl_ver.UpdateVersion),
            str(mkl_ver.MinorVersion),
        ]
    )


def test_get_include():
    incl = dpctl.get_include()
    assert type(incl) is str
    assert incl != ""
    assert os.path.isdir(incl)


def test_get_dpcppversion():
    """Intent of this test is to verify that libraries from dpcpp_cpp_rt
    conda package used at run-time are not from an older oneAPI. Since these
    libraries currently do not report the version, this test was using
    a proxy (version of Intel(R) Math Kernel Library).
    """
    incl_dir = dpctl.get_include()
    libs = glob.glob(os.path.join(incl_dir, "..", "*DPCTLSyclInterface*"))
    libs = sorted(libs)
    assert len(libs) > 0
    lib = ctypes.cdll.LoadLibrary(libs[0])
    fn = lib.DPCTLService_GetDPCPPVersion
    fn.restype = ctypes.c_char_p
    fn.argtypes = []
    dpcpp_ver = fn()
    assert len(dpcpp_ver) > 0
    dpcpp_ver = dpcpp_ver.decode("utf-8")
    mkl_ver = _get_mkl_version_if_present()
    if mkl_ver is not None:
        if not mkl_ver >= dpcpp_ver:
            pytest.xfail(
                reason="Flaky test: Investigate Math Kernel Library "
                f"library version {mkl_ver} being older than "
                f"DPC++ version {dpcpp_ver} used to build dpctl"
            )


def test___version__():
    dpctl_ver = getattr(dpctl, "__version__", None)
    assert type(dpctl_ver) is str
    assert "unknown" not in dpctl_ver
    # Reg expr from PEP-440, relaxed to allow for semantic variant
    # 0.9.0dev0 allowed, vs. PEP-440 compliant 0.9.0.dev0
    reg_expr = (
        r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))"
        r"*((a|b|rc)(0|[1-9][0-9]*))?(\.?post(0|[1-9][0-9]*))?(\.?dev("
        r"0|[1-9][0-9]*))?(\+.*)?$"
    )
    assert re.match(reg_expr, dpctl_ver) is not None


def test_dev_utils():
    import tempfile

    import dpctl._diagnostics as dd

    ctx_mngr = dd.syclinterface_diagnostics

    with ctx_mngr():
        dpctl.SyclDevice().parent_device
    with ctx_mngr(verbosity="error"):
        dpctl.SyclDevice().parent_device
    with pytest.raises(ValueError):
        with ctx_mngr(verbosity="blah"):
            dpctl.SyclDevice().parent_device
    with tempfile.TemporaryDirectory() as temp_dir:
        with ctx_mngr(log_dir=temp_dir):
            dpctl.SyclDevice().parent_device
    with pytest.raises(ValueError):
        with ctx_mngr(log_dir="/not_a_dir"):
            dpctl.SyclDevice().parent_device
