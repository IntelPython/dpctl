#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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
import subprocess
import sys

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
    assert os.path.exists(os.path.join(incl, "dpctl4pybind11.hpp"))
    assert os.path.exists(os.path.join(incl, "dpctl_capi.h"))
    assert os.path.exists(os.path.join(incl, "dpctl_sycl_interface.h"))
    assert os.path.exists(
        os.path.join(incl, "syclinterface", "Config", "dpctl_config.h")
    )
    assert os.path.exists(
        os.path.join(incl, "syclinterface", "dpctl_sycl_types.h")
    )
    assert os.path.exists(
        os.path.join(incl, "syclinterface", "dpctl_sycl_type_casters.hpp")
    )


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
    assert "untagged" not in dpctl_ver
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

    try:
        device = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default-constructed device could not be created")

    with ctx_mngr():
        device.parent_device
    with ctx_mngr(verbosity="error"):
        device.parent_device
    with pytest.raises(ValueError):
        with ctx_mngr(verbosity="blah"):
            device.parent_device
    with tempfile.TemporaryDirectory() as temp_dir:
        with ctx_mngr(log_dir=temp_dir):
            device.parent_device
    with pytest.raises(ValueError):
        with ctx_mngr(log_dir="/not_a_dir"):
            device.parent_device


def test_syclinterface():
    install_dir = os.path.dirname(os.path.abspath(dpctl.__file__))
    paths = glob.glob(os.path.join(install_dir, "*DPCTLSyclInterface*"))
    if "linux" in sys.platform:
        assert len(paths) > 1 and any(
            [os.path.islink(fn) for fn in paths]
        ), "All library instances are hard links"
    elif sys.platform in ["win32", "cygwin"]:
        exts = []
        for fn in paths:
            _, file_ext = os.path.splitext(fn)
            exts.append(file_ext.lower())
        assert (
            ".lib" in exts
        ), "Installation does not have DPCTLSyclInterface.lib"
        assert (
            ".dll" in exts
        ), "Installation does not have DPCTLSyclInterface.dll"
    else:
        raise RuntimeError("Unsupported system")


def test_main_include_dir():
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--include-dir"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    dir_path = res.stdout.decode("utf-8").strip()
    assert os.path.exists(dir_path)


def test_main_includes():
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--includes"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    flags = res.stdout.decode("utf-8")
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--include-dir"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    dir = res.stdout.decode("utf-8")
    assert flags == "-I " + dir


def test_main_library():
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--library"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    output = res.stdout.decode("utf-8")
    assert output.startswith("-L")
    assert "DPCTLSyclInterface" in output


def test_tensor_includes():
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--tensor-includes"],
        capture_output=True,
    )
    assert res.returncode == 0
    assert res.stdout
    flags = res.stdout.decode("utf-8")
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--tensor-include-dir"],
        capture_output=True,
    )
    assert res.returncode == 0
    assert res.stdout
    dir = res.stdout.decode("utf-8")
    assert flags == "-I " + dir


def test_main_library_dir():
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--library-dir"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    dir_path = res.stdout.decode("utf-8").strip()
    assert os.path.exists(dir_path)


def _run_main_lsplatform(flags: list) -> subprocess.CompletedProcess:
    status = subprocess.run(
        [sys.executable, "-m", "dpctl"] + flags, capture_output=True
    )
    return status


def _check_completed_process(cp: subprocess.CompletedProcess):
    assert cp.returncode == 0
    if cp.stdout:
        assert cp.stdout.decode("utf-8").strip()
    else:
        # Listing may be empty due to no devices known to DPC++ RT
        # or due to a bug in CPU RT
        assert dpctl.get_num_devices() == 0 or (
            dpctl.get_num_devices() == 1 and dpctl.has_cpu_devices()
        )


def _check_main_lsplatform(mode):
    # List platform using subprocess
    status = _run_main_lsplatform([mode])
    _check_completed_process(status)


def test_main_full_list():
    _check_main_lsplatform("-f")


def test_main_long_list():
    _check_main_lsplatform("-l")


def test_main_summary():
    _check_main_lsplatform("-s")


def test_main_warnings():
    res = _run_main_lsplatform(["-s", "--includes"])
    _check_completed_process(res)
    assert "UserWarning" in res.stderr.decode("utf-8")
    assert "is being ignored." in res.stderr.decode("utf-8")

    res = _run_main_lsplatform(["-s", "--includes", "--cmakedir"])
    _check_completed_process(res)
    assert "UserWarning" in res.stderr.decode("utf-8")
    assert "are being ignored." in res.stderr.decode("utf-8")


def test_cmakedir():
    res = subprocess.run(
        [sys.executable, "-m", "dpctl", "--cmakedir"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    cmake_dir = res.stdout.decode("utf-8").strip()
    assert os.path.exists(os.path.join(cmake_dir, "dpctl-config.cmake"))
