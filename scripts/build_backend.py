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

"""Invokes CMake build dpctl's C API library.
"""


def build_backend(
    l0_support=False, code_coverage=False, sycl_compiler_prefix=None
):
    import glob
    import os
    import shutil
    import subprocess
    import sys

    IS_WIN = False
    IS_LIN = False

    if "linux" in sys.platform:
        IS_LIN = True
    elif sys.platform in ["win32", "cygwin"]:
        IS_WIN = True
    else:
        assert False, sys.platform + " not supported"

    if sycl_compiler_prefix is None:
        oneapi_root = os.getenv("ONEAPI_ROOT")
        if IS_LIN:
            DPCPP_ROOT = os.path.join(oneapi_root, r"compiler/latest/linux")
        elif IS_WIN:
            DPCPP_ROOT = os.path.join(oneapi_root, r"compiler\latest\windows")
    else:
        DPCPP_ROOT = os.path.join(sycl_compiler_prefix)

    if not os.path.isdir(DPCPP_ROOT):
        raise ValueError(
            "SYCL compile prefix {} is not a directry".format(DPCPP_ROOT)
        )

    dpctl_dir = os.getcwd()
    build_cmake_dir = os.path.join(dpctl_dir, "build_cmake")
    if os.path.exists(build_cmake_dir):
        for f in os.listdir(build_cmake_dir):
            f_path = os.path.join(build_cmake_dir, f)
            if os.path.isdir(f_path):
                if (f == "level-zero") and os.path.isdir(
                    os.path.join(f_path, ".git")
                ):
                    # do not delete Git checkout of level zero headers
                    pass
                else:
                    shutil.rmtree(f_path)
            else:
                os.remove(f_path)
    else:
        os.mkdir(build_cmake_dir)
    os.chdir(build_cmake_dir)

    INSTALL_PREFIX = os.path.join(dpctl_dir, "install")
    if os.path.exists(INSTALL_PREFIX):
        shutil.rmtree(INSTALL_PREFIX)

    backends = os.path.join(dpctl_dir, "dpctl-capi")

    ENABLE_LO_PROGRAM_CREATION = "ON" if l0_support else "OFF"

    if IS_LIN:
        if os.path.exists(os.path.join(DPCPP_ROOT, "bin", "dpcpp")):
            cmake_compiler_args = [
                "-DCMAKE_C_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "clang"),
                "-DCMAKE_CXX_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "dpcpp"),
            ]
        else:
            cmake_compiler_args = [
                "-DDPCTL_CUSTOM_DPCPP_INSTALL_DIR=" + DPCPP_ROOT,
                "-DCMAKE_C_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "clang"),
                "-DCMAKE_CXX_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "clang++"),
            ]
        if code_coverage:
            cmake_args = (
                [
                    "cmake",
                    "-DCMAKE_BUILD_TYPE=Debug",
                    "-DCMAKE_INSTALL_PREFIX=" + INSTALL_PREFIX,
                    "-DCMAKE_PREFIX_PATH=" + INSTALL_PREFIX,
                ]
                + cmake_compiler_args
                + [
                    "-DDPCTL_ENABLE_LO_PROGRAM_CREATION="
                    + ENABLE_LO_PROGRAM_CREATION,
                    "-DDPCTL_BUILD_CAPI_TESTS=ON",
                    "-DDPCTL_GENERATE_COVERAGE=ON",
                    "-DDPCTL_COVERAGE_REPORT_OUTPUT_DIR=" + dpctl_dir,
                    backends,
                ]
            )
            subprocess.check_call(
                cmake_args, stderr=subprocess.STDOUT, shell=False
            )
            subprocess.check_call(["make", "V=1", "-j", "4"])
            subprocess.check_call(["make", "install"])
            subprocess.check_call(["make", "lcov-genhtml"])
        else:
            cmake_args = (
                [
                    "cmake",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_INSTALL_PREFIX=" + INSTALL_PREFIX,
                    "-DCMAKE_PREFIX_PATH=" + INSTALL_PREFIX,
                ]
                + cmake_compiler_args
                + [
                    "-DDPCTL_ENABLE_LO_PROGRAM_CREATION="
                    + ENABLE_LO_PROGRAM_CREATION,
                    backends,
                ]
            )
            subprocess.check_call(
                cmake_args, stderr=subprocess.STDOUT, shell=False
            )
            subprocess.check_call(["make", "V=1", "-j", "4"])
            subprocess.check_call(["make", "install"])

        os.chdir(dpctl_dir)
        for file in glob.glob(
            os.path.join(dpctl_dir, "install", "lib", "*.so*")
        ):
            shutil.copy(file, os.path.join(dpctl_dir, "dpctl"))
    elif IS_WIN:
        if os.path.exists(os.path.join(DPCPP_ROOT, "bin", "dpcpp.exe")):
            cmake_compiler_args = [
                "-DCMAKE_C_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "clang-cl.exe"),
                "-DCMAKE_CXX_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "dpcpp.exe"),
            ]
        else:
            cmake_compiler_args = [
                "-DDPCTL_CUSTOM_DPCPP_INSTALL_DIR=" + DPCPP_ROOT,
                "-DCMAKE_C_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "clang-cl.exe"),
                "-DCMAKE_CXX_COMPILER:PATH="
                + os.path.join(DPCPP_ROOT, "bin", "clang++.exe"),
            ]
        cmake_args = (
            [
                "cmake",
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_INSTALL_PREFIX=" + INSTALL_PREFIX,
                "-DCMAKE_PREFIX_PATH=" + INSTALL_PREFIX,
            ]
            + cmake_compiler_args
            + [
                "-DDPCTL_ENABLE_LO_PROGRAM_CREATION="
                + ENABLE_LO_PROGRAM_CREATION,
                backends,
            ]
        )
        subprocess.check_call(cmake_args, stderr=subprocess.STDOUT, shell=False)
        subprocess.check_call(["ninja", "-n"])
        subprocess.check_call(["ninja", "install"])

        os.chdir(dpctl_dir)
        for file in glob.glob(
            os.path.join(dpctl_dir, "install", "lib", "*.lib")
        ):
            shutil.copy(file, os.path.join(dpctl_dir, "dpctl"))

        for file in glob.glob(
            os.path.join(dpctl_dir, "install", "bin", "*.dll")
        ):
            shutil.copy(file, os.path.join(dpctl_dir, "dpctl"))

    include_dir = os.path.join(dpctl_dir, "dpctl", "include")
    if os.path.exists(include_dir):
        shutil.rmtree(include_dir)

    shutil.copytree(
        os.path.join(dpctl_dir, "dpctl-capi", "include"), include_dir
    )


if __name__ == "__main__":
    build_backend()
