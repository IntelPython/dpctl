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

import os
import subprocess
import sys


def run(
    use_oneapi=True,
    build_type="Release",
    c_compiler=None,
    cxx_compiler=None,
    level_zero=True,
    compiler_root=None,
    cmake_executable=None,
    use_glog=False,
    verbose=False,
    cmake_opts="",
):
    build_system = None

    if "linux" in sys.platform:
        build_system = "Ninja"
    elif sys.platform in ["win32", "cygwin"]:
        build_system = "Ninja"
    else:
        assert False, sys.platform + " not supported"

    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmake_args = [
        sys.executable,
        "setup.py",
        "develop",
    ]
    if cmake_executable:
        cmake_args += [
            "--cmake-executable=" + cmake_executable,
        ]
    cmake_args += [
        "--build-type=" + build_type,
        "--generator=" + build_system,
        "--",
        "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
        "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
        "-DDPCTL_ENABLE_L0_PROGRAM_CREATION=" + ("ON" if level_zero else "OFF"),
        "-DDPCTL_ENABLE_GLOG:BOOL=" + ("ON" if use_glog else "OFF"),
    ]
    if verbose:
        cmake_args += [
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
        ]
    if cmake_opts:
        cmake_args += cmake_opts.split()
    subprocess.check_call(
        cmake_args, shell=False, cwd=setup_dir, env=os.environ
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Driver to build dpctl for in-place installation"
    )
    driver = parser.add_argument_group(title="Coverage driver arguments")
    driver.add_argument("--c-compiler", help="Name of C compiler", default=None)
    driver.add_argument(
        "--cxx-compiler", help="Name of C++ compiler", default=None
    )
    driver.add_argument(
        "--oneapi",
        help="Is one-API installation",
        dest="oneapi",
        action="store_true",
    )
    driver.add_argument(
        "--debug",
        default="Release",
        const="Debug",
        action="store_const",
        help="Set the compilation mode to debugging",
    )
    driver.add_argument(
        "--compiler-root",
        type=str,
        help="Path to compiler home directory",
        default=None,
    )
    driver.add_argument(
        "--cmake-executable",
        type=str,
        help="Path to cmake executable",
        default=None,
    )
    driver.add_argument(
        "--no-level-zero",
        help="Enable Level Zero support",
        dest="level_zero",
        action="store_false",
    )
    driver.add_argument(
        "--glog",
        help="DPCTLSyclInterface uses Google logger",
        dest="glog",
        action="store_true",
    )
    driver.add_argument(
        "--verbose",
        help="Build using vebose makefile mode",
        dest="verbose",
        action="store_true",
    )
    driver.add_argument(
        "--cmake-opts",
        help="Options to pass through to cmake",
        dest="cmake_opts",
        default="",
        type=str,
    )
    args = parser.parse_args()

    args_to_validate = [
        "c_compiler",
        "cxx_compiler",
        "compiler_root",
    ]

    if args.oneapi or (
        args.c_compiler is None
        and args.cxx_compiler is None
        and args.compiler_root is None
    ):
        args.c_compiler = "icx"
        args.cxx_compiler = "icpx" if "linux" in sys.platform else "icx"
        args.compiler_root = None
    else:
        cr = args.compiler_root
        if isinstance(cr, str) and os.path.exists(cr):
            if args.c_compiler is None:
                args.c_compiler = "icx"
            if args.cxx_compiler is None:
                args.cxx_compiler = "icpx" if "linux" in sys.platform else "icx"
        else:
            raise RuntimeError(
                "Option 'compiler-root' must be provided when "
                "using non-default DPC++ layout."
            )
        args_to_validate = [
            "c_compiler",
            "cxx_compiler",
        ]
        for p in args_to_validate:
            arg = getattr(args, p)
            assert isinstance(arg, str)
            if not os.path.exists(arg):
                arg2 = os.path.join(cr, arg)
                if os.path.exists(arg2):
                    arg = arg2
                    setattr(args, p, arg)
            if not os.path.exists(arg):
                opt_name = p.replace("_", "-")
                raise RuntimeError(f"Option {opt_name} value {arg} must exist.")

    run(
        use_oneapi=args.oneapi,
        build_type=args.debug,
        c_compiler=args.c_compiler,
        cxx_compiler=args.cxx_compiler,
        level_zero=args.level_zero,
        compiler_root=args.compiler_root,
        cmake_executable=args.cmake_executable,
        use_glog=args.glog,
        verbose=args.verbose,
        cmake_opts=args.cmake_opts,
    )
