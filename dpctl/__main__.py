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

import argparse
import importlib
import os
import os.path
import platform
import sys
import warnings


def _dpctl_dir() -> str:
    dpctl_dir = importlib.util.find_spec("dpctl").submodule_search_locations[0]
    abs_dpctl_dir = os.path.abspath(dpctl_dir)
    return abs_dpctl_dir


def get_include_dir() -> str:
    "Prints include flags for dpctl and DPCTLSyclInterface library"
    return os.path.join(_dpctl_dir(), "include")


def print_include_flags() -> None:
    "Prints include flags for dpctl and DPCTLSyclInterface library"
    print("-I " + get_include_dir())


def get_tensor_include_dir() -> str:
    dpctl_dir = _dpctl_dir()
    libtensor_dir = os.path.join(dpctl_dir, "tensor", "libtensor", "include")
    return libtensor_dir


def print_tensor_include_flags() -> None:
    "Prints include flags for dpctl and DPCTLSyclInterface library"
    libtensor_dir = get_tensor_include_dir()
    print("-I " + libtensor_dir)


def print_cmake_dir() -> None:
    "Prints directory with dpctl-config.cmake"
    dpctl_dir = _dpctl_dir()
    cmake_dir = os.path.join(dpctl_dir, "resources", "cmake")
    print(cmake_dir)


def get_library_dir() -> str:
    dpctl_dir = _dpctl_dir()
    return dpctl_dir


def print_library() -> None:
    "Prints linker flags for DPCTLSyclInterface library"
    dpctl_dir = get_library_dir()
    plt = platform.platform()
    ld_flags = "-L " + dpctl_dir
    if plt != "Windows":
        ld_flags = ld_flags + " -Wl,-rpath," + dpctl_dir
    print(ld_flags + " -lDPCTLSyclInterface")


def _warn_if_any_set(args, li) -> None:
    opts_set = [it for it in li if getattr(args, it, True)]
    if opts_set:
        if len(opts_set) == 1:
            warnings.warn(
                "The option " + str(opts_set[0]) + " is being ignored.",
                stacklevel=3,
            )
        else:
            warnings.warn(
                "Options " + str(opts_set) + " are being ignored.", stacklevel=3
            )


def print_lsplatform(verbosity: int) -> None:
    import dpctl

    dpctl.lsplatform(verbosity=verbosity)


def main() -> None:
    """Main entry-point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--includes",
        action="store_true",
        help="Include flags for dpctl headers.",
    )
    parser.add_argument(
        "--include-dir",
        action="store_true",
        help="Path to dpctl include directory.",
    )
    parser.add_argument(
        "--tensor-includes",
        action="store_true",
        help="Include flags for dpctl libtensor headers.",
    )
    parser.add_argument(
        "--tensor-include-dir",
        action="store_true",
        help="Path to dpctl libtensor include directory.",
    )
    parser.add_argument(
        "--cmakedir",
        action="store_true",
        help="CMake module directory, ideal for setting -DDPCTL_ROOT in CMake.",
    )
    parser.add_argument(
        "--library",
        action="store_true",
        help="Linker flags for SyclInterface library.",
    )
    parser.add_argument(
        "--library-dir",
        action="store_true",
        help="Path to directory containing DPCTLSyclInterface library",
    )
    parser.add_argument(
        "-f",
        "--full-list",
        action="store_true",
        help="Enumerate system platforms, using dpctl.lsplatform(verbosity=2)",
    )
    parser.add_argument(
        "-l",
        "--long-list",
        action="store_true",
        help="Enumerate system platforms, using dpctl.lsplatform(verbosity=1)",
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Enumerate system platforms, using dpctl.lsplatform()",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.full_list:
        _warn_if_any_set(
            args, ["long_list", "summary", "includes", "cmakedir", "library"]
        )
        print_lsplatform(2)
        return
    if args.long_list:
        _warn_if_any_set(
            args, ["full_list", "summary", "includes", "cmakedir", "library"]
        )
        print_lsplatform(1)
        return
    if args.summary:
        _warn_if_any_set(
            args, ["long_list", "full_list", "includes", "cmakedir", "library"]
        )
        print_lsplatform(0)
        return
    if args.includes:
        print_include_flags()
    if args.include_dir:
        print(get_include_dir())
    if args.tensor_includes:
        print_tensor_include_flags()
    if args.tensor_include_dir:
        print(get_tensor_include_dir())
    if args.cmakedir:
        print_cmake_dir()
    if args.library:
        print_library()
    if args.library_dir:
        print(get_library_dir())


if __name__ == "__main__":
    main()
