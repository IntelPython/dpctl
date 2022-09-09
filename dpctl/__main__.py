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

import argparse
import os
import os.path
import platform
import sys

import dpctl


def _dpctl_dir() -> str:
    abs_path = os.path.abspath(dpctl.__file__)
    dpctl_dir = os.path.dirname(abs_path)
    return dpctl_dir


def print_includes() -> None:
    "Prints include flags for dpctl and SyclInterface library"
    print("-I " + dpctl.get_include())


def print_cmake_dir() -> None:
    "Prints directory with FindDpctl.cmake"
    dpctl_dir = _dpctl_dir()
    print(os.path.join(dpctl_dir, "resources", "cmake"))


def print_library() -> None:
    "Prints linker flags for SyclInterface library"
    dpctl_dir = _dpctl_dir()
    plt = platform.platform()
    ld_flags = "-L " + dpctl_dir
    if plt != "Windows":
        ld_flags = ld_flags + " -Wl,-rpath," + dpctl_dir
    print(ld_flags + " -lSyclInterface")


def main() -> None:
    """Main entry-point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--includes",
        action="store_true",
        help="Include flags dpctl headers.",
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
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.includes:
        print_includes()
    if args.cmakedir:
        print_cmake_dir()
    if args.library:
        print_library()


if __name__ == "__main__":
    main()
