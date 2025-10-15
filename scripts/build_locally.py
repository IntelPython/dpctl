#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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
import sys

# add scripts dir to Python path so we can import _build_helper
sys.path.insert(0, os.path.abspath("scripts"))

from _build_helper import (  # noqa: E402
    build_extension,
    clean_build_dir,
    err,
    install_editable,
    make_cmake_args,
    resolve_compilers,
    warn,
)


def parse_args():
    p = argparse.ArgumentParser(description="Local dpctl build driver")

    p.add_argument(
        "--c-compiler", default=None, help="Path or name of C compiler"
    )
    p.add_argument(
        "--cxx-compiler", default=None, help="Path or name of C++ compiler"
    )
    p.add_argument(
        "--compiler-root",
        type=str,
        default=None,
        help="Path to compiler installation root",
    )

    p.add_argument(
        "--oneapi",
        dest="oneapi",
        action="store_true",
        help="Use default oneAPI compiler layout",
    )
    p.add_argument(
        "--debug",
        dest="build_type",
        const="Debug",
        action="store_const",
        default="Release",
        help="Set build type to Debug (defaults to Release)",
    )

    p.add_argument(
        "--generator", type=str, default="Ninja", help="CMake generator"
    )
    p.add_argument(
        "--cmake-executable",
        type=str,
        default=None,
        help="Path to CMake executable used by build",
    )

    p.add_argument(
        "--glog",
        dest="glog",
        action="store_true",
        help="Enable DPCTL Google logger support",
    )
    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose makefile output",
    )

    p.add_argument(
        "--no-level-zero",
        dest="no_level_zero",
        action="store_true",
        default=False,
        help="Disable Level Zero backend (deprecated: use --target-level-zero "
        "OFF)",
    )
    p.add_argument(
        "--target-level-zero",
        action="store_true",
        help="Enable Level Zero backend explicitly",
    )

    p.add_argument(
        "--cmake-opts",
        type=str,
        default="",
        help="Additional options to pass directly to CMake",
    )

    p.add_argument(
        "--target-cuda",
        nargs="?",
        const="ON",
        default=None,
        help="Enable CUDA build. Architecture is optional to specify.",
    )
    p.add_argument(
        "--target-hip",
        required=False,
        type=str,
        help="Enable HIP backend. Architecture required to be specified.",
    )

    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove build dir before rebuild (default: False)",
    )
    p.add_argument(
        "--skip-editable",
        action="store_true",
        help="Skip pip editable install step",
    )

    return p.parse_args()


def main():
    if sys.platform not in ["cygwin", "win32", "linux"]:
        err(f"{sys.platform} not supported")
    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    c_compiler, cxx_compiler = resolve_compilers(
        args.oneapi, args.c_compiler, args.cxx_compiler, args.compiler_root
    )

    # clean build dir if --clean set
    if args.clean:
        clean_build_dir(setup_dir)

    if args.no_level_zero and args.target_level_zero:
        err("Cannot combine --no-level-zero and --target-level-zero")

    # Level Zero state (on unless explicitly disabled)
    if args.no_level_zero:
        level_zero_enabled = False
    elif args.target_level_zero:
        level_zero_enabled = True
    else:
        level_zero_enabled = True

    cmake_args = make_cmake_args(
        build_type=args.build_type,
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        level_zero=level_zero_enabled,
        glog=args.glog,
        generator=args.generator,
        verbose=args.verbose,
        other_opts=args.cmake_opts,
    )

    # handle architecture conflicts
    if args.target_hip is not None and not args.target_hip.strip():
        err("--target-hip requires an explicit architecture")

    # CUDA/HIP targets
    if args.target_cuda:
        cmake_args += f" -DDPCTL_TARGET_CUDA={args.target_cuda}"
    if args.target_hip:
        cmake_args += f" -DDPCTL_TARGET_HIP={args.target_hip}"

    cmake_args += (
        " -DDPCTL_ENABLE_L0_PROGRAM_CREATION="
        f"{'ON' if level_zero_enabled else 'OFF'}"
    )

    env = os.environ.copy()

    # ignore pre-existing CMAKE_ARGS for determinism in build driver
    if "CMAKE_ARGS" in env and env["CMAKE_ARGS"].strip():
        warn("Ignoring pre-existing CMAKE_ARGS in environment")
        del env["CMAKE_ARGS"]

    env["CMAKE_ARGS"] = cmake_args
    print(f"[build_locally] CMake args:\n {cmake_args}")

    print("[build_locally] Building extensions in-place...")

    build_extension(setup_dir, env)
    if not args.skip_editable:
        install_editable(setup_dir, env)
    else:
        print("[build_locally] Skipping editable install (--skip-editable)")

    print("[build_locally] Build complete")


if __name__ == "__main__":
    main()
