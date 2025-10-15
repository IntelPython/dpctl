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
import shutil
import subprocess
import sys


def run(cmd, env=None, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env, cwd=cwd or os.getcwd())


def _warn(msg: str):
    print(f"[build_locally][error] {msg}", file=sys.stderr)


def _err(msg: str):
    print(f"[build_locally][error] {msg}", file=sys.stderr)


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
        "--build-dir",
        default="build",
        help="CMake build directory (default: build)",
    )
    p.add_argument(
        "--clean",
        action="store_false",
        help="Remove build dir before rebuild (default: False)",
    )
    p.add_argument(
        "--skip-editable",
        action="store_true",
        help="Skip pip editable install step",
    )

    return p.parse_args()


def resolve_compilers(args):
    is_linux = "linux" in sys.platform

    if args.oneapi or (
        args.c_compiler is None
        and args.cxx_compiler is None
        and args.compiler_root is None
    ):
        args.c_compiler = "icx"
        args.cxx_compiler = "icpx" if is_linux else "icx"
        args.compiler_root = None
        return

    cr = args.compiler_root
    if isinstance(cr, str) and os.path.exists(cr):
        if args.c_compiler is None:
            args.c_compiler = "icx"
        if args.cxx_compiler is None:
            args.cxx_compiler = "icpx" if is_linux else "icx"
    else:
        raise RuntimeError(
            "'compiler-root' option must be set when using non-default DPC++ "
            "layout"
        )

    for opt_name in ("c_compiler", "cxx_compiler"):
        arg = getattr(args, opt_name)
        if not arg:
            continue
        if not os.path.exists(arg):
            probe = os.path.join(cr, arg)
            if os.path.exists(probe):
                setattr(args, opt_name, probe)
                continue
        if not os.path.exists(getattr(args, opt_name)):
            raise RuntimeError(
                f"{opt_name.replace('_', '-')} value {arg} not found"
            )


def main():
    if sys.platform not in ["cygwin", "win32", "linux"]:
        _err(f"{sys.platform} not supported")
    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(setup_dir, args.build_dir)

    resolve_compilers(args)

    # clean build dir if --clean set
    if args.clean and os.path.exists(build_dir):
        print(f"[build_locally] Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    env = os.environ.copy()

    # ignore pre-existing CMAKE_ARGS for determinism in build driver
    if "CMAKE_ARGS" in env and env["CMAKE_ARGS"].strip():
        _warn("Ignoring pre-existing CMAKE_ARGS in environment")
        del env["CMAKE_ARGS"]

    cmake_defs = []

    # handle architecture conflicts
    if args.target_hip is not None and not args.target_hip.strip():
        _err("--target-hip requires an explicit architecture")

    if args.no_level_zero and args.target_level_zero:
        _err("Cannot combine --no-level-zero and --target-level-zero")

    # CUDA/HIP targets
    if args.target_cuda:
        cmake_defs.append(f"-DDPCTL_TARGET_CUDA={args.target_cuda}")
    if args.target_hip:
        cmake_defs.append(f"-DDPCTL_TARGET_HIP={args.target_hip}")

    # Level Zero state (on unless explicitly disabled)
    if args.no_level_zero:
        level_zero_enabled = False
    elif args.target_level_zero:
        level_zero_enabled = True
    else:
        level_zero_enabled = True
    cmake_defs.append(
        "-DDPCTL_ENABLE_L0_PROGRAM_CREATION="
        f"{'ON' if level_zero_enabled else 'OFF'}"
    )

    # compilers and generator
    if args.c_compiler:
        cmake_defs.append(f"-DCMAKE_C_COMPILER:PATH={args.c_compiler}")
    if args.cxx_compiler:
        cmake_defs.append(f"-DCMAKE_CXX_COMPILER:PATH={args.cxx_compiler}")
    if args.generator:
        cmake_defs.append(f"-G{args.generator}")

    cmake_defs.append(
        f"-DDPCTL_ENABLE_GLOG:BOOL={'ON' if args.glog else 'OFF'}"
    )
    cmake_defs.append(f"-DCMAKE_BUILD_TYPE={args.build_type}")
    if args.verbose:
        cmake_defs.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")

    if args.cmake_opts:
        cmake_defs.extend(args.cmake_opts.split())

    env["CMAKE_ARGS"] = " ".join(cmake_defs)
    print(f"[build_locally] CMake args:\n {' '.join(cmake_defs)}")

    print("[build_locally] Building extensions in-place...")
    run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        env=env,
        cwd=setup_dir,
    )

    if not args.skip_editable:
        print("[build_locally] Installing dpctl in editable mode")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "--no-build-isolation",
            ],
            env=env,
            cwd=setup_dir,
        )
    else:
        print("[build_locally] Skipping editable install (--skip-editable)")

    print("[build_locally] Build complete")


if __name__ == "__main__":
    main()
