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
import subprocess
import sys

from _build_helper import (
    build_extension,
    capture_cmd_output,
    clean_build_dir,
    err,
    install_editable,
    log_cmake_args,
    make_cmake_args,
    resolve_compilers,
    run,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build dpctl and generate coverage")

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
        help="Disable Level Zero backend",
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
        "--cmake-opts",
        type=str,
        default="",
        help="Additional options to pass directly to CMake",
    )

    p.add_argument(
        "--doxyrest-root",
        type=str,
        help=(
            "Path to Doxyrest installation to use to generate Sphinx docs"
            + "for libsyclinterface"
        ),
    )

    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove build dir before rebuild (default: False)",
    )

    return p.parse_args()


def main():
    is_linux = "linux" in sys.platform
    if not is_linux:
        err(f"{sys.platform} not supported", "gen_docs")
    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    c_compiler, cxx_compiler = resolve_compilers(
        args.oneapi,
        args.c_compiler,
        args.cxx_compiler,
        args.compiler_root,
    )

    if args.clean:
        clean_build_dir(setup_dir)

    # Level Zero state (on unless explicitly disabled)
    level_zero_enabled = False if args.no_level_zero else True

    cmake_args = make_cmake_args(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        level_zero=level_zero_enabled,
        verbose=args.verbose,
    )

    cmake_args += ["-DDPCTL_GENERATE_DOCS=ON"]

    if args.doxyrest_root:
        cmake_args += ["-DDPCTL_ENABLE_DOXYREST=ON"]
        cmake_args += [f"-DDoxyrest_DIR={args.doxyrest_root}"]

    log_cmake_args(cmake_args, "gen_docs")

    env = os.environ.copy()

    build_extension(
        setup_dir,
        env,
        cmake_args,
        cmake_executable=args.cmake_executable,
        generator=args.generator,
        build_type="Release",
    )
    install_editable(setup_dir, env)
    cmake_build_dir = capture_cmd_output(
        ["find", "_skbuild", "-name", "cmake-build"], cwd=setup_dir
    )

    print(f"[gen_docs] Found CMake build dir: {cmake_build_dir}")

    run(
        ["cmake", "--build", ".", "--target", "Sphinx"],
        cwd=cmake_build_dir,
    )

    generated_doc_dir = (
        subprocess.check_output(
            ["find", "_skbuild", "-name", "index.html"], cwd=setup_dir
        )
        .decode("utf-8")
        .strip("\n")
    )
    print("Generated documentation placed under ", generated_doc_dir)

    print("[gen_docs] Done")


if __name__ == "__main__":
    main()
