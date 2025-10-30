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

# add scripts dir to Python path so we can import _build_helper
sys.path.insert(0, os.path.abspath("scripts"))

from _build_helper import (  # noqa: E402
    build_extension,
    clean_build_dir,
    err,
    get_output,
    install_editable,
    make_cmake_args,
    resolve_compilers,
    run,
    warn,
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
        help="Disable Level Zero backend (deprecated: use --target-level-zero "
        "OFF)",
    )
    p.add_argument(
        "--target-level-zero",
        action="store_true",
        help="Enable Level Zero backend explicitly",
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

    if args.no_level_zero and args.target_level_zero:
        err(
            "Cannot combine --no-level-zero and --target-level-zero",
            "gen_coverage",
        )

    # Level Zero state (on unless explicitly disabled)
    if args.no_level_zero:
        level_zero_enabled = False
    elif args.target_level_zero:
        level_zero_enabled = True
    else:
        level_zero_enabled = True

    cmake_args = make_cmake_args(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        level_zero=level_zero_enabled,
        verbose=args.verbose,
    )

    cmake_args += " -DDPCTL_GENERATE_DOCS=ON"

    if args.doxyrest_root:
        cmake_args += " -DDPCTL_ENABLE_DOXYREST=ON"
        cmake_args += f" -DDoxyrest_DIR={args.doxyrest_root}"

    env = os.environ.copy()

    if "CMAKE_ARGS" in env and env["CMAKE_ARGS"].strip():
        warn("Ignoring pre-existing CMAKE_ARGS in environment", "gen_docs")
        del env["CMAKE_ARGS"]

    env["CMAKE_ARGS"] = cmake_args

    print(f"[gen_docs] Using CMake args:\n {env['CMAKE_ARGS']}")

    build_extension(
        setup_dir,
        env,
        cmake_executable=args.cmake_executable,
        generator=args.generator,
        build_type="Release",
    )
    install_editable(setup_dir, env)
    cmake_build_dir = get_output(
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
