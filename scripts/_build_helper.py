#                      Data Parallel Control (dpctl)
#
# Copyright 2025 Intel Corporation
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
import shutil
import subprocess
import sys


def run(cmd, env=None, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(
        cmd, env=env or os.environ.copy(), cwd=cwd or os.getcwd()
    )


def warn(msg: str):
    print(f"[build_locally][error] {msg}", file=sys.stderr)


def err(msg: str):
    print(f"[build_locally][error] {msg}", file=sys.stderr)


def make_cmake_args(
    c_compiler=None,
    cxx_compiler=None,
    level_zero=True,
    glog=False,
    verbose=False,
    other_opts="",
):
    args = [
        f"-DCMAKE_C_COMPILER:PATH={c_compiler}" if c_compiler else "",
        f"-DCMAKE_CXX_COMPILER:PATH={cxx_compiler}" if cxx_compiler else "",
        f"-DDPCTL_ENABLE_L0_PROGRAM_CREATION={'ON' if level_zero else 'OFF'}",
        f"-DDPTL_ENABLE_GLOG:BOOL={'ON' if glog else 'OFF'}",
    ]

    if verbose:
        args.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")
    if other_opts:
        args.extend(other_opts.split())

    return " ".join(filter(None, args))


def build_extension(
    setup_dir, env, cmake_executable=None, generator=None, build_type=None
):
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    if cmake_executable:
        cmd.append(f"--cmake-executable={cmake_executable}")
    if generator:
        cmd.append(f"--generator={generator}")
    if build_type:
        cmd.append(f"--build-type={build_type}")
    run(
        cmd,
        env=env,
        cwd=setup_dir,
    )


def install_editable(setup_dir, env):
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


def clean_build_dir(setup_dir):
    target = os.path.join(setup_dir or os.getcwd(), "_skbuild")
    if os.path.exists(target):
        print(f"Cleaning build directory: {target}")
        shutil.rmtree(target)
