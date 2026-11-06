#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
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

"""
Intel Python CI (gh-ci) test for dpctl.

GitHub Actions, through ci/hooks.py, and Intel's Jenkins both run this with one
built package installed in a clean environment, from an empty working
directory, so the tests import the installed dpctl rather than the source tree.
The test requirements come from ci/config.yml. Writes a JUnit report for
Jenkins.

The environment is scrubbed, so this sets what dpctl needs itself. It prints
the SYCL devices and fails without any, so a run that skipped every device test
cannot pass. Jenkins sets GH_CI_DEVICE=gpu on its GPU agents, and then a GPU
device is required too. Without it the OpenCL CPU runtime from the test
environment is the device, as on GitHub's runners.
"""

import json
import os
import pathlib
import subprocess
import sys
import sysconfig
import tempfile

PACKAGE = "dpctl"

CPU_ICD = {"win32": "intelocl64.dll", "linux": "libintelocl.so"}

DEVICES = """
import json
import dpctl
print(json.dumps([
    [d.backend.name, d.device_type.name, d.name] for d in dpctl.get_devices()
]))
"""

PROBE_SETUP = """
import setuptools

setuptools.setup(
    name="probe",
    ext_modules=[
        setuptools.Extension("probe_c", ["probe_c.c"], language="c"),
        setuptools.Extension("probe_cpp", ["probe_cpp.cpp"], language="c++"),
    ],
)
"""

PROBE_SOURCE = """
#include <Python.h>
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "NAME", 0, -1, 0};
PyMODINIT_FUNC PyInit_NAME(void) { return PyModule_Create(&module); }
"""

# These two tests build extensions with the Python environment's compiler.
NEED_COMPILER = ["test_c_headers", "test_cython_api"]


def running_python() -> str:
    """This interpreter's version as gh-ci spells it, such as 3.14t."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    threaded = sysconfig.get_config_var("Py_GIL_DISABLED")
    return version + ("t" if threaded else "")


def has_compilers(env: dict) -> bool:
    """Whether setuptools can build a C and a C++ extension."""
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        (directory / "setup.py").write_text(PROBE_SETUP, encoding="utf-8")
        for language, suffix in (("c", "c"), ("c++", "cpp")):
            source = PROBE_SOURCE.replace("NAME", f"probe_{suffix}")
            (directory / f"probe_{suffix}.{suffix}").write_text(
                source, encoding="utf-8"
            )
        built = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if built.returncode:
        print(
            f"no usable compiler:\n{built.stdout[-1500:]}{built.stderr[-1500:]}"
        )
    return not built.returncode


def skipped_tests(env: dict) -> list:
    """The tests this environment cannot run, said out loud."""
    skipped = []
    if os.environ.get("GH_CI_KIND") == "wheel":
        print("deselecting test_syclinterface: a wheel keeps no symlinks")
        skipped.append("test_syclinterface")
    if not has_compilers(env):
        print(f"deselecting {', '.join(NEED_COMPILER)}: they need a compiler")
        skipped += NEED_COMPILER
    return skipped


def cpu_runtime_icd() -> str | None:
    """The OpenCL CPU runtime of the test environment, if it has one."""
    name = CPU_ICD.get(sys.platform, CPU_ICD["linux"])
    found = next(pathlib.Path(sys.prefix).rglob(name), None)
    return str(found) if found else None


def merge_icd_vendors(env: dict, merged: pathlib.Path) -> None:
    """Lets the environment's OpenCL loader see the system's GPU drivers too."""
    system = pathlib.Path("/etc/OpenCL/vendors")
    own = pathlib.Path(sys.prefix, "etc", "OpenCL", "vendors")
    if not (system.is_dir() and own.is_dir()):
        return
    for icd in [*own.glob("*.icd"), *system.glob("*.icd")]:
        if not (merged / icd.name).exists():
            (merged / icd.name).symlink_to(icd)
    env["OCL_ICD_VENDORS"] = str(merged)


def register_icd(icd: str) -> None:
    """Registers the ICD in the registry, which Windows' loader reads."""
    import winreg

    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Khronos\OpenCL\Vendors",
            0,
            winreg.KEY_SET_VALUE,
        ) as vendors:
            winreg.SetValueEx(vendors, icd, 0, winreg.REG_DWORD, 0)
    except OSError as error:
        print(f"could not register {icd}: {error}")


def sycl_env(gpu: bool, merged: pathlib.Path) -> dict:
    """The environment dpctl runs in."""
    env = dict(os.environ, SYCL_CACHE_PERSISTENT="1")
    if sys.platform == "win32":
        # setuptools finds Visual C++ from these, and gh-ci's scrubbed
        # environment lacks them.
        drive = env.get("SYSTEMDRIVE", "C:")
        env.setdefault("PROGRAMFILES", drive + "\\Program Files")
        env.setdefault("PROGRAMFILES(X86)", drive + "\\Program Files (x86)")
    if gpu:
        if sys.platform.startswith("linux"):
            merge_icd_vendors(env, merged)
        return env
    # Linux and newer Windows loaders take the ICD from here.
    icd = cpu_runtime_icd()
    if icd:
        env["OCL_ICD_FILENAMES"] = icd
        if sys.platform == "win32":
            register_icd(icd)
    return env


def devices(env: dict) -> list:
    """The [backend, type, name] of every SYCL device, in a fresh process."""
    listed = subprocess.run(
        [sys.executable, "-c", DEVICES],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode:
        print(listed.stderr)
        return []
    return json.loads(listed.stdout)


def devices_ok(found: list, gpu: bool) -> bool:
    """Prints the devices, and whether the cell has the ones it needs."""
    for backend, kind, name in found:
        print(f"SYCL device: {backend}:{kind} {name}")
    if not found:
        print("no SYCL device, so the tests would skip everything")
        return False
    if gpu and not any(kind == "gpu" for _, kind, _ in found):
        print("GH_CI_DEVICE=gpu, but there is no GPU device")
        return False
    return True


def main() -> int:
    """Runs the installed package's test suite."""
    sys.stdout.reconfigure(line_buffering=True)
    expected = os.environ["GH_CI_PYTHON"]
    if running_python() != expected:
        print(f"running Python {running_python()}, the cell is {expected}")
        return 1
    gpu = os.environ.get("GH_CI_DEVICE") == "gpu"
    with tempfile.TemporaryDirectory() as merged:
        env = sycl_env(gpu, pathlib.Path(merged))
        subprocess.run([sys.executable, "-m", "dpctl", "-f"], env=env)
        if not devices_ok(devices(env), gpu):
            return 1
        # Every cell of a Jenkins build reports into the same build, so each
        # names its own suite.
        cell = f"{os.environ.get('GH_CI_PLATFORM', 'linux')}.py{expected}"
        report = pathlib.Path(os.environ["GH_CI_RESULTS"]) / f"{PACKAGE}.xml"
        pytest = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            "-o",
            f"junit_suite_name={PACKAGE} {cell}",
            "--pyargs",
            PACKAGE,
        ]
        if skipped := skipped_tests(env):
            pytest += ["-k", "not " + " and not ".join(skipped)]
        return subprocess.run(pytest, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
