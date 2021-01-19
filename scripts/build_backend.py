import os
import sys
import subprocess
import shutil
import glob

IS_WIN = False
IS_LIN = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True
else:
    assert False, sys.platform + " not supported"

ONEAPI_ROOT = os.environ.get("ONEAPI_ROOT")

if IS_LIN:
    DPCPP_ROOT = os.path.join(ONEAPI_ROOT, "compiler/latest/linux")
if IS_WIN:
    DPCPP_ROOT = os.path.join(ONEAPI_ROOT, "compiler\latest\windows")

dpctl_dir = os.getcwd()
build_cmake_dir = os.path.join(dpctl_dir, "build_cmake")
if os.path.exists(build_cmake_dir):
    shutil.rmtree(build_cmake_dir)
os.mkdir(build_cmake_dir)
os.chdir(build_cmake_dir)

INSTALL_PREFIX = os.path.join(dpctl_dir, "install")
if os.path.exists(INSTALL_PREFIX):
    shutil.rmtree(INSTALL_PREFIX)

backends = os.path.join(dpctl_dir, "dpctl-capi")

if IS_LIN:
    cmake_args = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + INSTALL_PREFIX,
        "-DCMAKE_PREFIX_PATH=" + INSTALL_PREFIX,
        "-DDPCPP_INSTALL_DIR=" + DPCPP_ROOT,
        "-DCMAKE_C_COMPILER:PATH=" + os.path.join(DPCPP_ROOT, "bin", "clang"),
        "-DCMAKE_CXX_COMPILER:PATH=" + os.path.join(DPCPP_ROOT, "bin", "clang++"),
        backends,
    ]
    subprocess.check_call(cmake_args, stderr=subprocess.STDOUT, shell=False)
    subprocess.check_call(["make", "-j", "4"])
    subprocess.check_call(["make", "install"])

    os.chdir(dpctl_dir)
    for file in glob.glob(os.path.join(dpctl_dir, "install", "lib", "*.so")):
        shutil.copy(file, os.path.join(dpctl_dir, "dpctl"))

if IS_WIN:
    cmake_args = [
        "cmake",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + INSTALL_PREFIX,
        "-DCMAKE_PREFIX_PATH=" + INSTALL_PREFIX,
        "-DDPCPP_ROOT=" + DPCPP_ROOT,
        backends,
    ]
    subprocess.check_call(cmake_args, stderr=subprocess.STDOUT, shell=True)
    subprocess.check_call(["ninja", "-n"])
    subprocess.check_call(["ninja", "install"])

    os.chdir(dpctl_dir)
    for file in glob.glob(os.path.join(dpctl_dir, "install", "lib", "*.lib")):
        shutil.copy(file, os.path.join(dpctl_dir, "dpctl"))

    for file in glob.glob(os.path.join(dpctl_dir, "install", "bin", "*.dll")):
        shutil.copy(file, os.path.join(dpctl_dir, "dpctl"))

include_dir = os.path.join(dpctl_dir, "dpctl", "include")
if os.path.exists(include_dir):
    shutil.rmtree(include_dir)

shutil.copytree(os.path.join(dpctl_dir, "dpctl-capi", "include"), include_dir)
