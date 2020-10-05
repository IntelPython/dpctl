call "%ONEAPI_ROOT%compiler\latest\env\vars.bat"
IF ERRORLEVEL 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

set "CC=clang.exe"
set "CXX=dpcpp.exe"

rmdir /S /Q build_cmake
mkdir build_cmake
cd build_cmake

set "DPCPP_ROOT=%ONEAPI_ROOT%\compiler\latest\windows"
set "INSTALL_PREFIX=%cd%\..\install"

rmdir /S /Q "%INSTALL_PREFIX%"

cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX%" ^
    "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
    "%SRC_DIR%/backends"
IF %ERRORLEVEL% NEQ 0 exit 1

ninja -n
ninja install
IF %ERRORLEVEL% NEQ 0 exit 1

cd ..
xcopy install\lib\*.lib dpctl /E /Y
xcopy install\lib\*.dll dpctl /E /Y

mkdir dpctl\include
xcopy backends\include dpctl\include /E /Y


REM required by _opencl_core (dpctl.ocldrv)
set "DPPL_OPENCL_INTERFACE_LIBDIR=dpctl"
set "DPPL_OPENCL_INTERFACE_INCLDIR=dpctl\include"
set "OpenCL_LIBDIR=%DPCPP_ROOT%\lib"

REM required by _sycl_core(dpctl)
set "DPPL_SYCL_INTERFACE_LIBDIR=dpctl"
set "DPPL_SYCL_INTERFACE_INCLDIR=dpctl\include"

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py build install
IF %ERRORLEVEL% NEQ 0 exit 1
